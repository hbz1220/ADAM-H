"""
Experiment 3: Derivative-free quadratic saddle search — kappa crossover (Section 7.3)

Compares DF-ADAM-H, DF-ADAM-FC, and DF-HiSD on the quadratic saddle
E_kappa(x) = 0.5*lambda1*x1^2 + 0.5*lambda_plus*sum_{j=2}^d x_j^2,
where lambda1=-10 and lambda_plus=10/kappa. Thus kappa is the condition
number of |H|.

Parameters: d=10, 20 trials, n_g=5, n_h=3, l=1e-4,
            max_iter=3000, tol=1e-4 on true gradient norm.

Author: Jin Zhao
"""

import argparse
import torch
import numpy as np
import sys, os
from datetime import datetime
from typing import Any, Dict, Tuple

torch.set_default_dtype(torch.float64)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from derivative_free_optimizers import (
    DerivativeFreeADAMH, DerivativeFreeADAMFC, DerivativeFreeHiSD,
)
from experiment_utils import (
    final_validation,
    set_trial_seed,
    summarize_success,
    true_grad_norm,
    write_csv,
)


HI_SD_ETA_CANDIDATES = [0.005, 0.01, 0.02, 0.05, 0.1]
HI_SD_ETA_SELECTION_RULE = "success_rate_then_mean_success_iters_then_final_grad_norm"


class QuadraticKappaSaddleFunc:
    """Section 7.3 quadratic family used by this experiment."""

    def __init__(
        self,
        d: int = 10,
        kappa: float = 10.0,
        lambda1: float = -10.0,
        device: str = "cpu",
    ):
        self.d = d
        self.lambda1 = float(lambda1)
        self.kappa = float(kappa)
        if self.kappa <= 0:
            raise ValueError("kappa must be positive")
        self.lambda_plus = 10.0 / self.kappa
        self.device = device

        dtype = torch.get_default_dtype()
        self.eigenvalues = torch.full((d,), self.lambda_plus, dtype=dtype, device=device)
        self.eigenvalues[0] = self.lambda1
        self.H = torch.diag(self.eigenvalues)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum(self.eigenvalues * x ** 2)

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        return self.__call__(x)

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        return self.eigenvalues * x

    def hessian(self, x: torch.Tensor) -> torch.Tensor:
        return self.H


def run_trial(cls, func, x0, V0, kwargs, max_iter=3000, tol=1e-4):
    opt = cls(func=func, **kwargs)
    x, V = x0.clone(), V0.clone()
    last_info = {}
    for t in range(max_iter):
        x, V, last_info = opt.step(x, V)
        gn_true = true_grad_norm(func, x)
        if gn_true < tol:
            validation = final_validation(func, x, target_index=kwargs.get("k", 1), grad_tol=tol)
            return {
                "converged": bool(validation["grad_ok"] and validation["index_ok"]),
                "iterations": t + 1,
                "func_evals": int(getattr(opt, "func_evals", last_info.get("func_evals", 0))),
                **validation,
            }

    validation = final_validation(func, x, target_index=kwargs.get("k", 1), grad_tol=tol)
    return {
        "converged": False,
        "iterations": max_iter,
        "func_evals": int(getattr(opt, "func_evals", last_info.get("func_evals", 0))),
        **validation,
    }


def find_best_hisd_eta(
    func,
    x0,
    V0,
    max_iter,
    tol,
    n_quick=3,
    eta_candidates=None,
):
    """Quick sweep over fixed seeds without leaking formal trial outcomes."""
    etas = eta_candidates if eta_candidates is not None else HI_SD_ETA_CANDIDATES
    common = dict(k=1, eta_v=0.01, n_inner=5, l=1e-4,
                  n_grad_samples=5, n_hv_samples=3)
    best_eta = etas[0]
    best_score = -1
    best_iters = float("inf")
    best_grad = float("inf")
    sweep_rows = []
    candidate_summaries = []

    for eta in etas:
        succ = 0
        success_iters = []
        final_grad_sum = 0.0
        eta_rows = []

        for s in range(n_quick):
            quick_seed = 42 + s * 999
            set_trial_seed(quick_seed)
            r = run_trial(DerivativeFreeHiSD, func,
                          x0.clone(), V0.clone(),
                          {**common, 'eta': eta}, max_iter, tol)
            eta_rows.append({
                "eta": eta,
                "quick_seed": quick_seed,
                "converged": bool(r["converged"]),
                "iterations": int(r["iterations"]),
                "true_grad_norm": float(r["true_grad_norm"]),
                "func_evals": int(r["func_evals"]),
                "hessian_index": int(r["hessian_index"]),
            })
            succ += int(r["converged"])
            if r["converged"]:
                success_iters.append(int(r["iterations"]))
            final_grad_sum += float(r["true_grad_norm"])

        mean_success_iters = float(np.mean(success_iters)) if success_iters else float("inf")
        candidate_summary = {
            "eta": float(eta),
            "success_count": int(succ),
            "n_quick": int(n_quick),
            "mean_success_iterations": mean_success_iters,
            "total_final_grad_norm": float(final_grad_sum),
        }
        candidate_summaries.append(candidate_summary)
        sweep_rows.extend(eta_rows)

        if succ > best_score:
            best_score = succ
            best_eta = eta
            best_iters = mean_success_iters
            best_grad = final_grad_sum
        elif succ == best_score:
            if succ > 0 and mean_success_iters < best_iters:
                best_eta = eta
                best_iters = mean_success_iters
                best_grad = final_grad_sum
            elif succ == 0 and final_grad_sum < best_grad:
                best_eta = eta
                best_iters = mean_success_iters
                best_grad = final_grad_sum

    sweep_summary: Dict[str, Any] = {
        "selection_rule": HI_SD_ETA_SELECTION_RULE,
        "eta_candidates": list(etas),
        "n_quick": int(n_quick),
        "candidate_summaries": candidate_summaries,
    }
    return best_eta, {"rows": sweep_rows, "summary": sweep_summary}


def select_winner(results: Dict[str, Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    """Select by success rate first, then mean iterations on certified successes."""
    if all(r.get("successes", 0) == 0 for r in results.values()):
        return "All fail", {"success_rate": 0.0, "mean_success_iterations": float("inf")}

    best_name = None
    best_score = None
    best_detail = {"success_rate": 0.0, "mean_success_iterations": float("inf")}
    for name, r in results.items():
        mean_iters = float("inf") if r.get("successes", 0) == 0 else float(r["mean"])
        score = (r["rate"], -mean_iters)
        if best_score is None or score > best_score:
            best_score = score
            best_name = name
            best_detail = {
                "success_rate": r["rate"],
                "mean_success_iterations": mean_iters,
            }
    return best_name, best_detail



def experiment_df_quadratic_kappa(
    device='cpu',
    kappas=None,
    n_trials=20,
    max_iter=3000,
    tol=1e-4,
    results_dir=None,
):
    d = 10
    if kappas is None:
        kappas = [1, 10, 50, 100, 200, 500, 1000]

    common_adam = dict(k=1, eta=0.05, eta_v=0.01, n_inner=5,
                       l=1e-4, n_grad_samples=5, n_hv_samples=3,
                       device=device)

    print("=" * 80)
    print(f"DF Quadratic Saddle kappa-Crossover: d={d}, {n_trials} trials")
    print("=" * 80)

    all_rows = []
    raw_rows = []
    hisd_sweep_rows = []
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(__file__), "..", "result", "7.3")
    output_path = os.path.join(results_dir, "exp3_df_quadratic_kappa.csv")
    sweep_output_path = os.path.join(results_dir, "exp3_df_quadratic_kappa_hisd_sweep.csv")

    for kappa in kappas:
        lambda1 = -10.0
        lambda_plus = 10.0 / kappa
        func = QuadraticKappaSaddleFunc(d=d, kappa=float(kappa), lambda1=lambda1,
                                       device=device)

        # Fixed initial direction (axis-aligned)
        v0 = torch.zeros(d, device=device, dtype=torch.get_default_dtype())
        v0[0] = 1.0
        V0_base = v0.unsqueeze(0)

        # Find best HiSD eta for this kappa
        set_trial_seed(900000 + int(kappa))
        x0_probe = 0.5 * torch.randn(d, device=device, dtype=torch.get_default_dtype())
        hisd_eta, hisd_sweep = find_best_hisd_eta(
            func, x0_probe, V0_base.clone(), max_iter, tol
        )
        print(f"kappa={kappa}: selected DF-HiSD eta = {hisd_eta}")

        for row in hisd_sweep["rows"]:
            hisd_sweep_rows.append({
                "experiment": "df_quadratic_kappa",
                "d": d,
                "kappa": kappa,
                "lambda1": lambda1,
                "lambda_plus": lambda_plus,
                "selection_rule": hisd_sweep["summary"]["selection_rule"],
                "n_quick": hisd_sweep["summary"]["n_quick"],
                "candidates": str(hisd_sweep["summary"]["eta_candidates"]),
                **row,
            })

        common_hisd = dict(k=1, eta=hisd_eta, eta_v=0.01, n_inner=5,
                           l=1e-4, n_grad_samples=5, n_hv_samples=3,
                           device=device)

        methods = {
            'DF-ADAM-H': (DerivativeFreeADAMH, common_adam),
            'DF-ADAM-FC': (DerivativeFreeADAMFC, common_adam),
            'DF-HiSD': (DerivativeFreeHiSD, common_hisd),
        }

        results = {}
        for name, (cls, kw) in methods.items():
            method_rows = []
            for trial in range(n_trials):
                trial_seed = trial * 100 + 42
                set_trial_seed(trial_seed)
                x0 = 0.5 * torch.randn(d, device=device, dtype=torch.get_default_dtype())
                v0t = torch.randn(d, device=device, dtype=torch.get_default_dtype())
                v0t[0] = abs(v0t[0]) + 0.5
                v0t = v0t / torch.norm(v0t)
                V0t = v0t.unsqueeze(0)

                set_trial_seed(trial_seed)
                r = run_trial(cls, func, x0, V0t, kw, max_iter, tol)
                row = {
                    "experiment": "df_quadratic_kappa",
                    "d": d,
                    "kappa": kappa,
                    "lambda1": lambda1,
                    "lambda_plus": lambda_plus,
                    "method": name,
                    "trial": trial,
                    "seed": trial_seed,
                    "eta": kw["eta"],
                    "hisd_eta_selected": hisd_eta,
                    "hisd_eta_candidates": str(HI_SD_ETA_CANDIDATES),
                    "hisd_eta_selection_rule": HI_SD_ETA_SELECTION_RULE,
                    "eta_v": kw["eta_v"],
                    "n_inner": kw["n_inner"],
                    "n_grad_samples": kw["n_grad_samples"],
                    "n_hv_samples": kw["n_hv_samples"],
                    "l": kw["l"],
                    "tol": tol,
                    "max_iter": max_iter,
                    **r,
                }
                method_rows.append(row)
                raw_rows.append(row)
                status = f"{r['iterations']:4d}" if r["converged"] else "FAIL"
                print(f"  {name} kappa={kappa} trial {trial+1:2d}: {status}")

            summary = summarize_success(method_rows)
            results[name] = {
                'rate': summary["rate"],
                'mean': summary["mean"],
                'std': summary["std"],
                'successes': summary["successes"],
                'n': summary["n"],
            }
            print(f"  >> {name}: {summary['rate']:.0f}%, mean={summary['mean']}, std={summary['std']}")

        all_rows.append((kappa, results))

    # Summary table
    print("\n" + "=" * 90)
    print("Winner is selected by success rate first and mean iterations over certified successes second.")
    hdr = f"{'kappa':>6} | {'DF-ADAM-H':>25} | {'DF-ADAM-FC':>25} | {'DF-HiSD':>25} | {'Best by rule':>25}"
    print(hdr)
    print("-" * 90)
    for kappa, results in all_rows:
        parts = []
        for name in ['DF-ADAM-H', 'DF-ADAM-FC', 'DF-HiSD']:
            r = results[name]
            if r['rate'] < 10:
                parts.append(f"Fail ({r['rate']:.0f}%)")
            else:
                parts.append(f"{r['mean']} +/- {r['std']} ({r['rate']:.0f}%)")

        winner, _ = select_winner(results)
        winner = winner if winner == "All fail" else winner.replace('DF-', '')
        print(f"{kappa:>6} | {parts[0]:>25} | {parts[1]:>25} | {parts[2]:>25} | {winner}")

    # LaTeX
    print("\n=== LaTeX ===")
    print("Winner is selected by success rate first and mean iterations over certified successes second.")
    for kappa, results in all_rows:
        parts = []
        for name in ['DF-ADAM-H', 'DF-ADAM-FC', 'DF-HiSD']:
            r = results[name]
            if r['rate'] < 10:
                parts.append(f"Fail ({r['rate']:.0f}\\%)")
            else:
                parts.append(f"${r['mean']} \\pm {r['std']}$ ({r['rate']:.0f}\\%)")
        best, _ = select_winner(results)
        best_label = best if best == "All fail" else best.replace("DF-", "")
        print(f"{kappa} & {parts[0]} & {parts[1]} & {parts[2]} & {best_label} \\\\")

    write_csv(output_path, raw_rows)
    write_csv(sweep_output_path, hisd_sweep_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the derivative-free quadratic kappa sweep.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--kappas", nargs="+", type=int, default=[1, 10, 50, 100, 200, 500, 1000])
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument(
        "--results-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "result", "7.3"),
    )
    args = parser.parse_args()
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    experiment_df_quadratic_kappa(
        device=args.device,
        kappas=args.kappas,
        n_trials=args.n_trials,
        max_iter=args.max_iter,
        tol=args.tol,
        results_dir=args.results_dir,
    )
    print(f"\nFinished: {datetime.now().strftime('%H:%M:%S')}")
