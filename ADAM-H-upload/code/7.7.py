"""
Experiment 7: Randomly Rotated Quadratic Saddle (Table 7.7)

Tests DF-ADAM-H vs DF-ADAM-FC vs DF-HiSD on a randomly rotated quadratic,
where ALL eigenvectors are non-axis-aligned (worst case for pollution).

Parameters: d=10, 15 trials, n_g=5, n_h=3, l=1e-4, max_iter=5000, tol=1e-4.

Author: Jin Zhao
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from derivative_free_optimizers import DerivativeFreeADAMFC, DerivativeFreeADAMH, DerivativeFreeHiSD
from experiment_utils import final_validation, set_trial_seed, summarize_success, true_grad_norm, write_csv


def is_true(v):
    return str(v).strip().lower() in {"true", "1", "yes"}


class RandomlyRotatedQuadraticSaddleFunc:
    def __init__(self, d=10, kappa=100.0, lambda_unstable=-10.0, stable_min=None, stable_max=1.0, rotation_seed=123, stable_spectrum="logspace", device="cpu"):
        if stable_min is None:
            stable_min = 10.0 / float(kappa)
        self.d = int(d)
        self.kappa = float(kappa)
        self.lambda_unstable = float(lambda_unstable)
        self.stable_min = float(stable_min)
        self.stable_max = float(stable_max)
        self.stable_spectrum = stable_spectrum
        self.device = device
        q, _ = np.linalg.qr(np.random.RandomState(rotation_seed).randn(d, d))
        self.Q = torch.tensor(q, dtype=torch.float32, device=device)
        if stable_spectrum != "logspace":
            raise ValueError("Only stable_spectrum='logspace' is supported.")
        stable = torch.logspace(
            np.log10(self.stable_min),
            np.log10(self.stable_max),
            steps=d - 1,
            device=device,
            dtype=torch.float32,
        )
        self.eigs = torch.empty((d,), device=device, dtype=torch.float32)
        self.eigs[0] = self.lambda_unstable
        self.eigs[1:] = stable
        self.H = self.Q.T @ torch.diag(self.eigs) @ self.Q

    def __call__(self, x):
        y = self.Q @ x
        return 0.5 * torch.sum(self.eigs * y * y)

    def gradient(self, x):
        y = self.Q @ x
        return self.Q.T @ (self.eigs * y)

    def hessian(self, x):
        return self.H


def run_trial(cls, func, x0, V0, kwargs, max_iter=3000, tol=1e-4):
    opt = cls(func=func, **kwargs)
    x, V = x0.clone(), V0.clone()
    last_info = {}
    for t in range(max_iter):
        x, V, last_info = opt.step(x, V)
        gn = true_grad_norm(func, x)
        if gn != gn:
            return {
                "converged": False,
                "iterations": max_iter,
                "func_evals": int(getattr(opt, "func_evals", last_info.get("func_evals", 0))),
                "true_grad_norm": float("inf"),
                "hessian_index": -1,
                "index_ok": False,
                "grad_ok": False,
                "min_hessian_eval": float("nan"),
                "max_hessian_eval": float("nan"),
            }
        if gn < tol:
            v = final_validation(func, x, target_index=kwargs.get("k", 1), grad_tol=tol)
            return {"converged": bool(v["grad_ok"] and v["index_ok"]), "iterations": t + 1,
                    "func_evals": int(getattr(opt, "func_evals", last_info.get("func_evals", 0))), **v}
    v = final_validation(func, x, target_index=kwargs.get("k", 1), grad_tol=tol)
    return {"converged": False, "iterations": max_iter,
            "func_evals": int(getattr(opt, "func_evals", last_info.get("func_evals", 0))), **v}


def make_initial_state(func, start_seed, unstable_scale=0.5, stable_scale=0.10, v0_mode="random_coordinate_biased"):
    set_trial_seed(start_seed)
    d = func.d
    device = func.Q.device
    y0 = torch.zeros(d, device=device, dtype=torch.float32)
    sign = 1.0 if torch.rand(()) < 0.5 else -1.0
    y0[0] = sign * unstable_scale * (0.5 + torch.abs(torch.randn((), device=device)))
    if d > 1:
        y0[1:] = stable_scale * torch.randn(d - 1, device=device, dtype=torch.float32)
    x0 = func.Q.T @ y0
    if v0_mode == "random_coordinate_biased":
        v0 = torch.randn(d, device=device, dtype=torch.float32)
        v0[0] = torch.abs(v0[0]) + 0.5
    else:
        v0 = torch.randn(d, device=device, dtype=torch.float32)
    v0 = v0 / torch.norm(v0)
    return x0, v0.unsqueeze(0), y0


def find_best_hisd_eta(func, d, device, *, max_iter, tol, rotation_seed, kappa, n_inner, eta_v, l, n_grad_samples, n_hv_samples, n_quick=5, unstable_scale=0.5, stable_scale=0.10):
    etas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    base_kwargs = dict(k=1, eta_v=eta_v, n_inner=n_inner, l=l, n_grad_samples=n_grad_samples, n_hv_samples=n_hv_samples, device=device)
    best = {
        "eta": 0.05, "success_count": -1, "median_iterations": float("inf"),
        "total_final_grad_norm": float("inf"), "probe_scores": [],
    }
    for eta in etas:
        probe_rows = []
        for s in range(n_quick):
            probe_seed = 900000 + 10000 * int(rotation_seed) + 100 * s + int(kappa)
            x0, V0, _ = make_initial_state(func, probe_seed, unstable_scale=unstable_scale, stable_scale=stable_scale)
            set_trial_seed(probe_seed)
            r = run_trial(
                DerivativeFreeHiSD,
                func,
                x0,
                V0,
                {**base_kwargs, "eta": eta},
                max_iter=max_iter,
                tol=tol,
            )
            probe_rows.append(r)
        success_rows = [r for r in probe_rows if is_true(r.get("converged"))]
        succ = len(success_rows)
        med_iter = float(np.median([int(float(r["iterations"])) for r in success_rows])) if succ > 0 else float("inf")
        total_grad = float(sum(float(r["true_grad_norm"]) for r in probe_rows))
        best["probe_scores"].append({"eta": eta, "success_count": succ, "median_iterations": med_iter, "total_final_grad_norm": total_grad})
        better = False
        if succ > best["success_count"]:
            better = True
        elif succ == best["success_count"]:
            if succ > 0 and med_iter < best["median_iterations"]:
                better = True
            elif succ > 0 and med_iter == best["median_iterations"] and total_grad < best["total_final_grad_norm"]:
                better = True
            elif total_grad < best["total_final_grad_norm"]:
                better = True
            elif total_grad == best["total_final_grad_norm"] and eta < best["eta"]:
                better = True
        if better:
            best["eta"] = eta
            best["success_count"] = succ
            best["median_iterations"] = med_iter
            best["total_final_grad_norm"] = total_grad
    return best["eta"], {"probe_success_count": best["success_count"], "probe_median_iterations": best["median_iterations"], "probe_total_final_grad_norm": best["total_final_grad_norm"], "n_quick": n_quick, "probe_scores": best["probe_scores"]}


def expected_func_evals_per_iter(n_grad_samples, n_hv_samples, n_inner, k=1):
    return 2 * n_grad_samples + 4 * k * n_hv_samples * n_inner


def make_row(exp_name, d, kappa, stable_spectrum, stable_min, stable_max, rotation_seed, rotation_id, start_id, trial_id, start_seed, run_seed, method, eta, hisd_eta, eta_v, n_inner, n_grad_samples, n_hv_samples, l, tol, max_iter, x0_mode, unstable_scale, stable_scale, x0_norm, y0_unstable_abs, y0_stable_norm, run_result):
    return {
        "experiment": exp_name, "d": d, "kappa": int(kappa), "stable_spectrum": stable_spectrum, "stable_min": stable_min,
        "stable_max": stable_max, "rotation_seed": rotation_seed, "rotation_id": rotation_id, "start_id": start_id, "trial": trial_id,
        "start_seed": start_seed, "run_seed": run_seed, "method": method, "eta": eta,
        "hisd_sweep_eta": hisd_eta if method == "DF-HiSD" else "",
        "eta_v": eta_v, "n_inner": n_inner, "n_grad_samples": n_grad_samples, "n_hv_samples": n_hv_samples, "l": l,
        "tol": tol, "max_iter": max_iter, "x0_mode": x0_mode, "unstable_scale": unstable_scale, "stable_scale": stable_scale,
        "x0_norm": x0_norm, "y0_unstable_abs": y0_unstable_abs, "y0_stable_norm": y0_stable_norm, **run_result,
    }


def paired_h_fc(rows_h, rows_fc, total_trials):
    h_lookup = {int(float(r["trial"])): r for r in rows_h}
    fc_lookup = {int(float(r["trial"])): r for r in rows_fc}
    h_only = fc_only = common = h_faster = fc_faster = 0
    diffs = []
    for trial in range(total_trials):
        h_row = h_lookup.get(trial)
        fc_row = fc_lookup.get(trial)
        if h_row is None or fc_row is None:
            continue
        h_ok = is_true(h_row.get("converged"))
        fc_ok = is_true(fc_row.get("converged"))
        if h_ok and fc_ok:
            common += 1
            h_iter = int(float(h_row["iterations"]))
            fc_iter = int(float(fc_row["iterations"]))
            if h_iter < fc_iter:
                h_faster += 1
            elif fc_iter < h_iter:
                fc_faster += 1
            diffs.append(fc_iter - h_iter)
        elif h_ok:
            h_only += 1
        elif fc_ok:
            fc_only += 1
    return {
        "h_only": h_only, "fc_only": fc_only, "common": common, "h_faster": h_faster, "fc_faster": fc_faster,
        "median_fc_minus_h_iterations": float(np.median(diffs)) if diffs else float("nan"),
    }


def print_summary(reports):
    print("\n" + "=" * 80)
    hdr = f"{'kappa':>6} | {'DF-ADAM-H':>25} | {'DF-ADAM-FC':>25} | {'DF-HiSD':>25}"
    print(hdr)
    print("-" * 80)
    for r in reports:
        kappa = r["kappa"]
        res = r["methods"]
        parts = []
        for name in ["DF-ADAM-H", "DF-ADAM-FC", "DF-HiSD"]:
            if res[name]["successes"] == 0:
                parts.append(f"{int(res[name]['rate'])}% [0]")
            else:
                parts.append(f"{res[name]['median']} +/- {res[name]['std']} ({res[name]['rate']:.0f}%)")
        print(f"{kappa:>6} | {parts[0]:>25} | {parts[1]:>25} | {parts[2]:>25}")
    print("\nPaired H vs FC diagnostics")
    for r in reports:
        p = r["pair"]
        print(f"kappa={r['kappa']} | H-only={p['h_only']}, FC-only={p['fc_only']}, common={p['common']}, H faster={p['h_faster']}, FC faster={p['fc_faster']}, median(FC-H)={p['median_fc_minus_h_iterations']}")


def experiment_randomly_rotated(device="cpu", kappas=None, rotation_seeds=None, starts_per_rotation=5, max_iter=5000, tol=1e-4, stable_spectrum="logspace", stable_max=0.2, unstable_scale=0.5, stable_scale=0.10, adam_eta=0.02, eta_v=0.01, n_inner=5, n_grad_samples=5, n_hv_samples=3, l=1e-4, hisd_sweep_probes=5, results_dir=None):
    if kappas is None:
        kappas = [100, 500]
    if rotation_seeds is None:
        rotation_seeds = [123, 456, 789]
    methods = (("DF-ADAM-H", DerivativeFreeADAMH), ("DF-ADAM-FC", DerivativeFreeADAMFC), ("DF-HiSD", DerivativeFreeHiSD))

    print("=" * 80)
    print("Randomly Rotated Anisotropic Quadratic Saddle: d=10")
    print(f"stable_max={stable_max}, unstable_scale={unstable_scale}, stable_scale={stable_scale}, adam_eta={adam_eta}, max_iter={max_iter}")
    print("=" * 80)

    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "result", "7.7")
    output_path = os.path.join(results_dir, "exp10_randomly_rotated.csv")
    raw_rows = []
    per_kappa_reports = []
    eval_budget = expected_func_evals_per_iter(n_grad_samples, n_hv_samples, n_inner, k=1)

    for kappa in kappas:
        stable_min = 10.0 / float(kappa)
        total_trials = len(rotation_seeds) * starts_per_rotation
        rows_by_method = {name: [] for name, _ in methods}
        print(f"\n--- kappa = {kappa} (stable spectrum={stable_spectrum}) ---")

        for rotation_id, rotation_seed in enumerate(rotation_seeds):
            func = RandomlyRotatedQuadraticSaddleFunc(
                d=10, kappa=kappa, stable_spectrum=stable_spectrum, stable_min=stable_min,
                stable_max=stable_max, rotation_seed=rotation_seed, device=device,
            )
            hisd_eta, hisd_diag = find_best_hisd_eta(
                func,
                d=10,
                device=device,
                max_iter=max_iter,
                tol=tol,
                rotation_seed=rotation_seed,
                kappa=kappa,
                n_inner=n_inner,
                eta_v=eta_v,
                l=l,
                n_grad_samples=n_grad_samples,
                n_hv_samples=n_hv_samples,
                n_quick=hisd_sweep_probes,
                unstable_scale=unstable_scale,
                stable_scale=stable_scale,
            )
            print(f"  rotation_seed {rotation_seed}: best HiSD eta={hisd_eta}, probe success={hisd_diag['probe_success_count']}")

            base_kwargs = dict(k=1, eta_v=eta_v, n_inner=n_inner, l=l, n_grad_samples=n_grad_samples, n_hv_samples=n_hv_samples, device=device)
            method_specs = {
                "DF-ADAM-H": {**base_kwargs, "eta": adam_eta},
                "DF-ADAM-FC": {**base_kwargs, "eta": adam_eta},
                "DF-HiSD": {**base_kwargs, "eta": hisd_eta},
            }

            for start_id in range(starts_per_rotation):
                trial_id = rotation_id * starts_per_rotation + start_id
                start_seed = 100000 + 1000 * rotation_id + 100 * start_id + int(kappa)
                run_seed = start_seed
                x0, V0, y0 = make_initial_state(func, start_seed, unstable_scale=unstable_scale, stable_scale=stable_scale)
                x0_norm = float(torch.norm(x0).item())
                y0_unstable_abs = float(torch.abs(y0[0]).item())
                y0_stable_norm = float(torch.norm(y0[1:]).item()) if func.d > 1 else 0.0

                for method_name, cls in methods:
                    set_trial_seed(run_seed)
                    r = run_trial(cls, func, x0, V0, method_specs[method_name], max_iter=max_iter, tol=tol)
                    status = "FAIL" if not r["converged"] else f"{int(r['iterations']):4d}"
                    print(f"  {method_name} trial {trial_id + 1:2d}/{total_trials} seed {run_seed}: {status}")
                    row = make_row(
                        "randomly_rotated_anisotropic_quadratic",
                        10,
                        kappa,
                        stable_spectrum,
                        stable_min,
                        stable_max,
                        rotation_seed,
                        rotation_id,
                        start_id,
                        trial_id,
                        start_seed,
                        run_seed,
                        method_name,
                        method_specs[method_name]["eta"],
                        hisd_eta,
                        eta_v,
                        n_inner,
                        n_grad_samples,
                        n_hv_samples,
                        l,
                        tol,
                        max_iter,
                        "eigenbasis_pollution",
                        unstable_scale,
                        stable_scale,
                        x0_norm,
                        y0_unstable_abs,
                        y0_stable_norm,
                        r,
                    )
                    rows_by_method[method_name].append(row)
                    raw_rows.append(row)

        method_results = {}
        for name, _ in methods:
            method_rows = rows_by_method[name]
            summary = summarize_success(method_rows)

            successful_iterations = [
                 int(float(row["iterations"]))
                 for row in method_rows
                 if is_true(row.get("converged"))
            ]
            if successful_iterations:
                 summary["std"] = int(round(float(np.std(successful_iterations))))

            method_results[name] = summary
            print(f"  >> {name}: {summary['rate']:.0f}%, median={summary['median']}, std={summary['std']}")
        per_kappa_reports.append({"kappa": int(kappa), "methods": method_results, "pair": paired_h_fc(rows_by_method["DF-ADAM-H"], rows_by_method["DF-ADAM-FC"], total_trials)})

    print_summary(per_kappa_reports)
    write_csv(output_path, raw_rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run the randomly rotated anisotropic quadratic experiment.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--kappas", nargs="+", type=int, default=[100, 500])
    p.add_argument("--rotation-seeds", nargs="+", type=int, default=[123, 456, 789])
    p.add_argument("--starts-per-rotation", type=int, default=5)
    p.add_argument("--max-iter", type=int, default=5000)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--stable-spectrum", default="logspace")
    p.add_argument("--stable-max", type=float, default=0.2)
    p.add_argument("--unstable-scale", type=float, default=0.5)
    p.add_argument("--stable-scale", type=float, default=0.10)
    p.add_argument("--adam-eta", type=float, default=0.02)
    p.add_argument("--eta-v", type=float, default=0.01)
    p.add_argument("--n-inner", type=int, default=5)
    p.add_argument("--n-grad-samples", type=int, default=5)
    p.add_argument("--n-hv-samples", type=int, default=3)
    p.add_argument("--l", type=float, default=1e-4)
    p.add_argument("--hisd-sweep-probes", type=int, default=5)
    p.add_argument("--fresh", action="store_true", help="No-op; scripts are always run fresh.")
    p.add_argument("--results-dir", default=os.path.join(os.path.dirname(__file__), "..", "result", "7.7"))
    args = p.parse_args()
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    experiment_randomly_rotated(
        device=args.device,
        kappas=args.kappas,
        rotation_seeds=args.rotation_seeds,
        starts_per_rotation=args.starts_per_rotation,
        max_iter=args.max_iter,
        tol=args.tol,
        stable_spectrum=args.stable_spectrum,
        stable_max=args.stable_max,
        unstable_scale=args.unstable_scale,
        stable_scale=args.stable_scale,
        adam_eta=args.adam_eta,
        eta_v=args.eta_v,
        n_inner=args.n_inner,
        n_grad_samples=args.n_grad_samples,
        n_hv_samples=args.n_hv_samples,
        l=args.l,
        hisd_sweep_probes=args.hisd_sweep_probes,
        results_dir=args.results_dir,
    )
    print(f"\nFinished: {datetime.now().strftime('%H:%M:%S')}")
