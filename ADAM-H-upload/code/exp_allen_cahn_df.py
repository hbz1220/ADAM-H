"""
Derivative-free Allen-Cahn stress test.

This optional benchmark probes whether ADAM-H's decoupled second moment is more
robust when both gradients and Hessian-vector products are estimated from
function values. It is intentionally small (n=16) because zeroth-order PDE
experiments are function-evaluation intensive.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Dict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from derivative_free_optimizers import DerivativeFreeADAMFC, DerivativeFreeADAMH, DerivativeFreeHiSD
from experiment_utils import final_validation, set_trial_seed, summarize_success, true_grad_norm, write_csv


class AllenCahnFunctionOnly:
    """Function-only periodic Allen-Cahn energy with analytical validation helpers."""

    def __init__(self, n: int = 16, epsilon: float = 0.06):
        self.n = n
        self.d = n
        self.epsilon = epsilon
        self.h = 1.0 / n

    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        diff = torch.roll(u, shifts=-1) - u
        return self.h * (
            0.5 * self.epsilon**2 * torch.sum((diff / self.h) ** 2)
            + 0.25 * torch.sum((u**2 - 1.0) ** 2)
        )

    def gradient(self, u: torch.Tensor) -> torch.Tensor:
        lap = 2.0 * u - torch.roll(u, 1) - torch.roll(u, -1)
        return self.h * (u**3 - u) + (self.epsilon**2 / self.h) * lap

    def hessian(self, u: torch.Tensor) -> torch.Tensor:
        n = self.n
        diag = self.h * (3.0 * u**2 - 1.0) + 2.0 * self.epsilon**2 / self.h
        H = torch.diag(diag)
        off = -self.epsilon**2 / self.h
        for i in range(n):
            H[i, (i - 1) % n] = off
            H[i, (i + 1) % n] = off
        return H


def gram_schmidt(V: torch.Tensor) -> torch.Tensor:
    rows = []
    for i in range(V.shape[0]):
        v = V[i].clone()
        for q in rows:
            v = v - torch.dot(q, v) * q
        rows.append(v / (torch.norm(v) + 1e-10))
    return torch.stack(rows)


def target_index(problem: AllenCahnFunctionOnly) -> int:
    evals = torch.linalg.eigvalsh(problem.hessian(torch.zeros(problem.d)))
    return int((evals < -1e-6).sum().item())


def run_trial(cls, problem, x0, V0, kwargs, max_iter, tol, target_k) -> Dict:
    opt = cls(func=problem, **kwargs)
    x, V = x0.clone(), V0.clone()
    last_info = {}

    for t in range(max_iter):
        x, V, last_info = opt.step(x, V)
        gn = true_grad_norm(problem, x)
        if not np.isfinite(gn):
            break
        if gn < tol:
            validation = final_validation(problem, x, target_index=target_k, grad_tol=tol)
            return {
                "converged": bool(validation["grad_ok"] and validation["index_ok"]),
                "iterations": t + 1,
                "func_evals": int(getattr(opt, "func_evals", last_info.get("func_evals", 0))),
                **validation,
            }

    validation = final_validation(problem, x, target_index=target_k, grad_tol=tol)
    return {
        "converged": False,
        "iterations": max_iter,
        "func_evals": int(getattr(opt, "func_evals", last_info.get("func_evals", 0))),
        **validation,
    }


def experiment_df_allen_cahn(
    n_trials: int = 5,
    seed: int = 42,
    max_iter: int = 900,
    tol: float = 1e-3,
    results_dir: str | None = None,
):
    problem = AllenCahnFunctionOnly(n=16, epsilon=0.06)
    k = target_index(problem)
    common = dict(k=k, eta=0.02, eta_v=0.05, n_inner=3, l=1e-3,
                  n_grad_samples=5, n_hv_samples=3)
    methods = {
        "DF-ADAM-H": (DerivativeFreeADAMH, common),
        "DF-ADAM-FC": (DerivativeFreeADAMFC, common),
        "DF-HiSD": (DerivativeFreeHiSD, {**common, "eta": 0.05}),
    }

    raw_rows = []
    print("=" * 80)
    print("Derivative-free Allen-Cahn stress test")
    print("=" * 80)
    print(f"n={problem.n}, epsilon={problem.epsilon}, target index={k}")

    for name, (cls, kwargs) in methods.items():
        rows = []
        for trial in range(n_trials):
            trial_seed = seed + 1000 * trial
            set_trial_seed(trial_seed)
            x0 = 0.2 * torch.randn(problem.d)
            V0 = gram_schmidt(torch.randn(k, problem.d))
            set_trial_seed(trial_seed)
            r = run_trial(cls, problem, x0, V0, kwargs, max_iter, tol, k)
            row = {
                "experiment": "df_allen_cahn",
                "method": name,
                "trial": trial,
                "seed": trial_seed,
                "n": problem.n,
                "epsilon_ac": problem.epsilon,
                "target_index": k,
                "eta": kwargs["eta"],
                "eta_v": kwargs["eta_v"],
                "n_inner": kwargs["n_inner"],
                "n_grad_samples": kwargs["n_grad_samples"],
                "n_hv_samples": kwargs["n_hv_samples"],
                "l": kwargs["l"],
                **r,
            }
            raw_rows.append(row)
            rows.append(row)
            status = f"{r['iterations']:4d}" if r["converged"] else "FAIL"
            print(f"  {name:10s} trial {trial + 1:2d}: {status} "
                  f"(|g|={r['true_grad_norm']:.2e}, index={r['hessian_index']})")
        summary = summarize_success(rows)
        print(f"  >> {name}: {summary['rate']:.0f}% success, median={summary['median']}")

    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    write_csv(os.path.join(results_dir, "exp_allen_cahn_df.csv"), raw_rows)
    return raw_rows


if __name__ == "__main__":
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    experiment_df_allen_cahn()
    print(f"\nFinished: {datetime.now().strftime('%H:%M:%S')}")
