"""
Exact-gradient Allen-Cahn saddle-search experiment.

This script adds a small but genuine scientific-computing benchmark to the
ADAM-H package. The 1D periodic Allen-Cahn energy is discretized by finite
differences. With the default parameters, the homogeneous state u=0 is an
index-3 saddle point.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exact_adamh_baselines import ExactAdaptiveHiSD, gram_schmidt
from experiment_utils import final_validation, set_trial_seed, summarize_success, true_grad_norm, write_csv


class AllenCahn1D:
    """Periodic finite-difference Allen-Cahn energy on [0, 1]."""

    def __init__(self, n: int = 64, epsilon: float = 0.10, device: str = "cpu"):
        self.n = n
        self.d = n
        self.epsilon = epsilon
        self.device = device
        self.h = 1.0 / n

    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        diff = torch.roll(u, shifts=-1) - u
        grad_energy = 0.5 * self.epsilon**2 * torch.sum((diff / self.h) ** 2)
        potential = 0.25 * torch.sum((u**2 - 1.0) ** 2)
        return self.h * (grad_energy + potential)

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


def run_trial(
    mode: str,
    problem: AllenCahn1D,
    x0: torch.Tensor,
    V0: torch.Tensor,
    kwargs: Dict,
    max_iter: int,
    tol: float,
    target_index: int,
) -> Dict:
    opt = ExactAdaptiveHiSD(mode=mode, **kwargs)
    x, V = x0.clone(), V0.clone()
    last_info = {}

    for t in range(max_iter):
        x, V, last_info = opt.step(problem, x, V)
        gn = true_grad_norm(problem, x)
        if not np.isfinite(gn):
            return {
                "converged": False,
                "iterations": max_iter,
                "true_grad_norm": float("inf"),
                "hessian_index": -1,
                "index_ok": False,
                "grad_ok": False,
                "min_hessian_eval": float("nan"),
                "max_hessian_eval": float("nan"),
            }
        if gn < tol:
            validation = final_validation(problem, x, target_index=target_index, grad_tol=tol)
            return {
                "converged": bool(validation["grad_ok"] and validation["index_ok"]),
                "iterations": t + 1,
                **validation,
            }

    validation = final_validation(problem, x, target_index=target_index, grad_tol=tol)
    return {
        "converged": False,
        "iterations": max_iter,
        **validation,
    }


def experiment_allen_cahn_1d(
    n_trials: int = 8,
    seed: int = 42,
    device: str = "cpu",
    results_dir: str | None = None,
) -> Tuple[Dict, list]:
    torch.set_default_dtype(torch.float64)
    problem = AllenCahn1D(n=64, epsilon=0.10, device=device)
    target_index = 3
    max_iter = 3000
    tol = 1e-6

    common = dict(k=target_index, eta=0.01, eta_v=0.05, n_inner=5,
                  beta1=0.9, beta2=0.999, epsilon=1e-8, clip_threshold=0.25)

    methods = {
        "HiSD": ("hisd", {**common, "eta": 0.02, "clip_threshold": 0.25}),
        "A-HiSD": ("ahisd", {**common, "eta": 0.10, "momentum": 0.9,
                             "clip_threshold": 0.25}),
        "ADAM-FC": ("fc", common),
        "ADAM-H": ("adamh", common),
    }

    raw_rows = []
    print("=" * 80)
    print("Allen-Cahn 1D exact-gradient index-3 saddle experiment")
    print("=" * 80)

    for name, (mode, kwargs) in methods.items():
        for trial in range(n_trials):
            trial_seed = seed + 1000 * trial
            set_trial_seed(trial_seed)
            x0 = 0.15 * torch.randn(problem.d, dtype=torch.float64, device=device)
            V0 = gram_schmidt(torch.randn(target_index, problem.d, dtype=torch.float64, device=device))
            set_trial_seed(trial_seed)
            r = run_trial(mode, problem, x0, V0, kwargs, max_iter, tol, target_index)
            row = {
                "experiment": "allen_cahn_1d",
                "method": name,
                "trial": trial,
                "seed": trial_seed,
                "n": problem.n,
                "epsilon_ac": problem.epsilon,
                "target_index": target_index,
                "eta": kwargs["eta"],
                "eta_v": kwargs["eta_v"],
                "n_inner": kwargs["n_inner"],
                "momentum": kwargs.get("momentum", ""),
                **r,
            }
            raw_rows.append(row)
            status = f"{r['iterations']:4d}" if r["converged"] else "FAIL"
            print(f"  {name:8s} trial {trial + 1:2d}: {status} "
                  f"(|g|={r['true_grad_norm']:.2e}, index={r['hessian_index']})")

    summary = {}
    print("\nSummary")
    for name in methods:
        rows = [r for r in raw_rows if r["method"] == name]
        s = summarize_success(rows)
        summary[name] = s
        print(f"  {name:8s}: {s['rate']:.0f}% success, median={s['median']}, std={s['std']}")

    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(__file__), "..", "result", "7.6")
    write_csv(os.path.join(results_dir, "exp_allen_cahn_1d.csv"), raw_rows)
    return summary, raw_rows


if __name__ == "__main__":
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    experiment_allen_cahn_1d()
    print(f"\nFinished: {datetime.now().strftime('%H:%M:%S')}")
