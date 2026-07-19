"""
Experiment 4: Performance under varying stochastic gradient noise (Table 5)

Tests ADAM-H vs ADAM-FC vs HiSD with explicit stochastic noise at kappa=100.

Author: Jin Zhao
"""

import torch
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optimizers import (
    StochasticADAMH, StochasticADAMFC, OriginalHiSD,
    QuadraticSaddle, run_optimization
)
from experiment_utils import final_validation, set_trial_seed, write_csv


def experiment_stochastic_noise(device='cpu', results_dir=None):
    d = 10
    kappa = 100
    lambda1 = -10.0
    lambda2 = abs(lambda1) / kappa
    max_iter = 5000
    theta_deg = 18.0

    sigmas = [0.0, 0.01, 0.05, 0.1]
    tolerances = [1e-5, 5e-3, 2.5e-2, 5e-2]

    problem = QuadraticSaddle(d=d, lambda1=lambda1, lambda2=lambda2, device=device)

    x0 = torch.tensor([0.5, 1.0] + [0.1]*8, device=device, dtype=torch.float32)
    theta = theta_deg * np.pi / 180
    v0 = torch.zeros(d, device=device)
    v0[0] = np.cos(theta)
    v0[1] = np.sin(theta)
    V0 = v0.unsqueeze(0)

    print("=" * 70)
    print(f"Stochastic Noise Comparison: kappa={kappa}, d={d}")
    print("=" * 70)
    print(f"{'sigma':>8} | {'Tolerance':>12} | {'ADAM-H':>10} | {'ADAM-FC':>10} | {'HiSD':>10}")
    print("-" * 60)
    raw_rows = []

    for sigma, tol in zip(sigmas, tolerances):
        # ADAM-H
        set_trial_seed(42)
        opt_H = StochasticADAMH(k=1, eta=0.01, eta_v=0.01, n_inner=5, device=device)
        res_H = run_optimization(opt_H, problem, x0.clone(), V0.clone(),
                                 sigma=sigma, max_iter=max_iter, tol=tol)

        # ADAM-FC
        set_trial_seed(42)
        opt_FC = StochasticADAMFC(k=1, eta=0.01, eta_v=0.01, n_inner=5, device=device)
        res_FC = run_optimization(opt_FC, problem, x0.clone(), V0.clone(),
                                  sigma=sigma, max_iter=max_iter, tol=tol)

        # HiSD
        set_trial_seed(42)
        opt_O = OriginalHiSD(k=1, eta=0.01, eta_v=0.01, n_inner=5, device=device)
        res_O = run_optimization(opt_O, problem, x0.clone(), V0.clone(),
                                 sigma=sigma, max_iter=max_iter, tol=tol)

        def fmt(r):
            return str(r['iterations']) if r['converged'] else "FAIL"

        print(f"{sigma:>8.2f} | {tol:>12.1e} | {fmt(res_H):>10} | {fmt(res_FC):>10} | {fmt(res_O):>10}")

        for method, res in [("ADAM-H", res_H), ("ADAM-FC", res_FC), ("HiSD", res_O)]:
            validation = final_validation(problem, res["x_final"], target_index=1, grad_tol=tol)
            raw_rows.append({
                "experiment": "stochastic_noise",
                "method": method,
                "trial": 0,
                "seed": 42,
                "sigma": sigma,
                "tol": tol,
                "max_iter": max_iter,
                "iterations": res["iterations"],
                "converged": bool(validation["grad_ok"] and validation["index_ok"]),
                **validation,
            })

        if results_dir is None:
            results_dir = os.path.join(os.path.dirname(__file__), "..", "result", "7.4")
        write_csv(os.path.join(results_dir, "exp4_stochastic_noise.csv"), raw_rows)


if __name__ == "__main__":
    experiment_stochastic_noise()
