"""Numerical experiments for Section 7.5."""
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "result", "7.5")
)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from derivative_free_optimizers import (  # noqa: E402
    DerivativeFreeADAMFC,
    DerivativeFreeADAMH,
    DerivativeFreeHiSD,
)
from experiment_utils import final_validation, set_trial_seed, write_csv  # noqa: E402


ETA_CANDIDATES = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
HISD_ETA_SELECTION_RULE = (
    "max certified success_count; min median_success_iterations; "
    "min total_final_grad_norm; min eta"
)
COMMON_OPTIMIZER_KWARGS = dict(
    eta_v=0.01,
    n_inner=5,
    l=1e-4,
    n_grad_samples=5,
    n_hv_samples=3,
)


def read_existing_rows(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def truthy(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def select_hisd_eta(candidate_summaries):
    def selection_key(summary):
        median_iterations = float(summary["median_success_iterations"])
        total_grad_norm = float(summary["total_final_grad_norm"])
        if not np.isfinite(median_iterations):
            median_iterations = float("inf")
        if not np.isfinite(total_grad_norm):
            total_grad_norm = float("inf")
        return (
            -int(summary["success_count"]),
            median_iterations,
            total_grad_norm,
            float(summary["eta"]),
        )

    if not candidate_summaries:
        raise ValueError("candidate_summaries must not be empty")
    return min(candidate_summaries, key=selection_key)


def run_main_trial(cls, func, x0, V0, kwargs, max_iter, tol):
    opt = cls(func=func, **kwargs)
    x, V = x0.clone(), V0.clone()
    last_info = {}
    for t in range(max_iter):
        x, V, last_info = opt.step(x, V)
        gn = torch.norm(func.gradient(x)).item()
        if gn != gn:
            return {
                "converged": False,
                "iterations": max_iter,
                "g_norm": float("inf"),
                "func_evals": int(
                    getattr(opt, "func_evals", last_info.get("func_evals", 0))
                ),
                "true_grad_norm": float("inf"),
                "hessian_index": -1,
                "index_ok": False,
                "grad_ok": False,
            }
        if gn < tol:
            validation = final_validation(
                func, x, target_index=kwargs.get("k", 1), grad_tol=tol
            )
            return {
                "converged": bool(validation["grad_ok"] and validation["index_ok"]),
                "iterations": t + 1,
                "g_norm": gn,
                "func_evals": int(
                    getattr(opt, "func_evals", last_info.get("func_evals", 0))
                ),
                **validation,
            }
    validation = final_validation(
        func, x, target_index=kwargs.get("k", 1), grad_tol=tol
    )
    return {
        "converged": False,
        "iterations": max_iter,
        "g_norm": gn,
        "func_evals": int(
            getattr(opt, "func_evals", last_info.get("func_evals", 0))
        ),
        **validation,
    }


def run_stepsize_trial(cls, func, x0, V0, kwargs, max_iter, tol):
    opt = cls(func=func, **kwargs)
    x, V = x0.clone(), V0.clone()
    last_info = {}
    for t in range(max_iter):
        x, V, last_info = opt.step(x, V)
        gn = torch.norm(func.gradient(x)).item()
        if not np.isfinite(gn):
            return {
                "converged": False,
                "iterations": t + 1,
                "g_norm": float("inf"),
                "func_evals": int(
                    getattr(opt, "func_evals", last_info.get("func_evals", 0))
                ),
                "hessian_index": -1,
                "index_ok": False,
                "grad_ok": False,
                "true_grad_norm": float("inf"),
                "min_hessian_eval": float("nan"),
                "max_hessian_eval": float("nan"),
                "failure_reason": "nan_or_inf_grad_norm",
            }
        if gn < tol:
            validation = final_validation(
                func, x, target_index=kwargs.get("k", 1), grad_tol=tol
            )
            return {
                "converged": bool(validation["grad_ok"] and validation["index_ok"]),
                "iterations": t + 1,
                "g_norm": gn,
                "func_evals": int(
                    getattr(opt, "func_evals", last_info.get("func_evals", 0))
                ),
                **validation,
            }
    validation = final_validation(
        func, x, target_index=kwargs.get("k", 1), grad_tol=tol
    )
    return {
        "converged": False,
        "iterations": max_iter,
        "g_norm": gn,
        "func_evals": int(
            getattr(opt, "func_evals", last_info.get("func_evals", 0))
        ),
        **validation,
    }


def run_hisd_pilot(
    func,
    pilot_points,
    k,
    kappa,
    common_kwargs,
    max_iter,
    tol,
    experiment_label,
    panel_label,
):
    pilot_rows = []
    summaries = []
    for eta in ETA_CANDIDATES:
        kwargs = {**common_kwargs, "eta": eta, "k": k}
        eta_rows = []
        for pilot_id, (x0, V0, init_seed, run_seed) in enumerate(pilot_points):
            set_trial_seed(run_seed)
            result = run_main_trial(
                DerivativeFreeHiSD,
                func,
                x0.clone(),
                V0.clone(),
                kwargs,
                max_iter,
                tol,
            )
            row = {
                "experiment": experiment_label,
                "panel": panel_label,
                "k": k,
                "kappa": kappa,
                "eta": eta,
                "pilot_trial": pilot_id,
                "pilot_init_seed": init_seed,
                "pilot_run_seed": run_seed,
                "converged": result["converged"],
                "iterations": result["iterations"],
                "true_grad_norm": result["true_grad_norm"],
                "hessian_index": result["hessian_index"],
                "index_ok": result["index_ok"],
                "grad_ok": result["grad_ok"],
                "max_iter": max_iter,
                "tol": tol,
                "eta_v": kwargs["eta_v"],
                "n_inner": kwargs["n_inner"],
                "n_grad_samples": kwargs["n_grad_samples"],
                "n_hv_samples": kwargs["n_hv_samples"],
                "l": kwargs["l"],
            }
            pilot_rows.append(row)
            eta_rows.append(row)

        successful_iterations = [
            int(row["iterations"]) for row in eta_rows if truthy(row["converged"])
        ]
        median_iterations = (
            float(np.median(successful_iterations))
            if successful_iterations
            else float("inf")
        )
        grad_norms = []
        for row in eta_rows:
            grad_norm = float(row["true_grad_norm"])
            grad_norms.append(
                grad_norm if np.isfinite(grad_norm) else float("inf")
            )
        summaries.append({
            "experiment": experiment_label,
            "panel": panel_label,
            "k": k,
            "kappa": kappa,
            "eta": eta,
            "pilot_n": len(pilot_points),
            "success_count": len(successful_iterations),
            "median_success_iterations": median_iterations,
            "total_final_grad_norm": sum(grad_norms),
        })

    selected = select_hisd_eta(summaries)
    selected_eta = float(selected["eta"])
    for summary in summaries:
        key = (
            -int(summary["success_count"]),
            float(summary["median_success_iterations"]),
            float(summary["total_final_grad_norm"]),
            float(summary["eta"]),
        )
        summary["selection_key"] = str(key)
        summary["selected"] = float(summary["eta"]) == selected_eta
    return pilot_rows, summaries, selected


def main_method_specs(common_kwargs, k, hisd_eta):
    return {
        "DF-ADAM-H": (
            DerivativeFreeADAMH,
            {**common_kwargs, "k": k, "eta": 0.01},
        ),
        "DF-ADAM-FC": (
            DerivativeFreeADAMFC,
            {**common_kwargs, "k": k, "eta": 0.01},
        ),
        "DF-HiSD": (
            DerivativeFreeHiSD,
            {**common_kwargs, "k": k, "eta": hisd_eta},
        ),
    }


def summarize_main_trials(rows, n_trials):
    converged = [row for row in rows if truthy(row.get("converged"))]
    if converged:
        iterations = [int(float(row["iterations"])) for row in converged]
        median = int(np.median(iterations))
        std = int(np.std(iterations))
    else:
        median = 0
        std = 0
    return {
        "rate": len(converged) / n_trials * 100,
        "median": median,
        "std": std,
    }


def summarize_stepsize_rows(rows, n_trials):
    converged = [row for row in rows if truthy(row.get("converged"))]
    iterations = [int(float(row["iterations"])) for row in converged]
    return {
        "rate": 100.0 * len(converged) / n_trials,
        "median": int(np.median(iterations)) if iterations else 0,
        "std": int(np.std(iterations)) if len(iterations) > 1 else 0,
    }


def orthonormalize_higher_start(V0_raw):
    V0 = torch.zeros_like(V0_raw)
    for direction_id in range(V0_raw.shape[0]):
        v = V0_raw[direction_id].clone()
        for previous_id in range(direction_id):
            v = v - torch.dot(V0[previous_id], v) * V0[previous_id]
        V0[direction_id] = v / (torch.norm(v) + 1e-10)
    return V0


class HigherIndexRotatedSaddleFunc:
    def __init__(self, d=10, k=1, lambda2=0.02, theta_deg=30.0):
        self.d = d
        self.k_index = k
        self.lambda2 = lambda2
        self.kappa = 2.0 / lambda2
        if d < 2 * k:
            raise ValueError(
                "Need d >= 2k to rotate each unstable coordinate "
                "with a distinct stable coordinate."
            )
        theta = theta_deg * np.pi / 180.0
        c, s = np.cos(theta), np.sin(theta)
        Q = torch.eye(d, dtype=torch.float64)
        for i in range(k):
            j = k + i
            G = torch.eye(d, dtype=torch.float64)
            G[i, i] = c
            G[i, j] = -s
            G[j, i] = s
            G[j, j] = c
            Q = G @ Q
        self.Q = Q.float()
        self.eig_rotated = torch.zeros(d)
        self.eig_rotated[:k] = -2.0
        self.eig_rotated[k:] = lambda2

    def __call__(self, x):
        y = self.Q @ x
        energy = 0.0
        for i in range(self.k_index):
            energy += -y[i]**2
        for i in range(self.k_index, self.d):
            energy += 0.5 * self.lambda2 * y[i]**2
        energy += 0.25 * torch.sum(y**4)
        return energy

    def gradient(self, x):
        y = self.Q @ x
        grad_y = torch.zeros_like(y)
        for i in range(self.k_index):
            grad_y[i] = -2*y[i] + y[i]**3
        for i in range(self.k_index, self.d):
            grad_y[i] = self.lambda2 * y[i] + y[i]**3
        return self.Q.T @ grad_y


class RotatedQuarticSaddleFunc:
    def __init__(self, d=10, lambda2=0.02, theta_deg=30.0):
        self.d = d
        self.lambda2 = lambda2
        self.kappa = 2.0 / lambda2
        theta = theta_deg * np.pi / 180.0
        Q = torch.eye(d, dtype=torch.float64)
        Q[0, 0] = np.cos(theta)
        Q[0, 1] = -np.sin(theta)
        Q[1, 0] = np.sin(theta)
        Q[1, 1] = np.cos(theta)
        self.Q = Q.float()

    def __call__(self, x):
        y = self.Q @ x
        return (
            -y[0]**2
            + 0.5 * self.lambda2 * torch.sum(y[1:]**2)
            + 0.25 * torch.sum(y**4)
        )

    def gradient(self, x):
        y = self.Q @ x
        grad_y = torch.zeros_like(y)
        grad_y[0] = -2*y[0] + y[0]**3
        grad_y[1:] = self.lambda2 * y[1:] + y[1:]**3
        return self.Q.T @ grad_y


def same_higher_setting(
    row,
    *,
    d,
    kappa,
    theta_deg,
    index_k,
    method,
    trial,
    init_seed,
    run_seed,
    eta,
    eta_v,
    n_inner,
    n_grad_samples,
    n_hv_samples,
    l,
    tol,
    max_iter,
    rotation_scheme,
    hisd_eta_selected,
    hisd_pilot_n,
    hisd_pilot_success_count,
    hisd_eta_selection_rule,
):
    try:
        return (
            str(row.get("experiment")) == "higher_index_rotated"
            and int(float(row.get("d", -1))) == int(d)
            and int(float(row.get("kappa", -1))) == int(kappa)
            and abs(float(row.get("theta_deg", "nan")) - float(theta_deg)) < 1e-12
            and int(float(row.get("index_k", -1))) == int(index_k)
            and str(row.get("method")) == method
            and int(float(row.get("trial", -1))) == int(trial)
            and int(float(row.get("init_seed", -1))) == int(init_seed)
            and int(float(row.get("run_seed", -1))) == int(run_seed)
            and abs(float(row.get("eta", "nan")) - float(eta)) < 1e-12
            and abs(float(row.get("eta_v", "nan")) - float(eta_v)) < 1e-12
            and int(float(row.get("n_inner", -1))) == int(n_inner)
            and int(float(row.get("n_grad_samples", -1))) == int(n_grad_samples)
            and int(float(row.get("n_hv_samples", -1))) == int(n_hv_samples)
            and abs(float(row.get("l", "nan")) - float(l)) < 1e-12
            and abs(float(row.get("tol", "nan")) - float(tol)) < 1e-12
            and int(float(row.get("max_iter", -1))) == int(max_iter)
            and str(row.get("rotation_scheme")) == rotation_scheme
            and abs(
                float(row.get("hisd_eta_selected", "nan"))
                - float(hisd_eta_selected)
            ) < 1e-12
            and int(float(row.get("hisd_pilot_n", -1))) == int(hisd_pilot_n)
            and int(float(row.get("hisd_pilot_success_count", -1)))
            == int(hisd_pilot_success_count)
            and str(row.get("hisd_eta_selection_rule"))
            == hisd_eta_selection_rule
        )
    except (TypeError, ValueError):
        return False


def same_condition_setting(
    row,
    *,
    d,
    kappa,
    theta_deg,
    method,
    trial,
    init_seed,
    run_seed,
    eta,
    eta_v,
    n_inner,
    n_grad_samples,
    n_hv_samples,
    l,
    tol,
    max_iter,
    rotation_scheme,
    hisd_eta_selected,
    hisd_pilot_n,
    hisd_pilot_success_count,
    hisd_eta_selection_rule,
):
    try:
        return (
            str(row.get("experiment")) == "rotated_quartic"
            and int(float(row.get("d", -1))) == int(d)
            and int(float(row.get("kappa", -1))) == int(kappa)
            and abs(float(row.get("theta_deg", "nan")) - float(theta_deg)) < 1e-12
            and str(row.get("method")) == method
            and int(float(row.get("trial", -1))) == int(trial)
            and int(float(row.get("init_seed", -1))) == int(init_seed)
            and int(float(row.get("run_seed", -1))) == int(run_seed)
            and abs(float(row.get("eta", "nan")) - float(eta)) < 1e-12
            and abs(float(row.get("eta_v", "nan")) - float(eta_v)) < 1e-12
            and int(float(row.get("n_inner", -1))) == int(n_inner)
            and int(float(row.get("n_grad_samples", -1))) == int(n_grad_samples)
            and int(float(row.get("n_hv_samples", -1))) == int(n_hv_samples)
            and abs(float(row.get("l", "nan")) - float(l)) < 1e-12
            and abs(float(row.get("tol", "nan")) - float(tol)) < 1e-12
            and int(float(row.get("max_iter", -1))) == int(max_iter)
            and str(row.get("rotation_scheme")) == rotation_scheme
            and abs(
                float(row.get("hisd_eta_selected", "nan"))
                - float(hisd_eta_selected)
            ) < 1e-12
            and int(float(row.get("hisd_pilot_n", -1))) == int(hisd_pilot_n)
            and int(float(row.get("hisd_pilot_success_count", -1)))
            == int(hisd_pilot_success_count)
            and str(row.get("hisd_eta_selection_rule"))
            == hisd_eta_selection_rule
        )
    except (TypeError, ValueError):
        return False


def same_stepsize_setting(
    row,
    *,
    d,
    kappa,
    theta_deg,
    k,
    target_index,
    lambda2,
    eta_v,
    n_inner,
    n_grad_samples,
    n_hv_samples,
    l,
    method,
    eta,
    trial,
    seed,
    init_seed,
    tol,
    max_iter,
):
    try:
        return (
            str(row.get("experiment")) == "stepsize_sweep"
            and int(float(row.get("d", -1))) == int(d)
            and int(float(row.get("kappa", -1))) == int(kappa)
            and abs(float(row.get("theta_deg", "nan")) - float(theta_deg)) < 1e-12
            and abs(float(row.get("lambda2", "nan")) - float(lambda2)) < 1e-12
            and int(float(row.get("k", -1))) == int(k)
            and int(float(row.get("target_index", "nan"))) == int(target_index)
            and abs(float(row.get("eta_v", "nan")) - float(eta_v)) < 1e-12
            and int(float(row.get("n_inner", -1))) == int(n_inner)
            and int(float(row.get("n_grad_samples", -1))) == int(n_grad_samples)
            and int(float(row.get("n_hv_samples", -1))) == int(n_hv_samples)
            and abs(float(row.get("l", "nan")) - float(l)) < 1e-12
            and str(row.get("method")) == method
            and abs(float(row.get("eta", "nan")) - float(eta)) < 1e-12
            and int(float(row.get("trial", -1))) == int(trial)
            and int(float(row.get("seed", "nan"))) == int(seed)
            and int(float(row.get("init_seed", "nan"))) == int(init_seed)
            and abs(float(row.get("tol", "nan")) - float(tol)) < 1e-12
            and int(float(row.get("max_iter", -1))) == int(max_iter)
        )
    except ValueError:
        return False


def experiment_higher_index(
    n_trials=20,
    seed=42,
    results_dir=None,
    indices=None,
    resume=True,
    n_pilot=5,
    max_iter=3000,
    tol=1e-3,
):
    d = 10
    kappa = 100
    lambda2 = 2.0 / kappa
    theta_deg = 30.0
    rotation_scheme = "unstable_stable_pair_v1"
    common = dict(COMMON_OPTIMIZER_KWARGS)
    if indices is None:
        indices = [1, 2, 3]
    if n_pilot < 1:
        raise ValueError("n_pilot must be at least 1")
    if results_dir is None:
        results_dir = DEFAULT_RESULTS_DIR

    output_path = os.path.join(results_dir, "exp_higher_index_rotated_usmix.csv")
    pilot_output_path = os.path.join(
        results_dir, "exp_higher_index_rotated_hisd_sweep.csv"
    )
    raw_rows = read_existing_rows(output_path) if resume else []
    all_pilot_rows = []
    all_results = {}

    for k in indices:
        func = HigherIndexRotatedSaddleFunc(d, k, lambda2, theta_deg)
        init_points = []
        for trial in range(n_trials):
            init_seed = seed + k * 10000 + trial
            set_trial_seed(init_seed)
            x0 = torch.randn(d, dtype=torch.float32) * 0.5
            V0_raw = torch.randn(k, d, dtype=torch.float32)
            V0 = orthonormalize_higher_start(V0_raw)
            init_points.append((x0, V0, init_seed))

        pilot_points = []
        for pilot_id in range(n_pilot):
            pilot_init_seed = seed + 1_000_000 + 10_000 * k + pilot_id
            pilot_run_seed = seed + 2_000_000 + 10_000 * k + pilot_id
            set_trial_seed(pilot_init_seed)
            x0 = torch.randn(d, dtype=torch.float32) * 0.5
            V0_raw = torch.randn(k, d, dtype=torch.float32)
            V0 = orthonormalize_higher_start(V0_raw)
            pilot_points.append((x0, V0, pilot_init_seed, pilot_run_seed))

        pilot_rows, _pilot_summaries, selected = run_hisd_pilot(
            func,
            pilot_points,
            k,
            kappa,
            common,
            max_iter,
            tol,
            "higher_index_rotated",
            "higher_index",
        )
        all_pilot_rows.extend(pilot_rows)
        write_csv(pilot_output_path, all_pilot_rows)
        hisd_eta = float(selected["eta"])
        pilot_success_count = int(selected["success_count"])
        methods = main_method_specs(common, k, hisd_eta)

        k_results = {}
        for method, (cls, kwargs) in methods.items():
            trial_results = []
            for trial, (x0, V0, init_seed) in enumerate(init_points):
                run_seed = seed + trial * 1000 + k * 100
                cached = next(
                    (
                        row
                        for row in raw_rows
                        if same_higher_setting(
                            row,
                            d=d,
                            kappa=kappa,
                            theta_deg=theta_deg,
                            index_k=k,
                            method=method,
                            trial=trial,
                            init_seed=init_seed,
                            run_seed=run_seed,
                            eta=kwargs["eta"],
                            eta_v=kwargs["eta_v"],
                            n_inner=kwargs["n_inner"],
                            n_grad_samples=kwargs["n_grad_samples"],
                            n_hv_samples=kwargs["n_hv_samples"],
                            l=kwargs["l"],
                            tol=tol,
                            max_iter=max_iter,
                            rotation_scheme=rotation_scheme,
                            hisd_eta_selected=hisd_eta,
                            hisd_pilot_n=n_pilot,
                            hisd_pilot_success_count=pilot_success_count,
                            hisd_eta_selection_rule=HISD_ETA_SELECTION_RULE,
                        )
                    ),
                    None,
                )
                if cached is not None:
                    trial_results.append(cached)
                    continue

                set_trial_seed(run_seed)
                result = run_main_trial(
                    cls,
                    func,
                    x0.clone(),
                    V0.clone(),
                    kwargs,
                    max_iter,
                    tol,
                )
                trial_results.append(result)
                raw_rows.append({
                    "experiment": "higher_index_rotated",
                    "d": d,
                    "kappa": kappa,
                    "lambda2": lambda2,
                    "theta_deg": theta_deg,
                    "index_k": k,
                    "method": method,
                    "trial": trial,
                    "seed": run_seed,
                    "run_seed": run_seed,
                    "init_seed": init_seed,
                    "eta": kwargs["eta"],
                    "eta_v": kwargs["eta_v"],
                    "n_inner": kwargs["n_inner"],
                    "n_grad_samples": kwargs["n_grad_samples"],
                    "n_hv_samples": kwargs["n_hv_samples"],
                    "l": kwargs["l"],
                    "tol": tol,
                    "max_iter": max_iter,
                    "rotation_scheme": rotation_scheme,
                    "hisd_eta_selected": hisd_eta,
                    "hisd_pilot_n": n_pilot,
                    "hisd_pilot_success_count": pilot_success_count,
                    "hisd_eta_selection_rule": HISD_ETA_SELECTION_RULE,
                    **result,
                })
                write_csv(output_path, raw_rows)
            k_results[method] = summarize_main_trials(trial_results, n_trials)
        all_results[k] = k_results
        print(f"higher-index k={k}: HiSD eta={hisd_eta}")

    write_csv(output_path, raw_rows)
    return all_results


def experiment_condition_number(
    n_trials=20,
    seed=42,
    results_dir=None,
    kappas=None,
    resume=True,
    n_pilot=5,
    max_iter=3000,
    tol=1e-3,
):
    d = 10
    theta_deg = 30.0
    rotation_scheme = "plane_0_1_theta_30_v1"
    lambda2s = {10: 0.2, 100: 0.02, 500: 0.004}
    common = dict(COMMON_OPTIMIZER_KWARGS)
    if kappas is None:
        kappas = [10, 100, 500]
    if n_pilot < 1:
        raise ValueError("n_pilot must be at least 1")
    if results_dir is None:
        results_dir = DEFAULT_RESULTS_DIR

    output_path = os.path.join(results_dir, "exp_rotated_camel.csv")
    pilot_output_path = os.path.join(
        results_dir, "exp_rotated_camel_hisd_sweep.csv"
    )
    raw_rows = read_existing_rows(output_path) if resume else []
    all_pilot_rows = []
    all_results = {}

    for kappa in kappas:
        lambda2 = lambda2s[kappa]
        func = RotatedQuarticSaddleFunc(d, lambda2, theta_deg)
        init_points = []
        for trial in range(n_trials):
            init_seed = seed + kappa * 10000 + trial
            set_trial_seed(init_seed)
            x0 = torch.randn(d, dtype=torch.float32) * 0.5
            theta = np.random.uniform(0, np.pi / 4)
            v0 = torch.zeros(d, dtype=torch.float32)
            v0[0] = np.cos(theta)
            v0[1] = np.sin(theta)
            v0 = v0 / torch.norm(v0)
            V0 = v0.unsqueeze(0)
            init_points.append((x0, V0, init_seed))

        pilot_points = []
        for pilot_id in range(n_pilot):
            pilot_init_seed = seed + 30_000_000 + 10_000 * kappa + pilot_id
            pilot_run_seed = seed + 40_000_000 + 10_000 * kappa + pilot_id
            set_trial_seed(pilot_init_seed)
            x0 = torch.randn(d, dtype=torch.float32) * 0.5
            theta = np.random.uniform(0, np.pi / 4)
            v0 = torch.zeros(d, dtype=torch.float32)
            v0[0] = np.cos(theta)
            v0[1] = np.sin(theta)
            v0 = v0 / torch.norm(v0)
            V0 = v0.unsqueeze(0)
            pilot_points.append((x0, V0, pilot_init_seed, pilot_run_seed))

        pilot_rows, _pilot_summaries, selected = run_hisd_pilot(
            func,
            pilot_points,
            1,
            kappa,
            common,
            max_iter,
            tol,
            "rotated_quartic",
            "condition_number",
        )
        all_pilot_rows.extend(pilot_rows)
        write_csv(pilot_output_path, all_pilot_rows)
        hisd_eta = float(selected["eta"])
        pilot_success_count = int(selected["success_count"])
        methods = main_method_specs(common, 1, hisd_eta)

        kappa_results = {}
        for method, (cls, kwargs) in methods.items():
            trial_results = []
            for trial, (x0, V0, init_seed) in enumerate(init_points):
                run_seed = seed + trial * 1000 + kappa
                cached = next(
                    (
                        row
                        for row in raw_rows
                        if same_condition_setting(
                            row,
                            d=d,
                            kappa=kappa,
                            theta_deg=theta_deg,
                            method=method,
                            trial=trial,
                            init_seed=init_seed,
                            run_seed=run_seed,
                            eta=kwargs["eta"],
                            eta_v=kwargs["eta_v"],
                            n_inner=kwargs["n_inner"],
                            n_grad_samples=kwargs["n_grad_samples"],
                            n_hv_samples=kwargs["n_hv_samples"],
                            l=kwargs["l"],
                            tol=tol,
                            max_iter=max_iter,
                            rotation_scheme=rotation_scheme,
                            hisd_eta_selected=hisd_eta,
                            hisd_pilot_n=n_pilot,
                            hisd_pilot_success_count=pilot_success_count,
                            hisd_eta_selection_rule=HISD_ETA_SELECTION_RULE,
                        )
                    ),
                    None,
                )
                if cached is not None:
                    trial_results.append(cached)
                    continue

                set_trial_seed(run_seed)
                result = run_main_trial(
                    cls,
                    func,
                    x0.clone(),
                    V0.clone(),
                    kwargs,
                    max_iter,
                    tol,
                )
                trial_results.append(result)
                raw_rows.append({
                    "experiment": "rotated_quartic",
                    "d": d,
                    "kappa": kappa,
                    "lambda2": lambda2,
                    "theta_deg": theta_deg,
                    "method": method,
                    "trial": trial,
                    "seed": run_seed,
                    "run_seed": run_seed,
                    "init_seed": init_seed,
                    "eta": kwargs["eta"],
                    "eta_v": kwargs["eta_v"],
                    "n_inner": kwargs["n_inner"],
                    "n_grad_samples": kwargs["n_grad_samples"],
                    "n_hv_samples": kwargs["n_hv_samples"],
                    "l": kwargs["l"],
                    "tol": tol,
                    "max_iter": max_iter,
                    "rotation_scheme": rotation_scheme,
                    "hisd_eta_selected": hisd_eta,
                    "hisd_pilot_n": n_pilot,
                    "hisd_pilot_success_count": pilot_success_count,
                    "hisd_eta_selection_rule": HISD_ETA_SELECTION_RULE,
                    **result,
                })
                write_csv(output_path, raw_rows)
            kappa_results[method] = summarize_main_trials(trial_results, n_trials)
        all_results[kappa] = kappa_results
        print(f"condition-number kappa={kappa}: HiSD eta={hisd_eta}")

    write_csv(output_path, raw_rows)
    return all_results


def experiment_stepsize_sweep(
    n_trials=12,
    seed=42,
    results_dir=None,
    resume=True,
    max_iter=3000,
    tol=1e-3,
):
    d = 10
    kappa = 100
    lambda2 = 0.02
    theta_deg = 30.0
    if results_dir is None:
        results_dir = DEFAULT_RESULTS_DIR
    output_path = os.path.join(results_dir, "exp_stepsize_sweep.csv")
    raw_rows = read_existing_rows(output_path) if resume else []

    func = RotatedQuarticSaddleFunc(d, lambda2, theta_deg)
    init_points: List[Tuple[torch.Tensor, torch.Tensor, int]] = []
    for trial in range(n_trials):
        init_seed = seed + 50000 + trial
        set_trial_seed(init_seed)
        x0 = torch.randn(d, dtype=torch.float32) * 0.5
        theta = np.random.uniform(0, np.pi / 4)
        v0 = torch.zeros(d, dtype=torch.float32)
        v0[0] = np.cos(theta)
        v0[1] = np.sin(theta)
        v0 = v0 / torch.norm(v0)
        init_points.append((x0, v0.unsqueeze(0), init_seed))

    common = {"k": 1, **COMMON_OPTIMIZER_KWARGS}
    method_specs = [
        ("DF-ADAM-H", DerivativeFreeADAMH, ETA_CANDIDATES),
        ("DF-ADAM-FC", DerivativeFreeADAMFC, ETA_CANDIDATES),
        ("DF-HiSD", DerivativeFreeHiSD, ETA_CANDIDATES),
    ]
    results = {}
    for method, cls, etas in method_specs:
        results[method] = {}
        for eta in etas:
            kwargs = {**common, "eta": eta}
            trial_rows = []
            for trial, (x0, V0, init_seed) in enumerate(init_points):
                run_seed = seed + trial * 1000
                cached = next(
                    (
                        row
                        for row in raw_rows
                        if same_stepsize_setting(
                            row,
                            d=d,
                            kappa=kappa,
                            theta_deg=theta_deg,
                            lambda2=lambda2,
                            k=kwargs["k"],
                            target_index=kwargs["k"],
                            eta_v=kwargs["eta_v"],
                            n_inner=kwargs["n_inner"],
                            n_grad_samples=kwargs["n_grad_samples"],
                            n_hv_samples=kwargs["n_hv_samples"],
                            l=kwargs["l"],
                            method=method,
                            eta=eta,
                            trial=trial,
                            seed=run_seed,
                            init_seed=init_seed,
                            tol=tol,
                            max_iter=max_iter,
                        )
                    ),
                    None,
                )
                if cached is not None:
                    trial_rows.append(cached)
                    continue

                set_trial_seed(run_seed)
                result = run_stepsize_trial(
                    cls,
                    func,
                    x0.clone(),
                    V0.clone(),
                    kwargs,
                    max_iter,
                    tol,
                )
                row = {
                    "experiment": "stepsize_sweep",
                    "d": d,
                    "kappa": kappa,
                    "lambda2": lambda2,
                    "k": kwargs["k"],
                    "target_index": kwargs["k"],
                    "theta_deg": theta_deg,
                    "method": method,
                    "eta": eta,
                    "eta_v": kwargs["eta_v"],
                    "n_inner": kwargs["n_inner"],
                    "n_grad_samples": kwargs["n_grad_samples"],
                    "n_hv_samples": kwargs["n_hv_samples"],
                    "l": kwargs["l"],
                    "tol": tol,
                    "max_iter": max_iter,
                    "trial": trial,
                    "seed": run_seed,
                    "init_seed": init_seed,
                    **result,
                }
                trial_rows.append(row)
                raw_rows.append(row)
                write_csv(output_path, raw_rows)
            results[method][eta] = summarize_stepsize_rows(trial_rows, n_trials)

    write_csv(output_path, raw_rows)
    return results


def run_selected_mode(args) -> None:
    resume = not args.fresh
    if args.mode in {"all", "higher-index"}:
        experiment_higher_index(
            n_trials=args.n_trials,
            seed=args.seed,
            results_dir=args.results_dir,
            indices=args.indices,
            resume=resume,
            n_pilot=args.n_pilot,
            max_iter=args.max_iter,
            tol=args.tol,
        )
    if args.mode in {"all", "condition-number"}:
        experiment_condition_number(
            n_trials=args.n_trials,
            seed=args.seed,
            results_dir=args.results_dir,
            kappas=args.kappas,
            resume=resume,
            n_pilot=args.n_pilot,
            max_iter=args.max_iter,
            tol=args.tol,
        )
    if args.mode in {"all", "stepsize"}:
        experiment_stepsize_sweep(
            n_trials=args.aux_trials,
            seed=args.seed,
            results_dir=args.results_dir,
            resume=resume,
            max_iter=args.max_iter,
            tol=args.tol,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Numerical Experiments for Section 7.5."
    )
    parser.add_argument(
        "--mode",
        choices=["all", "higher-index", "condition-number", "stepsize"],
        default="all",
    )
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--aux-trials", type=int, default=12)
    parser.add_argument("--n-pilot", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--indices", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--kappas", nargs="+", type=int, default=[10, 100, 500])
    parser.add_argument(
        "--fresh", action="store_true", help="Ignore existing main CSV rows."
    )
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    run_selected_mode(args)
    print(f"Finished: {datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
