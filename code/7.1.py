"""Numerical experiments for Section 7.1."""
from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np
import torch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ADAM_DIR = os.path.join(SCRIPT_DIR, "ADAM")
DEFAULT_RESULTS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "result", "7.1")

if ADAM_DIR not in sys.path:
    sys.path.insert(0, ADAM_DIR)

from experiment_utils import hessian_index, true_grad_norm, write_csv  # noqa: E402
from optimizers import (  # noqa: E402
    OriginalHiSD,
    QuadraticSaddle,
    SevenHumpCamel,
    StochasticADAMFC,
    StochasticADAMH,
    run_optimization,
)


ETA = 0.01
TOL = 1e-5
DIMENSION = 10
STRESS_MAX_ITER = 3000
CONTROL_MAX_ITER = 10000


@contextmanager
def temporary_default_dtype(dtype: torch.dtype) -> Iterator[None]:
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


# Exact-gradient stress tests

STRESS_METHODS = {
    "HiSD": OriginalHiSD,
    "ADAM-FC": StochasticADAMFC,
    "ADAM-H": StochasticADAMH,
}


class MultiIndexQuadratic:
    def __init__(
        self, d: int = 10, k: int = 1,
        lambda_neg: float = -10.0, lambda_pos: float = 0.1,
    ) -> None:
        self.d = d
        self.k = k
        diag = torch.tensor(
            [lambda_neg] * k + [lambda_pos] * (d - k), dtype=torch.float32
        )
        self.H = torch.diag(diag)

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.dot(x, self.H @ x)

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        return self.H @ x

    def stochastic_gradient(self, x: torch.Tensor, sigma: float = 0.0) -> torch.Tensor:
        g = self.gradient(x)
        if sigma > 0:
            g = g + sigma * torch.randn_like(g)
        return g

    def hessian_vec_product(self, v: torch.Tensor) -> torch.Tensor:
        return self.H @ v


def stress_gram_schmidt(V: torch.Tensor) -> torch.Tensor:
    rows = []
    for i in range(V.shape[0]):
        v = V[i].clone()
        for q in rows:
            v = v - torch.dot(q, v) * q
        rows.append(v / (torch.norm(v) + 1e-10))
    return torch.stack(rows)


def stress_misaligned_v(d: int, theta_deg: float = 18.0) -> torch.Tensor:
    theta = theta_deg * np.pi / 180.0
    v = torch.zeros(d, dtype=torch.float32)
    v[0] = np.cos(theta)
    v[1] = np.sin(theta)
    return v.unsqueeze(0)


def stress_multi_index_v(k: int, d: int) -> torch.Tensor:
    V = torch.zeros(k, d, dtype=torch.float32)
    for i in range(k):
        V[i, i] = 1.0
        if i + 1 < d:
            V[i, i + 1] = 0.35
    return stress_gram_schmidt(V)


def seven_hump_hessian_index(x: torch.Tensor) -> Dict[str, object]:
    evals = torch.tensor([-2.0 + 3.0 * x[0] ** 2, 0.2 + 3.0 * x[1] ** 2])
    evals, _ = torch.sort(evals)
    return {
        "hessian_index": int(torch.sum(evals < -1e-6).item()),
        "min_hessian_eval": float(evals[0]),
        "max_hessian_eval": float(evals[-1]),
    }


def validate_stress_result(
    problem, x: torch.Tensor, target_index: int
) -> Dict[str, object]:
    grad_norm = true_grad_norm(problem, x)
    if isinstance(problem, SevenHumpCamel):
        hinfo = seven_hump_hessian_index(x)
    else:
        hi = hessian_index(problem, x)
        hinfo = {
            "hessian_index": hi["index"],
            "min_hessian_eval": hi["min_eval"],
            "max_hessian_eval": hi["max_eval"],
        }
    return {
        "true_grad_norm": grad_norm,
        "grad_ok": grad_norm < TOL,
        "index_ok": hinfo["hessian_index"] == target_index,
        **hinfo,
    }


def run_stress_method(
    problem, x0: torch.Tensor, V0: torch.Tensor,
    method: str, k: int, n_inner: int,
) -> Dict[str, object]:
    optimizer = STRESS_METHODS[method](
        k=k, eta=ETA, eta_v=ETA, n_inner=n_inner, device="cpu"
    )
    result = run_optimization(
        optimizer, problem, x0.clone(), V0.clone(), sigma=0.0,
        max_iter=STRESS_MAX_ITER, tol=TOL,
    )
    validation = validate_stress_result(problem, result["x_final"], k)
    return {
        "method": method,
        "converged": bool(
            result["converged"] and validation["grad_ok"] and validation["index_ok"]
        ),
        "iterations": result["iterations"],
        **validation,
    }


def append_stress_rows(
    rows: List[Dict[str, object]], experiment: str, test_problem: str, kappa,
    problem, x0: torch.Tensor, V0: torch.Tensor, k: int = 1, n_inner: int = 5,
) -> None:
    for method in STRESS_METHODS:
        torch.manual_seed(42)
        np.random.seed(42)
        result = run_stress_method(problem, x0, V0, method, k, n_inner)
        rows.append({
            "experiment": experiment,
            "test_problem": test_problem,
            "kappa": kappa,
            "k": k,
            "n_inner": n_inner,
            "tol": TOL,
            "max_iter": STRESS_MAX_ITER,
            **result,
        })


def stress_quadratic(kappa: int) -> QuadraticSaddle:
    return QuadraticSaddle(
        d=DIMENSION, lambda1=-10.0, lambda2=10.0 / kappa, device="cpu"
    )


def iter_stress_cases():
    x_quad = torch.tensor([0.5, 1.0] + [0.1] * 8, dtype=torch.float32)
    V_quad = stress_misaligned_v(DIMENSION)

    # ``exact_five_algo`` is a historical output label kept for CSV parity.
    for kappa in [100, 500]:
        yield (
            "exact_five_algo", f"quadratic_kappa_{kappa}", kappa,
            stress_quadratic(kappa), x_quad, V_quad, 1, 5,
        )

    yield (
        "exact_five_algo", "seven_hump_quartic", "", SevenHumpCamel(device="cpu"),
        torch.tensor([0.5, 1.0], dtype=torch.float32), stress_misaligned_v(2), 1, 5,
    )

    problem_100 = QuadraticSaddle(
        d=DIMENSION, lambda1=-10.0, lambda2=0.1, device="cpu"
    )
    for n_inner in [1, 3, 5, 10, 20, 50, 100]:
        yield (
            "exact_ninner", "quadratic_kappa_100", 100,
            problem_100, x_quad, V_quad, 1, n_inner,
        )

    for kappa in [10, 50, 100, 200, 500]:
        yield (
            "exact_kappa", "quadratic", kappa,
            stress_quadratic(kappa), x_quad, V_quad, 1, 5,
        )

    for k in [1, 2, 3]:
        problem = MultiIndexQuadratic(
            d=DIMENSION, k=k, lambda_neg=-10.0, lambda_pos=0.1
        )
        x0 = torch.tensor(
            [0.5] * k + [1.0] + [0.1] * (DIMENSION - k - 1), dtype=torch.float32
        )
        yield (
            "exact_index", "multi_index_quadratic", 100,
            problem, x0, stress_multi_index_v(k, DIMENSION), k, 5,
        )


def run_stress(results_dir: str) -> None:
    with temporary_default_dtype(torch.float32):
        torch.manual_seed(42)
        np.random.seed(42)
        rows: List[Dict[str, object]] = []
        for case in iter_stress_cases():
            append_stress_rows(rows, *case)
        filename = "exp_exact_stress_tests.csv"
        write_csv(os.path.join(results_dir, filename), rows)
        print(f"Wrote {len(rows)} exact-gradient stress-test rows to {filename}")


# Direction-solver control and shared exact-control helpers

def control_gram_schmidt(V: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    rows = []
    for i in range(V.shape[0]):
        v = V[i].clone()
        for q in rows:
            v = v - torch.dot(q, v) * q
        rows.append(v / (torch.norm(v) + eps))
    return torch.stack(rows)


def householder(g: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    out = g.clone()
    for v in V:
        out = out - 2.0 * torch.dot(v, g) * v
    return out


def plane_rotation(d: int, theta_deg: float) -> torch.Tensor:
    theta = torch.tensor(theta_deg * torch.pi / 180.0)
    Q = torch.eye(d)
    c, s = torch.cos(theta), torch.sin(theta)
    Q[0, 0] = c
    Q[0, 1] = -s
    Q[1, 0] = s
    Q[1, 1] = c
    return Q


@dataclass
class QuadraticProblem:
    d: int
    kappa: float
    theta_deg: float
    lambda_neg: float = -10.0

    def __post_init__(self) -> None:
        lambda_pos = abs(self.lambda_neg) / self.kappa
        diag = torch.ones(self.d) * lambda_pos
        diag[0] = self.lambda_neg
        Q = plane_rotation(self.d, self.theta_deg)
        self.H = Q @ torch.diag(diag) @ Q.T
        self.true_V = Q[:, :1].T

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        return self.H @ x

    def hessian(self, x: torch.Tensor) -> torch.Tensor:
        return self.H

    def hessian_index(self, x: torch.Tensor) -> int:
        evals = torch.linalg.eigvalsh(self.H)
        return int(torch.sum(evals < -1e-8).item())


def update_iterative(
    V: torch.Tensor, H: torch.Tensor, eta_v: float, n_inner: int
) -> torch.Tensor:
    Vn = V.clone()
    for _ in range(n_inner):
        for i in range(Vn.shape[0]):
            v = Vn[i]
            Hv = H @ v
            for j in range(i):
                Hv = Hv - 2.0 * torch.dot(Vn[j], Hv) * Vn[j]
            rayleigh = torch.dot(v, Hv)
            direction_grad = Hv - rayleigh * v
            Vn[i] = v - eta_v * direction_grad
        Vn = control_gram_schmidt(Vn)
    return Vn


def update_eig(H: torch.Tensor, k: int) -> torch.Tensor:
    evals, evecs = torch.linalg.eigh(H)
    order = torch.argsort(evals)
    return evecs[:, order[:k]].T.contiguous()


class MomentRunner:
    def __init__(
        self, method: str, solver: str, k: int = 1,
        eta: float = 0.01, eta_v: float = 0.01, n_inner: int = 5,
        beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8,
    ) -> None:
        self.method = method
        self.solver = solver
        self.k = k
        self.eta = eta
        self.eta_v = eta_v
        self.n_inner = n_inner
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = None
        self.s = None

    def step(
        self, problem: QuadraticProblem, x: torch.Tensor, V: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        H = problem.hessian(x)
        if self.solver == "iterative":
            Vn = update_iterative(V, H, self.eta_v, self.n_inner)
        elif self.solver == "eig":
            Vn = update_eig(H, self.k)
        else:
            raise ValueError(f"unknown solver {self.solver!r}")

        g = problem.gradient(x)
        g_ref = householder(g, Vn)
        if self.method == "HiSD":
            return x - self.eta * g_ref, Vn

        if self.m is None:
            self.m = torch.zeros_like(x)
            self.s = torch.zeros_like(x)
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * g_ref
        if self.method == "ADAM-H":
            second = g * g
        elif self.method == "ADAM-FC":
            second = g_ref * g_ref
        else:
            raise ValueError(f"unknown method {self.method!r}")
        self.s = self.beta2 * self.s + (1.0 - self.beta2) * second
        mhat = self.m / (1.0 - self.beta1 ** self.t)
        shat = self.s / (1.0 - self.beta2 ** self.t)
        return x - self.eta * mhat / (torch.sqrt(shat) + self.epsilon), Vn


def initial_control_v(problem: QuadraticProblem) -> torch.Tensor:
    return control_gram_schmidt(
        plane_rotation(problem.d, problem.theta_deg + 18.0)[:, :1].T
    )


def run_control_iterations(problem, runner, x: torch.Tensor, V: torch.Tensor):
    converged = False
    iterations = CONTROL_MAX_ITER
    for t in range(CONTROL_MAX_ITER):
        x, V = runner.step(problem, x, V)
        if torch.norm(problem.gradient(x)) < TOL:
            converged = True
            iterations = t + 1
            break
    final_grad = float(torch.norm(problem.gradient(x)))
    return x, V, converged, iterations, final_grad


def subspace_angle_deg(V: torch.Tensor, W: torch.Tensor) -> float:
    cos_val = torch.abs(torch.dot(V[0], W[0])).clamp(0.0, 1.0)
    return float(torch.arccos(cos_val) * 180.0 / torch.pi)


def run_direction_case(
    problem: QuadraticProblem, method: str, solver: str, n_inner: int
) -> Dict[str, object]:
    x = torch.tensor([0.5, 1.0] + [0.1] * (problem.d - 2))
    V = initial_control_v(problem)
    initial_angle = subspace_angle_deg(V, problem.true_V)
    runner = MomentRunner(
        method=method, solver=solver, n_inner=n_inner, eta=ETA
    )
    x, V, converged, iterations, final_grad = run_control_iterations(
        problem, runner, x, V
    )
    return {
        "method": method,
        "direction_solver": solver,
        "n_inner": n_inner if solver == "iterative" else "",
        "converged": converged,
        "iterations": iterations,
        "true_grad_norm": final_grad,
        "grad_ok": final_grad < TOL,
        "index_ok": problem.hessian_index(x) == 1,
        "hessian_index": problem.hessian_index(x),
        "initial_v_angle_deg": initial_angle,
        "final_v_angle_deg": subspace_angle_deg(V, problem.true_V),
    }


def run_direction_control(results_dir: str) -> None:
    with temporary_default_dtype(torch.float64):
        rows: List[Dict[str, object]] = []
        methods = ["HiSD", "ADAM-FC", "ADAM-H"]
        cases = [
            ("axis_aligned", QuadraticProblem(DIMENSION, 100.0, 0.0)),
            ("rotated_30deg", QuadraticProblem(DIMENSION, 100.0, 30.0)),
            ("rotated_30deg", QuadraticProblem(DIMENSION, 500.0, 30.0)),
        ]
        for problem_name, problem in cases:
            for solver, n_inner in [("iterative", 5), ("eig", 0)]:
                for method in methods:
                    result = run_direction_case(problem, method, solver, n_inner)
                    rows.append({
                        "experiment": "direction_solver_control",
                        "problem": problem_name,
                        "kappa": int(problem.kappa),
                        "theta_deg": problem.theta_deg,
                        "eta": ETA,
                        "tol": TOL,
                        "max_iter": CONTROL_MAX_ITER,
                        **result,
                    })
        filename = "exp_direction_solver_control.csv"
        write_csv(os.path.join(results_dir, filename), rows)
        print(f"Wrote {len(rows)} direction-solver control rows to {filename}")


# Exact-eigensolver regime map

INIT_MODES = {
    "mixed_default": lambda d: [0.5, 1.0] + [0.1] * (d - 2),
    "stable_large": lambda d: [0.05, 5.0] + [0.1] * (d - 2),
    "unstable_large": lambda d: [5.0, 0.05] + [0.1] * (d - 2),
    "opposite": lambda d: [1.0, -1.0] + [0.1] * (d - 2),
    "flat_tail": lambda d: [0.5, 0.5] + [2.0] * (d - 2),
}

CURATED_CASES = [
    ("equal_axis_aligned", 0.0, 100.0, "mixed_default"),
    ("h_favorable_stable", 15.0, 10.0, "stable_large"),
    ("h_favorable_mixed", 5.0, 50.0, "mixed_default"),
    ("fc_favorable_tail", 15.0, 500.0, "flat_tail"),
    ("fc_favorable_opposite", 5.0, 10.0, "opposite"),
]


def run_eig_once(
    problem: QuadraticProblem, method: str, init_mode: str
) -> Tuple[bool, int, float]:
    x = torch.tensor(INIT_MODES[init_mode](problem.d), dtype=torch.float64)
    V = initial_control_v(problem)
    runner = MomentRunner(method=method, solver="eig", n_inner=0, eta=ETA)
    _, _, converged, iterations, final_grad = run_control_iterations(
        problem, runner, x, V
    )
    return converged, iterations, final_grad


def eig_case_rows(
    case_name: str, theta_deg: float, kappa: float, init_mode: str
) -> List[Dict[str, object]]:
    problem = QuadraticProblem(DIMENSION, kappa, theta_deg)
    raw = {}
    for method in ["ADAM-FC", "ADAM-H"]:
        converged, iterations, grad_norm = run_eig_once(problem, method, init_mode)
        raw[method] = {
            "converged": converged,
            "iterations": iterations,
            "true_grad_norm": grad_norm,
            "grad_ok": grad_norm < TOL,
            "index_ok": problem.hessian_index(torch.zeros(problem.d)) == 1,
        }

    fc_it = raw["ADAM-FC"]["iterations"]
    h_it = raw["ADAM-H"]["iterations"]
    if raw["ADAM-FC"]["converged"] and raw["ADAM-H"]["converged"]:
        ratio = fc_it / h_it
    elif raw["ADAM-H"]["converged"]:
        ratio = float("inf")
    elif raw["ADAM-FC"]["converged"]:
        ratio = 0.0
    else:
        ratio = ""

    rows = []
    for method, result in raw.items():
        rows.append({
            "experiment": "exact_eig_regime_map",
            "case": case_name,
            "theta_deg": theta_deg,
            "kappa": int(kappa),
            "init_mode": init_mode,
            "direction_solver": "eig",
            "eta": ETA,
            "tol": TOL,
            "max_iter": CONTROL_MAX_ITER,
            "method": method,
            "speedup_fc_over_h": ratio,
            **result,
        })
    return rows


def full_grid_cases() -> Iterable[Tuple[str, float, float, str]]:
    for theta in [5.0, 15.0, 30.0, 45.0, 60.0, 75.0]:
        for kappa in [10.0, 50.0, 100.0, 500.0, 1000.0]:
            for init_mode in INIT_MODES:
                yield (
                    f"grid_theta{int(theta)}_kappa{int(kappa)}_{init_mode}",
                    theta, kappa, init_mode,
                )


def run_eig_regime_map(results_dir: str, full_grid: bool = False) -> None:
    with temporary_default_dtype(torch.float64):
        rows: List[Dict[str, object]] = []
        cases = list(full_grid_cases()) if full_grid else CURATED_CASES
        for case in cases:
            rows.extend(eig_case_rows(*case))
        suffix = "_full" if full_grid else ""
        filename = f"exp_exact_eig_regime_map{suffix}.csv"
        write_csv(os.path.join(results_dir, filename), rows)
        print(f"Wrote {len(rows)} exact-eig regime-map rows to {filename}")


def run_selected_mode(mode: str, results_dir: str, full_grid: bool = False) -> None:
    if mode == "stress":
        run_stress(results_dir)
    elif mode == "direction-control":
        run_direction_control(results_dir)
    elif mode == "eig-regime-map":
        run_eig_regime_map(results_dir, full_grid=full_grid)
    elif mode == "all":
        run_stress(results_dir)
        run_direction_control(results_dir)
        run_eig_regime_map(results_dir, full_grid=full_grid)
    else:
        raise ValueError(f"unknown mode {mode!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Numerical Experiments for Section 7.1."
    )
    parser.add_argument(
        "--mode",
        choices=["stress", "direction-control", "eig-regime-map", "all"],
        default="all",
        help="Experiment group to run (default: all).",
    )
    parser.add_argument(
        "--full-grid", action="store_true",
        help="Use the larger grid for eig-regime-map (also applies within all).",
    )
    parser.add_argument(
        "--results-dir", default=DEFAULT_RESULTS_DIR,
        help="Output directory (default: ../result/7.1 relative to this script).",
    )
    args = parser.parse_args()
    run_selected_mode(args.mode, args.results_dir, full_grid=args.full_grid)


if __name__ == "__main__":
    main()
