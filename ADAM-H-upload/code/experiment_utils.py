"""
Shared utilities for reproducible ADAM-H experiments.

The helpers in this module keep convergence checks, Hessian-index validation,
and raw-result logging consistent across derivative-free experiments.
"""
from __future__ import annotations

import csv
import os
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch


def set_trial_seed(seed: int) -> None:
    """Seed NumPy and PyTorch for a single trial."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def true_gradient(func: Any, x: torch.Tensor) -> torch.Tensor:
    """Return the analytical or autograd gradient at x."""
    if hasattr(func, "gradient"):
        return func.gradient(x).detach()

    x_req = x.detach().clone().requires_grad_(True)
    value = func(x_req)
    (grad,) = torch.autograd.grad(value, x_req)
    return grad.detach()


def true_grad_norm(func: Any, x: torch.Tensor) -> float:
    return float(torch.norm(true_gradient(func, x)).item())


def hessian_matrix(func: Any, x: torch.Tensor) -> torch.Tensor:
    """Return a dense Hessian matrix for small validation problems."""
    if hasattr(func, "hessian"):
        H = func.hessian(x)
        return H.detach()
    if hasattr(func, "H"):
        return func.H.detach()

    x_req = x.detach().clone().requires_grad_(True)

    def wrapped(y: torch.Tensor) -> torch.Tensor:
        return func(y)

    H = torch.autograd.functional.hessian(wrapped, x_req)
    return H.detach()


def hessian_index(func: Any, x: torch.Tensor, index_tol: float = 1e-6) -> Dict[str, Any]:
    """Count negative Hessian eigenvalues at x."""
    H = hessian_matrix(func, x)
    H = 0.5 * (H + H.T)
    evals = torch.linalg.eigvalsh(H).detach().cpu().numpy()
    return {
        "index": int(np.sum(evals < -index_tol)),
        "min_eval": float(evals[0]),
        "max_eval": float(evals[-1]),
        "evals": evals,
    }


def final_validation(
    func: Any,
    x: torch.Tensor,
    target_index: int,
    grad_tol: float,
    index_tol: float = 1e-6,
) -> Dict[str, Any]:
    """Validate a final iterate by true gradient norm and Hessian index."""
    grad_norm = true_grad_norm(func, x)
    hinfo = hessian_index(func, x, index_tol=index_tol)
    return {
        "true_grad_norm": grad_norm,
        "hessian_index": hinfo["index"],
        "index_ok": hinfo["index"] == target_index,
        "grad_ok": grad_norm < grad_tol,
        "min_hessian_eval": hinfo["min_eval"],
        "max_hessian_eval": hinfo["max_eval"],
    }


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    """Write dictionaries to CSV using the union of observed keys."""
    ensure_dir(os.path.dirname(path))
    if not rows:
        return

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            clean = {}
            for key, value in row.items():
                if isinstance(value, np.generic):
                    value = value.item()
                clean[key] = value
            writer.writerow(clean)


def summarize_success(rows: Iterable[Dict[str, Any]], iter_key: str = "iterations") -> Dict[str, Any]:
    """Summarize converged rows with mean, median, std, and success rate."""
    rows = list(rows)
    successes = [
        r for r in rows
        if str(r.get("converged", False)).strip().lower() in {"true", "1", "yes"}
    ]
    iters = [int(float(r[iter_key])) for r in successes]
    if iters:
        mean = int(np.mean(iters))
        median = int(np.median(iters))
        std = int(np.std(iters))
    else:
        mean = median = std = 0
    return {
        "n": len(rows),
        "successes": len(successes),
        "rate": 100.0 * len(successes) / len(rows) if rows else 0.0,
        "mean": mean,
        "median": median,
        "std": std,
    }
