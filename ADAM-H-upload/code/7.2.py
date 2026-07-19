"""Numerical experiments for Section 7.2."""
from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager
from typing import Dict, Iterator, List

import numpy as np
import torch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ADAM_DIR = os.path.join(SCRIPT_DIR, "ADAM")
DEFAULT_RESULTS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "result", "7.2")

if ADAM_DIR not in sys.path:
    sys.path.insert(0, ADAM_DIR)

from experiment_utils import write_csv  # noqa: E402
from optimizers import QuadraticSaddle  # noqa: E402


DEVICE = "cpu"
DIMENSION = 10
KAPPA = 100.0
FIXED_THETA_DEG = 5.0
SIGMAS = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
N_SAMPLES = 10000


@contextmanager
def temporary_default_dtype(dtype: torch.dtype) -> Iterator[None]:
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def householder_reflection(g: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return g - 2 * torch.dot(v, g) * v


def fixed_state(
    kappa: float = KAPPA,
    theta_deg: float = FIXED_THETA_DEG,
    device: str = DEVICE,
):
    lambda1 = -10.0
    lambda2 = lambda1 / (-kappa)
    problem = QuadraticSaddle(
        d=DIMENSION, lambda1=lambda1, lambda2=lambda2, device=device
    )
    x = torch.tensor(
        [0.5, 1.0] + [0.1] * (DIMENSION - 2),
        device=device,
        dtype=torch.float32,
    )
    theta = theta_deg * np.pi / 180
    v = torch.zeros(DIMENSION, device=device)
    v[0] = np.cos(theta)
    v[1] = np.sin(theta)
    g_det = problem.gradient(x)
    return g_det, householder_reflection(g_det, v), v


def verify_pollution_conservation(
    kappa: float = KAPPA,
    theta_deg: float = FIXED_THETA_DEG,
    sigmas: List[float] = SIGMAS,
    n_samples: int = N_SAMPLES,
    device: str = DEVICE,
) -> dict:
    g_det, g_tilde_det, v = fixed_state(kappa, theta_deg, device)
    theoretical_pollution = (g_tilde_det ** 2 - g_det ** 2).cpu().numpy()
    results = {
        "sigmas": sigmas,
        "theoretical_pollution": theoretical_pollution,
        "measured_pollution": [],
        "E_g2": [],
        "E_gtilde2": [],
        "g_det": g_det.cpu().numpy(),
        "g_tilde_det": g_tilde_det.cpu().numpy(),
    }
    raw_rows = []
    j = 1

    print("Section 7.2 fixed-state diagnostic")
    for sigma in sigmas:
        E_g2_samples = []
        E_gtilde2_samples = []
        for _ in range(n_samples):
            noise = torch.randn(DIMENSION, device=device) * sigma
            g = g_det + noise
            g_tilde = householder_reflection(g, v)
            E_g2_samples.append((g ** 2).cpu().numpy())
            E_gtilde2_samples.append((g_tilde ** 2).cpu().numpy())

        E_g2 = np.mean(E_g2_samples, axis=0)
        E_gtilde2 = np.mean(E_gtilde2_samples, axis=0)
        measured_pollution = E_gtilde2 - E_g2
        results["E_g2"].append(E_g2)
        results["E_gtilde2"].append(E_gtilde2)
        results["measured_pollution"].append(measured_pollution)
        raw_rows.append({
            "experiment": "pollution_conservation",
            "kappa": kappa,
            "theta_deg": theta_deg,
            "n_samples": n_samples,
            "sigma": sigma,
            "coordinate": j + 1,
            "E_gtilde2": float(E_gtilde2[j]),
            "E_g2": float(E_g2[j]),
            "measured_delta": float(measured_pollution[j]),
            "theory_delta": float(theoretical_pollution[j]),
        })
        print(
            f"sigma={sigma:>4.1f}: measured={measured_pollution[j]:.4f}, "
            f"theory={theoretical_pollution[j]:.4f}"
        )

    results["raw_rows"] = raw_rows
    return results


def plot_pollution_conservation(results: dict, save_path: str) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    sigmas = results["sigmas"]
    theoretical = results["theoretical_pollution"]

    ax1 = axes[0]
    E_g2_coord2 = [r[1] for r in results["E_g2"]]
    E_gtilde2_coord2 = [r[1] for r in results["E_gtilde2"]]
    ax1.plot(sigmas, E_g2_coord2, "o-", label=r"$\mathbb{E}[g_2^2]$", markersize=8)
    ax1.plot(
        sigmas, E_gtilde2_coord2, "s-",
        label=r"$\mathbb{E}[\tilde{g}_2^2]$", markersize=8,
    )
    ax1.set_xlabel(r"Noise level $\sigma$", fontsize=12)
    ax1.set_ylabel("Second moment", fontsize=12)
    ax1.set_title("(a) Second moments vs noise", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    measured_coord2 = [r[1] for r in results["measured_pollution"]]
    ax2.plot(sigmas, measured_coord2, "o-", label="Measured", markersize=8)
    ax2.axhline(
        y=theoretical[1], color="r", linestyle="--", label="Theory", linewidth=2
    )
    ax2.set_xlabel(r"Noise level $\sigma$", fontsize=12)
    ax2.set_ylabel(
        r"Pollution $\mathbb{E}[\tilde{g}_2^2] - \mathbb{E}[g_2^2]$", fontsize=12
    )
    ax2.set_title("(b) Pollution conservation", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    for i, sigma in enumerate(sigmas):
        if sigma in [0.0, 1.0, 5.0, 10.0]:
            ax3.plot(
                range(10), results["measured_pollution"][i], "o-",
                label=f"σ={sigma}", alpha=0.7, markersize=5,
            )
    ax3.plot(range(10), theoretical, "k--", label="Theory", linewidth=2)
    ax3.set_xlabel("Coordinate index", fontsize=12)
    ax3.set_ylabel("Pollution", fontsize=12)
    ax3.set_title("(c) Pollution by coordinate", fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    # Auxiliary output; currently not referenced by the manuscript.
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def verify_noise_dilution(
    kappa: float = KAPPA,
    theta_deg: float = FIXED_THETA_DEG,
    sigmas: List[float] = SIGMAS,
    device: str = DEVICE,
) -> Dict[str, object]:
    g_det, g_tilde_det, _ = fixed_state(kappa, theta_deg, device)
    j = 1
    P = (g_tilde_det[j] ** 2 / g_det[j] ** 2).item()
    rows = []

    print(f"noise-dilution diagnostic: pollution factor={P:.4f}")
    for sigma in sigmas:
        g_det_sq = g_det[j].item() ** 2
        advantage = np.sqrt((P * g_det_sq + sigma**2) / (g_det_sq + sigma**2))
        rows.append({
            "experiment": "noise_dilution",
            "kappa": kappa,
            "theta_deg": theta_deg,
            "sigma": sigma,
            "sigma_sq": sigma ** 2,
            "theory_ratio": float(advantage),
            "pollution_factor": float(P),
            "coordinate": j + 1,
        })
        print(f"sigma={sigma:>4.1f}: surrogate ratio={advantage:.4f}")
    return {"raw_rows": rows, "pollution_factor": P}


def run_fixed_state(results_dir: str) -> None:
    with temporary_default_dtype(torch.float32):
        torch.manual_seed(42)
        np.random.seed(42)
        os.makedirs(results_dir, exist_ok=True)
        results = verify_pollution_conservation()
        write_csv(
            os.path.join(results_dir, "exp1_pollution_conservation.csv"),
            results["raw_rows"],
        )
        plot_pollution_conservation(
            results, os.path.join(results_dir, "fig1_pollution_conservation.pdf")
        )
        dilution = verify_noise_dilution()
        write_csv(
            os.path.join(results_dir, "exp1_noise_dilution.csv"),
            dilution["raw_rows"],
        )
        print("Wrote fixed-state CSV diagnostics and auxiliary PDF")


def run_second_moment(
    results_dir: str,
    theta_deg: float = 10.0,
    sigma: float = 1.0,
    max_iter: int = 2000,
    beta2: float = 0.999,
    device: str = DEVICE,
) -> Dict[str, object]:
    with temporary_default_dtype(torch.float32):
        d = 2
        g_det = torch.tensor([10.0, 0.1], device=device)
        theta = theta_deg * np.pi / 180
        v = torch.tensor(
            [np.cos(theta), np.sin(theta)], device=device, dtype=torch.float32
        )
        g_tilde_det = householder_reflection(g_det, v)
        true_g2_sq = g_det[1].item()**2 + sigma**2
        true_gt2_sq = g_tilde_det[1].item()**2 + sigma**2
        s_H = torch.zeros(d, device=device)
        s_FC = torch.zeros(d, device=device)

        torch.manual_seed(42)
        for _ in range(1, max_iter + 1):
            noise = sigma * torch.randn(d, device=device)
            g = g_det + noise
            g_tilde = householder_reflection(g, v)
            s_H = beta2 * s_H + (1 - beta2) * (g ** 2)
            s_FC = beta2 * s_FC + (1 - beta2) * (g_tilde ** 2)

        s_hat_H = s_H / (1 - beta2 ** max_iter)
        s_hat_FC = s_FC / (1 - beta2 ** max_iter)
        ratio = s_hat_FC[1].item() / s_hat_H[1].item()
        row = {
            "experiment": "variance_estimation",
            "theta_deg": theta_deg,
            "sigma": sigma,
            "max_iter": max_iter,
            "beta2": beta2,
            "s_hat_H": s_hat_H[1].item(),
            "s_hat_FC": s_hat_FC[1].item(),
            "true_g2_sq": true_g2_sq,
            "true_gt2_sq": true_gt2_sq,
            "ratio": ratio,
        }
        write_csv(os.path.join(results_dir, "exp5_variance_estimation.csv"), [row])
        print(
            f"second-moment recursion: H={row['s_hat_H']:.2f}, "
            f"FC={row['s_hat_FC']:.2f}, ratio={ratio:.1f}x"
        )
        return row


def run_selected_mode(mode: str, results_dir: str) -> None:
    if mode == "fixed-state":
        run_fixed_state(results_dir)
    elif mode == "second-moment":
        run_second_moment(results_dir)
    elif mode == "all":
        run_fixed_state(results_dir)
        run_second_moment(results_dir)
    else:
        raise ValueError(f"unknown mode {mode!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Numerical Experiments for Section 7.2."
    )
    parser.add_argument(
        "--mode",
        choices=["fixed-state", "second-moment", "all"],
        default="all",
        help="Experiment group to run (default: all).",
    )
    parser.add_argument(
        "--results-dir", default=DEFAULT_RESULTS_DIR,
        help="Output directory (default: ../result/7.2 relative to this script).",
    )
    args = parser.parse_args()
    run_selected_mode(args.mode, args.results_dir)


if __name__ == "__main__":
    main()
