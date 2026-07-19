"""
Exact-gradient ADAM-H baseline implementations.

This file provides compact, reproducible exact-gradient variants used for the
merged ADAM-H manuscript. It is intended as a clean starting point for rerunning
or extending the deterministic stress tests; parameter sweeps should be logged
before final journal submission.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple, Optional
import torch

Tensor = torch.Tensor


def gram_schmidt(V: Tensor, eps: float = 1e-12) -> Tensor:
    rows = []
    for i in range(V.shape[0]):
        v = V[i].clone()
        for q in rows:
            v = v - torch.dot(q, v) * q
        rows.append(v / (torch.norm(v) + eps))
    return torch.stack(rows, dim=0)


def householder(g: Tensor, V: Tensor) -> Tensor:
    out = g.clone()
    for v in V:
        out = out - 2.0 * torch.dot(v, g) * v
    return out


def grad_and_hvp(E: Callable[[Tensor], Tensor], x: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
    x_req = x.detach().clone().requires_grad_(True)
    val = E(x_req)
    (g,) = torch.autograd.grad(val, x_req, create_graph=True)
    gv = torch.dot(g, v)
    (hvp,) = torch.autograd.grad(gv, x_req, retain_graph=False)
    return g.detach(), hvp.detach()


def gradient(E: Callable[[Tensor], Tensor], x: Tensor) -> Tensor:
    x_req = x.detach().clone().requires_grad_(True)
    val = E(x_req)
    (g,) = torch.autograd.grad(val, x_req)
    return g.detach()


@dataclass
class ExactAdaptiveHiSD:
    k: int = 1
    eta: float = 0.01
    eta_v: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    n_inner: int = 5
    mode: str = "adamh"  # "adamh", "fc", "hisd", or "ahisd"
    clip_threshold: Optional[float] = None
    momentum: float = 0.8

    def __post_init__(self) -> None:
        self.t = 0
        self.m: Optional[Tensor] = None
        self.s: Optional[Tensor] = None
        self.velocity: Optional[Tensor] = None

    def update_directions(self, E: Callable[[Tensor], Tensor], x: Tensor, V: Tensor) -> Tensor:
        Vn = V.clone()
        for _ in range(self.n_inner):
            for i in range(self.k):
                _, Hv = grad_and_hvp(E, x, Vn[i])
                for j in range(i):
                    Hv = Hv - 2.0 * torch.dot(Vn[j], Hv) * Vn[j]
                rayleigh = torch.dot(Vn[i], Hv)
                d = Hv - rayleigh * Vn[i]
                Vn[i] = Vn[i] - self.eta_v * d
            Vn = gram_schmidt(Vn)
        return Vn

    def step(self, E: Callable[[Tensor], Tensor], x: Tensor, V: Tensor):
        if self.m is None:
            self.m = torch.zeros_like(x)
            self.s = torch.zeros_like(x)
        self.t += 1
        g = gradient(E, x)
        V_new = self.update_directions(E, x, V)
        g_tilde = householder(g, V_new)

        if self.mode == "hisd":
            update = self.eta * g_tilde
        elif self.mode == "ahisd":
            if self.velocity is None:
                self.velocity = torch.zeros_like(x)
            self.velocity = self.momentum * self.velocity - self.eta * g_tilde
            if self.clip_threshold is not None:
                n = torch.norm(self.velocity)
                if n > self.clip_threshold:
                    self.velocity = self.velocity * (self.clip_threshold / n)
            x_new = x + self.velocity
            info = {"grad_norm": float(torch.norm(g)), "reflected_grad_norm": float(torch.norm(g_tilde))}
            return x_new.detach(), V_new.detach(), info
        else:
            self.m = self.beta1 * self.m + (1.0 - self.beta1) * g_tilde
            if self.mode == "fc":
                second = g_tilde * g_tilde
            elif self.mode == "adamh":
                second = g * g
            else:
                raise ValueError(f"unknown mode {self.mode!r}")
            self.s = self.beta2 * self.s + (1.0 - self.beta2) * second
            mhat = self.m / (1.0 - self.beta1 ** self.t)
            shat = self.s / (1.0 - self.beta2 ** self.t)
            update = self.eta * mhat / (torch.sqrt(shat) + self.epsilon)
        if self.clip_threshold is not None:
            n = torch.norm(update)
            if n > self.clip_threshold:
                update = update * (self.clip_threshold / n)
        x_new = x - update
        info = {"grad_norm": float(torch.norm(g)), "reflected_grad_norm": float(torch.norm(g_tilde))}
        return x_new.detach(), V_new.detach(), info


def quadratic_saddle(kappa: float = 100.0, d: int = 10, lambda1: float = -10.0):
    lambda2 = abs(lambda1) / kappa
    diag = torch.ones(d, dtype=torch.float64) * lambda2
    diag[0] = lambda1
    def E(x: Tensor) -> Tensor:
        return 0.5 * torch.sum(diag.to(x.device, x.dtype) * x * x)
    return E


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    E = quadratic_saddle(kappa=100, d=10)
    x = torch.randn(10) * 0.5
    V = torch.randn(1, 10); V = gram_schmidt(V)
    opt = ExactAdaptiveHiSD(k=1, eta=0.01, eta_v=0.01, n_inner=5, mode="adamh")
    for t in range(10):
        x, V, info = opt.step(E, x, V)
    print("smoke test grad_norm", info["grad_norm"])
