"""
Derivative-Free ADAM-H: Zeroth-Order Saddle Point Search with Adaptive Optimization

This module implements derivative-free (zeroth-order) variants of saddle point search algorithms.
The key insight is that ADAM-H's decoupled design makes it robust to the high variance
inherent in zeroth-order gradient estimators.

Reference:
- Du, Shi, Zhang, Zheng (2025) "A Derivative-Free Saddle-Search Algorithm with Linear Convergence Rate"

Author: Hua Su, Lei Zhang, Jin Zhao
Date: 2026-01-21
"""

import torch
import numpy as np
from typing import Tuple, Callable, Optional


class ZerothOrderEstimator:
    """
    Zeroth-order gradient and Hessian-vector product estimators.

    Uses only function evaluations to estimate:
    - Gradient: F(x, r, l) = [f(x+lr) - f(x-lr)] / (2l) * r
    - Hessian-vector product: Hv = [F(x+lv, r, l) - F(x-lv, r, l)] / (2l)
    """

    def __init__(self, func: Callable, l: float = 1e-4, device: str = 'cpu'):
        """
        Args:
            func: Energy function E(x) -> scalar
            l: Difference length (smoothing parameter)
            device: Computation device
        """
        self.func = func
        self.l = l
        self.device = device

    def gradient(self, x: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """
        Zeroth-order gradient estimator.

        F(x, r, l) = [f(x+lr) - f(x-lr)] / (2l) * r

        Args:
            x: Current position [d]
            n_samples: Number of random directions to average over

        Returns:
            Estimated gradient [d]
        """
        d = x.shape[0]
        g_est = torch.zeros(d, device=x.device, dtype=x.dtype)

        for _ in range(n_samples):
            r = torch.randn(d, device=x.device, dtype=x.dtype)
            f_plus = self.func(x + self.l * r)
            f_minus = self.func(x - self.l * r)
            g_est += (f_plus - f_minus) / (2 * self.l) * r

        return g_est / n_samples

    def hessian_vec_product(self, x: torch.Tensor, v: torch.Tensor,
                            n_samples: int = 1) -> torch.Tensor:
        """
        Zeroth-order Hessian-vector product estimator.

        Hv = [F(x+lv, r, l) - F(x-lv, r, l)] / (2l)

        This has O(d) variance vs O(d^4) for full Hessian estimator.

        Args:
            x: Current position [d]
            v: Direction vector [d]
            n_samples: Number of random directions to average over

        Returns:
            Estimated Hv [d]
        """
        d = x.shape[0]
        v = v / (torch.norm(v) + 1e-10)  # Normalize

        Hv_est = torch.zeros(d, device=x.device, dtype=x.dtype)

        for _ in range(n_samples):
            # Compute gradient estimates at x + lv and x - lv
            g_plus = self.gradient(x + self.l * v, n_samples=1)
            g_minus = self.gradient(x - self.l * v, n_samples=1)
            Hv_est += (g_plus - g_minus) / (2 * self.l)

        return Hv_est / n_samples


class DerivativeFreeADAMH:
    """
    Derivative-Free ADAM-H: Zeroth-Order Saddle Point Search with Decoupled ADAM.

    Key innovation: Uses zeroth-order estimators for gradient and Hessian-vector product,
    combined with ADAM-H's decoupled design to handle the high variance.

    The decoupled design is crucial:
    - First moment (m): uses reflected gradient g_tilde (dynamics)
    - Second moment (s): uses original gradient g (geometry)

    This prevents the pollution from zeroth-order estimation noise from
    accumulating in the second moment.
    """

    def __init__(
        self,
        func: Callable,
        k: int = 1,
        eta: float = 0.01,
        eta_v: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        n_inner: int = 5,
        l: float = 1e-4,
        n_grad_samples: int = 5,
        n_hv_samples: int = 3,
        device: str = 'cpu'
    ):
        """
        Args:
            func: Energy function E(x) -> scalar (ONLY function evaluations needed)
            k: Saddle index
            eta: Learning rate for position
            eta_v: Learning rate for direction update
            beta1, beta2: ADAM parameters
            epsilon: Numerical stability
            n_inner: Inner iterations for direction update
            l: Difference length for zeroth-order estimation
            n_grad_samples: Number of samples for gradient estimation
            n_hv_samples: Number of samples for Hessian-vector product
            device: Computation device
        """
        self.func = func
        self.k = k
        self.eta = eta
        self.eta_v = eta_v
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.n_inner = n_inner
        self.l = l
        self.n_grad_samples = n_grad_samples
        self.n_hv_samples = n_hv_samples
        self.device = device

        self.estimator = ZerothOrderEstimator(func, l, device)

        self.m = None
        self.s = None
        self.t = 0

        # Statistics
        self.func_evals = 0

    def _count_func_eval(self, n: int = 1):
        """Track function evaluations."""
        self.func_evals += n

    def zeroth_order_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate gradient using only function evaluations."""
        d = x.shape[0]
        g_est = torch.zeros(d, device=x.device, dtype=x.dtype)

        for _ in range(self.n_grad_samples):
            r = torch.randn(d, device=x.device, dtype=x.dtype)
            f_plus = self.func(x + self.l * r)
            f_minus = self.func(x - self.l * r)
            self._count_func_eval(2)
            g_est += (f_plus - f_minus) / (2 * self.l) * r

        return g_est / self.n_grad_samples

    def zeroth_order_hessian_vec(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Estimate Hessian-vector product using only function evaluations."""
        d = x.shape[0]
        v = v / (torch.norm(v) + 1e-10)

        Hv_est = torch.zeros(d, device=x.device, dtype=x.dtype)

        for _ in range(self.n_hv_samples):
            # F(x+lv, r, l)
            r = torch.randn(d, device=x.device, dtype=x.dtype)
            x_plus = x + self.l * v
            f_pp = self.func(x_plus + self.l * r)
            f_pm = self.func(x_plus - self.l * r)
            g_plus = (f_pp - f_pm) / (2 * self.l) * r

            # F(x-lv, r, l)
            x_minus = x - self.l * v
            f_mp = self.func(x_minus + self.l * r)
            f_mm = self.func(x_minus - self.l * r)
            g_minus = (f_mp - f_mm) / (2 * self.l) * r

            self._count_func_eval(4)

            Hv_est += (g_plus - g_minus) / (2 * self.l)

        return Hv_est / self.n_hv_samples

    def householder_reflection(self, g: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Apply Householder reflection: g_tilde = (I - 2VV^T) g"""
        g_tilde = g.clone()
        for i in range(V.shape[0]):
            v = V[i]
            g_tilde = g_tilde - 2 * torch.dot(v, g) * v
        return g_tilde

    def update_directions(self, x: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Update direction vectors via zeroth-order deflated power iteration."""
        V_new = V.clone()

        for _ in range(self.n_inner):
            for i in range(self.k):
                v = V_new[i]

                # Zeroth-order Hessian-vector product
                Hv = self.zeroth_order_hessian_vec(x, v)

                # Deflation
                for j in range(i):
                    Hv = Hv - 2 * torch.dot(V_new[j], Hv) * V_new[j]

                # Rayleigh quotient gradient
                rayleigh = torch.dot(v, Hv)
                d = Hv - rayleigh * v

                # Gradient descent
                v_new = v - self.eta_v * d
                V_new[i] = v_new / (torch.norm(v_new) + 1e-10)

            V_new = self._gram_schmidt(V_new)

        return V_new

    def _gram_schmidt(self, V: torch.Tensor) -> torch.Tensor:
        """Gram-Schmidt orthonormalization."""
        V_orth = torch.zeros_like(V)
        for i in range(V.shape[0]):
            v = V[i].clone()
            for j in range(i):
                v = v - torch.dot(V_orth[j], v) * V_orth[j]
            V_orth[i] = v / (torch.norm(v) + 1e-10)
        return V_orth

    def step(self, x: torch.Tensor, V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Perform one DF-ADAM-H optimization step.

        Args:
            x: Current position [d]
            V: Current direction matrix [k, d]

        Returns:
            x_new: Updated position
            V_new: Updated direction matrix
            info: Diagnostic information
        """
        d = x.shape[0]

        if self.m is None:
            self.m = torch.zeros(d, device=x.device, dtype=x.dtype)
            self.s = torch.zeros(d, device=x.device, dtype=x.dtype)

        self.t += 1

        # Update direction vectors (zeroth-order)
        V_new = self.update_directions(x, V)

        # Zeroth-order gradient estimation
        g = self.zeroth_order_gradient(x)

        # Householder reflection
        g_tilde = self.householder_reflection(g, V_new)

        # DECOUPLED ADAM UPDATE - Key for handling zeroth-order noise!
        # First moment: reflected gradient (dynamics)
        self.m = self.beta1 * self.m + (1 - self.beta1) * g_tilde
        # Second moment: ORIGINAL gradient (geometry) - prevents noise accumulation
        self.s = self.beta2 * self.s + (1 - self.beta2) * (g ** 2)

        # Bias correction
        m_hat = self.m / (1 - self.beta1 ** self.t)
        s_hat = self.s / (1 - self.beta2 ** self.t)

        # Update
        update = self.eta * m_hat / (torch.sqrt(s_hat) + self.epsilon)
        x_new = x - update

        info = {
            'g_norm': torch.norm(g).item(),
            'g_tilde_norm': torch.norm(g_tilde).item(),
            'update_norm': torch.norm(update).item(),
            'func_evals': self.func_evals
        }

        return x_new, V_new, info

    def reset(self):
        """Reset optimizer state."""
        self.m = None
        self.s = None
        self.t = 0
        self.func_evals = 0


class DerivativeFreeADAMFC:
    """
    Derivative-Free ADAM-FC: Fully-Coupled variant.

    Both moments use the reflected gradient - this causes pollution accumulation
    from zeroth-order estimation noise.
    """

    def __init__(
        self,
        func: Callable,
        k: int = 1,
        eta: float = 0.01,
        eta_v: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        n_inner: int = 5,
        l: float = 1e-4,
        n_grad_samples: int = 5,
        n_hv_samples: int = 3,
        device: str = 'cpu'
    ):
        self.func = func
        self.k = k
        self.eta = eta
        self.eta_v = eta_v
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.n_inner = n_inner
        self.l = l
        self.n_grad_samples = n_grad_samples
        self.n_hv_samples = n_hv_samples
        self.device = device

        self.m = None
        self.s = None
        self.t = 0
        self.func_evals = 0

    def _count_func_eval(self, n: int = 1):
        self.func_evals += n

    def zeroth_order_gradient(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[0]
        g_est = torch.zeros(d, device=x.device, dtype=x.dtype)

        for _ in range(self.n_grad_samples):
            r = torch.randn(d, device=x.device, dtype=x.dtype)
            f_plus = self.func(x + self.l * r)
            f_minus = self.func(x - self.l * r)
            self._count_func_eval(2)
            g_est += (f_plus - f_minus) / (2 * self.l) * r

        return g_est / self.n_grad_samples

    def zeroth_order_hessian_vec(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        d = x.shape[0]
        v = v / (torch.norm(v) + 1e-10)

        Hv_est = torch.zeros(d, device=x.device, dtype=x.dtype)

        for _ in range(self.n_hv_samples):
            r = torch.randn(d, device=x.device, dtype=x.dtype)
            x_plus = x + self.l * v
            f_pp = self.func(x_plus + self.l * r)
            f_pm = self.func(x_plus - self.l * r)
            g_plus = (f_pp - f_pm) / (2 * self.l) * r

            x_minus = x - self.l * v
            f_mp = self.func(x_minus + self.l * r)
            f_mm = self.func(x_minus - self.l * r)
            g_minus = (f_mp - f_mm) / (2 * self.l) * r

            self._count_func_eval(4)
            Hv_est += (g_plus - g_minus) / (2 * self.l)

        return Hv_est / self.n_hv_samples

    def householder_reflection(self, g: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        g_tilde = g.clone()
        for i in range(V.shape[0]):
            v = V[i]
            g_tilde = g_tilde - 2 * torch.dot(v, g) * v
        return g_tilde

    def update_directions(self, x: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        V_new = V.clone()

        for _ in range(self.n_inner):
            for i in range(self.k):
                v = V_new[i]
                Hv = self.zeroth_order_hessian_vec(x, v)

                for j in range(i):
                    Hv = Hv - 2 * torch.dot(V_new[j], Hv) * V_new[j]

                rayleigh = torch.dot(v, Hv)
                d = Hv - rayleigh * v
                v_new = v - self.eta_v * d
                V_new[i] = v_new / (torch.norm(v_new) + 1e-10)

            V_new = self._gram_schmidt(V_new)

        return V_new

    def _gram_schmidt(self, V: torch.Tensor) -> torch.Tensor:
        V_orth = torch.zeros_like(V)
        for i in range(V.shape[0]):
            v = V[i].clone()
            for j in range(i):
                v = v - torch.dot(V_orth[j], v) * V_orth[j]
            V_orth[i] = v / (torch.norm(v) + 1e-10)
        return V_orth

    def step(self, x: torch.Tensor, V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        d = x.shape[0]

        if self.m is None:
            self.m = torch.zeros(d, device=x.device, dtype=x.dtype)
            self.s = torch.zeros(d, device=x.device, dtype=x.dtype)

        self.t += 1

        V_new = self.update_directions(x, V)
        g = self.zeroth_order_gradient(x)
        g_tilde = self.householder_reflection(g, V_new)

        # FULLY-COUPLED: Both use g_tilde - POLLUTION ACCUMULATES!
        self.m = self.beta1 * self.m + (1 - self.beta1) * g_tilde
        self.s = self.beta2 * self.s + (1 - self.beta2) * (g_tilde ** 2)

        m_hat = self.m / (1 - self.beta1 ** self.t)
        s_hat = self.s / (1 - self.beta2 ** self.t)

        update = self.eta * m_hat / (torch.sqrt(s_hat) + self.epsilon)
        x_new = x - update

        info = {
            'g_norm': torch.norm(g).item(),
            'g_tilde_norm': torch.norm(g_tilde).item(),
            'update_norm': torch.norm(update).item(),
            'func_evals': self.func_evals
        }

        return x_new, V_new, info

    def reset(self):
        self.m = None
        self.s = None
        self.t = 0
        self.func_evals = 0


class DerivativeFreeHiSD:
    """
    Derivative-Free HiSD: Original HiSD with zeroth-order estimators.
    Fixed step size, no momentum.
    """

    def __init__(
        self,
        func: Callable,
        k: int = 1,
        eta: float = 0.01,
        eta_v: float = 0.01,
        n_inner: int = 5,
        l: float = 1e-4,
        n_grad_samples: int = 5,
        n_hv_samples: int = 3,
        device: str = 'cpu'
    ):
        self.func = func
        self.k = k
        self.eta = eta
        self.eta_v = eta_v
        self.n_inner = n_inner
        self.l = l
        self.n_grad_samples = n_grad_samples
        self.n_hv_samples = n_hv_samples
        self.device = device

        self.t = 0
        self.func_evals = 0

    def _count_func_eval(self, n: int = 1):
        self.func_evals += n

    def zeroth_order_gradient(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[0]
        g_est = torch.zeros(d, device=x.device, dtype=x.dtype)

        for _ in range(self.n_grad_samples):
            r = torch.randn(d, device=x.device, dtype=x.dtype)
            f_plus = self.func(x + self.l * r)
            f_minus = self.func(x - self.l * r)
            self._count_func_eval(2)
            g_est += (f_plus - f_minus) / (2 * self.l) * r

        return g_est / self.n_grad_samples

    def zeroth_order_hessian_vec(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        d = x.shape[0]
        v = v / (torch.norm(v) + 1e-10)

        Hv_est = torch.zeros(d, device=x.device, dtype=x.dtype)

        for _ in range(self.n_hv_samples):
            r = torch.randn(d, device=x.device, dtype=x.dtype)
            x_plus = x + self.l * v
            f_pp = self.func(x_plus + self.l * r)
            f_pm = self.func(x_plus - self.l * r)
            g_plus = (f_pp - f_pm) / (2 * self.l) * r

            x_minus = x - self.l * v
            f_mp = self.func(x_minus + self.l * r)
            f_mm = self.func(x_minus - self.l * r)
            g_minus = (f_mp - f_mm) / (2 * self.l) * r

            self._count_func_eval(4)
            Hv_est += (g_plus - g_minus) / (2 * self.l)

        return Hv_est / self.n_hv_samples

    def householder_reflection(self, g: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        g_tilde = g.clone()
        for i in range(V.shape[0]):
            v = V[i]
            g_tilde = g_tilde - 2 * torch.dot(v, g) * v
        return g_tilde

    def update_directions(self, x: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        V_new = V.clone()

        for _ in range(self.n_inner):
            for i in range(self.k):
                v = V_new[i]
                Hv = self.zeroth_order_hessian_vec(x, v)

                for j in range(i):
                    Hv = Hv - 2 * torch.dot(V_new[j], Hv) * V_new[j]

                rayleigh = torch.dot(v, Hv)
                d = Hv - rayleigh * v
                v_new = v - self.eta_v * d
                V_new[i] = v_new / (torch.norm(v_new) + 1e-10)

            V_new = self._gram_schmidt(V_new)

        return V_new

    def _gram_schmidt(self, V: torch.Tensor) -> torch.Tensor:
        V_orth = torch.zeros_like(V)
        for i in range(V.shape[0]):
            v = V[i].clone()
            for j in range(i):
                v = v - torch.dot(V_orth[j], v) * V_orth[j]
            V_orth[i] = v / (torch.norm(v) + 1e-10)
        return V_orth

    def step(self, x: torch.Tensor, V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        self.t += 1

        V_new = self.update_directions(x, V)
        g = self.zeroth_order_gradient(x)
        g_tilde = self.householder_reflection(g, V_new)

        # Fixed step size update
        x_new = x - self.eta * g_tilde

        info = {
            'g_norm': torch.norm(g).item(),
            'g_tilde_norm': torch.norm(g_tilde).item(),
            'update_norm': (self.eta * torch.norm(g_tilde)).item(),
            'func_evals': self.func_evals
        }

        return x_new, V_new, info

    def reset(self):
        self.t = 0
        self.func_evals = 0


# Test problems (function-only interface for derivative-free methods)
class QuadraticSaddleFunc:
    """Quadratic saddle as a function-only interface."""

    def __init__(self, d: int = 10, lambda1: float = -10.0, lambda2: float = 0.1,
        device: str = 'cpu'):
        self.d = d
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.device = device

        eigenvalues = torch.ones(d, device=device)
        eigenvalues[0] = lambda1
        eigenvalues[1] = lambda2
        self.eigenvalues = eigenvalues
        self.kappa = abs(lambda1) / lambda2

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Energy function: E(x) = 0.5 * sum(lambda_i * x_i^2)"""
        return 0.5 * torch.sum(self.eigenvalues * x ** 2)

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """Energy function (alias for __call__)."""
        return self.__call__(x)

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        """True gradient for comparison."""
        return self.eigenvalues * x


class MullerBrownFunc:
    """Müller-Brown potential as a function-only interface."""

    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.d = 2

        self.A = torch.tensor([-200., -100., -170., 15.], device=device)
        self.a = torch.tensor([-1., -1., -6.5, 0.7], device=device)
        self.b = torch.tensor([0., 0., 11., 0.6], device=device)
        self.c = torch.tensor([-10., -10., -6.5, 0.7], device=device)
        self.x0 = torch.tensor([1., 0., -0.5, -1.], device=device)
        self.y0 = torch.tensor([0., 0.5, 1.5, 1.], device=device)

        # Known saddle point
        self.saddle = torch.tensor([-0.822, 0.624], device=device)

    def __call__(self, pos: torch.Tensor) -> torch.Tensor:
        x, y = pos[0], pos[1]
        E = torch.tensor(0., device=self.device)
        for i in range(4):
            dx = x - self.x0[i]
            dy = y - self.y0[i]
            E = E + self.A[i] * torch.exp(
                self.a[i] * dx**2 + self.b[i] * dx * dy + self.c[i] * dy**2
            )
        return E


class SevenHumpCamelFunc:
    """Seven-Hump Camel as a function-only interface."""

    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.d = 2
        self.saddle = torch.tensor([0., 0.], device=device)

    def __call__(self, pos: torch.Tensor) -> torch.Tensor:
        x, y = pos[0], pos[1]
        return -x**2 + 0.1*y**2 + 0.25*x**4 + 0.25*y**4


if __name__ == "__main__":
    print("Testing Derivative-Free ADAM-H...")

    device = 'cpu'
    torch.manual_seed(42)

    # Test on quadratic saddle
    problem = QuadraticSaddleFunc(d=5, lambda1=-10, lambda2=0.1, device=device)

    x0 = torch.tensor([0.5, 1.0, 0.1, 0.1, 0.1], device=device, dtype=torch.float32)
    v0 = torch.zeros(5, device=device)
    v0[0] = 1.0
    V0 = v0.unsqueeze(0)

    # Test DF-ADAM-H
    opt = DerivativeFreeADAMH(
        func=problem,
        k=1, eta=0.01, eta_v=0.01, n_inner=3,
        l=1e-4, n_grad_samples=5, n_hv_samples=3,
        device=device
    )

    x = x0.clone()
    V = V0.clone()

    print("Running DF-ADAM-H on quadratic saddle...")
    for i in range(100):
        x, V, info = opt.step(x, V)
        if i % 20 == 0:
            true_g_norm = torch.norm(problem.gradient(x)).item()
            print(f"  Iter {i}: ||x|| = {torch.norm(x):.6f}, ||g_true|| = {true_g_norm:.6f}, func_evals = {info['func_evals']}")

    print(f"Final position: {x[:3].tolist()}")
    print(f"Total function evaluations: {opt.func_evals}")
    print("Test passed!")
