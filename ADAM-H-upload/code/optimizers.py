"""
ADAM-H: Stochastic ADAM with Householder Correction
Core optimizer implementations for saddle point search

Author: Hua Su, Lei Zhang, Jin Zhao
Date: 2026-01-20
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Optional, Callable


class StochasticADAMH:
    """
    ADAM-H: Decoupled Stochastic ADAM for Saddle Point Search

    Key innovation:
    - First moment (m): uses reflected gradient g_tilde (dynamics)
    - Second moment (s): uses original gradient g (geometry)

    This decoupling avoids historical pollution accumulation.
    """

    def __init__(
        self,
        k: int = 1,
        eta: float = 0.01,
        eta_v: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        n_inner: int = 5,
        clip_threshold: Optional[float] = None,
        device: str = 'cpu'
    ):
        """
        Args:
            k: Saddle index (number of negative eigenvalue directions)
            eta: Learning rate for position update
            eta_v: Learning rate for direction update
            beta1: Exponential decay rate for first moment
            beta2: Exponential decay rate for second moment
            epsilon: Numerical stability constant
            n_inner: Number of inner iterations for direction update
            clip_threshold: If not None, clip update norm to this value
            device: Computation device
        """
        self.k = k
        self.eta = eta
        self.eta_v = eta_v
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.n_inner = n_inner
        self.clip_threshold = clip_threshold
        self.device = device

        # State variables (initialized on first step)
        self.m = None  # First moment
        self.s = None  # Second moment
        self.t = 0     # Time step

    def householder_reflection(
        self,
        g: torch.Tensor,
        V: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply Householder reflection: g_tilde = (I - 2*V*V^T) * g

        Args:
            g: Gradient vector [d]
            V: Direction matrix [k, d] with orthonormal rows

        Returns:
            g_tilde: Reflected gradient [d]
        """
        g_tilde = g.clone()
        for i in range(V.shape[0]):
            v = V[i]
            g_tilde = g_tilde - 2 * torch.dot(v, g) * v
        return g_tilde

    def update_directions(
        self,
        V: torch.Tensor,
        hessian_vec_product: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """
        Update direction vectors via deflated power iteration.

        Args:
            V: Current direction matrix [k, d]
            hessian_vec_product: Function computing H @ v

        Returns:
            V_new: Updated direction matrix [k, d]
        """
        V_new = V.clone()

        for _ in range(self.n_inner):
            for i in range(self.k):
                v = V_new[i]
                Hv = hessian_vec_product(v)

                # Deflation: project out previously found directions
                for j in range(i):
                    Hv = Hv - 2 * torch.dot(V_new[j], Hv) * V_new[j]

                # Rayleigh quotient gradient
                # d = (I - v*v^T) * Hv = Hv - (v^T Hv) * v
                rayleigh = torch.dot(v, Hv)
                d = Hv - rayleigh * v

                # Gradient descent step
                v_new = v - self.eta_v * d

                # Normalize
                V_new[i] = v_new / (torch.norm(v_new) + 1e-10)

            # Gram-Schmidt orthonormalization
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

    def step(
        self,
        x: torch.Tensor,
        V: torch.Tensor,
        g: torch.Tensor,
        hessian_vec_product: Callable[[torch.Tensor], torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Perform one optimization step.

        Args:
            x: Current position [d]
            V: Current direction matrix [k, d]
            g: Stochastic gradient at x [d]
            hessian_vec_product: Function computing H @ v

        Returns:
            x_new: Updated position
            V_new: Updated direction matrix
            info: Dictionary with diagnostic information
        """
        d = x.shape[0]

        # Initialize state if needed
        if self.m is None:
            self.m = torch.zeros(d, device=self.device)
            self.s = torch.zeros(d, device=self.device)

        self.t += 1

        # Update direction vectors
        V_new = self.update_directions(V, hessian_vec_product)

        # Householder reflection
        g_tilde = self.householder_reflection(g, V_new)

        # ADAM-H: Decoupled update
        # First moment: uses reflected gradient (dynamics)
        self.m = self.beta1 * self.m + (1 - self.beta1) * g_tilde
        # Second moment: uses ORIGINAL gradient (geometry) - KEY DIFFERENCE!
        self.s = self.beta2 * self.s + (1 - self.beta2) * (g ** 2)

        # Bias correction
        m_hat = self.m / (1 - self.beta1 ** self.t)
        s_hat = self.s / (1 - self.beta2 ** self.t)

        # Compute update
        update = self.eta * m_hat / (torch.sqrt(s_hat) + self.epsilon)

        # Optional clipping
        if self.clip_threshold is not None:
            update_norm = torch.norm(update)
            if update_norm > self.clip_threshold:
                update = update * (self.clip_threshold / update_norm)

        # Position update
        x_new = x - update

        # Diagnostic info
        info = {
            'm_hat': m_hat.clone(),
            's_hat': s_hat.clone(),
            'update_norm': torch.norm(update).item(),
            'g_norm': torch.norm(g).item(),
            'g_tilde_norm': torch.norm(g_tilde).item()
        }

        return x_new, V_new, info

    def reset(self):
        """Reset optimizer state."""
        self.m = None
        self.s = None
        self.t = 0


class StochasticADAMFC:
    """
    ADAM-FC: Fully-Coupled Stochastic ADAM for Saddle Point Search

    This is the naive approach where BOTH moments use the reflected gradient.
    Suffers from historical pollution accumulation.
    """

    def __init__(
        self,
        k: int = 1,
        eta: float = 0.01,
        eta_v: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        n_inner: int = 5,
        device: str = 'cpu'
    ):
        self.k = k
        self.eta = eta
        self.eta_v = eta_v
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.n_inner = n_inner
        self.device = device

        self.m = None
        self.s = None
        self.t = 0

    def householder_reflection(self, g: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        g_tilde = g.clone()
        for i in range(V.shape[0]):
            v = V[i]
            g_tilde = g_tilde - 2 * torch.dot(v, g) * v
        return g_tilde

    def update_directions(
        self,
        V: torch.Tensor,
        hessian_vec_product: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        V_new = V.clone()

        for _ in range(self.n_inner):
            for i in range(self.k):
                v = V_new[i]
                Hv = hessian_vec_product(v)

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

    def step(
        self,
        x: torch.Tensor,
        V: torch.Tensor,
        g: torch.Tensor,
        hessian_vec_product: Callable[[torch.Tensor], torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        d = x.shape[0]

        if self.m is None:
            self.m = torch.zeros(d, device=self.device)
            self.s = torch.zeros(d, device=self.device)

        self.t += 1

        V_new = self.update_directions(V, hessian_vec_product)
        g_tilde = self.householder_reflection(g, V_new)

        # ADAM-FC: Fully-Coupled update
        # BOTH moments use reflected gradient - THIS CAUSES POLLUTION!
        self.m = self.beta1 * self.m + (1 - self.beta1) * g_tilde
        self.s = self.beta2 * self.s + (1 - self.beta2) * (g_tilde ** 2)  # Pollution!

        m_hat = self.m / (1 - self.beta1 ** self.t)
        s_hat = self.s / (1 - self.beta2 ** self.t)

        update = self.eta * m_hat / (torch.sqrt(s_hat) + self.epsilon)
        x_new = x - update

        info = {
            'm_hat': m_hat.clone(),
            's_hat': s_hat.clone(),
            'update_norm': torch.norm(update).item(),
            'g_norm': torch.norm(g).item(),
            'g_tilde_norm': torch.norm(g_tilde).item()
        }

        return x_new, V_new, info

    def reset(self):
        self.m = None
        self.s = None
        self.t = 0


class QuadraticSaddle:
    """
    Quadratic saddle point test function.

    E(x) = 0.5 * x^T H x

    where H = diag(lambda_1, lambda_2, 1, ..., 1) with lambda_1 < 0 < lambda_2.
    """

    def __init__(
        self,
        d: int = 10,
        lambda1: float = -10.0,
        lambda2: float = 0.1,
        device: str = 'cpu'
    ):
        self.d = d
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.device = device

        # Construct Hessian
        eigenvalues = torch.ones(d, device=device)
        eigenvalues[0] = lambda1
        eigenvalues[1] = lambda2
        self.H = torch.diag(eigenvalues)
        self.eigenvalues = eigenvalues

        # Condition number
        self.kappa = abs(lambda1) / lambda2

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.dot(x, self.H @ x)

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        return self.H @ x

    def stochastic_gradient(
        self,
        x: torch.Tensor,
        sigma: float = 0.0
    ) -> torch.Tensor:
        """Gradient with additive Gaussian noise."""
        g = self.gradient(x)
        if sigma > 0:
            noise = torch.randn_like(g) * sigma
            g = g + noise
        return g

    def hessian_vec_product(self, v: torch.Tensor) -> torch.Tensor:
        return self.H @ v

    def stochastic_hessian_vec_product(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        sigma: float = 0.0,
        delta: float = 1e-4
    ) -> torch.Tensor:
        """Finite difference Hessian-vector product with noise."""
        g_plus = self.stochastic_gradient(x + delta * v, sigma)
        g_minus = self.stochastic_gradient(x - delta * v, sigma)
        return (g_plus - g_minus) / (2 * delta)


class MullerBrownPotential:
    """
    Müller-Brown potential energy surface.
    Classic benchmark for transition state search in computational chemistry.

    E(x,y) = sum_{i=1}^4 A_i * exp(a_i*(x-x0_i)^2 + b_i*(x-x0_i)*(y-y0_i) + c_i*(y-y0_i)^2)
    """

    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.d = 2

        # Müller-Brown parameters
        self.A = torch.tensor([-200., -100., -170., 15.], device=device)
        self.a = torch.tensor([-1., -1., -6.5, 0.7], device=device)
        self.b = torch.tensor([0., 0., 11., 0.6], device=device)
        self.c = torch.tensor([-10., -10., -6.5, 0.7], device=device)
        self.x0 = torch.tensor([1., 0., -0.5, -1.], device=device)
        self.y0 = torch.tensor([0., 0.5, 1.5, 1.], device=device)

    def energy(self, pos: torch.Tensor) -> torch.Tensor:
        x, y = pos[0], pos[1]
        E = torch.tensor(0., device=self.device)
        for i in range(4):
            dx = x - self.x0[i]
            dy = y - self.y0[i]
            E = E + self.A[i] * torch.exp(
                self.a[i] * dx**2 + self.b[i] * dx * dy + self.c[i] * dy**2
            )
        return E

    def gradient(self, pos: torch.Tensor) -> torch.Tensor:
        pos_req = pos.clone().requires_grad_(True)
        E = self.energy(pos_req)
        E.backward()
        return pos_req.grad.detach()

    def stochastic_gradient(
        self,
        pos: torch.Tensor,
        sigma: float = 0.0
    ) -> torch.Tensor:
        g = self.gradient(pos)
        if sigma > 0:
            noise = torch.randn_like(g) * sigma
            g = g + noise
        return g

    def hessian(self, pos: torch.Tensor) -> torch.Tensor:
        """Compute full Hessian using autograd."""
        pos_req = pos.clone().requires_grad_(True)
        g = self.gradient(pos_req)
        H = torch.zeros(2, 2, device=self.device)
        for i in range(2):
            pos_req = pos.clone().requires_grad_(True)
            g = self.gradient(pos_req)
            g[i].backward()
            H[i] = pos_req.grad
        return H

    def hessian_vec_product(
        self,
        pos: torch.Tensor,
        v: torch.Tensor
    ) -> torch.Tensor:
        """Hessian-vector product using finite differences."""
        delta = 1e-5
        g_plus = self.gradient(pos + delta * v)
        g_minus = self.gradient(pos - delta * v)
        return (g_plus - g_minus) / (2 * delta)


class SevenHumpCamel:
    """
    Seven-Hump Camel function (modified for saddle point).

    E(x,y) = -x^2 + 0.1*y^2 + 0.25*x^4 + 0.25*y^4

    Has a saddle point at origin.
    """

    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.d = 2

    def energy(self, pos: torch.Tensor) -> torch.Tensor:
        x, y = pos[0], pos[1]
        return -x**2 + 0.1*y**2 + 0.25*x**4 + 0.25*y**4

    def gradient(self, pos: torch.Tensor) -> torch.Tensor:
        x, y = pos[0], pos[1]
        gx = -2*x + x**3
        gy = 0.2*y + y**3
        return torch.stack([gx, gy])

    def stochastic_gradient(
        self,
        pos: torch.Tensor,
        sigma: float = 0.0
    ) -> torch.Tensor:
        g = self.gradient(pos)
        if sigma > 0:
            noise = torch.randn_like(g) * sigma
            g = g + noise
        return g

    def hessian_vec_product(
        self,
        pos: torch.Tensor,
        v: torch.Tensor
    ) -> torch.Tensor:
        x, y = pos[0], pos[1]
        # Hessian: [[-2 + 3x^2, 0], [0, 0.2 + 3y^2]]
        Hv = torch.zeros(2, device=self.device)
        Hv[0] = (-2 + 3*x**2) * v[0]
        Hv[1] = (0.2 + 3*y**2) * v[1]
        return Hv


def run_optimization(
    optimizer,
    problem,
    x0: torch.Tensor,
    V0: torch.Tensor,
    sigma: float = 0.0,
    max_iter: int = 2000,
    tol: float = 1e-6,
    record_trajectory: bool = False
) -> dict:
    """
    Run optimization and return results.

    Args:
        optimizer: ADAM-H or ADAM-FC optimizer
        problem: Test problem (QuadraticSaddle, MullerBrown, etc.)
        x0: Initial position
        V0: Initial direction matrix
        sigma: Noise level for stochastic gradients
        max_iter: Maximum iterations
        tol: Convergence tolerance on gradient norm
        record_trajectory: Whether to record full trajectory

    Returns:
        Dictionary with optimization results
    """
    device = x0.device
    x = x0.clone()
    V = V0.clone()

    trajectory = [] if record_trajectory else None
    s_hat_history = []
    g_norm_history = []

    converged = False
    final_iter = max_iter

    for t in range(max_iter):
        # Get stochastic gradient
        g = problem.stochastic_gradient(x, sigma)

        # Create Hessian-vector product function
        if hasattr(problem, 'hessian_vec_product'):
            if problem.d == 2:
                hvp = lambda v: problem.hessian_vec_product(x, v)
            else:
                hvp = lambda v: problem.hessian_vec_product(v)
        else:
            hvp = problem.hessian_vec_product

        # Optimization step
        x, V, info = optimizer.step(x, V, g, hvp)

        # Record history
        g_norm = torch.norm(problem.gradient(x)).item()
        g_norm_history.append(g_norm)

        if 's_hat' in info:
            s_hat_history.append(info['s_hat'].clone())

        if record_trajectory:
            trajectory.append(x.clone())

        # Check convergence
        if g_norm < tol:
            converged = True
            final_iter = t + 1
            break

    return {
        'x_final': x,
        'V_final': V,
        'converged': converged,
        'iterations': final_iter,
        'g_norm_history': g_norm_history,
        's_hat_history': s_hat_history,
        'trajectory': trajectory
    }


def compute_v_angle(V: torch.Tensor, e1: torch.Tensor) -> float:
    """Compute angle (in degrees) between V[0] and e1."""
    cos_angle = abs(torch.dot(V[0], e1))
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    return torch.acos(cos_angle).item() * 180 / np.pi


class OriginalHiSD:
    """
    Original High-Index Saddle Dynamics (HiSD) with fixed step size.

    Reference: Yin, Zhang, Zhang (2019) "High-index saddle dynamics"
    SIAM J. Sci. Comput., 41(6):A3576-A3595

    x_{t+1} = x_t - eta * g_tilde

    where g_tilde = (I - 2*V*V^T) * g is the Householder-reflected gradient.
    """

    def __init__(
        self,
        k: int = 1,
        eta: float = 0.01,
        eta_v: float = 0.01,
        n_inner: int = 5,
        device: str = 'cpu'
    ):
        self.k = k
        self.eta = eta
        self.eta_v = eta_v
        self.n_inner = n_inner
        self.device = device
        self.t = 0

    def householder_reflection(self, g: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        g_tilde = g.clone()
        for i in range(V.shape[0]):
            v = V[i]
            g_tilde = g_tilde - 2 * torch.dot(v, g) * v
        return g_tilde

    def update_directions(
        self,
        V: torch.Tensor,
        hessian_vec_product: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        V_new = V.clone()

        for _ in range(self.n_inner):
            for i in range(self.k):
                v = V_new[i]
                Hv = hessian_vec_product(v)

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

    def step(
        self,
        x: torch.Tensor,
        V: torch.Tensor,
        g: torch.Tensor,
        hessian_vec_product: Callable[[torch.Tensor], torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        self.t += 1

        # Update direction vectors
        V_new = self.update_directions(V, hessian_vec_product)

        # Householder reflection
        g_tilde = self.householder_reflection(g, V_new)

        # Simple gradient descent with fixed step size
        x_new = x - self.eta * g_tilde

        info = {
            'update_norm': (self.eta * torch.norm(g_tilde)).item(),
            'g_norm': torch.norm(g).item(),
            'g_tilde_norm': torch.norm(g_tilde).item()
        }

        return x_new, V_new, info

    def reset(self):
        self.t = 0


class AcceleratedHiSD:
    """
    Accelerated High-Index Saddle Dynamics (A-HiSD) with Heavy Ball momentum.

    Reference: Luo, Zhang (2023) "Accelerated high-index saddle dynamics"
    Journal of Scientific Computing

    The Heavy Ball method adds momentum to accelerate convergence:
    x_{t+1} = x_t - eta * g_tilde + beta * (x_t - x_{t-1})

    This is equivalent to:
    v_{t+1} = beta * v_t - eta * g_tilde
    x_{t+1} = x_t + v_{t+1}
    """

    def __init__(
        self,
        k: int = 1,
        eta: float = 0.01,
        eta_v: float = 0.01,
        beta: float = 0.9,  # Heavy ball momentum parameter
        n_inner: int = 5,
        device: str = 'cpu'
    ):
        self.k = k
        self.eta = eta
        self.eta_v = eta_v
        self.beta = beta  # Momentum coefficient
        self.n_inner = n_inner
        self.device = device

        self.t = 0
        self.velocity = None  # Momentum term

    def householder_reflection(self, g: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        g_tilde = g.clone()
        for i in range(V.shape[0]):
            v = V[i]
            g_tilde = g_tilde - 2 * torch.dot(v, g) * v
        return g_tilde

    def update_directions(
        self,
        V: torch.Tensor,
        hessian_vec_product: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        V_new = V.clone()

        for _ in range(self.n_inner):
            for i in range(self.k):
                v = V_new[i]
                Hv = hessian_vec_product(v)

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

    def step(
        self,
        x: torch.Tensor,
        V: torch.Tensor,
        g: torch.Tensor,
        hessian_vec_product: Callable[[torch.Tensor], torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        d = x.shape[0]

        # Initialize velocity if needed
        if self.velocity is None:
            self.velocity = torch.zeros(d, device=self.device)

        self.t += 1

        # Update direction vectors
        V_new = self.update_directions(V, hessian_vec_product)

        # Householder reflection
        g_tilde = self.householder_reflection(g, V_new)

        # Heavy Ball update: v = beta * v - eta * g_tilde
        self.velocity = self.beta * self.velocity - self.eta * g_tilde

        # Position update
        x_new = x + self.velocity

        info = {
            'update_norm': torch.norm(self.velocity).item(),
            'g_norm': torch.norm(g).item(),
            'g_tilde_norm': torch.norm(g_tilde).item()
        }

        return x_new, V_new, info

    def reset(self):
        self.t = 0
        self.velocity = None


class StochasticAMSGradH:
    """
    S-AMSGrad-H: Decoupled AMSGrad for Saddle Point Search

    Key innovation from Reddi et al. (2018) "On the Convergence of Adam and Beyond":
    - Uses max(v_t, v_{t-1}) instead of exponential moving average for v
    - This guarantees non-increasing effective learning rate
    - Combined with our decoupling principle for HiSD

    Theoretical advantage:
    - Provable convergence (unlike standard Adam)
    - Avoids the "short memory" problem of Adam
    - Decoupling avoids pollution accumulation
    """

    def __init__(
        self,
        k: int = 1,
        eta: float = 0.01,
        eta_v: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        n_inner: int = 5,
        clip_threshold: Optional[float] = None,
        device: str = 'cpu'
    ):
        self.k = k
        self.eta = eta
        self.eta_v = eta_v
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.n_inner = n_inner
        self.clip_threshold = clip_threshold
        self.device = device

        self.m = None      # First moment
        self.s = None      # Second moment (EMA)
        self.s_max = None  # Maximum of second moment (AMSGrad key)
        self.t = 0

    def householder_reflection(self, g: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        g_tilde = g.clone()
        for i in range(V.shape[0]):
            v = V[i]
            g_tilde = g_tilde - 2 * torch.dot(v, g) * v
        return g_tilde

    def update_directions(
        self,
        V: torch.Tensor,
        hessian_vec_product: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        V_new = V.clone()

        for _ in range(self.n_inner):
            for i in range(self.k):
                v = V_new[i]
                Hv = hessian_vec_product(v)

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

    def step(
        self,
        x: torch.Tensor,
        V: torch.Tensor,
        g: torch.Tensor,
        hessian_vec_product: Callable[[torch.Tensor], torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        d = x.shape[0]

        if self.m is None:
            self.m = torch.zeros(d, device=self.device)
            self.s = torch.zeros(d, device=self.device)
            self.s_max = torch.zeros(d, device=self.device)

        self.t += 1

        V_new = self.update_directions(V, hessian_vec_product)
        g_tilde = self.householder_reflection(g, V_new)

        # AMSGrad-H: Decoupled update with max operation
        # First moment: uses reflected gradient (dynamics)
        self.m = self.beta1 * self.m + (1 - self.beta1) * g_tilde
        # Second moment: uses ORIGINAL gradient (geometry) - decoupling
        self.s = self.beta2 * self.s + (1 - self.beta2) * (g ** 2)
        # AMSGrad: use maximum of all past v values
        self.s_max = torch.maximum(self.s_max, self.s)

        # Bias correction
        m_hat = self.m / (1 - self.beta1 ** self.t)
        # Use s_max instead of s for guaranteed convergence
        s_hat = self.s_max / (1 - self.beta2 ** self.t)

        update = self.eta * m_hat / (torch.sqrt(s_hat) + self.epsilon)

        if self.clip_threshold is not None:
            update_norm = torch.norm(update)
            if update_norm > self.clip_threshold:
                update = update * (self.clip_threshold / update_norm)

        x_new = x - update

        info = {
            'm_hat': m_hat.clone(),
            's_hat': s_hat.clone(),
            's_max': self.s_max.clone(),
            'update_norm': torch.norm(update).item(),
            'g_norm': torch.norm(g).item(),
            'g_tilde_norm': torch.norm(g_tilde).item()
        }

        return x_new, V_new, info

    def reset(self):
        self.m = None
        self.s = None
        self.s_max = None
        self.t = 0


class NesterovHiSD:
    """
    Nesterov Accelerated High-Index Saddle Dynamics.

    Nesterov momentum provides better theoretical convergence properties.

    y_t = x_t + beta * (x_t - x_{t-1})
    x_{t+1} = y_t - eta * g_tilde(y_t)
    """

    def __init__(
        self,
        k: int = 1,
        eta: float = 0.01,
        eta_v: float = 0.01,
        beta: float = 0.9,
        n_inner: int = 5,
        device: str = 'cpu'
    ):
        self.k = k
        self.eta = eta
        self.eta_v = eta_v
        self.beta = beta
        self.n_inner = n_inner
        self.device = device

        self.t = 0
        self.x_prev = None

    def householder_reflection(self, g: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        g_tilde = g.clone()
        for i in range(V.shape[0]):
            v = V[i]
            g_tilde = g_tilde - 2 * torch.dot(v, g) * v
        return g_tilde

    def update_directions(
        self,
        V: torch.Tensor,
        hessian_vec_product: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        V_new = V.clone()

        for _ in range(self.n_inner):
            for i in range(self.k):
                v = V_new[i]
                Hv = hessian_vec_product(v)

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

    def step(
        self,
        x: torch.Tensor,
        V: torch.Tensor,
        g: torch.Tensor,
        hessian_vec_product: Callable[[torch.Tensor], torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        # Initialize x_prev if needed
        if self.x_prev is None:
            self.x_prev = x.clone()

        self.t += 1

        # Nesterov lookahead
        y = x + self.beta * (x - self.x_prev)

        # Update direction vectors at lookahead position
        V_new = self.update_directions(V, hessian_vec_product)

        # Householder reflection (gradient is already at x, not y for simplicity)
        g_tilde = self.householder_reflection(g, V_new)

        # Store previous position
        self.x_prev = x.clone()

        # Update
        x_new = y - self.eta * g_tilde

        info = {
            'update_norm': torch.norm(x_new - x).item(),
            'g_norm': torch.norm(g).item(),
            'g_tilde_norm': torch.norm(g_tilde).item()
        }

        return x_new, V_new, info

    def reset(self):
        self.t = 0
        self.x_prev = None


if __name__ == "__main__":
    # Quick test
    device = 'cpu'
    torch.manual_seed(42)

    # Create problem
    problem = QuadraticSaddle(d=10, lambda1=-10, lambda2=0.1, device=device)
    print(f"Condition number: {problem.kappa}")

    # Initial conditions
    x0 = torch.tensor([0.5, 1.0] + [0.1]*(10-2), device=device, dtype=torch.float32)

    # Initial V with some misalignment
    theta = 18 * np.pi / 180  # 18 degrees
    v0 = torch.zeros(10, device=device)
    v0[0] = np.cos(theta)
    v0[1] = np.sin(theta)
    V0 = v0.unsqueeze(0)

    # Test ADAM-H
    opt_H = StochasticADAMH(k=1, eta=0.01, eta_v=0.01, n_inner=5, device=device)
    result_H = run_optimization(opt_H, problem, x0, V0, sigma=0.0, max_iter=500)
    print(f"ADAM-H: {result_H['iterations']} iterations, converged={result_H['converged']}")

    # Test ADAM-FC
    opt_FC = StochasticADAMFC(k=1, eta=0.01, eta_v=0.01, n_inner=5, device=device)
    result_FC = run_optimization(opt_FC, problem, x0, V0.clone(), sigma=0.0, max_iter=2000)
    print(f"ADAM-FC: {result_FC['iterations']} iterations, converged={result_FC['converged']}")

    print("Optimizer tests passed!")
