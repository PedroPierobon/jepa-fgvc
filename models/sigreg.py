"""
SIGReg: Sketched Isotropic Gaussian Regularization
=================================================
Differentiable self-supervised regularizer that prevents representation collapse
and imposes standard isotropic Gaussian geometry in latent space via random 1D
projections (Cramér-Wold device) and empirical characteristic function testing
(Epps-Pulley test) discretized via symmetric trapezoidal quadrature.

Theoretical Background:
- Cramér-Wold Device: A multivariate distribution in R^K is standard normal N(0, I_K)
  if and only if every 1D projection u = a^T z (where ||a||_2 = 1) follows N(0, 1).
- Characteristic Function of N(0, 1): phi_0(t) = exp(-t^2 / 2).
- Empirical Characteristic Function (ECF) of {u_1, ..., u_B}:
    hat_phi(t) = (1/B) sum_{j=1}^B exp(i * t * u_j)
               = (1/B) sum_{j=1}^B cos(t * u_j) + i * (1/B) sum_{j=1}^B sin(t * u_j).
- Epps-Pulley Distance:
    T = int |hat_phi(t) - phi_0(t)|^2 w(t) dt.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SIGReg(nn.Module):
    """
    Sketched Isotropic Gaussian Regularizer.

    Samples M random 1D unit projection vectors on S^{K-1} at each step,
    computes the empirical characteristic function (ECF) of projected latents,
    and evaluates the L2 divergence against standard normal N(0, 1) using
    symmetric trapezoidal quadrature over frequency grid [-t_max, t_max].

    Args:
        num_projections (int): Number M of random projection directions in S^{K-1}.
        num_nodes (int): Number P of symmetric quadrature nodes in frequency domain t.
        t_max (float): Upper bound of the integration interval [0, t_max].
    """

    def __init__(
        self,
        num_projections: int = 128,
        num_nodes: int = 17,
        t_max: float = 3.0,
    ) -> None:
        super().__init__()
        self.num_projections = int(num_projections)
        self.num_nodes = int(num_nodes)
        self.t_max = float(t_max)

        # Setup integration nodes and trapezoidal weights on [0, t_max]
        # Exploiting even symmetry, integral on [-t_max, t_max] = 2 * integral on [0, t_max]
        nodes, weights = self._create_quadrature_grid(self.num_nodes, self.t_max)
        self.register_buffer("nodes", nodes)  # [P]
        self.register_buffer("weights", weights)  # [P]
        self.register_buffer("phi_target", torch.exp(-0.5 * (nodes ** 2)))  # [P]

    @staticmethod
    def _create_quadrature_grid(
        num_nodes: int, t_max: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Creates uniform nodes and trapezoidal rule weights on [0, t_max].

        Args:
            num_nodes: Number of discretization nodes.
            t_max: Maximum integration frequency.

        Returns:
            Tuple (nodes [P], weights [P])
        """
        if num_nodes < 2:
            raise ValueError(f"num_nodes must be >= 2, got {num_nodes}")

        nodes = torch.linspace(0.0, t_max, num_nodes, dtype=torch.float32)
        dt = t_max / (num_nodes - 1)

        # Trapezoidal weights with factor 2 due to [-t_max, t_max] symmetry:
        # Internal nodes: 2 * dt
        # Boundary nodes (t=0 and t=t_max): 2 * (dt / 2) = dt
        weights = torch.full((num_nodes,), 2.0 * dt, dtype=torch.float32)
        weights[0] = dt
        weights[-1] = dt

        return nodes, weights

    def sample_projection_directions(
        self,
        dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Samples M random unit directions uniformly distributed on the unit sphere S^{K-1}.

        Args:
            dim: Latent dimension K.
            device: Target torch device.
            dtype: Target torch data type.

        Returns:
            Tensor [M, K] with unit L2-norm rows.
        """
        # Normalized standard Gaussian vectors are uniformly distributed on the sphere
        a = torch.randn(self.num_projections, dim, device=device, dtype=dtype)
        a = F.normalize(a, p=2.0, dim=-1, eps=1e-8)
        return a

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Computes the SIGReg loss for a batch of latent embeddings.

        Args:
            z: Tensor [B, K] of latent representations.

        Returns:
            Scalar tensor containing the SIGReg regularization loss.
        """
        if z.ndim != 2:
            raise ValueError(f"z must have shape [B, K], got shape {z.shape}")

        batch_size, dim = z.shape
        if batch_size < 2:
            return torch.tensor(0.0, device=z.device, dtype=z.dtype, requires_grad=True)

        # 1. Sample M unit directions on sphere S^{K-1} (resampled each step)
        directions = self.sample_projection_directions(dim, device=z.device, dtype=z.dtype)  # [M, K]

        # 2. 1D Projections: u = Z * A^T in R^{B x M}
        u = torch.matmul(z, directions.t())  # [B, M]

        # 3. Evaluate empirical characteristic function at quadrature nodes
        nodes = self.nodes.to(device=z.device, dtype=z.dtype).view(-1, 1, 1)  # [P, 1, 1]
        t_u = nodes * u.unsqueeze(0)  # [P, B, M]

        # Real and imaginary components of ECF:
        ecf_re = torch.cos(t_u).mean(dim=1)  # [P, M]
        ecf_im = torch.sin(t_u).mean(dim=1)  # [P, M]

        # Target characteristic function for standard normal: phi_0(t) = exp(-t^2 / 2)
        target_re = self.phi_target.to(device=z.device, dtype=z.dtype).unsqueeze(1)  # [P, 1]

        # 4. Epps-Pulley squared distance:
        diff_re = ecf_re - target_re  # [P, M]
        diff_sq = (diff_re ** 2) + (ecf_im ** 2)  # [P, M]

        # 5. Trapezoidal quadrature integration:
        weights = self.weights.to(device=z.device, dtype=z.dtype).unsqueeze(1)  # [P, 1]
        loss_per_direction = torch.sum(weights * diff_sq, dim=0)  # [M]

        # Mean over all M sketched directions
        return loss_per_direction.mean()

    def compute_epps_pulley_stat(
        self, z: torch.Tensor, num_directions: int = 512
    ) -> float:
        """
        Computes high-precision Epps-Pulley statistic for diagnostics.

        Args:
            z: Tensor [B, K] of latent representations.
            num_directions: Number of directions for accurate estimation.

        Returns:
            Scalar float statistic.
        """
        orig_proj = self.num_projections
        self.num_projections = num_directions
        try:
            with torch.no_grad():
                loss = self.forward(z)
                return float(loss.item())
        finally:
            self.num_projections = orig_proj


def sigreg_loss(
    z: torch.Tensor,
    num_projections: int = 128,
    num_nodes: int = 17,
    t_max: float = 3.0,
) -> torch.Tensor:
    """
    Functional wrapper to compute SIGReg loss.
    """
    reg = SIGReg(num_projections=num_projections, num_nodes=num_nodes, t_max=t_max)
    return reg(z)
