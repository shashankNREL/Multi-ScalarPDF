"""Dirichlet Mixture Density Network.

Direct translation of DirichletMDN_proposal.md §3.1 + §5.2:

    moments -> Linear(n_in, 128) -> SiLU
            -> Linear(128, hidden) -> SiLU
            -> Linear(hidden, hidden) -> SiLU
            -> {head_pi: Linear(hidden, K) -> softmax,
                head_alpha: Linear(hidden, 3*K) -> softplus + alpha_min, clipped above}

The input moment vector has 4 or 5 components depending on whether the
training conditions on the cross-scalar covariance. The model itself is
input-dim-agnostic; the choice is set by ``n_in`` at construction time.
"""

from __future__ import annotations

import torch
from torch import nn


class DirichletMDN(nn.Module):
    def __init__(
        self,
        n_in: int,
        K: int = 8,
        hidden: int = 256,
        alpha_min: float = 1.0,
        alpha_clip: float = 1e3,
    ) -> None:
        super().__init__()
        if n_in not in (4, 5):
            raise ValueError(f"n_in must be 4 or 5, got {n_in}")
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")
        if hidden < 1:
            raise ValueError(f"hidden must be >= 1, got {hidden}")
        if alpha_min < 1.0:
            raise ValueError(
                "alpha_min must be >= 1.0 because grid cell masses are evaluated "
                "at cell centroids; concentrations below one have boundary "
                "singularities that this approximation cannot represent reliably"
            )
        if alpha_clip <= alpha_min:
            raise ValueError("alpha_clip must exceed alpha_min")

        self.n_in = n_in
        self.K = K
        self.alpha_min = float(alpha_min)
        self.alpha_clip = float(alpha_clip)

        self.trunk = nn.Sequential(
            nn.Linear(n_in, 128),
            nn.SiLU(),
            nn.Linear(128, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.head_pi = nn.Linear(hidden, K)
        self.head_alpha = nn.Linear(hidden, 3 * K)

    def forward(self, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map ``(B, n_in)`` moments to ``(pi (B, K), alpha (B, K, 3))``.

        ``pi`` is a softmax over K components; ``alpha`` is each component's
        3-vector of Dirichlet concentration parameters with
        ``alpha_min <= alpha_i <= alpha_clip``.
        """
        h = self.trunk(m)
        pi = torch.softmax(self.head_pi(h), dim=-1)
        raw_alpha = nn.functional.softplus(self.head_alpha(h)) + self.alpha_min
        alpha = raw_alpha.clamp(min=self.alpha_min, max=self.alpha_clip)
        alpha = alpha.view(-1, self.K, 3)
        return pi, alpha


__all__ = ["DirichletMDN"]
