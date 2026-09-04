"""Loss components for the Dirichlet Mixture Density Network.

Three terms (DirichletMDN_proposal.md §3.4):

  1. ``histogram_nll`` — categorical NLL on the simplex grid.
     For each batch element, evaluate the mixture density at every in-simplex
     clipped-cell centroid, multiply by the cell area, renormalize over the
     cells, and take ``L = -Σ_ij hist_ij * log(q_ij)``. Evaluated in log space
     throughout (logsumexp over components).

  2. ``moment_loss`` — closed-form mixture moment recovery against the
     conditioning moments. Predicts ``[E1, V1, E2, V2]`` (4-input) or
     ``[E1, V1, E2, V2, Cov12]`` (5-input) and takes MSE.

  3. ``entropy_reg`` — per-batch mean of ``-Σ_k π_k log π_k``. Combined as
     ``-λ_e * entropy`` (entropy is *maximized*) to discourage component
     collapse.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .bin_grid import BinGrid

LOG_EPS = 1e-12


# ---------------------------------------------------------------------------
# Dirichlet density utilities
# ---------------------------------------------------------------------------


def dirichlet_logpdf(z: torch.Tensor, alpha: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Log-density of the Dirichlet distribution.

    ``z`` and ``alpha`` must broadcast and have last dim = 3 (we operate on
    the 2-simplex; the third component is implicit). The output drops the
    trailing dim.
    """
    z_clamped = z.clamp(min=eps)
    z_norm = z_clamped / z_clamped.sum(dim=-1, keepdim=True)
    log_beta = torch.lgamma(alpha).sum(dim=-1) - torch.lgamma(alpha.sum(dim=-1))
    return ((alpha - 1.0) * torch.log(z_norm)).sum(dim=-1) - log_beta


def mixture_logpdf(
    z: torch.Tensor,
    pi: torch.Tensor,
    alpha: torch.Tensor,
    *,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Log-density of the Dirichlet mixture at points ``z``.

    Args:
        z:    (M, 3) points on the simplex, broadcast across the batch.
        pi:   (B, K) mixture weights summing to 1 along the K axis.
        alpha: (B, K, 3) per-component concentration parameters.

    Returns:
        log-density of shape ``(B, M)``.
    """
    # Broadcast: z -> (1, M, 1, 3), alpha -> (B, 1, K, 3) -> dirichlet_logpdf -> (B, M, K)
    z_e = z.view(1, z.shape[0], 1, 3)
    alpha_e = alpha.unsqueeze(1)
    log_p_k = dirichlet_logpdf(z_e, alpha_e, eps=eps)                   # (B, M, K)
    log_pi = torch.log(pi + LOG_EPS).unsqueeze(1)                       # (B, 1, K)
    return torch.logsumexp(log_pi + log_p_k, dim=-1)                    # (B, M)


# ---------------------------------------------------------------------------
# Histogram-categorical NLL on the simplex grid
# ---------------------------------------------------------------------------


class HistogramNLL(nn.Module):
    """Categorical NLL of the stored histogram under the Dirichlet mixture.

    The loss precomputes the in-simplex bin centers and cell areas as
    persistent buffers (so they move with ``.to(device)``).
    """

    def __init__(self, grid: BinGrid, eps: float = LOG_EPS,
                 dirichlet_eps: float = 1e-7) -> None:
        super().__init__()
        mask = torch.from_numpy(grid.simplex_mask)
        cell_area = torch.from_numpy(grid.cell_area)
        grid_a = torch.from_numpy(grid.cell_centroid_a)
        grid_b = torch.from_numpy(grid.cell_centroid_b)

        flat_mask = mask.reshape(-1)
        z1 = grid_a.reshape(-1)[flat_mask].to(torch.float64)
        z2 = grid_b.reshape(-1)[flat_mask].to(torch.float64)
        area = cell_area.reshape(-1)[flat_mask].to(torch.float64)
        z3 = 1.0 - z1 - z2

        z_simplex = torch.stack([z1, z2, z3], dim=-1).to(torch.float32)
        self.register_buffer("z_simplex", z_simplex)                    # (M, 3)
        self.register_buffer("cell_area_flat", area.to(torch.float32))  # (M,)
        self.register_buffer("simplex_mask", mask)                      # (N, N) bool
        self.register_buffer("flat_simplex_index",
                             flat_mask.nonzero(as_tuple=True)[0].to(torch.long))  # (M,)

        # ---- Static buffers used inside forward() ----
        # log(cell_area + eps): used to convert log-density to log-mass per cell.
        self.register_buffer("log_cell_area", torch.log(area.to(torch.float32))) # (M,)
        # log(z_norm) where z_norm = clamp(z, eps) / sum_clamp.
        # This term depends only on the grid (z is fixed across all batches).
        z_c = z_simplex.clamp(min=dirichlet_eps)
        z_n = z_c / z_c.sum(dim=-1, keepdim=True)
        # shape (1, M, 1, 3) — pre-broadcast for the mixture_logpdf inner product.
        self.register_buffer("log_z_norm", torch.log(z_n).view(1, -1, 1, 3))

        self.num_bins = grid.num_bins
        self.num_simplex_cells = int(flat_mask.sum().item())
        self.eps = eps
        self.dirichlet_eps = dirichlet_eps

    def per_record(
        self,
        pi: torch.Tensor,
        alpha: torch.Tensor,
        histogram: torch.Tensor,
    ) -> torch.Tensor:
        """Compute categorical NLL separately for every batch record.

        ``histogram`` may be either the full ``(B, N, N)`` 2-D grid (legacy)
        or the pre-flattened in-simplex ``(B, M)`` form (preferred — set by
        ``HybridPDFDataset(flat_histograms=True)``).
        """
        # ---- Mixture log-density evaluated at every in-simplex cell ----
        # alpha: (B, K, 3) -> (B, 1, K, 3); log_z_norm: (1, M, 1, 3)
        alpha_e = alpha.unsqueeze(1)
        log_p_k = (
            ((alpha_e - 1.0) * self.log_z_norm).sum(dim=-1)
            - torch.lgamma(alpha_e).sum(dim=-1)
            + torch.lgamma(alpha_e.sum(dim=-1))
        )                                                              # (B, M, K)
        log_pi = torch.log(pi + LOG_EPS).unsqueeze(1)                  # (B, 1, K)
        log_density = torch.logsumexp(log_pi + log_p_k, dim=-1)        # (B, M)

        log_q_unnorm = log_density + self.log_cell_area
        log_norm = torch.logsumexp(log_q_unnorm, dim=-1, keepdim=True)
        log_q = log_q_unnorm - log_norm                                # (B, M)

        if histogram.dim() == 2 and histogram.shape[-1] == self.num_simplex_cells:
            p_flat = histogram                                         # already (B, M)
        else:
            full_flat = histogram.reshape(histogram.shape[0], -1)
            if full_flat.shape[-1] != self.num_bins ** 2:
                raise ValueError(
                    f"target histogram has {full_flat.shape[-1]} cells; "
                    f"expected {self.num_bins ** 2}"
                )
            if not torch.isfinite(full_flat).all():
                raise ValueError("target histogram contains NaN or infinite values")
            if (full_flat < 0.0).any():
                raise ValueError("target histogram contains negative probability mass")
            p_flat = full_flat.index_select(-1, self.flat_simplex_index)
            total_mass = full_flat.sum(dim=-1, keepdim=True)
            inside_mass = p_flat.sum(dim=-1, keepdim=True)
            if ((total_mass - inside_mass) > 1e-7 * total_mass.clamp_min(1.0)).any():
                raise ValueError("target histogram contains mass outside the simplex")
        if not torch.isfinite(p_flat).all():
            raise ValueError("target histogram contains NaN or infinite values")
        if (p_flat < 0.0).any():
            raise ValueError("target histogram contains negative probability mass")
        target_mass = p_flat.sum(dim=-1, keepdim=True)
        if (target_mass <= self.eps).any():
            raise ValueError("target histogram has no mass inside the physical simplex")
        p_flat = p_flat / target_mass
        return -(p_flat * log_q).sum(dim=-1)                          # (B,)

    def forward(
        self,
        pi: torch.Tensor,
        alpha: torch.Tensor,
        histogram: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mean categorical NLL over the batch."""
        return self.per_record(pi, alpha, histogram).mean()


# ---------------------------------------------------------------------------
# Analytical moment recovery
# ---------------------------------------------------------------------------


def mixture_moments(pi: torch.Tensor, alpha: torch.Tensor) -> dict[str, torch.Tensor]:
    """Closed-form mixture moments (DirichletMDN_proposal.md §3.2).

    Returns a dict with per-batch tensors:
      ``E``    (B, 3): mixture mean E[Z_i]
      ``Var``  (B, 3): mixture variance Var[Z_i]
      ``Cov12``(B,):    mixture covariance Cov[Z_1, Z_2]
    """
    a0 = alpha.sum(dim=-1)                                              # (B, K)
    means_k = alpha / a0.unsqueeze(-1)                                  # (B, K, 3)
    var_k = means_k * (1.0 - means_k) / (a0.unsqueeze(-1) + 1.0)        # (B, K, 3)
    cov12_k = -alpha[..., 0] * alpha[..., 1] / (a0 ** 2 * (a0 + 1.0))   # (B, K)

    pi_b = pi.unsqueeze(-1)                                              # (B, K, 1)
    E = (pi_b * means_k).sum(dim=1)                                     # (B, 3)
    EZ2_per_comp = var_k + means_k ** 2                                 # (B, K, 3)
    EZ2 = (pi_b * EZ2_per_comp).sum(dim=1)                              # (B, 3)
    Var = EZ2 - E ** 2                                                  # (B, 3)
    EZ1Z2_per_comp = cov12_k + means_k[..., 0] * means_k[..., 1]        # (B, K)
    EZ1Z2 = (pi * EZ1Z2_per_comp).sum(dim=1)                            # (B,)
    Cov12 = EZ1Z2 - E[..., 0] * E[..., 1]                               # (B,)

    return {"E": E, "Var": Var, "Cov12": Cov12}


def stack_predicted_moments(pi: torch.Tensor, alpha: torch.Tensor, input_moments: int) -> torch.Tensor:
    """Stack the closed-form mixture moments into the conditioning order.

    4-input order: [E1, V1, E2, V2]
    5-input order: [E1, V1, E2, V2, Cov12]
    """
    m = mixture_moments(pi, alpha)
    E, V, C = m["E"], m["Var"], m["Cov12"]
    if input_moments == 4:
        return torch.stack([E[..., 0], V[..., 0], E[..., 1], V[..., 1]], dim=-1)
    elif input_moments == 5:
        return torch.stack([E[..., 0], V[..., 0], E[..., 1], V[..., 1], C], dim=-1)
    else:
        raise ValueError(f"input_moments must be 4 or 5, got {input_moments}")


def moment_loss(
    pi: torch.Tensor,
    alpha: torch.Tensor,
    m_input: torch.Tensor,
    input_moments: int,
    scales: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dimensionless MSE between predicted and target moments.

    Means naturally span ``[0, 1]``. Variances of a bounded scalar and the
    covariance span at most ``0.25`` in magnitude, so those physical ranges
    are used as default scales. This prevents raw mean errors from dominating
    the smaller-valued second moments.
    """
    pred = stack_predicted_moments(pi, alpha, input_moments)
    if scales is None:
        default = [1.0, 0.25, 1.0, 0.25]
        if input_moments == 5:
            default.append(0.25)
        scales = pred.new_tensor(default)
    return (((pred - m_input) / scales) ** 2).mean()


# ---------------------------------------------------------------------------
# Entropy regularizer
# ---------------------------------------------------------------------------


def entropy_reg(pi: torch.Tensor) -> torch.Tensor:
    """Mean per-batch entropy ``-Σ_k π_k log π_k``. Larger is better."""
    return -(pi * torch.log(pi + LOG_EPS)).sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


@dataclass
class LossWeights:
    lambda_mom: float = 0.5
    lambda_ent: float = 1e-3


@dataclass
class LossOutputs:
    total: torch.Tensor
    nll: torch.Tensor
    mom: torch.Tensor
    entropy: torch.Tensor


def combined_loss(
    pi: torch.Tensor,
    alpha: torch.Tensor,
    histogram: torch.Tensor,
    m_input: torch.Tensor,
    *,
    nll_module: HistogramNLL,
    input_moments: int,
    weights: LossWeights,
) -> LossOutputs:
    nll = nll_module(pi, alpha, histogram)
    mom = moment_loss(pi, alpha, m_input, input_moments)
    ent = entropy_reg(pi)
    total = nll + weights.lambda_mom * mom - weights.lambda_ent * ent
    return LossOutputs(total=total, nll=nll, mom=mom, entropy=ent)


__all__ = [
    "HistogramNLL",
    "dirichlet_logpdf",
    "mixture_logpdf",
    "mixture_moments",
    "stack_predicted_moments",
    "moment_loss",
    "entropy_reg",
    "combined_loss",
    "LossWeights",
    "LossOutputs",
]
