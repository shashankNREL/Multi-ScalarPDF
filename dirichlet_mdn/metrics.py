"""Evaluation metrics for the Dirichlet MDN.

All operate on histogram pairs ``(p, q)`` of shape ``(N, N)`` or batched
``(B, N, N)``, on per-record moments, or on mixture parameters ``(pi, alpha)``.

JSD and L1 are direct counterparts of ``MLPDF.py:79`` and ``MLPDF.py:225``.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from .bin_grid import BinGrid
from .losses import dirichlet_logpdf, mixture_logpdf, stack_predicted_moments

LOG_EPS = 1e-12


# ---------------------------------------------------------------------------
# Reconstruct a histogram on the simplex grid from (pi, alpha)
# ---------------------------------------------------------------------------


def predict_histograms(
    pi: torch.Tensor,
    alpha: torch.Tensor,
    grid: BinGrid,
) -> torch.Tensor:
    """Render the Dirichlet mixture as a probability histogram on ``grid``.

    Mirrors the renormalization used by ``HistogramNLL``: density × cell area,
    renormalized over the in-simplex cells. Out-of-simplex cells are zero.

    Returns: ``(B, N, N)`` float32 tensor summing to 1 over the simplex.
    """
    centers_a = torch.from_numpy(grid.centers_a).to(pi)
    centers_b = torch.from_numpy(grid.centers_b).to(pi)
    cell_area = torch.from_numpy(grid.cell_area).to(pi)
    mask = torch.from_numpy(grid.simplex_mask).to(pi.device)

    grid_a = centers_a.unsqueeze(1).expand(-1, grid.num_bins)
    grid_b = centers_b.unsqueeze(0).expand(grid.num_bins, -1)
    z1 = grid_a.reshape(-1)
    z2 = grid_b.reshape(-1)
    z3 = (1.0 - z1 - z2).clamp(min=0.0)
    z = torch.stack([z1, z2, z3], dim=-1)                                   # (N*N, 3)

    flat_mask = mask.reshape(-1)
    z_in = z[flat_mask]                                                     # (M, 3)
    area_in = cell_area.reshape(-1)[flat_mask]                              # (M,)

    log_density = mixture_logpdf(z_in, pi, alpha)                           # (B, M)
    log_unnorm = log_density + torch.log(area_in + LOG_EPS)
    log_norm = torch.logsumexp(log_unnorm, dim=-1, keepdim=True)
    q_flat = torch.exp(log_unnorm - log_norm)                               # (B, M)

    B = pi.shape[0]
    out = torch.zeros(B, grid.num_bins * grid.num_bins, dtype=pi.dtype, device=pi.device)
    out[:, flat_mask] = q_flat
    return out.view(B, grid.num_bins, grid.num_bins)


# ---------------------------------------------------------------------------
# JSD and L1 between two histograms (per-record)
# ---------------------------------------------------------------------------


def joint_l1(p: np.ndarray, q: np.ndarray, simplex_mask: np.ndarray) -> np.ndarray:
    """Per-record L1 over in-simplex cells. Returns shape (B,)."""
    diff = np.abs(p - q)
    mask = simplex_mask.astype(p.dtype)
    return (diff * mask).reshape(diff.shape[0], -1).sum(axis=-1)


def joint_jsd(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Per-record Jensen-Shannon divergence on flattened histograms.

    Bits (log base 2). Matches MLPDF.py:79 modulo broadcasting.
    """
    p = p.reshape(p.shape[0], -1) + eps
    q = q.reshape(q.shape[0], -1) + eps
    m = 0.5 * (p + q)
    kl_pm = (p * (np.log2(p) - np.log2(m))).sum(axis=-1)
    kl_qm = (q * (np.log2(q) - np.log2(m))).sum(axis=-1)
    return 0.5 * (kl_pm + kl_qm)


def marginal_jsd(
    p: np.ndarray,
    q: np.ndarray,
    *,
    axis: int,
    eps: float = 1e-12,
) -> np.ndarray:
    """JSD on marginals P(Z_axis) after summing along the other axis.

    ``axis`` ∈ {0, 1} refers to the Z1/Z2 axis to *keep*. Output shape (B,).
    """
    sum_axis = 1 if axis == 0 else 0
    pm = p.sum(axis=sum_axis + 1)   # +1 because of leading batch dim
    qm = q.sum(axis=sum_axis + 1)
    pm = pm[:, :, None] if False else pm  # already (B, N)
    pm = pm + eps
    qm = qm + eps
    m = 0.5 * (pm + qm)
    kl_pm = (pm * (np.log2(pm) - np.log2(m))).sum(axis=-1)
    kl_qm = (qm * (np.log2(qm) - np.log2(m))).sum(axis=-1)
    return 0.5 * (kl_pm + kl_qm)


# ---------------------------------------------------------------------------
# Mixture diagnostics
# ---------------------------------------------------------------------------


def mixture_active_count(pi: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """Number of mixture components with weight above ``threshold``."""
    return (pi > threshold).sum(axis=-1)


def moment_recovery_error(
    pi: torch.Tensor,
    alpha: torch.Tensor,
    target_moments: torch.Tensor,
    input_moments: int,
) -> Dict[str, np.ndarray]:
    """Per-record absolute and relative error for each conditioning moment.

    Returns a dict with arrays of shape ``(B, input_moments)`` for ``abs_err``,
    ``rel_err`` (relative to ``max(|target|, 1e-6)``), and the raw ``pred``.
    """
    with torch.no_grad():
        pred = stack_predicted_moments(pi, alpha, input_moments).cpu().numpy()
    tgt = target_moments.cpu().numpy()
    abs_err = np.abs(pred - tgt)
    rel_err = abs_err / np.maximum(np.abs(tgt), 1e-6)
    return {"pred": pred, "abs_err": abs_err, "rel_err": rel_err}


# ---------------------------------------------------------------------------
# Per-config / per-timestep breakdown helpers
# ---------------------------------------------------------------------------


def per_config_breakdown(
    metrics_per_record: pd.DataFrame,
    *,
    metric_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Mean and std of each metric grouped by ``scalar_config``.

    Expects ``metrics_per_record`` to contain a ``scalar_config`` column.
    """
    if metric_cols is None:
        metric_cols = [c for c in metrics_per_record.columns
                       if c not in ("scalar_config", "run_id", "timestep",
                                    "box_width", "sample_id")]
    grouped = metrics_per_record.groupby("scalar_config")[metric_cols]
    out = grouped.agg(["mean", "std", "count"])
    out.columns = [f"{m}_{stat}" for m, stat in out.columns]
    return out.reset_index()


def per_timestep_breakdown(
    metrics_per_record: pd.DataFrame,
    *,
    metric_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Mean and std of each metric grouped by ``(scalar_config, timestep)``."""
    if metric_cols is None:
        metric_cols = [c for c in metrics_per_record.columns
                       if c not in ("scalar_config", "run_id", "timestep",
                                    "box_width", "sample_id")]
    grouped = metrics_per_record.groupby(["scalar_config", "timestep"])[metric_cols]
    out = grouped.agg(["mean", "std", "count"])
    out.columns = [f"{m}_{stat}" for m, stat in out.columns]
    return out.reset_index()


__all__ = [
    "predict_histograms",
    "joint_l1",
    "joint_jsd",
    "marginal_jsd",
    "mixture_active_count",
    "moment_recovery_error",
    "per_config_breakdown",
    "per_timestep_breakdown",
]
