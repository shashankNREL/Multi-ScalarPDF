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
from .losses import mixture_logpdf, stack_predicted_moments


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
    centroid_a = torch.from_numpy(grid.cell_centroid_a).to(pi)
    centroid_b = torch.from_numpy(grid.cell_centroid_b).to(pi)
    cell_area = torch.from_numpy(grid.cell_area).to(pi)
    mask = torch.from_numpy(grid.simplex_mask).to(pi.device)

    z1 = centroid_a.reshape(-1)
    z2 = centroid_b.reshape(-1)
    z3 = 1.0 - z1 - z2
    z = torch.stack([z1, z2, z3], dim=-1)                                   # (N*N, 3)

    flat_mask = mask.reshape(-1)
    z_in = z[flat_mask]                                                     # (M, 3)
    area_in = cell_area.reshape(-1)[flat_mask]                              # (M,)

    log_density = mixture_logpdf(z_in, pi, alpha)                           # (B, M)
    log_unnorm = log_density + torch.log(area_in)
    log_norm = torch.logsumexp(log_unnorm, dim=-1, keepdim=True)
    q_flat = torch.exp(log_unnorm - log_norm)                               # (B, M)

    B = pi.shape[0]
    out = torch.zeros(B, grid.num_bins * grid.num_bins, dtype=pi.dtype, device=pi.device)
    out[:, flat_mask] = q_flat
    return out.view(B, grid.num_bins, grid.num_bins)


# ---------------------------------------------------------------------------
# JSD and L1 between two histograms (per-record)
# ---------------------------------------------------------------------------


def _normalise_rows(values: np.ndarray) -> np.ndarray:
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("histograms must contain finite, non-negative mass")
    totals = values.sum(axis=-1, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("histograms must have positive total mass")
    return values / totals


def joint_l1(p: np.ndarray, q: np.ndarray, simplex_mask: np.ndarray) -> np.ndarray:
    """Per-record L1 over in-simplex cells. Returns shape (B,)."""
    mask = simplex_mask.reshape(1, -1)
    p_flat = _normalise_rows(p.reshape(p.shape[0], -1) * mask)
    q_flat = _normalise_rows(q.reshape(q.shape[0], -1) * mask)
    return np.abs(p_flat - q_flat).sum(axis=-1)


def joint_jsd(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Per-record Jensen-Shannon divergence on flattened histograms.

    Bits (log base 2). Matches MLPDF.py:79 modulo broadcasting.
    """
    p = _normalise_rows(p.reshape(p.shape[0], -1))
    q = _normalise_rows(q.reshape(q.shape[0], -1))
    m = 0.5 * (p + q)
    ratio_p = np.ones_like(p)
    ratio_q = np.ones_like(q)
    np.divide(p, m, out=ratio_p, where=p > 0.0)
    np.divide(q, m, out=ratio_q, where=q > 0.0)
    kl_pm = (p * np.log2(ratio_p)).sum(axis=-1)
    kl_qm = (q * np.log2(ratio_q)).sum(axis=-1)
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
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")
    p = _normalise_rows(p.reshape(p.shape[0], -1)).reshape(p.shape)
    q = _normalise_rows(q.reshape(q.shape[0], -1)).reshape(q.shape)
    sum_axis = 1 if axis == 0 else 0
    pm = p.sum(axis=sum_axis + 1)   # +1 because of leading batch dim
    qm = q.sum(axis=sum_axis + 1)
    pm = _normalise_rows(pm)
    qm = _normalise_rows(qm)
    m = 0.5 * (pm + qm)
    ratio_p = np.ones_like(pm)
    ratio_q = np.ones_like(qm)
    np.divide(pm, m, out=ratio_p, where=pm > 0.0)
    np.divide(qm, m, out=ratio_q, where=qm > 0.0)
    kl_pm = (pm * np.log2(ratio_p)).sum(axis=-1)
    kl_qm = (qm * np.log2(ratio_q)).sum(axis=-1)
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
