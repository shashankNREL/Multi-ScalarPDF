"""Plotting utilities for Dirichlet MDN diagnostics.

All plots follow the MLPDF.py convention of overlaying the upper-triangle
"forbidden" region as a white polygon and the simplex outline as a black
triangle.

Plots are kept matplotlib-only and headless-safe (no plt.show()).
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Polygon

from .bin_grid import BinGrid


# ---------------------------------------------------------------------------
# Simplex overlay (port of MLPDF.py:282-300)
# ---------------------------------------------------------------------------


def _add_simplex_overlay(ax) -> None:
    points_outside = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    points_inside = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
    tri_mask = Polygon(points_outside, fc="white", ec="white", closed=None)
    tri_outline = Polygon(points_inside, ec="black", fill=None)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.add_patch(tri_mask)
    ax.add_patch(tri_outline)


# ---------------------------------------------------------------------------
# Twin contour plot of DNS vs predicted PDF
# ---------------------------------------------------------------------------


def plot_pdf_comparison(
    grid: BinGrid,
    hist_true: np.ndarray,           # (N, N)
    hist_pred: np.ndarray,           # (N, N)
    title: str = "",
    subtitle: str = "",
    cmap: str = "RdBu_r",
) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(9, 4), layout="constrained",
    )
    fig.suptitle(title)

    density_true = np.divide(
        hist_true, grid.cell_area,
        out=np.full_like(hist_true, np.nan, dtype=np.float64),
        where=grid.simplex_mask,
    )
    density_pred = np.divide(
        hist_pred, grid.cell_area,
        out=np.full_like(hist_pred, np.nan, dtype=np.float64),
        where=grid.simplex_mask,
    )
    vmax = max(float(np.nanmax(density_true)), float(np.nanmax(density_pred)))
    vmax = max(vmax, np.finfo(float).eps)
    mesh1 = ax1.pcolormesh(
        grid.edges_a, grid.edges_b, density_true.T,
        shading="flat", cmap=cmap, vmin=0.0, vmax=vmax,
    )
    _add_simplex_overlay(ax1)
    ax1.set_title("DNS density")
    ax1.set_xlabel("Z1")
    ax1.set_ylabel("Z2")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    ax2.pcolormesh(
        grid.edges_a, grid.edges_b, density_pred.T,
        shading="flat", cmap=cmap, vmin=0.0, vmax=vmax,
    )
    _add_simplex_overlay(ax2)
    ax2.set_title("Dirichlet MDN density")
    ax2.set_xlabel("Z1")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    fig.colorbar(mesh1, ax=(ax1, ax2), label="probability density")
    if subtitle:
        fig.text(0.5, 0.02, subtitle, ha="center", fontsize=8)
    return fig


# ---------------------------------------------------------------------------
# Metric-vs-time line plot
# ---------------------------------------------------------------------------


def plot_metric_vs_timestep(
    df: pd.DataFrame,
    metric: str,
    *,
    group_col: str = "scalar_config",
    timestep_col: str = "timestep",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    for cfg, sub in df.groupby(group_col):
        agg = sub.groupby(timestep_col)[metric].mean().reset_index()
        ax.plot(agg[timestep_col], agg[metric], marker="o", label=cfg)
    ax.set_xlabel("timestep")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs timestep, by {group_col}")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Mixture diagnostics
# ---------------------------------------------------------------------------


def plot_alpha_diagnostics(
    pi: np.ndarray,
    alpha: np.ndarray,
    moments: np.ndarray,
    moment_names: Optional[List[str]] = None,
    *,
    active_threshold: float = 0.01,
) -> plt.Figure:
    """Two-panel diagnostic: active-component count histogram + alpha_0 scatter."""
    if moment_names is None:
        moment_names = [f"m{i}" for i in range(moments.shape[1])]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    n_active = (pi > active_threshold).sum(axis=-1)
    ax1.hist(n_active, bins=np.arange(0, pi.shape[1] + 2) - 0.5, rwidth=0.9)
    ax1.set_xlabel(f"# mixture components with pi > {active_threshold}")
    ax1.set_ylabel("# records")
    ax1.set_title("Mixture activity")

    a0_per_comp = alpha.sum(axis=-1)                            # (B, K)
    a0_effective = (pi * a0_per_comp).sum(axis=-1)              # (B,)
    var_idx = moment_names.index("var_a") if "var_a" in moment_names else 1
    ax2.scatter(moments[:, var_idx], a0_effective, s=4, alpha=0.5)
    ax2.set_xlabel(moment_names[var_idx])
    ax2.set_ylabel("E[alpha_0] = sum_k pi_k * sum_i alpha_{k,i}")
    ax2.set_title("Concentration vs variance")
    ax2.set_yscale("log")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Moment recovery scatter
# ---------------------------------------------------------------------------


def plot_moment_recovery(
    pred: np.ndarray,                # (B, M)
    target: np.ndarray,              # (B, M)
    moment_names: List[str],
) -> plt.Figure:
    n = pred.shape[1]
    ncols = n
    fig, axes = plt.subplots(1, ncols, figsize=(3.2 * ncols, 3.2), squeeze=False)
    for i, name in enumerate(moment_names):
        ax = axes[0, i]
        ax.scatter(target[:, i], pred[:, i], s=4, alpha=0.5)
        lo = float(min(target[:, i].min(), pred[:, i].min()))
        hi = float(max(target[:, i].max(), pred[:, i].max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel(f"target {name}")
        ax.set_ylabel(f"pred {name}")
        ax.set_title(name)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# PDF album helper
# ---------------------------------------------------------------------------


def write_pdf_album(figs: Iterable[plt.Figure], out_path: str) -> None:
    with PdfPages(out_path) as pdf:
        for fig in figs:
            pdf.savefig(fig)
            plt.close(fig)


__all__ = [
    "plot_pdf_comparison",
    "plot_metric_vs_timestep",
    "plot_alpha_diagnostics",
    "plot_moment_recovery",
    "write_pdf_album",
]
