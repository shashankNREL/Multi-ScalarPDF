"""Dataset diversity diagnostics for the hybrid (parquet + HDF5) corpus.

Standalone CLI that audits a parquet metadata file *before* training:

  - per-(scalar_config, run_id, timestep, box_width) cell counts
  - simplex coverage of conditioning moments (mean_a, mean_b)
  - per-config moment distributions
  - retain_reason breakdown (the diversity policy that built this dataset)
  - shape diagnostic distributions (edge/corner/center mass, effective support)
  - variance regime per config (well-mixed vs highly segregated)
  - sample histograms drawn from a few records, for visual sanity check

Reports go under ``<out-dir>/`` as:

  summary.json            — machine-readable numbers
  cell_counts.csv         — (scalar_config, run_id) row matrix
  moment_stats.csv        — per-config moment percentiles
  plots/
    simplex_scatter.png       — (mean_a, mean_b) by config, simplex overlay
    simplex_coverage.png      — 2-D coverage heatmap, plus per-config maps
    moment_histograms.png     — per-config marginal histograms of all 5 moments
    variance_regime.png       — var/var_max scatter per config
    retain_reason.png         — diversity-policy breakdown
    shape_diagnostics.png     — distributions of shape_* columns
    timestep_counts.png       — records per timestep, per config
    sample_pdfs.png           — a grid of randomly drawn DNS histograms per config

Usage:

    PYTHONPATH=. python -m dirichlet_mdn.dataset_diversity \\
      --parquet hybrid_dataset_mpi/all_cases_w64_w128_metadata.parquet \\
      --hdf5-dir hybrid_dataset_mpi \\
      --out-dir diagnostics/hybrid_dataset_mpi
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .bin_grid import bin_grid


MOMENT_COLS = ["mean_a", "var_a", "mean_b", "var_b", "cov_ab"]
SHAPE_COLS = [
    "shape_entropy", "shape_max_prob", "shape_effective_support",
    "shape_corner_z1", "shape_corner_z2", "shape_corner_z3",
    "shape_edge_z1_zero", "shape_edge_z2_zero", "shape_edge_z3_zero",
    "shape_center_mass",
]


def _resolve(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} in parquet columns")


def load_meta(parquet_path: str) -> pd.DataFrame:
    df = pq.read_table(parquet_path).to_pandas()
    run_col = _resolve(df, ("run_id", "run_number"))
    width_col = _resolve(df, ("box_width", "filter_width"))
    return df.rename(columns={run_col: "run_id", width_col: "box_width"}).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Tabular reports
# ---------------------------------------------------------------------------

def cell_counts(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["scalar_config", "run_id"]).size()
        .unstack(fill_value=0).sort_index()
    )


def moment_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for cfg, sub in df.groupby("scalar_config"):
        row = {"scalar_config": cfg, "n_rows": len(sub)}
        for c in MOMENT_COLS:
            row[f"{c}_p10"] = float(sub[c].quantile(0.10))
            row[f"{c}_med"] = float(sub[c].median())
            row[f"{c}_p90"] = float(sub[c].quantile(0.90))
        out.append(row)
    return pd.DataFrame(out)


def simplex_coverage(
    df: pd.DataFrame, nb: int = 20
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (H, simplex_mask, summary) where H[i,j] is sample count in cell (i,j)."""
    xb = np.linspace(0.0, 1.0, nb + 1)
    yb = np.linspace(0.0, 1.0, nb + 1)
    H, _, _ = np.histogram2d(
        df["mean_a"].clip(0, 1), df["mean_b"].clip(0, 1), bins=[xb, yb]
    )
    mask = np.zeros_like(H, dtype=bool)
    for i in range(nb):
        for j in range(nb):
            if xb[i] + yb[j] < 1.0:
                mask[i, j] = True
    n_filled = int((H[mask] > 0).sum())
    n_cells = int(mask.sum())
    summary = {
        "grid_size": nb,
        "n_simplex_cells": n_cells,
        "n_simplex_cells_filled": n_filled,
        "fraction_simplex_covered": float(n_filled / max(n_cells, 1)),
        "median_samples_per_filled_cell": float(np.median(H[mask][H[mask] > 0])) if n_filled else 0.0,
        "max_samples_in_one_cell": float(H.max()),
    }
    return H, mask, summary


def per_config_simplex_coverage(df: pd.DataFrame, nb: int = 20) -> dict:
    xb = np.linspace(0.0, 1.0, nb + 1)
    yb = np.linspace(0.0, 1.0, nb + 1)
    mask = np.zeros((nb, nb), dtype=bool)
    for i in range(nb):
        for j in range(nb):
            if xb[i] + yb[j] < 1.0:
                mask[i, j] = True
    out = {}
    for cfg, sub in df.groupby("scalar_config"):
        H, _, _ = np.histogram2d(
            sub["mean_a"].clip(0, 1), sub["mean_b"].clip(0, 1), bins=[xb, yb]
        )
        out[cfg] = {
            "n_rows": int(len(sub)),
            "n_simplex_cells_filled": int((H[mask] > 0).sum()),
            "fraction_simplex_covered": float((H[mask] > 0).sum() / max(mask.sum(), 1)),
            "median_mean_a": float(sub["mean_a"].median()),
            "median_mean_b": float(sub["mean_b"].median()),
        }
    return out


def variance_regime(df: pd.DataFrame) -> dict:
    out = {}
    for cfg, sub in df.groupby("scalar_config"):
        max_var = sub["mean_a"] * (1.0 - sub["mean_a"])
        rel = (sub["var_a"] / max_var.replace(0, np.nan)).clip(0, 1)
        valid = rel.dropna()
        out[cfg] = {
            "frac_well_mixed_lt_0_1": (
                float((valid < 0.1).mean()) if len(valid) else None
            ),
            "frac_highly_segregated_gt_0_8": (
                float((valid > 0.8).mean()) if len(valid) else None
            ),
            "median_rel_var_a": float(valid.median()) if len(valid) else None,
            "n_undefined_at_bound_mean": int(rel.isna().sum()),
        }
    return out


def edge_concentration(df: pd.DataFrame) -> dict:
    edge_max = df[["shape_edge_z1_zero", "shape_edge_z2_zero", "shape_edge_z3_zero"]].max(axis=1)
    return {
        "frac_records_edge_mass_gt_0_25": float((edge_max > 0.25).mean()),
        "frac_records_edge_mass_gt_0_50": float((edge_max > 0.50).mean()),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _simplex_triangle(ax) -> None:
    ax.plot([0, 1, 0, 0], [0, 0, 1, 0], "k-", linewidth=1.0)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")


def plot_simplex_scatter(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    cfgs = sorted(df["scalar_config"].unique())
    cmap = plt.get_cmap("tab10")
    for i, cfg in enumerate(cfgs):
        sub = df[df["scalar_config"] == cfg]
        ax.scatter(
            sub["mean_a"], sub["mean_b"],
            s=4, alpha=0.30, color=cmap(i % 10), label=f"{cfg} (n={len(sub)})",
            edgecolors="none",
        )
    _simplex_triangle(ax)
    ax.set_xlabel(r"$\langle Z_1 \rangle$  (mean_a)")
    ax.set_ylabel(r"$\langle Z_2 \rangle$  (mean_b)")
    ax.set_title("Conditioning-moment coverage on the 2-D simplex, by scalar_config")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_simplex_coverage(df: pd.DataFrame, out_path: Path, nb: int = 20) -> None:
    cfgs = sorted(df["scalar_config"].unique())
    ncfg = len(cfgs)
    ncols = min(4, ncfg + 1)
    nrows = int(np.ceil((ncfg + 1) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).flatten()

    xb = np.linspace(0, 1, nb + 1)
    yb = np.linspace(0, 1, nb + 1)
    mask = np.zeros((nb, nb), dtype=bool)
    for i in range(nb):
        for j in range(nb):
            if xb[i] + yb[j] < 1.0:
                mask[i, j] = True

    # (0) global heatmap
    H, _, _ = np.histogram2d(df["mean_a"].clip(0, 1), df["mean_b"].clip(0, 1), bins=[xb, yb])
    H_show = np.where(mask, np.log10(H + 1), np.nan)
    im = axes[0].imshow(
        H_show.T, origin="lower", extent=[0, 1, 0, 1], aspect="equal",
        cmap="viridis",
    )
    axes[0].plot([0, 1, 0, 0], [0, 0, 1, 0], "w-", linewidth=1.0)
    axes[0].set_title(f"all configs  (cov={100*(H[mask]>0).sum()/mask.sum():.0f}%)")
    fig.colorbar(im, ax=axes[0], shrink=0.7, label="log10(N+1)")

    for k, cfg in enumerate(cfgs, start=1):
        sub = df[df["scalar_config"] == cfg]
        Hc, _, _ = np.histogram2d(sub["mean_a"].clip(0, 1), sub["mean_b"].clip(0, 1), bins=[xb, yb])
        H_show = np.where(mask, np.log10(Hc + 1), np.nan)
        im = axes[k].imshow(
            H_show.T, origin="lower", extent=[0, 1, 0, 1], aspect="equal",
            cmap="viridis",
        )
        axes[k].plot([0, 1, 0, 0], [0, 0, 1, 0], "w-", linewidth=1.0)
        axes[k].set_title(
            f"{cfg}  (cov={100*(Hc[mask]>0).sum()/mask.sum():.0f}%, n={len(sub)})"
        )

    for ax in axes[ncfg + 1:]:
        ax.axis("off")
    fig.suptitle("Per-config simplex coverage (log-scaled cell counts)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_moment_histograms(df: pd.DataFrame, out_path: Path) -> None:
    cfgs = sorted(df["scalar_config"].unique())
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for ax, col in zip(axes, MOMENT_COLS):
        for i, cfg in enumerate(cfgs):
            sub = df[df["scalar_config"] == cfg]
            ax.hist(
                sub[col], bins=60, alpha=0.45, density=True,
                color=cmap(i % 10), label=cfg, histtype="stepfilled",
            )
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("density")
    axes[0].legend(loc="upper right", fontsize=7, ncol=2)
    fig.suptitle("Per-config marginal distribution of conditioning moments")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_variance_regime(df: pd.DataFrame, out_path: Path) -> None:
    cfgs = sorted(df["scalar_config"].unique())
    ncols = min(4, len(cfgs))
    nrows = int(np.ceil(len(cfgs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for i, cfg in enumerate(cfgs):
        ax = axes[i]
        sub = df[df["scalar_config"] == cfg]
        m = sub["mean_a"].values
        v = sub["var_a"].values
        ax.scatter(m, v, s=2, alpha=0.20, color="C0", edgecolors="none")
        mx = np.linspace(0, 1, 200)
        ax.plot(mx, mx * (1 - mx), "k--", linewidth=1.0, label=r"$m(1-m)$")
        ax.set_xlim(0, 1); ax.set_ylim(0, 0.27)
        ax.set_xlabel("mean_a"); ax.set_ylabel("var_a")
        ax.set_title(f"{cfg}  (n={len(sub)})")
        if i == 0: ax.legend(fontsize=8)
    for ax in axes[len(cfgs):]: ax.axis("off")
    fig.suptitle("Variance vs mean (Z1): scatter with theoretical max envelope")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_retain_reason(df: pd.DataFrame, out_path: Path) -> None:
    counts = df["retain_reason"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index, counts.values, color="steelblue")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}\n({100*v/len(df):.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("# records retained")
    ax.set_title(
        f"Diversity-policy retain_reason breakdown  (total n={len(df):,})"
    )
    ax.set_ylim(0, counts.values.max() * 1.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_shape_diagnostics(df: pd.DataFrame, out_path: Path) -> None:
    cols = SHAPE_COLS
    ncols = 5
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.0 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for i, c in enumerate(cols):
        ax = axes[i]
        ax.hist(df[c], bins=60, color="C2", alpha=0.85)
        ax.set_title(c, fontsize=10)
        ax.set_yscale("log")
    for ax in axes[len(cols):]: ax.axis("off")
    fig.suptitle("Shape-diagnostic distributions over all records")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_timestep_counts(df: pd.DataFrame, out_path: Path) -> None:
    cfgs = sorted(df["scalar_config"].unique())
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, cfg in enumerate(cfgs):
        sub = df[df["scalar_config"] == cfg]
        g = sub.groupby("timestep").size()
        ax.plot(g.index, g.values, "o-", color=cmap(i % 10), label=cfg)
    ax.set_xlabel("timestep")
    ax.set_ylabel("# retained records")
    ax.set_title("Records per timestep, per scalar_config")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_sample_pdfs(
    df: pd.DataFrame, hdf5_dir: str, out_path: Path,
    grid, n_per_cfg: int = 4, seed: int = 0,
) -> None:
    cfgs = sorted(df["scalar_config"].unique())
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(
        len(cfgs), n_per_cfg, figsize=(3.0 * n_per_cfg, 3.0 * len(cfgs))
    )
    axes = np.atleast_2d(axes)
    hdf5_dir = Path(hdf5_dir)
    handles: dict[str, h5py.File] = {}
    try:
        for i, cfg in enumerate(cfgs):
            sub = df[df["scalar_config"] == cfg]
            picks = sub.sample(n=min(n_per_cfg, len(sub)), random_state=int(rng.integers(0, 1 << 31)))
            for j, (_, row) in enumerate(picks.iterrows()):
                shard = str(row["hdf5_file"])
                if shard not in handles:
                    handles[shard] = h5py.File(str(hdf5_dir / shard), "r")
                hist = np.asarray(handles[shard]["data/histograms"][int(row["local_index"])])
                density = np.divide(
                    hist,
                    grid.cell_area,
                    out=np.full_like(hist, np.nan, dtype=np.float64),
                    where=grid.simplex_mask,
                )
                show = np.where(grid.simplex_mask, density, np.nan).T
                ax = axes[i, j]
                ax.pcolormesh(
                    grid.edges_a, grid.edges_b, show,
                    shading="flat", cmap="magma",
                )
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_aspect("equal")
                ax.plot([0, 1, 0, 0], [0, 0, 1, 0], "w-", linewidth=0.8)
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(
                    f"{cfg} r{int(row['run_id'])} t{int(row['timestep'])} bw{int(row['box_width'])}",
                    fontsize=8,
                )
            for j in range(len(picks), n_per_cfg):
                axes[i, j].axis("off")
    finally:
        for h in handles.values():
            try: h.close()
            except Exception: pass
    fig.suptitle("Random DNS PDF density samples per scalar_config")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------

def run(
    parquet: str, hdf5_dir: str, out_dir: str,
    grid_size: int = 20, num_bins: int = 64, zst: float = 0.1,
    uniform_bins: bool = False, n_samples_per_cfg: int = 4, seed: int = 0,
) -> None:
    out = Path(out_dir)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    df = load_meta(parquet)
    print(f"[diag] loaded {len(df):,} rows from {parquet}")

    # tables
    cc = cell_counts(df)
    cc.to_csv(out / "cell_counts.csv")
    moment_percentiles(df).to_csv(out / "moment_stats.csv", index=False)

    # numeric summary
    H, mask, cov = simplex_coverage(df, nb=grid_size)
    per_cfg_cov = per_config_simplex_coverage(df, nb=grid_size)
    var_regime = variance_regime(df)
    edge = edge_concentration(df)
    if "moment_abs_err_max" in df:
        moment_discrepancy = {
            "maximum": float(df["moment_abs_err_max"].max()),
            "median": float(df["moment_abs_err_max"].median()),
            "p99": float(df["moment_abs_err_max"].quantile(0.99)),
            "per_config_maximum": (
                df.groupby("scalar_config")["moment_abs_err_max"].max().to_dict()
            ),
        }
    else:
        moment_discrepancy = {"available": False}

    summary = {
        "parquet": str(parquet),
        "n_rows": int(len(df)),
        "scalar_configs": sorted(df["scalar_config"].unique().tolist()),
        "n_runs": int(df["run_id"].nunique()),
        "n_timesteps": int(df["timestep"].nunique()),
        "box_widths": sorted(int(x) for x in df["box_width"].unique()),
        "per_config_n_rows": df["scalar_config"].value_counts().sort_index().to_dict(),
        "per_config_n_runs": df.groupby("scalar_config")["run_id"].nunique().to_dict(),
        "global_simplex_coverage": cov,
        "per_config_simplex_coverage": per_cfg_cov,
        "variance_regime_per_config": var_regime,
        "edge_concentration": edge,
        "direct_vs_histogram_moment_discrepancy": moment_discrepancy,
        "retain_reason_counts": df["retain_reason"].value_counts().to_dict(),
        "records_per_timestep": df.groupby("timestep").size().to_dict(),
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # plots
    g = bin_grid(num_bins=num_bins, zst=zst, uniform=uniform_bins)
    plot_simplex_scatter(df, out / "plots" / "simplex_scatter.png")
    plot_simplex_coverage(df, out / "plots" / "simplex_coverage.png", nb=grid_size)
    plot_moment_histograms(df, out / "plots" / "moment_histograms.png")
    plot_variance_regime(df, out / "plots" / "variance_regime.png")
    plot_retain_reason(df, out / "plots" / "retain_reason.png")
    plot_shape_diagnostics(df, out / "plots" / "shape_diagnostics.png")
    plot_timestep_counts(df, out / "plots" / "timestep_counts.png")
    plot_sample_pdfs(df, hdf5_dir, out / "plots" / "sample_pdfs.png",
                     grid=g, n_per_cfg=n_samples_per_cfg, seed=seed)
    print(f"[diag] wrote summary.json, cell_counts.csv, moment_stats.csv and 8 plots to {out}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dataset diversity diagnostics for hybrid PDF corpus")
    p.add_argument("--parquet", required=True)
    p.add_argument("--hdf5-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--grid-size", type=int, default=20, help="simplex coverage grid resolution")
    p.add_argument("--num-bins", type=int, default=64, help="histogram bin count of the stored PDFs")
    p.add_argument("--zst", type=float, default=0.1)
    p.add_argument("--uniform-bins", action="store_true")
    p.add_argument("--n-samples-per-cfg", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    return p


def cli(argv: Optional[Sequence[str]] = None) -> int:
    a = build_argparser().parse_args(list(argv) if argv is not None else None)
    run(
        parquet=a.parquet, hdf5_dir=a.hdf5_dir, out_dir=a.out_dir,
        grid_size=int(a.grid_size), num_bins=int(a.num_bins),
        zst=float(a.zst), uniform_bins=bool(a.uniform_bins),
        n_samples_per_cfg=int(a.n_samples_per_cfg), seed=int(a.seed),
    )
    return 0


if __name__ == "__main__":
    sys.exit(cli())
