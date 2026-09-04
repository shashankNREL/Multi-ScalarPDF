"""Evaluate a trained Dirichlet MDN on the test split.

Usage:
    python -m dirichlet_mdn.evaluate \\
        --run-dir dirichlet_mdn_runs/<timestamp>_<tag>

Reads the run directory's manifest, scaler, splits, and best checkpoint.
Writes under ``<run-dir>/eval/``:

  metrics_per_record.csv      one row per test sample, all metrics + meta
  metrics_per_config.csv      grouped by scalar_config
  metrics_per_timestep.csv    grouped by (scalar_config, timestep)
  metrics_per_config.md       Markdown rendering of metrics_per_config.csv
  plots/pdf_album.pdf         a curated grid of DNS-vs-MDN contour plots
  plots/l1_vs_time.pdf
  plots/jsd_vs_time.pdf
  plots/alpha_diagnostics.pdf
  plots/moment_recovery.pdf
  eval_summary.json           one-line summary metrics
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .bin_grid import BinGrid, bin_grid
from .data import HybridPDFDataset, collate_records, load_metadata
from .losses import HistogramNLL, mixture_moments, stack_predicted_moments
from .metrics import (
    joint_jsd, joint_l1, marginal_jsd, mixture_active_count,
    per_config_breakdown, per_timestep_breakdown, predict_histograms,
)
from .model import DirichletMDN
from .plotting import (
    plot_alpha_diagnostics, plot_metric_vs_timestep, plot_moment_recovery,
    plot_pdf_comparison, write_pdf_album,
)
from .splits import load_split
from .train import TorchScaler, resolve_device


@dataclass
class EvalContext:
    run_dir: Path
    config: dict
    grid: BinGrid
    device: torch.device
    model: DirichletMDN
    scaler: TorchScaler
    nll_module: HistogramNLL
    input_moments: int


def _load_context(run_dir: Path, device_override: Optional[str] = None) -> EvalContext:
    with open(run_dir / "manifest.json", "r") as f:
        manifest = json.load(f)
    cfg = manifest["config"]
    device = resolve_device(device_override or cfg.get("device", "cpu"))

    grid = bin_grid(
        num_bins=int(cfg.get("num_bins", 64)),
        zst=float(cfg.get("zst", 0.1)),
        uniform=bool(cfg.get("uniform_bins", False)),
    )
    model = DirichletMDN(
        n_in=int(cfg["input_moments"]),
        K=int(cfg["K"]),
        hidden=int(cfg["hidden"]),
        alpha_min=float(cfg["alpha_min"]),
        alpha_clip=float(cfg["alpha_clip"]),
    ).to(device)

    ckpt_path = run_dir / "model_best.pt"
    if not ckpt_path.exists():
        ckpt_path = run_dir / "model_last.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    sk_scaler = joblib.load(run_dir / "scaler.pkl")
    scaler = TorchScaler(sk_scaler).to(device)

    nll_module = HistogramNLL(grid).to(device)

    return EvalContext(
        run_dir=run_dir,
        config=cfg,
        grid=grid,
        device=device,
        model=model,
        scaler=scaler,
        nll_module=nll_module,
        input_moments=int(cfg["input_moments"]),
    )


@torch.no_grad()
def _gather_predictions(ctx: EvalContext, loader: DataLoader):
    rows = []
    all_pi: List[np.ndarray] = []
    all_alpha: List[np.ndarray] = []
    all_p: List[np.ndarray] = []
    all_q: List[np.ndarray] = []
    all_moments_pred: List[np.ndarray] = []
    all_moments_true: List[np.ndarray] = []
    meta_rows: List[dict] = []

    for batch in loader:
        m_raw = batch["moments"].to(ctx.device)
        h = batch["histogram"].to(ctx.device)
        m_scaled = ctx.scaler(m_raw)
        pi, alpha = ctx.model(m_scaled)
        q = predict_histograms(pi, alpha, ctx.grid)
        pred_moments = stack_predicted_moments(pi, alpha, ctx.input_moments)

        all_pi.append(pi.cpu().numpy())
        all_alpha.append(alpha.cpu().numpy())
        all_p.append(h.cpu().numpy())
        all_q.append(q.cpu().numpy())
        all_moments_pred.append(pred_moments.cpu().numpy())
        all_moments_true.append(m_raw.cpu().numpy())

        meta = batch["meta"]
        for i in range(m_raw.shape[0]):
            meta_rows.append({
                "scalar_config": meta["scalar_config"][i],
                "run_id": int(meta["run_id"][i].item()),
                "timestep": int(meta["timestep"][i].item()),
                "box_width": int(meta["box_width"][i].item()),
                "sample_id": int(meta["sample_id"][i].item()),
            })

    out = {
        "pi": np.concatenate(all_pi, axis=0),
        "alpha": np.concatenate(all_alpha, axis=0),
        "p": np.concatenate(all_p, axis=0),
        "q": np.concatenate(all_q, axis=0),
        "moments_pred": np.concatenate(all_moments_pred, axis=0),
        "moments_true": np.concatenate(all_moments_true, axis=0),
        "meta": pd.DataFrame(meta_rows),
    }
    return out


def _write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        formatted = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                formatted.append(f"{v:.4g}")
            else:
                formatted.append(str(v))
        lines.append("| " + " | ".join(formatted) + " |")
    path.write_text("\n".join(lines) + "\n")


def _moment_names(input_moments: int) -> List[str]:
    if input_moments == 4:
        return ["mean_a", "var_a", "mean_b", "var_b"]
    return ["mean_a", "var_a", "mean_b", "var_b", "cov_ab"]


def _build_metrics_df(preds: dict, grid: BinGrid, input_moments: int) -> pd.DataFrame:
    df = preds["meta"].copy()
    df["l1"] = joint_l1(preds["p"], preds["q"], grid.simplex_mask)
    df["jsd"] = joint_jsd(preds["p"], preds["q"])
    df["jsd_marg_z1"] = marginal_jsd(preds["p"], preds["q"], axis=0)
    df["jsd_marg_z2"] = marginal_jsd(preds["p"], preds["q"], axis=1)
    df["active_components"] = mixture_active_count(preds["pi"])
    df["alpha0_effective"] = (preds["pi"] * preds["alpha"].sum(axis=-1)).sum(axis=-1)

    abs_err = np.abs(preds["moments_pred"] - preds["moments_true"])
    for i, name in enumerate(_moment_names(input_moments)):
        df[f"mom_abs_err_{name}"] = abs_err[:, i]
    return df


def _select_album_examples(meta_df: pd.DataFrame, *, per_cfg: int = 3) -> List[int]:
    """Pick up to ``per_cfg`` rows per scalar_config, spanning timesteps."""
    rows: List[int] = []
    for cfg, sub in meta_df.groupby("scalar_config"):
        ts_sorted = sub.sort_values("timestep")
        n = min(per_cfg, len(ts_sorted))
        if n == 0:
            continue
        picks = np.linspace(0, len(ts_sorted) - 1, n).round().astype(int)
        rows.extend(ts_sorted.iloc[picks].index.tolist())
    return rows


def evaluate_run(run_dir: Path, *, device_override: Optional[str] = None,
                 album_per_cfg: int = 3) -> dict:
    ctx = _load_context(run_dir, device_override=device_override)
    cfg = ctx.config
    eval_dir = run_dir / "eval"
    plot_dir = eval_dir / "plots"
    eval_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    split = load_split(str(run_dir / "splits.json"))
    if len(split.test) == 0:
        print(f"[evaluate] WARNING: test split is empty. Run dir: {run_dir}", file=sys.stderr)
        return {"n_test": 0}

    ds = HybridPDFDataset(
        cfg["parquet"], cfg["hdf5_dir"],
        input_moments=ctx.input_moments,
        indices=split.test,
        num_bins=int(cfg.get("num_bins", 64)),
        zst=float(cfg.get("zst", 0.1)),
        uniform_bins=bool(cfg.get("uniform_bins", False)),
    )
    try:
        loader = DataLoader(
            ds, batch_size=int(cfg["batch_size"]), shuffle=False,
            num_workers=0, collate_fn=collate_records,
        )
        preds = _gather_predictions(ctx, loader)
    finally:
        ds.close()

    metrics_df = _build_metrics_df(preds, ctx.grid, ctx.input_moments)
    metrics_df.to_csv(eval_dir / "metrics_per_record.csv", index=False)

    cfg_break = per_config_breakdown(metrics_df)
    cfg_break.to_csv(eval_dir / "metrics_per_config.csv", index=False)
    _write_markdown_table(cfg_break, eval_dir / "metrics_per_config.md")

    ts_break = per_timestep_breakdown(metrics_df)
    ts_break.to_csv(eval_dir / "metrics_per_timestep.csv", index=False)

    figs = []
    for idx in _select_album_examples(metrics_df, per_cfg=album_per_cfg):
        meta_row = metrics_df.loc[idx]
        title = (f"{meta_row['scalar_config']}  run {meta_row['run_id']}  "
                 f"t={meta_row['timestep']}  w={meta_row['box_width']}")
        subtitle = (f"L1={meta_row['l1']:.4f}  JSD={meta_row['jsd']:.4f}  "
                    f"K_active={int(meta_row['active_components'])}")
        figs.append(plot_pdf_comparison(
            ctx.grid, preds["p"][idx], preds["q"][idx],
            title=title, subtitle=subtitle,
        ))
    write_pdf_album(figs, str(plot_dir / "pdf_album.pdf"))

    fig = plot_metric_vs_timestep(metrics_df, "l1")
    fig.savefig(plot_dir / "l1_vs_time.pdf")
    import matplotlib.pyplot as plt
    plt.close(fig)

    fig = plot_metric_vs_timestep(metrics_df, "jsd")
    fig.savefig(plot_dir / "jsd_vs_time.pdf")
    plt.close(fig)

    fig = plot_alpha_diagnostics(
        preds["pi"], preds["alpha"],
        preds["moments_true"], moment_names=_moment_names(ctx.input_moments),
    )
    fig.savefig(plot_dir / "alpha_diagnostics.pdf")
    plt.close(fig)

    fig = plot_moment_recovery(
        preds["moments_pred"], preds["moments_true"],
        moment_names=_moment_names(ctx.input_moments),
    )
    fig.savefig(plot_dir / "moment_recovery.pdf")
    plt.close(fig)

    summary = {
        "n_test": int(len(metrics_df)),
        "mean_l1": float(metrics_df["l1"].mean()),
        "mean_jsd": float(metrics_df["jsd"].mean()),
        "mean_active_components": float(metrics_df["active_components"].mean()),
        "per_config_mean_l1": metrics_df.groupby("scalar_config")["l1"].mean().to_dict(),
        "per_config_mean_jsd": metrics_df.groupby("scalar_config")["jsd"].mean().to_dict(),
    }
    with open(eval_dir / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def cli(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate a Dirichlet MDN run")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--album-per-cfg", type=int, default=3)
    args = p.parse_args(argv)
    evaluate_run(Path(args.run_dir),
                 device_override=args.device,
                 album_per_cfg=int(args.album_per_cfg))
    return 0


if __name__ == "__main__":
    sys.exit(cli())
