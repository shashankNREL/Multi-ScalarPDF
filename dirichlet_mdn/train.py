"""Training CLI for the Dirichlet Mixture Density Network.

Usage:
    python -m dirichlet_mdn.train \\
      --parquet hybrid_dataset_all/all_cases_w64_w128_checkpoint_metadata.parquet \\
      --hdf5-dir hybrid_dataset_all \\
      --input-moments 4 --K 8 --epochs 200 \\
      --tag baseline_M4_K8

Outputs land in ``<output-dir>/<tag>/``:
  manifest.json   - full config + dataset stats
  splits.json     - train/val/test parquet row indices and provenance
  scaler.pkl      - sklearn RobustScaler fit on the train moments
  model_best.pt   - best-val-NLL checkpoint
  model_last.pt   - last-epoch checkpoint
  tb/             - TensorBoard event files
  train.log       - per-epoch loss table
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import joblib
import numpy as np
import torch
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .bin_grid import bin_grid
from .data import (
    HybridPDFDataset,
    collate_records,
    load_metadata,
    validate_dataset,
)
from .losses import HistogramNLL, LossWeights, combined_loss
from .model import DirichletMDN
from .splits import load_split, make_split, save_split


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    parquet: str
    hdf5_dir: str
    input_moments: int
    K: int
    hidden: int
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    lambda_mom: float
    lambda_ent: float
    alpha_min: float
    alpha_clip: float
    seed: int
    output_dir: str
    tag: str
    device: str
    holdout_config: Optional[str]
    split_fallback: str
    splits_json: Optional[str]
    patience: int
    grad_clip: float
    num_workers: int
    num_bins: int
    zst: float
    uniform_bins: bool
    log_every_n_batches: int
    cache_histograms: bool
    flat_histograms: bool
    max_moment_discrepancy: float
    selection_metric: str
    moment_acceptance_tolerance: float


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


# ---------------------------------------------------------------------------
# Scaler wrapper
# ---------------------------------------------------------------------------


class TorchScaler:
    """Wraps an sklearn RobustScaler so it can be applied to a torch tensor.

    Avoids a numpy round-trip during the training inner loop.
    """

    def __init__(self, scaler: RobustScaler) -> None:
        self.scaler = scaler
        self.center_ = torch.as_tensor(scaler.center_, dtype=torch.float32)
        scale = np.asarray(scaler.scale_, dtype=np.float32)
        scale = np.where(scale < 1e-6, 1.0, scale)
        self.scale_ = torch.as_tensor(scale, dtype=torch.float32)

    @classmethod
    def from_artifact(cls, artifact: Dict) -> "TorchScaler":
        if artifact.get("format_version") != 1:
            raise ValueError("unsupported input-transform artifact version")
        if artifact.get("type") != "robust_scaler_affine":
            raise ValueError("unsupported input-transform artifact type")
        center = np.asarray(artifact["center"], dtype=np.float32)
        scale = np.asarray(artifact["scale"], dtype=np.float32)
        if center.ndim != 1 or scale.shape != center.shape:
            raise ValueError("invalid input-transform center/scale shapes")
        if not np.isfinite(center).all() or not np.isfinite(scale).all():
            raise ValueError("input-transform parameters must be finite")
        if np.any(scale <= 0.0):
            raise ValueError("input-transform scales must be positive")
        obj = cls.__new__(cls)
        obj.scaler = None
        obj.center_ = torch.as_tensor(center, dtype=torch.float32)
        obj.scale_ = torch.as_tensor(scale, dtype=torch.float32)
        return obj

    def to(self, device: torch.device) -> "TorchScaler":
        self.center_ = self.center_.to(device)
        self.scale_ = self.scale_.to(device)
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.center_) / self.scale_

    def to_artifact(self) -> Dict:
        """Return the exact affine transform used by the model."""
        return {
            "format_version": 1,
            "type": "robust_scaler_affine",
            "center": self.center_.detach().cpu().tolist(),
            "scale": self.scale_.detach().cpu().tolist(),
            "formula": "(x - center) / scale",
        }


# ---------------------------------------------------------------------------
# Evaluation pass
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    nll_module: HistogramNLL,
    scaler: TorchScaler,
    device: torch.device,
    input_moments: int,
    weights: LossWeights,
) -> Dict[str, float]:
    model.eval()
    # Accumulate on-device to avoid per-batch GPU->CPU sync (MPS is very
    # sensitive to .item() in hot loops). Sync once at the end.
    sums = {k: torch.zeros((), device=device, dtype=torch.float32)
            for k in ("nll", "mom", "ent", "total")}
    count = 0
    for batch in loader:
        m = batch["moments"].to(device, non_blocking=True)
        h = batch["histogram"].to(device, non_blocking=True)
        m_scaled = scaler(m)
        pi, alpha = model(m_scaled)
        out = combined_loss(
            pi, alpha, h, m, nll_module=nll_module,
            input_moments=input_moments, weights=weights,
        )
        bs = m.shape[0]
        sums["nll"]   += out.nll.detach()     * bs
        sums["mom"]   += out.mom.detach()     * bs
        sums["ent"]   += out.entropy.detach() * bs
        sums["total"] += out.total.detach()   * bs
        count += bs
    if count == 0:
        return {k: float("nan") for k in sums}
    return {k: float(v.item() / count) for k, v in sums.items()}


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


def _dataset_summary(meta) -> dict:
    return {
        "n_rows": int(len(meta)),
        "scalar_configs": sorted(meta["scalar_config"].unique().tolist()),
        "per_config_counts": meta["scalar_config"].value_counts().to_dict(),
        "box_widths": sorted(int(x) for x in meta["box_width"].unique().tolist()),
        "run_ids": sorted(int(x) for x in meta["run_id"].unique().tolist()),
        "timesteps": sorted(int(x) for x in meta["timestep"].unique().tolist()),
    }


def write_manifest(cfg: TrainConfig, run_dir: Path, extras: dict) -> None:
    manifest = {
        "config": asdict(cfg),
        "extras": extras,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------


def setup_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"dirichlet_mdn.train.{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(run_dir / "train.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def train(cfg: TrainConfig) -> Path:
    if cfg.selection_metric not in {"total", "nll"}:
        raise ValueError("selection_metric must be 'total' or 'nll'")
    if cfg.moment_acceptance_tolerance <= 0.0:
        raise ValueError("moment_acceptance_tolerance must be positive")
    if cfg.max_moment_discrepancy <= 0.0:
        raise ValueError("max_moment_discrepancy must be positive")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    timestamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path(cfg.output_dir) / f"{timestamp}_{cfg.tag}_{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    log = setup_logger(run_dir)
    log.info(f"Run dir: {run_dir}")
    log.info(f"Config: {asdict(cfg)}")

    meta = load_metadata(cfg.parquet)
    log.info(f"Loaded metadata: {len(meta)} rows, configs={sorted(meta['scalar_config'].unique().tolist())}")
    validation = validate_dataset(
        cfg.parquet,
        cfg.hdf5_dir,
        num_bins=cfg.num_bins,
        zst=cfg.zst,
        uniform_bins=cfg.uniform_bins,
        max_moment_discrepancy=cfg.max_moment_discrepancy,
    )
    with open(run_dir / "dataset_validation.json", "w") as f:
        json.dump(validation, f, indent=2)
    log.info(f"Validated all dataset rows: {validation}")

    if cfg.splits_json:
        split = load_split(cfg.splits_json, meta=meta)
        log.info(f"Loaded pre-computed split from {cfg.splits_json}")
    else:
        split = make_split(
            meta, seed=cfg.seed,
            holdout_config=(None if cfg.holdout_config in (None, "", "none") else cfg.holdout_config),
            fallback=cfg.split_fallback,
        )
    save_split(split, str(run_dir / "splits.json"))
    log.info(f"Split totals: {split.description['totals']}")
    if len(split.train) == 0 or len(split.val) == 0:
        raise ValueError("training and validation splits must both be non-empty")

    train_ds = HybridPDFDataset(
        cfg.parquet, cfg.hdf5_dir,
        input_moments=cfg.input_moments,
        indices=split.train,
        num_bins=cfg.num_bins, zst=cfg.zst, uniform_bins=cfg.uniform_bins,
        cache_histograms=cfg.cache_histograms,
        flat_histograms=cfg.flat_histograms,
        max_moment_discrepancy=cfg.max_moment_discrepancy,
    )
    val_ds = HybridPDFDataset(
        cfg.parquet, cfg.hdf5_dir,
        input_moments=cfg.input_moments,
        indices=split.val,
        num_bins=cfg.num_bins, zst=cfg.zst, uniform_bins=cfg.uniform_bins,
        cache_histograms=cfg.cache_histograms,
        flat_histograms=cfg.flat_histograms,
        max_moment_discrepancy=cfg.max_moment_discrepancy,
    )

    scaler = HybridPDFDataset.fit_input_scaler(train_ds.meta, cfg.input_moments)
    joblib.dump(scaler, run_dir / "scaler.pkl")
    log.info(f"Fit RobustScaler: center={scaler.center_}, scale={scaler.scale_}")

    device = resolve_device(cfg.device)
    log.info(f"Device: {device}")

    torch_scaler = TorchScaler(scaler).to(device)
    with open(run_dir / "input_transform.json", "w") as f:
        json.dump(torch_scaler.to_artifact(), f, indent=2)
    grid = train_ds.grid
    nll_module = HistogramNLL(grid).to(device)
    model = DirichletMDN(
        n_in=cfg.input_moments, K=cfg.K, hidden=cfg.hidden,
        alpha_min=cfg.alpha_min, alpha_clip=cfg.alpha_clip,
    ).to(device)
    log.info(f"Model params: {sum(p.numel() for p in model.parameters())}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=collate_records,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate_records,
        pin_memory=(device.type == "cuda"),
    )

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1))
    weights = LossWeights(lambda_mom=cfg.lambda_mom, lambda_ent=cfg.lambda_ent)

    writer = SummaryWriter(log_dir=str(run_dir / "tb"))

    best_val = float("inf")
    best_path = run_dir / "model_best.pt"
    last_path = run_dir / "model_last.pt"
    epochs_without_improve = 0

    n_train_batches = len(train_loader)
    log.info(
        f"Starting training: {cfg.epochs} epochs, "
        f"{n_train_batches} train batches/epoch (batch_size={cfg.batch_size}), "
        f"{len(val_loader)} val batches/epoch, log_every_n_batches={cfg.log_every_n_batches}"
    )

    def _zero_accum() -> Dict[str, torch.Tensor]:
        return {k: torch.zeros((), device=device, dtype=torch.float32)
                for k in ("nll", "mom", "ent", "total")}

    for epoch in range(cfg.epochs):
        model.train()
        epoch_sums = _zero_accum()
        win_sums = _zero_accum()
        n_seen = 0
        win_n = 0
        epoch_t0 = time.monotonic()
        batch_t0 = time.monotonic()
        last_log_batch = 0
        for batch_idx, batch in enumerate(train_loader):
            m = batch["moments"].to(device, non_blocking=True)
            h = batch["histogram"].to(device, non_blocking=True)
            m_scaled = torch_scaler(m)
            pi, alpha = model(m_scaled)
            out = combined_loss(
                pi, alpha, h, m, nll_module=nll_module,
                input_moments=cfg.input_moments, weights=weights,
            )
            opt.zero_grad(set_to_none=True)
            out.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            bs = m.shape[0]
            # On-device accumulation — no GPU->CPU sync in the hot loop.
            nll_d = out.nll.detach()
            mom_d = out.mom.detach()
            ent_d = out.entropy.detach()
            tot_d = out.total.detach()
            epoch_sums["nll"]   += nll_d * bs
            epoch_sums["mom"]   += mom_d * bs
            epoch_sums["ent"]   += ent_d * bs
            epoch_sums["total"] += tot_d * bs
            win_sums["nll"]   += nll_d * bs
            win_sums["mom"]   += mom_d * bs
            win_sums["ent"]   += ent_d * bs
            win_sums["total"] += tot_d * bs
            n_seen += bs
            win_n += bs

            if cfg.log_every_n_batches > 0 and (
                (batch_idx + 1) % cfg.log_every_n_batches == 0
                or (batch_idx + 1) == n_train_batches
            ):
                # One sync point — pull the window averages off the device.
                avg = {k: float((v / max(win_n, 1)).item()) for k, v in win_sums.items()}
                dt = time.monotonic() - batch_t0
                batches_this_win = (batch_idx + 1) - last_log_batch
                batches_per_sec = batches_this_win / max(dt, 1e-9)
                log.info(
                    f"epoch {epoch:3d}  [batch {batch_idx + 1:>4d}/{n_train_batches}]  "
                    f"nll={avg['nll']:.4f} mom={avg['mom']:.4f} "
                    f"ent={avg['ent']:.4f} total={avg['total']:.4f}  "
                    f"({batches_per_sec:.2f} batch/s, elapsed {time.monotonic() - epoch_t0:.1f}s)"
                )
                batch_t0 = time.monotonic()
                last_log_batch = batch_idx + 1
                win_sums = _zero_accum()
                win_n = 0
        sched.step()
        train_metrics = {k: float((v / max(n_seen, 1)).item()) for k, v in epoch_sums.items()}
        train_time = time.monotonic() - epoch_t0

        val_t0 = time.monotonic()
        val_metrics = evaluate_loss(
            model, val_loader, nll_module, torch_scaler, device,
            cfg.input_moments, weights,
        ) if len(val_ds) > 0 else {k: float("nan") for k in ("nll", "mom", "ent", "total")}
        val_time = time.monotonic() - val_t0

        for k, v in train_metrics.items():
            writer.add_scalar(f"train/{k}", v, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)
        writer.add_scalar("opt/lr", opt.param_groups[0]["lr"], epoch)
        writer.add_scalar("time/train_sec", train_time, epoch)
        writer.add_scalar("time/val_sec", val_time, epoch)
        writer.flush()

        log.info(
            f"epoch {epoch:3d}  "
            f"train nll={train_metrics['nll']:.4f} mom={train_metrics['mom']:.4f} "
            f"ent={train_metrics['ent']:.4f} total={train_metrics['total']:.4f}  "
            f"val nll={val_metrics['nll']:.4f} total={val_metrics['total']:.4f}  "
            f"lr={opt.param_groups[0]['lr']:.3g}  "
            f"(train {train_time:.1f}s, val {val_time:.1f}s)"
        )

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "val_metrics": val_metrics,
            "train_metrics": train_metrics,
        }, last_path)

        selection_value = val_metrics[cfg.selection_metric]
        if len(val_ds) > 0 and selection_value < best_val:
            best_val = selection_value
            epochs_without_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
                "train_metrics": train_metrics,
            }, best_path)
            log.info(
                f"  -> new best val {cfg.selection_metric} = {best_val:.4f}"
            )
        else:
            epochs_without_improve += 1
            if cfg.patience > 0 and epochs_without_improve >= cfg.patience:
                log.info(f"Early stop at epoch {epoch} (patience={cfg.patience})")
                break

    writer.close()
    train_ds.close()
    val_ds.close()

    extras = {
        "dataset_summary": _dataset_summary(meta),
        "split_totals": split.description["totals"],
        "selection_metric": cfg.selection_metric,
        "best_val_metric": float(best_val) if best_val != float("inf") else None,
        "num_simplex_cells": nll_module.num_simplex_cells,
        "deviation_note": (
            "Histogram-categorical NLL with simplex renormalization is used in "
            "place of the per-particle NLL of the proposal — the stored "
            "dataset is histograms, not particles. See "
            "DirichletMDN_implementation.md."
        ),
    }
    write_manifest(cfg, run_dir, extras)
    log.info(f"Done. Run dir: {run_dir}")
    return run_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a Dirichlet MDN on hybrid dataset")
    p.add_argument("--parquet", required=True, help="Path to parquet metadata")
    p.add_argument("--hdf5-dir", required=True, help="Directory of HDF5 shards")
    p.add_argument("--input-moments", type=int, default=4, choices=(4, 5))
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--lambda-mom", type=float, default=0.5)
    p.add_argument("--lambda-ent", type=float, default=1e-3)
    p.add_argument("--alpha-min", type=float, default=1.0)
    p.add_argument("--alpha-clip", type=float, default=1e3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="dirichlet_mdn_runs")
    p.add_argument("--tag", default="baseline")
    p.add_argument("--device", default="auto")
    p.add_argument("--holdout-config", default=None,
                   help="If set, this scalar_config is the entire test set")
    p.add_argument("--split-fallback", default="timestep",
                   choices=("timestep", "train_only"),
                   help="Deprecated compatibility option; splits never cross run IDs")
    p.add_argument("--splits-json", default=None,
                   help="Optional path to a pre-computed splits.json "
                        "(e.g. from preview_split.py). If set, "
                        "--seed/--holdout-config/--split-fallback are ignored "
                        "for split construction.")
    p.add_argument("--patience", type=int, default=30, help="0 disables early stop")
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--num-bins", type=int, default=64)
    p.add_argument("--zst", type=float, default=0.1)
    p.add_argument("--uniform-bins", action="store_true")
    p.add_argument("--log-every-n-batches", type=int, default=25,
                   help="Print a progress line every N train batches (0 disables)")
    p.add_argument("--cache-histograms", action="store_true",
                   help="Cache every histogram in RAM after first read. "
                        "~2.6 GB for the full dataset, eliminates per-batch h5py I/O.")
    p.add_argument("--flat-histograms", action="store_true",
                   help="Store/pass histograms pre-flattened to in-simplex cells "
                        "(skips a reshape+mask inside HistogramNLL every batch).")
    p.add_argument("--max-moment-discrepancy", type=float, default=0.05,
                   help="Maximum accepted direct-vs-histogram moment error")
    p.add_argument("--selection-metric", choices=("total", "nll"), default="total",
                   help="Validation metric used for checkpoints and early stopping")
    p.add_argument("--moment-acceptance-tolerance", type=float, default=0.05,
                   help="Maximum accepted moment error divided by physical range")
    return p


def cli(argv: Optional[Iterable[str]] = None) -> int:
    args = build_argparser().parse_args(list(argv) if argv is not None else None)
    cfg = TrainConfig(
        parquet=args.parquet,
        hdf5_dir=args.hdf5_dir,
        input_moments=int(args.input_moments),
        K=int(args.K),
        hidden=int(args.hidden),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        lambda_mom=float(args.lambda_mom),
        lambda_ent=float(args.lambda_ent),
        alpha_min=float(args.alpha_min),
        alpha_clip=float(args.alpha_clip),
        seed=int(args.seed),
        output_dir=str(args.output_dir),
        tag=str(args.tag),
        device=str(args.device),
        holdout_config=args.holdout_config,
        split_fallback=str(args.split_fallback),
        splits_json=args.splits_json,
        patience=int(args.patience),
        grad_clip=float(args.grad_clip),
        num_workers=int(args.num_workers),
        num_bins=int(args.num_bins),
        zst=float(args.zst),
        uniform_bins=bool(args.uniform_bins),
        log_every_n_batches=int(args.log_every_n_batches),
        cache_histograms=bool(args.cache_histograms),
        flat_histograms=bool(args.flat_histograms),
        max_moment_discrepancy=float(args.max_moment_discrepancy),
        selection_metric=str(args.selection_metric),
        moment_acceptance_tolerance=float(args.moment_acceptance_tolerance),
    )
    train(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(cli())
