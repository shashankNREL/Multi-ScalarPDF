"""Preview & freeze a train/val/test split before training.

``train.py`` will compute and save the split itself, deterministically per
seed. This CLI lets you:

  1. *See* the split's per-config composition before paying the training cost.
  2. Optionally write ``splits.json`` to a known path so multiple training
     runs (e.g. K=4 vs K=8, input_moments=4 vs 5) can be compared on the
     same train/val/test partition — eliminates seed-induced noise.

Outputs under ``<out-dir>/``:

  splits.json        — same format as the one ``train.py`` would write
  split_report.md    — human-readable per-config breakdown
  split_summary.csv  — per-config counts table

Usage:

  PYTHONPATH=. python -m dirichlet_mdn.preview_split \\
    --parquet hybrid_dataset_mpi/all_cases_w64_w128_metadata.parquet \\
    --out-dir splits/by_run_id_seed0 \\
    --seed 0 --train 0.7 --val 0.15 --test 0.15

  # OOD: hold out PI5 as the entire test set
  PYTHONPATH=. python -m dirichlet_mdn.preview_split \\
    --parquet hybrid_dataset_mpi/all_cases_w64_w128_metadata.parquet \\
    --out-dir splits/ood_holdout_PI5 \\
    --holdout-config PI5
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from .data import load_metadata
from .splits import make_split, save_split


def _format_runs(runs) -> str:
    if not runs:
        return "—"
    runs = sorted(int(r) for r in runs)
    return ", ".join(str(r) for r in runs)


def write_report(meta: pd.DataFrame, split, out_dir: Path) -> None:
    desc = split.description
    lines: list[str] = []
    lines.append("# Train / Val / Test split preview")
    lines.append("")
    lines.append(f"- **seed:** {desc['seed']}")
    lines.append(f"- **ratios (train/val/test):** {desc['ratios']}")
    lines.append(f"- **holdout_config:** {desc['holdout_config']}")
    lines.append(f"- **fallback:** {desc['fallback']}")
    t = desc["totals"]
    pct = lambda n: f"{100*n/max(t['all'],1):.1f}%"
    lines.append(
        f"- **totals:** train {t['train']:,} ({pct(t['train'])}), "
        f"val {t['val']:,} ({pct(t['val'])}), "
        f"test {t['test']:,} ({pct(t['test'])}), "
        f"all {t['all']:,}"
    )
    lines.append("")
    lines.append("## Per-config partition")
    lines.append("")
    lines.append("| scalar_config | mode | n_runs | n_rows | train_runs | val_runs | test_runs |")
    lines.append("|---|---|---|---|---|---|---|")
    rows: list[list[str]] = []
    for cfg, info in sorted(desc["per_config"].items()):
        mode = info["mode"]
        if "train_runs" in info:
            tr = _format_runs(info["train_runs"])
            va = _format_runs(info["val_runs"])
            te = _format_runs(info["test_runs"])
        else:  # by_timestep_fallback
            tr = "ts:" + _format_runs(info.get("train_timesteps", []))
            va = "ts:" + _format_runs(info.get("val_timesteps", []))
            te = "ts:" + _format_runs(info.get("test_timesteps", []))
        lines.append(
            f"| {cfg} | {mode} | {info['n_runs']} | {info['n_rows']:,} | {tr} | {va} | {te} |"
        )
        rows.append([cfg, mode, str(info["n_runs"]), str(info["n_rows"]), tr, va, te])
    lines.append("")

    # also a per-config row-count breakdown of the actual split
    by_cfg = (
        meta.assign(_split=_label_split(meta, split))
        .groupby(["scalar_config", "_split"]).size().unstack(fill_value=0)
        .reindex(columns=["train", "val", "test"], fill_value=0)
        .sort_index()
    )
    lines.append("## Per-config row counts in each split")
    lines.append("")
    lines.append("| scalar_config | train | val | test | total |")
    lines.append("|---|---:|---:|---:|---:|")
    for cfg, r in by_cfg.iterrows():
        total = int(r.sum())
        lines.append(
            f"| {cfg} | {int(r['train']):,} | {int(r['val']):,} | {int(r['test']):,} | {total:,} |"
        )
    lines.append("")

    out_md = out_dir / "split_report.md"
    out_md.write_text("\n".join(lines))

    with open(out_dir / "split_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scalar_config", "split", "n_rows"])
        for cfg, r in by_cfg.iterrows():
            for s in ("train", "val", "test"):
                w.writerow([cfg, s, int(r[s])])


def _label_split(meta: pd.DataFrame, split) -> pd.Series:
    labels = pd.Series(["unused"] * len(meta), index=meta.index)
    labels.iloc[split.train] = "train"
    labels.iloc[split.val] = "val"
    labels.iloc[split.test] = "test"
    return labels


def run(
    parquet: str, out_dir: str, *,
    seed: int = 0,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    holdout_config: Optional[str] = None,
    fallback: str = "timestep",
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = load_metadata(parquet)
    split = make_split(
        meta, seed=seed, ratios=ratios,
        holdout_config=holdout_config, fallback=fallback,
    )
    save_split(split, str(out / "splits.json"))
    write_report(meta, split, out)

    t = split.description["totals"]
    print(f"[split] wrote {out/'splits.json'}")
    print(f"[split] totals  train={t['train']:,}  val={t['val']:,}  test={t['test']:,}  all={t['all']:,}")
    print(f"[split] report  {out/'split_report.md'}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Preview & freeze a train/val/test split")
    p.add_argument("--parquet", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train", type=float, default=0.70, help="train fraction")
    p.add_argument("--val",   type=float, default=0.15, help="val fraction")
    p.add_argument("--test",  type=float, default=0.15, help="test fraction")
    p.add_argument("--holdout-config", default=None,
                   help="If set, this scalar_config is the entire test set "
                        "(remaining configs split 80/20 train/val)")
    p.add_argument("--split-fallback", default="timestep",
                   choices=("timestep", "train_only"))
    return p


def cli(argv: Optional[Sequence[str]] = None) -> int:
    a = build_argparser().parse_args(list(argv) if argv is not None else None)
    s = a.train + a.val + a.test
    if abs(s - 1.0) > 1e-6:
        raise SystemExit(f"--train + --val + --test must sum to 1.0, got {s}")
    run(
        parquet=a.parquet, out_dir=a.out_dir, seed=int(a.seed),
        ratios=(float(a.train), float(a.val), float(a.test)),
        holdout_config=a.holdout_config, fallback=str(a.split_fallback),
    )
    return 0


if __name__ == "__main__":
    sys.exit(cli())
