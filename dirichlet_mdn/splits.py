"""Train/val/test splits for the Dirichlet MDN.

Splits by ``(scalar_config, run_id)`` rather than by row. This is critical:
boxes from the same physical DNS run are highly correlated, so row-wise
splitting (as in ``MLPDF.py:177``) leaks information from train into val.

Two modes:

  - default          : per-config train/val/test split on run_ids
                       (default ratios 0.7 / 0.15 / 0.15).
  - holdout-config   : one named ``scalar_config`` is entirely held out as the
                       test set; the remaining configs are split into train and
                       val by run_id.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class SplitResult:
    train: np.ndarray   # parquet row indices
    val: np.ndarray
    test: np.ndarray
    description: dict


def _split_runs(
    run_ids: Sequence[int],
    ratios: Tuple[float, float, float],
    rng: np.random.Generator,
) -> Tuple[List[int], List[int], List[int]]:
    runs = sorted(set(int(r) for r in run_ids))
    n = len(runs)
    if n < 3:
        return runs, [], []
    perm = list(rng.permutation(n))
    shuffled = [runs[i] for i in perm]
    n_train = max(1, int(round(ratios[0] * n)))
    n_val = max(1, int(round(ratios[1] * n)))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
        n_train = n - n_val - 1
    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    return train, val, test


def _split_timesteps(
    timesteps: Sequence[int],
    ratios: Tuple[float, float, float],
) -> Tuple[List[int], List[int], List[int]]:
    """Contiguous early/mid/late split. Used as fallback when a config has only
    one run_id. No shuffle — keeps val/test from being interleaved time slices
    of the same Lagrangian evolution as train.
    """
    ts = sorted(set(int(t) for t in timesteps))
    n = len(ts)
    if n < 3:
        return ts, [], []
    n_train = max(1, int(round(ratios[0] * n)))
    n_val = max(1, int(round(ratios[1] * n)))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
        n_train = n - n_val - 1
    train = ts[:n_train]
    val = ts[n_train:n_train + n_val]
    test = ts[n_train + n_val:]
    return train, val, test


def make_split(
    meta: pd.DataFrame,
    *,
    seed: int = 0,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    holdout_config: Optional[str] = None,
    fallback: str = "timestep",
) -> SplitResult:
    """Build a leak-free split.

    Args:
        meta: parquet metadata as returned by ``data.load_metadata``.
        seed: deterministic seed for the by-run_id permutation.
        ratios: (train, val, test) fractions, used per-config.
        holdout_config: if set, this scalar_config goes entirely to ``test``;
            the rest are split with ``(0.8, 0.2, 0.0)``.
        fallback: behavior when a config has fewer than 3 distinct run_ids.
            * ``"timestep"`` — split by contiguous timestep blocks within that
              config. Acknowledged less-clean than by-run; flagged in the
              description.
            * ``"train_only"`` — put all rows of that config into train and
              log a warning. Reproduces the strict "no leak possible" guarantee
              at the cost of an empty val/test for that config.

    The returned ``description`` records, per config, the mode actually used
    and the resolved run/timestep partition.
    """
    if fallback not in ("timestep", "train_only"):
        raise ValueError(f"fallback must be 'timestep' or 'train_only', got {fallback!r}")
    rng = np.random.default_rng(seed)
    description: dict = {
        "seed": seed,
        "ratios": list(ratios),
        "holdout_config": holdout_config,
        "fallback": fallback,
        "per_config": {},
        "totals": {},
    }

    if holdout_config is not None:
        if holdout_config not in set(meta["scalar_config"].unique()):
            raise ValueError(
                f"holdout_config={holdout_config!r} not present in metadata "
                f"(have {sorted(meta['scalar_config'].unique())})"
            )

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for cfg, cfg_meta in meta.groupby("scalar_config", sort=True):
        idx_in_cfg = cfg_meta.index.to_numpy()
        if holdout_config is not None and cfg == holdout_config:
            test_idx.extend(idx_in_cfg.tolist())
            description["per_config"][cfg] = {
                "mode": "held_out_as_test",
                "n_runs": int(cfg_meta["run_id"].nunique()),
                "n_rows": int(len(cfg_meta)),
                "train_runs": [], "val_runs": [],
                "test_runs": sorted(set(int(r) for r in cfg_meta["run_id"].unique())),
            }
            continue

        if holdout_config is not None:
            cfg_ratios = (0.8, 0.2, 0.0)
        else:
            cfg_ratios = ratios

        n_runs = cfg_meta["run_id"].nunique()
        if n_runs >= 3:
            runs_train, runs_val, runs_test = _split_runs(
                cfg_meta["run_id"].unique().tolist(), cfg_ratios, rng,
            )
            train_mask = cfg_meta["run_id"].isin(runs_train)
            val_mask = cfg_meta["run_id"].isin(runs_val)
            test_mask = cfg_meta["run_id"].isin(runs_test)
            mode = "by_run_id" if holdout_config is None else "by_run_id_with_holdout"
            partition_log = {
                "train_runs": runs_train,
                "val_runs": runs_val,
                "test_runs": runs_test,
            }
        elif fallback == "timestep":
            t_train, t_val, t_test = _split_timesteps(
                cfg_meta["timestep"].unique().tolist(), cfg_ratios,
            )
            train_mask = cfg_meta["timestep"].isin(t_train)
            val_mask = cfg_meta["timestep"].isin(t_val)
            test_mask = cfg_meta["timestep"].isin(t_test)
            mode = f"by_timestep_fallback(n_runs={n_runs})"
            partition_log = {
                "train_timesteps": t_train,
                "val_timesteps": t_val,
                "test_timesteps": t_test,
            }
        else:  # train_only
            train_mask = pd.Series(True, index=cfg_meta.index)
            val_mask = pd.Series(False, index=cfg_meta.index)
            test_mask = pd.Series(False, index=cfg_meta.index)
            mode = f"train_only_fallback(n_runs={n_runs})"
            partition_log = {
                "train_runs": sorted(set(int(r) for r in cfg_meta["run_id"].unique())),
                "val_runs": [],
                "test_runs": [],
            }

        train_idx.extend(cfg_meta.index[train_mask].tolist())
        val_idx.extend(cfg_meta.index[val_mask].tolist())
        test_idx.extend(cfg_meta.index[test_mask].tolist())

        description["per_config"][cfg] = {
            "mode": mode,
            "n_runs": int(n_runs),
            "n_rows": int(len(cfg_meta)),
            "ratios_used": list(cfg_ratios),
            **partition_log,
        }

    train_arr = np.asarray(sorted(set(train_idx)), dtype=np.int64)
    val_arr = np.asarray(sorted(set(val_idx)), dtype=np.int64)
    test_arr = np.asarray(sorted(set(test_idx)), dtype=np.int64)

    description["totals"] = {
        "train": int(len(train_arr)),
        "val": int(len(val_arr)),
        "test": int(len(test_arr)),
        "all": int(len(meta)),
    }
    return SplitResult(train=train_arr, val=val_arr, test=test_arr, description=description)


def save_split(result: SplitResult, path: str) -> None:
    payload = {
        "train": result.train.tolist(),
        "val": result.val.tolist(),
        "test": result.test.tolist(),
        "description": result.description,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_split(path: str) -> SplitResult:
    with open(path, "r") as f:
        d = json.load(f)
    return SplitResult(
        train=np.asarray(d["train"], dtype=np.int64),
        val=np.asarray(d["val"], dtype=np.int64),
        test=np.asarray(d["test"], dtype=np.int64),
        description=d["description"],
    )


__all__ = ["SplitResult", "make_split", "save_split", "load_split"]
