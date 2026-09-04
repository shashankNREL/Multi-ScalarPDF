"""Train/val/test splits for the Dirichlet MDN.

Splits by global ``run_id`` rather than by row or independently by scalar
configuration. Boxes and scalar configurations from the same physical DNS run
share a velocity realization, so that run must stay in one partition.

Two modes:

  - default          : one global train/val/test split on run_ids
                       (default ratios 0.7 / 0.15 / 0.15).
  - holdout-config   : one named ``scalar_config`` is the entire test set; the
                       remaining configs are split into train and val only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .data import metadata_fingerprint


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
    perm = list(rng.permutation(n))
    shuffled = [runs[i] for i in perm]
    counts = _allocate_counts(n, ratios)
    n_train, n_val, _ = counts
    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    return train, val, test


def _allocate_counts(
    n: int,
    ratios: Tuple[float, float, float],
) -> Tuple[int, int, int]:
    """Allocate exactly ``n`` items while respecting zero-valued ratios."""
    values = np.asarray(ratios, dtype=np.float64)
    if np.any(values < 0.0) or not np.isclose(values.sum(), 1.0, atol=1e-9):
        raise ValueError(f"split ratios must be non-negative and sum to one: {ratios}")
    raw = values * n
    counts = np.floor(raw).astype(int)
    remainder = n - int(counts.sum())
    order = np.argsort(-(raw - counts), kind="stable")
    for index in order[:remainder]:
        counts[index] += 1

    positive = np.flatnonzero(values > 0.0)
    if n >= len(positive):
        for index in positive:
            if counts[index] == 0:
                donors = [
                    donor for donor in positive
                    if counts[donor] > 1
                ]
                if donors:
                    donor = max(donors, key=lambda item: counts[item])
                    counts[donor] -= 1
                    counts[index] += 1
    return tuple(int(value) for value in counts)


def make_split(
    meta: pd.DataFrame,
    *,
    seed: int = 0,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    holdout_config: Optional[str] = None,
    fallback: str = "timestep",
) -> SplitResult:
    """Build a run-isolated split.

    Args:
        meta: parquet metadata as returned by ``data.load_metadata``.
        seed: deterministic seed for the by-run_id permutation.
        ratios: (train, val, test) fractions, applied to global run IDs.
        holdout_config: if set, this scalar_config goes entirely to ``test``;
            the rest are split with ``(0.8, 0.2, 0.0)``.
        fallback: deprecated compatibility argument. Timestep fallback is never
            used because it would leak a physical run across partitions.

    The returned ``description`` records the grouping policy and resolved runs.
    """
    if fallback not in ("timestep", "train_only"):
        raise ValueError(f"fallback must be 'timestep' or 'train_only', got {fallback!r}")
    _allocate_counts(0, ratios)
    rng = np.random.default_rng(seed)
    description: dict = {
        "seed": seed,
        "ratios": list(ratios),
        "holdout_config": holdout_config,
        "deprecated_fallback_argument": fallback,
        "grouping": (
            "configuration_holdout_plus_global_train_val_run_id"
            if holdout_config is not None else "global_run_id"
        ),
        "dataset_fingerprint": metadata_fingerprint(meta),
        "per_config": {},
        "totals": {},
    }

    if holdout_config is not None:
        if holdout_config not in set(meta["scalar_config"].unique()):
            raise ValueError(
                f"holdout_config={holdout_config!r} not present in metadata "
                f"(have {sorted(meta['scalar_config'].unique())})"
            )
        holdout_runs = set(
            int(value) for value in
            meta.loc[meta["scalar_config"] == holdout_config, "run_id"].unique()
        )
        other_runs = set(
            int(value) for value in
            meta.loc[meta["scalar_config"] != holdout_config, "run_id"].unique()
        )
        shared_runs = sorted(holdout_runs & other_runs)
        if shared_runs:
            raise ValueError(
                "a pure configuration holdout cannot also isolate physical runs: "
                f"{len(shared_runs)} run_id value(s) occur in both {holdout_config!r} "
                "and non-held configurations. Use independent run IDs for the "
                "held configuration or use the ordinary global-run split."
            )

    eligible = (
        meta[meta["scalar_config"] != holdout_config]
        if holdout_config is not None else meta
    )
    unique_runs = sorted(int(value) for value in eligible["run_id"].unique())
    active_ratios = (0.8, 0.2, 0.0) if holdout_config is not None else ratios

    runs_train, runs_val, runs_test = _split_runs(
        unique_runs, active_ratios, rng,
    )
    train_mask = eligible["run_id"].isin(runs_train)
    val_mask = eligible["run_id"].isin(runs_val)
    test_mask = eligible["run_id"].isin(runs_test)
    split_mode = "global_run_id"

    train_idx = eligible.index[train_mask].tolist()
    val_idx = eligible.index[val_mask].tolist()
    test_idx = eligible.index[test_mask].tolist()
    if holdout_config is not None:
        test_idx.extend(meta.index[meta["scalar_config"] == holdout_config].tolist())

    description["global_partition"] = {
        "mode": split_mode,
        "ratios": list(active_ratios),
        "train_runs": runs_train,
        "val_runs": runs_val,
        "test_runs": runs_test,
    }
    train_set, val_set, test_set = set(train_idx), set(val_idx), set(test_idx)
    for cfg, cfg_meta in meta.groupby("scalar_config", sort=True):
        cfg_indices = set(int(index) for index in cfg_meta.index)
        mode = "held_out_as_test" if cfg == holdout_config else split_mode
        description["per_config"][cfg] = {
            "mode": mode,
            "n_runs": int(cfg_meta["run_id"].nunique()),
            "n_rows": int(len(cfg_meta)),
            "train_runs": sorted(
                int(value) for value in
                cfg_meta.loc[list(cfg_indices & train_set), "run_id"].unique()
            ),
            "val_runs": sorted(
                int(value) for value in
                cfg_meta.loc[list(cfg_indices & val_set), "run_id"].unique()
            ),
            "test_runs": sorted(
                int(value) for value in
                cfg_meta.loc[list(cfg_indices & test_set), "run_id"].unique()
            ),
        }

    train_arr = np.asarray(sorted(train_idx), dtype=np.int64)
    val_arr = np.asarray(sorted(val_idx), dtype=np.int64)
    test_arr = np.asarray(sorted(test_idx), dtype=np.int64)

    description["totals"] = {
        "train": int(len(train_arr)),
        "val": int(len(val_arr)),
        "test": int(len(test_arr)),
        "all": int(len(meta)),
    }
    result = SplitResult(
        train=train_arr, val=val_arr, test=test_arr, description=description,
    )
    validate_split(result, meta)
    return result


def validate_split(
    result: SplitResult,
    meta: pd.DataFrame,
    *,
    allow_legacy_fingerprint: bool = False,
    enforce_run_isolation: bool = True,
) -> None:
    """Validate identity, disjointness, uniqueness, and full row coverage."""
    expected_fingerprint = metadata_fingerprint(meta)
    actual_fingerprint = result.description.get("dataset_fingerprint")
    if (
        actual_fingerprint != expected_fingerprint
        and not (allow_legacy_fingerprint and actual_fingerprint is None)
    ):
        raise ValueError(
            "split dataset fingerprint does not match the loaded metadata; "
            "regenerate the split from this Parquet file"
        )

    arrays = {
        "train": result.train,
        "val": result.val,
        "test": result.test,
    }
    sets = {}
    for name, values in arrays.items():
        if len(values) != len(np.unique(values)):
            raise ValueError(f"{name} split contains duplicate indices")
        if len(values) and (values.min() < 0 or values.max() >= len(meta)):
            raise ValueError(f"{name} split contains out-of-range indices")
        sets[name] = set(int(value) for value in values)

    overlap = (
        (sets["train"] & sets["val"])
        | (sets["train"] & sets["test"])
        | (sets["val"] & sets["test"])
    )
    if overlap:
        raise ValueError(f"split partitions overlap at {len(overlap)} row(s)")
    covered = sets["train"] | sets["val"] | sets["test"]
    expected = set(range(len(meta)))
    if covered != expected:
        raise ValueError(
            f"split covers {len(covered)} of {len(expected)} metadata rows"
        )

    if enforce_run_isolation:
        run_sets = {
            name: set(meta.iloc[values]["run_id"])
            for name, values in arrays.items()
        }
        run_overlap = (
            (run_sets["train"] & run_sets["val"])
            | (run_sets["train"] & run_sets["test"])
            | (run_sets["val"] & run_sets["test"])
        )
        if run_overlap:
            raise ValueError(
                "physical run IDs cross split boundaries: "
                f"{sorted(run_overlap)}"
            )

    holdout_config = result.description.get("holdout_config")
    if holdout_config is not None:
        test_configs = set(meta.iloc[result.test]["scalar_config"])
        if test_configs != {holdout_config}:
            raise ValueError(
                "holdout test set is contaminated by non-held configurations: "
                f"{sorted(test_configs - {holdout_config})}"
            )


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


def load_split(
    path: str,
    meta: Optional[pd.DataFrame] = None,
    *,
    allow_legacy_fingerprint: bool = False,
    enforce_run_isolation: bool = True,
) -> SplitResult:
    with open(path, "r") as f:
        d = json.load(f)
    result = SplitResult(
        train=np.asarray(d["train"], dtype=np.int64),
        val=np.asarray(d["val"], dtype=np.int64),
        test=np.asarray(d["test"], dtype=np.int64),
        description=d["description"],
    )
    if meta is not None:
        validate_split(
            result,
            meta,
            allow_legacy_fingerprint=allow_legacy_fingerprint,
            enforce_run_isolation=enforce_run_isolation,
        )
    return result


__all__ = [
    "SplitResult", "make_split", "save_split", "load_split", "validate_split",
]
