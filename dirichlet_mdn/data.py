"""Hybrid parquet+HDF5 dataset adapter for Dirichlet MDN training.

Reads the metadata parquet (DNS-box moments + bookkeeping) and the
histograms from one-or-more HDF5 shards produced by ``EnsightPDFHybridDataset.py``.

Schema-tolerant: accepts either the older variant (``run_id``, ``box_width``,
``filter_width`` missing) or the newer one (``run_number``, ``filter_width``,
``box_width`` missing). The column used for the per-record ``hdf5_file`` is
resolved automatically.

Multi-shard from day one: shard file handles are opened lazily on first
access and cached for the life of the dataset.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.preprocessing import RobustScaler
from torch.utils.data import Dataset

from .bin_grid import BinGrid, bin_grid


SCALAR_CONFIG_COL = "scalar_config"
TIMESTEP_COL = "timestep"
SAMPLE_ID_COL = "sample_id"
LOCAL_INDEX_COL = "local_index"
HDF5_FILE_COL = "hdf5_file"
DATASET_FORMAT_VERSION = 2

MOMENT_COLS = ["mean_a", "var_a", "mean_b", "var_b", "cov_ab"]


def _resolve_column(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of the expected columns {candidates} found in parquet")


def load_metadata(parquet_path: str) -> pd.DataFrame:
    """Load the parquet metadata and normalize the schema to a canonical form.

    After this call, the returned frame has these guaranteed columns
    (regardless of which schema variant the writer used):
    ``scalar_config``, ``run_id``, ``timestep``, ``box_width``,
    ``mean_a``, ``var_a``, ``mean_b``, ``var_b``, ``cov_ab``,
    ``sample_id``, ``local_index``, ``hdf5_file``.
    """
    table = pq.read_table(parquet_path)
    df = table.to_pandas()

    run_col = _resolve_column(df, ("run_id", "run_number"))
    width_col = _resolve_column(df, ("box_width", "filter_width"))

    renamed = df.rename(columns={run_col: "run_id", width_col: "box_width"})

    required = [
        SCALAR_CONFIG_COL, "run_id", TIMESTEP_COL, "box_width",
        *MOMENT_COLS, SAMPLE_ID_COL, LOCAL_INDEX_COL, HDF5_FILE_COL,
    ]
    missing = [c for c in required if c not in renamed.columns]
    if missing:
        raise KeyError(f"parquet missing required columns: {missing}")

    result = renamed.reset_index(drop=True)
    if result[SCALAR_CONFIG_COL].isna().any():
        raise ValueError("scalar_config values must not be null")
    if result[HDF5_FILE_COL].isna().any():
        raise ValueError("hdf5_file values must not be null")
    numeric_columns = [
        *MOMENT_COLS, SAMPLE_ID_COL, LOCAL_INDEX_COL,
        "run_id", TIMESTEP_COL, "box_width",
    ]
    numeric = result[numeric_columns]
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError("parquet metadata contains non-finite moments or indices")
    integer_columns = [
        SAMPLE_ID_COL, LOCAL_INDEX_COL, "run_id", TIMESTEP_COL, "box_width",
    ]
    for column in integer_columns:
        values = result[column].to_numpy(dtype=np.float64)
        if not np.equal(values, np.floor(values)).all():
            raise ValueError(f"parquet {column} values must be integers")
    if (result[[SAMPLE_ID_COL, LOCAL_INDEX_COL, "run_id"]] < 0).any().any():
        raise ValueError("sample, local, and run indices must be non-negative")
    if (result["box_width"] <= 0).any():
        raise ValueError("box_width values must be positive")
    shard_names = result[HDF5_FILE_COL].astype(str)
    if any(
        Path(name).name != name or Path(name).is_absolute()
        for name in shard_names
    ):
        raise ValueError("hdf5_file values must be plain relative filenames")
    if (result[SCALAR_CONFIG_COL].astype(str).str.len() == 0).any():
        raise ValueError("scalar_config values must be non-empty")
    tol = 1e-6
    mean_a = result["mean_a"].to_numpy(dtype=np.float64)
    mean_b = result["mean_b"].to_numpy(dtype=np.float64)
    var_a = result["var_a"].to_numpy(dtype=np.float64)
    var_b = result["var_b"].to_numpy(dtype=np.float64)
    cov_ab = result["cov_ab"].to_numpy(dtype=np.float64)
    if (
        np.any(mean_a < -tol) or np.any(mean_b < -tol)
        or np.any(mean_a + mean_b > 1.0 + tol)
    ):
        raise ValueError("metadata means violate simplex bounds")
    if (
        np.any(var_a < -tol) or np.any(var_b < -tol)
        or np.any(var_a > mean_a * (1.0 - mean_a) + tol)
        or np.any(var_b > mean_b * (1.0 - mean_b) + tol)
    ):
        raise ValueError("metadata variances violate bounded-scalar limits")
    if np.any(cov_ab ** 2 > var_a * var_b + tol):
        raise ValueError("metadata covariance violates positive semidefiniteness")
    mean_z3 = 1.0 - mean_a - mean_b
    var_z3 = var_a + var_b + 2.0 * cov_ab
    if (
        np.any(var_z3 < -tol)
        or np.any(var_z3 > mean_z3 * (1.0 - mean_z3) + tol)
    ):
        raise ValueError("metadata implied third-scalar variance is not physical")
    if result[SAMPLE_ID_COL].duplicated().any():
        raise ValueError("parquet sample_id values must be unique")
    return result


def metadata_fingerprint(meta: pd.DataFrame) -> str:
    """Return a stable identity for the ordered dataset metadata."""
    columns = [
        SCALAR_CONFIG_COL, "run_id", TIMESTEP_COL, "box_width",
        *MOMENT_COLS, SAMPLE_ID_COL, LOCAL_INDEX_COL, HDF5_FILE_COL,
    ]
    missing = [column for column in columns if column not in meta.columns]
    if missing:
        raise KeyError(f"cannot fingerprint metadata; missing columns: {missing}")
    row_hashes = pd.util.hash_pandas_object(
        meta[columns], index=True, categorize=True,
    ).to_numpy(dtype=np.uint64)
    digest = hashlib.sha256()
    digest.update("\0".join(columns).encode("utf-8"))
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()


class HybridPDFDataset(Dataset):
    """One sample == one LES filter cell == one stored 2-D histogram + moments."""

    def __init__(
        self,
        parquet_path: str,
        hdf5_dir: str,
        *,
        input_moments: int = 5,
        indices: Optional[Sequence[int]] = None,
        num_bins: int = 64,
        zst: float = 0.1,
        uniform_bins: bool = False,
        sum_atol: float = 1e-4,
        cache_histograms: bool = False,
        flat_histograms: bool = False,
        max_moment_discrepancy: float | None = 0.05,
        grid_override: Optional[BinGrid] = None,
    ) -> None:
        if input_moments not in (4, 5):
            raise ValueError(f"input_moments must be 4 or 5, got {input_moments}")
        self.parquet_path = parquet_path
        self.hdf5_dir = Path(hdf5_dir)
        self.input_moments = input_moments
        self.sum_atol = sum_atol
        self.cache_histograms = cache_histograms
        self.flat_histograms = flat_histograms
        self.max_moment_discrepancy = max_moment_discrepancy

        self._meta_all = load_metadata(parquet_path)
        if (
            max_moment_discrepancy is not None
            and "moment_abs_err_max" in self._meta_all
        ):
            worst = float(self._meta_all["moment_abs_err_max"].max())
            if worst > max_moment_discrepancy:
                raise ValueError(
                    "direct and histogram moments disagree beyond the allowed "
                    f"tolerance: worst={worst:.6g}, "
                    f"limit={max_moment_discrepancy:.6g}"
                )
        if indices is None:
            self._indices = np.arange(len(self._meta_all), dtype=np.int64)
        else:
            self._indices = np.asarray(list(indices), dtype=np.int64)
        self.meta = self._meta_all.iloc[self._indices].reset_index(drop=True)

        self.grid: BinGrid = (
            grid_override
            if grid_override is not None
            else bin_grid(num_bins=num_bins, zst=zst, uniform=uniform_bins)
        )
        if int(self._meta_all.shape[0]) > 0 and grid_override is None:
            self._check_grid_matches_first_shard()

        # Precompute the flat in-simplex index (used when flat_histograms=True
        # or by callers that want to mask down to in-simplex cells).
        self._flat_simplex_index = np.flatnonzero(self.grid.simplex_mask.reshape(-1))

        self._handles: Dict[str, h5py.File] = {}
        self._hist_cache: Dict[int, np.ndarray] = {}

    def _check_grid_matches_first_shard(self) -> None:
        first_shard = self._meta_all[HDF5_FILE_COL].iloc[0]
        path = str(self.hdf5_dir / first_shard)
        from .bin_grid import validate_against_hdf5
        ok, diffs = validate_against_hdf5(
            path,
            num_bins=self.grid.num_bins,
            zst=self.grid.zst,
            uniform=self.grid.uniform,
        )
        if not ok:
            raise RuntimeError(
                f"Reconstructed bin grid does not match HDF5 shard {path}: {diffs}"
            )

    def _open_shard(self, shard_filename: str) -> h5py.File:
        handle = self._handles.get(shard_filename)
        if handle is None:
            full_path = str(self.hdf5_dir / shard_filename)
            handle = h5py.File(full_path, "r")
            self._handles[shard_filename] = handle
        return handle

    def close(self) -> None:
        for h in self._handles.values():
            try:
                h.close()
            except Exception:
                pass
        self._handles.clear()

    def __del__(self) -> None:
        self.close()

    def __len__(self) -> int:
        return int(len(self.meta))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.meta.iloc[int(idx)]
        shard_filename = str(row[HDF5_FILE_COL])
        local_index = int(row[LOCAL_INDEX_COL])

        global_index = int(self._indices[int(idx)])
        if self.cache_histograms and global_index in self._hist_cache:
            hist = self._hist_cache[global_index]
        else:
            handle = self._open_shard(shard_filename)
            hist = np.asarray(handle["data/histograms"][local_index], dtype=np.float32)
            if hist.shape != self.grid.simplex_mask.shape:
                raise RuntimeError(
                    f"Histogram at {shard_filename}[{local_index}] has shape "
                    f"{hist.shape}, expected {self.grid.simplex_mask.shape}"
                )
            if not np.isfinite(hist).all() or np.any(hist < 0.0):
                raise RuntimeError(
                    f"Histogram at {shard_filename}[{local_index}] contains "
                    "non-finite or negative values"
                )
            s = float(hist.sum())
            if abs(s - 1.0) > self.sum_atol:
                raise RuntimeError(
                    f"Histogram at {shard_filename}[{local_index}] sums to {s}, "
                    f"expected 1.0 ± {self.sum_atol}"
                )
            if self.flat_histograms:
                hist = hist.reshape(-1)[self._flat_simplex_index]
                inside_sum = float(hist.sum())
                if inside_sum <= 0.0:
                    raise RuntimeError(
                        f"Histogram at {shard_filename}[{local_index}] has no "
                        "mass in the physical simplex"
                    )
                hist = hist / inside_sum
            if self.cache_histograms:
                self._hist_cache[global_index] = hist

        if self.input_moments == 4:
            moments = np.array(
                [row["mean_a"], row["var_a"], row["mean_b"], row["var_b"]],
                dtype=np.float32,
            )
        else:
            moments = np.array(
                [row["mean_a"], row["var_a"], row["mean_b"], row["var_b"], row["cov_ab"]],
                dtype=np.float32,
            )

        meta = {
            "scalar_config": str(row[SCALAR_CONFIG_COL]),
            "run_id": int(row["run_id"]),
            "timestep": int(row[TIMESTEP_COL]),
            "box_width": int(row["box_width"]),
            "sample_id": int(row[SAMPLE_ID_COL]),
        }
        return {
            "moments": torch.from_numpy(moments),
            "histogram": torch.from_numpy(hist),
            "meta": meta,
        }

    @staticmethod
    def fit_input_scaler(meta: pd.DataFrame, input_moments: int) -> RobustScaler:
        if input_moments == 4:
            cols = ["mean_a", "var_a", "mean_b", "var_b"]
        else:
            cols = ["mean_a", "var_a", "mean_b", "var_b", "cov_ab"]
        X = meta[cols].to_numpy(dtype=np.float64)
        scaler = RobustScaler().fit(X)
        return scaler

    def get_moments_matrix(self) -> np.ndarray:
        if self.input_moments == 4:
            cols = ["mean_a", "var_a", "mean_b", "var_b"]
        else:
            cols = ["mean_a", "var_a", "mean_b", "var_b", "cov_ab"]
        return self.meta[cols].to_numpy(dtype=np.float32)


def collate_records(batch: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    moments = torch.stack([b["moments"] for b in batch], dim=0)
    histograms = torch.stack([b["histogram"] for b in batch], dim=0)
    meta = {
        "scalar_config": [b["meta"]["scalar_config"] for b in batch],
        "run_id": torch.tensor([b["meta"]["run_id"] for b in batch], dtype=torch.long),
        "timestep": torch.tensor([b["meta"]["timestep"] for b in batch], dtype=torch.long),
        "box_width": torch.tensor([b["meta"]["box_width"] for b in batch], dtype=torch.long),
        "sample_id": torch.tensor([b["meta"]["sample_id"] for b in batch], dtype=torch.long),
    }
    return {"moments": moments, "histogram": histograms, "meta": meta}


def validate_dataset(
    parquet_path: str,
    hdf5_dir: str,
    *,
    num_bins: int = 64,
    zst: float = 0.1,
    uniform_bins: bool = False,
    sum_atol: float = 1e-4,
    outside_mass_atol: float = 1e-7,
    max_moment_discrepancy: float = 0.05,
) -> dict[str, Any]:
    """Exhaustively validate a Parquet/HDF5 dataset before training."""
    from .bin_grid import validate_against_hdf5

    if max_moment_discrepancy <= 0.0:
        raise ValueError("max_moment_discrepancy must be positive")

    meta = load_metadata(parquet_path)
    grid = bin_grid(num_bins=num_bins, zst=zst, uniform=uniform_bins)
    root = Path(hdf5_dir)
    checked_rows = 0
    worst_sum_error = 0.0
    worst_outside_mass = 0.0
    worst_moment_discrepancy = 0.0
    worst_stored_moment_mismatch = 0.0
    histogram_moment_columns = [
        "hist_mean_a", "hist_var_a", "hist_mean_b", "hist_var_b", "hist_cov_ab",
    ]
    missing_hist_moments = [
        column for column in histogram_moment_columns
        if column not in meta.columns
    ]
    if missing_hist_moments or "moment_abs_err_max" not in meta.columns:
        raise KeyError(
            "format-version-2 metadata lacks required histogram moment fields: "
            f"{missing_hist_moments}"
        )
    derived_columns = [*histogram_moment_columns, "moment_abs_err_max"]
    derived_values = meta[derived_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(derived_values).all():
        raise ValueError("histogram moment metadata must be finite")
    if np.any(meta["moment_abs_err_max"].to_numpy(dtype=np.float64) < 0.0):
        raise ValueError("moment_abs_err_max must be non-negative")

    for shard_name, shard_meta in meta.groupby(HDF5_FILE_COL, sort=True):
        shard_path = root / str(shard_name)
        if not shard_path.is_file():
            raise FileNotFoundError(f"missing HDF5 shard: {shard_path}")
        grid_ok, grid_diffs = validate_against_hdf5(
            str(shard_path),
            num_bins=num_bins,
            zst=zst,
            uniform=uniform_bins,
        )
        if not grid_ok:
            raise RuntimeError(
                f"grid mismatch in {shard_path}: {grid_diffs}. "
                "Legacy center-masked datasets must be regenerated."
            )

        with h5py.File(shard_path, "r") as handle:
            if int(handle.attrs.get("format_version", -1)) != DATASET_FORMAT_VERSION:
                raise ValueError(
                    f"{shard_path} is not dataset format version 2; regenerate it"
                )
            if "data/histograms" not in handle:
                raise KeyError(f"{shard_path} lacks data/histograms")
            histograms = handle["data/histograms"]
            if histograms.ndim != 3 or histograms.shape[1:] != (num_bins, num_bins):
                raise ValueError(
                    f"{shard_path} histogram shape is {histograms.shape}, "
                    f"expected (records, {num_bins}, {num_bins})"
                )

            local_indices = shard_meta[LOCAL_INDEX_COL].to_numpy(dtype=np.int64)
            if len(np.unique(local_indices)) != len(local_indices):
                raise ValueError(f"{shard_path} has duplicate local_index mappings")
            expected = np.arange(histograms.shape[0], dtype=np.int64)
            if not np.array_equal(np.sort(local_indices), expected):
                raise ValueError(
                    f"{shard_path} metadata does not cover every HDF5 row exactly once"
                )

            ordered = shard_meta.sort_values(LOCAL_INDEX_COL)
            if "data/sample_id" not in handle or "data/local_index" not in handle:
                raise KeyError(
                    f"{shard_path} lacks required format-version-2 row identity arrays"
                )
            stored_ids = np.asarray(handle["data/sample_id"][:], dtype=np.int64)
            stored_local = np.asarray(handle["data/local_index"][:], dtype=np.int64)
            metadata_ids = ordered[SAMPLE_ID_COL].to_numpy(dtype=np.int64)
            if (
                stored_ids.shape != expected.shape
                or stored_local.shape != expected.shape
                or not np.array_equal(stored_ids, metadata_ids)
                or not np.array_equal(stored_local, expected)
            ):
                raise ValueError(
                    f"{shard_path} row identity arrays disagree with Parquet metadata"
                )

            chunk_rows = histograms.chunks[0] if histograms.chunks else 256
            for start in range(0, histograms.shape[0], chunk_rows):
                chunk = np.asarray(
                    histograms[start:start + chunk_rows], dtype=np.float64,
                )
                if not np.isfinite(chunk).all() or np.any(chunk < 0.0):
                    raise ValueError(
                        f"{shard_path} contains non-finite or negative histogram mass"
                    )
                sums = chunk.sum(axis=(1, 2))
                if np.any(sums <= 0.0):
                    raise ValueError(f"{shard_path} contains an empty histogram")
                worst_sum_error = max(
                    worst_sum_error, float(np.max(np.abs(sums - 1.0))),
                )
                outside = chunk[:, ~grid.simplex_mask].sum(axis=1)
                worst_outside_mass = max(
                    worst_outside_mass, float(np.max(outside, initial=0.0)),
                )
                probability = chunk / sums[:, None, None]
                mean_a = np.sum(
                    probability * grid.cell_centroid_a[None, :, :],
                    axis=(1, 2),
                )
                mean_b = np.sum(
                    probability * grid.cell_centroid_b[None, :, :],
                    axis=(1, 2),
                )
                var_a = np.sum(
                    probability
                    * (grid.cell_centroid_a[None, :, :] - mean_a[:, None, None]) ** 2,
                    axis=(1, 2),
                )
                var_b = np.sum(
                    probability
                    * (grid.cell_centroid_b[None, :, :] - mean_b[:, None, None]) ** 2,
                    axis=(1, 2),
                )
                cov_ab = np.sum(
                    probability
                    * (
                        grid.cell_centroid_a[None, :, :]
                        - mean_a[:, None, None]
                    )
                    * (
                        grid.cell_centroid_b[None, :, :]
                        - mean_b[:, None, None]
                    ),
                    axis=(1, 2),
                )
                actual_moments = np.column_stack(
                    [mean_a, var_a, mean_b, var_b, cov_ab],
                )
                metadata_chunk = ordered.iloc[start:start + len(chunk)]
                direct_moments = metadata_chunk[MOMENT_COLS].to_numpy(
                    dtype=np.float64,
                )
                stored_hist_moments = metadata_chunk[
                    histogram_moment_columns
                ].to_numpy(dtype=np.float64)
                direct_discrepancy = np.abs(actual_moments - direct_moments)
                stored_mismatch = np.abs(actual_moments - stored_hist_moments)
                worst_moment_discrepancy = max(
                    worst_moment_discrepancy,
                    float(direct_discrepancy.max(initial=0.0)),
                )
                worst_stored_moment_mismatch = max(
                    worst_stored_moment_mismatch,
                    float(stored_mismatch.max(initial=0.0)),
                )
                recorded_max = metadata_chunk[
                    "moment_abs_err_max"
                ].to_numpy(dtype=np.float64)
                actual_max = direct_discrepancy.max(axis=1)
                worst_stored_moment_mismatch = max(
                    worst_stored_moment_mismatch,
                    float(np.max(np.abs(recorded_max - actual_max), initial=0.0)),
                )
                checked_rows += len(chunk)

    if checked_rows != len(meta):
        raise ValueError(
            f"validated {checked_rows} HDF5 rows for {len(meta)} metadata records"
        )
    if worst_sum_error > sum_atol:
        raise ValueError(
            f"histogram normalization error {worst_sum_error:.6g} exceeds {sum_atol}"
        )
    if worst_outside_mass > outside_mass_atol:
        raise ValueError(
            f"histogram mass outside simplex {worst_outside_mass:.6g} "
            f"exceeds {outside_mass_atol}"
        )

    if worst_stored_moment_mismatch > 1e-6:
        raise ValueError(
            "stored histogram moments do not match the actual HDF5 histograms: "
            f"worst mismatch={worst_stored_moment_mismatch:.6g}"
        )
    if worst_moment_discrepancy > max_moment_discrepancy:
        raise ValueError(
            f"direct/histogram moment discrepancy {worst_moment_discrepancy:.6g} "
            f"exceeds {max_moment_discrepancy}"
        )

    return {
        "dataset_fingerprint": metadata_fingerprint(meta),
        "records_checked": checked_rows,
        "shards_checked": int(meta[HDF5_FILE_COL].nunique()),
        "worst_histogram_sum_error": worst_sum_error,
        "worst_outside_simplex_mass": worst_outside_mass,
        "worst_moment_discrepancy": worst_moment_discrepancy,
        "worst_stored_moment_mismatch": worst_stored_moment_mismatch,
    }


__all__ = [
    "HybridPDFDataset",
    "collate_records",
    "load_metadata",
    "metadata_fingerprint",
    "validate_dataset",
    "MOMENT_COLS",
]
