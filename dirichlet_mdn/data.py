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

    return renamed.reset_index(drop=True)


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
    ) -> None:
        if input_moments not in (4, 5):
            raise ValueError(f"input_moments must be 4 or 5, got {input_moments}")
        self.parquet_path = parquet_path
        self.hdf5_dir = Path(hdf5_dir)
        self.input_moments = input_moments
        self.sum_atol = sum_atol
        self.cache_histograms = cache_histograms
        self.flat_histograms = flat_histograms

        self._meta_all = load_metadata(parquet_path)
        if indices is None:
            self._indices = np.arange(len(self._meta_all), dtype=np.int64)
        else:
            self._indices = np.asarray(list(indices), dtype=np.int64)
        self.meta = self._meta_all.iloc[self._indices].reset_index(drop=True)

        self.grid: BinGrid = bin_grid(num_bins=num_bins, zst=zst, uniform=uniform_bins)
        if int(self._meta_all.shape[0]) > 0:
            self._check_grid_matches_first_shard()

        # Precompute the flat in-simplex index (used when flat_histograms=True
        # or by callers that want to mask down to in-simplex cells).
        self._flat_simplex_index = np.flatnonzero(self.grid.simplex_mask.reshape(-1))

        self._handles: Dict[str, h5py.File] = {}
        self._sum_verified_shards: set[str] = set()
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
            if shard_filename not in self._sum_verified_shards:
                s = float(hist.sum())
                if abs(s - 1.0) > self.sum_atol:
                    raise RuntimeError(
                        f"Histogram at {shard_filename}[{local_index}] sums to {s}, "
                        f"expected 1.0 ± {self.sum_atol}"
                    )
                self._sum_verified_shards.add(shard_filename)
            if self.flat_histograms:
                hist = hist.reshape(-1)[self._flat_simplex_index]
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


__all__ = [
    "HybridPDFDataset",
    "collate_records",
    "load_metadata",
    "MOMENT_COLS",
]
