"""Reconstruct the non-uniform Z1-Z2 bin grid used by EnsightPDFHybridDataset.

Mirrors EnsightPDFHybridDataset.py:123-186 so training code does not depend
on the writer script. ``validate_against_hdf5`` confirms agreement to 1e-9
against any shard produced by that writer.

The simplex convention follows ``indexing="ij"``: axis 0 is Z1 (A scalar),
axis 1 is Z2 (B scalar). ``simplex_mask[i, j]`` is True when
``centers_a[i] + centers_b[j] <= 1 + 1e-12``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import h5py
import numpy as np


@dataclass(frozen=True)
class BinGrid:
    num_bins: int
    zst: float
    uniform: bool
    centers_a: np.ndarray       # (N,)
    centers_b: np.ndarray       # (N,)
    edges_a: np.ndarray         # (N+1,)
    edges_b: np.ndarray         # (N+1,)
    cell_area: np.ndarray       # (N, N)
    simplex_mask: np.ndarray    # (N, N) bool

    @property
    def num_simplex_cells(self) -> int:
        return int(self.simplex_mask.sum())


def _bin_centers_1d(num_bins: int, zst: float, uniform: bool) -> np.ndarray:
    """Reproduce ``get_bin_centers`` from EnsightPDFHybridDataset.py:123-148."""
    if uniform:
        return np.linspace(0.0, 1.0, num_bins, dtype=np.float64)

    centers = np.zeros(num_bins, dtype=np.float64)
    zcut = int(num_bins / 2)
    dz = zst / float(zcut - 1)

    for idx in range(0, zcut):
        centers[idx] = float(idx) * dz

    m11 = float((num_bins - 1) ** 2 - (zcut - 1) ** 2)
    m12 = float(num_bins - zcut)
    m21 = float(2 * (zcut - 1) + 1)
    m22 = 1.0
    r1 = 1.0 - zst
    r2 = dz
    delta = m11 * m22 - m12 * m21
    coef_a = (+m22 * r1 - m12 * r2) / delta
    coef_b = (-m21 * r1 + m11 * r2) / delta
    coef_c = zst - coef_a * (zcut - 1) ** 2 - coef_b * (zcut - 1)
    for idx in range(zcut, num_bins):
        centers[idx] = coef_a * float(idx) ** 2 + coef_b * float(idx) + coef_c
    return centers


def _edges_from_centers(centers: np.ndarray) -> np.ndarray:
    n = len(centers)
    edges = np.zeros(n + 1, dtype=np.float64)
    edges[0] = centers[0] - (0.5 * centers[1] + 0.5 * centers[0] - centers[0])
    edges[-1] = centers[-1] + (centers[-1] - 0.5 * centers[-1] - 0.5 * centers[-2])
    for idx in range(1, n):
        edges[idx] = 0.5 * centers[idx] + 0.5 * centers[idx - 1]
    return edges


def bin_grid(num_bins: int = 64, zst: float = 0.1, uniform: bool = False) -> BinGrid:
    """Construct the full 2-D bin grid used by the hybrid dataset writer.

    Defaults match the writer's manifest defaults in this repo
    (``pdf_bins=64``, ``zst=0.1``, non-uniform).
    """
    centers_a = _bin_centers_1d(num_bins, zst, uniform)
    centers_b = _bin_centers_1d(num_bins, zst, uniform)
    edges_a = _edges_from_centers(centers_a)
    edges_b = _edges_from_centers(centers_b)

    cell_area = np.zeros((num_bins, num_bins), dtype=np.float64)
    for i in range(num_bins):
        for j in range(num_bins):
            cell_area[i, j] = (edges_a[i + 1] - edges_a[i]) * (edges_b[j + 1] - edges_b[j])

    grid_a, grid_b = np.meshgrid(centers_a, centers_b, indexing="ij")
    simplex_mask = (grid_a + grid_b) <= 1.0 + 1.0e-12

    return BinGrid(
        num_bins=num_bins,
        zst=zst,
        uniform=uniform,
        centers_a=centers_a,
        centers_b=centers_b,
        edges_a=edges_a,
        edges_b=edges_b,
        cell_area=cell_area,
        simplex_mask=simplex_mask,
    )


def validate_against_hdf5(
    h5_path: str,
    *,
    num_bins: int = 64,
    zst: float = 0.1,
    uniform: bool = False,
    atol: float = 1e-9,
) -> Tuple[bool, dict]:
    """Compare a reconstructed BinGrid against arrays stored under ``/bins`` in
    an HDF5 shard. Returns ``(ok, diffs)`` where ``diffs`` reports the max
    absolute difference for each field.
    """
    grid = bin_grid(num_bins=num_bins, zst=zst, uniform=uniform)
    diffs: dict = {}
    with h5py.File(h5_path, "r") as f:
        ref_centers_a = f["bins/centers_a"][:]
        ref_centers_b = f["bins/centers_b"][:]
        ref_edges_a = f["bins/edges_a"][:]
        ref_edges_b = f["bins/edges_b"][:]
        ref_cell_area = f["bins/cell_area"][:]
        ref_mask = f["bins/simplex_mask"][:].astype(bool)

    diffs["centers_a"] = float(np.abs(grid.centers_a - ref_centers_a).max())
    diffs["centers_b"] = float(np.abs(grid.centers_b - ref_centers_b).max())
    diffs["edges_a"] = float(np.abs(grid.edges_a - ref_edges_a).max())
    diffs["edges_b"] = float(np.abs(grid.edges_b - ref_edges_b).max())
    diffs["cell_area"] = float(np.abs(grid.cell_area - ref_cell_area).max())
    diffs["simplex_mask"] = int(np.sum(grid.simplex_mask != ref_mask))

    ok = (
        diffs["centers_a"] <= atol
        and diffs["centers_b"] <= atol
        and diffs["edges_a"] <= atol
        and diffs["edges_b"] <= atol
        and diffs["cell_area"] <= atol
        and diffs["simplex_mask"] == 0
    )
    return ok, diffs


__all__ = ["BinGrid", "bin_grid", "validate_against_hdf5"]
