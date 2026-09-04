"""Construct the non-uniform Z1-Z2 grid used by the dataset writers.

Rectangular histogram bins along the boundary are clipped to the physical
simplex.  ``cell_area`` is therefore the area of ``rectangle ∩ simplex``, and
``cell_centroid_*`` is the centroid of that clipped polygon.  This prevents
valid boundary mass from being discarded merely because a rectangular bin's
nominal centre lies just outside the simplex.

The simplex convention follows ``indexing="ij"``: axis 0 is Z1 (A scalar),
axis 1 is Z2 (B scalar). ``simplex_mask[i, j]`` is true whenever the clipped
cell has positive area.
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
    cell_area: np.ndarray       # (N, N), rectangle clipped to simplex
    cell_centroid_a: np.ndarray # (N, N)
    cell_centroid_b: np.ndarray # (N, N)
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


def _clip_polygon(
    polygon: list[tuple[float, float]],
    signed_distance,
) -> list[tuple[float, float]]:
    """Clip a convex polygon to ``signed_distance(point) >= 0``."""
    if not polygon:
        return []
    result: list[tuple[float, float]] = []
    previous = polygon[-1]
    previous_distance = float(signed_distance(previous))
    for current in polygon:
        current_distance = float(signed_distance(current))
        previous_inside = previous_distance >= 0.0
        current_inside = current_distance >= 0.0
        if previous_inside != current_inside:
            fraction = previous_distance / (previous_distance - current_distance)
            result.append((
                previous[0] + fraction * (current[0] - previous[0]),
                previous[1] + fraction * (current[1] - previous[1]),
            ))
        if current_inside:
            result.append(current)
        previous = current
        previous_distance = current_distance
    return result


def _simplex_cell_geometry(
    x0: float,
    x1: float,
    y0: float,
    y1: float,
) -> tuple[float, float, float]:
    """Return area and centroid of a rectangle intersected with the simplex."""
    polygon = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    for boundary in (
        lambda p: p[0],
        lambda p: p[1],
        lambda p: 1.0 - p[0] - p[1],
    ):
        polygon = _clip_polygon(polygon, boundary)
    if len(polygon) < 3:
        return 0.0, 0.0, 0.0

    twice_area = 0.0
    centroid_x_numerator = 0.0
    centroid_y_numerator = 0.0
    for index, (x_a, y_a) in enumerate(polygon):
        x_b, y_b = polygon[(index + 1) % len(polygon)]
        cross = x_a * y_b - x_b * y_a
        twice_area += cross
        centroid_x_numerator += (x_a + x_b) * cross
        centroid_y_numerator += (y_a + y_b) * cross

    area = 0.5 * twice_area
    if area <= np.finfo(np.float64).eps:
        return 0.0, 0.0, 0.0
    centroid_x = centroid_x_numerator / (6.0 * area)
    centroid_y = centroid_y_numerator / (6.0 * area)
    return area, centroid_x, centroid_y


def bin_grid(num_bins: int = 64, zst: float = 0.1, uniform: bool = False) -> BinGrid:
    """Construct the full 2-D bin grid used by the hybrid dataset writer.

    Defaults match the writer's manifest defaults in this repo
    (``pdf_bins=64``, ``zst=0.1``, non-uniform).
    """
    if num_bins < 2 or (not uniform and num_bins < 4):
        minimum = 2 if uniform else 4
        raise ValueError(f"num_bins must be >= {minimum} for this grid")
    if not 0.0 < zst < 1.0:
        raise ValueError(f"zst must lie strictly between 0 and 1, got {zst}")

    centers_a = _bin_centers_1d(num_bins, zst, uniform)
    centers_b = _bin_centers_1d(num_bins, zst, uniform)
    edges_a = _edges_from_centers(centers_a)
    edges_b = _edges_from_centers(centers_b)
    if not np.all(np.diff(centers_a) > 0.0) or not np.all(np.diff(edges_a) > 0.0):
        raise ValueError("constructed histogram grid is not strictly increasing")

    cell_area = np.zeros((num_bins, num_bins), dtype=np.float64)
    cell_centroid_a = np.zeros((num_bins, num_bins), dtype=np.float64)
    cell_centroid_b = np.zeros((num_bins, num_bins), dtype=np.float64)
    for i in range(num_bins):
        for j in range(num_bins):
            area, centroid_a, centroid_b = _simplex_cell_geometry(
                edges_a[i], edges_a[i + 1], edges_b[j], edges_b[j + 1],
            )
            cell_area[i, j] = area
            cell_centroid_a[i, j] = centroid_a
            cell_centroid_b[i, j] = centroid_b

    simplex_mask = cell_area > 0.0
    if not np.isclose(cell_area.sum(), 0.5, rtol=0.0, atol=1e-12):
        raise RuntimeError(
            f"clipped grid area must equal simplex area 0.5, got {cell_area.sum()}"
        )

    return BinGrid(
        num_bins=num_bins,
        zst=zst,
        uniform=uniform,
        centers_a=centers_a,
        centers_b=centers_b,
        edges_a=edges_a,
        edges_b=edges_b,
        cell_area=cell_area,
        cell_centroid_a=cell_centroid_a,
        cell_centroid_b=cell_centroid_b,
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
        ref_centroid_a = (
            f["bins/cell_centroid_a"][:]
            if "bins/cell_centroid_a" in f else None
        )
        ref_centroid_b = (
            f["bins/cell_centroid_b"][:]
            if "bins/cell_centroid_b" in f else None
        )
        ref_mask = f["bins/simplex_mask"][:].astype(bool)

    diffs["centers_a"] = float(np.abs(grid.centers_a - ref_centers_a).max())
    diffs["centers_b"] = float(np.abs(grid.centers_b - ref_centers_b).max())
    diffs["edges_a"] = float(np.abs(grid.edges_a - ref_edges_a).max())
    diffs["edges_b"] = float(np.abs(grid.edges_b - ref_edges_b).max())
    diffs["cell_area"] = float(np.abs(grid.cell_area - ref_cell_area).max())
    diffs["cell_centroid_a"] = (
        float(np.abs(grid.cell_centroid_a - ref_centroid_a).max())
        if ref_centroid_a is not None else float("inf")
    )
    diffs["cell_centroid_b"] = (
        float(np.abs(grid.cell_centroid_b - ref_centroid_b).max())
        if ref_centroid_b is not None else float("inf")
    )
    diffs["simplex_mask"] = int(np.sum(grid.simplex_mask != ref_mask))

    ok = (
        diffs["centers_a"] <= atol
        and diffs["centers_b"] <= atol
        and diffs["edges_a"] <= atol
        and diffs["edges_b"] <= atol
        and diffs["cell_area"] <= atol
        and diffs["cell_centroid_a"] <= atol
        and diffs["cell_centroid_b"] <= atol
        and diffs["simplex_mask"] == 0
    )
    return ok, diffs


__all__ = ["BinGrid", "bin_grid", "validate_against_hdf5"]
