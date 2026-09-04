#!/usr/bin/env python3
"""
MPI-parallel driver for the hybrid LES-PDF dataset builder.

How to run (52 ranks, ct-env):
    conda activate ct-env
    mpirun -n 52 python EnsightPDFHybridDatasetMPI.py \
        -f /path/to/runs --run-ids 0:27 \
        --scalar-configs I1,I4,I5,L1,PI1,PI4,PI5,PL1 \
        --box-widths 64,128 --strides 32,64 \
        --output-dir hybrid_dataset_mpi

Phase 1: every rank losslessly processes its slice of (run_id, scalar_config)
work units and writes per-rank parquet+HDF5 shards under
output_dir/_rank/rank_NNNN/.

Phase 2: rank 0 streams the per-rank candidates back through the same
decide_retention logic with the original per-bin cap and produces the final
merged dataset. Root memory is bounded by the retained-set size (identical to
the serial baseline), not by the total per-rank candidate volume.

The serial entry point delegates to this implementation with one MPI rank.
"""

import argparse
import glob
import hashlib
import heapq
import json
import os
import re
import shutil
import sys
import tempfile
import time
import uuid
from collections import Counter, OrderedDict, defaultdict
from datetime import timedelta

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dirichlet_mdn.bin_grid import bin_grid
from dirichlet_mdn.data import validate_dataset


SCALAR_CONFIGS = ["I1", "I4", "I5", "L1", "PI1", "PI4", "PI5", "PL1"]
DATASET_FORMAT_VERSION = 2
MOMENT_NAMES = ["mean_a", "var_a", "mean_b", "var_b", "cov_ab"]
SHAPE_NAMES = [
    "shape_entropy",
    "shape_max_prob",
    "shape_effective_support",
    "shape_corner_z1",
    "shape_corner_z2",
    "shape_corner_z3",
    "shape_edge_z1_zero",
    "shape_edge_z2_zero",
    "shape_edge_z3_zero",
    "shape_center_mass",
]


def parse_int_spec(text):
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            pieces = [item.strip() for item in part.split(":")]
            if len(pieces) not in (2, 3):
                raise ValueError("Range specifications must be start:end or start:end:step")
            start = int(pieces[0])
            end = int(pieces[1])
            step = int(pieces[2]) if len(pieces) == 3 else 1
            if step == 0:
                raise ValueError("Range step cannot be zero")
            values.extend(list(range(start, end, step)))
        else:
            values.append(int(part))
    if not values:
        raise ValueError("Expected at least one integer value")
    return values


def parse_str_list(text):
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one string value")
    return values


def fill_periodic(orig, nx, width):
    mod = np.zeros((nx + 2 * width, nx + 2 * width, nx + 2 * width), dtype=orig.dtype)

    mod[width:nx + width, width:nx + width, width:nx + width] = orig[:, :, :]

    mod[0:width, width:nx + width, width:nx + width] = orig[nx - width:nx, :, :]
    mod[(nx + width):(nx + 2 * width), width:nx + width, width:nx + width] = orig[0:width, :, :]

    mod[:, 0:width, :] = mod[:, nx:(nx + width), :]
    mod[:, (nx + width):(nx + 2 * width), :] = mod[:, width:2 * width, :]

    mod[:, :, 0:width] = mod[:, :, nx:(nx + width)]
    mod[:, :, (nx + width):(nx + 2 * width)] = mod[:, :, width:2 * width]

    return mod


def read_ensight(file_scalar_a, file_scalar_b, nx, pad_width=32, npx=8):
    nx_local = nx // npx

    with open(file_scalar_a, "rb") as handle:
        handle.read(80)
        handle.read(80)
        np.fromfile(handle, dtype=np.int32, count=1)
        handle.read(80)
        scalar_a = np.fromfile(handle, dtype=np.float32, count=-1)
    expected_values = nx ** 3
    if scalar_a.size != expected_values:
        raise ValueError(
            f"{file_scalar_a} contains {scalar_a.size} scalar values; "
            f"expected {expected_values}"
        )

    scalar_a = np.reshape(scalar_a, (nx_local, nx_local, nx_local, npx, npx, npx), order="F")
    scalar_a_reshaped = np.zeros((nx, nx, nx), dtype=np.float32)
    for ip in range(npx):
        for jp in range(npx):
            for kp in range(npx):
                scalar_a_reshaped[
                    ip * nx_local:(ip + 1) * nx_local,
                    jp * nx_local:(jp + 1) * nx_local,
                    kp * nx_local:(kp + 1) * nx_local,
                ] = scalar_a[:, :, :, kp, jp, ip]

    with open(file_scalar_b, "rb") as handle:
        handle.read(80)
        handle.read(80)
        np.fromfile(handle, dtype=np.int32, count=1)
        handle.read(80)
        scalar_b = np.fromfile(handle, dtype=np.float32, count=-1)
    if scalar_b.size != expected_values:
        raise ValueError(
            f"{file_scalar_b} contains {scalar_b.size} scalar values; "
            f"expected {expected_values}"
        )

    scalar_b = np.reshape(scalar_b, (nx_local, nx_local, nx_local, npx, npx, npx), order="F")
    scalar_b_reshaped = np.zeros((nx, nx, nx), dtype=np.float32)
    for ip in range(npx):
        for jp in range(npx):
            for kp in range(npx):
                scalar_b_reshaped[
                    ip * nx_local:(ip + 1) * nx_local,
                    jp * nx_local:(jp + 1) * nx_local,
                    kp * nx_local:(kp + 1) * nx_local,
                ] = scalar_b[:, :, :, kp, jp, ip]

    full_scalar_a = fill_periodic(scalar_a_reshaped, nx, pad_width)
    full_scalar_b = fill_periodic(scalar_b_reshaped, nx, pad_width)

    return full_scalar_a, full_scalar_b


def get_histogram_layout(num_bins, zst=0.1, non_uniform=True):
    grid = bin_grid(num_bins=num_bins, zst=zst, uniform=not non_uniform)

    return {
        "edges_a": grid.edges_a,
        "edges_b": grid.edges_b,
        "centers_a": grid.centers_a,
        "centers_b": grid.centers_b,
        "grid_a": grid.cell_centroid_a,
        "grid_b": grid.cell_centroid_b,
        "z1z2": grid.cell_centroid_a * grid.cell_centroid_b,
        "cell_area": grid.cell_area,
        "simplex_mask": grid.simplex_mask,
    }


def compute_direct_moments(z1_values, z2_values):
    mean_a = np.mean(z1_values)
    mean_b = np.mean(z2_values)
    var_a = np.mean((z1_values - mean_a) ** 2)
    var_b = np.mean((z2_values - mean_b) ** 2)
    cov_ab = np.mean((z1_values - mean_a) * (z2_values - mean_b))
    return mean_a, var_a, mean_b, var_b, cov_ab


def compute_histogram_pdf(z1_values, z2_values, layout):
    counts, _, _ = np.histogram2d(
        z1_values,
        z2_values,
        bins=[layout["edges_a"], layout["edges_b"]],
        density=False,
    )
    total = counts.sum()
    if total <= 0.0:
        raise ValueError("Histogram count total must be positive")
    if int(total) != int(z1_values.size):
        raise ValueError(
            f"histogram dropped {int(z1_values.size - total)} validated scalar pairs"
        )
    outside_count = int(counts[~layout["simplex_mask"]].sum())
    if outside_count:
        raise ValueError(
            f"histogram assigned {outside_count} validated scalar pairs to "
            "zero-area cells outside the simplex"
        )
    return counts / total


def prepare_scalar_samples(z1_values, z2_values, tolerance=1e-6, policy="error"):
    """Validate and tolerance-correct one filter cell before using its values."""
    if tolerance < 0.0:
        raise ValueError("scalar tolerance must be non-negative")
    if policy not in ("error", "drop"):
        raise ValueError("invalid scalar policy must be 'error' or 'drop'")

    z1 = np.asarray(z1_values, dtype=np.float64).reshape(-1)
    z2 = np.asarray(z2_values, dtype=np.float64).reshape(-1)
    if z1.shape != z2.shape:
        raise ValueError("paired scalar arrays must have the same shape")
    finite = np.isfinite(z1) & np.isfinite(z2)
    physical = (
        (z1 >= -tolerance)
        & (z1 <= 1.0 + tolerance)
        & (z2 >= -tolerance)
        & (z2 <= 1.0 + tolerance)
        & ((z1 + z2) <= 1.0 + tolerance)
    )
    valid = finite & physical
    invalid_count = int((~valid).sum())
    if invalid_count and policy == "error":
        raise ValueError(
            f"filter cell contains {invalid_count}/{z1.size} non-finite or "
            "non-physical scalar pairs"
        )
    if policy == "drop":
        z1 = z1[valid]
        z2 = z2[valid]
    if z1.size == 0:
        raise ValueError("filter cell contains no valid scalar pairs")

    original_z1 = z1.copy()
    original_z2 = z2.copy()
    z1 = np.clip(z1, 0.0, 1.0)
    z2 = np.clip(z2, 0.0, 1.0)
    pair_sum = z1 + z2
    over = pair_sum > 1.0
    z1[over] /= pair_sum[over]
    z2[over] /= pair_sum[over]
    adjusted_count = int(
        np.count_nonzero((z1 != original_z1) | (z2 != original_z2))
    )
    return z1, z2, {
        "scalar_pairs_raw": int(np.asarray(z1_values).size),
        "scalar_pairs_invalid": invalid_count,
        "scalar_pairs_adjusted": adjusted_count,
        "scalar_pairs_used": int(z1.size),
    }


def compute_histogram_moments(pdf, layout):
    mean_a = np.sum(pdf * layout["grid_a"])
    mean_b = np.sum(pdf * layout["grid_b"])
    var_a = np.sum(pdf * ((layout["grid_a"] - mean_a) ** 2))
    var_b = np.sum(pdf * ((layout["grid_b"] - mean_b) ** 2))
    cov_ab = np.sum(pdf * (layout["z1z2"] - mean_a * mean_b))
    return mean_a, var_a, mean_b, var_b, cov_ab


def build_shape_masks(layout, corner_threshold, edge_threshold, center_radius):
    grid_a = layout["grid_a"]
    grid_b = layout["grid_b"]
    simplex_mask = layout["simplex_mask"]
    z3 = 1.0 - grid_a - grid_b
    center_distance = np.sqrt((grid_a - 1.0 / 3.0) ** 2 + (grid_b - 1.0 / 3.0) ** 2)

    return {
        "corner_z1": simplex_mask & (grid_a >= 1.0 - corner_threshold) & (grid_b <= corner_threshold),
        "corner_z2": simplex_mask & (grid_b >= 1.0 - corner_threshold) & (grid_a <= corner_threshold),
        "corner_z3": simplex_mask & (grid_a <= corner_threshold) & (grid_b <= corner_threshold),
        "edge_z1_zero": simplex_mask & (grid_a <= edge_threshold),
        "edge_z2_zero": simplex_mask & (grid_b <= edge_threshold),
        "edge_z3_zero": simplex_mask & (z3 <= edge_threshold),
        "center_mass": simplex_mask & (center_distance <= center_radius),
    }


def compute_shape_features(pdf, shape_masks):
    flat_pdf = np.ravel(pdf)
    safe_pdf = np.clip(flat_pdf, 1.0e-16, None)
    entropy = -np.sum(flat_pdf * np.log(safe_pdf)) / np.log(float(flat_pdf.size))
    effective_support = 1.0 / np.sum(flat_pdf ** 2)
    effective_support /= float(flat_pdf.size)

    features = {
        "shape_entropy": float(entropy),
        "shape_max_prob": float(np.max(flat_pdf)),
        "shape_effective_support": float(effective_support),
        "shape_corner_z1": float(np.sum(pdf[shape_masks["corner_z1"]])),
        "shape_corner_z2": float(np.sum(pdf[shape_masks["corner_z2"]])),
        "shape_corner_z3": float(np.sum(pdf[shape_masks["corner_z3"]])),
        "shape_edge_z1_zero": float(np.sum(pdf[shape_masks["edge_z1_zero"]])),
        "shape_edge_z2_zero": float(np.sum(pdf[shape_masks["edge_z2_zero"]])),
        "shape_edge_z3_zero": float(np.sum(pdf[shape_masks["edge_z3_zero"]])),
        "shape_center_mass": float(np.sum(pdf[shape_masks["center_mass"]])),
    }
    return features


def jensen_shannon_distance(pdf_a, pdf_b):
    flat_a = np.ravel(pdf_a).astype(np.float64)
    flat_b = np.ravel(pdf_b).astype(np.float64)
    flat_a = np.clip(flat_a / np.sum(flat_a), 1.0e-16, None)
    flat_b = np.clip(flat_b / np.sum(flat_b), 1.0e-16, None)
    mix = 0.5 * (flat_a + flat_b)
    kl_a = np.sum(flat_a * np.log(flat_a / mix))
    kl_b = np.sum(flat_b * np.log(flat_b / mix))
    return float(np.sqrt(0.5 * (kl_a + kl_b)))


def js_distance_to_stack(candidate, stack):
    """Vectorized JS distance from one candidate PDF to a stack of existing PDFs.

    candidate: (B, B) array.  stack: (N, B, B) array (possibly N == 0).
    Returns a length-N float64 array.
    """
    if stack.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    a = np.ravel(candidate).astype(np.float64)
    a_sum = a.sum()
    if a_sum > 0:
        a = a / a_sum
    a = np.clip(a, 1.0e-16, None)

    flat = stack.reshape(stack.shape[0], -1).astype(np.float64)
    sums = flat.sum(axis=1, keepdims=True)
    sums = np.where(sums > 0, sums, 1.0)
    flat = flat / sums
    flat = np.clip(flat, 1.0e-16, None)

    mix = 0.5 * (a[None, :] + flat)
    kl_a = (a[None, :] * np.log(a[None, :] / mix)).sum(axis=1)
    kl_b = (flat * np.log(flat / mix)).sum(axis=1)
    return np.sqrt(np.maximum(0.0, 0.5 * (kl_a + kl_b)))


def pairwise_min_distances(stack):
    """For each PDF in stack, the min JS distance to any other PDF in stack.

    stack: (N, B, B) array.  Returns length-N float64 array (zeros for N <= 1).
    """
    n = stack.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=np.float64)
    flat = stack.reshape(n, -1).astype(np.float64)
    sums = flat.sum(axis=1, keepdims=True)
    sums = np.where(sums > 0, sums, 1.0)
    flat = flat / sums
    flat = np.clip(flat, 1.0e-16, None)

    novelties = np.full(n, np.inf, dtype=np.float64)
    for i in range(n):
        a = flat[i:i + 1]
        mix = 0.5 * (a + flat)
        kl_a = (a * np.log(a / mix)).sum(axis=1)
        kl_b = (flat * np.log(flat / mix)).sum(axis=1)
        d = np.sqrt(np.maximum(0.0, 0.5 * (kl_a + kl_b)))
        d[i] = np.inf
        novelties[i] = float(d.min())
    return novelties


def compute_moment_bin(moment_values, moment_bin_counts, variance_limit, covariance_limit):
    ranges = [
        (0.0, 1.0),
        (0.0, variance_limit),
        (0.0, 1.0),
        (0.0, variance_limit),
        (-covariance_limit, covariance_limit),
    ]

    bin_ids = []
    for value, count, limits in zip(moment_values, moment_bin_counts, ranges):
        low, high = limits
        clamped = min(max(float(value), low), np.nextafter(high, low))
        position = (clamped - low) / (high - low)
        bin_id = int(position * count)
        bin_id = max(0, min(count - 1, bin_id))
        bin_ids.append(bin_id)
    return tuple(bin_ids)


def min_distance_to_existing(pdf, indices, selected_pdfs):
    if not indices:
        return 1.0, []
    distances = [jensen_shannon_distance(pdf, selected_pdfs[idx]) for idx in indices]
    return float(min(distances)), distances


def get_existing_novelties(indices, selected_pdfs):
    novelties = []
    for idx in indices:
        other_indices = [other for other in indices if other != idx]
        if not other_indices:
            novelties.append(0.0)
            continue
        distance, _ = min_distance_to_existing(selected_pdfs[idx], other_indices, selected_pdfs)
        novelties.append(distance)
    return novelties


def decide_retention(pdf, metadata, bin_indices, bin_source_counts, selected_pdfs, max_per_bin, shape_threshold):
    source_key = (metadata["scalar_config"], metadata["run_id"], metadata["timestep"], metadata["box_width"])
    unseen_source = bin_source_counts[source_key] == 0
    min_distance, _ = min_distance_to_existing(pdf, bin_indices, selected_pdfs)

    if not bin_indices:
        return True, None, 1.0, "first_in_bin"

    if len(bin_indices) < max_per_bin:
        if min_distance >= shape_threshold:
            return True, None, min_distance, "novel_shape"
        if unseen_source:
            return True, None, min_distance, "source_diversity"
        return False, None, min_distance, "redundant_shape"

    existing_novelties = get_existing_novelties(bin_indices, selected_pdfs)
    replace_position = int(np.argmin(existing_novelties))
    replace_index = bin_indices[replace_position]
    replace_novelty = existing_novelties[replace_position]

    if min_distance > replace_novelty and (min_distance >= shape_threshold or unseen_source):
        return True, replace_index, min_distance, "replaced_low_novelty"

    return False, None, min_distance, "bin_full"


def iter_filter_records(
    scalar_a,
    scalar_b,
    nx,
    pad_width,
    box_width,
    stride,
    layout,
    shape_masks,
    run_id,
    scalar_config,
    timestep,
    scalar_tolerance=1e-6,
    invalid_scalar_policy="error",
    max_moment_discrepancy=0.05,
):
    half_width = box_width // 2
    ranges = [
        range(pad_width, nx + pad_width, stride),
        range(pad_width, nx + pad_width, stride),
        range(pad_width, nx + pad_width, stride),
    ]

    for center_i, center_j, center_k in ((i, j, k) for i in ranges[0] for j in ranges[1] for k in ranges[2]):
        block = np.s_[
            center_i - half_width:center_i + half_width,
            center_j - half_width:center_j + half_width,
            center_k - half_width:center_k + half_width,
        ]

        z1_values, z2_values, scalar_stats = prepare_scalar_samples(
            scalar_a[block],
            scalar_b[block],
            tolerance=scalar_tolerance,
            policy=invalid_scalar_policy,
        )
        direct_moments = compute_direct_moments(z1_values, z2_values)
        pdf = compute_histogram_pdf(z1_values, z2_values, layout)
        hist_moments = compute_histogram_moments(pdf, layout)
        moment_abs_err_max = float(np.max(
            np.abs(np.asarray(direct_moments) - np.asarray(hist_moments))
        ))
        if moment_abs_err_max > max_moment_discrepancy:
            raise ValueError(
                "direct and histogram moments disagree: "
                f"{moment_abs_err_max:.6g} > {max_moment_discrepancy:.6g}"
            )
        shape_features = compute_shape_features(pdf, shape_masks)

        metadata = {
            "scalar_config": scalar_config,
            "run_id": int(run_id),
            "timestep": int(timestep),
            "box_width": int(box_width),
            "stride": int(stride),
            "center_i": int(center_i - pad_width),
            "center_j": int(center_j - pad_width),
            "center_k": int(center_k - pad_width),
            "n_dns_cells": int(z1_values.size),
            "mean_a": float(direct_moments[0]),
            "var_a": float(direct_moments[1]),
            "mean_b": float(direct_moments[2]),
            "var_b": float(direct_moments[3]),
            "cov_ab": float(direct_moments[4]),
            "hist_mean_a": float(hist_moments[0]),
            "hist_var_a": float(hist_moments[1]),
            "hist_mean_b": float(hist_moments[2]),
            "hist_var_b": float(hist_moments[3]),
            "hist_cov_ab": float(hist_moments[4]),
            "moment_abs_err_max": moment_abs_err_max,
        }
        metadata.update(scalar_stats)
        metadata.update(shape_features)
        yield metadata, pdf.astype(np.float32)


def write_hdf5_shards(output_dir, dataset_tag, histograms, metadata_df, layout, shard_size, compression):
    shard_paths = []
    num_samples = len(histograms)

    for shard_id, start in enumerate(range(0, num_samples, shard_size)):
        end = min(start + shard_size, num_samples)
        shard_name = f"{dataset_tag}_{shard_id:04d}.h5"
        shard_path = os.path.join(output_dir, shard_name)
        shard_paths.append(shard_path)

        shard_histograms = np.stack(histograms[start:end]).astype(np.float32)
        local_indices = np.arange(end - start, dtype=np.int32)

        with h5py.File(shard_path, "w") as handle:
            handle.attrs["dataset_tag"] = dataset_tag
            handle.attrs["format_version"] = DATASET_FORMAT_VERSION
            handle.attrs["num_samples"] = end - start
            handle.attrs["num_bins_a"] = shard_histograms.shape[1]
            handle.attrs["num_bins_b"] = shard_histograms.shape[2]

            bins_group = handle.create_group("bins")
            bins_group.create_dataset("edges_a", data=layout["edges_a"])
            bins_group.create_dataset("edges_b", data=layout["edges_b"])
            bins_group.create_dataset("centers_a", data=layout["centers_a"])
            bins_group.create_dataset("centers_b", data=layout["centers_b"])
            bins_group.create_dataset("cell_centroid_a", data=layout["grid_a"])
            bins_group.create_dataset("cell_centroid_b", data=layout["grid_b"])
            bins_group.create_dataset("simplex_mask", data=layout["simplex_mask"].astype(np.uint8))
            bins_group.create_dataset("cell_area", data=layout["cell_area"])

            data_group = handle.create_group("data")
            data_group.create_dataset(
                "histograms",
                data=shard_histograms,
                compression=compression,
                shuffle=True,
                chunks=(min(256, end - start), shard_histograms.shape[1], shard_histograms.shape[2]),
            )
            data_group.create_dataset("sample_id", data=metadata_df.iloc[start:end]["sample_id"].to_numpy(np.int64))
            data_group.create_dataset("local_index", data=local_indices)

    return shard_paths


class StreamingRankWriter:
    """Write lossless Phase-1 candidates one bounded-size shard at a time."""

    def __init__(
        self, output_dir, dataset_tag, layout, shard_size, compression,
    ):
        if shard_size <= 0:
            raise ValueError("shard_size must be positive")
        self.output_dir = output_dir
        self.dataset_tag = dataset_tag
        self.layout = layout
        self.shard_size = int(shard_size)
        self.compression = compression
        self.metadata_path = os.path.join(
            output_dir, f"{dataset_tag}_metadata.parquet",
        )
        self._metadata = []
        self._pdfs = []
        self._parquet_writer = None
        self._next_sample_id = 0
        self._next_shard_id = 0
        self._closed = False
        os.makedirs(output_dir, exist_ok=True)
        remove_existing_shards(output_dir, dataset_tag)
        if os.path.exists(self.metadata_path):
            os.remove(self.metadata_path)

    def add(self, metadata, pdf):
        if self._closed:
            raise RuntimeError("cannot add to a closed Phase-1 writer")
        self._metadata.append(dict(metadata))
        self._pdfs.append(np.asarray(pdf, dtype=np.float32))
        if len(self._metadata) >= self.shard_size:
            self._flush()

    def _flush(self):
        if not self._metadata:
            return
        shard_id = self._next_shard_id
        shard_name = f"{self.dataset_tag}_{shard_id:04d}.h5"
        shard_path = os.path.join(self.output_dir, shard_name)
        histograms = np.stack(self._pdfs).astype(np.float32, copy=False)
        count = len(self._metadata)
        sample_ids = np.arange(
            self._next_sample_id, self._next_sample_id + count, dtype=np.int64,
        )
        local_indices = np.arange(count, dtype=np.int32)

        with h5py.File(shard_path, "w") as handle:
            handle.attrs["dataset_tag"] = self.dataset_tag
            handle.attrs["format_version"] = DATASET_FORMAT_VERSION
            handle.attrs["num_samples"] = count
            handle.attrs["num_bins_a"] = histograms.shape[1]
            handle.attrs["num_bins_b"] = histograms.shape[2]
            bins_group = handle.create_group("bins")
            bins_group.create_dataset("edges_a", data=self.layout["edges_a"])
            bins_group.create_dataset("edges_b", data=self.layout["edges_b"])
            bins_group.create_dataset("centers_a", data=self.layout["centers_a"])
            bins_group.create_dataset("centers_b", data=self.layout["centers_b"])
            bins_group.create_dataset("cell_centroid_a", data=self.layout["grid_a"])
            bins_group.create_dataset("cell_centroid_b", data=self.layout["grid_b"])
            bins_group.create_dataset(
                "simplex_mask",
                data=self.layout["simplex_mask"].astype(np.uint8),
            )
            bins_group.create_dataset("cell_area", data=self.layout["cell_area"])
            data_group = handle.create_group("data")
            data_group.create_dataset(
                "histograms",
                data=histograms,
                compression=self.compression,
                shuffle=True,
                chunks=(min(256, count), histograms.shape[1], histograms.shape[2]),
            )
            data_group.create_dataset("sample_id", data=sample_ids)
            data_group.create_dataset("local_index", data=local_indices)

        metadata_df = pd.DataFrame(self._metadata)
        metadata_df.insert(0, "sample_id", sample_ids)
        metadata_df.insert(1, "shard_id", np.full(count, shard_id, dtype=np.int32))
        metadata_df.insert(2, "local_index", local_indices)
        metadata_df.insert(3, "hdf5_file", np.full(count, shard_name, dtype=object))
        table = pa.Table.from_pandas(metadata_df, preserve_index=False)
        if self._parquet_writer is None:
            self._parquet_writer = pq.ParquetWriter(self.metadata_path, table.schema)
        else:
            table = table.cast(self._parquet_writer.schema)
        self._parquet_writer.write_table(table)

        self._next_sample_id += count
        self._next_shard_id += 1
        self._metadata.clear()
        self._pdfs.clear()

    def close(self):
        if not self._closed:
            self._flush()
            if self._parquet_writer is not None:
                self._parquet_writer.close()
            self._closed = True
        return self._next_sample_id

    def abort(self):
        if self._parquet_writer is not None:
            self._parquet_writer.close()
        self._closed = True
        if os.path.exists(self.metadata_path):
            os.remove(self.metadata_path)
        remove_existing_shards(self.output_dir, self.dataset_tag)


def remove_existing_shards(output_dir, dataset_tag):
    if not os.path.isdir(output_dir):
        return
    pattern = re.compile(rf"^{re.escape(dataset_tag)}_[0-9]{{4,}}\.h5$")
    for entry in os.listdir(output_dir):
        if pattern.fullmatch(entry):
            os.remove(os.path.join(output_dir, entry))


def build_metadata_frame(selected_metadata, shard_paths, shard_size):
    metadata_df = pd.DataFrame(selected_metadata)
    metadata_df.insert(0, "sample_id", np.arange(len(metadata_df), dtype=np.int64))

    shard_id = np.zeros(len(metadata_df), dtype=np.int32)
    local_index = np.zeros(len(metadata_df), dtype=np.int32)
    hdf5_file = np.empty(len(metadata_df), dtype=object)
    for sid, start_index in enumerate(range(0, len(metadata_df), shard_size)):
        end_index = min(start_index + shard_size, len(metadata_df))
        shard_id[start_index:end_index] = sid
        local_index[start_index:end_index] = np.arange(end_index - start_index, dtype=np.int32)
        relative_path = os.path.basename(shard_paths[sid])
        hdf5_file[start_index:end_index] = relative_path

    metadata_df.insert(1, "shard_id", shard_id)
    metadata_df.insert(2, "local_index", local_index)
    metadata_df.insert(3, "hdf5_file", hdf5_file)
    return metadata_df


def materialize_dataset(output_dir, dataset_tag, selected_pdfs, selected_metadata, layout, args, counters, missing_paths, fdir, scalar_configs, run_ids, box_widths, stride_values, start, is_checkpoint, extra_manifest=None):
    if not selected_metadata:
        return None

    os.makedirs(output_dir, exist_ok=True)
    staging_dir = os.path.join(
        output_dir, f".{dataset_tag}.staging-{uuid.uuid4().hex}",
    )
    os.makedirs(staging_dir)
    temp_metadata = pd.DataFrame(selected_metadata)
    temp_metadata.insert(0, "sample_id", np.arange(len(temp_metadata), dtype=np.int64))
    shard_paths = write_hdf5_shards(
        staging_dir,
        dataset_tag,
        selected_pdfs,
        temp_metadata,
        layout,
        args.hdf5_shard_size,
        args.compression,
    )

    metadata_df = build_metadata_frame(selected_metadata, shard_paths, args.hdf5_shard_size)
    metadata_path = os.path.join(staging_dir, f"{dataset_tag}_metadata.parquet")
    metadata_df.to_parquet(metadata_path, index=False)

    manifest = {
        "dataset_tag": dataset_tag,
        "format_version": DATASET_FORMAT_VERSION,
        "simplex_geometry": "rectangle_intersection_area_and_centroid",
        "input_folder": fdir,
        "input_layout": "run_00**/ensight-3D/ZA* and ZB*",
        "metadata_path": os.path.basename(metadata_path),
        "hdf5_shards": [os.path.basename(path) for path in shard_paths],
        "scalar_configs": scalar_configs,
        "run_ids": run_ids,
        "box_widths": box_widths,
        "strides": stride_values,
        "pdf_bins": args.pdf_bins,
        "zst": args.zst,
        "uniform_bins": bool(args.uniform_bins),
        "timesteps": {
            "start": args.tstart,
            "end": args.tend,
            "jump": args.tjump,
        },
        "moment_bins": parse_int_spec(args.moment_bins),
        "max_per_moment_bin": args.max_per_moment_bin,
        "shape_threshold": args.shape_threshold,
        "scalar_tolerance": args.scalar_tolerance,
        "invalid_scalar_policy": args.invalid_scalar_policy,
        "max_moment_discrepancy": args.max_moment_discrepancy,
        "counts": dict(counters),
        "num_retained": int(len(metadata_df)),
        "missing_paths": missing_paths,
        "elapsed_seconds": time.time() - start,
        "is_checkpoint": bool(is_checkpoint),
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    manifest_path = os.path.join(staging_dir, f"{dataset_tag}_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    final_manifest = os.path.join(output_dir, os.path.basename(manifest_path))
    final_metadata = os.path.join(output_dir, os.path.basename(metadata_path))
    final_shards = [
        os.path.join(output_dir, os.path.basename(path))
        for path in shard_paths
    ]
    # The manifest is the commit marker. Remove the old marker first, publish
    # every data file with atomic same-filesystem renames, and publish the new
    # marker last. Readers can therefore never mistake a partial write for a
    # complete current dataset.
    try:
        if os.path.exists(final_manifest):
            os.remove(final_manifest)
        remove_existing_shards(output_dir, dataset_tag)
        for staged, final in zip(shard_paths, final_shards):
            os.replace(staged, final)
        os.replace(metadata_path, final_metadata)
        os.replace(manifest_path, final_manifest)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    return final_metadata, final_shards, final_manifest, len(metadata_df)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="MPI driver for the LES-filtered histogram dataset builder")
    parser.add_argument("-f", "--folder", dest="folder", type=str, default=".", help="Folder containing run_00** directories")
    parser.add_argument("--run-ids", dest="run_ids", type=str, default="0:27", help="Run ids or ranges, e.g. 0:27 or 0,3,5")
    parser.add_argument("--scalar-configs", dest="scalar_configs", type=str, default=",".join(SCALAR_CONFIGS), help="Comma-separated scalar configuration names")
    parser.add_argument("--box-widths", dest="box_widths", type=str, default="64,128", help="Full LES box widths in cells; each value must be even")
    parser.add_argument("--strides", dest="strides", type=str, default="32,64", help="Stride values in cells; one value or one per box width")
    parser.add_argument("--pdf-bins", dest="pdf_bins", type=int, default=64, help="Number of histogram cells in each PDF dimension")
    parser.add_argument("--zst", dest="zst", type=float, default=0.1, help="Transition point for the non-uniform bin layout")
    parser.add_argument("--uniform-bins", dest="uniform_bins", action="store_true", help="Use uniform histogram bins")
    parser.add_argument("--tstart", dest="tstart", type=int, default=17, help="Starting timestep index")
    parser.add_argument("--tend", dest="tend", type=int, default=18, help="Ending timestep index, exclusive")
    parser.add_argument("--tjump", dest="tjump", type=int, default=1, help="Timestep increment")
    parser.add_argument("--nx", dest="nx", type=int, default=256, help="Global cube size in each direction")
    parser.add_argument("--npx", dest="npx", type=int, default=8, help="Number of processor partitions in each direction")
    parser.add_argument("--output-dir", dest="output_dir", type=str, default="hybrid_dataset_mpi", help="Directory for final parquet/HDF5 outputs")
    parser.add_argument("--dataset-tag", dest="dataset_tag", type=str, default="dns_pdf_hybrid", help="Prefix used in output file names")
    parser.add_argument("--max-per-moment-bin", dest="max_per_moment_bin", type=int, default=50, help="Final per-bin cap (global). Phase 1 uses this * --per-rank-cap-multiplier.")
    parser.add_argument("--moment-bins", dest="moment_bins", type=str, default="12,12,12,12,12", help="Coarse bin counts for meanA,varA,meanB,varB,covAB")
    parser.add_argument("--variance-limit", dest="variance_limit", type=float, default=0.25, help="Upper clip used when binning variances")
    parser.add_argument("--covariance-limit", dest="covariance_limit", type=float, default=0.25, help="Symmetric clip used when binning covariance")
    parser.add_argument("--shape-threshold", dest="shape_threshold", type=float, default=0.03, help="Minimum Jensen-Shannon distance to keep similar shapes apart")
    parser.add_argument("--corner-threshold", dest="corner_threshold", type=float, default=0.08, help="Threshold used for corner-mass shape features")
    parser.add_argument("--edge-threshold", dest="edge_threshold", type=float, default=0.04, help="Threshold used for edge-mass shape features")
    parser.add_argument("--center-radius", dest="center_radius", type=float, default=0.12, help="Radius around the simplex barycenter for center-mass feature")
    parser.add_argument("--hdf5-shard-size", dest="hdf5_shard_size", type=int, default=25000, help="Maximum number of samples per HDF5 shard")
    parser.add_argument("--compression", dest="compression", type=str, default="gzip", help="HDF5 compression filter name")
    parser.add_argument("--strict-missing", dest="strict_missing", action="store_true", help="Abort on the first missing Ensight folder or file")
    parser.add_argument("--scalar-tolerance", dest="scalar_tolerance", type=float, default=1e-6, help="Tolerance for tiny scalar bound/simplex roundoff before correction")
    parser.add_argument("--invalid-scalar-policy", dest="invalid_scalar_policy", choices=("error", "drop"), default="error", help="Reject a filter cell with materially invalid scalar pairs, or explicitly drop and count them")
    parser.add_argument("--max-moment-discrepancy", dest="max_moment_discrepancy", type=float, default=0.05, help="Maximum allowed direct-vs-histogram moment difference")
    parser.add_argument("--per-rank-cap-multiplier", dest="per_rank_cap_multiplier", type=int, default=2, help="Deprecated compatibility option; phase 1 is now lossless")
    parser.add_argument("--rank-output-subdir", dest="rank_output_subdir", type=str, default="_rank", help="Subdirectory under --output-dir holding per-rank phase-1 outputs")
    parser.add_argument("--keep-rank-outputs", dest="keep_rank_outputs", action="store_true", help="Keep per-rank phase-1 outputs after the final dataset is written")
    parser.add_argument("--rank-dataset-tag", dest="rank_dataset_tag", type=str, default="rank_phase1", help="Dataset tag used for per-rank phase-1 outputs")
    parser.add_argument("--skip-phase1", dest="skip_phase1", action="store_true", help="Skip phase 1 and only run the merge over an existing _rank/ tree. Run with `python ...` (no mpirun needed).")
    parser.add_argument("--merge-progress-every", dest="merge_progress_every", type=int, default=10000, help="Print merge progress every N candidates. Set 0 to disable.")
    parser.add_argument("--merge-sort-chunk-rows", dest="merge_sort_chunk_rows", type=int, default=50000, help="Maximum metadata rows held while externally sorting Phase-1 candidates")
    parser.add_argument("--merge-max-open-files", dest="merge_max_open_files", type=int, default=128, help="Maximum Phase-1 HDF5 files kept open during the merge")
    parser.add_argument("--selection-seed", dest="selection_seed", type=int, default=0, help="Seed for deterministic hash ordering during global diversity selection")
    return parser


def enumerate_work_units(run_ids, scalar_configs):
    units = []
    for run_id in run_ids:
        for scalar_config in scalar_configs:
            units.append((int(run_id), str(scalar_config)))
    return units


def partition_work_units(units, rank, size):
    return [unit for index, unit in enumerate(units) if index % size == rank]


def phase1_configuration(
    args,
    *,
    fdir,
    run_ids,
    scalar_configs,
    box_widths,
    stride_values,
):
    return {
        "format_version": DATASET_FORMAT_VERSION,
        "input_folder": os.path.realpath(fdir),
        "run_ids": [int(value) for value in run_ids],
        "scalar_configs": [str(value) for value in scalar_configs],
        "box_widths": [int(value) for value in box_widths],
        "strides": [int(value) for value in stride_values],
        "timesteps": [int(args.tstart), int(args.tend), int(args.tjump)],
        "nx": int(args.nx),
        "npx": int(args.npx),
        "pdf_bins": int(args.pdf_bins),
        "zst": float(args.zst),
        "uniform_bins": bool(args.uniform_bins),
        "moment_bins": parse_int_spec(args.moment_bins),
        "variance_limit": float(args.variance_limit),
        "covariance_limit": float(args.covariance_limit),
        "shape_threshold": float(args.shape_threshold),
        "corner_threshold": float(args.corner_threshold),
        "edge_threshold": float(args.edge_threshold),
        "center_radius": float(args.center_radius),
        "scalar_tolerance": float(args.scalar_tolerance),
        "invalid_scalar_policy": str(args.invalid_scalar_policy),
        "max_moment_discrepancy": float(args.max_moment_discrepancy),
        "strict_missing": bool(args.strict_missing),
        "lossless_phase1": True,
    }


def configuration_signature(configuration):
    payload = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_rank_outputs(
    rank_dirs,
    rank_dataset_tag,
    expected_signature,
    *,
    expected_size=None,
    expected_work_units=None,
):
    """Reject stale, missing, extra, or incompatible phase-1 rank outputs."""
    if not rank_dirs:
        raise RuntimeError("no phase-1 rank directories were found")
    actual_ranks = []
    actual_work_units = []
    for rank_dir in rank_dirs:
        match = re.fullmatch(r"rank_([0-9]{4})", os.path.basename(rank_dir))
        if not match:
            raise RuntimeError(f"invalid rank directory name: {rank_dir}")
        rank_number = int(match.group(1))
        actual_ranks.append(rank_number)
        manifest_path = os.path.join(
            rank_dir, f"{rank_dataset_tag}_manifest.json",
        )
        if not os.path.isfile(manifest_path):
            raise RuntimeError(f"missing rank manifest: {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("format_version") != DATASET_FORMAT_VERSION:
            raise RuntimeError(f"incompatible dataset format in {manifest_path}")
        if manifest.get("phase1_signature") != expected_signature:
            raise RuntimeError(
                f"phase-1 configuration mismatch in {manifest_path}; "
                "rerun phase 1 with the current arguments"
            )
        if manifest.get("mpi_rank") != rank_number:
            raise RuntimeError(f"rank identity mismatch in {manifest_path}")
        if not manifest.get("lossless_phase1", False):
            raise RuntimeError(f"lossy phase-1 output is not accepted: {manifest_path}")
        work_units = [
            (int(unit[0]), str(unit[1]))
            for unit in manifest.get("rank_work_units", [])
        ]
        actual_work_units.extend(work_units)
        retained = int(manifest.get("num_retained", 0))
        collected = int(
            manifest.get("counts", {}).get("phase1_candidates_collected", 0)
        )
        if retained != collected:
            raise RuntimeError(
                f"lossless candidate count mismatch in {manifest_path}: "
                f"manifest={retained}, collected={collected}"
            )
        if retained > 0:
            parquet_path = os.path.join(
                rank_dir, f"{rank_dataset_tag}_metadata.parquet",
            )
            if not os.path.isfile(parquet_path):
                raise RuntimeError(f"missing rank metadata: {parquet_path}")
            parquet_file = pq.ParquetFile(parquet_path)
            rows_seen = 0
            shard_rows = {}
            expected_shards = set()
            for batch in parquet_file.iter_batches(
                batch_size=65536,
                columns=["sample_id", "local_index", "hdf5_file"],
            ):
                frame = batch.to_pandas()
                count = len(frame)
                sample_ids = frame["sample_id"].to_numpy(dtype=np.int64)
                expected_ids = np.arange(
                    rows_seen, rows_seen + count, dtype=np.int64,
                )
                if not np.array_equal(sample_ids, expected_ids):
                    raise RuntimeError(
                        f"rank metadata sample IDs are not canonical in {parquet_path}"
                    )
                for sample_id, local_index, shard_name in frame.itertuples(
                    index=False, name=None,
                ):
                    shard_name = str(shard_name)
                    expected_shards.add(shard_name)
                    info = shard_rows.setdefault(
                        shard_name,
                        {"start": int(sample_id), "count": 0},
                    )
                    if int(local_index) != info["count"]:
                        raise RuntimeError(
                            f"rank metadata local indices are not canonical in "
                            f"{parquet_path}"
                        )
                    info["count"] += 1
                rows_seen += count
            if rows_seen != retained:
                raise RuntimeError(
                    f"rank metadata count mismatch in {parquet_path}: "
                    f"{rows_seen} != {retained}"
                )
            shard_pattern = re.compile(
                rf"^{re.escape(rank_dataset_tag)}_[0-9]{{4,}}\.h5$"
            )
            actual_shards = {
                name for name in os.listdir(rank_dir)
                if shard_pattern.fullmatch(name)
            }
            if actual_shards != expected_shards:
                raise RuntimeError(
                    f"rank shard set does not match metadata in {rank_dir}"
                )
            for shard_name, info in shard_rows.items():
                shard_path = os.path.join(rank_dir, shard_name)
                with h5py.File(shard_path, "r") as handle:
                    if int(handle.attrs.get("format_version", -1)) != DATASET_FORMAT_VERSION:
                        raise RuntimeError(
                            f"incompatible rank shard: {shard_path}"
                        )
                    required = (
                        "data/histograms", "data/sample_id", "data/local_index",
                    )
                    if any(name not in handle for name in required):
                        raise RuntimeError(
                            f"rank shard lacks row identity arrays: {shard_path}"
                        )
                    count = int(info["count"])
                    if handle["data/histograms"].shape[0] != count:
                        raise RuntimeError(
                            f"rank shard row count disagrees with metadata: {shard_path}"
                        )
                    expected_ids = np.arange(
                        info["start"], info["start"] + count, dtype=np.int64,
                    )
                    if (
                        not np.array_equal(handle["data/sample_id"][:], expected_ids)
                        or not np.array_equal(
                            handle["data/local_index"][:],
                            np.arange(count, dtype=np.int64),
                        )
                    ):
                        raise RuntimeError(
                            f"rank shard row identity disagrees with metadata: "
                            f"{shard_path}"
                        )
        else:
            parquet_path = os.path.join(
                rank_dir, f"{rank_dataset_tag}_metadata.parquet",
            )
            shard_pattern = re.compile(
                rf"^{re.escape(rank_dataset_tag)}_[0-9]{{4,}}\.h5$"
            )
            stale_shards = [
                name for name in os.listdir(rank_dir)
                if shard_pattern.fullmatch(name)
            ]
            if os.path.exists(parquet_path) or stale_shards:
                raise RuntimeError(
                    f"empty rank output contains stale data files: {rank_dir}"
                )

    actual_ranks = sorted(actual_ranks)
    inferred_size = len(actual_ranks) if expected_size is None else expected_size
    expected_ranks = list(range(inferred_size))
    if actual_ranks != expected_ranks:
        raise RuntimeError(
            f"phase-1 rank set mismatch: expected {expected_ranks}, "
            f"found {actual_ranks}"
        )
    if expected_work_units is not None:
        expected_units = sorted(
            (int(run_id), str(config))
            for run_id, config in expected_work_units
        )
        if sorted(actual_work_units) != expected_units:
            raise RuntimeError(
                "phase-1 work-unit coverage is incomplete, duplicated, or stale"
            )
    return inferred_size


def process_work_units_on_rank(
    work_units,
    rank,
    size,
    fdir,
    args,
    layout,
    shape_masks,
    pad_width,
    box_widths,
    stride_values,
    moment_bin_counts,
    per_rank_cap,
    rank_output_dir,
    writer,
):
    """Extract every candidate and stream it to bounded-size Phase-1 shards."""
    width_stride_pairs = list(zip(box_widths, stride_values))
    counters = Counter()
    missing_paths = []

    for run_id, scalar_config in work_units:
        run_dir = os.path.join(fdir, f"run_{run_id:04d}")
        ensight_dir = os.path.join(run_dir, "ensight-3D")

        if not os.path.isdir(run_dir):
            message = f"[rank {rank}] Missing run folder for run {run_id}: {run_dir}"
            missing_paths.append(message)
            if args.strict_missing:
                raise FileNotFoundError(message)
            counters["missing_run_folders"] += 1
            continue
        if not os.path.isdir(ensight_dir):
            message = f"[rank {rank}] Missing ensight-3D folder for run {run_id}: {ensight_dir}"
            missing_paths.append(message)
            if args.strict_missing:
                raise FileNotFoundError(message)
            counters["missing_ensight_folders"] += 1
            continue

        folder_a = os.path.join(ensight_dir, f"ZA{scalar_config}{run_id}")
        folder_b = os.path.join(ensight_dir, f"ZB{scalar_config}{run_id}")
        if not os.path.isdir(folder_a) or not os.path.isdir(folder_b):
            message = f"[rank {rank}] Missing scalar folders for {scalar_config} run {run_id} in {ensight_dir}"
            missing_paths.append(message)
            if args.strict_missing:
                raise FileNotFoundError(message)
            counters["missing_folders"] += 1
            continue

        for timestep in range(args.tstart, args.tend, args.tjump):
            file_a = os.path.join(folder_a, f"ZA{scalar_config}{run_id}.{str(timestep).zfill(6)}")
            file_b = os.path.join(folder_b, f"ZB{scalar_config}{run_id}.{str(timestep).zfill(6)}")
            if not os.path.isfile(file_a) or not os.path.isfile(file_b):
                message = f"[rank {rank}] Missing timestep files for {scalar_config} run {run_id} timestep {timestep}"
                missing_paths.append(message)
                if args.strict_missing:
                    raise FileNotFoundError(message)
                counters["missing_files"] += 1
                continue

            print(f"[rank {rank:03d}] processing run {run_id:04d}, scalar {scalar_config}, timestep {timestep}", flush=True)
            counters["snapshots_read"] += 1
            scalar_a, scalar_b = read_ensight(file_a, file_b, args.nx, pad_width=pad_width, npx=args.npx)

            for box_width, stride in width_stride_pairs:
                for metadata, pdf in iter_filter_records(
                    scalar_a, scalar_b, args.nx, pad_width, box_width, stride,
                    layout, shape_masks, run_id, scalar_config, timestep,
                    scalar_tolerance=args.scalar_tolerance,
                    invalid_scalar_policy=args.invalid_scalar_policy,
                    max_moment_discrepancy=args.max_moment_discrepancy,
                ):
                    counters["raw_records"] += 1
                    counters["scalar_pairs_raw"] += metadata["scalar_pairs_raw"]
                    counters["scalar_pairs_invalid"] += metadata["scalar_pairs_invalid"]
                    counters["scalar_pairs_adjusted"] += metadata["scalar_pairs_adjusted"]
                    counters["scalar_pairs_used"] += metadata["scalar_pairs_used"]
                    if metadata["scalar_pairs_invalid"]:
                        counters["records_with_invalid_pairs"] += 1
                    if metadata["scalar_pairs_adjusted"]:
                        counters["records_with_adjusted_pairs"] += 1
                    moment_values = tuple(metadata[name] for name in MOMENT_NAMES)
                    bin_key = compute_moment_bin(
                        moment_values, moment_bin_counts,
                        args.variance_limit, args.covariance_limit,
                    )
                    metadata["moment_bin_0"] = int(bin_key[0])
                    metadata["moment_bin_1"] = int(bin_key[1])
                    metadata["moment_bin_2"] = int(bin_key[2])
                    metadata["moment_bin_3"] = int(bin_key[3])
                    metadata["moment_bin_4"] = int(bin_key[4])
                    metadata["moment_bin_key"] = "-".join(str(item) for item in bin_key)
                    metadata["shape_novelty"] = 1.0
                    metadata["retain_reason"] = "phase1_lossless"
                    writer.add(metadata, pdf)
                    counters["phase1_candidates_collected"] += 1

    return counters, missing_paths


def write_rank_outputs(
    rank,
    rank_output_dir,
    rank_dataset_tag,
    num_retained,
    counters,
    missing_paths,
    layout,
    args,
    fdir,
    scalar_configs,
    run_ids,
    box_widths,
    stride_values,
    rank_work_units,
    start_time,
    per_rank_cap,
    phase1_config,
    phase1_signature,
):
    os.makedirs(rank_output_dir, exist_ok=True)
    extra = {
        "mpi_rank": int(rank),
        "rank_work_units": [list(unit) for unit in rank_work_units],
        "per_rank_cap": int(per_rank_cap),
        "phase1_configuration": phase1_config,
        "phase1_signature": phase1_signature,
        "lossless_phase1": True,
    }

    manifest = {
        "dataset_tag": rank_dataset_tag,
        "format_version": DATASET_FORMAT_VERSION,
        "num_retained": int(num_retained),
        "counts": dict(counters),
        "missing_paths": missing_paths,
        "elapsed_seconds": time.time() - start_time,
        "is_checkpoint": False,
        **extra,
    }
    manifest_path = os.path.join(
        rank_output_dir, f"{rank_dataset_tag}_manifest.json",
    )
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return int(num_retained)


def _physical_candidate_key(metadata):
    return (
        int(metadata["run_id"]),
        str(metadata["scalar_config"]),
        int(metadata["timestep"]),
        int(metadata["box_width"]),
        int(metadata["center_i"]),
        int(metadata["center_j"]),
        int(metadata["center_k"]),
    )


def _candidate_sort_key(metadata, selection_seed=0):
    physical_key = _physical_candidate_key(metadata)
    payload = json.dumps(
        [int(selection_seed), *physical_key],
        separators=(",", ":"),
    ).encode("utf-8")
    return (hashlib.sha256(payload).digest(), physical_key)


def _write_sorted_candidate_runs(
    rank_dirs,
    rank_dataset_tag,
    selection_seed,
    sort_chunk_rows,
    temporary_dir,
):
    """External-sort bounded metadata chunks and return their Parquet paths."""
    stale_cols = ("moment_bin_0", "moment_bin_1", "moment_bin_2", "moment_bin_3",
                  "moment_bin_4", "moment_bin_key", "shape_novelty", "retain_reason")
    int_cols = ("run_id", "timestep", "box_width", "stride",
                "center_i", "center_j", "center_k", "n_dns_cells")
    required = {
        "sample_id", "local_index", "hdf5_file", "scalar_config",
        "run_id", "timestep", "box_width", "center_i", "center_j", "center_k",
    }
    run_paths = []
    run_number = 0
    for rank_dir in rank_dirs:
        parquet_path = os.path.join(
            rank_dir, f"{rank_dataset_tag}_metadata.parquet",
        )
        if not os.path.isfile(parquet_path):
            continue
        parquet_file = pq.ParquetFile(parquet_path)
        if not required.issubset(parquet_file.schema.names):
            missing = sorted(required - set(parquet_file.schema.names))
            raise RuntimeError(
                f"rank metadata lacks required columns {missing}: {parquet_path}"
            )
        for batch in parquet_file.iter_batches(batch_size=sort_chunk_rows):
            frame = batch.to_pandas()
            if frame.empty:
                continue
            cols_to_drop = [column for column in stale_cols if column in frame]
            if cols_to_drop:
                frame = frame.drop(columns=cols_to_drop)
            for column in int_cols:
                if column in frame:
                    frame[column] = frame[column].astype(int)
            if frame[list(required)].isna().any().any():
                raise RuntimeError(f"null values in rank metadata: {parquet_path}")
            frame["scalar_config"] = frame["scalar_config"].astype(str)
            frame["_selection_digest"] = [
                _candidate_sort_key(record, selection_seed)[0].hex()
                for record in frame.to_dict(orient="records")
            ]
            frame["_rank_dir"] = os.path.abspath(rank_dir)
            frame = frame.sort_values(
                [
                    "_selection_digest", "run_id", "scalar_config", "timestep",
                    "box_width", "center_i", "center_j", "center_k",
                ],
                kind="stable",
            ).reset_index(drop=True)
            run_path = os.path.join(
                temporary_dir, f"candidate_run_{run_number:06d}.parquet",
            )
            frame.to_parquet(run_path, index=False)
            run_paths.append(run_path)
            run_number += 1
    return run_paths


def _iter_sorted_metadata_run(run_path, batch_size=512):
    parquet_file = pq.ParquetFile(run_path)
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        for metadata in batch.to_pandas().to_dict(orient="records"):
            key = (
                bytes.fromhex(metadata["_selection_digest"]),
                _physical_candidate_key(metadata),
            )
            yield key, metadata


def _candidate_pdf(metadata, handles, expected_num_bins, max_open_handles=32):
    rank_dir = metadata["_rank_dir"]
    shard_name = str(metadata["hdf5_file"])
    shard_path = os.path.join(rank_dir, shard_name)
    handle = handles.pop(shard_path, None)
    if handle is None:
        handle = h5py.File(shard_path, "r")
        if int(handle.attrs.get("format_version", -1)) != DATASET_FORMAT_VERSION:
            handle.close()
            raise RuntimeError(f"incompatible rank shard: {shard_name}")
        if "bins/cell_centroid_a" not in handle or "bins/cell_centroid_b" not in handle:
            handle.close()
            raise RuntimeError(
                f"rank shard lacks clipped-simplex geometry: {shard_name}"
            )
        if "data/sample_id" not in handle or "data/local_index" not in handle:
            handle.close()
            raise RuntimeError(
                f"rank shard lacks row identity arrays: {shard_name}"
            )
        histograms = handle["data/histograms"]
        if (
            histograms.ndim != 3
            or (
                expected_num_bins is not None
                and histograms.shape[1:] != (
                    expected_num_bins, expected_num_bins,
                )
            )
        ):
            handle.close()
            raise RuntimeError(
                f"rank shard histogram shape is incompatible: "
                f"{shard_name} {histograms.shape}"
            )
    handles[shard_path] = handle
    while len(handles) > max_open_handles:
        _, oldest = handles.popitem(last=False)
        oldest.close()

    local_index = int(metadata["local_index"])
    if not 0 <= local_index < handle["data/histograms"].shape[0]:
        raise RuntimeError(
            f"rank metadata local_index is out of range in {shard_name}"
        )
    if (
        int(handle["data/sample_id"][local_index]) != int(metadata["sample_id"])
        or int(handle["data/local_index"][local_index]) != local_index
    ):
        raise RuntimeError(
            f"rank HDF5 row identity disagrees with metadata in {shard_name}"
        )
    return np.asarray(handle["data/histograms"][local_index], dtype=np.float32)


def iter_rank_candidates(
    rank_dirs,
    rank_dataset_tag,
    selection_seed=0,
    expected_num_bins=None,
    sort_chunk_rows=50000,
    max_open_files=128,
):
    """External-sort and merge candidates in rank-count-independent hash order."""
    if sort_chunk_rows <= 0:
        raise ValueError("sort_chunk_rows must be positive")
    if max_open_files <= 0:
        raise ValueError("max_open_files must be positive")
    if not rank_dirs:
        return
    temporary_parent = os.path.dirname(os.path.abspath(rank_dirs[0]))
    handles = OrderedDict()
    drop_cols_final = {
        "sample_id", "shard_id", "local_index", "hdf5_file",
        "_selection_digest", "_rank_dir",
    }
    try:
        with tempfile.TemporaryDirectory(
            prefix=".candidate-sort-", dir=temporary_parent,
        ) as temporary_dir:
            run_paths = _write_sorted_candidate_runs(
                rank_dirs,
                rank_dataset_tag,
                selection_seed,
                sort_chunk_rows,
                temporary_dir,
            )
            iterators = [
                iter(_iter_sorted_metadata_run(run_path))
                for run_path in run_paths
            ]
            heap = []
            for run_index, iterator in enumerate(iterators):
                try:
                    key, metadata = next(iterator)
                except StopIteration:
                    continue
                heapq.heappush(
                    heap, (key, run_index, metadata, iterator),
                )

            while heap:
                _, run_index, metadata, iterator = heapq.heappop(heap)
                pdf = _candidate_pdf(
                    metadata,
                    handles,
                    expected_num_bins,
                    max_open_handles=max_open_files,
                )
                clean_metadata = {
                    key: value for key, value in metadata.items()
                    if key not in drop_cols_final
                }
                yield clean_metadata, pdf
                try:
                    next_key, next_metadata = next(iterator)
                except StopIteration:
                    continue
                heapq.heappush(
                    heap, (next_key, run_index, next_metadata, iterator),
                )
    finally:
        for handle in handles.values():
            handle.close()


def merge_per_rank_outputs(
    rank_dirs,
    rank_dataset_tag,
    layout,
    args,
    moment_bin_counts,
    progress_every=10000,
):
    """Replay decide_retention over all rank candidates with the global per-bin cap.

    Vectorized JS distance + incremental novelty cache keep this O(candidates * N)
    instead of the original O(candidates * N^2) hot path inside full bins.
    RAM footprint is bounded by the retained set (same as the serial baseline).
    """
    selected_metadata = []
    selected_pdfs = []
    selected_bin_keys = []
    bin_to_indices = defaultdict(list)
    bin_source_counts = defaultdict(Counter)
    bin_novelties = {}
    counters = Counter()
    candidate_source_counts = Counter()

    max_per_bin = int(args.max_per_moment_bin)
    shape_threshold = float(args.shape_threshold)
    merge_start = time.time()
    last_print_time = merge_start
    last_print_seen = 0

    for metadata, pdf in iter_rank_candidates(
        rank_dirs, rank_dataset_tag, selection_seed=args.selection_seed,
        expected_num_bins=int(args.pdf_bins),
        sort_chunk_rows=int(args.merge_sort_chunk_rows),
        max_open_files=int(args.merge_max_open_files),
    ):
        counters["candidates_seen"] += 1
        moment_values = tuple(metadata[name] for name in MOMENT_NAMES)
        bin_key = compute_moment_bin(
            moment_values, moment_bin_counts,
            args.variance_limit, args.covariance_limit,
        )

        bin_indices = bin_to_indices[bin_key]
        source_counts = bin_source_counts[bin_key]
        source_key = (
            metadata["scalar_config"], metadata["run_id"],
            metadata["timestep"], metadata["box_width"],
        )
        candidate_source_counts[source_key] += 1
        unseen_source = source_counts[source_key] == 0

        metadata["moment_bin_0"] = int(bin_key[0])
        metadata["moment_bin_1"] = int(bin_key[1])
        metadata["moment_bin_2"] = int(bin_key[2])
        metadata["moment_bin_3"] = int(bin_key[3])
        metadata["moment_bin_4"] = int(bin_key[4])
        metadata["moment_bin_key"] = "-".join(str(item) for item in bin_key)

        n_bin = len(bin_indices)
        if n_bin == 0:
            metadata["shape_novelty"] = 1.0
            metadata["retain_reason"] = "first_in_bin"
            selected_metadata.append(metadata)
            selected_pdfs.append(pdf)
            selected_bin_keys.append(bin_key)
            bin_indices.append(len(selected_metadata) - 1)
            source_counts[source_key] += 1
            counters["merge_keep_first_in_bin"] += 1
        else:
            existing_stack = np.stack([selected_pdfs[i] for i in bin_indices])
            distances = js_distance_to_stack(pdf, existing_stack)
            min_distance = float(distances.min())
            metadata["shape_novelty"] = min_distance

            if n_bin < max_per_bin:
                if min_distance >= shape_threshold:
                    reason = "novel_shape"
                    admit = True
                elif unseen_source:
                    reason = "source_diversity"
                    admit = True
                else:
                    reason = "redundant_shape"
                    admit = False

                if admit:
                    metadata["retain_reason"] = reason
                    selected_metadata.append(metadata)
                    selected_pdfs.append(pdf)
                    selected_bin_keys.append(bin_key)
                    bin_indices.append(len(selected_metadata) - 1)
                    source_counts[source_key] += 1
                    if bin_key in bin_novelties:
                        old_novelties = bin_novelties[bin_key]
                        updated = np.minimum(old_novelties, distances)
                        bin_novelties[bin_key] = np.append(updated, min_distance)
                    counters[f"merge_keep_{reason}"] += 1
                else:
                    metadata["retain_reason"] = reason
                    counters[f"merge_discard_{reason}"] += 1
            else:
                novelties = bin_novelties.get(bin_key)
                if novelties is None:
                    novelties = pairwise_min_distances(existing_stack)
                    bin_novelties[bin_key] = novelties
                replace_position = int(np.argmin(novelties))
                replace_novelty = float(novelties[replace_position])

                if min_distance > replace_novelty and (min_distance >= shape_threshold or unseen_source):
                    reason = "replaced_low_novelty"
                    metadata["retain_reason"] = reason
                    replace_global_index = bin_indices[replace_position]
                    old_metadata = selected_metadata[replace_global_index]
                    old_source_key = (
                        old_metadata["scalar_config"], old_metadata["run_id"],
                        old_metadata["timestep"], old_metadata["box_width"],
                    )
                    source_counts[old_source_key] -= 1
                    if source_counts[old_source_key] <= 0:
                        del source_counts[old_source_key]
                    selected_metadata[replace_global_index] = metadata
                    selected_pdfs[replace_global_index] = pdf
                    selected_bin_keys[replace_global_index] = bin_key
                    source_counts[source_key] += 1
                    # Neighbors' novelties may change unpredictably; force recompute next time.
                    bin_novelties.pop(bin_key, None)
                    counters[f"merge_keep_{reason}"] += 1
                else:
                    metadata["retain_reason"] = "bin_full"
                    counters["merge_discard_bin_full"] += 1

        seen = counters["candidates_seen"]
        if progress_every > 0 and seen % progress_every == 0:
            now = time.time()
            dt = max(now - last_print_time, 1.0e-6)
            rate = (seen - last_print_seen) / dt
            elapsed = now - merge_start
            print(
                f"[rank 000] merge progress: seen={seen} retained={len(selected_metadata)} "
                f"populated_bins={len(bin_to_indices)} "
                f"cached_full_bins={len(bin_novelties)} "
                f"rate={rate:.0f} cand/s elapsed={elapsed:.0f}s",
                flush=True,
            )
            last_print_time = now
            last_print_seen = seen

    retained_source_counts = Counter(
        (
            metadata["scalar_config"], metadata["run_id"],
            metadata["timestep"], metadata["box_width"],
        )
        for metadata in selected_metadata
    )
    for metadata in selected_metadata:
        source_key = (
            metadata["scalar_config"], metadata["run_id"],
            metadata["timestep"], metadata["box_width"],
        )
        candidates = candidate_source_counts[source_key]
        retained = retained_source_counts[source_key]
        metadata["source_candidate_count"] = int(candidates)
        metadata["source_retained_count"] = int(retained)
        metadata["source_retention_fraction"] = float(retained / candidates)
        metadata["source_inverse_retention_weight"] = float(candidates / retained)

    return selected_metadata, selected_pdfs, counters


def aggregate_rank_manifests(rank_dirs, rank_dataset_tag):
    """Sum per-rank counters and concatenate missing_paths in rank order."""
    aggregated_counters = Counter()
    aggregated_missing = []
    rank_summaries = []
    for rank_dir in rank_dirs:
        manifest_path = os.path.join(rank_dir, f"{rank_dataset_tag}_manifest.json")
        if not os.path.isfile(manifest_path):
            continue
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        for key, value in manifest.get("counts", {}).items():
            aggregated_counters[f"phase1_{key}"] += int(value)
        aggregated_missing.extend(manifest.get("missing_paths", []))
        rank_summaries.append({
            "rank": manifest.get("mpi_rank"),
            "num_retained": manifest.get("num_retained", 0),
            "elapsed_seconds": manifest.get("elapsed_seconds"),
            "work_units": manifest.get("rank_work_units", []),
        })
    return aggregated_counters, aggregated_missing, rank_summaries


def main():
    start_time = time.time()
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        from mpi4py import MPI  # noqa: F401
    except ImportError as exc:
        sys.stderr.write(
            "mpi4py is required; install it in ct-env: "
            "conda activate ct-env && pip install mpi4py\n"
        )
        raise exc
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if args.per_rank_cap_multiplier < 1:
        raise ValueError("--per-rank-cap-multiplier must be >= 1")
    if args.nx <= 0 or args.npx <= 0 or args.nx % args.npx != 0:
        raise ValueError("--nx and --npx must be positive and nx must be divisible by npx")
    if args.tjump <= 0 or args.tend <= args.tstart:
        raise ValueError("timesteps require --tjump > 0 and --tend > --tstart")
    if (
        args.max_per_moment_bin <= 0
        or args.hdf5_shard_size <= 0
        or args.merge_sort_chunk_rows <= 0
        or args.merge_max_open_files <= 0
    ):
        raise ValueError("moment-bin, shard, and merge resource caps must be positive")
    if args.variance_limit <= 0.0 or args.covariance_limit <= 0.0:
        raise ValueError("variance and covariance limits must be positive")
    if args.scalar_tolerance < 0.0:
        raise ValueError("--scalar-tolerance must be non-negative")
    if args.max_moment_discrepancy <= 0.0:
        raise ValueError("--max-moment-discrepancy must be positive")

    run_ids = parse_int_spec(args.run_ids)
    scalar_configs = parse_str_list(args.scalar_configs)
    box_widths = parse_int_spec(args.box_widths)
    stride_values = parse_int_spec(args.strides)
    moment_bin_counts = parse_int_spec(args.moment_bins)

    if len(set(run_ids)) != len(run_ids):
        raise ValueError("--run-ids must not contain duplicates")
    if len(set(scalar_configs)) != len(scalar_configs):
        raise ValueError("--scalar-configs must not contain duplicates")
    if len(set(box_widths)) != len(box_widths):
        raise ValueError("--box-widths must not contain duplicates")
    if len(moment_bin_counts) != 5:
        raise ValueError("--moment-bins must provide exactly 5 integers")
    if any(count <= 0 for count in moment_bin_counts):
        raise ValueError("--moment-bins values must be positive")
    if len(stride_values) == 1:
        stride_values = stride_values * len(box_widths)
    if len(stride_values) != len(box_widths):
        raise ValueError("--strides must provide one value or one value per box width")
    if any(width <= 0 or width % 2 != 0 for width in box_widths):
        raise ValueError("All --box-widths values must be positive even integers")
    if any(stride <= 0 for stride in stride_values):
        raise ValueError("All stride values must be positive")
    if not (0.0 <= args.shape_threshold <= 1.0):
        raise ValueError("--shape-threshold must lie in [0, 1]")
    if args.corner_threshold < 0.0 or args.edge_threshold < 0.0:
        raise ValueError("corner and edge thresholds must be non-negative")
    if args.center_radius < 0.0:
        raise ValueError("--center-radius must be non-negative")
    if max(box_widths) // 2 > args.nx:
        raise ValueError("half of the largest box width cannot exceed --nx")

    layout = get_histogram_layout(args.pdf_bins, zst=args.zst, non_uniform=not args.uniform_bins)
    shape_masks = build_shape_masks(layout, args.corner_threshold, args.edge_threshold, args.center_radius)
    pad_width = max(width // 2 for width in box_widths)

    fdir = os.path.abspath(args.folder)
    if not args.skip_phase1 and not os.path.isdir(fdir):
        raise FileNotFoundError(f"input folder does not exist: {fdir}")
    output_dir = os.path.abspath(args.output_dir)
    rank_root_dir = os.path.join(output_dir, args.rank_output_subdir)
    rank_output_dir = os.path.join(rank_root_dir, f"rank_{rank:04d}")

    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        if not args.skip_phase1:
            if os.path.isdir(rank_root_dir):
                shutil.rmtree(rank_root_dir)
            os.makedirs(rank_root_dir)
    comm.Barrier()

    per_rank_cap = max(1, args.max_per_moment_bin * args.per_rank_cap_multiplier)
    phase1_config = phase1_configuration(
        args,
        fdir=fdir,
        run_ids=run_ids,
        scalar_configs=scalar_configs,
        box_widths=box_widths,
        stride_values=stride_values,
    )
    phase1_signature = configuration_signature(phase1_config)

    if args.skip_phase1:
        if rank != 0:
            return
        if not os.path.isdir(rank_root_dir):
            raise FileNotFoundError(
                f"--skip-phase1 set but {rank_root_dir} does not exist. "
                "Run phase 1 first, or point --output-dir at the directory that contains it."
            )
        print(f"[rank 000] --skip-phase1: jumping straight to merge over {rank_root_dir}", flush=True)
    else:
        all_units = enumerate_work_units(run_ids, scalar_configs)
        rank_units = partition_work_units(all_units, rank, size)

        if rank == 0:
            print(
                f"[mpi] size={size} total_work_units={len(all_units)} "
                f"per_rank_units≈{len(all_units)/max(size,1):.2f} "
                f"phase1=lossless global_cap={args.max_per_moment_bin}",
                flush=True,
            )

        rank_writer = StreamingRankWriter(
            rank_output_dir,
            args.rank_dataset_tag,
            layout,
            args.hdf5_shard_size,
            args.compression,
        )
        try:
            rank_counters, rank_missing = process_work_units_on_rank(
                work_units=rank_units,
                rank=rank,
                size=size,
                fdir=fdir,
                args=args,
                layout=layout,
                shape_masks=shape_masks,
                pad_width=pad_width,
                box_widths=box_widths,
                stride_values=stride_values,
                moment_bin_counts=moment_bin_counts,
                per_rank_cap=per_rank_cap,
                rank_output_dir=rank_output_dir,
                writer=rank_writer,
            )
            rank_candidate_count = rank_writer.close()
        except Exception:
            rank_writer.abort()
            raise

        rank_kept = write_rank_outputs(
            rank=rank,
            rank_output_dir=rank_output_dir,
            rank_dataset_tag=args.rank_dataset_tag,
            num_retained=rank_candidate_count,
            counters=rank_counters,
            missing_paths=rank_missing,
            layout=layout,
            args=args,
            fdir=fdir,
            scalar_configs=scalar_configs,
            run_ids=run_ids,
            box_widths=box_widths,
            stride_values=stride_values,
            rank_work_units=rank_units,
            start_time=start_time,
            per_rank_cap=per_rank_cap,
            phase1_config=phase1_config,
            phase1_signature=phase1_signature,
        )
        print(f"[rank {rank:03d}] phase 1 done: retained={rank_kept} elapsed={time.time()-start_time:.1f}s", flush=True)

        comm.Barrier()

        if rank != 0:
            return

    print(f"[rank 000] phase 2 (merge) start; reading per-rank outputs from {rank_root_dir}", flush=True)
    rank_dirs = sorted(glob.glob(os.path.join(rank_root_dir, "rank_*")))
    source_mpi_size = validate_rank_outputs(
        rank_dirs,
        args.rank_dataset_tag,
        phase1_signature,
        expected_size=(None if args.skip_phase1 else size),
        expected_work_units=enumerate_work_units(run_ids, scalar_configs),
    )
    phase1_counters, phase1_missing, rank_summaries = aggregate_rank_manifests(rank_dirs, args.rank_dataset_tag)

    final_metadata, final_pdfs, merge_counters = merge_per_rank_outputs(
        rank_dirs=rank_dirs,
        rank_dataset_tag=args.rank_dataset_tag,
        layout=layout,
        args=args,
        moment_bin_counts=moment_bin_counts,
        progress_every=args.merge_progress_every,
    )
    print(f"[rank 000] merge complete; retained={len(final_metadata)}", flush=True)

    if not final_metadata:
        raise RuntimeError("No samples were retained after the merge; check inputs and CLI args")

    combined_counters = Counter()
    combined_counters.update(phase1_counters)
    combined_counters.update(merge_counters)

    extra = {
        "mpi_size": int(source_mpi_size),
        "per_rank_cap": int(per_rank_cap),
        "per_rank_cap_multiplier": int(args.per_rank_cap_multiplier),
        "phase1_configuration": phase1_config,
        "phase1_signature": phase1_signature,
        "lossless_phase1": True,
        "merge_order": "sha256_seeded_physical_candidate_key",
        "selection_seed": int(args.selection_seed),
        "rank_summaries": rank_summaries,
    }

    result = materialize_dataset(
        output_dir=output_dir,
        dataset_tag=args.dataset_tag,
        selected_pdfs=final_pdfs,
        selected_metadata=final_metadata,
        layout=layout,
        args=args,
        counters=combined_counters,
        missing_paths=phase1_missing,
        fdir=fdir,
        scalar_configs=scalar_configs,
        run_ids=run_ids,
        box_widths=box_widths,
        stride_values=stride_values,
        start=start_time,
        is_checkpoint=False,
        extra_manifest=extra,
    )
    metadata_path, shard_paths, manifest_path, final_count = result
    try:
        post_write_validation = validate_dataset(
            metadata_path,
            output_dir,
            num_bins=args.pdf_bins,
            zst=args.zst,
            uniform_bins=args.uniform_bins,
            max_moment_discrepancy=args.max_moment_discrepancy,
        )
    except Exception:
        if os.path.exists(manifest_path):
            os.remove(manifest_path)
        raise
    with open(manifest_path, "r", encoding="utf-8") as handle:
        final_manifest_data = json.load(handle)
    final_manifest_data["post_write_validation"] = post_write_validation
    validation_manifest_path = manifest_path + ".tmp"
    with open(validation_manifest_path, "w", encoding="utf-8") as handle:
        json.dump(final_manifest_data, handle, indent=2)
    os.replace(validation_manifest_path, manifest_path)

    if not args.keep_rank_outputs:
        shutil.rmtree(rank_root_dir, ignore_errors=True)

    elapsed = time.time() - start_time
    print(f"Wrote metadata parquet: {metadata_path}")
    print(f"Wrote HDF5 shards: {', '.join(os.path.basename(path) for path in shard_paths)}")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Retained samples: {final_count}")
    print("Elapsed time " + str(timedelta(seconds=elapsed)) + f" (or {elapsed:f} seconds)")


if __name__ == "__main__":
    main()
