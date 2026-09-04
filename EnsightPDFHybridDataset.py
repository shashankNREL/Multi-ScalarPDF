#!/usr/bin/env python3

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from datetime import timedelta

import h5py
import numpy as np
import pandas as pd


SCALAR_CONFIGS = ["I1", "I4", "I5", "L1", "PI1", "PI4", "PI5", "PL1"]
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


def get_bin_centers(num_bins, zst, non_uniform=True):
    centers = np.zeros(num_bins, dtype=np.float64)

    if non_uniform:
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
    else:
        centers = np.linspace(0.0, 1.0, num_bins)

    return centers


def get_histogram_layout(num_bins, zst=0.1, non_uniform=True):
    centers_a = get_bin_centers(num_bins, zst, non_uniform)
    centers_b = get_bin_centers(num_bins, zst, non_uniform)

    edges_a = np.zeros(num_bins + 1, dtype=np.float64)
    edges_b = np.zeros(num_bins + 1, dtype=np.float64)
    cell_area = np.zeros((num_bins, num_bins), dtype=np.float64)

    edges_a[0] = centers_a[0] - (0.5 * centers_a[1] + 0.5 * centers_a[0] - centers_a[0])
    edges_b[0] = centers_b[0] - (0.5 * centers_b[1] + 0.5 * centers_b[0] - centers_b[0])
    edges_a[-1] = centers_a[-1] + (centers_a[-1] - 0.5 * centers_a[-1] - 0.5 * centers_a[-2])
    edges_b[-1] = centers_b[-1] + (centers_b[-1] - 0.5 * centers_b[-1] - 0.5 * centers_b[-2])

    for idx in range(1, len(centers_a)):
        edges_a[idx] = 0.5 * centers_a[idx] + 0.5 * centers_a[idx - 1]
    for idx in range(1, len(centers_b)):
        edges_b[idx] = 0.5 * centers_b[idx] + 0.5 * centers_b[idx - 1]

    for i in range(len(centers_a)):
        for j in range(len(centers_b)):
            cell_area[i, j] = (edges_a[i + 1] - edges_a[i]) * (edges_b[j + 1] - edges_b[j])

    grid_a, grid_b = np.meshgrid(centers_a, centers_b, indexing="ij")
    simplex_mask = (grid_a + grid_b) <= 1.0 + 1.0e-12

    return {
        "edges_a": edges_a,
        "edges_b": edges_b,
        "centers_a": centers_a,
        "centers_b": centers_b,
        "grid_a": grid_a,
        "grid_b": grid_b,
        "z1z2": grid_a * grid_b,
        "cell_area": cell_area,
        "simplex_mask": simplex_mask,
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
    return counts / total


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


def decide_retention(pdf, metadata, bin_key, bin_indices, bin_source_counts, selected_pdfs, args):
    source_key = (metadata["scalar_config"], metadata["run_id"], metadata["timestep"], metadata["box_width"])
    unseen_source = bin_source_counts[source_key] == 0
    min_distance, distances = min_distance_to_existing(pdf, bin_indices, selected_pdfs)

    if not bin_indices:
        return True, None, 1.0, "first_in_bin"

    if len(bin_indices) < args.max_per_moment_bin:
        if min_distance >= args.shape_threshold:
            return True, None, min_distance, "novel_shape"
        if unseen_source:
            return True, None, min_distance, "source_diversity"
        return False, None, min_distance, "redundant_shape"

    existing_novelties = get_existing_novelties(bin_indices, selected_pdfs)
    replace_position = int(np.argmin(existing_novelties))
    replace_index = bin_indices[replace_position]
    replace_novelty = existing_novelties[replace_position]

    if min_distance > replace_novelty and (min_distance >= args.shape_threshold or unseen_source):
        return True, replace_index, min_distance, "replaced_low_novelty"

    return False, None, min_distance, "bin_full"


def iter_filter_records(scalar_a, scalar_b, nx, pad_width, box_width, stride, layout, shape_masks, run_id, scalar_config, timestep):
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

        z1_values = np.ravel(scalar_a[block]).astype(np.float64)
        z2_values = np.ravel(scalar_b[block]).astype(np.float64)
        direct_moments = compute_direct_moments(z1_values, z2_values)
        pdf = compute_histogram_pdf(z1_values, z2_values, layout)
        hist_moments = compute_histogram_moments(pdf, layout)
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
            "n_dns_cells": int(box_width ** 3),
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
            "moment_abs_err_max": float(np.max(np.abs(np.asarray(direct_moments) - np.asarray(hist_moments)))),
        }
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
            handle.attrs["num_samples"] = end - start
            handle.attrs["num_bins_a"] = shard_histograms.shape[1]
            handle.attrs["num_bins_b"] = shard_histograms.shape[2]

            bins_group = handle.create_group("bins")
            bins_group.create_dataset("edges_a", data=layout["edges_a"])
            bins_group.create_dataset("edges_b", data=layout["edges_b"])
            bins_group.create_dataset("centers_a", data=layout["centers_a"])
            bins_group.create_dataset("centers_b", data=layout["centers_b"])
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


def remove_existing_shards(output_dir, dataset_tag):
    prefix = f"{dataset_tag}_"
    suffix = ".h5"
    for entry in os.listdir(output_dir):
        if entry.startswith(prefix) and entry.endswith(suffix):
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


def materialize_dataset(output_dir, dataset_tag, selected_pdfs, selected_metadata, layout, args, counters, missing_paths, fdir, scalar_configs, run_ids, box_widths, stride_values, start, is_checkpoint):
    if not selected_metadata:
        return None

    remove_existing_shards(output_dir, dataset_tag)
    temp_metadata = pd.DataFrame(selected_metadata)
    temp_metadata.insert(0, "sample_id", np.arange(len(temp_metadata), dtype=np.int64))
    shard_paths = write_hdf5_shards(
        output_dir,
        dataset_tag,
        selected_pdfs,
        temp_metadata,
        layout,
        args.hdf5_shard_size,
        args.compression,
    )

    metadata_df = build_metadata_frame(selected_metadata, shard_paths, args.hdf5_shard_size)
    metadata_path = os.path.join(output_dir, f"{dataset_tag}_metadata.parquet")
    metadata_df.to_parquet(metadata_path, index=False)

    manifest = {
        "dataset_tag": dataset_tag,
        "created_in_env": "mlProp",
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
        "counts": dict(counters),
        "num_retained": int(len(metadata_df)),
        "missing_paths": missing_paths,
        "elapsed_seconds": time.time() - start,
        "is_checkpoint": bool(is_checkpoint),
    }
    manifest_path = os.path.join(output_dir, f"{dataset_tag}_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return metadata_path, shard_paths, manifest_path, len(metadata_df)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Generate a balanced LES-filtered histogram dataset from DNS Ensight files")
    parser.add_argument("-f", "--folder", dest="folder", type=str, default=".", help="Folder containing run_00** directories; each run directory must contain ensight-3D/ZA* and ensight-3D/ZB* folders")
    parser.add_argument("--run-ids", dest="run_ids", type=str, default="0:27", help="Run ids or ranges, e.g. 0:27 or 0,3,5")
    parser.add_argument("--scalar-configs", dest="scalar_configs", type=str, default=",".join(SCALAR_CONFIGS), help="Comma-separated scalar configuration names")
    parser.add_argument("--box-widths", dest="box_widths", type=str, default="64,128", help="Full LES box widths in cells; each value must be even")
    parser.add_argument("--strides", dest="strides", type=str, default="32,64", help="Stride values in cells; one value or one per box width")
    parser.add_argument("--pdf-bins", dest="pdf_bins", type=int, default=64, help="Number of histogram cells in each PDF dimension")
    parser.add_argument("--zst", dest="zst", type=float, default=0.1, help="Transition point for the non-uniform bin layout")
    parser.add_argument("--uniform-bins", dest="uniform_bins", action="store_true", help="Use uniform histogram bins instead of the non-uniform layout")
    parser.add_argument("--tstart", dest="tstart", type=int, default=17, help="Starting timestep index")
    parser.add_argument("--tend", dest="tend", type=int, default=18, help="Ending timestep index, exclusive")
    parser.add_argument("--tjump", dest="tjump", type=int, default=1, help="Timestep increment")
    parser.add_argument("--nx", dest="nx", type=int, default=256, help="Global cube size in each direction")
    parser.add_argument("--npx", dest="npx", type=int, default=8, help="Number of processor partitions in each direction")
    parser.add_argument("--output-dir", dest="output_dir", type=str, default="hybrid_dataset", help="Directory for parquet, HDF5, and manifest outputs")
    parser.add_argument("--dataset-tag", dest="dataset_tag", type=str, default="dns_pdf_hybrid", help="Prefix used in output file names")
    parser.add_argument("--max-per-moment-bin", dest="max_per_moment_bin", type=int, default=50, help="Maximum retained samples per coarse moment-space bin")
    parser.add_argument("--moment-bins", dest="moment_bins", type=str, default="12,12,12,12,12", help="Coarse bin counts for meanA,varA,meanB,varB,covAB")
    parser.add_argument("--variance-limit", dest="variance_limit", type=float, default=0.25, help="Upper clip used when binning variances")
    parser.add_argument("--covariance-limit", dest="covariance_limit", type=float, default=0.25, help="Symmetric clip used when binning covariance")
    parser.add_argument("--shape-threshold", dest="shape_threshold", type=float, default=0.03, help="Minimum Jensen-Shannon distance used to keep similar shapes apart")
    parser.add_argument("--corner-threshold", dest="corner_threshold", type=float, default=0.08, help="Threshold used for corner-mass shape features")
    parser.add_argument("--edge-threshold", dest="edge_threshold", type=float, default=0.04, help="Threshold used for edge-mass shape features")
    parser.add_argument("--center-radius", dest="center_radius", type=float, default=0.12, help="Radius around the simplex barycenter used for center-mass shape feature")
    parser.add_argument("--hdf5-shard-size", dest="hdf5_shard_size", type=int, default=25000, help="Maximum number of samples written to one HDF5 shard")
    parser.add_argument("--compression", dest="compression", type=str, default="gzip", help="HDF5 compression filter name")
    parser.add_argument("--strict-missing", dest="strict_missing", action="store_true", help="Abort on the first missing Ensight folder or file")
    parser.add_argument("--checkpoint-every-snapshots", dest="checkpoint_every_snapshots", type=int, default=1, help="Write checkpoint parquet/HDF5 outputs after this many processed snapshots; set 0 to disable")
    parser.add_argument("--checkpoint-tag-suffix", dest="checkpoint_tag_suffix", type=str, default="checkpoint", help="Suffix appended to dataset-tag for periodic checkpoint outputs")
    return parser


def main():
    start = time.time()
    parser = build_arg_parser()
    args = parser.parse_args()

    run_ids = parse_int_spec(args.run_ids)
    scalar_configs = parse_str_list(args.scalar_configs)
    box_widths = parse_int_spec(args.box_widths)
    stride_values = parse_int_spec(args.strides)
    moment_bin_counts = parse_int_spec(args.moment_bins)

    if len(moment_bin_counts) != 5:
        raise ValueError("--moment-bins must provide exactly 5 integers")
    if len(stride_values) == 1:
        stride_values = stride_values * len(box_widths)
    if len(stride_values) != len(box_widths):
        raise ValueError("--strides must provide one value or one value per box width")
    if any(width <= 0 or width % 2 != 0 for width in box_widths):
        raise ValueError("All --box-widths values must be positive even integers")
    if any(stride <= 0 for stride in stride_values):
        raise ValueError("All stride values must be positive")

    width_stride_pairs = list(zip(box_widths, stride_values))
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    layout = get_histogram_layout(args.pdf_bins, zst=args.zst, non_uniform=not args.uniform_bins)
    shape_masks = build_shape_masks(layout, args.corner_threshold, args.edge_threshold, args.center_radius)
    pad_width = max(width // 2 for width in box_widths)

    selected_metadata = []
    selected_pdfs = []
    selected_bin_keys = []
    bin_to_indices = defaultdict(list)
    bin_source_counts = defaultdict(Counter)
    counters = Counter()
    missing_paths = []
    processed_snapshots = 0

    fdir = os.path.abspath(args.folder)
    for run_id in run_ids:
        run_dir = os.path.join(fdir, f"run_{run_id:04d}")
        ensight_dir = os.path.join(run_dir, "ensight-3D")

        if not os.path.isdir(run_dir):
            message = f"Missing run folder for run {run_id}: {run_dir}"
            missing_paths.append(message)
            if args.strict_missing:
                raise FileNotFoundError(message)
            counters["missing_run_folders"] += 1
            continue

        if not os.path.isdir(ensight_dir):
            message = f"Missing ensight-3D folder for run {run_id}: {ensight_dir}"
            missing_paths.append(message)
            if args.strict_missing:
                raise FileNotFoundError(message)
            counters["missing_ensight_folders"] += 1
            continue

        for scalar_config in scalar_configs:
            folder_a = os.path.join(ensight_dir, f"ZA{scalar_config}{run_id}")
            folder_b = os.path.join(ensight_dir, f"ZB{scalar_config}{run_id}")

            if not os.path.isdir(folder_a) or not os.path.isdir(folder_b):
                message = f"Missing scalar folders for {scalar_config} run {run_id} in {ensight_dir}"
                missing_paths.append(message)
                if args.strict_missing:
                    raise FileNotFoundError(message)
                counters["missing_folders"] += 1
                continue

            for timestep in range(args.tstart, args.tend, args.tjump):
                file_a = os.path.join(folder_a, f"ZA{scalar_config}{run_id}.{str(timestep).zfill(6)}")
                file_b = os.path.join(folder_b, f"ZB{scalar_config}{run_id}.{str(timestep).zfill(6)}")
                if not os.path.isfile(file_a) or not os.path.isfile(file_b):
                    message = f"Missing timestep files for {scalar_config} run {run_id} timestep {timestep}"
                    missing_paths.append(message)
                    if args.strict_missing:
                        raise FileNotFoundError(message)
                    counters["missing_files"] += 1
                    continue

                print(f"Processing run {run_id:04d}, scalar {scalar_config}, timestep {timestep}")
                counters["snapshots_read"] += 1
                scalar_a, scalar_b = read_ensight(file_a, file_b, args.nx, pad_width=pad_width, npx=args.npx)

                for box_width, stride in width_stride_pairs:
                    print(f"  width={box_width} stride={stride}")
                    for metadata, pdf in iter_filter_records(
                        scalar_a,
                        scalar_b,
                        args.nx,
                        pad_width,
                        box_width,
                        stride,
                        layout,
                        shape_masks,
                        run_id,
                        scalar_config,
                        timestep,
                    ):
                        counters["raw_records"] += 1
                        moment_values = tuple(metadata[name] for name in MOMENT_NAMES)
                        bin_key = compute_moment_bin(
                            moment_values,
                            moment_bin_counts,
                            args.variance_limit,
                            args.covariance_limit,
                        )
                        keep, replace_index, min_distance, reason = decide_retention(
                            pdf,
                            metadata,
                            bin_key,
                            bin_to_indices[bin_key],
                            bin_source_counts[bin_key],
                            selected_pdfs,
                            args,
                        )

                        metadata["moment_bin_0"] = int(bin_key[0])
                        metadata["moment_bin_1"] = int(bin_key[1])
                        metadata["moment_bin_2"] = int(bin_key[2])
                        metadata["moment_bin_3"] = int(bin_key[3])
                        metadata["moment_bin_4"] = int(bin_key[4])
                        metadata["moment_bin_key"] = "-".join(str(item) for item in bin_key)
                        metadata["shape_novelty"] = float(min_distance)
                        metadata["retain_reason"] = reason

                        if not keep:
                            counters[f"discard_{reason}"] += 1
                            continue

                        source_key = (metadata["scalar_config"], metadata["run_id"], metadata["timestep"], metadata["box_width"])
                        if replace_index is None:
                            selected_metadata.append(metadata)
                            selected_pdfs.append(pdf)
                            selected_bin_keys.append(bin_key)
                            new_index = len(selected_metadata) - 1
                            bin_to_indices[bin_key].append(new_index)
                        else:
                            old_bin_key = selected_bin_keys[replace_index]
                            old_metadata = selected_metadata[replace_index]
                            old_source_key = (
                                old_metadata["scalar_config"],
                                old_metadata["run_id"],
                                old_metadata["timestep"],
                                old_metadata["box_width"],
                            )
                            bin_source_counts[old_bin_key][old_source_key] -= 1
                            if bin_source_counts[old_bin_key][old_source_key] <= 0:
                                del bin_source_counts[old_bin_key][old_source_key]
                            selected_metadata[replace_index] = metadata
                            selected_pdfs[replace_index] = pdf
                            selected_bin_keys[replace_index] = bin_key

                        bin_source_counts[bin_key][source_key] += 1
                        counters[f"keep_{reason}"] += 1

                processed_snapshots += 1
                print(
                    f"Completed snapshot count={processed_snapshots}, retained={len(selected_metadata)}, raw_seen={counters['raw_records']}"
                )

                if args.checkpoint_every_snapshots > 0 and processed_snapshots % args.checkpoint_every_snapshots == 0 and selected_metadata:
                    checkpoint_tag = f"{args.dataset_tag}_{args.checkpoint_tag_suffix}"
                    checkpoint_result = materialize_dataset(
                        output_dir=output_dir,
                        dataset_tag=checkpoint_tag,
                        selected_pdfs=selected_pdfs,
                        selected_metadata=selected_metadata,
                        layout=layout,
                        args=args,
                        counters=counters,
                        missing_paths=missing_paths,
                        fdir=fdir,
                        scalar_configs=scalar_configs,
                        run_ids=run_ids,
                        box_widths=box_widths,
                        stride_values=stride_values,
                        start=start,
                        is_checkpoint=True,
                    )
                    if checkpoint_result is not None:
                        checkpoint_metadata, checkpoint_shards, checkpoint_manifest, checkpoint_count = checkpoint_result
                        print(
                            f"Checkpoint wrote {checkpoint_count} samples to {checkpoint_metadata} and {len(checkpoint_shards)} shard(s)"
                        )

    if not selected_metadata:
        raise RuntimeError("No samples were retained; adjust the CLI arguments or source paths")
    result = materialize_dataset(
        output_dir=output_dir,
        dataset_tag=args.dataset_tag,
        selected_pdfs=selected_pdfs,
        selected_metadata=selected_metadata,
        layout=layout,
        args=args,
        counters=counters,
        missing_paths=missing_paths,
        fdir=fdir,
        scalar_configs=scalar_configs,
        run_ids=run_ids,
        box_widths=box_widths,
        stride_values=stride_values,
        start=start,
        is_checkpoint=False,
    )
    metadata_path, shard_paths, manifest_path, final_count = result

    elapsed = time.time() - start
    print(f"Wrote metadata parquet: {metadata_path}")
    print(f"Wrote HDF5 shards: {', '.join(os.path.basename(path) for path in shard_paths)}")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Retained samples: {final_count}")
    print("Elapsed time " + str(timedelta(seconds=elapsed)) + f" (or {elapsed:f} seconds)")


if __name__ == "__main__":
    main()


def parse_csv_strings(value: str) -> List[str]:
    return [token.strip() for token in value.split(",") if token.strip()]


def parse_run_numbers(value: str) -> List[int]:
    runs: List[int] = []
    for token in parse_csv_strings(value):
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError("Run-number range must be ascending")
            runs.extend(list(range(start, end + 1)))
        else:
            runs.append(int(token))
    if not runs:
        raise ValueError("At least one run number is required")
    return sorted(set(runs))


def parse_moment_bins(value: str) -> Tuple[int, int, int, int, int]:
    parts = parse_csv_ints(value)
    if len(parts) == 1:
        parts = parts * 5
    if len(parts) != 5:
        raise ValueError("Moment bins must contain 1 or 5 integers")
    return tuple(parts)  # type: ignore[return-value]


def parse_moment_ranges(value: str) -> Tuple[Tuple[float, float], ...]:
    groups = [group.strip() for group in value.split(";") if group.strip()]
    if len(groups) != 5:
        raise ValueError(
            "Moment ranges must contain five min:max groups separated by ';'"
        )
    ranges: List[Tuple[float, float]] = []
    for group in groups:
        lower_text, upper_text = group.split(":", 1)
        lower = float(lower_text)
        upper = float(upper_text)
        if upper <= lower:
            raise ValueError("Moment-range upper bound must exceed lower bound")
        ranges.append((lower, upper))
    return tuple(ranges)


def ensure_even_widths(widths: Sequence[int]) -> List[int]:
    result: List[int] = []
    for width in widths:
        if width <= 0 or width % 2 != 0:
            raise ValueError("Filter widths must be positive even integers")
        result.append(int(width))
    return sorted(set(result))


def resolve_stride(filter_width: int, stride: int | None, stride_scale: float) -> int:
    if stride is not None:
        if stride <= 0:
            raise ValueError("Stride must be positive")
        return stride
    computed = int(round(filter_width * stride_scale))
    return max(1, computed)


def fill_periodic(orig: np.ndarray, nx: int, ghost_width: int) -> np.ndarray:
    mod = np.zeros((nx + 2 * ghost_width, nx + 2 * ghost_width, nx + 2 * ghost_width))

    mod[
        ghost_width:nx + ghost_width,
        ghost_width:nx + ghost_width,
        ghost_width:nx + ghost_width,
    ] = orig[:, :, :]

    mod[0:ghost_width, ghost_width:nx + ghost_width, ghost_width:nx + ghost_width] = (
        orig[nx - ghost_width:nx, :, :]
    )
    mod[
        (nx + ghost_width):(nx + 2 * ghost_width),
        ghost_width:nx + ghost_width,
        ghost_width:nx + ghost_width,
    ] = orig[0:ghost_width, :, :]

    mod[:, 0:ghost_width, :] = mod[:, nx:(nx + ghost_width), :]
    mod[:, (nx + ghost_width):(nx + 2 * ghost_width), :] = mod[:, ghost_width:2 * ghost_width, :]

    mod[:, :, 0:ghost_width] = mod[:, :, nx:(nx + ghost_width)]
    mod[:, :, (nx + ghost_width):(nx + 2 * ghost_width)] = mod[:, :, ghost_width:2 * ghost_width]

    return mod


def read_ensight(
    file_scalar_a: str,
    file_scalar_b: str,
    nx: int,
    ghost_width: int,
    npx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    nx_local = nx // npx
    with open(file_scalar_a, "rb") as handle:
        handle.read(80)
        handle.read(80)
        np.fromfile(handle, dtype=np.int32, count=1)
        handle.read(80)
        scalar_a = np.fromfile(handle, dtype=np.float32, count=-1)
    scalar_a = np.reshape(scalar_a, (nx_local, nx_local, nx_local, npx, npx, npx), order="F")
    scalar_a_reshape = np.zeros((nx, nx, nx))
    for ip in range(npx):
        for jp in range(npx):
            for kp in range(npx):
                scalar_a_reshape[
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
    scalar_b = np.reshape(scalar_b, (nx_local, nx_local, nx_local, npx, npx, npx), order="F")
    scalar_b_reshape = np.zeros((nx, nx, nx))
    for ip in range(npx):
        for jp in range(npx):
            for kp in range(npx):
                scalar_b_reshape[
                    ip * nx_local:(ip + 1) * nx_local,
                    jp * nx_local:(jp + 1) * nx_local,
                    kp * nx_local:(kp + 1) * nx_local,
                ] = scalar_b[:, :, :, kp, jp, ip]

    full_scalar_a = fill_periodic(scalar_a_reshape, nx, ghost_width)
    full_scalar_b = fill_periodic(scalar_b_reshape, nx, ghost_width)
    return full_scalar_a, full_scalar_b


def get_bin_centers(nbins_a: int, nbins_b: int, zst: float, non_uniform: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    bins_a = np.zeros(nbins_a)
    bins_b = np.zeros(nbins_b)

    if non_uniform:
        zcut_a = int(nbins_a / 2)
        zcut_b = int(nbins_b / 2)

        dz_a = zst / float(zcut_a - 1)
        dz_b = zst / float(zcut_b - 1)

        for index in range(0, zcut_a):
            bins_a[index] = float(index) * dz_a

        for index in range(0, zcut_b):
            bins_b[index] = float(index) * dz_b

        m11 = float((nbins_a - 1) ** 2 - (zcut_a - 1) ** 2)
        m12 = float(nbins_a - zcut_a)
        m21 = float(2 * (zcut_a - 1) + 1)
        m22 = 1.0
        r1 = 1.0 - zst
        r2 = dz_a
        delta = m11 * m22 - m12 * m21
        a_coef = (+m22 * r1 - m12 * r2) / delta
        b_coef = (-m21 * r1 + m11 * r2) / delta
        c_coef = zst - a_coef * (zcut_a - 1) ** 2 - b_coef * (zcut_a - 1)
        for index in range(zcut_a, nbins_a):
            bins_a[index] = a_coef * float(index) ** 2 + b_coef * float(index) + c_coef

        m11 = float((nbins_b - 1) ** 2 - (zcut_b - 1) ** 2)
        m12 = float(nbins_b - zcut_b)
        m21 = float(2 * (zcut_b - 1) + 1)
        m22 = 1.0
        r1 = 1.0 - zst
        r2 = dz_b
        delta = m11 * m22 - m12 * m21
        a_coef = (+m22 * r1 - m12 * r2) / delta
        b_coef = (-m21 * r1 + m11 * r2) / delta
        c_coef = zst - a_coef * (zcut_b - 1) ** 2 - b_coef * (zcut_b - 1)
        for index in range(zcut_b, nbins_b):
            bins_b[index] = a_coef * float(index) ** 2 + b_coef * float(index) + c_coef
    else:
        bins_a = np.linspace(0.0, 1.0, nbins_a)
        bins_b = np.linspace(0.0, 1.0, nbins_b)

    z1_mesh, z2_mesh = np.meshgrid(bins_a, bins_b, indexing="ij")
    z1z2 = z1_mesh * z2_mesh
    return z1_mesh, z2_mesh, z1z2


def gen_bin_edges(nbins_a: int, nbins_b: int, zst: float, non_uniform: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    bins_a = np.zeros(nbins_a)
    bins_b = np.zeros(nbins_b)

    if non_uniform:
        zcut_a = int(nbins_a / 2)
        zcut_b = int(nbins_b / 2)

        dz_a = zst / float(zcut_a - 1)
        dz_b = zst / float(zcut_b - 1)

        for index in range(0, zcut_a):
            bins_a[index] = float(index) * dz_a

        for index in range(0, zcut_b):
            bins_b[index] = float(index) * dz_b

        m11 = float((nbins_a - 1) ** 2 - (zcut_a - 1) ** 2)
        m12 = float(nbins_a - zcut_a)
        m21 = float(2 * (zcut_a - 1) + 1)
        m22 = 1.0
        r1 = 1.0 - zst
        r2 = dz_a
        delta = m11 * m22 - m12 * m21
        a_coef = (+m22 * r1 - m12 * r2) / delta
        b_coef = (-m21 * r1 + m11 * r2) / delta
        c_coef = zst - a_coef * (zcut_a - 1) ** 2 - b_coef * (zcut_a - 1)
        for index in range(zcut_a, nbins_a):
            bins_a[index] = a_coef * float(index) ** 2 + b_coef * float(index) + c_coef

        m11 = float((nbins_b - 1) ** 2 - (zcut_b - 1) ** 2)
        m12 = float(nbins_b - zcut_b)
        m21 = float(2 * (zcut_b - 1) + 1)
        m22 = 1.0
        r1 = 1.0 - zst
        r2 = dz_b
        delta = m11 * m22 - m12 * m21
        a_coef = (+m22 * r1 - m12 * r2) / delta
        b_coef = (-m21 * r1 + m11 * r2) / delta
        c_coef = zst - a_coef * (zcut_b - 1) ** 2 - b_coef * (zcut_b - 1)
        for index in range(zcut_b, nbins_b):
            bins_b[index] = a_coef * float(index) ** 2 + b_coef * float(index) + c_coef
    else:
        bins_a = np.linspace(0.0, 1.0, nbins_a)
        bins_b = np.linspace(0.0, 1.0, nbins_b)

    edges_a = np.zeros(len(bins_a) + 1)
    edges_b = np.zeros(len(bins_b) + 1)
    dz_ab = np.zeros((nbins_a, nbins_b))

    edges_a[0] = bins_a[0] - (0.5 * bins_a[1] + 0.5 * bins_a[0] - bins_a[0])
    edges_b[0] = bins_b[0] - (0.5 * bins_b[1] + 0.5 * bins_b[0] - bins_b[0])
    edges_a[-1] = bins_a[-1] + (bins_a[-1] - 0.5 * bins_a[-1] - 0.5 * bins_a[-2])
    edges_b[-1] = bins_b[-1] + (bins_b[-1] - 0.5 * bins_b[-1] - 0.5 * bins_b[-2])

    for index in range(1, len(bins_a)):
        edges_a[index] = 0.5 * bins_a[index] + 0.5 * bins_a[index - 1]
    for index in range(1, len(bins_b)):
        edges_b[index] = 0.5 * bins_b[index] + 0.5 * bins_b[index - 1]

    for row in range(len(bins_a)):
        for col in range(len(bins_b)):
            dz_ab[row, col] = (edges_a[row + 1] - edges_a[row]) * (edges_b[col + 1] - edges_b[col])

    return edges_a, edges_b, dz_ab


def compute_direct_moments(block_a: np.ndarray, block_b: np.ndarray) -> np.ndarray:
    mean_a = np.mean(block_a)
    mean_b = np.mean(block_b)
    var_a = np.mean((block_a - mean_a) ** 2)
    var_b = np.mean((block_b - mean_b) ** 2)
    cov_ab = np.mean((block_a - mean_a) * (block_b - mean_b))
    return np.array([mean_a, var_a, mean_b, var_b, cov_ab], dtype=np.float64)


def compute_histogram_pdf(
    block_a: np.ndarray,
    block_b: np.ndarray,
    bins_a: np.ndarray,
    bins_b: np.ndarray,
    dz_ab: np.ndarray,
) -> np.ndarray:
    pdf, _, _ = np.histogram2d(
        np.ravel(block_a),
        np.ravel(block_b),
        bins=[bins_a, bins_b],
        density=True,
    )
    pdf = np.multiply(pdf, dz_ab)
    total_mass = float(np.sum(pdf))
    if total_mass <= 0.0:
        raise ValueError("Encountered zero-mass histogram PDF")
    return (pdf / total_mass).astype(np.float32)


def compute_histogram_moments(
    pdf: np.ndarray,
    z1_centers: np.ndarray,
    z2_centers: np.ndarray,
    z1z2: np.ndarray,
) -> np.ndarray:
    mean_a = np.sum(pdf * z1_centers)
    mean_b = np.sum(pdf * z2_centers)
    var_a = np.sum(pdf * ((z1_centers - mean_a) ** 2))
    var_b = np.sum(pdf * ((z2_centers - mean_b) ** 2))
    cov_ab = np.sum(pdf * (z1z2 - mean_a * mean_b))
    return np.array([mean_a, var_a, mean_b, var_b, cov_ab], dtype=np.float64)


def build_shape_masks(
    z1_centers: np.ndarray,
    z2_centers: np.ndarray,
    corner_threshold: float,
    edge_threshold: float,
    center_threshold: float,
    diag_threshold: float,
) -> Dict[str, np.ndarray]:
    z3_centers = 1.0 - z1_centers - z2_centers
    masks = {
        "corner_1": z1_centers >= (1.0 - corner_threshold),
        "corner_2": z2_centers >= (1.0 - corner_threshold),
        "corner_3": z3_centers >= (1.0 - corner_threshold),
        "edge_1": z1_centers <= edge_threshold,
        "edge_2": z2_centers <= edge_threshold,
        "edge_3": z3_centers <= edge_threshold,
        "center": (z1_centers >= center_threshold)
        & (z2_centers >= center_threshold)
        & (z3_centers >= center_threshold),
        "diag12": np.abs(z1_centers - z2_centers) <= diag_threshold,
    }
    return masks


def compute_shape_features(pdf: np.ndarray, masks: Dict[str, np.ndarray]) -> np.ndarray:
    flat = pdf.astype(np.float64).ravel(order="F")
    flat = flat / max(np.sum(flat), 1e-15)
    support_fraction = float(np.count_nonzero(flat > 1e-10)) / float(flat.size)
    entropy = -np.sum(flat * np.log(flat + 1e-15))
    entropy_norm = entropy / math.log(float(flat.size))
    peak_mass = float(np.max(flat))
    concentration = float(np.sum(flat ** 2))

    features = np.array(
        [
            entropy_norm,
            concentration,
            peak_mass,
            support_fraction,
            float(np.sum(pdf[masks["corner_1"]])),
            float(np.sum(pdf[masks["corner_2"]])),
            float(np.sum(pdf[masks["corner_3"]])),
            float(np.sum(pdf[masks["edge_1"]])),
            float(np.sum(pdf[masks["edge_2"]])),
            float(np.sum(pdf[masks["edge_3"]])),
            float(np.sum(pdf[masks["center"]])),
            float(np.sum(pdf[masks["diag12"]])),
        ],
        dtype=np.float32,
    )
    return features


def clip_to_range(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def quantize_moment_vector(
    moments: Sequence[float],
    moment_bins: Sequence[int],
    moment_ranges: Sequence[Tuple[float, float]],
) -> Tuple[Tuple[int, ...], str]:
    indices: List[int] = []
    for axis, value in enumerate(moments):
        lower, upper = moment_ranges[axis]
        clipped = clip_to_range(float(value), lower, upper - 1e-12)
        fraction = (clipped - lower) / (upper - lower)
        index = min(moment_bins[axis] - 1, max(0, int(fraction * moment_bins[axis])))
        indices.append(index)
    key = "|".join(str(index) for index in indices)
    return tuple(indices), key


def jensen_shannon_distance(pdf_a: np.ndarray, pdf_b: np.ndarray) -> float:
    flat_a = pdf_a.astype(np.float64).ravel(order="F")
    flat_b = pdf_b.astype(np.float64).ravel(order="F")
    flat_a /= max(np.sum(flat_a), 1e-15)
    flat_b /= max(np.sum(flat_b), 1e-15)
    mixture = 0.5 * (flat_a + flat_b)
    kl_a = np.sum(flat_a * np.log((flat_a + 1e-15) / (mixture + 1e-15)))
    kl_b = np.sum(flat_b * np.log((flat_b + 1e-15) / (mixture + 1e-15)))
    return float(np.sqrt(max(0.0, 0.5 * (kl_a + kl_b))))


def farthest_first_order(features: np.ndarray) -> List[int]:
    if len(features) == 0:
        return []
    if len(features) == 1:
        return [0]

    mean = np.mean(features, axis=0, keepdims=True)
    scale = np.std(features, axis=0, keepdims=True) + 1e-12
    normalized = (features - mean) / scale
    centroid = np.mean(normalized, axis=0, keepdims=True)
    first = int(np.argmax(np.linalg.norm(normalized - centroid, axis=1)))
    order = [first]
    min_dist = np.linalg.norm(normalized - normalized[first:first + 1], axis=1)
    min_dist[first] = -1.0

    while len(order) < len(features):
        next_index = int(np.argmax(min_dist))
        if min_dist[next_index] < 0.0:
            break
        order.append(next_index)
        next_dist = np.linalg.norm(normalized - normalized[next_index:next_index + 1], axis=1)
        min_dist = np.minimum(min_dist, next_dist)
        min_dist[next_index] = -1.0
    return order


@dataclass
class OutputPaths:
    raw_h5: str
    raw_meta: str
    final_h5: str
    final_meta: str
    summary_json: str


class HDF5Writer:
    def __init__(
        self,
        filepath: str,
        histogram_shape: Tuple[int, int],
        compression: str,
        bin_edges_a: np.ndarray,
        bin_edges_b: np.ndarray,
        z1_centers: np.ndarray,
        z2_centers: np.ndarray,
    ) -> None:
        self.filepath = filepath
        self.handle = h5py.File(filepath, "w")
        chunk_rows = 128
        self.histograms = self.handle.create_dataset(
            "histograms",
            shape=(0, histogram_shape[0], histogram_shape[1]),
            maxshape=(None, histogram_shape[0], histogram_shape[1]),
            chunks=(chunk_rows, histogram_shape[0], histogram_shape[1]),
            compression=compression,
            dtype="f4",
        )
        self.moments = self.handle.create_dataset(
            "moments",
            shape=(0, 5),
            maxshape=(None, 5),
            chunks=(chunk_rows, 5),
            compression=compression,
            dtype="f4",
        )
        self.histogram_moments = self.handle.create_dataset(
            "histogram_moments",
            shape=(0, 5),
            maxshape=(None, 5),
            chunks=(chunk_rows, 5),
            compression=compression,
            dtype="f4",
        )
        self.shape_features = self.handle.create_dataset(
            "shape_features",
            shape=(0, len(SHAPE_FEATURE_COLUMNS)),
            maxshape=(None, len(SHAPE_FEATURE_COLUMNS)),
            chunks=(chunk_rows, len(SHAPE_FEATURE_COLUMNS)),
            compression=compression,
            dtype="f4",
        )
        self.handle.create_dataset("bin_edges_a", data=bin_edges_a, dtype="f8")
        self.handle.create_dataset("bin_edges_b", data=bin_edges_b, dtype="f8")
        self.handle.create_dataset("z1_centers", data=z1_centers, dtype="f8")
        self.handle.create_dataset("z2_centers", data=z2_centers, dtype="f8")
        self.count = 0

    def append_batch(
        self,
        histograms: np.ndarray,
        moments: np.ndarray,
        histogram_moments: np.ndarray,
        shape_features: np.ndarray,
    ) -> Tuple[int, int]:
        start = self.count
        count = int(histograms.shape[0])
        end = start + count
        self.histograms.resize((end,) + self.histograms.shape[1:])
        self.moments.resize((end, 5))
        self.histogram_moments.resize((end, 5))
        self.shape_features.resize((end, len(SHAPE_FEATURE_COLUMNS)))
        self.histograms[start:end] = histograms
        self.moments[start:end] = moments
        self.histogram_moments[start:end] = histogram_moments
        self.shape_features[start:end] = shape_features
        self.count = end
        return start, end

    def close(self) -> None:
        self.handle.close()


def build_output_paths(output_dir: str, dataset_name: str) -> OutputPaths:
    return OutputPaths(
        raw_h5=os.path.join(output_dir, dataset_name + ".raw.h5"),
        raw_meta=os.path.join(output_dir, dataset_name + ".raw.parquet"),
        final_h5=os.path.join(output_dir, dataset_name + ".h5"),
        final_meta=os.path.join(output_dir, dataset_name + ".parquet"),
        summary_json=os.path.join(output_dir, dataset_name + ".summary.json"),
    )


def maybe_remove_output(filepath: str, overwrite: bool) -> None:
    if os.path.exists(filepath):
        if not overwrite:
            raise FileExistsError(f"Output already exists: {filepath}")
        os.remove(filepath)


def generate_file_paths(base_dir: str, scalar_config: str, run_number: int, timestep: int) -> Tuple[str, str]:
    folder_a = os.path.join(base_dir, f"ZA{scalar_config}{run_number}")
    folder_b = os.path.join(base_dir, f"ZB{scalar_config}{run_number}")
    filename_a = os.path.join(folder_a, f"ZA{scalar_config}{run_number}.{str(timestep).zfill(6)}")
    filename_b = os.path.join(folder_b, f"ZB{scalar_config}{run_number}.{str(timestep).zfill(6)}")
    return filename_a, filename_b


def extract_records_for_width(
    scalar_a: np.ndarray,
    scalar_b: np.ndarray,
    nx: int,
    ghost_width: int,
    filter_width: int,
    stride: int,
    bins_a: np.ndarray,
    bins_b: np.ndarray,
    dz_ab: np.ndarray,
    z1_centers: np.ndarray,
    z2_centers: np.ndarray,
    z1z2: np.ndarray,
    masks: Dict[str, np.ndarray],
    moment_bins: Sequence[int],
    moment_ranges: Sequence[Tuple[float, float]],
    run_number: int,
    scalar_config: str,
    timestep: int,
    starting_sample_id: int,
) -> Tuple[List[Dict[str, object]], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    half_width = filter_width // 2
    centers = range(ghost_width, nx + ghost_width, stride)
    rows: List[Dict[str, object]] = []
    histograms: List[np.ndarray] = []
    moments_list: List[np.ndarray] = []
    histogram_moments_list: List[np.ndarray] = []
    shape_features_list: List[np.ndarray] = []

    sample_id = starting_sample_id
    for i_center in centers:
        for j_center in centers:
            for k_center in centers:
                block = np.s_[
                    i_center - half_width:i_center + half_width,
                    j_center - half_width:j_center + half_width,
                    k_center - half_width:k_center + half_width,
                ]
                block_a = scalar_a[block]
                block_b = scalar_b[block]
                moments = compute_direct_moments(block_a, block_b)
                histogram = compute_histogram_pdf(block_a, block_b, bins_a, bins_b, dz_ab)
                histogram_moments = compute_histogram_moments(histogram, z1_centers, z2_centers, z1z2)
                shape_features = compute_shape_features(histogram, masks)
                moment_indices, moment_key = quantize_moment_vector(moments, moment_bins, moment_ranges)

                row: Dict[str, object] = {
                    "sample_id": sample_id,
                    "source_sample_id": sample_id,
                    "run_number": run_number,
                    "scalar_config": scalar_config,
                    "timestep": timestep,
                    "filter_width": filter_width,
                    "half_width": half_width,
                    "stride": stride,
                    "center_i": i_center - ghost_width,
                    "center_j": j_center - ghost_width,
                    "center_k": k_center - ghost_width,
                    "mean_a": float(moments[0]),
                    "var_a": float(moments[1]),
                    "mean_b": float(moments[2]),
                    "var_b": float(moments[3]),
                    "cov_ab": float(moments[4]),
                    "hist_mean_a": float(histogram_moments[0]),
                    "hist_var_a": float(histogram_moments[1]),
                    "hist_mean_b": float(histogram_moments[2]),
                    "hist_var_b": float(histogram_moments[3]),
                    "hist_cov_ab": float(histogram_moments[4]),
                    "moment_bin_0": int(moment_indices[0]),
                    "moment_bin_1": int(moment_indices[1]),
                    "moment_bin_2": int(moment_indices[2]),
                    "moment_bin_3": int(moment_indices[3]),
                    "moment_bin_4": int(moment_indices[4]),
                    "moment_bin_key": moment_key,
                }
                for index, column in enumerate(SHAPE_FEATURE_COLUMNS):
                    row[column] = float(shape_features[index])
                rows.append(row)
                histograms.append(histogram)
                moments_list.append(moments.astype(np.float32))
                histogram_moments_list.append(histogram_moments.astype(np.float32))
                shape_features_list.append(shape_features.astype(np.float32))
                sample_id += 1

    return (
        rows,
        np.stack(histograms, axis=0).astype(np.float32),
        np.stack(moments_list, axis=0).astype(np.float32),
        np.stack(histogram_moments_list, axis=0).astype(np.float32),
        np.stack(shape_features_list, axis=0).astype(np.float32),
    )


def select_samples_within_bin(
    group: pd.DataFrame,
    raw_histograms: h5py.Dataset,
    bin_cap: int,
    js_threshold: float,
) -> List[int]:
    if len(group) <= bin_cap:
        return [int(value) for value in group["sample_id"].tolist()]

    features = group[SHAPE_FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    order = farthest_first_order(features)
    ordered_group = group.iloc[order].reset_index(drop=True)
    ordered_ids = ordered_group["sample_id"].astype(int).tolist()
    ordered_histograms = raw_histograms[ordered_ids, :, :]

    kept_ids: List[int] = []
    kept_histograms: List[np.ndarray] = []
    for row_index, sample_id in enumerate(ordered_ids):
        candidate_pdf = ordered_histograms[row_index]
        if not kept_histograms:
            kept_ids.append(sample_id)
            kept_histograms.append(candidate_pdf)
            if len(kept_ids) >= bin_cap:
                break
            continue

        min_distance = min(
            jensen_shannon_distance(candidate_pdf, selected_pdf)
            for selected_pdf in kept_histograms
        )
        if min_distance >= js_threshold:
            kept_ids.append(sample_id)
            kept_histograms.append(candidate_pdf)
            if len(kept_ids) >= bin_cap:
                break

    if len(kept_ids) < bin_cap:
        for row_index, sample_id in enumerate(ordered_ids):
            if sample_id in kept_ids:
                continue
            kept_ids.append(sample_id)
            kept_histograms.append(ordered_histograms[row_index])
            if len(kept_ids) >= bin_cap:
                break
    return kept_ids


def write_selected_outputs(
    raw_meta: pd.DataFrame,
    raw_h5_path: str,
    final_h5_path: str,
    final_meta_path: str,
    compression: str,
) -> Dict[str, object]:
    with h5py.File(raw_h5_path, "r") as raw_handle:
        raw_histograms = raw_handle["histograms"]
        raw_moments = raw_handle["moments"]
        raw_histogram_moments = raw_handle["histogram_moments"]
        raw_shape_features = raw_handle["shape_features"]

        final_writer = HDF5Writer(
            final_h5_path,
            histogram_shape=(raw_histograms.shape[1], raw_histograms.shape[2]),
            compression=compression,
            bin_edges_a=raw_handle["bin_edges_a"][:],
            bin_edges_b=raw_handle["bin_edges_b"][:],
            z1_centers=raw_handle["z1_centers"][:],
            z2_centers=raw_handle["z2_centers"][:],
        )

        kept_rows: List[pd.DataFrame] = []
        selected_per_bin: Dict[str, int] = {}
        next_sample_id = 0

        for moment_key, group in raw_meta.groupby("moment_bin_key", sort=True):
            kept_ids = select_samples_within_bin(
                group,
                raw_histograms,
                bin_cap=int(group["bin_cap"].iloc[0]),
                js_threshold=float(group["shape_js_threshold"].iloc[0]),
            )
            selected_per_bin[str(moment_key)] = len(kept_ids)
            kept_rows.append(group[group["sample_id"].isin(kept_ids)].copy())

        selected_meta = pd.concat(kept_rows, ignore_index=True)
        selected_meta = selected_meta.sort_values(["sample_id"]).reset_index(drop=True)
        source_ids = selected_meta["sample_id"].astype(int).to_numpy()
        selected_meta["source_sample_id"] = source_ids
        selected_meta["sample_id"] = np.arange(len(selected_meta), dtype=np.int64)

        final_writer.append_batch(
            raw_histograms[source_ids, :, :],
            raw_moments[source_ids, :],
            raw_histogram_moments[source_ids, :],
            raw_shape_features[source_ids, :],
        )
        final_writer.close()
        selected_meta.to_parquet(final_meta_path, index=False)

    summary = {
        "selected_count": int(len(selected_meta)),
        "selected_per_bin": selected_per_bin,
        "next_sample_id": next_sample_id,
    }
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate balanced LES-PDF histogram datasets from DNS Ensight files"
    )
    parser.add_argument(
        "-f",
        "--folder",
        dest="folder",
        type=str,
        default=".",
        help="Folder containing ZA*/ZB* Ensight subdirectories",
    )
    parser.add_argument(
        "--run-numbers",
        dest="run_numbers",
        type=str,
        default="0-26",
        help="Comma-separated run numbers and ranges, e.g. '0-26' or '0,3,7-12'",
    )
    parser.add_argument(
        "--scalar-configs",
        dest="scalar_configs",
        type=str,
        default=",".join(DEFAULT_SCALAR_CONFIGS),
        help="Comma-separated scalar configuration names",
    )
    parser.add_argument(
        "-W",
        "--widths",
        dest="widths",
        type=str,
        default="64,128",
        help="Comma-separated LES filter widths in DNS cells; values are full box widths",
    )
    parser.add_argument(
        "-s",
        "--stride",
        dest="stride",
        type=int,
        default=None,
        help="Explicit traversal stride in DNS cells; if omitted, width*stride-scale is used",
    )
    parser.add_argument(
        "--stride-scale",
        dest="stride_scale",
        type=float,
        default=0.5,
        help="Stride multiplier used when --stride is omitted",
    )
    parser.add_argument(
        "-b",
        "--bins",
        dest="bins",
        type=int,
        default=65,
        help="Histogram bins per scalar direction",
    )
    parser.add_argument(
        "--zst",
        dest="zst",
        type=float,
        default=0.1,
        help="Stoichiometric mixture fraction used by the non-uniform binning rule",
    )
    parser.add_argument(
        "--uniform-bins",
        dest="uniform_bins",
        action="store_true",
        help="Use uniform histogram bins instead of the existing non-uniform mapping",
    )
    parser.add_argument(
        "--tstart",
        dest="tstart",
        type=int,
        default=17,
        help="Starting Ensight timestep index (inclusive)",
    )
    parser.add_argument(
        "--tend",
        dest="tend",
        type=int,
        default=18,
        help="Ending Ensight timestep index (exclusive)",
    )
    parser.add_argument(
        "--tjump",
        dest="tjump",
        type=int,
        default=1,
        help="Stride between successive Ensight timestep indices",
    )
    parser.add_argument(
        "--nx",
        dest="nx",
        type=int,
        default=256,
        help="Global DNS cube size in one direction",
    )
    parser.add_argument(
        "--npx",
        dest="npx",
        type=int,
        default=8,
        help="Processor decomposition per direction used in the Ensight files",
    )
    parser.add_argument(
        "--dataset-name",
        dest="dataset_name",
        type=str,
        default="les_pdf_dataset",
        help="Base filename used for parquet/HDF5 outputs",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=str,
        default="./dataset_out",
        help="Directory where parquet/HDF5 outputs will be written",
    )
    parser.add_argument(
        "--moment-bins",
        dest="moment_bins",
        type=str,
        default=",".join(str(value) for value in DEFAULT_MOMENT_BINS),
        help="Moment-space bin counts: one integer or five comma-separated integers",
    )
    parser.add_argument(
        "--moment-ranges",
        dest="moment_ranges",
        type=str,
        default="0:1;0:0.25;0:1;0:0.25;-0.25:0.25",
        help="Five min:max groups for moment-space quantization",
    )
    parser.add_argument(
        "--bin-cap",
        dest="bin_cap",
        type=int,
        default=50,
        help="Maximum retained samples per coarse 5-D moment bin",
    )
    parser.add_argument(
        "--shape-js-threshold",
        dest="shape_js_threshold",
        type=float,
        default=0.05,
        help="Minimum Jensen-Shannon distance for shape-diverse samples inside one moment bin",
    )
    parser.add_argument(
        "--corner-threshold",
        dest="corner_threshold",
        type=float,
        default=0.10,
        help="Corner-mass threshold in simplex coordinates for shape features",
    )
    parser.add_argument(
        "--edge-threshold",
        dest="edge_threshold",
        type=float,
        default=0.05,
        help="Edge-mass threshold in simplex coordinates for shape features",
    )
    parser.add_argument(
        "--center-threshold",
        dest="center_threshold",
        type=float,
        default=0.20,
        help="Minimum Z1, Z2, Z3 used to define center-mass shape feature",
    )
    parser.add_argument(
        "--diag12-threshold",
        dest="diag12_threshold",
        type=float,
        default=0.05,
        help="Distance-to-Z1=Z2 line used in the diagonal shape feature",
    )
    parser.add_argument(
        "--compression",
        dest="compression",
        type=str,
        default="gzip",
        help="HDF5 compression filter used for dense arrays",
    )
    parser.add_argument(
        "--missing-policy",
        dest="missing_policy",
        choices=["skip", "error"],
        default="skip",
        help="How to handle missing Ensight files",
    )
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "--drop-raw",
        dest="drop_raw",
        action="store_true",
        help="Delete raw parquet/HDF5 extraction outputs after final pruning",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    widths = ensure_even_widths(parse_csv_ints(args.widths))
    run_numbers = parse_run_numbers(args.run_numbers)
    scalar_configs = parse_csv_strings(args.scalar_configs)
    moment_bins = parse_moment_bins(args.moment_bins)
    moment_ranges = parse_moment_ranges(args.moment_ranges)
    non_uniform = not bool(args.uniform_bins)
    pad_half_width = max(widths) // 2

    os.makedirs(args.output_dir, exist_ok=True)
    output_paths = build_output_paths(args.output_dir, args.dataset_name)
    for filepath in [
        output_paths.raw_h5,
        output_paths.raw_meta,
        output_paths.final_h5,
        output_paths.final_meta,
        output_paths.summary_json,
    ]:
        maybe_remove_output(filepath, args.overwrite)

    bin_edges_a, bin_edges_b, dz_ab = gen_bin_edges(args.bins, args.bins, args.zst, non_uniform)
    z1_centers, z2_centers, z1z2 = get_bin_centers(args.bins, args.bins, args.zst, non_uniform)
    shape_masks = build_shape_masks(
        z1_centers,
        z2_centers,
        corner_threshold=args.corner_threshold,
        edge_threshold=args.edge_threshold,
        center_threshold=args.center_threshold,
        diag_threshold=args.diag12_threshold,
    )

    raw_writer = HDF5Writer(
        output_paths.raw_h5,
        histogram_shape=(args.bins, args.bins),
        compression=args.compression,
        bin_edges_a=bin_edges_a,
        bin_edges_b=bin_edges_b,
        z1_centers=z1_centers,
        z2_centers=z2_centers,
    )

    metadata_rows: List[Dict[str, object]] = []
    missing_files: List[str] = []
    next_sample_id = 0
    start_time = time.time()

    for run_number in run_numbers:
        for scalar_config in scalar_configs:
            for timestep in range(args.tstart, args.tend, args.tjump):
                filename_a, filename_b = generate_file_paths(
                    os.path.abspath(args.folder),
                    scalar_config,
                    run_number,
                    timestep,
                )
                if not (os.path.exists(filename_a) and os.path.exists(filename_b)):
                    message = f"Missing files for scalar={scalar_config}, run={run_number}, timestep={timestep}"
                    if args.missing_policy == "error":
                        raise FileNotFoundError(message)
                    missing_files.append(message)
                    continue

                scalar_a, scalar_b = read_ensight(
                    filename_a,
                    filename_b,
                    args.nx,
                    ghost_width=pad_half_width,
                    npx=args.npx,
                )

                for filter_width in widths:
                    stride = resolve_stride(filter_width, args.stride, args.stride_scale)
                    rows, histograms, moments, histogram_moments, shape_features = extract_records_for_width(
                        scalar_a=scalar_a,
                        scalar_b=scalar_b,
                        nx=args.nx,
                        ghost_width=pad_half_width,
                        filter_width=filter_width,
                        stride=stride,
                        bins_a=bin_edges_a,
                        bins_b=bin_edges_b,
                        dz_ab=dz_ab,
                        z1_centers=z1_centers,
                        z2_centers=z2_centers,
                        z1z2=z1z2,
                        masks=shape_masks,
                        moment_bins=moment_bins,
                        moment_ranges=moment_ranges,
                        run_number=run_number,
                        scalar_config=scalar_config,
                        timestep=timestep,
                        starting_sample_id=next_sample_id,
                    )
                    batch_start, batch_end = raw_writer.append_batch(
                        histograms,
                        moments,
                        histogram_moments,
                        shape_features,
                    )
                    if batch_start != next_sample_id or batch_end != next_sample_id + len(rows):
                        raise RuntimeError("HDF5 sample-id alignment mismatch")
                    for row in rows:
                        row["bin_cap"] = args.bin_cap
                        row["shape_js_threshold"] = args.shape_js_threshold
                    metadata_rows.extend(rows)
                    next_sample_id += len(rows)

    raw_writer.close()
    raw_meta = pd.DataFrame(metadata_rows)
    if raw_meta.empty:
        raise RuntimeError("No samples were extracted; check the input selection")
    raw_meta.to_parquet(output_paths.raw_meta, index=False)

    selection_summary = write_selected_outputs(
        raw_meta=raw_meta,
        raw_h5_path=output_paths.raw_h5,
        final_h5_path=output_paths.final_h5,
        final_meta_path=output_paths.final_meta,
        compression=args.compression,
    )

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": args.dataset_name,
        "folder": os.path.abspath(args.folder),
        "output_dir": os.path.abspath(args.output_dir),
        "run_numbers": run_numbers,
        "scalar_configs": scalar_configs,
        "filter_widths": widths,
        "timesteps": {
            "start": args.tstart,
            "end": args.tend,
            "jump": args.tjump,
        },
        "raw_sample_count": int(len(raw_meta)),
        "selected_sample_count": int(selection_summary["selected_count"]),
        "histogram_shape": [int(args.bins), int(args.bins)],
        "moment_bins": list(moment_bins),
        "moment_ranges": [list(bounds) for bounds in moment_ranges],
        "shape_feature_columns": SHAPE_FEATURE_COLUMNS,
        "missing_files": missing_files,
        "paths": {
            "raw_h5": output_paths.raw_h5,
            "raw_meta": output_paths.raw_meta,
            "final_h5": output_paths.final_h5,
            "final_meta": output_paths.final_meta,
        },
        "elapsed_seconds": time.time() - start_time,
    }
    with open(output_paths.summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if args.drop_raw:
        os.remove(output_paths.raw_h5)
        os.remove(output_paths.raw_meta)

    elapsed = timedelta(seconds=(time.time() - start_time))
    print(f"Wrote dataset '{args.dataset_name}' in {elapsed}")


if __name__ == "__main__":
    main()