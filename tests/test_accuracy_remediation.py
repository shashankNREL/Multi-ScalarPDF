import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import h5py
import joblib
from sklearn.preprocessing import RobustScaler

import EnsightPDFHybridDataset as serial_sampler
import EnsightPDFHybridDatasetMPI as sampler
from dirichlet_mdn.bin_grid import bin_grid, load_bin_grid_from_hdf5
from dirichlet_mdn.data import load_metadata, validate_dataset
from dirichlet_mdn.dataset_diversity import variance_regime
from dirichlet_mdn.evaluate import _load_context
from dirichlet_mdn.losses import HistogramNLL
from dirichlet_mdn.metrics import joint_jsd, joint_l1
from dirichlet_mdn.model import DirichletMDN
from dirichlet_mdn.splits import (
    SplitResult,
    make_split,
    validate_split,
)
from dirichlet_mdn.train import TorchScaler


def _metadata() -> pd.DataFrame:
    rows = []
    sample_id = 0
    for run_id in range(6):
        for config in ("A", "B", "C"):
            rows.append({
                "scalar_config": config,
                "run_id": run_id,
                "timestep": 10 + run_id,
                "box_width": 64,
                "mean_a": 0.2,
                "var_a": 0.02,
                "mean_b": 0.3,
                "var_b": 0.03,
                "cov_ab": -0.01,
                "sample_id": sample_id,
                "local_index": sample_id,
                "hdf5_file": "shard.h5",
            })
            sample_id += 1
    return pd.DataFrame(rows)


class BinGeometryTests(unittest.TestCase):
    def test_clipped_cells_cover_exact_simplex(self):
        grid = bin_grid(num_bins=16, zst=0.1)
        self.assertAlmostEqual(float(grid.cell_area.sum()), 0.5, places=12)
        self.assertTrue(np.all(grid.cell_centroid_a[grid.simplex_mask] >= 0.0))
        self.assertTrue(np.all(grid.cell_centroid_b[grid.simplex_mask] >= 0.0))
        centroid_sum = (
            grid.cell_centroid_a[grid.simplex_mask]
            + grid.cell_centroid_b[grid.simplex_mask]
        )
        self.assertTrue(np.all(centroid_sum <= 1.0 + 1e-14))

    def test_boundary_intersections_are_not_center_masked_away(self):
        grid = bin_grid(num_bins=16, zst=0.1)
        nominal_a, nominal_b = np.meshgrid(
            grid.centers_a, grid.centers_b, indexing="ij",
        )
        recovered = grid.simplex_mask & (nominal_a + nominal_b > 1.0)
        self.assertTrue(recovered.any())
        self.assertTrue(np.all(grid.cell_area[recovered] > 0.0))


class ScalarValidationTests(unittest.TestCase):
    def test_serial_entry_point_uses_canonical_mpi_implementation(self):
        self.assertIs(serial_sampler.main, sampler.main)

    def test_roundoff_is_corrected_and_counted(self):
        z1, z2, counts = sampler.prepare_scalar_samples(
            np.array([-1e-8, 0.6]),
            np.array([0.2, 0.40000001]),
            tolerance=1e-6,
        )
        self.assertTrue(np.all(z1 >= 0.0))
        self.assertTrue(np.all(z1 + z2 <= 1.0))
        self.assertEqual(counts["scalar_pairs_adjusted"], 2)

    def test_materially_invalid_pairs_are_explicitly_dropped(self):
        z1, z2, counts = sampler.prepare_scalar_samples(
            np.array([0.2, 1.2]), np.array([0.3, 0.1]), policy="drop",
        )
        self.assertEqual(len(z1), 1)
        self.assertEqual(len(z2), 1)
        self.assertEqual(counts["scalar_pairs_invalid"], 1)
        self.assertEqual(counts["scalar_pairs_used"], 1)

    def test_histogram_refuses_silent_sample_loss(self):
        layout = sampler.get_histogram_layout(16)
        with self.assertRaises(ValueError):
            sampler.compute_histogram_pdf(
                np.array([0.2, 2.0]), np.array([0.3, 0.0]), layout,
            )


class SplitTests(unittest.TestCase):
    def test_default_split_keeps_each_run_in_one_partition(self):
        meta = _metadata()
        split = make_split(meta, seed=7)
        labels = np.full(len(meta), "unset", dtype=object)
        labels[split.train] = "train"
        labels[split.val] = "val"
        labels[split.test] = "test"
        for run_id in meta["run_id"].unique():
            self.assertEqual(len(set(labels[meta["run_id"] == run_id])), 1)

    def test_zero_ratio_is_respected(self):
        split = make_split(_metadata(), ratios=(0.8, 0.2, 0.0))
        self.assertEqual(len(split.test), 0)

    def test_holdout_rejects_shared_physical_runs(self):
        meta = _metadata()
        with self.assertRaisesRegex(ValueError, "pure configuration holdout"):
            make_split(meta, holdout_config="C")

    def test_holdout_test_is_pure_and_run_isolated(self):
        meta = _metadata()
        meta.loc[meta["scalar_config"] == "C", "run_id"] += 100
        split = make_split(meta, holdout_config="C")
        self.assertEqual(set(meta.iloc[split.test]["scalar_config"]), {"C"})
        self.assertNotIn("C", set(meta.iloc[split.train]["scalar_config"]))
        self.assertNotIn("C", set(meta.iloc[split.val]["scalar_config"]))
        test_runs = set(meta.iloc[split.test]["run_id"])
        self.assertFalse(test_runs & set(meta.iloc[split.train]["run_id"]))
        self.assertFalse(test_runs & set(meta.iloc[split.val]["run_id"]))

    def test_fingerprint_mismatch_is_rejected(self):
        meta = _metadata()
        split = make_split(meta)
        bad_description = dict(split.description)
        bad_description["dataset_fingerprint"] = "wrong"
        bad = SplitResult(split.train, split.val, split.test, bad_description)
        with self.assertRaises(ValueError):
            validate_split(bad, meta)


class ModelAndLossTests(unittest.TestCase):
    def test_model_rejects_boundary_singular_concentrations(self):
        with self.assertRaises(ValueError):
            DirichletMDN(4, alpha_min=0.5)
        _, alpha = DirichletMDN(4, alpha_min=1.0)(torch.zeros(2, 4))
        self.assertTrue(torch.all(alpha >= 1.0))
        legacy = DirichletMDN(
            4, alpha_min=0.5, allow_alpha_below_one=True,
        )
        _, legacy_alpha = legacy(torch.zeros(1, 4))
        self.assertTrue(torch.all(legacy_alpha >= 0.5))

    def test_nll_normalizes_targets_and_rejects_outside_mass(self):
        grid = bin_grid(num_bins=8, zst=0.1)
        nll = HistogramNLL(grid)
        pi = torch.ones(1, 1)
        alpha = torch.tensor([[[2.0, 3.0, 4.0]]])
        target = torch.zeros(1, 8, 8)
        target[0][torch.from_numpy(grid.simplex_mask)] = 1.0
        first = nll(pi, alpha, target)
        second = nll(pi, alpha, 7.0 * target)
        self.assertTrue(torch.allclose(first, second))
        outside = target.clone()
        outside[0][torch.from_numpy(~grid.simplex_mask)] = 1.0
        with self.assertRaises(ValueError):
            nll(pi, alpha, outside)

    def test_histogram_metrics_have_exact_limiting_values(self):
        p = np.array([[[1.0, 0.0], [0.0, 0.0]]])
        q = np.array([[[0.0, 1.0], [0.0, 0.0]]])
        mask = np.array([[True, True], [False, False]])
        self.assertEqual(float(joint_l1(p, q, mask)[0]), 2.0)
        self.assertEqual(float(joint_jsd(p, q)[0]), 1.0)


class ArtifactAndDiagnosticTests(unittest.TestCase):
    def test_effective_input_transform_round_trip(self):
        scaler = RobustScaler().fit(np.array([[1.0, 2.0], [1.0, 4.0]]))
        original = TorchScaler(scaler)
        restored = TorchScaler.from_artifact(original.to_artifact())
        values = torch.tensor([[1.0, 3.0]])
        self.assertTrue(torch.equal(original(values), restored(values)))

    def test_variance_regime_labels_match_physics(self):
        frame = pd.DataFrame({
            "scalar_config": ["case", "case"],
            "mean_a": [0.5, 0.5],
            "var_a": [0.0, 0.25],
        })
        result = variance_regime(frame)["case"]
        self.assertEqual(result["frac_well_mixed_lt_0_1"], 0.5)
        self.assertEqual(result["frac_highly_segregated_gt_0_8"], 0.5)

    def test_legacy_evaluation_requires_opt_in_and_loads_old_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            grid = bin_grid(num_bins=8, zst=0.1)
            shard = root / "legacy.h5"
            legacy_area = np.outer(
                np.diff(grid.edges_a), np.diff(grid.edges_b),
            )
            center_a, center_b = np.meshgrid(
                grid.centers_a, grid.centers_b, indexing="ij",
            )
            legacy_mask = center_a + center_b <= 1.0 + 1e-12
            with h5py.File(shard, "w") as handle:
                bins = handle.create_group("bins")
                bins.create_dataset("centers_a", data=grid.centers_a)
                bins.create_dataset("centers_b", data=grid.centers_b)
                bins.create_dataset("edges_a", data=grid.edges_a)
                bins.create_dataset("edges_b", data=grid.edges_b)
                bins.create_dataset("cell_area", data=legacy_area)
                bins.create_dataset(
                    "simplex_mask", data=legacy_mask.astype(np.uint8),
                )
            parquet = root / "metadata.parquet"
            pd.DataFrame([{
                "scalar_config": "A", "run_id": 0, "timestep": 1,
                "box_width": 2, "mean_a": 0.2, "var_a": 0.01,
                "mean_b": 0.3, "var_b": 0.01, "cov_ab": 0.0,
                "sample_id": 0, "local_index": 0, "hdf5_file": shard.name,
            }]).to_parquet(parquet, index=False)
            config = {
                "parquet": str(parquet), "hdf5_dir": str(root),
                "input_moments": 4, "K": 1, "hidden": 8,
                "alpha_min": 1e-3, "alpha_clip": 1e3,
                "device": "cpu", "num_bins": 8, "zst": 0.1,
                "uniform_bins": False,
            }
            (root / "manifest.json").write_text(json.dumps({"config": config}))
            model = DirichletMDN(
                4, K=1, hidden=8, alpha_min=1e-3,
                allow_alpha_below_one=True,
            )
            torch.save(
                {"model_state_dict": model.state_dict()},
                root / "model_best.pt",
            )
            joblib.dump(
                RobustScaler().fit(np.array([[0.0] * 4, [1.0] * 4])),
                root / "scaler.pkl",
            )
            with self.assertRaisesRegex(ValueError, "allow-legacy-artifacts"):
                _load_context(root)
            with self.assertWarnsRegex(RuntimeWarning, "not directly comparable"):
                context = _load_context(root, allow_legacy_artifacts=True)
            self.assertTrue(context.legacy_artifacts)
            self.assertEqual(context.model.alpha_min, 1e-3)
            self.assertTrue(
                np.array_equal(context.grid.cell_centroid_a, center_a)
            )


class RankProvenanceTests(unittest.TestCase):
    def test_phase1_writer_bounds_shards_and_preserves_identity(self):
        layout = sampler.get_histogram_layout(4)
        with tempfile.TemporaryDirectory() as tmp:
            writer = sampler.StreamingRankWriter(
                tmp, "phase", layout, shard_size=2, compression=None,
            )
            for index in range(5):
                writer.add(
                    {
                        "scalar_config": "A", "run_id": index, "timestep": 1,
                        "box_width": 2, "stride": 1, "center_i": 0,
                        "center_j": 0, "center_k": 0, "n_dns_cells": 1,
                    },
                    np.full((4, 4), 1.0 / 16.0, dtype=np.float32),
                )
            self.assertEqual(writer.close(), 5)
            metadata = pd.read_parquet(Path(tmp) / "phase_metadata.parquet")
            self.assertEqual(metadata["sample_id"].tolist(), list(range(5)))
            self.assertEqual(len(list(Path(tmp).glob("phase_*.h5"))), 3)
            for shard_name, rows in metadata.groupby("hdf5_file", sort=False):
                with h5py.File(Path(tmp) / shard_name, "r") as handle:
                    self.assertEqual(
                        handle["data/sample_id"][:].tolist(),
                        rows["sample_id"].tolist(),
                    )
                    self.assertEqual(
                        handle["data/local_index"][:].tolist(),
                        list(range(len(rows))),
                    )

    def test_rank_output_validation_checks_work_unit_coverage(self):
        signature = "abc"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rank_dirs = []
            for rank, units in enumerate(([[0, "A"]], [[1, "A"]])):
                rank_dir = root / f"rank_{rank:04d}"
                rank_dir.mkdir()
                rank_dirs.append(str(rank_dir))
                manifest = {
                    "format_version": sampler.DATASET_FORMAT_VERSION,
                    "phase1_signature": signature,
                    "mpi_rank": rank,
                    "lossless_phase1": True,
                    "num_retained": 0,
                    "counts": {},
                    "rank_work_units": units,
                }
                (rank_dir / "phase_manifest.json").write_text(json.dumps(manifest))

            sampler.validate_rank_outputs(
                rank_dirs, "phase", signature,
                expected_size=2, expected_work_units=[(0, "A"), (1, "A")],
            )
            stale = Path(rank_dirs[0]) / "phase_0000.h5"
            stale.touch()
            with self.assertRaises(RuntimeError):
                sampler.validate_rank_outputs(
                    rank_dirs, "phase", signature,
                    expected_size=2,
                    expected_work_units=[(0, "A"), (1, "A")],
                )
            stale.unlink()
            with self.assertRaises(RuntimeError):
                sampler.validate_rank_outputs(
                    rank_dirs, "phase", signature,
                    expected_size=2, expected_work_units=[(0, "A"), (2, "A")],
                )

    def test_exact_shard_cleanup_does_not_delete_prefix_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in (
                "foo_0000.h5",
                "foo_checkpoint_0000.h5",
                "foo_v2_0000.h5",
            ):
                (root / name).touch()
            sampler.remove_existing_shards(str(root), "foo")
            self.assertFalse((root / "foo_0000.h5").exists())
            self.assertTrue((root / "foo_checkpoint_0000.h5").exists())
            self.assertTrue((root / "foo_v2_0000.h5").exists())

    def test_seeded_candidate_order_is_independent_of_rank_partition(self):
        records = []
        for index in range(4):
            records.append({
                "sample_id": index,
                "shard_id": 0,
                "local_index": 0,
                "hdf5_file": f"record_{index}.h5",
                "scalar_config": "A",
                "run_id": index,
                "timestep": 10,
                "box_width": 64,
                "stride": 64,
                "center_i": 0,
                "center_j": 0,
                "center_k": 0,
                "n_dns_cells": 1,
                "mean_a": 0.2,
                "var_a": 0.0,
                "mean_b": 0.3,
                "var_b": 0.0,
                "cov_ab": 0.0,
            })

        def write_partition(root: Path, partitions):
            rank_dirs = []
            for rank, indices in enumerate(partitions):
                rank_dir = root / f"rank_{rank:04d}"
                rank_dir.mkdir(parents=True)
                rank_dirs.append(str(rank_dir))
                rank_records = []
                for index in indices:
                    record = dict(records[index])
                    record["local_index"] = 0
                    rank_records.append(record)
                    with h5py.File(rank_dir / record["hdf5_file"], "w") as handle:
                        handle.attrs["format_version"] = sampler.DATASET_FORMAT_VERSION
                        bins = handle.create_group("bins")
                        bins.create_dataset("cell_centroid_a", data=np.zeros((4, 4)))
                        bins.create_dataset("cell_centroid_b", data=np.zeros((4, 4)))
                        data = handle.create_group("data")
                        data.create_dataset(
                            "histograms",
                            data=np.full((1, 4, 4), 1.0 / 16.0),
                        )
                        data.create_dataset("sample_id", data=np.array([index]))
                        data.create_dataset("local_index", data=np.array([0]))
                pd.DataFrame(rank_records).to_parquet(
                    rank_dir / "phase_metadata.parquet", index=False,
                )
            return rank_dirs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            order_one = [
                metadata["run_id"]
                for metadata, _ in sampler.iter_rank_candidates(
                    write_partition(root / "one", [[0, 1, 2, 3]]),
                    "phase", selection_seed=11, expected_num_bins=4,
                    sort_chunk_rows=1,
                )
            ]
            order_two = [
                metadata["run_id"]
                for metadata, _ in sampler.iter_rank_candidates(
                    write_partition(root / "two", [[0, 2], [1, 3]]),
                    "phase", selection_seed=11, expected_num_bins=4,
                    sort_chunk_rows=1,
                )
            ]
            self.assertEqual(order_one, order_two)


class DatasetValidationTests(unittest.TestCase):
    def test_exhaustive_validator_accepts_consistent_synthetic_dataset(self):
        grid = bin_grid(num_bins=8, zst=0.1)
        hist = np.zeros((8, 8), dtype=np.float32)
        cell = tuple(np.argwhere(grid.simplex_mask)[5])
        hist[cell] = 1.0
        mean_a = float(grid.cell_centroid_a[cell])
        mean_b = float(grid.cell_centroid_b[cell])
        row = {
            "scalar_config": "A",
            "run_id": 0,
            "timestep": 1,
            "box_width": 2,
            "mean_a": mean_a,
            "var_a": 0.0,
            "mean_b": mean_b,
            "var_b": 0.0,
            "cov_ab": 0.0,
            "hist_mean_a": mean_a,
            "hist_var_a": 0.0,
            "hist_mean_b": mean_b,
            "hist_var_b": 0.0,
            "hist_cov_ab": 0.0,
            "moment_abs_err_max": 0.0,
            "sample_id": 0,
            "local_index": 0,
            "hdf5_file": "synthetic_0000.h5",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parquet = root / "metadata.parquet"
            pd.DataFrame([row]).to_parquet(parquet, index=False)
            with h5py.File(root / row["hdf5_file"], "w") as handle:
                handle.attrs["format_version"] = sampler.DATASET_FORMAT_VERSION
                bins = handle.create_group("bins")
                bins.create_dataset("centers_a", data=grid.centers_a)
                bins.create_dataset("centers_b", data=grid.centers_b)
                bins.create_dataset("edges_a", data=grid.edges_a)
                bins.create_dataset("edges_b", data=grid.edges_b)
                bins.create_dataset("cell_area", data=grid.cell_area)
                bins.create_dataset("cell_centroid_a", data=grid.cell_centroid_a)
                bins.create_dataset("cell_centroid_b", data=grid.cell_centroid_b)
                bins.create_dataset(
                    "simplex_mask", data=grid.simplex_mask.astype(np.uint8),
                )
                data = handle.create_group("data")
                data.create_dataset("histograms", data=hist[None, ...])
                data.create_dataset("sample_id", data=np.array([0]))
                data.create_dataset("local_index", data=np.array([0]))
            result = validate_dataset(
                str(parquet), str(root), num_bins=8, zst=0.1,
            )
            self.assertEqual(result["records_checked"], 1)
            altered = np.zeros_like(hist)
            altered[tuple(np.argwhere(grid.simplex_mask)[-1])] = 1.0
            with h5py.File(root / row["hdf5_file"], "r+") as handle:
                handle["data/histograms"][0] = altered
            with self.assertRaisesRegex(
                ValueError, "stored histogram moments do not match",
            ):
                validate_dataset(
                    str(parquet), str(root), num_bins=8, zst=0.1,
                )
            with h5py.File(root / row["hdf5_file"], "r+") as handle:
                handle["data/histograms"][0] = hist
                del handle["data/local_index"]
            with self.assertRaisesRegex(KeyError, "row identity arrays"):
                validate_dataset(
                    str(parquet), str(root), num_bins=8, zst=0.1,
                )
            with h5py.File(root / row["hdf5_file"], "r+") as handle:
                handle["data"].create_dataset(
                    "local_index", data=np.array([0]),
                )

            bad_row = dict(row)
            bad_row["hist_mean_a"] = np.nan
            pd.DataFrame([bad_row]).to_parquet(parquet, index=False)
            with self.assertRaisesRegex(ValueError, "must be finite"):
                validate_dataset(
                    str(parquet), str(root), num_bins=8, zst=0.1,
                )

            bad_row = dict(row)
            bad_row["hdf5_file"] = None
            pd.DataFrame([bad_row]).to_parquet(parquet, index=False)
            with self.assertRaisesRegex(ValueError, "must not be null"):
                load_metadata(str(parquet))

    def test_legacy_grid_loader_uses_stored_center_geometry(self):
        grid = bin_grid(num_bins=8, zst=0.1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.h5"
            with h5py.File(path, "w") as handle:
                bins = handle.create_group("bins")
                bins.create_dataset("centers_a", data=grid.centers_a)
                bins.create_dataset("centers_b", data=grid.centers_b)
                bins.create_dataset("edges_a", data=grid.edges_a)
                bins.create_dataset("edges_b", data=grid.edges_b)
                bins.create_dataset("cell_area", data=grid.cell_area)
                bins.create_dataset(
                    "simplex_mask", data=grid.simplex_mask.astype(np.uint8),
                )
            loaded = load_bin_grid_from_hdf5(str(path), zst=0.1)
            centers_a, centers_b = np.meshgrid(
                grid.centers_a, grid.centers_b, indexing="ij",
            )
            self.assertTrue(np.array_equal(loaded.cell_centroid_a, centers_a))
            self.assertTrue(np.array_equal(loaded.cell_centroid_b, centers_b))


if __name__ == "__main__":
    unittest.main()
