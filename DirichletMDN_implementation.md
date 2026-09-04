# DirichletMDN — Implementation Log

## Purpose

This file is the running record of the implementation of `dirichlet_mdn/` (PyTorch training, validation, and verification of a Dirichlet Mixture Density Network for three-stream subgrid PDF closure). It is updated incrementally as work happens, not retroactively.

> **Current-status note (2026-09-04):** early sections preserve the original
> implementation history. The later **Accuracy remediation** section is
> authoritative wherever behavior has changed.

Companion documents:

- `DirichletMDN_proposal.md` — the scientific methodology and motivation.
- `DirichletMDN_implementation_plan.md` — the approved implementation plan (the design intent).
- `EnsightPDFHybridDataset.md` — the data-pipeline counterpart this file is modeled after.

When this log conflicts with the plan, this log wins: the plan documents intent, this file documents reality. Update the plan to match when divergence is intentional.

## Environment

Conda environment: `ct-env`.

Versions confirmed working at the first successful 3-epoch smoke training run (2026-06-16):

```text
torch==2.11.0
pyarrow==24.0.0
tensorboard==2.20.0
h5py==3.14.0
numpy==2.4.4
pandas==3.0.2
scipy==1.17.1
scikit-learn==1.8.0
matplotlib==3.10.8
joblib==1.5.3
```

`tabulate` is NOT installed and not needed — the evaluator writes Markdown tables by hand.

`pyarrow` and `tensorboard` were `pip install`ed into `ct-env` during this implementation; all other packages were already present.

## Implementation summary

One paragraph per module. Updated as each module is committed.

### `bin_grid.py`
Stand-alone reproduction of the non-uniform Z1-Z2 bin grid used by `EnsightPDFHybridDataset.py:123-186`. Exposes `bin_grid(num_bins, zst, uniform) -> BinGrid` and `validate_against_hdf5(...)`. Validated to **zero numerical difference** against `hybrid_dataset_all/all_cases_w64_w128_checkpoint_0000.h5`.

### `data.py`
`HybridPDFDataset` — schema-tolerant parquet+HDF5 reader (accepts older `run_id`/`box_width` or newer `run_number`/`filter_width` column names). Multi-shard: opens each unique shard on first access and caches the handle. Histogram-sum sanity check (`|sum-1| < 1e-4`) runs once per shard. `__getitem__` returns `{"moments", "histogram", "meta"}` with `moments` of length 4 or 5 according to `input_moments`. Class method `fit_input_scaler` fits an `sklearn.preprocessing.RobustScaler` on the train-split moment matrix — same persistence pattern as `MLPDF.py:71`.

### `model.py`
`DirichletMDN` — three-layer MLP trunk (`Linear→SiLU` × 3, hidden 128→256→256), two heads (softmax `π`, softplus `α`+`alpha_min`, clipped above by `alpha_clip`). `n_in` configurable (4 or 5), `K` configurable. Default `K=8` gives 107k params for `n_in=4` and 108k for `n_in=5`.

### `losses.py`
Three components plus a combined wrapper.

- `dirichlet_logpdf(z, alpha)` — closed-form Dirichlet log-density on the 2-simplex (3-vector `z`, normalized after clamping to ≥ `eps`).
- `mixture_logpdf(z, pi, alpha)` — `(B, M)` log-density of the K-component mixture at M simplex points, via explicit `(1, M, 1, 3)` × `(B, 1, K, 3)` broadcasting and `logsumexp` over K. (See Deviation 2 for the broadcasting layout fix.)
- `HistogramNLL` — `nn.Module` that precomputes the in-simplex centers and cell areas as registered buffers, then takes `loss = -Σ_ij p_ij log q_ij` where `q_ij = density × area`, renormalized over the in-simplex cells. Operates entirely in log-space (no exp before `logsumexp`).
- `mixture_moments(pi, alpha)` — closed-form E, Var, Cov12 from the Perry-Mueller / Dirichlet formulas.
- `stack_predicted_moments(pi, alpha, input_moments)` — packs predictions into either `[E1, V1, E2, V2]` or `[E1, V1, E2, V2, Cov12]`.
- `moment_loss` — MSE in that order.
- `entropy_reg` — mean `-Σ_k π_k log π_k`.
- `combined_loss` returns a `LossOutputs` dataclass with `total`, `nll`, `mom`, `entropy` so the trainer can log each component separately.

Verified properties: `moment_loss` is exactly **0** when fed its own predicted moments (K ∈ {1, 4, 8}, n_in ∈ {4, 5}); NLL is exactly invariant under K-axis component permutation; gradients are finite at initialization.

### `metrics.py`
`predict_histograms(pi, alpha, grid)` renders the mixture as a probability histogram on the grid using the same renormalization the NLL uses. Per-record metrics: `joint_l1` (in-simplex masked), `joint_jsd` (log-base-2, port of `MLPDF.py:79`), `marginal_jsd` for `axis=0` or `1`, `mixture_active_count`, `moment_recovery_error`. Group helpers: `per_config_breakdown`, `per_timestep_breakdown`.

### `splits.py`
Leak-free splits at the `(scalar_config, run_id)` level. Default `(0.7, 0.15, 0.15)`. `--holdout-config` puts one whole config into test. `--split-fallback timestep` (default) handles configs with fewer than 3 unique `run_id`s by contiguous timestep blocks (early→train, mid→val, late→test). See Deviation 1.

### `verify.py`
Standalone PyTest-style CLI exposing five checks (`--check-bins`, `--check-loss-gradient`, `--check-moment-loss-zero-on-true-params`, `--check-permutation-invariance`, `--check-k1-fits-analytical-dirichlet`). The K=1 check builds a synthetic Dirichlet histogram from a known `(2, 3, 5)` truth and recovers it to machine precision (Adam, 1500 steps, method-of-moments init). All five pass on this dataset.

### `train.py`
Full training CLI. AdamW (`weight_decay=1e-5`), cosine LR schedule over `epochs`, gradient clipping at `--grad-clip` (default 5.0), early-stopping on val NLL with `--patience` (default 30). Writes `manifest.json`, `splits.json`, `scaler.pkl`, `model_best.pt`, `model_last.pt`, `tb/` (`torch.utils.tensorboard.SummaryWriter`), and `train.log`. Device resolution: `auto` picks CUDA → MPS → CPU. A `TorchScaler` wraps the sklearn `RobustScaler` so per-batch scaling happens on-device without a numpy round-trip.

### `evaluate.py`
Loads a run dir, runs predictions on the test split, produces `metrics_per_record.csv`, `metrics_per_config.{csv,md}`, `metrics_per_timestep.csv`, `eval_summary.json`, and a `plots/` directory with `pdf_album.pdf` (~3 DNS-vs-MDN contour plots per config), `l1_vs_time.pdf`, `jsd_vs_time.pdf`, `alpha_diagnostics.pdf`, `moment_recovery.pdf`. Markdown tables are written by hand to avoid a `tabulate` dependency (see Deviation 3).

### `plotting.py`
`plot_pdf_comparison` (twin contour with the `MLPDF.py:283-300` simplex polygon overlay), `plot_metric_vs_timestep`, `plot_alpha_diagnostics` (active-component histogram + `α_0` vs `var_a` scatter), `plot_moment_recovery` (predicted-vs-target scatter with `y=x` reference). Headless `matplotlib.use("Agg")`. PDF album written via `PdfPages`.

## Important implementation details

### 1. Simplex-renormalized categorical NLL (`losses.HistogramNLL`)

The per-cell predicted mass is `q_ij = density(center_ij) * cell_area_ij`. Two corrections are needed before computing `-Σ p log q`:

- **Out-of-simplex cells** carry density only because the Dirichlet has support on the full simplex; the stored histograms zero them out. We mask them out of the log-density evaluation and ignore them in the NLL sum.
- **Renormalization** over the surviving in-simplex cells. Without this step, `Σ q_ij` over the in-simplex cells does not equal 1 for two reasons: (i) the simplex-truncated Dirichlet integrates to 1 only when the full continuous measure is used, not the discrete midpoint rule, and (ii) bin centers near the simplex boundary are systematically biased. Renormalizing with `log_norm = logsumexp(log_unnorm)` makes the loss the correct categorical cross-entropy.

Everything happens in log-space: `log_unnorm = log_density + log(cell_area)`, then `log_q = log_unnorm - logsumexp(log_unnorm)`. No `exp` of the unnormalized log density.

### 2. `mixture_logpdf` broadcasting (`losses.mixture_logpdf`)

First implementation used `z.unsqueeze(-2)` to inject a component axis. This collapses incorrectly when both batch (B) and cell (M) axes are >1 — broadcasting tries to align M against B and errors at `(8,) vs (3642,)`. Fixed to explicit shapes:

- `z` viewed as `(1, M, 1, 3)`
- `alpha` viewed as `(B, 1, K, 3)`
- `dirichlet_logpdf` then returns `(B, M, K)`
- `log_pi` unsqueezed to `(B, 1, K)`
- `logsumexp` over K yields `(B, M)`

### 3. K=1 verification needs method-of-moments init (`verify.check_k1_fits_analytical_dirichlet`)

Initializing the K=1 fit at `softplus(0) ≈ 0.69` per component (α_0 ≈ 2) is too far from the truth (α_0 = 10) for Adam at lr=0.05 to converge within 600 steps — final relative error was ~6%. Method-of-moments initialization (estimating α_0 from `E[Z_1] (1 - E[Z_1]) / Var[Z_1] - 1` and α_i from α_0 * E[Z_i]) brings the optimizer to within machine precision in 1500 steps. The check is now a true convergence test, not a warmup test.

### 4. `RobustScaler` zero-scale guard (`train.TorchScaler`)

In the preliminary dataset, `var_b` and `cov_ab` are essentially zero for many records (Z2 is sparse). `RobustScaler.scale_` for those columns lands at ~4e-7, which makes the scaled inputs numerically unstable (a real zero divided by 4e-7 is still zero, but any perturbation explodes). `TorchScaler` replaces any `scale_` below `1e-6` with `1.0` before dividing, so degenerate columns become a passthrough rather than an amplifier.

### 5. Split fallback for single-run configs (`splits.make_split`)

See Deviation 1.

### 6. Multi-shard HDF5 handles, single-process DataLoader

The `HybridPDFDataset` opens shard handles lazily and caches them. Because `h5py.File` handles are not pickle-safe, `num_workers=0` is the default (and only tested) configuration. When the full dataset spans many shards and disk I/O becomes the bottleneck, switch to `num_workers > 0` with a worker-init that opens the handles in each worker — not needed for the current 3,931-record file.

## Deviations from the plan

Numbered subsections, each with four fields. Every divergence from `DirichletMDN_implementation_plan.md` goes here, however small.

### Deviation 1: timestep-fallback split mode added to `splits.py`

- **Planned:** split train/val/test strictly by `(scalar_config, run_id)`. Configs with fewer than 3 distinct run_ids fall back to "all rows in train" with a warning.
- **Implemented:** an additional `fallback="timestep"` mode (default) that splits a config's rows by contiguous timestep blocks (early → train, mid → val, late → test) when that config has fewer than 3 distinct run_ids. `fallback="train_only"` reproduces the originally planned behavior.
- **Reason:** every `scalar_config` in the preliminary checkpoint dataset has only one `run_id`, so the strict by-run_id split produced an empty val and test set, blocking any end-to-end pipeline verification. Timestep-contiguous splits are still leakage-bounded (val/test are *unseen* timesteps of the same evolution rather than interleaved or shuffled samples) and let us exercise the loop now. The full dataset (once `EnsightPDFHybridDataset.py` finishes) will have many run_ids per config and the by-run_id mode will activate automatically — no behavior change required at that point.
- **Consequence:** with the current dataset, val/test metrics measure *out-of-time* generalization within one DNS run, not the stronger *out-of-run* generalization the plan intends. The split-mode is recorded in `splits.json` so this caveat travels with each run's artifacts. Reruns on the full dataset should not show the `by_timestep_fallback(...)` mode in any per-config entry.

### Deviation 2: explicit broadcasting in `mixture_logpdf`

- **Planned:** evaluate the K-component mixture log-density via implicit broadcasting (`z.unsqueeze(-2)` to inject a component axis).
- **Implemented:** explicit `(1, M, 1, 3)` × `(B, 1, K, 3)` reshape before `dirichlet_logpdf`, returning a clean `(B, M, K)` tensor that is `logsumexp`'d over K with `log_pi` viewed as `(B, 1, K)`.
- **Reason:** the implicit form collapsed when both batch size and cell count exceed 1 (broadcasting tried to align M against B and errored).
- **Consequence:** no behavior change for correctly broadcasting inputs; the loss now works for arbitrary `(B, M, K)` combinations including the `(B=64, M=3642, K=8)` workload.

### Deviation 3: hand-written Markdown table writer in `evaluate.py`

- **Planned:** `pandas.DataFrame.to_markdown` for the per-config table.
- **Implemented:** custom `_write_markdown_table` that emits the same pipe-delimited table without calling `to_markdown`.
- **Reason:** `pandas.to_markdown` requires the optional `tabulate` package which is not installed in `ct-env`. Avoided one more pip install for a tiny formatting helper.
- **Consequence:** none — output format is identical.

### Deviation 4: zero-scale guard in `TorchScaler`

- **Planned:** apply the fitted `RobustScaler` as-is.
- **Implemented:** any `scale_` component below `1e-6` is replaced with `1.0` before division.
- **Reason:** in the preliminary dataset, the IQR of `var_b` and `cov_ab` is essentially zero (Z2 is sparse), so the raw scale would amplify floating-point noise by ~6 orders of magnitude.
- **Consequence:** for degenerate input dimensions, the scaler becomes a passthrough; the model sees the raw column. This is the correct fallback when the column has no variance to scale against. Will be a no-op on the full dataset once `var_b` distributes properly.

## Exact output schema

Artifacts produced under `<output-dir>/<timestamp>_<tag>/`.

### `manifest.json`
JSON with three top-level keys:
- `config`: dict of every `TrainConfig` field (parquet, hdf5_dir, input_moments, K, hidden, epochs, batch_size, lr, weight_decay, lambda_mom, lambda_ent, alpha_min, alpha_clip, seed, output_dir, tag, device, holdout_config, split_fallback, patience, grad_clip, num_workers, num_bins, zst, uniform_bins).
- `extras`: dataset_summary (n_rows, scalar_configs, per_config_counts, box_widths, run_ids, timesteps), split_totals (train/val/test counts), best_val_nll, num_simplex_cells, deviation_note.
- `created_utc`: ISO-8601 timestamp.

### `scaler.pkl`
`joblib`-pickled `sklearn.preprocessing.RobustScaler` fit on the train-split moment matrix. Loaded by `evaluate.py` and wrapped in a `TorchScaler` for inference.

### `model_best.pt` and `model_last.pt`
Torch checkpoint dicts with keys: `epoch`, `model_state_dict`, `val_metrics` (dict of nll/mom/ent/total), `train_metrics`. `model_last.pt` additionally has `optimizer_state_dict` for resumability.

### `splits.json`
Three arrays of parquet row indices: `train`, `val`, `test`. Plus a `description` dict carrying seed, ratios, holdout_config, fallback mode, per-config partition (run_ids or timesteps per split), and the overall totals dict.

### `eval/metrics_per_record.csv`
One row per test record. Columns:
- `scalar_config`, `run_id`, `timestep`, `box_width`, `sample_id` (meta).
- `l1`, `jsd`, `jsd_marg_z1`, `jsd_marg_z2` (PDF metrics).
- `active_components`, `alpha0_effective` (mixture diagnostics).
- `mom_abs_err_<name>` for each of the conditioning moments (4 or 5).

### `eval/metrics_per_config.csv` and `eval/metrics_per_config.md`
`per_config_breakdown` output: one row per `scalar_config`, columns `<metric>_mean`, `<metric>_std`, `<metric>_count` for every numeric metric column. `.md` is a Markdown rendering of the same data.

### `eval/metrics_per_timestep.csv`
Same shape as `metrics_per_config.csv` but grouped by `(scalar_config, timestep)`.

### `eval/eval_summary.json`
Headline summary: `n_test`, `mean_l1`, `mean_jsd`, `mean_active_components`, `per_config_mean_l1`, `per_config_mean_jsd`.

### `eval/plots/`
- `pdf_album.pdf` — multi-page; up to `--album-per-cfg` (default 3) DNS-vs-MDN twin contour plots per `scalar_config`, spanning timesteps. Simplex polygon overlay.
- `l1_vs_time.pdf` — mean L1 vs timestep, one line per config.
- `jsd_vs_time.pdf` — mean JSD vs timestep, one line per config.
- `alpha_diagnostics.pdf` — active-component histogram + `α_0` vs `var_a` log-y scatter.
- `moment_recovery.pdf` — pred-vs-target scatter per input moment, with `y = x` reference line.

### `tb/`
TensorBoard event files written by `torch.utils.tensorboard.SummaryWriter`. Scalars: `train/{nll,mom,ent,total}`, `val/{nll,mom,ent,total}`, `opt/lr`.

### `train.log`
Plain-text per-epoch log with timestamps. Mirrors stdout.

## CLI arguments

### `python -m dirichlet_mdn.train`

| Flag | Default | Notes |
|---|---|---|
| `--parquet` | required | Path to parquet metadata file |
| `--hdf5-dir` | required | Directory holding HDF5 shards referenced by parquet `hdf5_file` |
| `--input-moments` | 4 | 4 (LES-deployable, no covariance) or 5 (with covariance) |
| `--K` | 8 | Mixture components |
| `--hidden` | 256 | Trunk hidden width |
| `--epochs` | 200 | Max epochs |
| `--batch-size` | 64 | |
| `--lr` | 3e-4 | AdamW learning rate |
| `--weight-decay` | 1e-5 | AdamW weight decay |
| `--lambda-mom` | 0.5 | Weight on moment recovery loss |
| `--lambda-ent` | 1e-3 | Weight on entropy reg (subtracted: entropy maximized) |
| `--alpha-min` | 1e-3 | Floor on Dirichlet concentrations |
| `--alpha-clip` | 1e3 | Ceiling on Dirichlet concentrations |
| `--seed` | 0 | Deterministic split + torch seed |
| `--output-dir` | dirichlet_mdn_runs | Parent of timestamped run dirs |
| `--tag` | baseline | Appended to `<timestamp>_<tag>` |
| `--device` | auto | auto / cpu / mps / cuda |
| `--holdout-config` | None | If set, this scalar_config is the full test set |
| `--split-fallback` | timestep | timestep / train_only — behavior when <3 run_ids in a config |
| `--patience` | 30 | 0 disables early stopping |
| `--grad-clip` | 5.0 | Gradient norm clip |
| `--num-workers` | 0 | DataLoader workers (only 0 is tested) |
| `--num-bins` | 64 | Must match the dataset's stored bin count |
| `--zst` | 0.1 | Non-uniform bin transition point |
| `--uniform-bins` | False | Use uniform binning instead |

### `python -m dirichlet_mdn.evaluate`

| Flag | Default | Notes |
|---|---|---|
| `--run-dir` | required | Directory produced by `train.py` |
| `--device` | inherits from run | Override for evaluation device |
| `--album-per-cfg` | 3 | DNS-vs-MDN contour plots per scalar_config in `pdf_album.pdf` |

### `python -m dirichlet_mdn.verify`

| Flag | Default | Notes |
|---|---|---|
| `--parquet` | None | Required for shard-bound checks |
| `--hdf5-dir` | None | Required for shard-bound checks |
| `--check-bins` | off | reconstructed bin grid == HDF5 |
| `--check-loss-gradient` | off | no NaN/Inf at init |
| `--check-moment-loss-zero-on-true-params` | off | algebra correctness |
| `--check-permutation-invariance` | off | NLL invariant under K-axis permutation |
| `--check-k1-fits-analytical-dirichlet` | off | K=1 recovers known Dirichlet |
| `--check-all` | False | If no individual checks selected, all five run by default |

## Example commands

All run from the repo root with `conda activate ct-env` (set `PYTHONPATH=.` if you have not installed the package).

### Sanity (under 30 s)

```bash
PYTHONPATH=. python -m dirichlet_mdn.verify --check-all \
  --parquet hybrid_dataset_all/all_cases_w64_w128_checkpoint_metadata.parquet \
  --hdf5-dir hybrid_dataset_all
```

Expected output: `5/5 checks passed.`

### Smoke training (a few minutes on CPU)

```bash
PYTHONPATH=. python -m dirichlet_mdn.train \
  --parquet hybrid_dataset_all/all_cases_w64_w128_checkpoint_metadata.parquet \
  --hdf5-dir hybrid_dataset_all \
  --input-moments 4 --K 4 --epochs 3 --batch-size 64 \
  --tag smoke_M4_K4 --device cpu
```

Expected: training NLL drops, val NLL is finite, run dir `dirichlet_mdn_runs/<timestamp>_smoke_M4_K4` contains `manifest.json`, `model_best.pt`, `splits.json`, `tb/`, `train.log`.

### Baseline training (target on full dataset)

```bash
PYTHONPATH=. python -m dirichlet_mdn.train \
  --parquet hybrid_dataset_all/all_cases_w64_w128_checkpoint_metadata.parquet \
  --hdf5-dir hybrid_dataset_all \
  --input-moments 4 --K 8 --epochs 200 \
  --tag baseline_M4_K8 --device auto

PYTHONPATH=. python -m dirichlet_mdn.train \
  --parquet hybrid_dataset_all/all_cases_w64_w128_checkpoint_metadata.parquet \
  --hdf5-dir hybrid_dataset_all \
  --input-moments 5 --K 8 --epochs 200 \
  --tag baseline_M5_K8 --device auto
```

### Evaluation

```bash
PYTHONPATH=. python -m dirichlet_mdn.evaluate \
  --run-dir dirichlet_mdn_runs/<timestamp>_baseline_M4_K8
```

Writes `eval/metrics_per_record.csv`, `eval/metrics_per_config.csv`, `eval/plots/pdf_album.pdf`, ...

### OOD stress (when the full dataset has multiple configs with diverse runs)

```bash
PYTHONPATH=. python -m dirichlet_mdn.train \
  --parquet <full_dataset>/metadata.parquet --hdf5-dir <full_dataset> \
  --input-moments 4 --K 8 --epochs 200 \
  --holdout-config L1 --tag ood_holdout_L1_M4_K8
```

## Notes for future follow-up

- $S_3$ permutation augmentation — add as a dataset-layer flag `--augment-s3` that relabels both moments and histograms (the latter via index transposition over the simplex grid).
- $K$-ablation sweep — wrap `train.py` invocations in a small driver script; aggregate per-K NLL/L1/JSD curves into a single plot.
- Logistic-normal base family — alternative `model.py` class with a logistic-normal head; loss in `losses.py` reused with a different `component_logpdf` callable.
- Adaptive-$K$ via stick-breaking — replace softmax $\pi$ head with stick-breaking parameterization; everything else stays the same.
- A-posteriori LES drop-in — out of this code's scope (needs the CFD harness).
- Active-learning data-selection loop — out of this code's scope (needs DNS access).
- Conditional reaction-rate $\overline{\dot\omega}$ comparison — needs a chemistry table; defer to a follow-up.

## Accuracy remediation (2026-09-04)

This section supersedes older implementation descriptions where they conflict.
It records the fixes approved after `CODE_REVIEW.md`.

### Implemented corrections

- **B1 — canonical sampler:** `EnsightPDFHybridDataset.py` is now a small
  single-rank wrapper around `EnsightPDFHybridDatasetMPI.py`. Serial and MPI use
  one extraction algorithm and one HDF5/Parquet schema.
- **B2/B3 — honest splits:** counts respect zero ratios; a named holdout is the
  only test configuration; ordinary splits partition global physical `run_id`
  values once across every scalar configuration. Timestep fallback was removed
  because it leaked one evolution across partitions. A pure configuration
  holdout is rejected when its run IDs also occur in other configurations,
  because pure holdout, complete row coverage, and run isolation are then
  mathematically incompatible.
- **B4/H1 — one simplex convention:** rectangular cells are clipped against
  the physical triangle. Stored areas and centroids describe the clipped
  polygons, and area sums to 0.5. Writing, loading, NLL, rendering, moments,
  metrics, and plots now use this convention. Targets are validated and
  normalized. Concentrations are restricted to `alpha >= 1` so centroid
  quadrature never pretends to resolve an infinite boundary density.
- **B5/H2 — MPI correctness and provenance:** Phase 1 collects every candidate.
  The merge rejects stale, missing, extra, lossy, or configuration-incompatible
  rank output. Candidates are globally ordered by a seeded SHA-256 hash of
  physical identity, so changing rank count does not change arrival order.
  Phase 1 writes bounded-size shards incrementally; Phase 2 external-sorts
  bounded metadata chunks and uses a configurable bounded HDF5 handle cache.
- **H3 — safe publication:** cleanup matches only canonical shard names for the
  exact dataset tag. New output is built in a staging directory, data files are
  renamed into place, and the manifest is published last as the commit marker.
- **H4 — scalar validation:** paired arrays must match; all values are checked
  for finiteness, bounds, and the simplex rule. Tolerance corrections and
  explicit drops are counted. Histogramming must account for every validated
  pair. EnSight files must contain exactly the expected number of scalar values.
- **H5/M6 — truthful moment handling:** moment agreement remains a regularizer,
  not a construction guarantee. Moment MSE is dimensionless, using physical
  scales `[1, .25, 1, .25, .25]`. Checkpoint selection defaults to total
  validation loss. Evaluation reports absolute, target-relative, and
  physical-scale-relative errors plus an explicit tolerance pass/fail result.
- **H6 — independent references:** moment formulas are compared with independent
  NumPy Monte Carlo samples. The K=1 recovery target is also made from NumPy
  samples rather than the density helper being tested.
- **H7 — complete available evaluation:** per-record held-out NLL is reported
  with L1, joint/marginal JSD, and moment errors. The reaction-rate metric remains
  explicitly deferred because no chemistry function or table is present.
- **H8/M1/M2 — integrity:** all direct-versus-histogram moment discrepancies are
  audited against a configurable limit. Training exhaustively validates every
  shard, histogram, mapping, sample ID, and geometry. Splits carry a metadata
  fingerprint and are checked for identity, uniqueness, overlap, coverage,
  holdout purity, and physical-run isolation.
- **M3 — exact preprocessing artifact:** `input_transform.json` stores the exact
  effective center and scale used by PyTorch. Evaluation uses this artifact;
  `scaler.pkl` is retained only for inspection and compatibility.
- **M4/M5 — accurate diagnostics:** non-uniform-grid plots show density
  (`mass / clipped area`) at actual bin edges. Low relative variance is labeled
  well mixed; high relative variance is labeled highly segregated.
- **M7 — visible selection weighting:** deterministic hash ordering removes
  systematic run/configuration arrival order and can be varied with
  `--selection-seed`. Every retained row stores source candidate/retained counts
  and an inverse retention weight. Evaluation reports micro, macro, per-config,
  per-timestep, and source-weighted summaries with dispersion.
- **M8 — bounded claims:** README, proposal, and LaTeX now state that finite,
  bounded mixtures cannot exactly represent edge/vertex atoms or guarantee
  moment equality.
- **L1/L2/L3 — reproducibility:** run directories use microseconds plus a random
  suffix and refuse reuse. Grid, decomposition, threshold, ID, and binary-size
  invariants fail before expensive work. `pixi run test` provides deterministic
  synthetic regression coverage without proprietary DNS data.

### Validation and review follow-up

- All 23 deterministic regression tests pass, including exact clipped-simplex
  area, serial/MPI identity, global-run split isolation, rank provenance,
  zero-safe metrics, and exhaustive synthetic-dataset validation. The validator
  test also alters an HDF5 row after writing and confirms that the stored
  metadata mismatch is rejected.
- Independent NumPy verification passes: the largest analytical-versus-Monte
  Carlo moment difference was \(9.683\times10^{-5}\), and the recovered
  single-Dirichlet concentration had 0.38% maximum relative error.
- A fresh 30-record synthetic dataset passed exhaustive preflight validation,
  completed one CPU training epoch, and completed held-out evaluation. Its
  one-epoch model failed the configured moment-accuracy threshold, as expected;
  the important workflow check is that the failure was measured and reported.
- Python byte-compilation and `git diff --check` pass. The declared Pixi/TeX
  environment was unavailable in this runner, so validation used the installed
  Python packages and the LaTeX guide could not be compiled here.
- Review follow-up tied every Phase-1 HDF5 row to Parquet sample/local IDs,
  made dataset-bound verification accept non-default grid arguments, and
  removed the remaining proposal claims of exact moment enforcement or exact
  subsumption of the analytical hierarchy.
- Final follow-up rejects null shard/configuration names, non-finite derived
  moments, missing row-identity arrays, incomplete HDF5 coverage, and shared-run
  configuration holdouts. New runs declare artifact schema version 2. Older
  runs require the explicit `evaluate --allow-legacy-artifacts` option, which
  loads their stored grid and scaler and warns that their metrics are not
  comparable with corrected version-2 runs.

### Deviations from the approved remediation plan

1. **Boundary integration was replaced by a restricted family.**
   - Planned: numerically integrate components with concentrations below one in
     every clipped boundary cell.
   - Implemented: require every concentration to be at least one and evaluate
     density at each clipped-cell centroid.
   - Reason: reliable singular-cell integration needs a substantially more
     expensive quadrature design and convergence study.
   - Consequence: the model is stable and its limitation is explicit, but exact
     edge/corner atoms and below-one Dirichlets are unsupported.

2. **Lossless Phase 1 initially used rank-wide buffering.**
   - Planned preference: a lossless collector with bounded memory.
   - Initial implementation: lossless collection was buffered per rank.
   - Follow-up: candidates now flush at `--hdf5-shard-size`, while merge metadata
     is externally sorted in `--merge-sort-chunk-rows` batches.
   - Consequence: memory is bounded by one extraction shard per rank and one
     metadata sort chunk on rank 0; the complete lossless candidate set still
     requires corresponding disk space.

3. **Downstream closure error is deferred rather than fabricated.**
   - Planned: report conditional reaction-rate error.
   - Implemented: held-out NLL and all available PDF/moment metrics; no reaction
     metric.
   - Reason: the repository has no reaction-rate function, chemistry table, or
     accepted reference values.
   - Consequence: no LES reaction-rate accuracy claim is supported yet.

4. **Legacy datasets are rejected rather than repaired.**
   - Planned: measure whether existing center-masked files were salvageable.
   - Implemented: format version 2 requires clipped centroids and exhaustive
     validation rejects legacy geometry.
   - Reason: discarded boundary mass cannot be reconstructed from normalized
     legacy histograms.
   - Consequence: datasets must be regenerated before new training. Existing
     historical runs can still be evaluated only through the explicit legacy
     option, using their stored geometry; those metrics are labeled
     non-comparable with corrected runs.

5. **Validation in this change uses synthetic data only.**
   - Planned: also regenerate and train on DNS data.
   - Implemented: deterministic synthetic tests and source compilation.
   - Reason: raw DNS, generated HDF5/Parquet datasets, and checkpoints are not
     present in this repository clone.
   - Consequence: real-data MPI sampling and training remain operator acceptance
     steps; no empirical accuracy number is claimed here.

### Superseded earlier deviations

- The previous timestep split fallback is removed.
- The K=1 self-check no longer generates its target with the code under test.
- The scaler's zero-scale guard is now captured exactly in
  `input_transform.json`.
- Older output-schema and best-validation-NLL descriptions above are historical;
  the corrected artifacts and default total-loss selection are authoritative.

## Change log

Newest first. Format: `- YYYY-MM-DD: <one-line summary>`.

- 2026-09-04: Implemented the approved accuracy remediation across sampling,
  simplex geometry, MPI provenance, splits, training artifacts, evaluation,
  diagnostics, independent verification, tests, and novice LaTeX documentation.
  Deviations and unavailable real-data acceptance steps are recorded above.
- 2026-06-16: First end-to-end pipeline landed and validated. All five verify checks pass; 3-epoch smoke training in both 4-input and 5-input variants runs cleanly (train NLL 4.22 → 3.45 in 3 epochs); evaluator produces 322-record metrics table and full plot set on the smoke run. Implementation summary, exact output schema, CLI arguments, and example commands sections all populated. Five deviations from the plan documented (timestep-fallback split mode, explicit broadcasting in `mixture_logpdf`, hand-written Markdown table writer, zero-scale guard in `TorchScaler`, K=1 verify init via method-of-moments).
- 2026-06-16: Created implementation log skeleton. Plan persisted to `DirichletMDN_implementation_plan.md`.
