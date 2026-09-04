# Dirichlet Mixture Density Network — Training, Test, and Verification Pipeline

> Approved implementation plan. Companion to `DirichletMDN_proposal.md` (methodology) and `EnsightPDFHybridDataset.md` (data pipeline). The running implementation log is `DirichletMDN_implementation.md`.

## Context

`DirichletMDN_proposal.md` lays out the methodology — a small MLP that maps LES subgrid moments to the parameters of a $K$-component Dirichlet mixture on the 2-simplex, replacing the 4096-output pixel-wise softmax DNN of `MLPDF.py`. The hybrid dataset writer `EnsightPDFHybridDataset.py` has produced a checkpoint dataset in `hybrid_dataset_all/` containing **3,931 retained LES filter-cell PDFs** as 64×64 histograms with the five DNS-box moments stored separately in parquet metadata. We now need a PyTorch training, validation, and verification routine that operates on this dataset, runs in conda env `ct-env`, and supports both 5-input (moments+covariance) and 4-input (moments only, no covariance) variants — the latter motivated by the fact that in actual LES the scalar-scalar covariance transport equation has unclosed terms and is generally not solved.

### Dataset status — preliminary

`EnsightPDFHybridDataset.py` is **currently running** to produce a larger, better-balanced dataset with broader coverage of `(scalar_config, run_id, timestep, box_width)` and richer shape diversity. The 3,931-sample checkpoint file in `hybrid_dataset_all/` is **a partial extract intended only to enable this first cut of the PyTorch pipeline**. The code below is therefore written to be dataset-size-agnostic:

- All split, scaling, and batching logic operates on parquet row counts at load time — no constants pinned to 3,931.
- The schema-tolerant reader accepts either the older or newer hybrid-dataset schema variants.
- `--parquet` and `--hdf5-dir` can be repointed to the new dataset directory without code changes.
- A multi-shard reader is included from the start (the new dataset is expected to span several `*.h5` shards even though the current checkpoint has only one).

When the full dataset lands, the same `train.py` / `evaluate.py` / `verify.py` commands rerun with new arguments. Any tuning that turns out to be dataset-specific (e.g. `lambda_mom`, `alpha_clip`, `batch_size`) is captured in the run manifest, not hard-coded.

### Key facts from inspection

| Item | Value |
|---|---|
| Records | 3,931 retained LES PDFs |
| Storage | one parquet metadata file + one HDF5 shard |
| Histogram grid | 64×64 non-uniform bins, 3,642 cells inside simplex |
| Stored moments | DNS-box direct moments (`mean_a, var_a, mean_b, var_b, cov_ab`) plus histogram-recovered diagnostics |
| Config breakdown | L1: 2247, I1: 650, I4: 392, PI1: 336, PI4: 270, I5: 27, PI5: 9 |
| Box widths | 64 and 128 |
| Run ids | 0–25 |
| Note | Many records have `mean_b ≈ 0`, `var_b ≈ 0` — Z2 is sparse; the 3-stream problem is effectively 2-stream for a large fraction of records |
| Status | **Partial checkpoint — the extractor is still running and will produce a larger, more diverse dataset.** The pipeline must work on this file without baking in its quirks. |

### Confirmed design decisions

1. **Likelihood** — histogram-categorical NLL: `L = -Σ_j p_j log q_j` with `q_j` = Dirichlet mixture density at bin center × cell area, renormalized over the simplex mask.
2. **Ablation scope** — land a single $K=8$ end-to-end run first; ablation over $K \in \{1, 2, 4, 8, 16, 32\}$ as a follow-up driven by a config-file knob (already exposed via CLI for free).
3. **Augmentation** — $S_3$ permutation off for v1 (would invent boxes with $Z_1 \approx 0$ that don't reflect the physical initial-condition coverage).
4. **Input dimensionality** — configurable: 4 (no covariance) or 5 (with covariance). Single code path, single CLI flag `--input-moments {4,5}`. The 4-input variant is the deployable LES-ready model; the 5-input variant is the upper-bound diagnostic.

## File layout

All new files under `/Users/syellapa/Documents/Research/2026/ANSYS/Multi-ScalarPDF/dirichlet_mdn/`:

```
dirichlet_mdn/
  __init__.py
  data.py            # HybridPDFDataset: parquet + HDF5 reader, splits, simplex-mask handling
  model.py           # DirichletMDN: MLP trunk + π and α heads
  losses.py          # histogram NLL, analytical moment recovery, entropy reg
  metrics.py         # JSD, L1, R^2 on moments, marginal JSD, all on the 64x64 grid
  splits.py          # train/val/test split by (scalar_config, run_id), deterministic
  train.py           # CLI training entry point
  evaluate.py        # CLI evaluation entry point (test set + per-config breakdown + plots)
  verify.py          # CLI sanity-check entry point (analytical-Dirichlet fit, K=1 sanity, etc.)
  bin_grid.py        # non-uniform bin reconstruction (mirrors EnsightPDFHybridDataset.py)
  plotting.py        # contour plots of DNS vs MDN PDFs with simplex polygon overlay
```

Outputs land in `dirichlet_mdn_runs/<timestamp>_<tag>/` containing the model checkpoint, scaler pickle, manifest JSON, TensorBoard logs, per-config metrics tables, and PDF plots — matching the style of `MLPDF.py`.

## Component specifications

### `bin_grid.py` — non-uniform bin reconstruction

Stand-alone reproduction of the bin construction in `EnsightPDFHybridDataset.py:920-931` so the training code does not depend on the writer script. Two functions:

- `bin_centers_and_edges(num_bins: int, zst: float, uniform: bool) -> (centers_a, edges_a, cell_area, simplex_mask)` — returns numpy arrays for one axis (centers, edges) and the 2D `cell_area` and `simplex_mask`. Uses the same lower-half-linear, upper-half-quadratic-with-C1-continuity formula. Sanity-checked at construction time against the values stored in the HDF5 shard.
- `validate_against_hdf5(h5_path, num_bins, zst, uniform)` — reads `/bins/edges_a`, `/bins/centers_a`, `/bins/simplex_mask`, `/bins/cell_area` from the dataset and asserts agreement to 1e-9.

### `data.py` — `HybridPDFDataset`

Reads the parquet metadata file and the HDF5 shards. **Multi-shard from day one** — the parquet `hdf5_file` column tells each row which shard to read; the dataset opens each unique shard once. Schema-tolerant: accepts either the older (`box_width`, `run_id`) or newer (`filter_width`, `run_number`) column names; chooses the available one. The current checkpoint file uses the older schema but the in-progress run may emit either, so the reader must handle both. Lazy histogram reads via `h5py` file handles cached per shard.

`__init__` signature:
```
HybridPDFDataset(parquet_path, hdf5_dir, *,
                 input_moments=5,     # 4 or 5
                 indices=None,        # row indices to include (from splits.py)
                 cache_histograms=False)
```

`__getitem__(i)` returns a dict:
```
{
  "moments":   torch.float32, shape (input_moments,),
  "histogram": torch.float32, shape (num_bins, num_bins),  # already simplex-masked + renormalized
  "meta":      {"scalar_config", "run_id", "timestep", "box_width", "sample_id"}
}
```

Notes:

- `input_moments=4` drops `cov_ab` and keeps `[mean_a, var_a, mean_b, var_b]`.
- `input_moments=5` keeps `[mean_a, var_a, mean_b, var_b, cov_ab]`.
- The histogram is `float32` and already sums to 1 in the stored data; an assertion confirms `|sum - 1| < 1e-5` on first read of each shard.
- A class method `compute_input_scaler(records, input_moments)` returns a fitted `sklearn.preprocessing.RobustScaler` — same pattern as `MLPDF.py:71`. Stored in the run directory.

A dataset-level helper exposes the precomputed `cell_area` and `simplex_mask` as `torch.float32` and `torch.bool` tensors so the loss can use them without per-batch overhead.

### `splits.py` — leak-free splits

Critical: split by `(scalar_config, run_id)` — not by row, and not by `(scalar_config, run_id, timestep)`. This avoids the leak in `MLPDF.py:177` where rows from the same physical DNS run end up in train and val.

Default scheme:
- For each `scalar_config`, partition its run_ids into train/val/test with ratios 0.7 / 0.15 / 0.15, deterministic under a `--seed` argument.
- Configs with fewer than 3 distinct run_ids get all rows into train with a logged warning.
- Output: three lists of parquet row indices, written to `<run_dir>/splits.json` for full reproducibility.

CLI override: `--holdout-config L1` for the out-of-distribution stress test (train on everything except L1, test on L1 only) — this is the §5.5 ablation #4 from the proposal.

### `model.py` — `DirichletMDN`

Direct translation of proposal §3.1 + §5.2. Single class, input dimension configurable:

```
class DirichletMDN(nn.Module):
    def __init__(self, n_in: int, hidden: int = 256, K: int = 8,
                 alpha_min: float = 1e-3, alpha_clip: float = 1e3):
        # trunk: Linear(n_in, 128) -> SiLU -> Linear(128, hidden) -> SiLU
        #        -> Linear(hidden, hidden) -> SiLU
        # head_pi:    Linear(hidden, K)      -> softmax
        # head_alpha: Linear(hidden, 3 * K)  -> softplus + alpha_min, clipped above
    def forward(self, m): -> pi (B, K), alpha (B, K, 3)
```

`alpha_clip` keeps `α_0 = Σ_i α_i` finite — a hard upper bound prevents the late-time near-delta cases from driving any single `α_i` into the `1e+12` regime where `lgamma` loses precision.

### `losses.py`

Three losses, summed with configurable weights.

1. **`histogram_nll(pi, alpha, hist, centers_a, centers_b, cell_area, simplex_mask, eps=1e-12)`**

   For each batch element:
   - Compute `q_ij = mixture_density(z1_center_i, z2_center_j; pi, alpha) * cell_area_ij` for all `(i, j)` inside `simplex_mask`.
   - Renormalize `q_ij` over the in-simplex cells so `Σ q_ij = 1`. (Compensates for the simplex truncation and discretization-induced normalization drift in the analytical Dirichlet mass.)
   - Loss: `-Σ_ij hist_ij * log(q_ij + eps)`, averaged over batch.

   The mixture density at a 2-simplex point `(z1, z2)` with implicit `z3 = 1 - z1 - z2`:
   ```
   logsumexp_k[ log pi_k + dirichlet_logpdf((z1, z2, z3); alpha_k) ]
   ```
   evaluated in log-space throughout to prevent underflow at the corners (where `α_i < 1` blows up the density).

   Cost: $O(B \cdot N_{\text{cells}} \cdot K)$ per batch. With $B = 64$, $N_{\text{cells}} = 3642$, $K = 8$ → 1.9M evaluations per batch — easily tractable on CPU/MPS/CUDA. Centers and cell area are pre-broadcast tensors registered as buffers on the loss module.

2. **`moment_loss(pi, alpha, m_input, input_moments)`**

   Closed-form moment recovery (proposal §3.4, §5.2). Already analytical:
   - Predicted means $E[Z_i]$, variances $\mathrm{Var}[Z_i]$, and covariance $\mathrm{Cov}[Z_1, Z_2]$ from `(pi, alpha)`.
   - Stack into either `[E1, V1, E2, V2, C12]` (5-input) or `[E1, V1, E2, V2]` (4-input).
   - MSE against `m_input` (which was already in the same order at the dataset layer).

   Note: in the 4-input variant the mixture still has a well-defined covariance, but the loss does not penalize it — the data simply does not condition on it.

3. **`entropy_reg(pi)`** — `-Σ_k π_k log π_k` averaged over batch. The training step subtracts $\lambda_e \cdot$ entropy (i.e., maximizes entropy) per §3.4.

Combined loss:
```
loss = nll + lambda_mom * moment_loss - lambda_ent * entropy
```

Defaults `lambda_mom = 0.5`, `lambda_ent = 1e-3` — tunable via CLI.

### `metrics.py`

All operate on either `(pred_pi, pred_alpha, hist)` or on a binned reconstructed PDF `q_ij`.

| Function | Formula | Returns |
|---|---|---|
| `joint_l1(p, q, simplex_mask)` | $\sum_{ij \in \mathrm{mask}} \|p_{ij} - q_{ij}\|$ | scalar per record |
| `joint_jsd(p, q)` | Jensen–Shannon divergence on flattened in-simplex PDFs (port of `MLPDF.py:79`) | scalar per record |
| `marginal_jsd(p, q, axis)` | JSD on $P(Z_1)$ or $P(Z_2)$ after summing along the other axis | scalar per record |
| `moment_recovery_error(pred_moments, true_moments)` | Per-component absolute and relative error | dict |
| `mixture_active_count(pi, threshold=0.01)` | Number of components with $\pi_k > 0.01$ — diagnostic for mode collapse | scalar per record |
| `per_config_breakdown(records, predictions)` | Mean ± std of each metric grouped by `scalar_config` | DataFrame |
| `per_timestep_breakdown(records, predictions)` | Mean of each metric grouped by `timestep` (proxy for $t/\tau_{\text{eddy}}$) | DataFrame |

JSD and L1 are reused from `MLPDF.py:79,97` where the formulae match.

### `train.py` — CLI

```
python -m dirichlet_mdn.train \
  --parquet /Users/syellapa/.../hybrid_dataset_all/all_cases_w64_w128_checkpoint_metadata.parquet \
  --hdf5-dir /Users/syellapa/.../hybrid_dataset_all \
  --input-moments 4      # or 5
  --K 8
  --hidden 256
  --epochs 200
  --batch-size 64
  --lr 3e-4
  --lambda-mom 0.5
  --lambda-ent 1e-3
  --seed 0
  --output-dir dirichlet_mdn_runs
  --tag baseline_K8_M4
  --device auto          # auto-detects CUDA > MPS > CPU
  --holdout-config none  # or e.g. L1 for OOD stress test
```

Loop mirrors §5.3 of the proposal:

- AdamW, `weight_decay=1e-5`, cosine LR schedule over `epochs`.
- Gradient clipping at 5.0.
- Early stopping: track best val NLL; if it does not improve for `--patience 30` epochs, stop.
- Per-epoch logging to TensorBoard (loss components, val NLL, mean active component count). Uses `torch.utils.tensorboard`.
- Checkpoint state dict + scaler + config to `<output-dir>/<tag>/`.

The same `train.py` covers both `--input-moments 4` and `--input-moments 5`. The only differences propagate via:
- `DirichletMDN(n_in=input_moments, ...)`
- `HybridPDFDataset(..., input_moments=input_moments)`
- `moment_loss(..., input_moments=input_moments)`

### `evaluate.py` — CLI

Loads a trained run directory and runs the full §5.4–§5.5 protocol on the test split:

1. Per-record metrics from `metrics.py`.
2. Per-config and per-timestep breakdown tables → CSV + Markdown.
3. **Plots** (use `plotting.py`):
   - Predicted vs DNS contour PDFs for a fixed grid of representative records (one per `(scalar_config, timestep)` pair), with simplex polygon overlay borrowed from `MLPDF.py:283-300`.
   - L1 and JSD error vs `timestep`, one line per `scalar_config`.
   - Mixture diagnostics: histogram of active-component count, scatter of `α_0` vs `var_a`.
4. **Moment self-consistency check** — predicted mixture moments vs input moments, scatter + R² (proposal §5.4 row 2).
5. Saves all artifacts under `<run_dir>/eval/`.

### `verify.py` — CLI

Lightweight standalone sanity checks, runnable in seconds:

1. **`--check-bins`** — `bin_grid.validate_against_hdf5` confirms reconstructed bin geometry matches the dataset to numerical precision.
2. **`--check-loss-gradient`** — single-batch forward+backward with the histogram-NLL loss; reports gradient norms per parameter group. Confirms no NaN/Inf at initialization.
3. **`--check-k1-fits-analytical-dirichlet`** — fits `K=1` on a synthetic single-Dirichlet histogram of known $(α_1, α_2, α_3)$, asserts recovered concentration vector matches to <2% relative error. This is the §3.5 / proposal §6 sanity claim: `K=1` ≡ Perry & Mueller's analytical Dirichlet baseline.
4. **`--check-moment-loss-zero-on-true-params`** — given known $(\pi, \alpha)$, compute mixture moments and pass them as `m_input` — `moment_loss` should be `< 1e-10`. Catches algebra errors in the closed-form moment expressions.
5. **`--check-permutation-invariance`** — confirms that swapping the order of mixture components in `(pi, alpha)` leaves the NLL and moment outputs unchanged. Standard MDN test.

These run as PyTest-style functions; CLI prints PASS/FAIL for each.

### `plotting.py`

- `plot_pdf_comparison(hist_true, hist_pred, moments, ax)` — twin contour plot, simplex polygon overlay (`pointsO/pointsI/Polygon` from `MLPDF.py:283-300`).
- `plot_error_vs_time(metrics_df, metric_name, ax)` — line per config.
- `plot_alpha_diagnostics(alpha, ax)` — diagnostic for mixture collapse.

## Reuse from existing code

| New file | Reuses | Source |
|---|---|---|
| `data.py` | RobustScaler pattern + joblib persistence | `MLPDF.py:71` |
| `metrics.py` | `jensen_shannon_divergence`, `calculate_jsd`, `summarize_training` | `MLPDF.py:79, 97, 110` |
| `bin_grid.py` | non-uniform bin construction | `EnsightPDFHybridDataset.py:920-931, 954, 1004-1016` |
| `plotting.py` | simplex polygon overlay, twin contour layout | `MLPDF.py:283-300` |
| `train.py` | `create_logdir`, TensorBoard layout | `MLPDF.py:33, 421` |
| `evaluate.py` | per-case L1/JSD plotting against time | `MLPDF.py:230` |

## Risks and deviations from the proposal — flagged honestly

1. **Histogram-categorical NLL instead of particle NLL.** Proposal §3.4 evaluates NLL on raw DNS particles. The dataset stores histograms only. The categorical formulation is exact in the small-bin limit and is the established practice for binned-data MLE; we lose ~1-2% likelihood precision from the discretization. Documented explicitly in the run manifest.
2. **Dataset size and balance — preliminary file only.** The current 3,931 records (L1 = 57%, PI5 = 0.2%, many `mean_b ≈ 0` boxes) are a partial checkpoint while `EnsightPDFHybridDataset.py` continues to run. Metrics from this first cut should be read as pipeline-correctness checks, not as the final scientific result. Per-config metrics for I5 and PI5 will have high variance; the K=16/32 ablation may be data-limited rather than architecture-limited; the Z2 ≈ 0 dominance means the 3-stream advantage of Dirichlet over independent betas is only weakly tested. **Plan to rerun the full training and evaluation matrix on the larger dataset once available — without code changes**, only by repointing `--parquet` and `--hdf5-dir`.
3. **No permutation augmentation in v1.** Decided above. Easy to enable later via a dataset-layer flag (`--augment-s3`) that applies the relabeling to both `(mean, var, cov)` and the histogram (via index transposition over the simplex grid).
4. **4-input variant cannot constrain off-diagonal cross-moments.** The trained model will produce a mixture that does have a covariance, but training does not penalize it — only NLL on the histogram does. Expect higher covariance recovery error in the 4-input variant; this is *the* central diagnostic for whether covariance-blind LES closures lose meaningful joint-PDF accuracy.
5. **`alpha_clip`** — needed to keep $\lgamma$ stable on late-time near-delta cases. A clip at $10^3$ corresponds to $\sigma \sim 0.03$ per component, well below the discretization width. Documented; revisit if it caps a real signal.
6. **Train/val/test by `(scalar_config, run_id)`** — fixes the leak in `MLPDF.py`. Means the new model's headline NLL will look *worse* than a naive row-shuffle baseline, but it will be honest. Worth stating in the writeup.

## Environment

`ct-env` already has `torch 2.11`, `h5py 3.14`, `numpy 2.4`, `pandas 3.0`, `scipy 1.17`, `sklearn 1.8`, `matplotlib 3.10`. **Missing**: `pyarrow` (required to read the parquet metadata) and `tensorboard` (for logging). Single one-time install:

```bash
conda activate ct-env
pip install pyarrow tensorboard
```

`pyarrow 24.0.0` was installed during plan exploration. `tensorboard` install happens as the first implementation step.

## Implementation order (first actions)

Before touching any Python:

1. **Persist this plan into the repo as a permanent reference.** Copy the contents of this plan file to `/Users/syellapa/Documents/Research/2026/ANSYS/Multi-ScalarPDF/DirichletMDN_implementation_plan.md` so the implementation rationale travels with the code and survives across Claude sessions.
2. **Create the running implementation log** `DirichletMDN_implementation.md` (see next subsection).
3. Install missing packages into `ct-env` (`pip install pyarrow tensorboard`).
4. Create the `dirichlet_mdn/` package skeleton with empty modules in the layout above.
5. Implement files in the order: `bin_grid.py` → `data.py` → `model.py` → `losses.py` → `verify.py` (sanity-check the first three) → `metrics.py` → `splits.py` → `train.py` → `plotting.py` → `evaluate.py`.
6. Run the verification commands below to confirm the pipeline.

### Running implementation log

Create and maintain `/Users/syellapa/Documents/Research/2026/ANSYS/Multi-ScalarPDF/DirichletMDN_implementation.md` throughout the build. It mirrors the style of `EnsightPDFHybridDataset.md` and acts as the working record of what was actually built versus what was planned. **Update it incrementally as work progresses, not retroactively** — append entries the same turn the work happens.

Required sections (skeleton populated up front, contents grow):

- **Purpose** — one paragraph: this file logs the implementation of `dirichlet_mdn/`; references `DirichletMDN_proposal.md` and `DirichletMDN_implementation_plan.md`.
- **Environment** — exact `ct-env` package versions used. Pasted from `pip freeze` output of the run that produced the first working baseline.
- **Implementation summary** — module-by-module: one paragraph per file describing what it does and what existing code (if any) it reuses. Updated as each module is committed.
- **Important implementation details** — numbered subsections for each non-obvious decision made during coding.
- **Deviations from the plan** — numbered subsections, each with **Planned**, **Implemented**, **Reason**, and **Consequence** fields. Every divergence from this plan file goes here, however small.
- **Exact output schema** — for each artifact in `<run_dir>/`: file name, format, fields. Mirrors the "Exact output schema" section of `EnsightPDFHybridDataset.md`.
- **CLI arguments** — every flag, default, and accepted range for `train.py`, `evaluate.py`, `verify.py`. Mirrors the "CLI arguments" section of `EnsightPDFHybridDataset.md`.
- **Example commands** — at least one worked example per script, with the exact dataset paths that were used. Updated when the dataset is repointed to the full extract.
- **Notes for future follow-up** — items deferred (S₃ augmentation, logistic-normal base, adaptive-K, etc.) with one-line pointers to where in the code they would slot in.
- **Change log** — dated bullet entries, newest first.

This file is the single source of truth for "what does the current code actually do" — when planning conflicts with reality, this file wins and the plan file is updated to match.

## Verification — end-to-end

```bash
conda activate ct-env

# 1. Sanity: bin geometry, gradients, K=1 analytical equivalence, moment-loss zero-on-truth,
#    permutation invariance. Runs in < 30 seconds, no training.
python -m dirichlet_mdn.verify --check-all \
  --parquet hybrid_dataset_all/all_cases_w64_w128_checkpoint_metadata.parquet \
  --hdf5-dir hybrid_dataset_all

# 2. Quick smoke training: 4-input, K=4, 5 epochs. Confirms wiring end-to-end.
python -m dirichlet_mdn.train \
  --parquet hybrid_dataset_all/all_cases_w64_w128_checkpoint_metadata.parquet \
  --hdf5-dir hybrid_dataset_all \
  --input-moments 4 --K 4 --epochs 5 --batch-size 32 \
  --tag smoke_M4_K4 --output-dir dirichlet_mdn_runs

# 3. Baseline run: 4-input (LES-deployable), K=8, full epoch budget.
python -m dirichlet_mdn.train --input-moments 4 --K 8 --epochs 200 \
  --tag baseline_M4_K8 ...

# 4. Upper-bound diagnostic: 5-input (with covariance), K=8.
python -m dirichlet_mdn.train --input-moments 5 --K 8 --epochs 200 \
  --tag baseline_M5_K8 ...

# 5. Evaluate both, side-by-side. Produces per-config / per-timestep tables and plots.
python -m dirichlet_mdn.evaluate --run-dir dirichlet_mdn_runs/baseline_M4_K8
python -m dirichlet_mdn.evaluate --run-dir dirichlet_mdn_runs/baseline_M5_K8

# 6. OOD stress (optional in first pass): hold out L1 (the dominant config) entirely.
python -m dirichlet_mdn.train --input-moments 4 --K 8 --epochs 200 \
  --holdout-config L1 --tag ood_holdout_L1_M4_K8
```

### Rerun on the full dataset (when ready)

When `EnsightPDFHybridDataset.py` finishes its current run, the larger dataset will land in a new directory with one parquet metadata file and one-or-more `*.h5` shards. The same commands rerun unchanged except for `--parquet` and `--hdf5-dir`:

```bash
# Reproduce the baselines on the full dataset
python -m dirichlet_mdn.train \
  --parquet <NEW_DIR>/<NEW>_metadata.parquet --hdf5-dir <NEW_DIR> \
  --input-moments 4 --K 8 --epochs 200 --tag full_M4_K8 ...
python -m dirichlet_mdn.train \
  --parquet <NEW_DIR>/<NEW>_metadata.parquet --hdf5-dir <NEW_DIR> \
  --input-moments 5 --K 8 --epochs 200 --tag full_M5_K8 ...

# Then the K-ablation (deferred to this stage so the curve is computed on real data)
for K in 1 2 4 8 16 32; do
  python -m dirichlet_mdn.train --input-moments 4 --K $K --epochs 200 \
    --tag full_M4_K${K} ...
done
```

Both 4-input and 5-input runs are repeated so the covariance-blind vs covariance-aware comparison is reported on the same dataset and split.

A run is considered "working" when:

- `verify --check-all` passes all five sanity checks.
- Smoke training completes 5 epochs without NaN, decreasing training loss, finite validation NLL.
- Baseline run produces per-config metrics tables, predicted-vs-DNS contour plots, and a TensorBoard log with non-trivial structure (NLL decreasing, entropy non-degenerate).

## Out of scope for this plan (explicitly deferred)

- $S_3$ permutation augmentation (flag stub only; not implemented).
- $K$-ablation sweep (config knob exists; sweep script and result aggregator deferred).
- Logistic-normal alternative base family (proposal §2.6, future work).
- Adaptive-$K$ via stick-breaking (proposal §3.3, future work).
- A-posteriori LES drop-in evaluation (proposal §5.5 item 5 — needs CFD harness; out of scope for this code).
- Active-learning data-selection loop (proposal §4.3 item 5 — needs DNS access).
- Conditional reaction-rate $\overline{\dot\omega}$ comparison (needs a chemistry table; ask in a follow-up).
