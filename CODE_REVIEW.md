# Implementation Accuracy Review

**Review date:** 2026-09-04
**Status:** Findings and proposed remediation only. No implementation fixes have been made.

## Scope and review standard

This review prioritizes scientific and implementation correctness over software
sophistication. It covers both dataset generators, the Dirichlet MDN, data
loading and splitting, training, evaluation, diagnostics, and the claims made
in the accompanying documentation.

This was a static review. The raw DNS fields, generated HDF5/Parquet dataset,
and trained checkpoints are intentionally absent from the repository, so
dataset-wide numerical audits and end-to-end reproduction were not possible.
Several findings therefore include an explicit validation step for the real
artifacts.

## Frank overall assessment

The core closed-form mixture moment algebra is correct, the main likelihood
calculation uses appropriate log-space operations, and fitting the input scaler
on training rows only is correct. The repository also documents its intent
unusually well.

However, I would not treat the current headline validation results as
scientifically reliable yet. The serial reference sampler is broken, the
holdout split does not implement a true holdout, the default split likely leaks
physical runs across partitions, and the treatment of histogram cells along
the simplex boundary can silently remove target mass. MPI reruns can also
ingest stale rank outputs. These are accuracy and provenance problems, not
style complaints.

## Findings

### Blockers

#### B1. The serial dataset generator is two concatenated programs and cannot complete cleanly

**Evidence**

- `EnsightPDFHybridDataset.py:542-761` contains one complete program and calls
  its `main()` at lines 760-761.
- A second, incompatible program starts at line 764, defines another `main()`
  at line 1689, and calls it at lines 1852-1853.
- The second program uses undefined annotation/import names such as `List`,
  `Tuple`, `Sequence`, `Dict`, `dataclass`, `math`, `datetime`, and
  `SHAPE_FEATURE_COLUMNS`.
- `EnsightPDFHybridDatasetMPI.py:22-23` explicitly acknowledges that importing
  the serial module crashes after line 762.
- The two programs also write incompatible HDF5/Parquet schemas. The second
  writes root-level datasets, while `dirichlet_mdn/data.py:65-71,168` requires
  `hdf5_file`, `local_index`, and `data/histograms`.

**Impact**

Importing the module fails. Running it as a script executes the first pipeline,
then falls through and fails while defining the second, so orchestration sees a
non-zero exit even after outputs were written. The advertised serial reference
cannot currently provide a trustworthy cross-check of MPI output.

**Proposed correction**

Choose one canonical serial implementation, remove the other, and make its
output schema exactly match the training reader and MPI writer. Add import,
CLI-help, synthetic extraction, and serial-versus-MPI equivalence checks.

#### B2. `holdout_config` does not produce a pure held-out test set

**Evidence**

- `dirichlet_mdn/splits.py:143-161` assigns non-held configurations ratios
  `(0.8, 0.2, 0.0)`.
- `_split_runs` at lines 46-53 and `_split_timesteps` at lines 69-77 always
  reserve at least one item for the final test slice when train and validation
  counts consume the input.
- This contradicts `splits.py:94-95` and
  `DirichletMDN_implementation.md:68-69`, which state that the named
  configuration is the test set.

**Impact**

The OOD test contains rows from configurations seen during training. That
invalidates the intended unseen-configuration experiment and can make its
aggregate score look better than the true holdout score.

**Proposed correction**

Use an explicit two-way train/validation allocation for non-held
configurations, assert that their test contribution is empty, and test the
membership invariant rather than only the split sizes.

#### B3. The default split likely leaks the same physical DNS run across partitions

**Evidence**

- `dirichlet_mdn/splits.py:130-161` groups by `scalar_config` and independently
  assigns run IDs for each configuration.
- The writer reads every scalar configuration from the same
  `run_NNNN/ensight-3D` tree
  (`EnsightPDFHybridDataset.py:582-616`), strongly indicating that a run ID
  denotes one shared velocity realization.
- The committed `splits/by_run_id_seed0/split_report.md:13-20` demonstrates the
  overlap. For example, run 0 is training data for I1 but test data for I4.
- The split nevertheless calls itself leak-free in `splits.py:1-6`.

**Impact**

If scalar configurations under one run share the same DNS flow realization,
the model sees the test run's turbulence during training under another label.
Test scores then overstate generalization to unseen physical runs.

**Proposed correction**

Confirm the run semantics with the data owner. If they are shared realizations,
partition run IDs globally once and apply that partition to every scalar
configuration. Keep configuration holdout as a separate experiment.

#### B4. Valid target mass can be removed at the simplex boundary

**Evidence**

- Histograms are built on rectangular bins without applying the simplex mask
  (`EnsightPDFHybridDataset.py:198-208`;
  `EnsightPDFHybridDatasetMPI.py:224-234`).
- The mask classifies a cell only by whether its center satisfies
  `Z1 + Z2 <= 1` (`dirichlet_mdn/bin_grid.py:91-92`).
- A physically valid point can lie in the triangular portion of a rectangular
  cell whose center is outside the simplex.
- `dirichlet_mdn/losses.py:151-157` drops those target cells but does not
  renormalize the remaining target mass.
- `dirichlet_mdn/data.py:177-180` does the same for flattened histograms, and
  `dirichlet_mdn/metrics.py:72-76` excludes the mass from L1.
- `DirichletMDN_implementation.md:89` incorrectly states that stored
  histograms already zero these cells.

**Impact**

The supposed categorical target may sum to less than one. Boundary-rich
records then receive a smaller effective loss weight, real edge structure is
discarded, and NLL, L1, and JSD do not evaluate the same support convention.
The full rectangular `cell_area` also overstates the area of cells cut by the
simplex boundary.

**Proposed correction**

Adopt one geometrically consistent definition of boundary-cell mass, apply it
in the writer, loss, prediction renderer, and metrics, and enforce normalized
targets. Measure current off-mask mass globally and by mixing regime before
deciding whether existing datasets can be repaired or must be regenerated.

#### B5. MPI merges can silently include stale or incompatible rank outputs

**Evidence**

- Phase 1 creates the rank root but does not clear or uniquely namespace it
  (`EnsightPDFHybridDatasetMPI.py:1090-1098`).
- An empty current rank writes only a manifest and leaves any old Parquet/HDF5
  files in place (`EnsightPDFHybridDatasetMPI.py:792-808`).
- Phase 2 merges every directory matching `rank_*`, not only ranks from the
  current invocation (`EnsightPDFHybridDatasetMPI.py:1169-1179`).
- `--skip-phase1` does not validate grid or selection settings against the rank
  manifests.
- The standard Pixi tasks preserve rank outputs (`pixi.toml:43-89`), making
  reuse a normal path rather than a remote edge case.

**Impact**

A rerun after interruption, a changed MPI size, or changed bin/configuration
arguments can mix old candidates into a new dataset. With changed grid
settings, final histogram arrays can even be paired with incorrect bin
metadata.

**Proposed correction**

Isolate each phase-1 run with a complete configuration signature, clean the
current namespace before writing, require the exact expected rank set, and
reject any manifest or histogram shape that does not match the merge request.

### High-priority correctness issues

#### H1. Boundary-peaked Dirichlet components are rendered with an arbitrary epsilon-dependent approximation

The grid deliberately includes centers at 0 and 1
(`dirichlet_mdn/bin_grid.py:38-62`), while Dirichlet densities with
concentrations below one are singular on edges and corners. The implementation
clamps zero coordinates to an epsilon and renormalizes the point
(`dirichlet_mdn/losses.py:37-47,114-117`), then treats density at that point
times rectangular area as cell mass (`losses.py:138-149`).

This makes edge/corner probabilities depend on the arbitrary clamp, especially
in the early-time regimes the model is intended to capture. It also makes the
rendered histogram used by NLL inconsistent with the exact continuous moments
used by `moment_loss`.

**Plan:** replace boundary point evaluation with a cell-integrated or
independently converged quadrature definition, and add convergence tests over
epsilon/grid resolution for concentrations below, near, and above one.

#### H2. MPI phase-1 pruning makes the dataset depend on rank count and arrival order

Work allocation depends on `index % size`
(`EnsightPDFHybridDatasetMPI.py:628-637`). Each rank applies a finite local cap
before the global merge (`EnsightPDFHybridDatasetMPI.py:709-724,1100`).
Candidates discarded locally can never be selected globally. A multiplier of
two is a heuristic, not a guarantee of equivalence to the serial selection.

**Plan:** either make phase 1 a lossless collector or define and document the
MPI result as an approximation, then test selection stability across rank
counts and cap multipliers.

#### H3. Prefix-based shard cleanup can delete other datasets and orphan checkpoints

`remove_existing_shards` deletes every HDF5 filename beginning with
`<dataset_tag>_` (`EnsightPDFHybridDataset.py:425-430`;
`EnsightPDFHybridDatasetMPI.py:503-510`). A final serial write for tag `foo`
therefore deletes `foo_checkpoint_*.h5` while leaving checkpoint metadata and
manifest files. It can also delete a separate dataset such as `foo_v2`.

**Plan:** delete only files recorded by the exact dataset manifest or matching
an exact shard-name pattern, and update all related metadata atomically.

#### H4. Invalid or dropped DNS values are not accounted for

The writers do not explicitly check finiteness, `0 <= Z1,Z2 <= 1`, or
`Z1 + Z2 <= 1`. They only require that the histogram contain some mass
(`EnsightPDFHybridDataset.py:198-208`;
`EnsightPDFHybridDatasetMPI.py:224-234`). Values outside the histogram edges
are omitted by `numpy.histogram2d`, after which the remaining counts are
renormalized without recording how many cells were lost.

**Plan:** validate every source block, record tolerance-qualified violations
and dropped counts, fail on material violations, and include these totals in
the dataset manifest.

#### H5. Moment consistency with the conditioning inputs is not “by construction”

The network emits unconstrained mixture parameters. Equality between predicted
and input moments is encouraged only by a soft MSE term
(`dirichlet_mdn/losses.py:206-214,246-260`), and the best checkpoint is selected
only by validation NLL (`dirichlet_mdn/train.py:409-417`). Therefore the model
can be valid as a PDF while still violating the moments supplied by the LES
solver.

This contradicts `README.md:16-19`, the “matching the moments exactly” heading
in `docs/mdn_and_sampling_explained.tex:1542-1546`, and similar claims in the
proposal.

**Plan:** either implement and verify a genuinely constrained construction, or
state accurately that consistency is regularized rather than guaranteed.
Define acceptance tolerances and include them in model selection/evaluation.

#### H6. The main mathematical verification checks are circular

`verify.py:82-95` creates a target with `stack_predicted_moments` and then tests
`moment_loss`, which calls the same function again. It will pass even if the
moment algebra is wrong. The K=1 check at `verify.py:121-205` generates its
synthetic target with the same log-density/grid approximation it fits. The bin
check verifies agreement with the writer, not correctness of the writer.

The current closed-form mixture formulas in `losses.py:165-187` appear
mathematically correct; the problem is that the checks do not establish that.

**Plan:** compare moments and densities against independent Monte Carlo and
trusted analytical references, add hand-calculated cases, and test the
writer/loss boundary convention separately.

#### H7. Test evaluation omits the primary objective and part of the stated scientific protocol

`evaluate.py:94-103` constructs and stores `HistogramNLL`, but no evaluation
path calls it. `_build_metrics_df` at lines 178-190 reports L1, JSD, marginal
JSD, component count, and absolute moment errors only. The proposal requires
held-out NLL and conditional reaction-rate error
(`DirichletMDN_proposal.md:602-625`).

**Plan:** report per-record and aggregate test NLL using the corrected support
convention, summarize absolute/relative moment errors, and implement or
explicitly defer the downstream closure metric before making LES-accuracy
claims.

#### H8. Direct moments and histogram moments can disagree without affecting data acceptance

The writers compute and store both direct DNS moments and histogram-recovered
moments, plus a maximum discrepancy
(`EnsightPDFHybridDatasetMPI.py:429-456`). Training conditions and penalizes
against direct moments while fitting the histogram, but the loader neither
enforces nor reports the discrepancy (`dirichlet_mdn/data.py:38,65-73`).

**Plan:** audit discrepancy distributions by regime and boundary mass, set a
documented tolerance or quality flag, and ensure the likelihood and moment
targets are not materially contradictory.

### Moderate issues

#### M1. Precomputed split files have no identity, overlap, or coverage validation

`dirichlet_mdn/train.py:227-238` checks only that loaded indices are in range.
It does not reject overlap, missing/duplicate rows, or a different Parquet file
with the same row count. Generated splits also silently deduplicate indices at
`splits.py:198-200`.

**Plan:** save a dataset fingerprint and grouping policy with every split, then
assert disjointness, full intended coverage, uniqueness, and identity when it
is loaded.

#### M2. Dataset integrity checks sample too little

The grid is checked against only the first shard
(`dirichlet_mdn/data.py:109-134`), and histogram normalization is checked only
for the first accessed record in each shard (`data.py:169-176`). Shape,
finiteness, simplex mass, metadata-to-HDF5 IDs, and all later records are
trusted.

**Plan:** provide a complete offline dataset validator and require it before
training; keep lightweight runtime checks as a separate option.

#### M3. The saved scaler is not the exact preprocessing artifact used by the model

Training saves the original sklearn scaler, then `TorchScaler` independently
replaces every scale below `1e-6` with 1
(`dirichlet_mdn/train.py:113-116,265-272`). Repository evaluation repeats that
wrapper and is consistent, but an external deployment that follows the
documentation and calls the saved scaler directly can produce different model
inputs.

**Plan:** persist the exact effective center/scale transform used by the model
and make it the single deployment API.

#### M4. Non-uniform-grid plots display probability mass as if it were density

`dirichlet_mdn/plotting.py:54-67` contours per-cell probability mass. The
dataset sample plot uses `imshow` with a uniform extent
(`dirichlet_mdn/dataset_diversity.py:350-374`), ignoring non-uniform bin
geometry as well. Large cells can look artificially intense merely because
they cover more area.

**Plan:** plot mass divided by the physically clipped cell area and use actual
bin edges for geometry. Label mass and density plots distinctly.

#### M5. The variance-regime diagnostic reverses the physical interpretation

`dirichlet_mdn/dataset_diversity.py:152-162` computes
`variance / (mean * (1 - mean))`. Values near one describe maximum segregation
for a bounded scalar, but the code labels them `well_mixed`; values near zero
are the well-mixed/delta-at-mean limit.

**Plan:** rename the regimes, update the documentation, and validate the labels
on analytical delta and two-point distributions.

#### M6. The auxiliary moment MSE weights unlike moment scales equally in raw units

`moment_loss` averages squared errors in means, variances, and covariance
without normalization (`dirichlet_mdn/losses.py:206-214`). The documentation
itself notes that these quantities have very different scales
(`docs/mdn_and_sampling_explained.tex:1761-1769`). Mean errors can therefore
dominate the term intended to enforce all moments.

**Plan:** define dimensionless or separately weighted errors from physical
tolerances/training statistics and report every moment independently.

#### M7. Diversity selection and headline metrics retain ordering/weighting bias

The greedy retention logic processes runs, configurations, timesteps, and
spatial centers in fixed order. Once a moment bin is full, later sources face a
different admission rule (`EnsightPDFHybridDatasetMPI.py:388-411,665-760`).
The evaluator then reports row-weighted global means
(`dirichlet_mdn/evaluate.py:283-290`) from this deliberately stratified,
configuration-imbalanced retained library.

**Plan:** quantify selection stability under reordered input, preserve source
weights/provenance, and report both micro-averaged and macro/stratified metrics
with uncertainty.

#### M8. Universal/exact boundary-limit claims exceed the implemented model family

The implementation uses finite `K`, `alpha_min=1e-3`, and
`alpha_clip=1e3` (`dirichlet_mdn/model.py:23-30`). It can approximate many
interior densities but cannot exactly represent edge/vertex atoms or an exact
delta. Claims that it universally or exactly recovers every analytical limit
should be treated as asymptotic motivation unless demonstrated at the chosen
bounds and grid.

**Plan:** narrow the claims or add explicit discrete boundary components and
benchmark approximation error on the claimed limiting cases.

### Lower-priority robustness and reproducibility issues

#### L1. Run directories can collide despite documentation saying they cannot

`dirichlet_mdn/train.py:217-220` uses second-resolution timestamps and
`exist_ok=True`. Concurrent runs with the same tag can share and overwrite
artifacts.

**Plan:** fail on an existing directory or use a collision-resistant run ID.

#### L2. Important numerical and file-layout parameters are under-validated

The writers do not comprehensively validate bin count, `zst`, `nx % npx`,
filter width relative to the periodic halo, or monotonic/positive bin geometry.
Some invalid combinations fail late; others can produce misleading artifacts.

**Plan:** validate all geometry and decomposition invariants before reading DNS
files, and store the validated values in a schema-versioned manifest.

#### L3. There is no independent automated regression suite

The repository has a useful `verify.py` harness but no independent tests for
split membership, boundary-cell normalization, serial/MPI equivalence, stale
rank rejection, artifact schemas, evaluation metrics, or diagnostic physical
labels. The current blockers are exactly the kinds of defects such tests should
catch.

**Plan:** turn the corrected independent checks into a small deterministic
synthetic test suite runnable without proprietary DNS data.

## Proposed fix sequence

No step below should begin without approval.

1. **Establish independent ground truth and acceptance criteria**
   - Add synthetic scalar fields and analytical/Monte Carlo references.
   - Define tolerances for normalization, simplex support, moments, serial/MPI
     equivalence, and boundary-grid convergence.
   - Record dataset and split fingerprints.

2. **Restore one canonical dataset format**
   - Remove the duplicate serial program.
   - Make serial and MPI writers emit the same schema and manifest.
   - Make writes and cleanup exact, atomic, and non-destructive.

3. **Correct the simplex-boundary convention**
   - Measure off-mask mass in the current dataset.
   - Define physically clipped cell areas and integrated predicted cell mass.
   - Apply the same convention to writing, loading, NLL, rendering, moments,
     L1/JSD, and plots.
   - Decide from the audit whether existing datasets are salvageable.

4. **Correct experimental splitting**
   - Confirm whether run IDs are shared physical realizations.
   - Implement global run holdout if confirmed.
   - Implement a genuinely pure configuration holdout.
   - Validate and fingerprint all saved splits.

5. **Make MPI provenance safe and characterize selection**
   - Isolate phase-1 outputs per exact run configuration.
   - Reject stale, missing, extra, or incompatible ranks.
   - Test results over rank counts and either remove local information loss or
     document and bound the approximation.

6. **Strengthen artifact and preprocessing integrity**
   - Validate every shard and metadata/HDF5 mapping before training.
   - Persist the exact preprocessing transform.
   - Prevent run-directory collisions.

7. **Repair scientific evaluation and diagnostics**
   - Add test NLL and per-moment tolerance reporting.
   - Correct mixing-regime labels and non-uniform-grid plots.
   - Report balanced and population-relevant aggregates.
   - Add the reaction-rate/LES metric or narrow the stated scope.

8. **Regenerate and re-baseline**
   - Regenerate affected datasets and splits.
   - Retrain all reported models.
   - Re-run independent tests, analytical baselines, ablations, and OOD
     evaluation before using results in a publication or LES claim.

## Decisions needed before implementation

1. Does one `run_id` represent the same physical velocity realization for all
   scalar configurations?
2. Must the model represent true edge/vertex probability atoms, or is a
   quantitatively bounded continuous approximation acceptable?
3. Must MPI output be invariant to rank count, or is a measured approximation
   acceptable?
4. Should headline evaluation represent the naturally occurring DNS
   distribution, an equal-weight configuration average, or both?

## Approval gate

This document is the only intended change from the review. No production code,
tests, datasets, or documentation claims have been altered. Implementation
should start only after the findings, priorities, and decisions above are
approved.
