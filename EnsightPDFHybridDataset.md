# EnsightPDFHybridDataset

## Purpose

This file keeps a running account of the implementation for `EnsightPDFHybridDataset.py`.

The routine is intended to generate LES-style training samples from DNS scalar fields using the existing Ensight framework, while writing the resulting dataset in a hybrid format:

- metadata in parquet for fast tabular queries
- histogram samples in HDF5 shards for efficient numeric I/O

Development target environment:

```bash
conda activate mlProp
```

No execution was performed during implementation.

## Implementation summary

The routine reuses the main ideas from `EnsightPDFml.py`:

- read ZA/ZB Ensight scalar fields
- pad periodically
- traverse the DNS volume with a translated cube filter
- build a `Z1-Z2` histogram per LES box

Input discovery now assumes `--folder` points to the parent directory that contains
`run_0000`, `run_0001`, ..., and that each run directory contains an `ensight-3D`
subdirectory holding the `ZA*` and `ZB*` folders.

The new routine changes the dataset layer:

- each retained LES box becomes one histogram-valued training record
- metadata are stored separately from the histogram arrays
- coarse moment-space balancing is applied during extraction
- shape novelty is tracked within each moment bin to avoid retaining too many near-duplicate PDFs
- periodic checkpoint outputs can be written during execution so partial data survive job timeout or queue eviction

## Important implementation details

### 1. Full box width CLI

The new script uses **full box widths** in the CLI.

Examples:

- `--box-widths 64` means a `64^3` LES box
- `--box-widths 64,128` means both `64^3` and `128^3` LES boxes

Internally the code converts each full width to a half-width for symmetric slicing.

This removes the ambiguity in the older script where `width` was used directly in slice bounds.

### 2. Input moments are computed directly from DNS cell values

The retained conditioning moments are computed directly from the box values, not reconstructed from the histogram.

Reason:

- the input moments should represent the LES-filtered DNS box as accurately as possible
- histogram moments are still computed and stored as a validation diagnostic

Stored diagnostics include:

- `hist_mean_a`, `hist_var_a`, `hist_mean_b`, `hist_var_b`, `hist_cov_ab`
- `moment_abs_err_max`

### 3. Histogram target remains the LES PDF

The output target is still a histogram-valued LES sample.

Each record contains:

- the 5 conditioning moments
- one `pdf_bins x pdf_bins` histogram PDF for `Z1, Z2`

`Z3` is not stored explicitly.

### 4. Greedy online retention

The implementation uses a **greedy online retention policy**.

For each candidate record:

1. bin it in coarse 5D moment space
2. compare its histogram to already-kept samples in that same moment bin
3. retain it if it is shape-novel enough or improves source diversity
4. if the bin is full, optionally replace the least novel kept sample

This avoids staging the full raw dataset before pruning.

### 4a. Periodic checkpoint output

The script now prints progress during the run and can periodically write a checkpoint dataset.

Checkpoint behavior:

1. The script prints the current `(run, scalar, timestep)` before reading each snapshot.
2. It prints the `box_width` and `stride` before traversing each filter width.
3. After each processed snapshot, it prints cumulative counts.
4. Every `--checkpoint-every-snapshots` processed snapshots, it rewrites a checkpoint parquet/HDF5 dataset using the current retained sample set.

Checkpoint files use a dataset tag of:

```text
<dataset-tag>_<checkpoint-tag-suffix>
```

By default that means a tag like `dns_pdf_hybrid_checkpoint`.

### 5. Shape novelty metric

Shape similarity is measured with Jensen-Shannon distance between flattened histograms.

Stored shape descriptors include:

- normalized entropy
- maximum bin probability
- normalized effective support
- mass near the three simplex corners
- mass near the three simplex edges
- mass near the simplex barycenter

These are stored as metadata for downstream filtering and analysis.

## Deviations from the original planning discussion

### Deviation 1: single-pass greedy selection instead of two-pass global balancing

Planned idea:

- extract all raw samples first
- then run a global moment-space and shape-space pruning pass

Implemented instead:

- a single-pass greedy selector with bounded per-bin retention

Reason:

- avoids holding the entire raw dataset in memory
- keeps the script usable on larger sweeps without needing a raw staging database

Consequence:

- retention depends on traversal order more than a full offline clustering pass would

### Deviation 2: HDF5 sharding is implemented, but metadata remain in one parquet file

This is intentional and consistent with the hybrid pattern.

Reason:

- parquet is the fast metadata query surface
- HDF5 shards are the numeric payload surface

### Deviation 3: square histogram storage is preserved

The histograms are stored as full square arrays of shape `(pdf_bins, pdf_bins)`.

Reason:

- keeps compatibility with existing square-grid workflows
- avoids custom triangular packing logic

Mitigation:

- the HDF5 files store a `simplex_mask` dataset under `/bins/simplex_mask`

## Exact output schema

### Parquet metadata file

Output file pattern:

```text
<output-dir>/<dataset-tag>_metadata.parquet
```

One row per retained LES sample.

Required columns:

- `sample_id` : int64
- `shard_id` : int32
- `local_index` : int32
- `hdf5_file` : string
- `scalar_config` : string
- `run_id` : int32
- `timestep` : int32
- `box_width` : int32
- `stride` : int32
- `center_i` : int32
- `center_j` : int32
- `center_k` : int32
- `n_dns_cells` : int32
- `mean_a` : float64
- `var_a` : float64
- `mean_b` : float64
- `var_b` : float64
- `cov_ab` : float64
- `hist_mean_a` : float64
- `hist_var_a` : float64
- `hist_mean_b` : float64
- `hist_var_b` : float64
- `hist_cov_ab` : float64
- `moment_abs_err_max` : float64
- `moment_bin_0` : int32
- `moment_bin_1` : int32
- `moment_bin_2` : int32
- `moment_bin_3` : int32
- `moment_bin_4` : int32
- `moment_bin_key` : string
- `shape_novelty` : float64
- `retain_reason` : string
- `shape_entropy` : float64
- `shape_max_prob` : float64
- `shape_effective_support` : float64
- `shape_corner_z1` : float64
- `shape_corner_z2` : float64
- `shape_corner_z3` : float64
- `shape_edge_z1_zero` : float64
- `shape_edge_z2_zero` : float64
- `shape_edge_z3_zero` : float64
- `shape_center_mass` : float64

### HDF5 shard files

Output file pattern:

```text
<output-dir>/<dataset-tag>_0000.h5
<output-dir>/<dataset-tag>_0001.h5
...
```

Each shard contains the following layout:

```text
/
  attrs:
    dataset_tag
    num_samples
    num_bins_a
    num_bins_b

  /bins
    edges_a          float64[num_bins+1]
    edges_b          float64[num_bins+1]
    centers_a        float64[num_bins]
    centers_b        float64[num_bins]
    simplex_mask     uint8[num_bins, num_bins]
    cell_area        float64[num_bins, num_bins]

  /data
    histograms       float32[n_samples_in_shard, num_bins, num_bins]
    sample_id        int64[n_samples_in_shard]
    local_index      int32[n_samples_in_shard]
```

### Manifest JSON

Output file pattern:

```text
<output-dir>/<dataset-tag>_manifest.json
```

The manifest records:

- input folder
- run ids
- scalar configs
- widths and strides
- histogram settings
- balancing settings
- output file names
- missing source paths
- overall counts

## CLI arguments

### Required in practice

- `-f, --folder`
  - folder containing the `run_00**` directories
  - each run directory must contain `ensight-3D/ZA*` and `ensight-3D/ZB*`

### Dataset coverage

- `--run-ids`
  - run ids or ranges
  - examples: `0:27`, `0,1,2,8`, `0:27:3`
- `--scalar-configs`
  - comma-separated scalar configuration names
  - default: `I1,I4,I5,L1,PI1,PI4,PI5,PL1`
- `--tstart`
  - starting timestep index
- `--tend`
  - ending timestep index, exclusive
- `--tjump`
  - timestep increment

### LES filter geometry

- `--box-widths`
  - full LES cube widths in cells
  - default: `64,128`
- `--strides`
  - stride values in cells
  - either one value for all widths or one value per width

### Histogram layout

- `--pdf-bins`
  - number of histogram cells in each dimension
  - default: `64`
- `--zst`
  - transition point for the non-uniform bins
  - default: `0.1`
- `--uniform-bins`
  - switch from non-uniform bins to uniform bins

### DNS layout

- `--nx`
  - global cube size in each direction
  - default: `256`
- `--npx`
  - domain decomposition count per direction in the Ensight files
  - default: `8`

### Output control

- `--output-dir`
  - directory for parquet, HDF5, and manifest outputs
- `--dataset-tag`
  - file prefix used for output naming
- `--hdf5-shard-size`
  - max number of retained samples per HDF5 file
- `--compression`
  - HDF5 compression name
- `--checkpoint-every-snapshots`
  - write checkpoint parquet/HDF5 outputs after this many processed snapshots
  - set to `0` to disable checkpoint writing
  - default: `1`
- `--checkpoint-tag-suffix`
  - suffix appended to `--dataset-tag` for checkpoint files
  - default: `checkpoint`

### Balancing and uniqueness control

- `--max-per-moment-bin`
  - cap on retained samples per coarse 5D moment bin
- `--moment-bins`
  - five comma-separated bin counts for `meanA,varA,meanB,varB,covAB`
- `--variance-limit`
  - clipping range used when binning variances
- `--covariance-limit`
  - clipping range used when binning covariance
- `--shape-threshold`
  - Jensen-Shannon novelty threshold

### Shape-feature geometry

- `--corner-threshold`
  - threshold for corner mass descriptors
- `--edge-threshold`
  - threshold for edge mass descriptors
- `--center-radius`
  - radius around the simplex barycenter for center-mass descriptor

### Error handling

- `--strict-missing`
  - abort on missing folders or timestep files instead of skipping them

## Example commands

### Example 1: one configuration, one run, one timestep, one width

```bash
conda activate mlProp
python EnsightPDFHybridDataset.py \
  --folder . \
  --run-ids 0 \
  --scalar-configs I1 \
  --box-widths 64 \
  --strides 32 \
  --tstart 17 \
  --tend 18 \
  --dataset-tag I1_run0_w64
```

### Example 2: all 27 runs, all scalar configurations, both LES widths

```bash
conda activate mlProp
python EnsightPDFHybridDataset.py \
  --folder . \
  --run-ids 0:27 \
  --scalar-configs I1,I4,I5,L1,PI1,PI4,PI5,PL1 \
  --box-widths 64,128 \
  --strides 32,64 \
  --tstart 17 \
  --tend 30 \
  --tjump 1 \
  --dataset-tag all_cases_w64_w128 \
  --output-dir hybrid_dataset_all
```

### Example 3: denser moment-space coverage with stronger pruning

```bash
conda activate mlProp
python EnsightPDFHybridDataset.py \
  --folder . \
  --run-ids 0:27 \
  --box-widths 64,128 \
  --strides 16,32 \
  --max-per-moment-bin 30 \
  --moment-bins 16,16,16,16,20 \
  --shape-threshold 0.04 \
  --dataset-tag dense_stride_pruned
```

### Example 4: keep current non-uniform histogram layout but store fewer samples per shard

```bash
conda activate mlProp
python EnsightPDFHybridDataset.py \
  --folder . \
  --run-ids 0:27 \
  --box-widths 64,128 \
  --strides 32,64 \
  --checkpoint-every-snapshots 2 \
  --hdf5-shard-size 10000 \
  --dataset-tag sharded_dataset
```

## Notes for future follow-up

Potential next improvements:

- add optional permutation augmentation after the base pipeline is validated
- add a second offline refinement pass that re-clusters retained samples inside each moment bin
- add optional width-conditioned output splits
- add a small summary parquet or CSV of counts by run, width, timestep, and retain reason

## Change log

- Added progress prints for each snapshot and width.
- Added periodic checkpoint parquet/HDF5 output so partial retained samples are available before the final write stage.

## Change log

- Updated input-path handling so the script loops over `run_00**` directories and
  reads scalar folders from `run_00**/ensight-3D` instead of assuming all `ZA*`
  and `ZB*` folders live directly under `--folder`.

Given:

- `--output-dir OUTPUT_DIR`
- `--dataset-name DATASET`

The routine writes:

- `OUTPUT_DIR/DATASET.raw.parquet`
- `OUTPUT_DIR/DATASET.raw.h5`
- `OUTPUT_DIR/DATASET.parquet`
- `OUTPUT_DIR/DATASET.h5`
- `OUTPUT_DIR/DATASET.summary.json`

If `--drop-raw` is used, the two `.raw.*` files are deleted after the final
pruned outputs are written.

## Exact Schema

### Raw and Final Parquet Metadata Schema

Each row corresponds to one LES histogram sample.

- `sample_id` : `int64`
- `source_sample_id` : `int64`
- `run_number` : `int32`
- `scalar_config` : `string`
- `timestep` : `int32`
- `filter_width` : `int32`
- `half_width` : `int32`
- `stride` : `int32`
- `center_i` : `int32`
- `center_j` : `int32`
- `center_k` : `int32`
- `mean_a` : `float64`
- `var_a` : `float64`
- `mean_b` : `float64`
- `var_b` : `float64`
- `cov_ab` : `float64`
- `hist_mean_a` : `float64`
- `hist_var_a` : `float64`
- `hist_mean_b` : `float64`
- `hist_var_b` : `float64`
- `hist_cov_ab` : `float64`
- `moment_bin_0` : `int32`
- `moment_bin_1` : `int32`
- `moment_bin_2` : `int32`
- `moment_bin_3` : `int32`
- `moment_bin_4` : `int32`
- `moment_bin_key` : `string`
- `shape_entropy_norm` : `float64`
- `shape_concentration` : `float64`
- `shape_peak_mass` : `float64`
- `shape_support_fraction` : `float64`
- `shape_corner_1_mass` : `float64`
- `shape_corner_2_mass` : `float64`
- `shape_corner_3_mass` : `float64`
- `shape_edge_1_mass` : `float64`
- `shape_edge_2_mass` : `float64`
- `shape_edge_3_mass` : `float64`
- `shape_center_mass` : `float64`
- `shape_diag12_mass` : `float64`
- `bin_cap` : `int32`
- `shape_js_threshold` : `float64`

Notes:

- In the raw parquet file, `sample_id == source_sample_id`.
- In the final parquet file, `sample_id` is reindexed from `0..N-1`, while `source_sample_id` points back to the raw extraction id.

### Raw and Final HDF5 Schema

Root datasets:

- `/histograms` : `float32`, shape `(nsamples, nbins, nbins)`
- `/moments` : `float32`, shape `(nsamples, 5)`
- `/histogram_moments` : `float32`, shape `(nsamples, 5)`
- `/shape_features` : `float32`, shape `(nsamples, 12)`
- `/bin_edges_a` : `float64`, shape `(nbins + 1,)`
- `/bin_edges_b` : `float64`, shape `(nbins + 1,)`
- `/z1_centers` : `float64`, shape `(nbins, nbins)`
- `/z2_centers` : `float64`, shape `(nbins, nbins)`

Mapping rule:

- Parquet row `sample_id = k` maps to HDF5 row `k` in the matching file.

This is true for both the raw outputs and the final retained outputs.

## Shape Bookkeeping

Within each coarse 5-D moment bin, the code keeps shape diversity using two layers:

1. A farthest-first ordering in low-dimensional shape-feature space.
2. A Jensen-Shannon distance threshold applied directly to histogram PDFs.

The current shape features are:

- normalized entropy
- concentration index `sum(p^2)`
- peak mass
- support fraction
- mass near the three simplex corners
- mass near the three simplex edges
- central-simplex mass
- mass near the `Z1 = Z2` diagonal

This is intended to avoid storing many nearly identical LES PDFs that also share
similar moment vectors.

## CLI Arguments

### Data Selection

- `-f, --folder`
  - Root folder containing `ZA*` and `ZB*` Ensight directories.
- `--run-numbers`
  - Comma-separated run numbers and inclusive ranges.
  - Default: `0-26`
- `--scalar-configs`
  - Comma-separated scalar configuration names.
  - Default: `I1,I4,I5,L1,PI1,PI4,PI5,PL1`
- `--tstart`
  - Starting timestep, inclusive.
- `--tend`
  - Ending timestep, exclusive.
- `--tjump`
  - Timestep increment.

### LES Filter Geometry

- `-W, --widths`
  - Comma-separated LES filter widths in DNS cells.
  - Important: these are full cube widths, not half-widths.
  - Default: `64,128`
- `-s, --stride`
  - Explicit box-translation stride.
  - If omitted, stride is computed from `width * --stride-scale`.
- `--stride-scale`
  - Default stride multiplier when `--stride` is not given.
  - Default: `0.5`

### Histogram Definition

- `-b, --bins`
  - Number of histogram bins in each scalar direction.
  - Default: `65`
- `--zst`
  - Stoichiometric mixture fraction used by the existing non-uniform binning map.
  - Default: `0.1`
- `--uniform-bins`
  - Use uniform bins instead of the non-uniform bin map.

### DNS Layout

- `--nx`
  - Global DNS cube size in one direction.
  - Default: `256`
- `--npx`
  - Processor tiles per direction used in the Ensight files.
  - Default: `8`

### Output Naming

- `--dataset-name`
  - Base output name.
  - Default: `les_pdf_dataset`
- `--output-dir`
  - Output directory.
  - Default: `./dataset_out`

### Moment-Space Balancing

- `--moment-bins`
  - Either one integer for all five dimensions or five comma-separated integers.
  - Default: `24,16,24,16,24`
- `--moment-ranges`
  - Five `min:max` groups separated by `;`.
  - Default: `0:1;0:0.25;0:1;0:0.25;-0.25:0.25`
- `--bin-cap`
  - Maximum retained samples per coarse moment bin.
  - Default: `50`

### Shape-Diversity Pruning

- `--shape-js-threshold`
  - Minimum Jensen-Shannon distance between retained PDFs inside one moment bin.
  - Default: `0.05`
- `--corner-threshold`
  - Corner-mass threshold in simplex coordinates.
  - Default: `0.10`
- `--edge-threshold`
  - Edge-mass threshold in simplex coordinates.
  - Default: `0.05`
- `--center-threshold`
  - Threshold used to define central-simplex mass.
  - Default: `0.20`
- `--diag12-threshold`
  - Distance to the `Z1 = Z2` line used in the diagonal feature.
  - Default: `0.05`

### I/O Behavior

- `--compression`
  - HDF5 compression filter name.
  - Default: `gzip`
- `--missing-policy`
  - `skip` or `error` for missing Ensight files.
  - Default: `skip`
- `--overwrite`
  - Overwrite existing outputs.
- `--drop-raw`
  - Remove raw outputs after the final pruned dataset is written.

## Usage Examples

### Example 1: Single width, one timestep, all 27 cases

```bash
conda activate mlProp
python EnsightPDFHybridDataset.py \
  --folder ./ensight-3D \
  --run-numbers 0-26 \
  --scalar-configs I1,I4,I5,L1,PI1,PI4,PI5,PL1 \
  --widths 64 \
  --stride 32 \
  --bins 65 \
  --tstart 17 \
  --tend 18 \
  --dataset-name dns_w64_t17 \
  --output-dir ./dataset_out \
  --overwrite
```

### Example 2: Two widths, multiple timesteps, default balancing

```bash
conda activate mlProp
python EnsightPDFHybridDataset.py \
  --folder ./ensight-3D \
  --run-numbers 0-26 \
  --widths 64,128 \
  --tstart 17 \
  --tend 28 \
  --tjump 2 \
  --dataset-name dns_multiwidth \
  --output-dir ./dataset_out \
  --overwrite
```

### Example 3: Denser coverage with stronger pruning

```bash
conda activate mlProp
python EnsightPDFHybridDataset.py \
  --folder ./ensight-3D \
  --run-numbers 0-26 \
  --widths 64 \
  --stride-scale 0.25 \
  --bin-cap 30 \
  --shape-js-threshold 0.08 \
  --dataset-name dns_dense_pruned \
  --output-dir ./dataset_out \
  --overwrite
```

### Example 4: Keep only final curated outputs

```bash
conda activate mlProp
python EnsightPDFHybridDataset.py \
  --folder ./ensight-3D \
  --run-numbers 0-26 \
  --widths 64,128 \
  --dataset-name dns_final_only \
  --output-dir ./dataset_out \
  --overwrite \
  --drop-raw
```

## Implementation Details Worth Remembering

1. The code computes the five LES moments directly from the raw DNS box values.
2. The histogram moments are also stored for quality checks and later diagnostics.
3. The largest requested filter width determines the periodic halo width for file reads.
4. One read of the Ensight files is reused across all requested widths for the same `(run, scalar, timestep)` combination.
5. The output datasets are sample-major, so metadata queries resolve sample ids first and HDF5 fetches only the needed histogram blocks.

## Deviations From The Earlier Plan

1. Filter-width semantics are user-facing full cube widths.
   - Earlier legacy code used a variable named `width` as the half-width in slicing.
   - This implementation defines `--widths 64,128` to mean actual `64^3` and `128^3` filters, matching the LES language used in the planning discussion.

2. Sample selection is implemented as a two-stage greedy diversity pass rather than a full clustering or medoid optimization.
   - Reason: it stays scalable and deterministic without materializing pairwise distances for all extracted samples.

3. Permutation augmentation is not implemented in the first version.
   - Reason: the immediate objective was a clean native-label dataset with strong bookkeeping and pruning.
   - This can be added later as a separate stage without changing the hybrid storage layout.

4. The first version writes both raw and curated outputs.
   - Reason: this preserves traceability and makes it possible to revisit pruning thresholds later.

## Future Extensions

1. Add optional permutation augmentation after the base dataset is validated.
2. Add case-aware or width-aware diversity quotas inside each moment bin.
3. Add approximate nearest-neighbor shape indexing if raw sample counts become very large.
4. Add train/validation/test split manifests keyed by run number, timestep, and filter width.