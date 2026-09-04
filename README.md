# Multi-ScalarPDF

**Data-driven joint subgrid PDF closure for three-stream turbulent mixing via Dirichlet Mixture Density Networks.**

This project learns a mapping from LES-filtered subgrid *moments* to the full joint
subgrid PDF $P(Z_1, Z_2)$ of two mixture fractions in three-stream mixing. It replaces
the pixel-wise softmax DNN of Yellapantula et al. (2019) with a parametric mixture model
that lives natively on the 2-simplex and is evaluated against the analytical
bivariate-beta hierarchy of Perry & Mueller (2018).

Two halves:

1. **Dataset generation** — an MPI sampler (`EnsightPDFHybridDatasetMPI.py`) that reads
   DNS EnSight fields, applies box filters at multiple widths/strides, and builds a
   moment-stratified library of filter-cell PDFs stored as HDF5 + a Parquet metadata index.
2. **Dirichlet MDN** — the `dirichlet_mdn/` package: a mixture density network whose
   output is a mixture of $K$ Dirichlet components on the 2-simplex, so predicted PDFs are
   valid (non-negative, normalized, correctly supported). Moment agreement is encouraged
   during training and measured during evaluation; it is not exact by construction.

## Scientific context

Three-component passive mixing produces two independent mixture fractions $(Z_1, Z_2)$ with
$Z_1 + Z_2 + Z_3 = 1$, $Z_i \ge 0$, so the joint subgrid PDF is supported on the unit
triangle (the 2-simplex). For LES closure we need a model
$(\tilde Z_1, \tilde Z_2, \widetilde{Z_1''^2}, \widetilde{Z_2''^2}, \widetilde{Z_1''Z_2''}) \mapsto P(Z_1, Z_2)$
that is valid, moment-consistent, and accurate across every mixing regime (equal, favored,
layered, premixed) without needing to know the regime a priori.

The Dirichlet MDN outputs mixture weights $\pi_k$ (softmax) and concentration parameters
$\alpha_k \in \mathbb{R}^3_{>0}$ (softplus) for $K$ Dirichlet components, trained by
maximum likelihood with auxiliary moment-consistency and entropy terms. Design intent is in
`DirichletMDN_proposal.md`; the implementation plan and running log are in
`DirichletMDN_implementation_plan.md` and `DirichletMDN_implementation.md`.

## Repository layout

```
EnsightPDFHybridDatasetMPI.py   MPI sampler (Phase 1 per-rank sampling + Phase 2 merge)
EnsightPDFHybridDataset.py      Single-process reference sampler (slow; cross-checks)

dirichlet_mdn/                  The model package
  model.py          DirichletMDN architecture
  losses.py         NLL + moment-consistency + entropy losses
  data.py           Dataset/dataloader over Parquet index + HDF5 histograms
  splits.py         by-run-id train/val/test splitting
  bin_grid.py       2-simplex bin geometry
  train.py          training entry point
  evaluate.py       evaluation of a finished run
  verify.py         five correctness self-checks
  dataset_diversity.py   coverage/diagnostics audit
  preview_split.py  build + report a split without training
  metrics.py, plotting.py

docs/               Plain-English guide (LaTeX + compiled PDF)
diagnostics/        Dataset coverage reports and plots
splits/             Saved train/val/test split reports
*.md                Proposal, implementation plan, implementation log, dataset notes
pixi.toml           Environment + task runner definitions
```

## Data & artifacts (not in git)

The following are **intentionally excluded** via `.gitignore` because of size or copyright —
regenerate or obtain them separately:

| Path | What | How to get it |
|---|---|---|
| `hybrid_dataset_mpi/` (`*.h5`, `*.parquet`) | ~450 MB generated dataset | `pixi run sample-mpi <nranks>` |
| `hybrid_dataset_all/` | serial-sampler dataset | `pixi run sample-serial` |
| `dirichlet_mdn_runs/` (`*.pt`, `*.pkl`) | training runs / checkpoints | `pixi run train ...` |
| `.pixi/` | ~1.4 GB conda environment | `pixi install` |
| `1-s2.0-*.pdf` | copyrighted journal papers | from the publisher |

The raw DNS EnSight fields (pointed to by `DNS_ROOT`) are not part of this repo.

## Setup

Uses [pixi](https://pixi.sh) for a reproducible conda environment (Python 3.11, PyTorch,
h5py, mpi4py/OpenMPI, scikit-learn, tectonic, …):

```bash
pixi install
```

Default paths and knobs (`DNS_ROOT`, `DATASET_TAG`, `PARQUET`, `HDF5_DIR`, `RUNS_DIR`) are
set under `[activation.env]` in `pixi.toml` and can be overridden per command.

## Workflow

```bash
# 0. Sanity-check the plumbing (tiny 2-rank run)
pixi run sample-mpi-smoke

# 1. Generate the dataset (MPI; pass number of ranks)
pixi run sample-mpi 8

# 2. Inspect the dataset
pixi run verify        # five self-checks: bin grid, gradients, Monte Carlo moments,
                       # permutation invariance, K=1 recovery
pixi run test          # deterministic remediation regression tests
pixi run diversity     # coverage audit: cell counts, moment percentiles, plots
pixi run preview-split # build + report a train/val/test split (no training)

# 3. Train the Dirichlet MDN
pixi run train-smoke                       # 2-epoch end-to-end check
pixi run train 16 5 500 mpi_M5_K16_500ep   # train <K> <input_moments> <epochs> <tag>

# 4. Evaluate / monitor
pixi run evaluate dirichlet_mdn_runs/<run_dir>
pixi run tensorboard
```

New run manifests use artifact schema version 2. Historical runs are rejected by
default; trusted pre-v2 artifacts can be inspected with
`python -m dirichlet_mdn.evaluate --run-dir <run_dir> --allow-legacy-artifacts`.
Their stored legacy grid is preserved, so those metrics are not directly comparable
with version-2 runs.

### Task reference

| Task | Purpose |
|---|---|
| `sample-mpi <nranks>` | MPI sampler, Phase 1 (per-rank) + Phase 2 (merge) |
| `sample-mpi-smoke` | tiny MPI run to check plumbing |
| `merge-only` | re-run Phase 2 merge over an existing `_rank/` tree |
| `sample-serial` | single-process reference sampler |
| `verify` | five dataset correctness self-checks |
| `test` | deterministic geometry/sampling/split/loss/provenance tests |
| `diversity` | dataset coverage audit + plots |
| `preview-split <seed>` | build/report a split without training |
| `train <k> <moments> <epochs> <tag>` | train the Dirichlet MDN |
| `train-smoke` | 2-epoch training smoke test |
| `evaluate <run_dir>` | evaluate a finished run |
| `tensorboard` | TensorBoard over all runs |
| `doc` / `doc-clean` | compile / clean the plain-English guide |

## References

- Yellapantula et al. (2019) — pixel-wise ML subgrid PDF closure (baseline being replaced).
- Perry & Mueller (2018) — analytical bivariate-beta hierarchy for three-stream mixing.
- Bishop (1994) — Mixture Density Networks.

## Author

Shashank Yellapantula, National Renewable Energy Laboratory (NREL).
