"""Lightweight sanity-check harness for the Dirichlet MDN pipeline.

Five checks, each runnable in under 30s on CPU:

  --check-bins                          : reconstructed bin grid == HDF5 shard
  --check-loss-gradient                 : single forward+backward, no NaN/Inf
  --check-moments-against-monte-carlo    : closed-form moments match samples
  --check-permutation-invariance        : NLL invariant under K-axis permutation
  --check-k1-fits-analytical-dirichlet  : K=1 MDN recovers a known Dirichlet to <2%

All checks run by default. Each prints PASS/FAIL. Exits non-zero on any FAIL.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import torch

from .bin_grid import bin_grid, validate_against_hdf5
from .data import HybridPDFDataset, collate_records
from .losses import (
    HistogramNLL,
    LossWeights,
    combined_loss,
    stack_predicted_moments,
)
from .model import DirichletMDN


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def check_bins(
    parquet_path: str,
    hdf5_dir: str,
    *,
    num_bins: int = 64,
    zst: float = 0.1,
    uniform_bins: bool = False,
) -> CheckResult:
    import pandas as pd
    import pyarrow.parquet as pq
    df = pq.read_table(parquet_path).to_pandas()
    shard_filename = str(df["hdf5_file"].iloc[0])
    full = f"{hdf5_dir.rstrip('/')}/{shard_filename}"
    ok, diffs = validate_against_hdf5(
        full, num_bins=num_bins, zst=zst, uniform=uniform_bins,
    )
    detail = "max diffs: " + ", ".join(f"{k}={v}" for k, v in diffs.items())
    return CheckResult("check_bins", ok, detail)


def check_loss_gradient(
    parquet_path: str,
    hdf5_dir: str,
    *,
    num_bins: int = 64,
    zst: float = 0.1,
    uniform_bins: bool = False,
) -> CheckResult:
    ds = HybridPDFDataset(
        parquet_path, hdf5_dir, input_moments=4,
        num_bins=num_bins, zst=zst, uniform_bins=uniform_bins,
    )
    try:
        if len(ds) == 0:
            return CheckResult("check_loss_gradient", False, "dataset is empty")
        batch = collate_records([ds[i] for i in range(min(8, len(ds)))])
    finally:
        ds.close()
    torch.manual_seed(0)
    model = DirichletMDN(n_in=4, K=8)
    nll_mod = HistogramNLL(ds.grid)
    pi, alpha = model(batch["moments"])
    out = combined_loss(
        pi, alpha, batch["histogram"], batch["moments"],
        nll_module=nll_mod, input_moments=4,
        weights=LossWeights(),
    )
    if not torch.isfinite(out.total):
        return CheckResult("check_loss_gradient", False,
                           f"non-finite total loss {out.total.item()}")
    out.total.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if any((not torch.isfinite(g).all()) for g in grads):
        return CheckResult("check_loss_gradient", False, "NaN/Inf in gradients")
    gn = float(sum(g.norm().item() ** 2 for g in grads) ** 0.5)
    detail = (f"nll={out.nll.item():.4g}  mom={out.mom.item():.4g}  "
              f"ent={out.entropy.item():.4g}  grad_norm={gn:.4g}")
    return CheckResult("check_loss_gradient", True, detail)


def check_moments_against_monte_carlo() -> CheckResult:
    """Compare model moment formulas with independent NumPy random samples."""
    rng = np.random.default_rng(1)
    pi_np = np.array([0.25, 0.75], dtype=np.float64)
    alpha_np = np.array([[2.0, 3.0, 4.0], [7.0, 2.0, 5.0]])
    n_samples = 400_000
    component = rng.choice(2, size=n_samples, p=pi_np)
    samples = np.empty((n_samples, 3), dtype=np.float64)
    for index in range(2):
        chosen = component == index
        samples[chosen] = rng.dirichlet(alpha_np[index], size=int(chosen.sum()))

    empirical = np.array([
        samples[:, 0].mean(),
        samples[:, 0].var(),
        samples[:, 1].mean(),
        samples[:, 1].var(),
        np.cov(samples[:, 0], samples[:, 1], ddof=0)[0, 1],
    ])
    predicted = stack_predicted_moments(
        torch.tensor(pi_np, dtype=torch.float64).unsqueeze(0),
        torch.tensor(alpha_np, dtype=torch.float64).unsqueeze(0),
        5,
    )[0].numpy()
    worst = float(np.max(np.abs(predicted - empirical)))
    return CheckResult(
        "check_moments_against_monte_carlo",
        worst < 2.5e-3,
        f"max absolute moment error={worst:.3e} over {n_samples} samples",
    )


def check_permutation_invariance(
    parquet_path: str,
    hdf5_dir: str,
    *,
    num_bins: int = 64,
    zst: float = 0.1,
    uniform_bins: bool = False,
) -> CheckResult:
    ds = HybridPDFDataset(
        parquet_path, hdf5_dir, input_moments=4,
        num_bins=num_bins, zst=zst, uniform_bins=uniform_bins,
    )
    try:
        if len(ds) == 0:
            return CheckResult(
                "check_permutation_invariance", False, "dataset is empty",
            )
        batch = collate_records([ds[i] for i in range(min(4, len(ds)))])
        nll_mod = HistogramNLL(ds.grid)
        torch.manual_seed(2)
        K = 6
        batch_size = batch["moments"].shape[0]
        pi = torch.softmax(torch.randn(batch_size, K), dim=-1)
        alpha = torch.rand(batch_size, K, 3) * 5.0 + 1.0

        perm = torch.tensor([3, 1, 5, 0, 4, 2])
        pi_p = pi[:, perm]
        alpha_p = alpha[:, perm, :]
        a = nll_mod(pi, alpha, batch["histogram"])
        b = nll_mod(pi_p, alpha_p, batch["histogram"])
        diff = float((a - b).abs().item())
        passed = diff < 1e-6
        detail = f"abs(NLL_orig - NLL_perm) = {diff:.2e}"
    finally:
        ds.close()
    return CheckResult("check_permutation_invariance", passed, detail)


def _synthetic_dirichlet_histogram(grid, alpha_true: torch.Tensor) -> torch.Tensor:
    """Build an independent Monte Carlo histogram from NumPy samples."""
    rng = np.random.default_rng(3)
    samples = rng.dirichlet(alpha_true.numpy(), size=750_000)
    counts, _, _ = np.histogram2d(
        samples[:, 0], samples[:, 1],
        bins=[grid.edges_a, grid.edges_b],
    )
    counts[:, :] = np.where(grid.simplex_mask, counts, 0.0)
    return torch.tensor(counts / counts.sum(), dtype=torch.float32)


def _method_of_moments_alpha(grid, hist: torch.Tensor) -> torch.Tensor:
    """Quick method-of-moments alpha init from a histogram.

    Uses E[Z_i] and Var[Z_1] to solve for alpha_0 = E[Z_1](1 - E[Z_1]) / Var[Z_1] - 1,
    then alpha_i = alpha_0 * E[Z_i]. Returns a 3-vector. Clipped to a sane range.
    """
    grid_a = torch.from_numpy(grid.cell_centroid_a).to(torch.float64)
    grid_b = torch.from_numpy(grid.cell_centroid_b).to(torch.float64)
    p = hist.to(torch.float64)
    E1 = (p * grid_a).sum()
    E2 = (p * grid_b).sum()
    E3 = (1.0 - E1 - E2).clamp(min=1e-6)
    V1 = (p * (grid_a - E1) ** 2).sum().clamp(min=1e-8)
    a0 = (E1 * (1.0 - E1) / V1 - 1.0).clamp(min=0.5, max=1e3)
    alpha0 = (a0 * torch.stack([E1, E2, E3])).clamp(min=1.0001)
    return alpha0.to(torch.float32)


def check_k1_fits_analytical_dirichlet(
    *,
    steps: int = 1500,
    lr: float = 0.05,
    rel_tol: float = 0.02,
) -> CheckResult:
    """Fit a K=1 Dirichlet MDN against a synthetic single-Dirichlet histogram.

    The model trunk is bypassed and we directly optimize alpha as a free
    parameter; this isolates the loss numerics from the moment-to-parameter
    map. The init uses method-of-moments on the synthetic histogram so the
    test exercises convergence to the optimum, not optimizer-warmup quality.
    Recovered alpha should match the truth to within ``rel_tol``.
    """
    torch.manual_seed(3)
    grid = bin_grid()
    nll_mod = HistogramNLL(grid)
    alpha_true = torch.tensor([2.0, 3.0, 5.0])
    hist = _synthetic_dirichlet_histogram(grid, alpha_true).unsqueeze(0)  # (1, N, N)

    alpha_init = _method_of_moments_alpha(grid, hist[0])
    raw = torch.nn.Parameter(torch.log(torch.expm1((alpha_init - 1.0).clamp(min=1e-4))))
    pi = torch.ones(1, 1)
    opt = torch.optim.Adam([raw], lr=lr)
    last = None
    for _ in range(steps):
        opt.zero_grad()
        alpha_pred = (torch.nn.functional.softplus(raw) + 1.0).view(1, 1, 3)
        loss = nll_mod(pi, alpha_pred, hist)
        loss.backward()
        opt.step()
        last = loss.item()
    alpha_pred = (torch.nn.functional.softplus(raw) + 1.0).detach()
    rel_err = ((alpha_pred - alpha_true).abs() / alpha_true).max().item()
    passed = rel_err < rel_tol
    detail = (f"alpha_true={alpha_true.tolist()}, "
              f"alpha_init={[round(x,3) for x in alpha_init.tolist()]}, "
              f"alpha_pred={[round(x,3) for x in alpha_pred.tolist()]}, "
              f"max_rel_err={rel_err:.4f}, final_nll={last:.4f}")
    return CheckResult("check_k1_fits_analytical_dirichlet", passed, detail)


def run_all(
    parquet_path: Optional[str],
    hdf5_dir: Optional[str],
    *,
    num_bins: int = 64,
    zst: float = 0.1,
    uniform_bins: bool = False,
) -> List[CheckResult]:
    results: List[CheckResult] = []
    if parquet_path and hdf5_dir:
        grid_args = {
            "num_bins": num_bins, "zst": zst, "uniform_bins": uniform_bins,
        }
        results.append(check_bins(parquet_path, hdf5_dir, **grid_args))
        results.append(check_loss_gradient(parquet_path, hdf5_dir, **grid_args))
        results.append(check_permutation_invariance(
            parquet_path, hdf5_dir, **grid_args,
        ))
    results.append(check_moments_against_monte_carlo())
    results.append(check_k1_fits_analytical_dirichlet())
    return results


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Dirichlet MDN sanity checks")
    p.add_argument("--parquet", default=None,
                   help="Path to the parquet metadata file (required for shard-bound checks)")
    p.add_argument("--hdf5-dir", default=None,
                   help="Directory containing HDF5 shards")
    p.add_argument("--num-bins", type=int, default=64)
    p.add_argument("--zst", type=float, default=0.1)
    p.add_argument("--uniform-bins", action="store_true")
    p.add_argument("--check-bins", action="store_true")
    p.add_argument("--check-loss-gradient", action="store_true")
    p.add_argument("--check-moments-against-monte-carlo", action="store_true")
    p.add_argument("--check-permutation-invariance", action="store_true")
    p.add_argument("--check-k1-fits-analytical-dirichlet", action="store_true")
    p.add_argument("--check-all", action="store_true", default=False)
    args = p.parse_args(argv)

    any_specific = any([
        args.check_bins,
        args.check_loss_gradient,
        args.check_moments_against_monte_carlo,
        args.check_permutation_invariance,
        args.check_k1_fits_analytical_dirichlet,
    ])
    run_everything = args.check_all or not any_specific

    results: List[CheckResult] = []
    grid_args = {
        "num_bins": int(args.num_bins),
        "zst": float(args.zst),
        "uniform_bins": bool(args.uniform_bins),
    }
    if run_everything or args.check_bins:
        if args.parquet and args.hdf5_dir:
            results.append(check_bins(args.parquet, args.hdf5_dir, **grid_args))
        else:
            results.append(CheckResult("check_bins", False,
                                       "skipped: --parquet and --hdf5-dir required"))
    if run_everything or args.check_loss_gradient:
        if args.parquet and args.hdf5_dir:
            results.append(check_loss_gradient(
                args.parquet, args.hdf5_dir, **grid_args,
            ))
        else:
            results.append(CheckResult("check_loss_gradient", False,
                                       "skipped: --parquet and --hdf5-dir required"))
    if run_everything or args.check_moments_against_monte_carlo:
        results.append(check_moments_against_monte_carlo())
    if run_everything or args.check_permutation_invariance:
        if args.parquet and args.hdf5_dir:
            results.append(check_permutation_invariance(
                args.parquet, args.hdf5_dir, **grid_args,
            ))
        else:
            results.append(CheckResult("check_permutation_invariance", False,
                                       "skipped: --parquet and --hdf5-dir required"))
    if run_everything or args.check_k1_fits_analytical_dirichlet:
        results.append(check_k1_fits_analytical_dirichlet())

    print()
    pass_count = sum(1 for r in results if r.passed)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"[{status}] {r.name}: {r.detail}")
    print()
    print(f"{pass_count}/{len(results)} checks passed.")
    return 0 if pass_count == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
