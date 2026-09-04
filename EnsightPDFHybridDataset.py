#!/usr/bin/env python3
"""Single-process entry point for the canonical LES-PDF dataset generator.

The implementation lives in :mod:`EnsightPDFHybridDatasetMPI` so serial and
MPI runs use exactly the same extraction, selection, file schema, and
validation rules. Running this file without ``mpirun`` creates an MPI world of
size one and therefore provides the single-process reference result.
"""

from EnsightPDFHybridDatasetMPI import main


if __name__ == "__main__":
    main()
