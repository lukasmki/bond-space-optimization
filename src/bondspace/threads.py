"""Thread configuration for PySCF under a scheduler allocation.

Lives in the library rather than beside a study because both ``data/`` and
``experiments/`` depend on it, and a copy in each is how the two drift apart.
``data/util.py`` re-exports these names, so the pipeline scripts are unchanged.
"""

import os
import warnings

# Thread-count environment variables for the BLAS libraries PySCF may be
# linked against.  These are only read when the library initialises, so
# setting them here is best-effort -- see configure_threads.
_BLAS_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def available_cpus() -> tuple[int, str]:
    """How many CPUs this process may actually use, and where that came from.

    On a cluster "all available" means the allocation, not the node: sizing
    from `os.cpu_count()` under a partial allocation oversubscribes the cores
    granted and slows the job down, so the scheduler's own count wins where
    it is published.
    """
    explicit = os.environ.get("BONDSPACE_THREADS")
    if explicit:
        return max(1, int(explicit)), "BONDSPACE_THREADS"

    # Scheduler allocations, most specific first.
    for var in (
        "SLURM_CPUS_PER_TASK",
        "SLURM_CPUS_ON_NODE",
        "PBS_NP",
        "NCPUS",
        "NSLOTS",
    ):
        value = os.environ.get(var)
        if value and value.isdigit():
            return max(1, int(value)), var

    # Respects CPU affinity, so it already reflects cgroup-based pinning.
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0))), "sched_getaffinity"

    return max(1, os.cpu_count() or 1), "os.cpu_count"


def configure_threads(n: int | None = None, *, verbose: bool = True) -> int:
    """Point PySCF at every CPU this process is allowed to use.

    PySCF parallelises its integral and exchange-correlation kernels with
    OpenMP and defaults to whatever `omp_get_max_threads()` reports, which on
    many clusters is 1 because the site module sets `OMP_NUM_THREADS=1`.  This
    sets it explicitly instead, and is the single largest lever available: the
    SCF and CPHF solves are the whole runtime of this pipeline.

    `lib.num_threads` applies at runtime and so works whenever it is called.
    The BLAS variables set alongside it do not -- those libraries read them
    once, at load, which has already happened by the time this module is
    imported.  Export them in the job script to be sure:

        export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

    Returns the thread count actually in effect.
    """
    from pyscf import lib

    if n is None:
        n, source = available_cpus()
    else:
        n, source = max(1, int(n)), "explicit argument"

    previous = os.environ.get("OMP_NUM_THREADS")
    for var in _BLAS_VARS:
        os.environ.setdefault(var, str(n))

    with warnings.catch_warnings():
        # PySCF warns when built without OpenMP; reported below instead.
        warnings.simplefilter("ignore", UserWarning)
        lib.num_threads(n=n)
    effective = lib.num_threads()

    if verbose:
        print(
            f"[threads] requesting {n} (from {source}); pyscf reports {effective}",
            flush=True,
        )
        if previous is not None and previous != str(n):
            print(
                f"[threads] note: OMP_NUM_THREADS was already {previous}; BLAS may "
                f"keep using that, since it is read at load time",
                flush=True,
            )
        if effective < n:
            print(
                f"[threads] WARNING: pyscf is using {effective} thread(s), not {n}. "
                "This build has no OpenMP support, so it cannot be threaded "
                "(common in macOS wheels; Linux cluster wheels are fine).",
                flush=True,
            )

    return effective
