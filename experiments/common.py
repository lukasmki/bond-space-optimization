"""Shared harness: provenance, caching, resume, and SLURM array plumbing.

The invariant this module exists to enforce: **experiment scripts write JSON
and extxyz, nothing else.**  `analysis/aggregate.py` turns those into CSVs and
the figure scripts read only CSVs.  That is what makes every figure in the
paper regenerable in seconds during revisions, without a cluster.

The second invariant: **a failed run writes a record too.**  Silently missing
data points are how a benchmark lies about its denominator -- see E10, which
cannot distinguish "the method failed" from "we could not tell" unless every
failure carries its own status and traceback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np

HERE = Path(__file__).parent
REPO = HERE.parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
CACHE = RESULTS / "cache"

#: Bump when a RunRecord field changes meaning.  Records with a different
#: version are never treated as valid cache hits.
SCHEMA_VERSION = 1

#: One seed for the whole suite; per-job streams are derived from it.
SEED = 20260810


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------


def _git_state() -> dict:
    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args],
                cwd=REPO,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def _package_versions() -> dict:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for name in ("numpy", "scipy", "ase", "pyscf", "sella", "geometric"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[name] = None
    return versions


_PROVENANCE: dict | None = None


def provenance() -> dict:
    """Git state, package versions and host, computed once per process."""
    global _PROVENANCE
    if _PROVENANCE is None:
        _PROVENANCE = {
            "git": _git_state(),
            "packages": _package_versions(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "cpus_allocated": os.environ.get("SLURM_CPUS_PER_TASK"),
            "slurm_job": os.environ.get("SLURM_JOB_ID"),
        }
    return _PROVENANCE


# --------------------------------------------------------------------------
# job specs and hashing
# --------------------------------------------------------------------------


def canonical(obj: Any) -> Any:
    """A JSON-serialisable form with deterministic ordering.

    Used both for hashing and for writing records, so that a hash computed
    today matches one computed after an unrelated dict-ordering change.
    """
    if isinstance(obj, dict):
        return {k: canonical(obj[k]) for k in sorted(obj, key=str)}
    if isinstance(obj, (list, tuple)):
        return [canonical(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return canonical(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if hasattr(obj, "as_dict"):
        return canonical(obj.as_dict())
    return str(obj)


def hash_inputs(spec: dict, level_dict: dict) -> str:
    """Content hash of everything that determines a job's result.

    Deliberately **excludes** the git commit: an unrelated commit elsewhere in
    the repo must not invalidate a twelve-hour cluster result.  The commit is
    recorded in the run record, so provenance is not lost -- it is simply not
    part of the cache key.
    """
    payload = canonical(
        {"spec": spec, "level": level_dict, "schema": SCHEMA_VERSION}
    )
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class JobSpec:
    """One unit of work: everything a script needs to run one calculation."""

    experiment: str
    key: str  # unique within the experiment; becomes the filename
    params: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"experiment": self.experiment, "key": self.key,
                "params": canonical(self.params)}

    def rng(self) -> np.random.Generator:
        """A reproducible RNG for this job alone, independent of run order."""
        digest = hashlib.sha256(f"{self.experiment}/{self.key}".encode()).digest()
        return np.random.default_rng(SEED + int.from_bytes(digest[:4], "big"))


# --------------------------------------------------------------------------
# run records
# --------------------------------------------------------------------------

STATUSES = ("ok", "scf_fail", "step_limit", "exception", "skipped")


@dataclass
class RunRecord:
    experiment: str
    key: str
    schema_version: int = SCHEMA_VERSION
    inputs_hash: str = ""
    status: str = "ok"
    spec: dict = field(default_factory=dict)
    level: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    counters: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    restarted: bool = False
    traceback: str | None = None

    def as_dict(self) -> dict:
        return canonical(asdict(self))


def record_path(experiment: str, key: str) -> Path:
    return RESULTS / "runs" / experiment / f"{key}.json"


def traj_path(experiment: str, key: str) -> Path:
    """Where a job's trajectory goes, with the directory guaranteed to exist.

    Callers hand this straight to `ase.io.write`, which does not create parent
    directories -- and a job that computed a whole path and then died writing
    it out is the most expensive possible way to fail.
    """
    path = RESULTS / "runs" / experiment / f"{key}.xyz"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_record(rec: RunRecord) -> Path:
    """Write a record atomically.

    SLURM preemption partway through a write leaves a truncated file that is
    indistinguishable from a completed job to a skip-if-exists check -- the
    exact failure mode data/1-run.py has today.  Write to a temporary name in
    the same directory and rename, which is atomic on every filesystem this
    will run on.
    """
    path = record_path(rec.experiment, rec.key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec.as_dict(), indent=2))
    os.replace(tmp, path)
    return path


def read_record(experiment: str, key: str) -> dict | None:
    path = record_path(experiment, key)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        # A record from a job killed mid-write before atomic writes landed.
        return None


def load_records(experiment: str) -> list[dict]:
    """Every record for an experiment, ordered by key."""
    directory = RESULTS / "runs" / experiment
    if not directory.exists():
        return []
    out = []
    for path in sorted(directory.glob("*.json")):
        try:
            out.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            continue
    return out


def is_complete(spec: JobSpec, inputs_hash: str, *, retry_failed: bool) -> bool:
    """Whether this job can be skipped."""
    rec = read_record(spec.experiment, spec.key)
    if rec is None:
        return False
    if rec.get("schema_version") != SCHEMA_VERSION:
        return False
    if rec.get("inputs_hash") != inputs_hash:
        return False
    if retry_failed and rec.get("status") != "ok":
        return False
    return True


# --------------------------------------------------------------------------
# budgets: timing and call counting
# --------------------------------------------------------------------------


class Budget:
    """Wall-clock and calculator-call accounting for one job.

    CPHF solves are *derived*, not instrumented: the direct path costs one
    solve per perturbed atom (3N perturbations solved as a block, one solve
    per atom) and the adjoint path costs exactly one, regardless of size.  The
    code path fixes the count, so counting calls and multiplying is exact and
    needs no monkeypatching of `bondspace.bond`.  E09 instruments PySCF
    directly, under a context manager, and only there.
    """

    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.calls: dict[str, int] = {}
        self.sections: dict[str, float] = {}

    def count(self, name: str, n: int = 1) -> None:
        self.calls[name] = self.calls.get(name, 0) + n

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.sections[name] = self.sections.get(name, 0.0) + (
                time.perf_counter() - start
            )

    def attach_counter(self, calc, name: str) -> None:
        """Count `calculate()` invocations on an ASE calculator in place."""
        original = calc.calculate
        budget = self

        def counted(*args, **kwargs):
            budget.count(name)
            return original(*args, **kwargs)

        calc.calculate = counted  # type: ignore[method-assign]

    def cphf_solves(self, *, n_calls: int, natoms: int, zvector: bool) -> int:
        return n_calls * (1 if zvector else 3 * natoms)

    def finish(self) -> tuple[dict, dict]:
        timings = {"wall_seconds": time.perf_counter() - self.t0, **self.sections}
        return timings, dict(self.calls)


# --------------------------------------------------------------------------
# electronic-structure cache
# --------------------------------------------------------------------------


def geometry_key(atoms, charge: int, spin: int, level_dict: dict, tag: str) -> str:
    """Cache key from *exact* coordinate bytes, never rounded values.

    Rounding would collide geometries that differ below the rounding scale and
    silently serve one result for another; at 1e-4 Bohr displacements that is
    precisely the regime the finite-difference tests operate in.
    """
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(atoms.get_positions(), dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(atoms.get_atomic_numbers(), dtype=np.int64).tobytes())
    h.update(json.dumps(canonical(level_dict), sort_keys=True).encode())
    h.update(f"{charge}/{spin}/{tag}".encode())
    return h.hexdigest()[:32]


class ResultCache:
    """npz-backed cache of single-point results, keyed on exact geometry.

    Pays for itself in E04 and E06, where several baselines revisit the same
    structures.  ``verify_fraction`` recomputes a sample of hits and asserts
    agreement, so a corrupted or mis-keyed cache is caught rather than quietly
    poisoning a figure.
    """

    def __init__(self, enabled: bool = True, verify_fraction: float = 0.0) -> None:
        self.enabled = enabled
        self.verify_fraction = verify_fraction
        self.hits = 0
        self.misses = 0
        self.verified = 0
        self._rng = np.random.default_rng(SEED)
        if enabled:
            CACHE.mkdir(parents=True, exist_ok=True)

    def get_or_compute(self, key: str, compute: Callable[[], dict]) -> dict:
        if not self.enabled:
            return compute()
        path = CACHE / f"{key}.npz"
        if path.exists():
            self.hits += 1
            stored = {k: v for k, v in np.load(path, allow_pickle=False).items()}
            if self._rng.random() < self.verify_fraction:
                fresh = compute()
                for k, v in fresh.items():
                    if not np.allclose(np.asarray(v), stored[k], atol=1e-8):
                        raise RuntimeError(
                            f"cache verification failed for {key!r} on {k!r}"
                        )
                self.verified += 1
            return stored
        self.misses += 1
        result = compute()
        tmp = path.with_suffix(".npz.tmp")
        np.savez(tmp, **{k: np.asarray(v) for k, v in result.items()})
        os.replace(tmp, path)
        return result

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses, "verified": self.verified}


# --------------------------------------------------------------------------
# trajectory checkpointing
# --------------------------------------------------------------------------


class TrajectoryWriter:
    """Append frames to an extxyz file as they are produced.

    extxyz round-trips the ``bond-order`` array and the JSON ``connectivity``
    info field, which `data/` already relies on.  Writing incrementally means a
    preempted job leaves a usable partial trajectory rather than nothing.

    A run restarted from a partial trajectory is flagged, and flagged runs are
    **excluded from every timing statistic**: FIRE2's velocity state is lost
    across a restart, so neither the step count nor the wall-clock is
    comparable to an uninterrupted run.
    """

    def __init__(self, path: Path, tag: str = "") -> None:
        from ase import io  # noqa: F401  (import cost paid once, lazily)

        self.path = path
        self.tag = tag
        self.n = 0
        path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, atoms) -> None:
        from ase import io

        image = atoms.copy()
        if self.tag:
            image.info["id"] = self.tag
        # Carry the calculator's published arrays, which `copy()` keeps, and
        # the results, which it does not.
        io.write(self.path, image, format="extxyz", append=self.path.exists())
        self.n += 1

    def frames(self) -> list:
        from ase import io

        if not self.path.exists():
            return []
        return io.read(self.path, index=":")  # type: ignore[return-value]


# --------------------------------------------------------------------------
# CLI and SLURM array plumbing
# --------------------------------------------------------------------------


def add_standard_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--index", type=int, default=None,
                        help="run only job N (0-based); defaults to "
                             "SLURM_ARRAY_TASK_ID when set")
    parser.add_argument("--chunk", type=int, default=1,
                        help="jobs per --index, for when array size limits bite")
    parser.add_argument("--list", action="store_true",
                        help="print the job count and exit (for array bounds)")
    parser.add_argument("--dry-run", action="store_true",
                        help="list the jobs that would run, without running")
    parser.add_argument("--force", action="store_true",
                        help="recompute even completed jobs")
    parser.add_argument("--force-failed", action="store_true",
                        help="recompute only jobs whose status is not ok")
    parser.add_argument("--smoke", action="store_true",
                        help="one job at a minimal basis with tight caps")
    parser.add_argument("--no-cache", action="store_true",
                        help="bypass the single-point cache")
    parser.add_argument("--verify-cache", type=float, default=0.0,
                        metavar="FRACTION",
                        help="recompute this fraction of cache hits and assert "
                             "agreement")
    parser.add_argument("--threads", type=int, default=None,
                        help="override the thread count")
    return parser


#: True once a `--smoke` invocation has parsed its arguments.  Modules that
#: gate on data quality consult this so that a smoke run still exercises the
#: code path instead of skipping (see `01_reference_states.load_reference`).
#: It must never widen what a *production* run accepts.
SMOKE = False


def parse_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """`parser.parse_args()`, plus recording flags other modules need to see.

    Job selection happens before `setup()` runs -- a script asks E01 which
    reactions have references in order to pick its smoke jobs -- so the smoke
    flag has to be visible from the moment it is parsed.
    """
    global SMOKE
    args = parser.parse_args()
    SMOKE = bool(getattr(args, "smoke", False))
    return args


def explicit_index(args: argparse.Namespace) -> int | None:
    """The array index this invocation was pinned to, if any.

    `--index` and `SLURM_ARRAY_TASK_ID` are the same statement; a script has
    to be able to ask whether one was made before it narrows the job list on
    its own account (see `pick_smoke_jobs`).
    """
    if args.index is not None:
        return args.index
    env = os.environ.get("SLURM_ARRAY_TASK_ID")
    if env is not None and env.isdigit():
        return int(env)
    return None


def select_jobs(jobs: Sequence[JobSpec], args: argparse.Namespace) -> list[JobSpec]:
    """Apply --index/--chunk/SLURM_ARRAY_TASK_ID to an ordered job list.

    The job list must be deterministic and ordered, since the array index is
    the only thing linking a task to its work.  `registry.py` guarantees that.
    """
    index = explicit_index(args)
    if index is None:
        return list(jobs)
    start = index * args.chunk
    return list(jobs[start:start + args.chunk])


@contextmanager
def run_job(
    spec: JobSpec,
    level,
    args: argparse.Namespace,
) -> Iterator[RunRecord | None]:
    """Run one job with resume, provenance, timing and failure capture.

    Yields ``None`` when the job is skipped, so the caller can `continue`.
    On an exception the record is written with ``status`` set and the full
    traceback preserved -- E10 needs the traceback to tell an SCF blow-up
    apart from converging to the wrong structure.
    """
    inputs_hash = hash_inputs(spec.as_dict(), level.as_dict())
    retry_failed = args.force_failed
    if not args.force and is_complete(spec, inputs_hash, retry_failed=retry_failed):
        print(f"  skip {spec.experiment}/{spec.key} (complete)", flush=True)
        yield None
        return
    if args.dry_run:
        print(f"  would run {spec.experiment}/{spec.key}", flush=True)
        yield None
        return

    rec = RunRecord(
        experiment=spec.experiment,
        key=spec.key,
        inputs_hash=inputs_hash,
        spec=spec.as_dict(),
        level=level.as_dict(),
        provenance=provenance(),
    )
    print(f"  run  {spec.experiment}/{spec.key}", flush=True)
    started = time.perf_counter()
    try:
        yield rec
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # noqa: BLE001 -- recorded, not swallowed
        rec.status = _classify_exception(exc)
        rec.traceback = traceback.format_exc()
        print(f"       FAILED [{rec.status}] {exc}", flush=True)
    finally:
        rec.timings.setdefault("wall_seconds", time.perf_counter() - started)
        write_record(rec)


def _classify_exception(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "scf" in text and ("converg" in text or "diverg" in text):
        return "scf_fail"
    if "convergencefailure" in text or "not converged" in text:
        return "scf_fail"
    return "exception"


def setup(args: argparse.Namespace) -> None:
    """Threading and output directories.  Call once at the top of a script."""
    sys.path.insert(0, str(HERE))
    from bondspace.threads import configure_threads

    configure_threads(args.threads)
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def main_loop(
    experiment: str,
    jobs: Sequence[JobSpec],
    args: argparse.Namespace,
    level,
    body: Callable[[JobSpec, RunRecord], None],
) -> None:
    """The shape every experiment script shares."""
    if args.list:
        print(len(jobs))
        return
    setup(args)
    selected = select_jobs(jobs, args)
    print(f"[{experiment}] {len(selected)} of {len(jobs)} jobs", flush=True)
    for spec in selected:
        with run_job(spec, level, args) as rec:
            if rec is None:
                continue
            body(spec, rec)


def pick_smoke_jobs(
    jobs: Sequence[JobSpec],
    args: argparse.Namespace,
    n: int = 1,
    prefer: Callable[[JobSpec], bool] | None = None,
) -> list[JobSpec]:
    """The first `n` jobs, preferring ones whose prerequisites are satisfied.

    A smoke run that silently skips because its E01 reference is missing has
    exercised nothing.  `prefer` lets a script say "pick a job that will
    actually do work"; if none qualifies the first jobs are returned anyway,
    so the run still reports the missing prerequisite rather than passing
    vacuously.

    An explicit `--index` (or `SLURM_ARRAY_TASK_ID`) wins: truncating first
    and indexing after means `--smoke --index 2` selects nothing, which is how
    you end up unable to smoke-test the one reaction you care about.
    """
    if explicit_index(args) is not None:
        return list(jobs)
    if prefer is not None:
        eligible = [spec for spec in jobs if prefer(spec)]
        if eligible:
            return eligible[:n]
    return list(jobs[:n])


def iter_progress(items: Iterable, label: str) -> Iterator:
    for i, item in enumerate(items):
        print(f"    [{label}] {i}", flush=True)
        yield item
