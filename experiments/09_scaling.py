"""E09 -- what the Z-vector adjoint actually buys, as a function of size.

src/bondspace/README.md reports 2.1x at four atoms and 2.3x at eight,
"improving with size", with `make_h1` as the remaining floor.  Those are two
points and an extrapolation.  This measures the curve.

The benchmark cannot supply it: HCombustion is 2-6 atoms, where every cost is
trivial and the asymptotics are invisible.  So the ladder is **synthetic** --
water clusters give a homologous series at fixed element ratio, alkanes vary
the basis functions per atom -- and **no accuracy claim attaches to any of
it**.  This is a cost curve only.

Also measured here: `BondFluxCalculator.calculate` calls the *forces* scanner
and discards the nuclear gradient, using only the SCF object and the free PES
energy.  If that waste is material at the top of the ladder it is a concrete
implementation finding for the cost discussion.

All timings must come from one pinned node type -- the run record stores the
hostname and allocation so mixed hardware can be detected and excluded at
analysis time.  On macOS wheels PySCF has no OpenMP and everything here is
single-threaded, which `configure_threads` warns about; those numbers are not
comparable to a cluster's.

    sbatch --array=0-21 --exclusive experiments/run.slurm 09_scaling.py
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from ase import Atoms

import common
from common import JobSpec, RunRecord
from levels import PRODUCTION, Level
from registry import jobs_for

EXPERIMENT = "09_scaling"

REPEATS = 3


def build_system(name: str) -> tuple[Atoms, int, int]:
    """The cost ladder.  Returns (atoms, charge, spin)."""
    from ase.build import molecule

    if name.startswith("H2O-"):
        n = int(name.split("-")[1])
        water = molecule("H2O")
        cluster = Atoms()
        for k in range(n):
            piece = water.copy()
            # A ring rather than a line, so the cluster stays compact and the
            # integral screening behaves like a real system rather than a
            # chain of isolated monomers.
            angle = 2 * np.pi * k / max(n, 1)
            piece.translate((3.0 * np.cos(angle), 3.0 * np.sin(angle), 0.0))
            cluster += piece
        return cluster, 0, 0

    aliases = {"C4H10": "trans-butane", "C6H6": "C6H6"}
    return molecule(aliases.get(name, name)), 0, 0


def time_gradient(
    atoms: Atoms, charge: int, spin: int, level: Level, mode: str, bonds
) -> dict:
    """One BondFluxCalculator force evaluation, timed and decomposed."""
    from bondspace.ase import BondFluxCalculator

    work = atoms.copy()
    work.calc = BondFluxCalculator(
        bonds,
        charge=charge,
        spin=spin,
        thresh=1e-9,  # never early-stop: we are timing the full path
        ovlp_thresh=0.5,
        zvector=(mode == "zvector"),
        restrict_gradient=(mode == "restrict"),
        **level.calc_kwargs(),
    )
    start = time.perf_counter()
    forces = work.get_forces()
    elapsed = time.perf_counter() - start
    return {"seconds": elapsed, "forces": np.asarray(forces)}


def time_scanners(atoms: Atoms, charge: int, spin: int, level: Level) -> dict:
    """Energy-only vs energy+gradient scanner, to price the discarded gradient."""
    from pyscf import dft

    from bondspace.util import ase_to_pyscf

    mol = ase_to_pyscf(atoms, basis=level.basis, charge=charge, spin=spin)
    mol.verbose = 0

    def build():
        mf = dft.UKS(mol, xc=level.xc)
        if level.density_fit:
            mf = mf.density_fit()
        mf.conv_tol = level.conv_tol
        mf.level_shift = level.level_shift
        return mf

    start = time.perf_counter()
    build().kernel()
    energy_only = time.perf_counter() - start

    mf = build()
    start = time.perf_counter()
    mf.kernel()
    mf.nuc_grad_method().kernel()
    with_gradient = time.perf_counter() - start

    return {
        "energy_only_seconds": energy_only,
        "energy_gradient_seconds": with_gradient,
        "discarded_gradient_fraction": (
            (with_gradient - energy_only) / with_gradient if with_gradient else 0.0
        ),
    }


def run_one(spec: JobSpec, rec: RunRecord, args) -> None:
    name = spec.params["system"]
    basis = spec.params["basis"]
    atoms, charge, spin = build_system(name)
    level = Level(
        f"scaling-{basis}",
        PRODUCTION.xc,
        basis,
        PRODUCTION.conv_tol,
        PRODUCTION.level_shift,
        PRODUCTION.density_fit,
    )

    # One constrained bond, always the first two atoms: the objective's size
    # must not vary with the system, or the curve would confound the cost of
    # the gradient with the cost of the target list.
    bonds = [(0, 1, 1.0)]

    from bondspace.util import ase_to_pyscf

    mol = ase_to_pyscf(atoms, basis=basis, charge=charge, spin=spin)
    metrics: dict = {
        "system": name,
        "basis": basis,
        "natoms": len(atoms),
        "nbasis": int(mol.nao_nr()),
        "formula": atoms.get_chemical_formula(),
    }

    modes = ("zvector", "direct") if args.smoke else ("zvector", "direct", "restrict")
    repeats = 1 if args.smoke else REPEATS
    forces: dict[str, np.ndarray] = {}

    for mode in modes:
        times = []
        for _ in range(repeats):
            result = time_gradient(atoms, charge, spin, level, mode, bonds)
            times.append(result["seconds"])
            forces[mode] = result["forces"]
        metrics[f"{mode}_seconds_median"] = float(np.median(times))
        metrics[f"{mode}_seconds_all"] = times
        metrics[f"{mode}_cphf_solves"] = 1 if mode == "zvector" else 3 * len(atoms)

    if "direct" in forces and "zvector" in forces:
        # Exactness at production sizes, not just in the STO-3G unit tests:
        # does the krylov tolerance hold up as the system grows?
        metrics["zvector_vs_direct_max_abs"] = float(
            np.abs(forces["zvector"] - forces["direct"]).max()
        )
        metrics["speedup_direct_over_zvector"] = (
            metrics["direct_seconds_median"] / metrics["zvector_seconds_median"]
        )
    if "restrict" in forces and "direct" in forces:
        # What restrict_gradient throws away.  The calculator docstring records
        # 4.6e-2 on H2O2; as a curve over size this becomes the justification
        # for the default being off.
        metrics["restrict_discarded_force_max"] = float(
            np.abs(forces["direct"] - forces["restrict"]).max()
        )

    metrics.update(time_scanners(atoms, charge, spin, level))

    rec.metrics = metrics
    rec.counters = {"repeats": repeats}
    print(
        f"       {name}/{basis}: N={len(atoms)} nbf={metrics['nbasis']}  "
        f"zvector={metrics['zvector_seconds_median']:.2f}s "
        f"direct={metrics.get('direct_seconds_median', float('nan')):.2f}s "
        f"speedup={metrics.get('speedup_direct_over_zvector', float('nan')):.2f}x",
        flush=True,
    )


def main() -> None:
    parser = common.add_standard_args(argparse.ArgumentParser(description=__doc__))
    args = common.parse_args(parser)
    jobs = jobs_for(EXPERIMENT)
    if args.smoke:
        jobs = common.pick_smoke_jobs(
            jobs, args, prefer=lambda s: s.params["system"] == "H2O-1"
        )
    common.main_loop(
        EXPERIMENT,
        jobs,
        args,
        PRODUCTION,
        lambda spec, rec: run_one(spec, rec, args),
    )


if __name__ == "__main__":
    main()
