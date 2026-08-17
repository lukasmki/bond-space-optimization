"""E08 -- robustness: how sensitive is the result to knobs a user must choose?

One factor at a time around E02's L1 configuration, on the pre-registered
subset in systems.ABLATION_SUBSET -- seven reactions, the eighth having been
dropped for cause once E01 found it barrierless.  T4 is not scored here --
hundreds of saddle refinements would dominate the cost of the whole suite --
so the metrics stop at T3, which is enough to see a knob change the answer.

The two rows a referee will demand:

  * **basis** -- Mayer bond orders are notoriously basis-dependent.  If the
    target 0.5 corresponds to different geometries in different bases, the
    half-bond heuristic is soft in a way no optimiser tuning can fix.  This is
    the biggest scientific risk in the whole method and it has never been
    measured here.
  * **perturbed start** -- does the single-ended claim survive a reactant that
    is not the exact reference geometry?  Ten seeded displacements at each of
    three magnitudes; the spread is the basin-of-attraction result.

Two rows are correctness checks rather than tuning, and a dependence in either
is a bug rather than a finding:

  * **level_shift** shifts nothing at convergence, so converged endpoints must
    be independent of it.
  * **zvector** is exact, not an approximation, so it must give the same
    endpoint as the direct path to solver tolerance.

With seven reactions most cells have wide intervals.  The analysis reports
them and refuses to order settings whose intervals overlap.

    sbatch --array=0-433%50 experiments/run.slurm 08_ablations.py
"""

from __future__ import annotations

import argparse
import importlib
from dataclasses import replace

from ase.optimize import BFGS, FIRE2, LBFGS

import common
import drive
import quality
import targets as targets_mod
from common import JobSpec, RunRecord
from levels import PRODUCTION, VERIFY, Level, smoke_variant
from registry import jobs_for
from systems import by_id

e01 = importlib.import_module("01_reference_states")

EXPERIMENT = "08_ablations"

OPTIMIZERS = {
    "FIRE2-0.02": (FIRE2, 0.02),
    "FIRE2-0.1": (FIRE2, 0.1),
    "FIRE2-0.2": (FIRE2, 0.2),
    "BFGS": (BFGS, 0.05),
    "LBFGS": (LBFGS, 0.05),
}


def _configure(knob, value, base_level: Level, args) -> tuple[Level, dict]:
    """Turn one (knob, value) pair into a level and a set of drive kwargs."""
    level = base_level
    kwargs: dict = {
        "steps": 30 if args.smoke else drive.DEFAULT_STEPS,
        "thresh": 0.05,
        "ovlp_thresh": 0.5,
        "zvector": True,
        "restrict_gradient": False,
        "maxstep": drive.DEFAULT_MAXSTEP,
        "fmax": 0.1,
        "optimizer": FIRE2,
        "fresh_guess": False,
    }
    if knob in ("reference", "perturb"):
        return level, kwargs
    # `replace` throughout: these were positional rebuilds, which stop at
    # whatever fields existed when they were written.  Adding `grid_level` to
    # `Level` silently reverted every ablation row to PySCF's default grid --
    # a one-factor-at-a-time study is worthless if the rows change two things.
    if knob == "basis":
        level = replace(base_level, name=f"{base_level.name}-{value}", basis=str(value))
    elif knob == "level_shift":
        shift = tuple(value) if isinstance(value, (list, tuple)) else float(value)
        level = replace(base_level, name=f"{base_level.name}-ls", level_shift=shift)
    elif knob == "density_fit":
        level = replace(
            base_level, name=f"{base_level.name}-nodf", density_fit=bool(value)
        )
    elif knob == "optimizer":
        optimizer, maxstep = OPTIMIZERS[str(value)]
        kwargs["optimizer"] = optimizer
        kwargs["maxstep"] = maxstep
    elif knob == "steps":
        kwargs["steps"] = int(value)
    elif knob in ("thresh", "ovlp_thresh", "fmax"):
        kwargs[knob] = float(value)
    elif knob in ("zvector", "restrict_gradient", "fresh_guess"):
        kwargs[knob] = bool(value)
    elif knob == "full_connectivity":
        pass  # handled at target-construction time
    else:
        raise ValueError(f"unknown knob {knob!r}")
    return level, kwargs


def _build_targets(rxn, knob, value):
    if knob == "full_connectivity":
        # The `-fullbonds` variant on disk, reproduced with provenance this
        # time: every bond constrained rather than only the changing ones.
        bonds = list(targets_mod.l1_targets(rxn, "ts"))
        changing = {(i, j) for i, j, _ in bonds}
        for i, j, order in rxn.lewis_r:
            if (i, j) not in changing:
                bonds.append((i, j, float(order)))
        return sorted(bonds)
    return targets_mod.l1_targets(rxn, "ts")


def _start_geometry(reference, knob, value, spec):
    if knob != "perturb":
        return reference["r_atoms"]
    atoms = reference["r_atoms"].copy()
    rng = spec.rng()
    atoms.set_positions(
        atoms.get_positions() + rng.normal(scale=float(value), size=(len(atoms), 3))
    )
    return atoms


def run_one(spec: JobSpec, rec: RunRecord, base_production, verify, args) -> None:
    rxn = by_id()[spec.params["reaction"]]
    knob = spec.params["knob"]
    value = spec.params.get("value")
    reference = e01.load_reference(rxn.id)
    if reference is None:
        rec.status = "skipped"
        rec.metrics = {
            "reaction": rxn.id,
            "knob": knob,
            "value": value,
            "excluded": True,
            "exclusion_reason": "no verified reference from E01",
        }
        print("       skipped: no verified reference", flush=True)
        return

    level, kwargs = _configure(knob, value, base_production, args)
    bonds = _build_targets(rxn, knob, value)
    start = _start_geometry(reference, knob, value, spec)

    result = drive.drive(start, bonds, rxn.charge, rxn.spin, level, **kwargs)

    metrics = {
        "reaction": rxn.id,
        "category": rxn.category,
        "spin_changes": rxn.spin_changes,
        "knob": knob,
        "value": value,
        "seed": spec.params.get("seed"),
        "level": level.as_dict(),
        "drive_kwargs": {
            k: (v if not callable(v) else v.__name__) for k, v in kwargs.items()
        },
        "n_targets": len(bonds),
        "drive_outcome": result.outcome,
        "drive_converged": result.converged,
        "drive_steps": result.steps,
        "drive_calls": result.calculate_calls,
        # max|dB| at the moment the optimiser stopped: the number that shows
        # whether a loose `thresh` is hiding error rather than removing it.
        "drive_target_error": result.max_target_error,
        "cphf_solves": result.calculate_calls
        * (1 if kwargs["zvector"] else 3 * rxn.natoms),
    }

    # Metrics are scored at VERIFY regardless of what the drive used, so a
    # basis ablation measures the geometry the basis produced rather than
    # re-scoring it in its own basis and comparing apples to oranges.
    verdict = drive.verify_ts(
        result.atoms, rxn, verify, reference, bonds=bonds, full=False
    )
    metrics.update(verdict.metrics)

    if knob == "restrict_gradient" and value:
        # Turn the calculator docstring's anecdotal 4.6e-2 spectator force into
        # a measured quantity: how much force is being discarded, and does the
        # endpoint move as a result?
        free = drive.drive(
            start,
            bonds,
            rxn.charge,
            rxn.spin,
            level,
            **{**kwargs, "restrict_gradient": False},
        )
        metrics["restricted_vs_free_rmsd"] = quality.permutation_rmsd(
            result.atoms, free.atoms, heavy_only=False
        )
        metrics["free_drive_steps"] = free.steps

    rec.metrics = metrics
    print(
        f"       {knob}={value}: outcome={result.outcome} "
        f"rmsd={metrics['rmsd_heavy']:.3f} n_imag={metrics['n_imaginary']} "
        f"dB_at_stop={result.max_target_error:.3f}",
        flush=True,
    )


def main() -> None:
    parser = common.add_standard_args(argparse.ArgumentParser(description=__doc__))
    args = common.parse_args(parser)
    production = smoke_variant(PRODUCTION) if args.smoke else PRODUCTION
    verify = smoke_variant(VERIFY) if args.smoke else VERIFY
    jobs = jobs_for(EXPERIMENT)
    if args.smoke:
        jobs = common.pick_smoke_jobs(
            jobs,
            args,
            n=2,
            prefer=lambda s: e01.load_reference(s.params["reaction"]) is not None,
        )
    common.main_loop(
        EXPERIMENT,
        jobs,
        args,
        production,
        lambda spec, rec: run_one(spec, rec, production, verify, args),
    )


if __name__ == "__main__":
    main()
