"""E02 -- the flagship: ask for half bonds from the reactant alone.

Starts from E01's **relaxed** reactant (not the dataset geometry, which would
be a free half-step of information toward the saddle), drives with L1 targets
-- integer Lewis orders and the atom mapping, nothing geometric -- and then
scores the result outside bond space.

The result is a five-tier ladder, the same runs scored five ways:

    T0  max|dB| < 0.5 over reference-bonded pairs   (the existing metric)
    T1  ... over ALL pairs                          (catches invented bonds)
    T2  ... plus heavy-atom RMSD to the verified TS < 0.25 Ang
    T3  ... plus exactly one imaginary frequency
    T4  ... plus Sella refines it and its IRC connects the correct R and P

The paper reports T4.  The number is expected to be well below the 12/19 the
existing bond-order metric gives, and that attrition is the point: a method
honest about a verified rate is more useful than one claiming an unverified
one.

Two things reported for every run, success or failure, because they answer the
claim without reference to any external structure:

  * ``pes_residual_ev_ang`` -- how stationary the located structure actually
    is.  A bond-flux endpoint has no reason to be a stationary point, since
    the calculator returns no PES force.
  * ``mode_overlap_bonds`` -- whether the imaginary mode is the motion that
    was requested.  Energy and RMSD cannot establish this on their own.

    uv run python experiments/02_ts_single_ended.py --smoke
    sbatch --array=0-18 experiments/run.slurm 02_ts_single_ended.py
"""

from __future__ import annotations

import argparse

from ase import io

import baselines
import common
import drive
import quality
import targets as targets_mod
from common import JobSpec, RunRecord
from levels import PRODUCTION, VERIFY, smoke_variant
from registry import jobs_for
from systems import by_id

import importlib

e01 = importlib.import_module("01_reference_states")

EXPERIMENT = "02_ts_single_ended"


def run_one(spec: JobSpec, rec: RunRecord, production, verify, args) -> None:
    rxn = by_id()[spec.params["reaction"]]
    reference = e01.load_reference(rxn.id)
    if reference is None:
        # A reference-side failure is not a method failure.  Recording it as
        # "skipped" rather than "ok with tier 0" keeps the two apart in E10 and
        # keeps the denominator honest.
        rec.status = "skipped"
        rec.metrics = {
            "reaction": rxn.id,
            "excluded": True,
            "exclusion_reason": "no verified reference from E01",
        }
        print("       skipped: no verified reference (run E01 first)", flush=True)
        return

    bonds = targets_mod.l1_targets(rxn, "ts")
    steps = 30 if args.smoke else drive.DEFAULT_STEPS

    result = drive.drive(
        reference["r_atoms"], bonds, rxn.charge, rxn.spin, production, steps=steps
    )

    metrics: dict = {
        "reaction": rxn.id,
        "equation": rxn.equation,
        "category": rxn.category,
        "spin_changes": rxn.spin_changes,
        "rung": "L1",
        "targets": [list(b) for b in bonds],
        "drive_outcome": result.outcome,
        "drive_converged": result.converged,
        "drive_steps": result.steps,
        "drive_calls": result.calculate_calls,
        "drive_target_error": result.max_target_error,
        "cphf_solves": result.calculate_calls,  # zvector: one solve per call
    }

    verdict = drive.verify_ts(
        result.atoms, rxn, verify, reference,
        bonds=bonds,
        refine_steps=30 if args.smoke else 100,
        irc_steps=15 if args.smoke else 60,
    )
    metrics.update(verdict.metrics)

    # The null baseline lives in this table, not only in E04: "would
    # interpolating the endpoints have done as well?" is the first question a
    # reader has, and it should be answerable from Table 1 alone.
    for name, guess in (
        ("midpoint", baselines.midpoint_guess(rxn)),
        ("idpp", baselines.idpp_guess(rxn)),
    ):
        metrics[f"{name}_rmsd_heavy"] = quality.permutation_rmsd(
            guess.atoms, reference["ts_atoms"], heavy_only=True
        )

    frames = result.frames + [result.atoms]
    if verdict.refined is not None:
        frames.append(verdict.refined)
    io.write(common.traj_path(EXPERIMENT, spec.key), frames, format="extxyz")

    rec.metrics = metrics
    rec.counters = {"calculate_calls": result.calculate_calls}
    print(
        f"       outcome={result.outcome}  tier={verdict.tiers.label}  "
        f"rmsd={metrics['rmsd_heavy']:.3f} (midpoint {metrics['midpoint_rmsd_heavy']:.3f})  "
        f"n_imag={metrics['n_imaginary']}  "
        f"|F|_PES={metrics['pes_residual_ev_ang']:.3f} eV/Ang",
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
            jobs, args, prefer=lambda s: e01.load_reference(s.params["reaction"]) is not None
        )
    common.main_loop(
        EXPERIMENT, jobs, args, production,
        lambda spec, rec: run_one(spec, rec, production, verify, args),
    )


if __name__ == "__main__":
    main()
