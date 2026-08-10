"""E03 -- L0 vs L1 vs L2: how much came from the reference product geometry?

This is the experiment that pre-empts the fatal objection.  data/1-run.py
derives its half-bond targets from 0.5*(B_R + B_P), which means the converged
bond-order matrix of the *product* -- and therefore the product geometry -- is
an input.  Starting the optimisation from the reactant does not make that
single-ended.

    L0  reference R and P bond-order matrices   -- the existing protocol
    L1  integer Lewis orders and the atom map   -- the paper's claim
    L2  the reactant and an enumerated move set -- product-agnostic

If L1 matches L0, the reference geometries were worth nothing and the method
is genuinely single-ended; that is the strongest single result available here
and belongs in the abstract.  If L1 collapses, the method needs the product
and the claim has to be rewritten.

L2 is scored differently, and deliberately so: it is handed no product at all,
so "success" is that the true transition state appears *somewhere* in the set
of saddles its enumerated moves produce.  Precision (how many of the other
saddles are chemically sensible) and the rank of the true TS by barrier height
are reported alongside, because a method that returns the right answer among
fifty wrong ones has not solved the problem.

    sbatch --array=0-56 experiments/run.slurm 03_information_ladder.py
"""

from __future__ import annotations

import argparse
import importlib

from ase import io

import common
import drive
import targets as targets_mod
from common import JobSpec, RunRecord
from levels import PRODUCTION, VERIFY, smoke_variant
from registry import jobs_for
from systems import by_id

e01 = importlib.import_module("01_reference_states")

EXPERIMENT = "03_information_ladder"

#: L2 enumerates every break/form/transfer available; on a six-atom system
#: that is a few dozen moves, each needing a drive and a verification.  Capped
#: so a single array task stays inside a wall-clock limit; the cap is recorded
#: in the metrics so a truncated enumeration is never mistaken for a complete
#: one.
MAX_L2_MOVES = 24


def _run_l0_l1(spec, rec, rxn, reference, rung, production, verify, args) -> None:
    bonds = targets_mod.targets_for(rxn, rung, "ts")
    steps = 30 if args.smoke else drive.DEFAULT_STEPS
    result = drive.drive(
        reference["r_atoms"], bonds, rxn.charge, rxn.spin, production, steps=steps
    )

    metrics = {
        "reaction": rxn.id,
        "rung": rung,
        "rung_information": targets_mod.RUNG_DESCRIPTION[rung],
        "category": rxn.category,
        "spin_changes": rxn.spin_changes,
        "targets": [list(b) for b in bonds],
        "n_targets": len(bonds),
        "drive_outcome": result.outcome,
        "drive_steps": result.steps,
        "drive_calls": result.calculate_calls,
        "drive_target_error": result.max_target_error,
    }
    verdict = drive.verify_ts(
        result.atoms, rxn, verify, reference, bonds=bonds,
        refine_steps=30 if args.smoke else 100,
        irc_steps=15 if args.smoke else 60,
    )
    metrics.update(verdict.metrics)

    frames = result.frames + [result.atoms]
    io.write(common.traj_path(EXPERIMENT, spec.key), frames, format="extxyz")
    rec.metrics = metrics
    print(
        f"       {rung}: tier={verdict.tiers.label} rmsd={metrics['rmsd_heavy']:.3f} "
        f"n_imag={metrics['n_imaginary']}",
        flush=True,
    )


def _run_l2(spec, rec, rxn, reference, production, verify, args) -> None:
    """Drive every enumerated move and see whether the true TS is among them."""
    start = reference["r_atoms"]
    moves = targets_mod.enumerate_l2_moves(start)
    truncated = len(moves) > MAX_L2_MOVES
    moves = moves[:MAX_L2_MOVES]
    steps = 20 if args.smoke else 120
    if args.smoke:
        moves = moves[:2]

    found = []
    for index, move in enumerate(moves):
        bonds = targets_mod.l2_targets(start, move)
        entry = {
            "move": move.describe(start),
            "targets": [list(b) for b in bonds],
        }
        try:
            result = drive.drive(
                start, bonds, rxn.charge, rxn.spin, production,
                steps=steps, keep_frames=False
            )
            entry["drive_outcome"] = result.outcome
            verdict = drive.verify_ts(
                result.atoms, rxn, verify, reference, bonds=bonds,
                refine_steps=30 if args.smoke else 60,
                irc_steps=10 if args.smoke else 40,
                full=True,
            )
            entry.update({
                "rmsd_heavy": verdict.metrics["rmsd_heavy"],
                "n_imaginary": verdict.metrics["n_imaginary"],
                "energy_ev": verdict.metrics["energy_ev"],
                "barrier_kcal": verdict.metrics["barrier_kcal"],
                "irc_connects": verdict.metrics.get("irc_connects", False),
                "irc_observed": verdict.metrics.get("irc_observed"),
                "highest_tier": verdict.metrics["highest_tier"],
            })
        except Exception as exc:  # noqa: BLE001 -- one move failing is data
            entry["error"] = f"{type(exc).__name__}: {exc}"
        found.append(entry)
        print(f"       move {index + 1}/{len(moves)}: {entry.get('move')}", flush=True)

    saddles = [e for e in found if e.get("n_imaginary") == 1]
    correct = [e for e in saddles if e.get("irc_connects")]
    # Rank by barrier: a chemist screening the output would look at the lowest
    # barriers first, so the rank of the true TS in that ordering is what says
    # whether the enumeration is usable in practice.
    ranked = sorted(saddles, key=lambda e: e.get("barrier_kcal", float("inf")))
    rank = next(
        (i + 1 for i, e in enumerate(ranked) if e.get("irc_connects")), None
    )

    rec.metrics = {
        "reaction": rxn.id,
        "rung": "L2",
        "rung_information": targets_mod.RUNG_DESCRIPTION["L2"],
        "category": rxn.category,
        "spin_changes": rxn.spin_changes,
        "n_moves_enumerated": len(moves),
        "enumeration_truncated": truncated,
        "n_saddles": len(saddles),
        "n_correct": len(correct),
        "recall": bool(correct),
        "precision": len(correct) / len(saddles) if saddles else 0.0,
        "true_ts_rank_by_barrier": rank,
        "moves": found,
    }
    print(
        f"       L2: {len(moves)} moves -> {len(saddles)} saddles, "
        f"true TS found={bool(correct)} rank={rank}",
        flush=True,
    )


def run_one(spec: JobSpec, rec: RunRecord, production, verify, args) -> None:
    rxn = by_id()[spec.params["reaction"]]
    rung = spec.params["rung"]
    reference = e01.load_reference(rxn.id)
    if reference is None:
        rec.status = "skipped"
        rec.metrics = {"reaction": rxn.id, "rung": rung, "excluded": True,
                       "exclusion_reason": "no verified reference from E01"}
        print("       skipped: no verified reference", flush=True)
        return
    if rung == "L2":
        _run_l2(spec, rec, rxn, reference, production, verify, args)
    else:
        _run_l0_l1(spec, rec, rxn, reference, rung, production, verify, args)


def main() -> None:
    parser = common.add_standard_args(argparse.ArgumentParser(description=__doc__))
    args = common.parse_args(parser)
    production = smoke_variant(PRODUCTION) if args.smoke else PRODUCTION
    verify = smoke_variant(VERIFY) if args.smoke else VERIFY
    jobs = jobs_for(EXPERIMENT)
    if args.smoke:
        jobs = common.pick_smoke_jobs(
            jobs, args, n=3,
            prefer=lambda s: e01.load_reference(s.params["reaction"]) is not None,
        )
    common.main_loop(
        EXPERIMENT, jobs, args, production,
        lambda spec, rec: run_one(spec, rec, production, verify, args),
    )


if __name__ == "__main__":
    main()
