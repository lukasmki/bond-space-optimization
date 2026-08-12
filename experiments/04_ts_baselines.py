"""E04 -- what else could you have done?

A transition-state result with no baseline is not reviewable.  Every method
here is scored by the identical T0-T4 ladder from `quality.Tiers`, refined by
the same Sella call, and charged its **full** cost including the cost of
producing the guess -- CI-NEB's nine images are not free because the
refinement afterwards happens to be short.

    B0a  Cartesian midpoint of the aligned endpoints   L0   the null
    B0b  IDPP midpoint                                 L0   stronger null, ~free
    B1   Dimer from the reactant, bond-direction seed  L1   like-for-like
    B1r  Dimer, random direction (5 seeds)             L2   what the hint is worth
    B2   Sella straight from the reactant              L2   why this is hard
    B3   CI-NEB peak image -> Sella                    L0   the ceiling
    B4   Bond space L1 -> Sella                        L1   the hybrid

Every row carries the information rung it consumed, because "better" and
"better informed" are different claims and the accuracy-cost figure has to be
able to tell them apart.  The expected honest story is that CI-NEB wins on
success rate while using the product geometry and roughly an order of
magnitude more gradient evaluations.

This is where a referee looks hardest for unfairness, so the Dimer and NEB
parameters are tuned rather than left at defaults, and the tuning is recorded.

    sbatch --array=0-132 experiments/run.slurm 04_ts_baselines.py
"""

from __future__ import annotations

import argparse
import importlib


import baselines
import common
import drive
import quality
import targets as targets_mod
from common import JobSpec, RunRecord
from levels import PRODUCTION, VERIFY, smoke_variant
from registry import jobs_for
from systems import by_id

e01 = importlib.import_module("01_reference_states")

EXPERIMENT = "04_ts_baselines"

#: B1r's random-direction seeds.  Reported as a distribution, not a best-of:
#: quoting the luckiest seed would be exactly the unfairness this experiment
#: exists to avoid.
RANDOM_SEEDS = (0, 1, 2, 3, 4)


def _make_guess(method: str, rxn, reference, production, args):
    """Produce one TS guess, with its cost and its information rung."""
    steps = 20 if args.smoke else 60
    if method == "B0a":
        return [baselines.midpoint_guess(rxn)]
    if method == "B0b":
        return [baselines.idpp_guess(rxn)]
    if method == "B1":
        direction = targets_mod.bond_direction(
            reference["r_atoms"], targets_mod.l1_targets(rxn, "ts")
        )
        return [
            baselines.dimer_guess(rxn, production, direction=direction, steps=steps)
        ]
    if method == "B1r":
        seeds = RANDOM_SEEDS[:2] if args.smoke else RANDOM_SEEDS
        return [
            baselines.dimer_guess(rxn, production, seed=seed, steps=steps)
            for seed in seeds
        ]
    if method == "B2":
        return [
            baselines.sella_from_reactant(
                rxn, production, steps=30 if args.smoke else 100
            )
        ]
    if method == "B3":
        return [
            baselines.cineb_guess(
                rxn,
                production,
                n_images=5 if args.smoke else 9,
                steps=10 if args.smoke else 50,
            )
        ]
    if method == "B4":
        bonds = targets_mod.l1_targets(rxn, "ts")
        result = drive.drive(
            reference["r_atoms"],
            bonds,
            rxn.charge,
            rxn.spin,
            production,
            steps=30 if args.smoke else drive.DEFAULT_STEPS,
            keep_frames=False,
        )
        return [
            baselines.Guess(
                result.atoms,
                "bondspace-L1",
                "L1",
                pes_calls=result.calculate_calls,
                converged=result.converged,
                note=result.outcome,
            )
        ]
    raise ValueError(f"unknown baseline {method!r}")


def run_one(spec: JobSpec, rec: RunRecord, production, verify, args) -> None:
    rxn = by_id()[spec.params["reaction"]]
    method = spec.params["method"]
    reference = e01.load_reference(rxn.id)
    if reference is None:
        rec.status = "skipped"
        rec.metrics = {
            "reaction": rxn.id,
            "method": method,
            "excluded": True,
            "exclusion_reason": "no verified reference from E01",
        }
        print("       skipped: no verified reference", flush=True)
        return

    bonds = targets_mod.l1_targets(rxn, "ts")
    guesses = _make_guess(method, rxn, reference, production, args)

    runs = []
    for guess in guesses:
        entry = {
            "guess_method": guess.method,
            "rung": guess.rung,
            "guess_converged": guess.converged,
            "guess_pes_calls": guess.pes_calls,
            "note": guess.note,
            "guess_rmsd_heavy": quality.permutation_rmsd(
                guess.atoms, reference["ts_atoms"], heavy_only=True
            ),
        }
        try:
            verdict = drive.verify_ts(
                guess.atoms,
                rxn,
                verify,
                reference,
                bonds=bonds,
                refine_steps=30 if args.smoke else 100,
                irc_steps=15 if args.smoke else 60,
            )
            entry.update(verdict.metrics)
            # The fair cost currency: true-PES gradient evaluations, counting
            # both the guess and the refinement.  NEB makes many cheap calls
            # and bond space makes few expensive ones, so wall-clock alone
            # would flatter whichever happens to suit the hardware.
            entry["total_pes_calls"] = guess.pes_calls + verdict.metrics.get(
                "refine_gradient_calls", 0
            )
        except Exception as exc:  # noqa: BLE001
            # A baseline that blows up is a data point, not a crash -- B2 is
            # *expected* to fail, and that is the "why single-ended is hard"
            # datum.  The traceback is kept because E10 has to tell an SCF
            # divergence apart from converging to the wrong structure.
            import traceback as tb

            entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["traceback"] = tb.format_exc()
        runs.append(entry)

    best = min(
        (r for r in runs if "highest_tier" in r),
        key=lambda r: (-r["highest_tier"], r["rmsd_heavy"]),
        default=None,
    )
    rec.metrics = {
        "reaction": rxn.id,
        "method": method,
        "label": baselines.BASELINES[method]["label"],
        "rung": baselines.BASELINES[method]["rung"],
        "category": rxn.category,
        "spin_changes": rxn.spin_changes,
        "runs": runs,
        "n_runs": len(runs),
        **(
            {}
            if best is None
            else {
                "highest_tier": best["highest_tier"],
                "tiers": best["tiers"],
                "rmsd_heavy": best["rmsd_heavy"],
                "barrier_error_kcal": best["barrier_error_kcal"],
                "n_imaginary": best["n_imaginary"],
                "total_pes_calls": best["total_pes_calls"],
                "irc_connects": best.get("irc_connects", False),
            }
        ),
    }
    if best is not None:
        print(
            f"       {method}: tier={best['tier_label']} "
            f"rmsd={best['rmsd_heavy']:.3f} pes_calls={best['total_pes_calls']}",
            flush=True,
        )
    else:
        reasons = sorted({r.get("error", "?") for r in runs})
        print(
            f"       {method}: every run errored -- {'; '.join(reasons)}",
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
