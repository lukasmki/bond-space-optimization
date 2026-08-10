"""E05 -- is 0.5 special, or does any value near it work?

The method's slogan is that a transition state is reached by asking for *half*
bonds.  That is either a sharp statement about where saddles sit in bond space
or a coincidence of these systems, and nothing in the repo distinguishes the
two.  This sweeps the target fraction and asks where the located structure is
actually closest to the saddle.

The real baseline is not 0.5 -- it is **the reference transition state's own
Mayer bond orders**, computed at the same level.  The gap between those and
0.5 is the quantitative statement of how good the heuristic is, and for early
or late transition states (Hammond) it will be systematically nonzero.  That
is a genuine limitation of the heuristic and is reported as one.

The one thing this experiment must not do is tune tau per reaction and then
quote the tuned result as the headline.  tau* is reported per reaction *and*
as a spread; E02's headline stays at tau = 0.5.

    sbatch --array=0-111 experiments/run.slurm 05_target_sharpness.py
"""

from __future__ import annotations

import argparse
import importlib

import numpy as np

import common
import drive
import targets as targets_mod
from common import JobSpec, RunRecord
from levels import PRODUCTION, VERIFY, smoke_variant
from registry import jobs_for
from systems import by_id

e01 = importlib.import_module("01_reference_states")

EXPERIMENT = "05_target_sharpness"


def reference_mayer_orders(rxn, reference) -> dict:
    """What the verified TS's own bond orders actually are on the changing pairs.

    If the method's heuristic were exactly right these would all be 0.5 on a
    1 -> 0 / 0 -> 1 pair.  Where they are not, 0.5 is asking for the wrong
    structure and no amount of optimiser tuning will fix it.
    """
    B = reference["ts_atoms"].get_array("bond-order")
    out = {}
    for i, j in rxn.changing_pairs():
        n_r = rxn.lewis_order("r", i, j)
        n_p = rxn.lewis_order("p", i, j)
        span = n_p - n_r
        observed = float(B[i, j])
        # Express as the fraction of the way from R to P that the reference
        # saddle actually sits, so pairs with different integer spans (a 2 -> 0
        # O2 dissociation vs a 1 -> 0 transfer) are comparable.
        fraction = (observed - n_r) / span if span else float("nan")
        out[f"{i}-{j}"] = {
            "mayer": observed,
            "lewis_r": n_r,
            "lewis_p": n_p,
            "fraction": float(fraction),
        }
    fractions = [v["fraction"] for v in out.values() if np.isfinite(v["fraction"])]
    return {
        "per_pair": out,
        "mean_fraction": float(np.mean(fractions)) if fractions else float("nan"),
        "deviation_from_half": (
            float(np.mean(fractions) - 0.5) if fractions else float("nan")
        ),
    }


def run_one(spec: JobSpec, rec: RunRecord, production, verify, args) -> None:
    rxn = by_id()[spec.params["reaction"]]
    tau = float(spec.params["tau"])
    mode = spec.params["mode"]
    reference = e01.load_reference(rxn.id)
    if reference is None:
        rec.status = "skipped"
        rec.metrics = {"reaction": rxn.id, "tau": tau, "mode": mode,
                       "excluded": True,
                       "exclusion_reason": "no verified reference from E01"}
        print("       skipped: no verified reference", flush=True)
        return

    if mode == "symmetric":
        bonds = targets_mod.l1_targets(rxn, "ts", half=tau)
    else:
        bonds = targets_mod.l1_asymmetric_targets(rxn, tau)

    steps = 30 if args.smoke else drive.DEFAULT_STEPS
    result = drive.drive(
        reference["r_atoms"], bonds, rxn.charge, rxn.spin, production, steps=steps
    )

    metrics = {
        "reaction": rxn.id,
        "category": rxn.category,
        "spin_changes": rxn.spin_changes,
        "tau": tau,
        "mode": mode,
        "targets": [list(b) for b in bonds],
        "drive_outcome": result.outcome,
        "drive_steps": result.steps,
        "drive_target_error": result.max_target_error,
        "reference_mayer": reference_mayer_orders(rxn, reference),
    }
    # Hoisted to the top level so they survive into the CSV: the figure needs
    # to draw where the verified saddle's own bond orders actually sit, which
    # is the real baseline for "is 0.5 special".
    metrics["reference_mayer_mean_fraction"] = metrics["reference_mayer"][
        "mean_fraction"
    ]
    metrics["reference_mayer_deviation_from_half"] = metrics["reference_mayer"][
        "deviation_from_half"
    ]

    # T4 is not scored here: a saddle refinement per tau per reaction would
    # dominate the cost of the whole suite, and the question this experiment
    # asks -- how close does tau get you -- is answered by RMSD, the true PES
    # gradient, and the curvature at the located point.
    verdict = drive.verify_ts(
        result.atoms, rxn, verify, reference, bonds=bonds, full=False
    )
    metrics.update(verdict.metrics)

    rec.metrics = metrics
    print(
        f"       tau={tau:.2f} ({mode}): rmsd={metrics['rmsd_heavy']:.3f} "
        f"|F|_PES={metrics['pes_residual_ev_ang']:.3f} n_imag={metrics['n_imaginary']} "
        f"| reference sits at {metrics['reference_mayer']['mean_fraction']:.2f}",
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
            jobs, args, n=2,
            prefer=lambda s: e01.load_reference(s.params["reaction"]) is not None,
        )
    common.main_loop(
        EXPERIMENT, jobs, args, production,
        lambda spec, rec: run_one(spec, rec, production, verify, args),
    )


if __name__ == "__main__":
    main()
