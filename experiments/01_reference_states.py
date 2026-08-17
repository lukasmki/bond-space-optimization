"""E01 -- level-consistent reference states.  Run this first.

The HCombustion geometries come from an external dataset at an unknown level
of theory, and are almost certainly not stationary points under
wB97X-D3/cc-pVTZ/DF.  Until they are re-optimised here, every RMSD and every
barrier in the suite would be measuring the level-of-theory shift as much as
the method, and there would be no way to tell the two apart.

For each reaction this relaxes R and P, refines the dataset TS to a genuine
first-order saddle with Sella, and runs an IRC to confirm the saddle actually
connects the intended endpoints.  A reaction whose TS cannot be verified is
**excluded from every downstream TS metric** -- and, crucially, is recorded as
a reference-side failure so E10 can tell "the method failed" apart from "we
could not tell".

`systems.BARRIERLESS` -- the five single-bond homolyses -- is skipped outright.
Those reactions have no first-order saddle to find at a fixed multiplicity, so
refinement walks down the dissociation coordinate until the SCF gives up; the
first run spent 19-53 minutes each proving it.  They are recorded as excluded
with a reason rather than run.

It also reports every place where round(Mayer B) on the reference frames
disagrees with the Lewis integers in systems.py.  Those are the reactions
where the L0 and L1 rungs genuinely ask for different chemistry.

    uv run python experiments/01_reference_states.py --smoke
    sbatch --array=0-18 experiments/run.slurm 01_reference_states.py
"""

from __future__ import annotations

import argparse

import numpy as np
from ase import io

import common
import quality
import spectra
import targets as targets_mod
from common import JobSpec, RunRecord
from levels import VERIFY, shifted_variant, smoke_variant
from registry import jobs_for
from systems import BARRIERLESS, by_id, exclusion_reason

EXPERIMENT = "01_reference_states"

# An IRC branch that moves less than this has not left the transition state,
# so its endpoint label describes the saddle and says nothing about products.
IRC_MIN_DISPLACEMENT = 0.05  # Angstrom, max component


def reference_path(reaction: str, suffix: str = ".xyz"):
    """Where a reference artifact goes, with the directory guaranteed to exist.

    Same reasoning as `common.traj_path`: these writes come after the whole
    expensive part of the job, so a missing parent directory throws away hours.
    `exist_ok` makes it safe for concurrent array tasks.
    """
    path = common.RESULTS / "reference" / f"{reaction}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def run_one(spec: JobSpec, rec: RunRecord, level, args) -> None:
    rxn = by_id()[spec.params["reaction"]]
    metrics: dict = {
        "reaction": rxn.id,
        "equation": rxn.equation,
        "category": rxn.category,
        "spin_changes": rxn.spin_changes,
        "charge": rxn.charge,
        "spin": rxn.spin,
        "natoms": rxn.natoms,
    }
    # Bound to the record now, not at the end: every field added below survives
    # an exception, because `run_job` writes whatever `rec` holds from its
    # `finally`.  Assigning only on the last line is how the first run threw
    # away four relaxed endpoint pairs and four refined saddles -- 19 to 127
    # minutes each -- and left `metrics: {}` behind, which is also why a
    # `grep verified` over the records did not list those reactions at all.
    rec.metrics = metrics

    if rxn.id in BARRIERLESS:
        # Nothing below would find a saddle, and the search runs off down the
        # dissociation coordinate until the SCF fails.  Recorded as a result so
        # the denominator stays visible, at the cost of a second rather than an
        # hour.
        reason = exclusion_reason(rxn.id)
        metrics["verified"] = False
        metrics["excluded"] = True
        metrics["exclusion_reason"] = reason
        print(f"       skipped: {reason}", flush=True)
        return

    # Where L0 and L1 diverge.  Reported here once, for every reaction, so the
    # ladder comparison in E03 can point at a cause rather than a mystery.
    disagreements = targets_mod.integer_disagreements(rxn)
    metrics["integer_disagreements"] = disagreements
    metrics["n_integer_disagreements"] = len(disagreements)

    # Smoke caps are tight but not so tight that nothing converges: a smoke
    # run that leaves `verified` false has produced no reference, and every
    # downstream experiment then skips, exercising only its skip path.
    steps = 150 if args.smoke else 300
    irc_steps = 30 if args.smoke else 80

    # --- endpoints ------------------------------------------------------
    # Relaxed to the *separated* asymptote, not to whatever contact complex the
    # dataset supplies.  Two fragments in contact are not the state their label
    # claims: rxn_11's supplied product is two OH radicals with every atom
    # 0.97-0.98 Ang from a neighbour, and relaxing it as given recombines to
    # H2O + O -- which is rxn_04's product, to the last digit of its energy.
    # Every barrier below is therefore an asymptotic barrier, the convention
    # they are quoted against anyway.  See `spectra.separate_fragments`.
    relaxed = {}
    for side, frame in (("r", rxn.reactant), ("p", rxn.product)):
        atoms, converged, n, separated = spectra.relax_asymptote(
            frame,
            rxn.charge,
            rxn.spin,
            level,
            fmax=0.01,
            steps=steps,
        )
        point = spectra.single_point(atoms, rxn.charge, rxn.spin, level)
        relaxed[side] = atoms
        metrics[f"{side}_relaxed"] = converged
        metrics[f"{side}_relax_steps"] = n
        metrics[f"{side}_separated"] = separated
        metrics[f"{side}_energy_ev"] = point["energy"]
        metrics[f"{side}_max_force"] = point["max_force"]
        metrics[f"{side}_species"] = spectra.species_label(atoms)

    # --- the dataset TS, before anything is done to it ------------------
    # This is the baseline for the whole experiment: it says how far the
    # supplied geometry is from a stationary point at our level, and therefore
    # how much of any later RMSD is reference drift rather than method error.
    dataset_point = spectra.single_point(rxn.ts, rxn.charge, rxn.spin, level)
    metrics["dataset_ts_max_force"] = dataset_point["max_force"]
    metrics["dataset_ts_energy_ev"] = dataset_point["energy"]
    try:
        dataset_spectrum = spectra.hessian_spectrum(rxn.ts, rxn.charge, rxn.spin, level)
        metrics["dataset_ts_n_imaginary"] = dataset_spectrum.n_imaginary
        metrics["dataset_ts_omega"] = dataset_spectrum.omega_imaginary
    except RuntimeError as exc:
        metrics["dataset_ts_n_imaginary"] = None
        metrics["dataset_ts_hessian_error"] = str(exc)

    # --- refine to a genuine saddle -------------------------------------
    refinement = spectra.refine_saddle(
        rxn.ts, rxn.charge, rxn.spin, level, fmax=0.02, steps=steps
    )
    if not refinement["converged"]:
        refinement = _retry_shifted(rxn, refinement, level, steps, metrics)
    ts_atoms = refinement["atoms"]
    metrics["ts_refine_converged"] = refinement["converged"]
    metrics["ts_refine_steps"] = refinement["steps"]
    metrics["ts_refine_gradient_calls"] = refinement["gradient_calls"]
    metrics["ts_shift_rmsd_heavy"] = quality.permutation_rmsd(
        rxn.ts, ts_atoms, heavy_only=True
    )
    metrics["ts_shift_rmsd_all"] = quality.permutation_rmsd(
        rxn.ts, ts_atoms, heavy_only=False
    )

    ts_point = spectra.single_point(ts_atoms, rxn.charge, rxn.spin, level)
    metrics["ts_energy_ev"] = ts_point["energy"]
    metrics["ts_max_force"] = ts_point["max_force"]

    metrics["barrier_kcal"] = quality.barrier_kcal(
        ts_point["energy"], metrics["r_energy_ev"]
    )
    metrics["reverse_barrier_kcal"] = quality.barrier_kcal(
        ts_point["energy"], metrics["p_energy_ev"]
    )
    metrics["reaction_energy_kcal"] = quality.barrier_kcal(
        metrics["p_energy_ev"], metrics["r_energy_ev"]
    )

    # Written now and again at the end.  The Hessian below is the single most
    # failure-prone call in this script, and losing the refined saddle to it
    # means paying for the whole refinement again just to see where it went.
    # The duplicate write is seconds against hours.
    write_reference(rxn, relaxed, ts_atoms, verified=False, mode=None)

    # --- harmonic analysis ----------------------------------------------
    # A Hessian that cannot be computed makes the reaction unverifiable, which
    # is a *result*: it belongs in the record next to everything above it, not
    # in a traceback that discards them.  The dataset-TS Hessian above is
    # already treated this way.
    try:
        ts_spectrum = spectra.hessian_spectrum(ts_atoms, rxn.charge, rxn.spin, level)
    except RuntimeError as exc:
        ts_spectrum = None
        metrics["ts_hessian_error"] = str(exc)

    # A second imaginary frequency is usually the optimiser stopping early on a
    # floppy mode rather than a genuine second-order saddle: rxn_16 exited in 5
    # Sella steps having moved 0.045 Ang, and its extra mode is 84i cm^-1 --
    # a hindered rotation of a loosely bound complex, barely past
    # IMAGINARY_CUTOFF_CM.  Tightening once and re-diagonalising distinguishes
    # the two, and is kept only if it actually produces a first-order saddle,
    # so nothing that already verified can change.
    if ts_spectrum is not None and ts_spectrum.n_imaginary > 1:
        metrics["ts_n_imaginary_loose"] = ts_spectrum.n_imaginary
        metrics["ts_wavenumbers_loose"] = ts_spectrum.wavenumbers.tolist()
        tighter = spectra.refine_saddle(
            ts_atoms, rxn.charge, rxn.spin, level, fmax=0.005, steps=steps
        )
        try:
            retried = spectra.hessian_spectrum(
                tighter["atoms"], rxn.charge, rxn.spin, level
            )
        except RuntimeError as exc:
            retried = None
            metrics["ts_retry_hessian_error"] = str(exc)
        metrics["ts_retry_converged"] = tighter["converged"]
        metrics["ts_retry_n_imaginary"] = retried.n_imaginary if retried else None
        if retried is not None and retried.n_imaginary == 1:
            ts_atoms = tighter["atoms"]
            ts_spectrum = retried
            # The retry starts from the first refinement's output, so the cost
            # of reaching this saddle is both legs -- E02 and E04 compare
            # against `ts_refine_gradient_calls` as a basin-of-attraction
            # measurement and would otherwise be handed a free head start.
            refinement = {
                **tighter,
                "steps": refinement["steps"] + tighter["steps"],
                "gradient_calls": (
                    refinement["gradient_calls"] + tighter["gradient_calls"]
                ),
            }
            metrics["ts_refine_converged"] = refinement["converged"]
            metrics["ts_refine_steps"] = refinement["steps"]
            metrics["ts_refine_gradient_calls"] = refinement["gradient_calls"]
            ts_point = spectra.single_point(ts_atoms, rxn.charge, rxn.spin, level)
            metrics["ts_energy_ev"] = ts_point["energy"]
            metrics["ts_max_force"] = ts_point["max_force"]
            metrics["ts_shift_rmsd_heavy"] = quality.permutation_rmsd(
                rxn.ts, ts_atoms, heavy_only=True
            )
            metrics["ts_shift_rmsd_all"] = quality.permutation_rmsd(
                rxn.ts, ts_atoms, heavy_only=False
            )
            metrics["barrier_kcal"] = quality.barrier_kcal(
                ts_point["energy"], metrics["r_energy_ev"]
            )
            metrics["reverse_barrier_kcal"] = quality.barrier_kcal(
                ts_point["energy"], metrics["p_energy_ev"]
            )

    metrics["ts_n_imaginary"] = ts_spectrum.n_imaginary if ts_spectrum else None
    metrics["ts_omega_imaginary"] = ts_spectrum.omega_imaginary if ts_spectrum else None
    metrics["ts_wavenumbers"] = (
        ts_spectrum.wavenumbers.tolist() if ts_spectrum else None
    )

    # --- does it connect the reaction we think it does? -----------------
    verified = False
    if refinement["converged"] and ts_spectrum and ts_spectrum.n_imaginary == 1:
        irc = spectra.run_irc(ts_atoms, rxn.charge, rxn.spin, level, steps=irc_steps)
        connection = spectra.irc_connects(
            irc, relaxed["r"], relaxed["p"], rxn.charge, rxn.spin, level
        )
        metrics["irc_expected"] = connection["expected"]
        metrics["irc_observed"] = connection["observed"]
        metrics["irc_connects"] = connection["connects"]
        metrics["irc_forward_converged"] = irc["forward"]["converged"]
        metrics["irc_reverse_converged"] = irc["reverse"]["converged"]
        metrics["irc_forward_displacement"] = irc["forward"]["displacement"]
        metrics["irc_reverse_displacement"] = irc["reverse"]["displacement"]
        metrics["irc_left_saddle"] = all(
            irc[d]["displacement"] > IRC_MIN_DISPLACEMENT
            for d in ("forward", "reverse")
        )
        # Whether each terminus actually reached a minimum before it was
        # labelled.  Without this, an endpoint label that disagrees with the
        # intended product cannot be told apart from one whose relaxation ran
        # out of steps -- which is exactly what rxn_11 turned out to be.
        for direction, endpoint in connection["endpoints"].items():
            metrics[f"irc_{direction}_label"] = endpoint["label"]
            metrics[f"irc_{direction}_relax_converged"] = endpoint["relax_converged"]
            metrics[f"irc_{direction}_fragment_separation"] = endpoint["separation"]
            # What the terminus would have been called had it been relaxed in
            # contact.  Equal to `label` unless separating changed the answer,
            # which is the whole reason rxn_11 and rxn_13 failed the first run.
            metrics[f"irc_{direction}_separated"] = endpoint["separated"]
            metrics[f"irc_{direction}_label_contact"] = endpoint["label_contact"]
        verified = bool(connection["connects"])

        path = (
            list(reversed(irc["reverse"]["frames"]))
            + [ts_atoms]
            + irc["forward"]["frames"]
        )
        io.write(
            reference_path(rxn.id, "-irc.xyz"),
            path,
            format="extxyz",
        )
    else:
        metrics["irc_connects"] = False

    metrics["verified"] = verified
    if not verified:
        # Named explicitly so downstream scripts exclude rather than score it,
        # and so the exclusion appears in the paper with a reason attached.
        metrics["exclusion_reason"] = _exclusion_reason(
            refinement, ts_spectrum, metrics
        )

    write_reference(
        rxn,
        relaxed,
        ts_atoms,
        verified=verified,
        mode=ts_spectrum.imaginary_mode if ts_spectrum else None,
    )

    print(
        f"       verified={verified}  n_imag={metrics['ts_n_imaginary']}  "
        f"barrier={metrics['barrier_kcal']:.1f} kcal/mol  "
        f"ts_shift={metrics['ts_shift_rmsd_heavy']:.3f} Ang",
        flush=True,
    )


def _retry_shifted(rxn, refinement: dict, level, steps: int, metrics: dict) -> dict:
    """Resume a failed saddle refinement under a level shift.

    rxn_18 is why this exists.  Its dataset TS has a clean imaginary mode
    (-2020i), so there is a saddle there, but refinement spent all 300 steps
    drifting 0.30 Ang without converging and the Hessian then failed outright
    with "SCF did not converge".  That is an SCF problem on a stretched
    open-shell geometry, not a missing stationary point, and a level shift is
    what `data/1-run.py` already uses to get through exactly those.

    A shift moves nothing at convergence (see `levels.shifted_variant`), so the
    resumed geometry is legitimate -- but only the geometry.  Everything
    recorded downstream is re-evaluated at the caller's unshifted level, and the
    record's `level` field stays VERIFY, which is also what `inputs_hash`
    covers: the shift is scaffolding and must not look like a second method.

    Both legs are charged to `steps` and `gradient_calls`.  E02 and E04 read
    `ts_refine_gradient_calls` as a basin-of-attraction measurement, and a
    reference that reached its saddle in two goes must not be reported as
    having done it in one.
    """
    shifted = spectra.refine_saddle(
        refinement["atoms"],
        rxn.charge,
        rxn.spin,
        shifted_variant(level),
        fmax=0.02,
        steps=steps,
    )
    # Named apart from the `ts_retry_*` block below: that one re-refines a
    # second-order saddle, this one re-refines a non-converged SCF, and a run
    # can need both.
    metrics["ts_shift_retry_level_shift"] = list(shifted_variant(level).level_shift)
    metrics["ts_shift_retry_converged"] = shifted["converged"]
    metrics["ts_shift_retry_steps"] = shifted["steps"]
    if not shifted["converged"]:
        # Nothing gained, and the unshifted geometry is the one every other
        # number in this record was computed from.  Keep it.
        return refinement
    return {
        **shifted,
        "steps": refinement["steps"] + shifted["steps"],
        "gradient_calls": refinement["gradient_calls"] + shifted["gradient_calls"],
    }


def _exclusion_reason(refinement: dict, ts_spectrum, metrics: dict) -> str:
    """Why this reaction has no trustworthy reference, in one phrase."""
    if not refinement["converged"]:
        return "sella did not converge"
    if ts_spectrum is None:
        return "SCF did not converge at the refined saddle"
    if ts_spectrum.n_imaginary != 1:
        return f"n_imaginary={ts_spectrum.n_imaginary}"
    if not metrics.get("irc_left_saddle", True):
        return "IRC did not leave the saddle"
    return "IRC did not connect the intended endpoints"


def write_reference(rxn, relaxed: dict, ts_atoms, *, verified: bool, mode) -> None:
    """Persist the three reference frames and the imaginary mode."""
    frames = [relaxed["r"], ts_atoms, relaxed["p"]]
    for name, frame in zip(("r", "ts", "p"), frames):
        frame.info["id"] = f"{rxn.id}-{name}"
        frame.info["verified"] = verified
    io.write(reference_path(rxn.id), frames, format="extxyz")
    np.save(
        reference_path(rxn.id, "-mode.npy"),
        mode if mode is not None else np.zeros((rxn.natoms, 3)),
    )


def load_reference(reaction: str) -> dict | None:
    """The E01 output that every downstream TS experiment consumes.

    Returns None when the reaction has no verified reference, which callers
    must treat as an exclusion rather than as a zero.

    `--smoke` is the one exception: at a minimal basis several of these
    reactions have no product well left to fall into, so nothing verifies and
    every downstream script would exercise only its skip path.  A smoke run
    therefore accepts an unverified reference -- the geometries are still real
    output of the same code.  A smoke record cannot be mistaken for a result:
    its level is named `*-smoke` and `inputs_hash` covers the level, so a
    production run recomputes it rather than resuming from it.
    """
    rec = common.read_record(EXPERIMENT, reaction)
    if rec is None or rec.get("status") != "ok":
        return None
    metrics = rec["metrics"]
    if not metrics.get("verified") and not common.SMOKE:
        return None
    frames = io.read(reference_path(reaction), index=":")
    assert isinstance(frames, list)
    mode_path = reference_path(reaction, "-mode.npy")
    return {
        "r_atoms": frames[0],
        "ts_atoms": frames[1],
        "p_atoms": frames[2],
        "r_energy": metrics["r_energy_ev"],
        "ts_energy": metrics["ts_energy_ev"],
        "p_energy": metrics["p_energy_ev"],
        "barrier_kcal": metrics["barrier_kcal"],
        "ts_imaginary_mode": np.load(mode_path) if mode_path.exists() else None,
        "ts_omega": metrics["ts_omega_imaginary"],
    }


def excluded_reactions() -> dict[str, str]:
    """Reactions with no trustworthy reference, and why.

    Every figure caption that reports a success rate must state this set.

    `BARRIERLESS` wins over whatever the record says, because those reactions
    have no saddle to find at all.  Records written before that set existed
    report `scf_fail`, which E10 would otherwise class as an infrastructure
    failure -- the opposite of the truth, since the SCF failed precisely
    *because* there was no saddle to walk to.  rxn_09's older records say
    `n_imaginary=0` for the same reason and are overridden the same way.

    `systems.exclusion_reason` distinguishes the four that were pre-registered
    from rxn_09, which was excluded after E01 measured it.
    """
    # Listed whether or not a record exists: the exclusion is a property of the
    # reaction, not of whether anyone got round to running it.
    out = {rid: exclusion_reason(rid) for rid in sorted(BARRIERLESS)}
    for rec in common.load_records(EXPERIMENT):
        metrics = rec.get("metrics", {})
        if rec["key"] in BARRIERLESS:
            continue
        if rec.get("status") != "ok":
            out[rec["key"]] = f"reference run failed ({rec.get('status')})"
        elif not metrics.get("verified"):
            out[rec["key"]] = metrics.get("exclusion_reason", "not verified")
    return out


def main() -> None:
    parser = common.add_standard_args(argparse.ArgumentParser(description=__doc__))
    args = common.parse_args(parser)
    level = smoke_variant(VERIFY) if args.smoke else VERIFY
    jobs = jobs_for(EXPERIMENT)
    if args.smoke:
        jobs = common.pick_smoke_jobs(jobs, args)
    common.main_loop(
        EXPERIMENT,
        jobs,
        args,
        level,
        lambda spec, rec: run_one(spec, rec, level, args),
    )


if __name__ == "__main__":
    main()
