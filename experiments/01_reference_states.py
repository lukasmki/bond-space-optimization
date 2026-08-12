"""E01 -- level-consistent reference states.  Run this first.

The HCombustion geometries come from an external dataset at an unknown level
of theory, and are almost certainly not stationary points under
wB97X-D3/cc-pVDZ/DF.  Until they are re-optimised here, every RMSD and every
barrier in the suite would be measuring the level-of-theory shift as much as
the method, and there would be no way to tell the two apart.

For each reaction this relaxes R and P, refines the dataset TS to a genuine
first-order saddle with Sella, and runs an IRC to confirm the saddle actually
connects the intended endpoints.  A reaction whose TS cannot be verified is
**excluded from every downstream TS metric** -- and, crucially, is recorded as
a reference-side failure so E10 can tell "the method failed" apart from "we
could not tell".

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
from levels import VERIFY, smoke_variant
from registry import jobs_for
from systems import by_id

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
    relaxed = {}
    for side, frame in (("r", rxn.reactant), ("p", rxn.product)):
        atoms, converged, n = spectra.relax(
            frame,
            rxn.charge,
            rxn.spin,
            level,
            fmax=0.01,
            steps=steps,
            internal=True,
        )
        point = spectra.single_point(atoms, rxn.charge, rxn.spin, level)
        relaxed[side] = atoms
        metrics[f"{side}_relaxed"] = converged
        metrics[f"{side}_relax_steps"] = n
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

    ts_spectrum = spectra.hessian_spectrum(ts_atoms, rxn.charge, rxn.spin, level)
    metrics["ts_n_imaginary"] = ts_spectrum.n_imaginary
    metrics["ts_omega_imaginary"] = ts_spectrum.omega_imaginary
    metrics["ts_wavenumbers"] = ts_spectrum.wavenumbers.tolist()

    metrics["barrier_kcal"] = quality.barrier_kcal(
        ts_point["energy"], metrics["r_energy_ev"]
    )
    metrics["reverse_barrier_kcal"] = quality.barrier_kcal(
        ts_point["energy"], metrics["p_energy_ev"]
    )
    metrics["reaction_energy_kcal"] = quality.barrier_kcal(
        metrics["p_energy_ev"], metrics["r_energy_ev"]
    )

    # --- does it connect the reaction we think it does? -----------------
    verified = False
    if refinement["converged"] and ts_spectrum.n_imaginary == 1:
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
        metrics["exclusion_reason"] = (
            "sella did not converge"
            if not refinement["converged"]
            else (
                f"n_imaginary={ts_spectrum.n_imaginary}"
                if ts_spectrum.n_imaginary != 1
                else (
                    "IRC did not leave the saddle"
                    if not metrics.get("irc_left_saddle", True)
                    else "IRC did not connect the intended endpoints"
                )
            )
        )

    # --- persist the reference geometries -------------------------------
    path = reference_path(rxn.id)
    frames = [relaxed["r"], ts_atoms, relaxed["p"]]
    for name, frame in zip(("r", "ts", "p"), frames):
        frame.info["id"] = f"{rxn.id}-{name}"
        frame.info["verified"] = verified
    io.write(path, frames, format="extxyz")
    np.save(
        reference_path(rxn.id, "-mode.npy"),
        (
            ts_spectrum.imaginary_mode
            if ts_spectrum.imaginary_mode is not None
            else np.zeros((rxn.natoms, 3))
        ),
    )

    rec.metrics = metrics
    print(
        f"       verified={verified}  n_imag={metrics['ts_n_imaginary']}  "
        f"barrier={metrics['barrier_kcal']:.1f} kcal/mol  "
        f"ts_shift={metrics['ts_shift_rmsd_heavy']:.3f} Ang",
        flush=True,
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
    """
    out = {}
    for rec in common.load_records(EXPERIMENT):
        metrics = rec.get("metrics", {})
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
