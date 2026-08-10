"""E06 -- is the R->P drive a chemically meaningful path?

One thing must be said plainly before any number: **a bond-space drive is an
optimisation trajectory, not a reaction path.**  It is FIRE2-momentum
dependent, unevenly spaced in any coordinate, and not mass-weighted.  It
cannot be an intrinsic reaction coordinate and the paper must not imply that
it is.  What can honestly be asked is whether it stays in the neighbourhood of
one -- whether it passes through chemically reasonable structures on its way,
or takes a shortcut over a ridge that no molecule would climb.

So the metrics are:

  * **tube distance** -- per frame, the distance to the nearest reference IRC
    frame.  The *maximum* is the honest headline; a path can have a fine mean
    while excursioning somewhere absurd.
  * **monotonicity** -- does it advance along the reference coordinate, or
    backtrack?
  * **barrier recovery** -- the true PES energy along the drive, against the
    verified barrier.  A path that overshoots by tens of kcal/mol is not a
    reaction path however good its endpoints are.  This is the number most
    likely to hurt, so it is reported as a distribution.

The drive never uses the PES, so re-evaluating every frame's true energy is
free evidence about the path it chose rather than a change to the method.

Baselines: linear interpolation, IDPP, and the converged CI-NEB band, scored
identically.  IDPP is nearly free, so a bond-space path no better than IDPP on
tube distance is a null result and must be stated as one.

    sbatch --array=0-75 experiments/run.slurm 06_path_vs_irc.py
"""

from __future__ import annotations

import argparse
import importlib

import numpy as np
from ase import Atoms, io

import baselines
import common
import drive
import quality
import spectra
import targets as targets_mod
from common import JobSpec, RunRecord
from levels import PRODUCTION, VERIFY, smoke_variant
from registry import jobs_for
from systems import by_id

e01 = importlib.import_module("01_reference_states")

EXPERIMENT = "06_path_vs_irc"

#: Frames re-evaluated on the true PES.  Every frame would be wasteful on a
#: 200-step drive and adds nothing: the profile is smooth at this resolution.
MAX_PROFILE_FRAMES = 30


def _subsample(frames: list[Atoms], limit: int) -> list[Atoms]:
    if len(frames) <= limit:
        return frames
    idx = np.linspace(0, len(frames) - 1, limit).round().astype(int)
    return [frames[i] for i in idx]


def _reference_irc(reaction: str) -> list[Atoms]:
    """E01's verified IRC, which supersedes data/HCombustion-irc/.

    The shipped IRCs are at an unknown level of theory; comparing a cc-pVDZ
    drive against them would fold level-of-theory error into the path metric.
    They are still used as a cross-check of the IRC machinery itself -- see
    `legacy_irc` below.
    """
    path = common.RESULTS / "reference" / f"{reaction}-irc.xyz"
    if not path.exists():
        return []
    frames = io.read(path, index=":")
    return frames if isinstance(frames, list) else [frames]


def legacy_irc(reaction: str) -> list[Atoms]:
    """The dataset's own IRC, used only to calibrate the tube metric.

    If E01's computed IRC lies inside the tube of the shipped one, both the
    IRC machinery and the metric are behaving; the check validates them
    against each other rather than trusting either alone.
    """
    index = reaction.split("_")[1]
    path = common.REPO / "data" / "HCombustion-irc" / f"{index}_irc.xyz"
    if not path.exists():
        return []
    frames = io.read(path, index=":")
    return frames if isinstance(frames, list) else [frames]


def _build_path(method: str, rxn, reference, production, args) -> dict:
    steps = 30 if args.smoke else drive.DEFAULT_STEPS
    if method == "bondspace":
        bonds = targets_mod.l1_targets(rxn, "p")
        result = drive.drive(
            reference["r_atoms"], bonds, rxn.charge, rxn.spin, production,
            steps=steps
        )
        return {
            "frames": result.frames + [result.atoms],
            "pes_calls": result.calculate_calls,
            "outcome": result.outcome,
            "converged": result.converged,
            "targets": [list(b) for b in bonds],
        }
    if method in ("linear", "idpp"):
        from ase.mep import NEB

        product = rxn.product.copy()
        product.set_positions(baselines._aligned_product(rxn))
        n = 5 if args.smoke else 15
        images = [reference["r_atoms"].copy() for _ in range(n - 1)] + [product]
        neb = NEB(images)
        neb.interpolate(method="idpp" if method == "idpp" else "linear")
        return {"frames": images, "pes_calls": 0, "outcome": "interpolated",
                "converged": True}
    if method == "cineb":
        from ase.mep import NEB
        from ase.optimize import FIRE

        product = rxn.product.copy()
        product.set_positions(baselines._aligned_product(rxn))
        n = 5 if args.smoke else 9
        images = [reference["r_atoms"].copy() for _ in range(n - 1)] + [product]
        neb = NEB(images, climb=True, allow_shared_calculator=False)
        neb.interpolate(method="idpp")
        calls = {"n": 0}
        for image in images[1:-1]:
            image.calc = spectra.make_calculator(rxn.charge, rxn.spin, production)
            original = image.calc.calculate

            def counted(*a, _o=original, **k):
                calls["n"] += 1
                return _o(*a, **k)

            image.calc.calculate = counted  # type: ignore[method-assign]
        converged = bool(
            FIRE(neb, logfile=None).run(
                fmax=0.1, steps=10 if args.smoke else 50
            )
        )
        return {"frames": images, "pes_calls": calls["n"], "outcome": "neb",
                "converged": converged}
    raise ValueError(f"unknown path method {method!r}")


def run_one(spec: JobSpec, rec: RunRecord, production, verify, args) -> None:
    rxn = by_id()[spec.params["reaction"]]
    method = spec.params["method"]
    reference = e01.load_reference(rxn.id)
    if reference is None:
        rec.status = "skipped"
        rec.metrics = {"reaction": rxn.id, "method": method, "excluded": True,
                       "exclusion_reason": "no verified reference from E01"}
        print("       skipped: no verified reference", flush=True)
        return

    irc = _reference_irc(rxn.id)
    built = _build_path(method, rxn, reference, production, args)
    frames = built["frames"]

    metrics: dict = {
        "reaction": rxn.id,
        "method": method,
        "category": rxn.category,
        "spin_changes": rxn.spin_changes,
        "n_frames": len(frames),
        "pes_calls": built["pes_calls"],
        "path_outcome": built["outcome"],
        "path_converged": built["converged"],
        "reference_irc_frames": len(irc),
    }

    if irc:
        metrics.update(
            {f"tube_{k}": v for k, v in quality.tube_distance(frames, irc).items()}
        )
        metrics.update(
            {f"progress_{k}": v
             for k, v in quality.progress_monotonicity(frames, irc).items()}
        )
        # Mutual validation: E01's IRC against the shipped one.  Only run on
        # the bondspace job so it is computed once per reaction.
        if method == "bondspace":
            shipped = legacy_irc(rxn.id)
            if shipped and len(shipped[0]) == len(irc[0]):
                metrics["irc_vs_shipped"] = quality.tube_distance(irc, shipped)

    # True PES energies along the path.  The drive never used them, so this is
    # evidence about the path it chose rather than a modification of it.
    profile_frames = _subsample(frames, 10 if args.smoke else MAX_PROFILE_FRAMES)
    energies = []
    for frame in profile_frames:
        try:
            energies.append(
                spectra.single_point(frame, rxn.charge, rxn.spin, verify)["energy"]
            )
        except Exception:  # noqa: BLE001 -- a diverged frame is data
            energies.append(float("nan"))
    finite = [e for e in energies if np.isfinite(e)]
    metrics["profile_energies_ev"] = energies
    metrics["profile_scf_failures"] = len(energies) - len(finite)
    if finite:
        metrics.update(
            quality.barrier_recovery(
                finite, reference["r_energy"], reference["barrier_kcal"]
            )
        )

    final = frames[-1]
    metrics["final_species"] = spectra.species_label(final)
    metrics["expected_species"] = spectra.species_label(reference["p_atoms"])
    metrics["endpoint_correct"] = (
        metrics["final_species"] == metrics["expected_species"]
    )
    metrics["min_rmsd_to_ts"] = min(
        quality.permutation_rmsd(f, reference["ts_atoms"], heavy_only=True)
        for f in profile_frames
    )

    io.write(common.traj_path(EXPERIMENT, spec.key), frames, format="extxyz")
    rec.metrics = metrics
    print(
        f"       {method}: tube_max={metrics.get('tube_max', float('nan')):.3f} "
        f"overshoot={metrics.get('overshoot_kcal', float('nan')):.1f} kcal/mol "
        f"endpoint_ok={metrics['endpoint_correct']}",
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
