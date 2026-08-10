"""E10 -- the failure atlas.  When it fails, why?

Not a new computation: an analysis over every RunRecord already on disk.  It
is an experiment rather than a footnote because two of its classes decide what
a success rate even means.

Class 8 (reference-side failure) is the difference between "the method failed"
and "we could not tell" -- a reaction whose reference TS could not be verified
must not count against the method.  Class 6 (spin-state error) makes the
fixed-spin limitation quantitative rather than a caveat in prose.  Class 9
(barrierless) tells you the denominator: a reaction with no saddle at this
level of theory cannot have one found.

A paper that publishes this table is much harder to attack than one that
publishes a success rate.

    uv run python experiments/10_failure_atlas.py
    uv run python experiments/10_failure_atlas.py --spin-check
"""

from __future__ import annotations

import argparse
import importlib
import json


import common
import spectra
from levels import VERIFY
from systems import by_id

e01 = importlib.import_module("01_reference_states")

EXPERIMENT = "10_failure_atlas"

CLASSES = {
    1: "SCF non-convergence",
    2: "optimizer step limit",
    3: "flat-gradient stall (|F| < fmax while max|dB| > 0.2)",
    4: "early-stop artifact (thresh triggered with error elsewhere)",
    5: "converged to the wrong structure (minimum, higher saddle, other reaction)",
    6: "spin-state error (another multiplicity is lower at this geometry)",
    7: "spectator freezing under restrict_gradient",
    8: "reference-side failure -- NOT a method failure",
    9: "barrierless: no saddle exists at this level",
}

ANALYSED = (
    "02_ts_single_ended",
    "03_information_ladder",
    "04_ts_baselines",
    "05_target_sharpness",
    "06_path_vs_irc",
    "08_ablations",
)


def classify(rec: dict, excluded: dict[str, str]) -> list[int]:
    """Every class a record belongs to.  Records can belong to more than one."""
    metrics = rec.get("metrics", {})
    reaction = metrics.get("reaction")
    classes: list[int] = []

    reference_failed = bool(reaction in excluded or metrics.get("excluded"))
    if reference_failed:
        classes.append(8)

    status = rec.get("status")
    if status == "scf_fail":
        classes.append(1)
    elif status == "exception":
        text = (rec.get("traceback") or "").lower()
        classes.append(1 if "scf" in text or "converg" in text else 5)

    outcome = metrics.get("drive_outcome") or metrics.get("path_outcome")
    if outcome == "step_limit":
        classes.append(2)
    if outcome == "flat_gradient_stall":
        classes.append(3)
    if outcome == "early_stop" and metrics.get("bond_max_all_pairs", 0) > 0.5:
        classes.append(4)

    # Classes 1-4 above are properties of the *run* and stay diagnosable
    # whatever the reference did: a drive that stalled on fmax stalled.  Class
    # 5 is the one that needs the reference, since "wrong structure" is only
    # meaningful relative to a trusted right one -- so a reference-side failure
    # suppresses class 5 and nothing else.  Collapsing everything into class 8
    # the moment E01 fails would hide precisely the failure modes E10 exists to
    # count.
    if not reference_failed:
        n_imag = metrics.get("n_imaginary")
        if n_imag is not None and n_imag != 1:
            classes.append(5)
        elif n_imag == 1 and metrics.get("irc_connects") is False:
            classes.append(5)

    if metrics.get("knob") == "restrict_gradient" and metrics.get("value"):
        if (metrics.get("restricted_vs_free_rmsd") or 0) > 0.05:
            classes.append(7)

    return classes


def spin_check(args) -> dict:
    """Is another multiplicity lower at the located geometry?

    The fixed-spin limitation is documented in prose everywhere in this repo.
    This is the measurement: re-evaluate each located structure at spin, spin-2
    and spin+2 and report where the assumed multiplicity is not the lowest.
    Expensive enough to be opt-in, cheap enough to be worth running once.
    """
    out = {}
    for rec in common.load_records("02_ts_single_ended"):
        metrics = rec.get("metrics", {})
        reaction = metrics.get("reaction")
        if rec.get("status") != "ok" or not reaction:
            continue
        rxn = by_id()[reaction]
        traj = common.traj_path("02_ts_single_ended", rec["key"])
        if not traj.exists():
            continue
        from ase import io

        frames = io.read(traj, index=":")
        located = frames[-1] if isinstance(frames, list) else frames

        energies = {}
        for spin in {max(0, rxn.spin - 2), rxn.spin, rxn.spin + 2}:
            if (spin - rxn.spin) % 2 != 0:
                continue
            try:
                energies[spin] = spectra.single_point(
                    located, rxn.charge, spin, VERIFY
                )["energy"]
            except Exception:  # noqa: BLE001
                energies[spin] = None
        valid = {k: v for k, v in energies.items() if v is not None}
        if not valid:
            continue
        lowest = min(valid, key=lambda k: valid[k])
        out[reaction] = {
            "assumed_spin": rxn.spin,
            "lowest_spin": lowest,
            "energies_ev": valid,
            "spin_error": lowest != rxn.spin,
            "gap_kcal": (
                (valid[rxn.spin] - valid[lowest]) * 23.0605
                if rxn.spin in valid else None
            ),
        }
    return out


def barrierless(excluded: dict[str, str]) -> list[str]:
    """Reactions E01 could not give a saddle to at all."""
    return sorted(
        rid for rid, reason in excluded.items()
        if "n_imaginary=0" in reason or "did not converge" in reason
    )


def main() -> None:
    parser = common.add_standard_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--spin-check", action="store_true",
                        help="re-evaluate located structures at spin +/- 2 "
                             "(class 6); expensive, so opt-in")
    args = common.parse_args(parser)
    common.setup(args)

    excluded = e01.excluded_reactions()

    counts = {k: 0 for k in CLASSES}
    rows = []
    for experiment in ANALYSED:
        for rec in common.load_records(experiment):
            classes = classify(rec, excluded)
            if not classes:
                continue
            for c in classes:
                counts[c] += 1
            rows.append({
                "experiment": experiment,
                "key": rec["key"],
                "reaction": rec.get("metrics", {}).get("reaction"),
                "status": rec.get("status"),
                "classes": classes,
                "class_names": [CLASSES[c] for c in classes],
            })

    atlas = {
        "class_definitions": CLASSES,
        "counts": counts,
        "rows": rows,
        "excluded_reactions": excluded,
        "n_excluded": len(excluded),
        "barrierless_candidates": barrierless(excluded),
        "experiments_analysed": list(ANALYSED),
    }

    if args.spin_check:
        checks = spin_check(args)
        atlas["spin_check"] = checks
        atlas["counts"][6] = sum(1 for v in checks.values() if v["spin_error"])

    path = common.RESULTS / "failure_atlas.json"
    path.write_text(json.dumps(common.canonical(atlas), indent=2))

    print(f"\nfailure atlas -> {path}")
    print(f"  {len(rows)} records in a failure class; "
          f"{len(excluded)} reactions excluded on the reference side\n")
    for c, name in CLASSES.items():
        print(f"  class {c}  {counts[c]:5d}  {name}")
    if excluded:
        print("\n  excluded reactions (class 8 -- NOT method failures):")
        for rid, reason in sorted(excluded.items()):
            print(f"    {rid}: {reason}")


if __name__ == "__main__":
    main()
