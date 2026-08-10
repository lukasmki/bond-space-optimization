"""Run a bond-space drive and then score it outside bond space.

E02, E03, E05 and E08 are the same pipeline with different targets and knobs,
so it lives here once.  Four experiments agreeing on what "success" means is
the point: three independent scoring implementations would be three ways to
disagree about the same run.

The pipeline is deliberately in two halves:

  1. `drive` -- PRODUCTION level, exactly data/1-run.py's settings.
  2. `verify` -- VERIFY level, tight convergence, no level shift: the Hessian,
     the saddle refinement and the IRC.

Nothing in half 2 is allowed to influence half 1.  A bond-space run must not
be able to consult the potential energy surface it claims not to use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from ase import Atoms
from ase.optimize import FIRE2

from bondspace.ase import BondFluxCalculator

import quality
import spectra
import targets as targets_mod
from levels import Level
from systems import Reaction

Target = tuple[int, int, float]

#: Raised from data/1-run.py's 80.  The step budget must not be the thing that
#: limits the headline result; E08 sweeps it to show whether it binds.
DEFAULT_STEPS = 200

#: FIRE2's own default is 0.2 Ang, which overshoots into SCF-diverging
#: geometries because the constraint objective's scale is unrelated to a PES.
DEFAULT_MAXSTEP = 0.05


@dataclass
class DriveResult:
    atoms: Atoms
    frames: list[Atoms]
    converged: bool
    steps: int
    calculate_calls: int
    outcome: str  # converged | step_limit | flat_gradient_stall | early_stop
    max_target_error: float


def drive(
    start: Atoms,
    bonds: Sequence[Target],
    charge: int,
    spin: int,
    level: Level,
    *,
    steps: int = DEFAULT_STEPS,
    maxstep: float = DEFAULT_MAXSTEP,
    fmax: float = 0.1,
    thresh: float = 0.05,
    ovlp_thresh: float = 0.5,
    zvector: bool = True,
    restrict_gradient: bool = False,
    optimizer=FIRE2,
    fresh_guess: bool = False,
    keep_frames: bool = True,
) -> DriveResult:
    """Drive a geometry toward a target bond-order matrix.

    Takes ``charge``/``spin`` rather than a `Reaction` so that E07 can drive
    structures the benchmark has never heard of -- network discovery invents
    its own species, and requiring a Reaction there would mean faking one.

    ``outcome`` distinguishes cases the existing analysis collapses into
    "converged".  In particular **flat_gradient_stall**: the optimiser can
    reach ``fmax`` while the bond orders are still far from target, because a
    bond that must form from far away produces almost no force.  Counting
    those as successes is the single most misleading thing this pipeline could
    do, so they get their own label.
    """
    atoms = start.copy()
    calc = BondFluxCalculator(
        list(bonds),
        charge=charge,
        spin=spin,
        thresh=thresh,
        ovlp_thresh=ovlp_thresh,
        zvector=zvector,
        restrict_gradient=restrict_gradient,
        # E08 asks whether the located structure depends on trajectory
        # history: by default BondFluxCalculator carries mo_coeff between
        # geometries, so the SCF guess is path-dependent.
        reuse_guess=not fresh_guess,
        **level.calc_kwargs(),
    )
    calls = {"n": 0}
    original = calc.calculate

    def counted(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    calc.calculate = counted  # type: ignore[method-assign]
    atoms.calc = calc

    frames: list[Atoms] = []
    opt = optimizer(atoms, logfile=None, maxstep=maxstep)
    if keep_frames:
        opt.attach(lambda: frames.append(atoms.copy()), interval=1)
    converged = bool(opt.run(fmax=fmax, steps=steps))
    n_steps = opt.get_number_of_steps()

    error = quality.target_error(atoms, bonds)
    if not converged:
        outcome = "step_limit"
    elif error < thresh:
        outcome = "early_stop"
    elif error > 0.2:
        # Converged on force while still far from the requested bond orders:
        # the documented flat-gradient-at-long-range failure, not a success.
        outcome = "flat_gradient_stall"
    else:
        outcome = "converged"

    return DriveResult(
        atoms=atoms,
        frames=frames,
        converged=converged,
        steps=n_steps,
        calculate_calls=calls["n"],
        outcome=outcome,
        max_target_error=error,
    )


@dataclass
class Verdict:
    """Everything the verification half produces about one located structure."""

    metrics: dict
    tiers: quality.Tiers
    refined: Atoms | None


def verify_ts(
    located: Atoms,
    rxn: Reaction,
    level: Level,
    reference: dict,
    *,
    bonds: Sequence[Target] | None = None,
    rmsd_threshold: float = 0.25,
    bond_threshold: float = 0.5,
    refine_steps: int = 100,
    irc_steps: int = 60,
    full: bool = True,
) -> Verdict:
    """Score a transition-state guess against a verified reference saddle.

    ``reference`` is one entry of E01's output: the refined TS geometry, its
    energy, its spectrum and the relaxed endpoints.  Every tier above T1 is
    meaningless without it, which is why E01 must run first and why reactions
    whose reference could not be verified are excluded rather than scored.

    ``full=False`` skips the saddle refinement and IRC, for the ablation
    sweeps where hundreds of runs make T4 unaffordable and T0-T3 suffice.
    """
    ref_ts: Atoms = reference["ts_atoms"]
    metrics: dict = {}

    metrics.update(
        {f"bond_{k}": v for k, v in quality.bond_errors(located, ref_ts).items()}
    )
    metrics["rmsd_heavy"] = quality.permutation_rmsd(located, ref_ts, heavy_only=True)
    metrics["rmsd_all"] = quality.permutation_rmsd(located, ref_ts, heavy_only=False)

    point = spectra.single_point(located, rxn.charge, rxn.spin, level)
    metrics["energy_ev"] = point["energy"]
    metrics["pes_residual_ev_ang"] = point["max_force"]
    metrics["energy_gap_kcal"] = quality.energy_gap_kcal(
        point["energy"], reference["ts_energy"]
    )
    metrics["barrier_kcal"] = quality.barrier_kcal(
        point["energy"], reference["r_energy"]
    )
    metrics["barrier_error_kcal"] = (
        metrics["barrier_kcal"] - reference["barrier_kcal"]
    )

    spectrum = spectra.hessian_spectrum(located, rxn.charge, rxn.spin, level)
    metrics["n_imaginary"] = spectrum.n_imaginary
    metrics["omega_imaginary"] = spectrum.omega_imaginary
    metrics["wavenumbers"] = spectrum.wavenumbers.tolist()

    # Is the mode you got the mode you asked for?  Neither the energy nor the
    # RMSD can answer this: two saddles of the same reaction can sit close in
    # both while describing different motions.
    mode = spectrum.imaginary_mode
    if mode is not None:
        if bonds:
            direction = targets_mod.bond_direction(located, bonds)
            metrics["mode_overlap_bonds"] = quality.mode_overlap(mode, direction)
        ref_mode = reference.get("ts_imaginary_mode")
        if ref_mode is not None:
            metrics["mode_overlap_reference"] = quality.mode_overlap(
                mode, np.asarray(ref_mode)
            )

    t0 = metrics["bond_max_ref_pairs"] < bond_threshold
    t1 = metrics["bond_max_all_pairs"] < bond_threshold
    t2 = metrics["rmsd_heavy"] < rmsd_threshold
    t3 = spectrum.n_imaginary == 1
    t4 = False
    refined = None

    if full:
        refinement = spectra.refine_saddle(
            located, rxn.charge, rxn.spin, level, steps=refine_steps
        )
        refined = refinement["atoms"]
        metrics["refine_converged"] = refinement["converged"]
        metrics["refine_steps"] = refinement["steps"]
        metrics["refine_gradient_calls"] = refinement["gradient_calls"]
        metrics["refine_displacement_rmsd"] = quality.permutation_rmsd(
            located, refined, heavy_only=False
        )

        if refinement["converged"]:
            refined_spectrum = spectra.hessian_spectrum(
                refined, rxn.charge, rxn.spin, level
            )
            metrics["refined_n_imaginary"] = refined_spectrum.n_imaginary
            if refined_spectrum.n_imaginary == 1:
                irc = spectra.run_irc(
                    refined, rxn.charge, rxn.spin, level, steps=irc_steps
                )
                connection = spectra.irc_connects(
                    irc,
                    reference["r_atoms"],
                    reference["p_atoms"],
                    rxn.charge,
                    rxn.spin,
                    level,
                )
                metrics["irc_expected"] = connection["expected"]
                metrics["irc_observed"] = connection["observed"]
                metrics["irc_connects"] = connection["connects"]
                t4 = bool(connection["connects"])

    tiers = quality.Tiers(
        t0_ref_bond_error=bool(t0),
        t1_all_bond_error=bool(t1),
        t2_rmsd=bool(t2),
        t3_one_imaginary=bool(t3),
        t4_verified=bool(t4),
    )
    metrics["tiers"] = tiers.as_dict()
    metrics["highest_tier"] = tiers.highest
    metrics["tier_label"] = tiers.label
    return Verdict(metrics=metrics, tiers=tiers, refined=refined)
