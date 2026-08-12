"""Tests for the experiment harness.

In the spirit of `conftest.py`: check the thing that would silently corrupt a
result, not the thing that is easy to check.  Three of these earn their place
because getting them wrong would be invisible in the output --

  * permutation-aware RMSD: with equivalent hydrogens, index-matched RMSD
    reports multi-Angstrom errors for physically identical structures, which
    would inflate every geometric number in the paper;
  * the L1 targets: if they silently disagreed with the hand-written example
    the whole information-ladder argument would be about the wrong thing;
  * the harmonic analysis sign convention: an imaginary frequency reported as
    a positive number turns every saddle into a minimum.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "experiments"))

import common  # noqa: E402
import levels  # noqa: E402
import quality  # noqa: E402
import systems  # noqa: E402
import targets as targets_mod  # noqa: E402


# --------------------------------------------------------------------------
# systems registry
# --------------------------------------------------------------------------


def test_registry_loads_and_agrees_with_data_util():
    """The consistency assert in systems.py must actually be exercised."""
    reactions = systems.load_set()
    assert len(reactions) == 19
    assert {r.id for r in reactions} == {f"rxn_{i:02d}" for i in range(1, 20)}


def test_every_reaction_has_a_bond_change():
    """A reaction with no integer change would give L1 no targets at all."""
    for rxn in systems.load_set():
        assert rxn.changing_pairs(), rxn.id


def test_spin_subset_is_preregistered():
    declared = systems.SPIN_NONCONSERVING
    flagged = {r.id for r in systems.load_set() if r.spin_changes}
    assert declared == flagged


# --------------------------------------------------------------------------
# targets -- the information ladder
# --------------------------------------------------------------------------


def test_l1_targets_match_the_hand_written_example():
    """examples/5-reaction-path.py asks for [(0,1,0.5), (1,3,0.5)] on rxn_03.

    That file is the documented statement of what the method does.  If the L1
    constructor disagrees with it, the ladder is measuring something other
    than the claim.
    """
    rxn = systems.by_id()["rxn_03"]
    assert targets_mod.l1_targets(rxn, "ts") == [(0, 1, 0.5), (1, 3, 0.5)]


def test_l1_product_targets_match_the_example():
    rxn = systems.by_id()["rxn_03"]
    assert targets_mod.l1_targets(rxn, "p") == [(0, 1, 0.0), (1, 3, 1.0)]


def test_l1_tau_interpolates_between_the_integer_endpoints():
    rxn = systems.by_id()["rxn_03"]
    at_zero = targets_mod.l1_targets(rxn, "ts", half=0.0)
    at_one = targets_mod.l1_targets(rxn, "ts", half=1.0)
    assert at_zero == targets_mod.l1_targets(rxn, "r")
    assert at_one == targets_mod.l1_targets(rxn, "p")


def test_asymmetric_targets_split_breaking_and_forming():
    """tau on the breaking pair, 1 - tau on the forming one."""
    rxn = systems.by_id()["rxn_03"]
    bonds = dict(((i, j), v) for i, j, v in targets_mod.l1_asymmetric_targets(rxn, 0.3))
    assert bonds[(0, 1)] == pytest.approx(0.7)  # breaking 1 -> 0, 30% of the way
    assert bonds[(1, 3)] == pytest.approx(0.7)  # forming 0 -> 1, 70% of the way


def test_l0_and_l1_differ_where_the_integers_disagree():
    """rxn_01 is the documented case: an O-O "bond" at 2.05 Ang.

    L0 rounds the Mayer order to 1 and therefore believes there is a bond to
    break; L1 reads the chemistry and knows there is not.  If this test ever
    passes trivially, the disagreement machinery has stopped working.
    """
    rxn = systems.by_id()["rxn_01"]
    disagreements = targets_mod.integer_disagreements(rxn)
    assert any(d["pair"] == [0, 1] and d["side"] == "r" for d in disagreements)
    offending = next(d for d in disagreements if d["pair"] == [0, 1])
    assert offending["distance_ang"] > 2.0
    assert offending["rounded"] == 1 and offending["lewis"] == 0


def test_bond_direction_is_normalised_and_signed():
    rxn = systems.by_id()["rxn_03"]
    direction = targets_mod.bond_direction(
        rxn.reactant, targets_mod.l1_targets(rxn, "ts")
    )
    assert direction.shape == (len(rxn.reactant), 3)
    assert np.linalg.norm(direction) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# quality -- the metric that would silently corrupt everything
# --------------------------------------------------------------------------


def test_rmsd_is_zero_for_identical_structures():
    water = molecule("H2O")
    assert quality.permutation_rmsd(water, water) == pytest.approx(0.0, abs=1e-10)


def test_rmsd_is_invariant_under_rotation_and_translation():
    water = molecule("H2O")
    moved = water.copy()
    moved.rotate(37, "xyz", rotate_cell=False)
    moved.translate((1.3, -0.7, 2.2))
    assert quality.permutation_rmsd(water, moved) == pytest.approx(0.0, abs=1e-8)


def test_rmsd_is_zero_when_like_atoms_are_relabelled():
    """The case naive index-matched RMSD gets badly wrong.

    rxn_03 is H2 + OH: three hydrogens in chemically different positions, one
    of them transferring.  Relabelling two of them leaves the *structure*
    identical -- nothing moved, only the indices changed -- but index-matched
    RMSD reports a large error, because it insists atom 0 be compared against
    atom 0.  A bond-space drive is under no obligation to move the hydrogen
    the reference happened to label as transferring, so without permutation
    awareness every RMSD in the paper would inherit this error.

    (Water is the wrong example here: swapping its two hydrogens *is* a
    symmetry operation, so Kabsch rotates the difference away by itself.)
    """
    rxn = systems.by_id()["rxn_03"]
    original = rxn.reactant
    swapped = original.copy()
    positions = swapped.get_positions()
    positions[[0, 2]] = positions[[2, 0]]  # two inequivalent hydrogens
    swapped.set_positions(positions)

    naive = quality.kabsch_rmsd(original.get_positions(), swapped.get_positions())
    assert naive > 0.5, "the naive metric should be badly wrong here"
    assert quality.permutation_rmsd(original, swapped) == pytest.approx(0.0, abs=1e-8)


def test_heavy_only_falls_back_when_there_are_no_heavy_atoms():
    h2 = Atoms("H2", positions=[(0, 0, 0), (0, 0, 0.74)])
    other = Atoms("H2", positions=[(0, 0, 0), (0, 0, 1.20)])
    value = quality.permutation_rmsd(h2, other, heavy_only=True)
    assert value > 0.0 and np.isfinite(value)


def test_heavy_only_falls_back_with_a_single_heavy_atom():
    """One heavy atom carries no shape: Kabsch superimposes it exactly.

    Four of the nineteen reactions (OH + H2 and friends) have exactly one
    oxygen.  Masking to heavy atoms there would make T2 report 0.000 A for
    every structure ever compared, so it would pass unconditionally.
    """
    a = Atoms("OH3", positions=[(0, 0, 0), (0, 0, 1.0), (0, 1.0, 0), (1.0, 0, 0)])
    b = a.copy()
    positions = b.get_positions()
    positions[1] += (0.0, 0.0, 0.8)
    b.set_positions(positions)
    assert quality.permutation_rmsd(a, b, heavy_only=True) > 0.1


def test_bond_errors_see_a_spurious_bond_that_the_old_metric_cannot():
    """T1 exists because T0 iterates the reference connectivity only."""
    a, b = molecule("H2O"), molecule("H2O")
    B = np.zeros((3, 3))
    B[0, 1] = B[1, 0] = 1.0
    invented = B.copy()
    invented[1, 2] = invented[2, 1] = 0.9  # a bond the reference does not have
    a.set_array("bond-order", invented)
    b.set_array("bond-order", B)

    errors = quality.bond_errors(a, b)
    assert errors["max_ref_pairs"] == pytest.approx(0.0)
    assert errors["max_all_pairs"] == pytest.approx(0.9)


def test_wilson_interval_brackets_the_point_estimate():
    low, high = quality.wilson_interval(12, 19)
    assert 0.0 <= low < 12 / 19 < high <= 1.0


def test_tiers_report_the_highest_contiguous_level():
    tiers = quality.Tiers(True, True, False, True, False)
    assert tiers.highest == 1  # T2 fails, so T3 passing does not count


# --------------------------------------------------------------------------
# harness -- records and hashing
# --------------------------------------------------------------------------


def test_inputs_hash_is_stable_under_key_reordering():
    a = common.hash_inputs({"x": 1, "y": 2}, levels.PRODUCTION.as_dict())
    b = common.hash_inputs({"y": 2, "x": 1}, levels.PRODUCTION.as_dict())
    assert a == b


def test_inputs_hash_changes_with_the_level():
    spec = {"reaction": "rxn_03"}
    assert common.hash_inputs(spec, levels.PRODUCTION.as_dict()) != common.hash_inputs(
        spec, levels.VERIFY.as_dict()
    )


def test_run_record_round_trips_through_json():
    import json

    rec = common.RunRecord(
        experiment="test",
        key="k",
        inputs_hash="abc",
        metrics={"a": np.float64(1.5), "b": [np.int64(2)]},
    )
    restored = json.loads(json.dumps(rec.as_dict()))
    assert restored["metrics"]["a"] == 1.5
    assert restored["metrics"]["b"] == [2]


def test_job_specs_are_deterministic_and_ordered():
    """The SLURM array index is the only link between a task and its work."""
    from registry import REGISTRY, jobs_for

    for name in REGISTRY:
        first = [j.key for j in jobs_for(name)]
        second = [j.key for j in jobs_for(name)]
        assert first == second, name
        assert len(set(first)) == len(first), f"{name} has duplicate keys"


def test_smoke_truncation_defers_to_an_explicit_index():
    """`--smoke --index N` must run job N, not nothing.

    Truncating the list to one job and *then* slicing by index selects the
    empty set for every index above zero, which silently makes it impossible
    to smoke-test a chosen reaction.
    """
    import argparse

    jobs = [common.JobSpec("e", f"k{i}", {}) for i in range(5)]
    free = argparse.Namespace(index=None, chunk=1)
    pinned = argparse.Namespace(index=2, chunk=1)

    assert [s.key for s in common.pick_smoke_jobs(jobs, free)] == ["k0"]
    kept = common.pick_smoke_jobs(jobs, pinned)
    assert [s.key for s in common.select_jobs(kept, pinned)] == ["k2"]


def test_job_rng_is_reproducible_and_spec_dependent():
    a = common.JobSpec("e", "k", {}).rng().normal(size=3)
    b = common.JobSpec("e", "k", {}).rng().normal(size=3)
    c = common.JobSpec("e", "other", {}).rng().normal(size=3)
    assert np.allclose(a, b)
    assert not np.allclose(a, c)


def test_levels_carry_distinct_convergence_tolerances():
    """The whole point of the PRODUCTION/VERIFY split."""
    assert levels.VERIFY.conv_tol < levels.PRODUCTION.conv_tol
    assert levels.VERIFY.level_shift == 0.0
    assert levels.PRODUCTION.level_shift == (0.3, 0.2)


# --------------------------------------------------------------------------
# spectra -- the sign convention that turns saddles into minima
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_relaxed_water_has_no_imaginary_frequencies():
    import spectra

    water = molecule("H2O")
    relaxed, _, _ = spectra.relax(water, 0, 0, levels.CHEAP, fmax=0.02, steps=60)
    spectrum = spectra.hessian_spectrum(relaxed, 0, 0, levels.CHEAP)
    assert spectrum.n_imaginary == 0
    assert spectrum.imaginary_mode is None
    # Three vibrational modes for a triatomic, all real and positive.
    assert (spectrum.wavenumbers > 0).sum() == 3


@pytest.mark.slow
def test_a_transition_state_has_exactly_one_imaginary_mode():
    """Sign convention check: PySCF returns imaginary frequencies as complex.

    If they were flattened to positive reals, every saddle in the suite would
    be classified as a minimum and T3 would pass for nothing.
    """
    import spectra

    rxn = systems.by_id()["rxn_03"]
    spectrum = spectra.hessian_spectrum(rxn.ts, rxn.charge, rxn.spin, levels.CHEAP)
    assert spectrum.n_imaginary == 1
    assert spectrum.omega_imaginary < 0
    assert spectrum.imaginary_mode.shape == (len(rxn.ts), 3)
