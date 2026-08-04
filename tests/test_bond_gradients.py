"""Finite-difference checks for every analytic gradient in `bondspace.bond`.

These run bottom-up, so a failure points at a layer rather than just at the
final answer: overlap derivative, then atom-blocked overlap, then the density
response, then the bond order itself, then the adjoint contraction.

Comparing the analytic paths only against each other is not enough.  Two bugs
in this file's history -- an asymmetric dS/dR and a double-counted
occupied-occupied density response -- cancelled each other well enough that
the analytic routes agreed while both were wrong.  Only finite differences
separated them, which is why every layer is checked independently here.
"""

import numpy as np
import pytest

from bondspace.bond import (
    atom_overlap,
    bo,
    bo_flux_gradient,
    bo_gradient,
    dm_gradient,
    ov_gradient,
)

from conftest import central_difference


# --------------------------------------------------------------------------
# overlap derivative -- pure integrals, no density response
# --------------------------------------------------------------------------


def test_ov_gradient_matches_finite_difference(system, scf_at, mf):
    _, _, coords, _ = system
    analytic = ov_gradient(mf)
    numeric = central_difference(lambda c: scf_at(c).get_ovlp(), coords)
    assert np.abs(analytic - numeric).max() < 1e-7


def test_ov_gradient_is_symmetric(mf):
    """dS/dR must be symmetric in (mu, nu) because S is.

    Reading the transposed slice of `int1e_ipovlp` for the ket term breaks
    exactly this, and it is the cheapest signature of that bug.
    """
    analytic = ov_gradient(mf)
    assert np.abs(analytic - analytic.swapaxes(-1, -2)).max() < 1e-12


def test_ov_gradient_translationally_invariant(mf):
    """Rigidly translating the molecule cannot change any overlap."""
    assert np.abs(ov_gradient(mf).sum(axis=0)).max() < 1e-10


# --------------------------------------------------------------------------
# atom-blocked overlap -- feeds the BondFluxCalculator overlap correction
# --------------------------------------------------------------------------


def test_atom_overlap_gradient_matches_finite_difference(system, scf_at, mf):
    _, _, coords, _ = system
    _, analytic = atom_overlap(mf)
    numeric = central_difference(lambda c: atom_overlap(scf_at(c))[0], coords)
    assert np.abs(analytic - numeric).max() < 1e-7


def test_atom_overlap_gradient_translationally_invariant(mf):
    assert np.abs(atom_overlap(mf)[1].sum(axis=0)).max() < 1e-10


# --------------------------------------------------------------------------
# density response -- the CPHF layer
# --------------------------------------------------------------------------


def test_dm_gradient_matches_finite_difference(system, scf_at, mf):
    """dP/dR for both spin channels.

    The occupied-occupied rotations are already inside PySCF's `mo1`
    (`solve_withs1` fills that block with -S1/2); adding a separate
    occupied-occupied term on top double counts and shows up here.
    """
    _, _, coords, _ = system
    dma, dmb = dm_gradient(mf)
    numeric = central_difference(lambda c: scf_at(c).make_rdm1(), coords)

    # make_rdm1 returns (2, nao, nao); split the spin axis back out.
    assert np.abs(dma - numeric[:, :, 0]).max() < 1e-4
    assert np.abs(dmb - numeric[:, :, 1]).max() < 1e-4


# --------------------------------------------------------------------------
# bond order -- the quantity the whole package optimizes
# --------------------------------------------------------------------------


def test_bo_gradient_matches_finite_difference(system, scf_at, mf):
    _, _, coords, _ = system
    analytic = bo_gradient(mf)
    numeric = central_difference(lambda c: bo(scf_at(c)), coords)
    assert np.abs(analytic - numeric).max() < 1e-5


def test_bo_gradient_translationally_invariant(mf):
    """Sum over atoms must vanish; this is what exposed the original bug."""
    assert np.abs(bo_gradient(mf).sum(axis=0)).max() < 1e-8


def test_bo_gradient_symmetric_in_pair_indices(mf):
    """B is symmetric, so its gradient must be symmetric in (i, j) too."""
    BG = bo_gradient(mf)
    assert np.abs(BG - BG.swapaxes(-1, -2)).max() < 1e-12


@pytest.mark.parametrize("atomlist", [[0], [0, 2]])
def test_atomlist_rows_match_unrestricted(mf, atomlist):
    """Requested rows are computed exactly; the rest are left at zero.

    The residual on the requested rows is CPHF block-convergence noise --
    PySCF solves all perturbations against one criterion, so asking for fewer
    lands slightly differently inside that tolerance.
    """
    full = bo_gradient(mf)
    restricted = bo_gradient(mf, atomlist=atomlist)

    assert np.abs(restricted[atomlist] - full[atomlist]).max() < 1e-5
    skipped = [k for k in range(mf.mol.natm) if k not in atomlist]
    assert np.all(restricted[skipped] == 0.0)


# --------------------------------------------------------------------------
# adjoint (Z-vector) gradient of a scalar bond-order objective
# --------------------------------------------------------------------------


def objective_weights(B: np.ndarray, target: np.ndarray, mask: np.ndarray):
    """G = dE/dB for E = 1/2 sum mask*(B - target)^2."""
    return mask * (B - target)


def bond_flux_energy(B: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    return 0.5 * float(np.sum(np.square(mask * (B - target))))


@pytest.fixture
def objective(mf):
    """A constraint on the 0-1 pair, away from its current value."""
    natm = mf.mol.natm
    target = np.full((natm, natm), -1.0)
    target[0, 1] = target[1, 0] = 0.5
    return target, target >= 0


def test_bo_flux_gradient_matches_finite_difference(system, scf_at, mf, objective):
    """The adjoint path against finite differences of the scalar objective."""
    _, _, coords, _ = system
    target, mask = objective

    B = bo(mf)
    analytic = bo_flux_gradient(mf, objective_weights(B, target, mask))
    numeric = central_difference(
        lambda c: bond_flux_energy(bo(scf_at(c)), target, mask), coords
    )
    assert np.abs(analytic - numeric).max() < 1e-5


def test_bo_flux_gradient_matches_direct_contraction(mf, objective):
    """One adjoint solve must reproduce contracting the per-atom gradients.

    Agreement is at the krylov tolerance of the single solve, not exact
    arithmetic equality.
    """
    target, mask = objective
    G = objective_weights(bo(mf), target, mask)

    adjoint = bo_flux_gradient(mf, G)
    direct = np.einsum("ab,kxab->kx", G, bo_gradient(mf))
    assert np.abs(adjoint - direct).max() < 1e-5


def test_bo_flux_gradient_translationally_invariant(mf, objective):
    target, mask = objective
    G = objective_weights(bo(mf), target, mask)
    assert np.abs(bo_flux_gradient(mf, G).sum(axis=0)).max() < 1e-7
