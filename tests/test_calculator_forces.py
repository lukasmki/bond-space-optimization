"""Finite-difference checks for the ASE calculators.

`test_bond_gradients.py` validates the gradients in isolation.  This file
checks what the calculators actually hand to an optimizer, which is a
different question: the constraint force is assembled from the bond-order
gradient *plus* the overlap correction, and an error in that assembly is
invisible to the lower-level tests.

That is not hypothetical.  The overlap correction once applied its force to
the two atoms of a pair with opposite signs, which -- since dS/dR is already
antisymmetric between them -- produced a net translation of the pair instead
of separating it, with a net force roughly forty times the real constraint
force.  `test_forces_sum_to_zero` is the direct check for that class of bug
and does not depend on finite-difference precision at all.

Both calculators report forces in eV/Angstrom, as ASE requires, so a
numerical derivative of the energy with respect to Angstrom positions should
match the reported force directly.  This is worth testing rather than assuming:
the gradients underneath are all per Bohr, since they are built from PySCF
integrals, and a missing conversion is a uniform factor of 0.529 that leaves
optimizations still converging -- just to a differently scaled `fmax`.
"""

import numpy as np
import pytest
from ase import Atoms

from bondspace.ase import BondFluxCalculator, PySCFCalculator

# Angstrom.  Larger than the Bohr step used for the bare gradients: the
# calculators run DFT on a grid, and a bigger step keeps grid noise from
# dominating the difference quotient.
STEP = 1e-3

BASIS = "sto-3g"
BONDS = [(0, 1, 0.5)]


def h3() -> Atoms:
    """H2 + H as a doublet: open shell, and small enough to differentiate."""
    return Atoms("HHH", positions=[[-0.74, 0, 0], [0.0, 0, 0], [1.30, 0, 0]])


def bond_flux(atoms: Atoms, **kwargs) -> Atoms:
    atoms = atoms.copy()
    atoms.calc = BondFluxCalculator(
        BONDS,
        charge=0,
        spin=1,
        basis=BASIS,
        # The default 0.05 early-returns zero energy and force once the error
        # is small, which would silently flatten the numerical derivative.
        thresh=1e-9,
        level_shift=(0.3, 0.2),
        **kwargs,
    )
    return atoms


def numerical_force(make: "callable", atoms: Atoms, step: float = STEP) -> np.ndarray:
    """-dE/dR by central differences, in energy units per Angstrom."""
    out = np.zeros((len(atoms), 3))
    for k in range(len(atoms)):
        for x in range(3):
            plus, minus = atoms.copy(), atoms.copy()
            plus.positions[k, x] += step
            minus.positions[k, x] -= step
            e_plus = make(plus).get_potential_energy()
            e_minus = make(minus).get_potential_energy()
            out[k, x] = -(e_plus - e_minus) / (2 * step)
    return out


@pytest.mark.parametrize(
    "ovlp_thresh, label",
    [(0.0, "correction off"), (1.0, "correction on")],
    ids=["no-overlap-correction", "with-overlap-correction"],
)
def test_bond_flux_forces_match_finite_difference(ovlp_thresh, label):
    """Analytic constraint force against a numerical derivative of its energy.

    With `ovlp_thresh=1.0` the 0.5 target falls below the threshold, so the
    overlap repulsion is active and its contribution is exercised too.
    """
    atoms = h3()

    def make(a: Atoms) -> Atoms:
        return bond_flux(a, ovlp_thresh=ovlp_thresh)

    analytic = make(atoms).get_forces()
    numeric = numerical_force(make, atoms)

    assert np.abs(analytic - numeric).max() < 2e-3, (
        f"{label}: analytic {analytic[:, 0]} vs numeric {numeric[:, 0]}"
    )


@pytest.mark.parametrize("ovlp_thresh", [0.0, 1.0])
def test_forces_sum_to_zero(ovlp_thresh):
    """No net force on an isolated molecule, whatever the objective.

    The objective depends only on internal coordinates, so every term has to
    conserve momentum.  The tolerance is relative because the calculator runs
    DFT on a grid with density fitting and `conv_tol=1e-6`, which leaves a
    residual around 1e-4; the bare gradients reach 1e-15 under tight UHF (see
    test_bond_gradients.py).  A wrong-signed force term is not a small
    residual -- the overlap-correction bug produced a net force twice the
    size of the largest real force -- so this still discriminates by orders
    of magnitude.
    """
    forces = bond_flux(h3(), ovlp_thresh=ovlp_thresh).get_forces()
    scale = np.abs(forces).max()
    assert np.abs(forces.sum(axis=0)).max() < 1e-2 * scale


def test_overlap_correction_conserves_momentum():
    """The overlap term alone must carry no net force.

    Sharper than the test above: taking the difference against the run
    without the correction cancels the SCF noise, and what remains is built
    from analytic integrals only, so it can be held to a tight tolerance.
    This is the precise signature of applying dS/dR with the wrong relative
    sign between the two atoms of a pair.
    """
    atoms = h3()
    without = bond_flux(atoms, ovlp_thresh=0.0).get_forces()
    with_ = bond_flux(atoms, ovlp_thresh=1.0).get_forces()

    extra = with_ - without
    assert np.abs(extra).max() > 1e-3, "correction did not activate"
    assert np.abs(extra.sum(axis=0)).max() < 1e-9


def test_overlap_correction_separates_the_pair():
    """The correction must push the constrained pair apart, not shove it along.

    H0 and H1 are the constrained pair and lie on the x axis, so the extra
    force from the correction must be equal and opposite along x.
    """
    atoms = h3()
    without = bond_flux(atoms, ovlp_thresh=0.0).get_forces()
    with_ = bond_flux(atoms, ovlp_thresh=1.0).get_forces()
    extra = with_ - without

    assert extra[0, 0] * extra[1, 0] < 0, "pair is not being pushed apart"
    assert np.isclose(extra[0, 0], -extra[1, 0], rtol=1e-6, atol=1e-8)


def test_zvector_matches_direct_forces():
    """The adjoint path must agree with the per-atom path through the calculator."""
    atoms = h3()
    direct = bond_flux(atoms, ovlp_thresh=1.0).get_forces()
    adjoint = bond_flux(atoms, ovlp_thresh=1.0, zvector=True).get_forces()
    assert np.abs(direct - adjoint).max() < 1e-5


def test_restrict_gradient_zeroes_unconstrained_atoms():
    """Restricting the solve freezes atoms outside `bonds` -- by construction.

    Atom 2 appears in no constrained bond, so its force is dropped entirely.
    This is the documented approximation, not a bug, but it is worth pinning
    down: `BondFluxCalculator` adds no PES force, so a zeroed atom cannot move
    at all.
    """
    atoms = h3()
    full = bond_flux(atoms, ovlp_thresh=0.0).get_forces()
    restricted = bond_flux(atoms, ovlp_thresh=0.0, restrict_gradient=True).get_forces()

    assert np.abs(restricted[2]).max() == 0.0
    assert np.abs(full[2]).max() > 1e-4, "spectator force should be non-trivial"
    assert np.abs(restricted[[0, 1]] - full[[0, 1]]).max() < 1e-5


def test_pyscf_calculator_forces_match_finite_difference():
    """The physical calculator, which does convert to eV/Angstrom."""
    atoms = h3()

    def make(a: Atoms) -> Atoms:
        a = a.copy()
        a.calc = PySCFCalculator(charge=0, spin=1, basis=BASIS, level_shift=(0.3, 0.2))
        return a

    analytic = make(atoms).get_forces()
    numeric = numerical_force(make, atoms)
    assert np.abs(analytic - numeric).max() < 1e-2


def test_bond_order_array_is_published():
    """Both calculators leave the bond-order matrix and connectivity behind."""
    atoms = bond_flux(h3(), ovlp_thresh=0.0)
    atoms.get_potential_energy()

    B = atoms.get_array("bond-order")
    assert B.shape == (3, 3)
    assert np.allclose(B, B.T, atol=1e-10)
    assert "connectivity" in atoms.info
