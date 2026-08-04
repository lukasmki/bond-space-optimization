"""Shared fixtures for the finite-difference gradient tests.

Everything here is built to make finite differences trustworthy, because the
analytic gradients are only as validated as the reference they are compared
against:

  * **UHF, not DFT** -- an exchange-correlation grid adds noise at roughly the
    level of the errors we are trying to detect.
  * **STO-3G and tight `conv_tol`** -- small enough that 6N SCF runs per test
    is cheap, converged tightly enough that the difference quotient is clean.
  * **Coordinates in Bohr** -- every gradient in `bondspace.bond` is a
    derivative with respect to nuclear position in atomic units (they come from
    PySCF's `int1e_ipovlp`), while ASE works in Angstrom.  Displacing in the
    wrong unit produces a consistent factor-of-0.529 error that looks like a
    subtle bug.

Both a closed-shell and an open-shell system are provided, and tests run
against both.  This is not redundant: for a closed shell the spin density
R = P_alpha - P_beta vanishes identically, so the entire RS branch of `bo` and
`bo_gradient` is multiplied by zero and any error in it is invisible.
"""

import numpy as np
import pytest
from pyscf import gto, scf

# Central-difference step, in Bohr.  1e-4 balances truncation error (~step^2)
# against the SCF convergence floor.
STEP = 1e-4

# (symbols, coordinates in Bohr, number of unpaired electrons)
SYSTEMS = {
    "h2o": (
        ["O", "H", "H"],
        [[0.0, 0.0, 0.0], [0.0, 1.43, 1.11], [0.0, -1.43, 1.11]],
        0,
    ),
    "h3": (  # open shell: exercises the spin-density terms
        ["H", "H", "H"],
        [[-1.40, 0.0, 0.0], [0.0, 0.0, 0.0], [2.40, 0.0, 0.0]],
        1,
    ),
}


def make_scf(symbols: list[str], coords: np.ndarray, spin: int) -> scf.uhf.UHF:
    """A tightly converged UHF reference at the given geometry (Bohr)."""
    mol = gto.M(
        atom=[(s, tuple(c)) for s, c in zip(symbols, coords)],
        basis="sto-3g",
        spin=spin,
        charge=0,
        unit="Bohr",
        verbose=0,
    )
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-14
    mf.kernel()
    return mf


def central_difference(func, coords: np.ndarray, step: float = STEP) -> np.ndarray:
    """d func / d coords by central differences.

    `func` maps coordinates to a scalar or array; the result has shape
    ``(natm, 3) + func_shape``, matching the layout every analytic gradient in
    this package uses.
    """
    coords = np.asarray(coords, dtype=float)
    probe = np.asarray(func(coords), dtype=float)
    out = np.empty(coords.shape + probe.shape)

    for k in range(coords.shape[0]):
        for x in range(3):
            plus, minus = coords.copy(), coords.copy()
            plus[k, x] += step
            minus[k, x] -= step
            out[k, x] = (
                np.asarray(func(plus), dtype=float)
                - np.asarray(func(minus), dtype=float)
            ) / (2 * step)
    return out


@pytest.fixture(params=sorted(SYSTEMS), ids=sorted(SYSTEMS))
def system(request):
    """Symbols, coordinates (Bohr) and spin for each test system."""
    symbols, coords, spin = SYSTEMS[request.param]
    return request.param, symbols, np.array(coords, dtype=float), spin


@pytest.fixture
def scf_at(system):
    """Factory building a converged UHF object at arbitrary coordinates."""
    _, symbols, _, spin = system

    def build(coords: np.ndarray) -> scf.uhf.UHF:
        return make_scf(symbols, coords, spin)

    return build


@pytest.fixture
def mf(system, scf_at):
    """The converged reference calculation at the undisplaced geometry."""
    _, _, coords, _ = system
    return scf_at(coords)
