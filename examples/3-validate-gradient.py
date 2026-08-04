"""Validate the analytic bond-order gradient against finite differences.

There is no test suite, so this script is the correctness gate for `bond.py`.
It caught two real bugs that had been cancelling each other out: an asymmetric
dS/dR, and an occupied-occupied density-response term that was counted twice.
Fixing either one alone made the answer worse, so neither showed up when the
analytic paths were only compared against each other -- only finite differences
exposed them.

Two conventions matter here:

  * `bo_gradient` differentiates with respect to nuclear positions in **Bohr**
    (it is built from PySCF's `int1e_ipovlp`), while ASE works in Angstrom.
    The molecule is therefore built with `unit="Bohr"` and displaced in Bohr.
  * UHF/STO-3G is used rather than DFT: no exchange-correlation grid means no
    grid noise in the finite differences, and it keeps the 6N+1 SCF runs quick.

Run: uv run python examples/3-validate-gradient.py
"""

import numpy as np
from pyscf import gto, scf

from bondspace.bond import bo, bo_gradient

# H2O, coordinates in Bohr
GEOM = [
    ("O", (0.0, 0.0, 0.0)),
    ("H", (0.0, 1.43, 1.11)),
    ("H", (0.0, -1.43, 1.11)),
]
STEP = 1e-4  # Bohr


def scf_at(coords: np.ndarray) -> scf.hf.SCF:
    mol = gto.M(
        atom=[(s, tuple(c)) for (s, _), c in zip(GEOM, coords)],
        basis="sto-3g",
        spin=0,
        unit="Bohr",
        verbose=0,
    )
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-14
    mf.kernel()
    return mf


if __name__ == "__main__":
    c0 = np.array([c for _, c in GEOM])
    BG = bo_gradient(scf_at(c0))
    sym = [s for s, _ in GEOM]
    natm = len(GEOM)

    print("d(bond order O0-H1) / dR, analytic vs central differences\n")
    print("  atom  dir      analytic    finite-diff        error")
    worst = 0.0
    for k in range(natm):
        for x in range(3):
            cp, cm = c0.copy(), c0.copy()
            cp[k, x] += STEP
            cm[k, x] -= STEP
            fd = (bo(scf_at(cp))[0, 1] - bo(scf_at(cm))[0, 1]) / (2 * STEP)
            an = BG[k, x, 0, 1]
            if abs(fd) < 1e-9 and abs(an) < 1e-9:
                continue  # symmetry-zero component, nothing to compare
            worst = max(worst, abs(an - fd))
            print(f"  {sym[k]}{k}    {'xyz'[x]}    {an:12.6f} {fd:12.6f} {an - fd:12.2e}")

    # Rigidly translating the molecule cannot change any bond order, so every
    # column of the gradient must sum to zero over atoms.  This is the check
    # that originally revealed the bug.
    drift = np.abs(BG.sum(axis=0)).max()

    print(f"\nmax |analytic - finite difference| : {worst:.2e}")
    print(f"translational invariance residual  : {drift:.2e}")

    ok = worst < 1e-5 and drift < 1e-8
    print("\nPASS - analytic gradient is correct" if ok else
          "\nFAIL - bond.py has regressed")
    raise SystemExit(0 if ok else 1)
