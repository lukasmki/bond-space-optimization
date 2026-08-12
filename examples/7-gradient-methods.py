"""Three ways to get the constraint force, and what each costs.

The bond-order gradient needs a CPHF solve, which is 75-85% of the work, and
the direct route needs one solve *per atom*.  There are two ways to cut that,
and they are not equivalent:

  direct                 one CPHF solve per atom.  The reference.
  zvector=True           one solve total.  The objective is a single scalar,
                         so the adjoint (Z-vector) method contracts first and
                         solves once.  Mathematically the same answer.
  restrict_gradient=True limits the solve to atoms named in `bonds`.  Rows
                         that are skipped are set to exactly zero, so those
                         atoms feel no force at all -- an approximation, and
                         since this calculator supplies no PES force, they are
                         frozen for the whole optimization.

Run: uv run python examples/7-gradient-methods.py
"""

import time

import numpy as np
from ase import Atoms

from bondspace.ase import BondFluxCalculator

# H2O2: constrain only the O-O bond, so the two hydrogens are "spectators"
# and restrict_gradient has something to skip.
ATOMS = Atoms(
    "OOHH",
    positions=[
        [0.0, 0.0, 0.00],
        [0.0, 0.0, 1.45],
        [0.9, 0.0, -0.30],
        [-0.9, 0.0, 1.75],
    ],
)
BONDS = [(0, 1, 2.0)]  # O-O targeted well away from its actual value
SPECTATORS = [2, 3]


def run(label: str, **kwargs) -> tuple[np.ndarray, float]:
    atoms = ATOMS.copy()
    atoms.calc = BondFluxCalculator(
        BONDS,
        charge=0,
        spin=0,
        basis="cc-pvdz",
        thresh=0.05,
        ovlp_thresh=0.5,
        **kwargs,
    )
    t0 = time.perf_counter()
    forces = atoms.get_forces()
    dt = time.perf_counter() - t0
    print(f"  {label:<24} {dt:6.2f} s")
    return forces, dt


if __name__ == "__main__":
    print(f"H2O2, {len(ATOMS)} atoms, constraining only the O-O bond.\n")
    f_direct, t_direct = run("direct (per-atom CPHF)")
    f_zvec, t_zvec = run("zvector=True", zvector=True)
    f_restr, t_restr = run("restrict_gradient=True", restrict_gradient=True)

    print("\nzvector vs direct:")
    print(
        f"  max force difference : {np.abs(f_zvec - f_direct).max():.2e}"
        f"   (force scale {np.abs(f_direct).max():.3f})"
    )
    print(f"  speedup              : {t_direct / t_zvec:.2f}x")
    print("  Exact -- the residual is the krylov tolerance of the single solve,")
    print("  not an approximation, and forces are produced for every atom.")

    print("\nrestrict_gradient vs direct:")
    print(
        f"  constrained atoms    : {sorted({a for i, j, _ in BONDS for a in (i, j)})}"
    )
    print(
        f"  force on O atoms     : differs by "
        f"{np.abs(f_restr[[0, 1]] - f_direct[[0, 1]]).max():.2e}"
    )
    print(
        f"  force on H spectators: {np.abs(f_restr[SPECTATORS]).max():.2e} "
        f"(discarded {np.abs(f_direct[SPECTATORS]).max():.2e})"
    )
    print(f"  speedup              : {t_direct / t_restr:.2f}x")
    print("  The discarded spectator force is the same order as a typical fmax,")
    print("  so those atoms are not merely slow to move -- they cannot move.")

    print("""
Recommendation: use `zvector=True`.  It is exact, returns forces on every
atom, and its advantage grows with system size, which makes
`restrict_gradient` unnecessary in most cases.  Reach for
`restrict_gradient` only when freezing the unconstrained atoms is what you
actually want.""")
