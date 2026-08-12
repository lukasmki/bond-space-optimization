"""Mayer bond orders from a DFT calculation.

`PySCFCalculator` is an ordinary energy+forces calculator, but every time it
runs it also leaves the Mayer bond-order matrix on the Atoms object.  That
matrix is the coordinate system the rest of this package works in, so this is
the place to start.

Run: uv run python examples/1-bond-order.py
"""

import numpy as np
from ase import Atoms, build

from bondspace.ase import PySCFCalculator

# spin is the number of unpaired electrons and has to match the electron
# count parity of each system -- O2 is a ground-state triplet.
SYSTEMS = [
    ("H2", 0, "single bond"),
    ("N2", 0, "triple bond"),
    ("O2", 2, "double bond, triplet"),
]


def report(atoms: Atoms, label: str) -> np.ndarray:
    energy = atoms.get_potential_energy()
    B = atoms.get_array("bond-order")
    sym = atoms.get_chemical_symbols()

    print(f"\n--- {label}   E = {energy:.4f} eV")
    print("bond-order matrix (the diagonal is an atomic term, not a bond):")
    print("        " + "".join(f"{s}{i:<7}" for i, s in enumerate(sym)))
    for i, s in enumerate(sym):
        print(f"  {s}{i:<5}" + "".join(f"{B[i, j]:8.3f}" for j in range(len(atoms))))

    # Mayer valence: how much bonding capacity each atom is using.
    valence = B.sum(axis=1) - np.diag(B)
    print(
        "Mayer valences: "
        + "  ".join(f"{s}{i}={v:.3f}" for i, (s, v) in enumerate(zip(sym, valence)))
    )
    return B


if __name__ == "__main__":
    print("Bond order recovers chemical bond multiplicity:\n")
    print("  system   expected            Mayer bond order")
    results = []
    for name, spin, expected in SYSTEMS:
        atoms: Atoms = build.molecule(name)
        atoms.calc = PySCFCalculator(charge=0, spin=spin, basis="cc-pvdz")
        atoms.get_potential_energy()
        b = atoms.get_array("bond-order")[0, 1]
        results.append((name, expected, b))
        print(f"  {name:<8} {expected:<20}{b:6.3f}")

    # A polyatomic: the matrix carries every pair at once, and the calculator
    # also stores the thresholded pair list it derived from it.
    water: Atoms = build.molecule("H2O")
    water.calc = PySCFCalculator(charge=0, spin=0, basis="cc-pvdz")
    report(water, "H2O")
    print("connectivity (i, j, bond order):")
    sym = water.get_chemical_symbols()
    for i, j, b in water.info["connectivity"]:
        if b > 0.1:  # skip the ~0.01 residual overlap of non-bonded pairs
            print(f"  {sym[i]}{i}-{sym[j]}{j}  {b:.3f}")

    print("\nBond order is continuous, not integral, so it can be differentiated")
    print("and optimized -- that is what the rest of these examples do.")
