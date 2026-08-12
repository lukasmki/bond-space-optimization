"""Bond-order gradients: d(bond order ij) / d(position of atom k).

This is the quantity that makes bond space an optimizable coordinate.  It is
opt-in (`bo_grad=True`) because it needs a CPHF solve per perturbed atom, which
dominates the runtime.

Two things worth knowing are demonstrated here:

  * the storage convention -- ASE requires per-atom arrays, so the (natm, 3,
    natm, natm) tensor is flattened to (natm, 3*natm*natm) and must be reshaped
    after reading;
  * `bo_grad_atoms`, which restricts the CPHF to selected atoms.

Run: uv run python examples/2-bond-order-gradient.py
"""

import time
from pathlib import Path

import numpy as np
from ase import Atoms, build, io
from ase.calculators.singlepoint import SinglePointCalculator

from bondspace.ase import PySCFCalculator

OUT = Path(__file__).parent / "out"


def bond_order_grad(atoms: Atoms) -> np.ndarray:
    """Read back the flattened array as (natm, 3, natm, natm)."""
    n = len(atoms)
    return np.reshape(atoms.get_array("bond-order-grad"), (n, 3, n, n))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)

    atoms: Atoms = build.molecule("H2O")
    atoms.calc = PySCFCalculator(charge=0, spin=0, basis="cc-pvdz", bo_grad=True)

    t0 = time.perf_counter()
    atoms.get_potential_energy()
    t_full = time.perf_counter() - t0

    BG = bond_order_grad(atoms)
    sym = atoms.get_chemical_symbols()
    print(f"gradient tensor shape {BG.shape}   ({t_full:.1f} s, all atoms)")

    # BG[k, :, i, j] is the direction to move atom k to change bond ij.
    print("\nd(bond order)/dR, as a force-like vector on each atom:")
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        print(f"  bond {sym[i]}{i}-{sym[j]}{j}:")
        for k in range(len(atoms)):
            v = BG[k, :, i, j]
            print(f"     move {sym[k]}{k}: [{v[0]:8.4f} {v[1]:8.4f} {v[2]:8.4f}]")

    # Translational invariance: shifting the whole molecule changes nothing,
    # so the gradient must sum to zero over atoms.  Here it only reaches ~1e-3,
    # limited by the DFT grid, density fitting and conv_tol=1e-6 -- not by the
    # gradient itself.  3-validate-gradient.py repeats this with tight UHF and
    # gets ~1e-15.
    print(
        f"\nsum over atoms (0 by translational invariance): "
        f"{np.abs(BG.sum(axis=0)).max():.2e}"
    )

    # Write each pair's gradient as a set of pseudo-forces, viewable in any
    # trajectory viewer that draws force vectors.
    images = []
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            mol: Atoms = atoms.copy()
            mol.calc = SinglePointCalculator(mol, forces=BG[:, :, i, j])
            mol.info["pair"] = f"{sym[i]}{i}-{sym[j]}{j}"
            images.append(mol)
    io.write(OUT / "2-bond-order-modes.xyz", images, format="extxyz")
    print(f"wrote {len(images)} modes to {OUT / '2-bond-order-modes.xyz'}")

    # Restricting the CPHF to selected atoms.  Rows requested are exact; rows
    # skipped are left at exactly zero, so this trades coverage for speed.
    restricted: Atoms = build.molecule("H2O")
    restricted.calc = PySCFCalculator(
        charge=0, spin=0, basis="cc-pvdz", bo_grad=True, bo_grad_atoms=[0]
    )
    t0 = time.perf_counter()
    restricted.get_potential_energy()
    t_sub = time.perf_counter() - t0
    BG_sub = bond_order_grad(restricted)

    print(f"\nbo_grad_atoms=[0]: {t_sub:.1f} s vs {t_full:.1f} s full")
    print(f"  requested row 0 vs full   : {np.abs(BG_sub[0] - BG[0]).max():.2e}")
    print(f"  skipped rows 1,2 are zero : {np.all(BG_sub[1:] == 0)}")
    print("""
  Row 0 is computed exactly; it differs only at CPHF convergence level,
  because PySCF solves all perturbations as one block against a single
  criterion, so asking for fewer atoms lands slightly differently inside
  that tolerance.  Skipped rows are exactly zero, not approximate.

  The saving is marginal on 3 atoms -- part of make_h1 runs for every atom
  regardless -- and grows with system size.  7-gradient-methods.py shows
  the exact alternative that does not zero any rows.""")
