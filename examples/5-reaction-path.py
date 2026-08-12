"""Walk a real reaction: H2 + OH -> H + H2O.

`rxn_03.xyz` holds reference reactant, transition-state and product geometries
for one of the HCombustion reactions.  Starting from the reactant only, this
drives to each of the other two by asking for their bond orders -- no energy
gradient, no TS search, no knowledge of the path.

The transition state is reached simply by asking for *half* bonds: the
transferring hydrogen is targeted at 0.5 to each partner.  That is the point
of working in bond space -- a half-formed bond is a perfectly ordinary target
here, while on the potential energy surface it is a saddle point that needs
dedicated machinery to find.

This is the slowest example (~4-6 min); everything else runs in about a minute.

Run: uv run python examples/5-reaction-path.py
"""

from pathlib import Path
from typing import cast

import numpy as np
from ase import Atoms, io
from ase.optimize import FIRE2

from bondspace.ase import BondFluxCalculator

HERE = Path(__file__).parent
OUT = HERE / "out"

# H0-H1 is the H2 molecule, H2-O3 is the hydroxyl; H1 is the atom that
# transfers.  Targets are read off the reference frames' connectivity.
CHARGE, SPIN = 0, 1  # H,H,H,O = 11 electrons -> one unpaired


def drive(start: Atoms, bonds, label: str) -> tuple[Atoms, list[Atoms], bool]:
    atoms = start.copy()
    atoms.calc = BondFluxCalculator(
        bonds,
        charge=CHARGE,
        spin=SPIN,
        basis="cc-pvdz",
        thresh=0.05,
        ovlp_thresh=0.5,  # applies to the 0.0 target only, not to 0.5 or 1.0
        level_shift=(0.3, 0.2),
    )
    images: list[Atoms] = []
    # maxstep well below FIRE2's 0.2 A default.  The constraint "energy" is a
    # sum of squared bond orders, so its scale -- and therefore the size of
    # the force FIRE accelerates on -- has no relation to a real potential
    # energy surface.  With the default step the optimizer overshoots into
    # geometries where the SCF diverges.
    opt = FIRE2(atoms, logfile=None, maxstep=0.05)
    opt.attach(lambda: images.append(atoms.copy()), 1)
    print(f"\n--- {label}   targets {bonds}")
    converged = bool(opt.run(fmax=0.1, steps=80))
    print(f"    {len(images)} steps, converged={converged}")
    return atoms, images, converged


def compare(atoms: Atoms, reference: Atoms, label: str) -> float:
    """Largest bond-order deviation from the reference frame, over all pairs."""
    B = atoms.get_array("bond-order")
    Bref = reference.get_array("bond-order")
    n = len(atoms)
    iu = np.triu_indices(n, 1)
    err = np.abs(B[iu] - Bref[iu])
    print(f"    bond orders vs reference {label}:")
    sym = atoms.get_chemical_symbols()
    for (i, j), e in zip(zip(*iu), err):
        if Bref[i, j] > 0.1 or B[i, j] > 0.1:
            print(
                f"      {sym[i]}{i}-{sym[j]}{j}   got {B[i, j]:.3f}   "
                f"reference {Bref[i, j]:.3f}   err {e:.3f}"
            )
    return float(err.max())


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    RS, TS, PS = cast(list[Atoms], io.read(HERE / "rxn_03.xyz", ":"))

    print("reference connectivity:")
    for name, frame in (("R ", RS), ("TS", TS), ("P ", PS)):
        pairs = [
            (int(a), int(b), float(c))
            for a, b, c in frame.info["connectivity"]
            if c > 0.1
        ]
        print(f"  {name}: " + "  ".join(f"{i}-{j}={v:.2f}" for i, j, v in pairs))

    # Reactant -> transition state: ask for two half bonds.
    ts_atoms, ts_traj, ts_ok = drive(RS, [(0, 1, 0.5), (1, 3, 0.5)], "R -> TS")
    ts_err = compare(ts_atoms, TS, "TS")

    # Reactant -> product: break H0-H1, form H1-O3.
    ps_atoms, ps_traj, ps_ok = drive(RS, [(0, 1, 0.0), (1, 3, 1.0)], "R -> P")
    ps_err = compare(ps_atoms, PS, "P")

    io.write(OUT / "5-reaction-path.xyz", ts_traj + ps_traj, format="extxyz")

    print(f"\nmax bond-order error   R->TS: {ts_err:.3f}   R->P: {ps_err:.3f}")
    print(
        f"wrote {len(ts_traj) + len(ps_traj)} frames to {OUT / '5-reaction-path.xyz'}"
    )
    # data/analysis.ipynb counts a run as successful below 0.5.
    ok = ts_err < 0.5 and ps_err < 0.5
    print("both endpoints reached" if ok else "at least one endpoint missed")
