"""Runs bond space optimizations to check the following:
1. TS -> R
2. TS -> P
3. R -> TS
4. P -> TS
5. R -> P
6. P -> R

Set BONDSPACE_RXN to a single reaction stem (e.g. "rxn_03") to run only that
one.  The reactions are independent, so a job array is the effective way to
parallelise this -- see run.slurm.  Selecting explicitly is required rather
than optional: the skip-if-output-exists check only sees a finished reaction,
because the trajectory is written at the end, so array tasks launched together
would all pick the same first pending reaction and duplicate the whole thing.
"""

import os

from bondspace.ase import BondFluxCalculator

from ase import Atoms, io
from ase.optimize import FIRE2
from typing import cast

import numpy as np

from pathlib import Path

from util import configure_threads, rxn_data


def run(
    id: str, atoms: Atoms, bonds: list[tuple], charge: int, spin: int
) -> list[Atoms]:
    images: list[Atoms] = []

    def add_image(image: Atoms) -> None:
        image.info["id"] = id
        images.append(image.copy())

    # bonds = [(int(row[0]), int(row[1]), float(np.round(row[2], 0))) for row in conn]

    print("ID:", id, "TARGET:", bonds, flush=True)
    print("CONN:\n", atoms.info["connectivity"], flush=True)

    atoms.calc = BondFluxCalculator(
        bonds,
        charge=charge,
        spin=spin,
        basis="cc-pvdz",
        thresh=0.05,
        # Above the break targets (0.0), below every target meant to survive.
        # This used to be 2.0, which put the overlap repulsion on all the 0.5
        # and 1.0 targets as well.  That was survivable only while the
        # correction was mis-signed and merely translated each pair; now that
        # it genuinely separates them it fights the bonds being formed.  On
        # rxn_03 R->TS it costs an order of magnitude: 0.479 vs 0.044 max
        # bond-order error against the reference TS.
        ovlp_thresh=0.5,
        # One CPHF solve for the whole objective instead of one per atom.
        # Exact, and the gain grows with system size.
        zvector=True,
        level_shift=(0.3, 0.2),
    )
    # maxstep well under FIRE2's 0.2 A default: the constraint energy is a sum
    # of squared bond orders, so its scale is unrelated to a real PES and the
    # default step overshoots into geometries where the SCF diverges.
    opt = FIRE2(atoms, maxstep=0.05)
    opt.attach(add_image, 1, atoms)
    # Smaller steps need a larger budget; zvector roughly halves the per-step
    # cost, so 80 steps here costs about what 50 used to.
    opt.run(fmax=0.1, steps=80)
    return images


if __name__ == "__main__":
    configure_threads()
    root = Path(__file__).parent
    (root / "HCombustion-paths").mkdir(parents=True, exist_ok=True)

    only = os.environ.get("BONDSPACE_RXN")
    if only:
        print(f"restricted to {only}", flush=True)

    for file in reversed(sorted((root / "HCombustion-bso").glob("*.xyz"))):
        if only and file.stem != only:
            continue

        outfile = root / "HCombustion-paths" / f"{file.stem}.xyz"
        if outfile.exists():
            continue

        print(file.stem)
        atoms = cast(list[Atoms], io.read(file, index=":"))
        RS, TS, PS = atoms
        rdata = rxn_data[file.stem]

        Bi = RS.get_array("bond-order")
        Bf = PS.get_array("bond-order")
        Bm = 0.5 * (Bi + Bf)

        conn = []
        for i in range(Bm.shape[0]):
            for j in range(i + 1, Bm.shape[1]):
                if (bij := Bm[i, j]) > 0:
                    conn.append((i, j, bij))
        TS.info["connectivity"] = conn

        ts_conn = [
            (int(row[0]), int(row[1]), float(0.5 * np.round(2 * row[2], 0)))
            for row in TS.info["connectivity"]
        ]
        rs_conn = [
            (int(row[0]), int(row[1]), float(np.round(row[2], 0)))
            for row in RS.info["connectivity"]
        ]
        ps_conn = [
            (int(row[0]), int(row[1]), float(np.round(row[2], 0)))
            for row in PS.info["connectivity"]
        ]

        ts_bonds = list(set(ts_conn) - set(rs_conn) - set(ps_conn))
        rs_bonds = list(set(rs_conn) - set(ps_conn))
        ps_bonds = list(set(ps_conn) - set(rs_conn))

        print(rs_bonds)
        print(ps_bonds)

        charge = rdata["charge"]
        spin = rdata["spin"]

        # transition state
        ts2rs = run(
            f"{file.stem}-ts2rs",
            TS.copy(),
            bonds=rs_bonds,
            charge=charge,
            spin=spin,
        )
        ts2ps = run(
            f"{file.stem}-ts2ps",
            TS.copy(),
            bonds=ps_bonds,
            charge=charge,
            spin=spin,
        )
        # reactant
        rs2ts = run(
            f"{file.stem}-rs2ts",
            RS.copy(),
            bonds=ts_bonds,
            charge=charge,
            spin=spin,
        )
        rs2ps = run(
            f"{file.stem}-rs2ps",
            RS.copy(),
            bonds=ps_bonds,
            charge=charge,
            spin=spin,
        )
        # product
        ps2ts = run(
            f"{file.stem}-ps2ts",
            PS.copy(),
            bonds=ts_bonds,
            charge=charge,
            spin=spin,
        )
        ps2rs = run(
            f"{file.stem}-ps2rs",
            PS.copy(),
            bonds=rs_bonds,
            charge=charge,
            spin=spin,
        )
        io.write(
            outfile,
            ts2rs + ts2ps + rs2ts + rs2ps + ps2ts + ps2rs,
            format="extxyz",
        )
