from typing import cast
from pathlib import Path
from bondspace.ase import BondFluxCalculator
from ase import Atoms, io
from ase.optimize import FIRE2

RS, TS, PS = cast(list[Atoms], io.read(Path(__file__).parent / "rxn_03.xyz", ":"))
atoms = RS.copy()

atoms.calc = BondFluxCalculator(
    bonds=[(0, 1, 0.0), (1, 3, 1.0)],
    basis="cc-pvdz",
    charge=0,
    spin=1,
    verbose=1,
)

opt = FIRE2(atoms)
opt.attach(io.write, 1, "rxn3.xyz", atoms, append=True)
opt.run(fmax=0.05)


atoms = RS.copy()

atoms.calc = BondFluxCalculator(
    bonds=[(0, 1, 0.5), (1, 3, 0.5)],
    basis="cc-pvdz",
    charge=0,
    spin=1,
    verbose=1,
    thresh=0.05,
    ovlp_thresh=1.0,
)

opt = FIRE2(atoms)
opt.attach(io.write, 1, "rxn3.xyz", atoms, append=True)
opt.run(fmax=0.05)
