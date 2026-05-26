from bondspace.ase import BondFluxCalculator
from ase import Atoms, Atom, io
from ase.optimize import FIRE2

atoms = Atoms(
    [
        Atom("H", [0.0, 0.0, 0.0]),
        Atom("H", [2.0, 0.0, 0.0]),
        Atom("H", [4.0, 0.0, 0.0]),
    ]
)
atoms.calc = BondFluxCalculator(
    bonds=[(0, 1, 1.0), (1, 2, 0.0)],
    charge=0,
    spin=1,
    verbose=1,
)

opt = FIRE2(atoms)
opt.attach(io.write, 1, "examples/h2.xyz", atoms, append=True)
opt.run(fmax=0.10)
