from bondspace.ase import PySCFCalculator
from typing import cast
from pathlib import Path
from ase import io, Atoms

from util import rxn_data

if __name__ == "__main__":
    root = Path(__file__).parent
    (root / "HCombustion-bso").mkdir(parents=True, exist_ok=True)
    for file in sorted((root / "HCombustion").glob("*.xyz")):
        print(file.stem)
        atoms = cast(list[Atoms], io.read(file, index=":"))
        rdata = rxn_data[file.stem]
        images = []
        for at in atoms:
            at.calc = PySCFCalculator(**rdata, basis="cc-pvdz", level_shift=(3, 2))
            at.get_potential_energy()
            images.append(at.copy())
        io.write((root / "HCombustion-bso" / file.name), images, format="extxyz")
