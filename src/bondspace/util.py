from ase import Atoms
from pyscf import gto


def ase_to_pyscf(
    atoms: Atoms,
    basis: str = "cc-pvdz",
    charge: int | None = None,
    spin: int | None = None,
) -> gto.Mole:
    Z = atoms.get_atomic_numbers()
    R = atoms.get_positions()

    molstr = "\n".join(
        f"{Z[i]} {R[i][0]} {R[i][1]} {R[i][2]}\n" for i in range(len(atoms))
    )
    mol: gto.Mole = gto.M(
        atom=molstr,
        basis=basis,
        charge=charge,
        spin=spin,
    )
    return mol


def pyscf_to_ase(mol: gto.Mole) -> Atoms:
    return Atoms(
        symbols=mol.elements,
        positions=mol.atom_coords(unit="Angstrom"),
    )
