from typing import cast
import numpy as np
from bondspace.bond import bo, bo_gradient
from bondspace.util import ase_to_pyscf
from pyscf import gto, dft, grad, lib, scf

from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes


class PySCFCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        charge: int | None = 0,
        spin: int | None = 0,
        xc: str = "HYB_GGA_XC_WB97X_D3",
        basis: str = "cc-pvtz",
        verbose: int = 0,
        threads: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.basis: str = basis
        self.charge: int | None = charge
        self.spin: int | None = spin

        self.energy_pipe = (
            gto.M()
            .set(verbose=verbose)
            .apply(dft.UKS, xc=xc)
            .set(conv_tol=1e-6)
            .density_fit()
        )
        self.forces_scanner: grad.rhf.SCF_GradScanner = (
            self.energy_pipe.nuc_grad_method().as_scanner()
        )
        if threads is not None:
            self.threads = lib.num_threads(n=threads)

        # private
        self.mo_coeff = None

    def calculate(
        self,
        atoms=None,
        properties=["energy", "forces"],
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)
        atoms = cast(Atoms, atoms)

        # energy and forces
        mol: gto.Mole = ase_to_pyscf(
            atoms, basis=self.basis, charge=self.charge, spin=self.spin
        )
        energy, gradient = self.forces_scanner(mol, mo_coeff=self.mo_coeff)
        forces = -gradient

        # bond order
        mf: scf.hf.SCF = self.forces_scanner.base
        bond_order = bo(mf)
        atoms.set_array("bond-order", bond_order, bond_order.dtype)

        conn = []
        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                if (bij := bond_order[i, j]) > 0:
                    conn.append((i, j, bij))
        atoms.info["connectivity"] = conn

        self.results = {
            "energy": energy * units.Hartree,
            "forces": forces * units.Hartree / units.Bohr,
        }


class BondFluxCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        bonds: list[tuple[int, int, float]],
        charge: int = 0,
        spin: int | None = 0,
        xc: str = "HYB_GGA_XC_WB97X_D3",
        basis: str = "cc-pvtz",
        verbose: int = 0,
        threads: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.basis: str = basis
        self.charge: int = charge
        self.spin: int | None = spin

        self.bonds: list[tuple[int, int, int | float]] = bonds

        self.energy_pipe = (
            gto.M()
            .set(verbose=verbose)
            .apply(dft.UKS, xc=xc)
            .set(conv_tol=1e-6)
            .density_fit()
        )
        self.forces_scanner: grad.rhf.SCF_GradScanner = (
            self.energy_pipe.nuc_grad_method().as_scanner()
        )
        if threads is not None:
            self.threads = lib.num_threads(n=threads)

        # private
        self.mo_coeff = None

    def harmonic_constraint(
        self,
        BO: np.ndarray,
        BO_grad: np.ndarray,
    ):
        # norm_bond_grads = np.nan_to_num(
        #     BO_grad / (np.linalg.norm(x=BO_grad, axis=1, keepdims=True) + 1e-8),
        #     nan=0.0,
        #     posinf=0.0,
        #     neginf=0.0,
        # )

        bmat = np.full_like(BO, -1)
        for i, j, v in self.bonds:
            bmat[i, j] = v
            bmat[j, i] = v

        mask = bmat >= 0
        bond_energy = 0.5 * np.sum(np.square(mask * (BO - bmat)))
        bond_forces = -np.sum(mask * (BO - bmat) * BO_grad, (-2, -1))
        # bond_forces = -np.sum(mask * (BO - bmat) * norm_bond_grads, (-2, -1))

        # bond_energy = 0.0
        # bond_forces = np.zeros_like(pes_forces)
        # for i, j, val in self.bonds:
        #     bond_energy += 0.5 * np.square(bond_order[i][j] - val)
        #     bond_forces += 1.0 * (bond_order[i][j] - val) * norm_bond_grads[:, :, i, j]
        #     print(f"bond {i}, {j}, {val:0.2f}, {bond_order[i][j]:0.2f}")
        return bond_energy, bond_forces

    def calculate(
        self,
        atoms=None,
        properties=["energy", "forces"],
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)
        assert isinstance(atoms, Atoms)

        mol: gto.Mole = ase_to_pyscf(
            atoms, basis=self.basis, charge=self.charge, spin=self.spin
        )

        pes_energy, pes_gradient = self.forces_scanner(mol, mo_coeff=self.mo_coeff)
        pes_forces = -pes_gradient

        mf: scf.hf.SCF = self.forces_scanner.base

        bond_order = bo(mf)
        bond_order_grad = bo_gradient(mf)

        conn = []
        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                if (bij := bond_order[i, j]) > 0:
                    conn.append((i, j, bij))
        atoms.info["connectivity"] = conn
        atoms.set_array("bond-order", bond_order, bond_order.dtype)

        bond_energy, bond_forces = self.harmonic_constraint(
            bond_order,
            bond_order_grad,
        )

        self.results = {
            "energy": bond_energy,
            "forces": bond_forces,
        }
