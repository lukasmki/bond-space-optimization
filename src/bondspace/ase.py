from typing import cast
import numpy as np
from bondspace.bond import bo, bo_gradient, atom_overlap
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
        bo_grad: bool = False,
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
        self.bo_grad = bo_grad

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
        if self.bo_grad:
            bond_order_grad = bo_gradient(mf)
            atoms.set_array(
                "bond-order-grad",
                np.reshape(
                    bond_order_grad, (len(atoms), -1)
                ),  # (natm, 3, natm, natm) -> (natm, 3*natm*natm)
                bond_order_grad.dtype,
            )
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
        thresh: float = 0.05,
        ovlp_thresh: float = 0.1,
        verbose: int = 0,
        threads: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.basis: str = basis
        self.charge: int = charge
        self.spin: int | None = spin
        self.thresh: float = thresh
        self.ovlp_thresh: float = ovlp_thresh
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
        ovlp: np.ndarray,
        ovlp_grad: np.ndarray,
    ):
        # target bond order matrix
        bmat = np.full_like(BO, -1)
        for i, j, v in self.bonds:
            bmat[i, j] = v
            bmat[j, i] = v
        bond_mask = bmat >= 0

        bond_energy = 0.5 * np.sum(np.square(bond_mask * (BO - bmat)))
        bond_forces = -np.sum(bond_mask * (BO - bmat) * BO_grad, (-2, -1))

        max_error = np.max(bond_mask * (BO - bmat))
        if max_error < self.thresh:
            return 0.0, np.zeros_like(bond_forces)

        # set bonds that should break
        ovlp_mask = bond_mask & (bmat < self.ovlp_thresh)
        for i, j in zip(*np.where(ovlp_mask)):
            if j < i:
                continue
            ovlp_energy_ij = 0.5 * ovlp[i, j] * ovlp[i, j]
            ovlp_force_ij = -ovlp[i, j] * ovlp_grad[:, :, i, j]
            bond_energy += ovlp_energy_ij
            bond_forces[i] += ovlp_force_ij[i]
            bond_forces[j] -= ovlp_force_ij[j]

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
        aov, aov_grad = atom_overlap(mf)

        # save bond/connectivity info
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
            aov,
            aov_grad,
        )

        self.results = {
            "energy": bond_energy,
            "forces": bond_forces,
        }
