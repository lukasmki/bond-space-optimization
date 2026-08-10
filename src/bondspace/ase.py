from typing import cast
import numpy as np
from bondspace.bond import bo, bo_flux_gradient, bo_gradient, atom_overlap
from bondspace.util import ase_to_pyscf
from pyscf import gto, lib, scf

from pyscf import dft, grad

# from gpu4pyscf import dft, grad

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
        bo_grad_atoms: list[int] | None = None,
        level_shift: float | tuple[float, float] = 0.0,
        conv_tol: float = 1e-6,
        reuse_guess: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.basis: str = basis
        # Seed each SCF from the previous geometry's orbitals.  Cheap when it
        # works, but it makes the surface path-dependent: seeded from a stale
        # guess the SCF can land on a different solution, and the energy then
        # jumps by ~1 eV between adjacent geometries while the forces stay
        # consistent with neither.  A BFGS relaxation driven that way oscillates
        # instead of converging.  Set False whenever the *surface* has to be
        # single-valued -- minimisations, Hessians, IRCs.
        self.reuse_guess: bool = reuse_guess
        self.charge: int | None = charge
        self.spin: int | None = spin
        # Which atoms to solve the density response for; None means all.
        # One CPHF solve per atom, so naming a subset is the way to make
        # bo_grad affordable on larger systems.
        self.bo_grad_atoms: list[int] | None = bo_grad_atoms

        self.energy_pipe = (
            gto.M()
            .set(verbose=verbose)
            .apply(dft.UKS, xc=xc)
            .set(conv_tol=conv_tol, level_shift=level_shift)
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
        self.mo_coeff = mf.mo_coeff if self.reuse_guess else None
        bond_order = bo(mf)
        if self.bo_grad:
            bond_order_grad = bo_gradient(mf, atomlist=self.bo_grad_atoms)
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
        level_shift: float | tuple[float, float] = 0.0,
        restrict_gradient: bool = False,
        zvector: bool = False,
        verbose: int = 0,
        threads: int | None = None,
        conv_tol: float = 1e-6,
        reuse_guess: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.basis: str = basis
        # See PySCFCalculator: reusing the previous geometry's orbitals makes
        # the endpoint of a drive depend on the trajectory that reached it.
        self.reuse_guess: bool = reuse_guess
        self.charge: int = charge
        self.spin: int | None = spin
        self.thresh: float = thresh
        self.ovlp_thresh: float = ovlp_thresh
        self.bonds: list[tuple[int, int, int | float]] = bonds
        self.restrict_gradient: bool = restrict_gradient
        # Adjoint gradient: one CPHF solve for the whole objective instead of
        # one per atom.  Equivalent to the direct path, not an approximation.
        self.zvector: bool = zvector

        self.energy_pipe = (
            gto.M()
            .set(verbose=verbose)
            .apply(dft.UKS, xc=xc)
            .set(conv_tol=conv_tol, level_shift=level_shift)
            .density_fit()
        )
        self.forces_scanner: grad.rhf.SCF_GradScanner = (
            self.energy_pipe.nuc_grad_method().as_scanner()
        )
        if threads is not None:
            self.threads = lib.num_threads(n=threads)

        # private
        self.mo_coeff = None

    def gradient_atoms(self) -> list[int] | None:
        """Atoms whose density response is worth solving for.

        The constraint force on atom k contracts d(bond order)/dR_k, and one
        CPHF solve is needed per atom k, so the cost scales with how many
        atoms are asked for.  Only atoms appearing in a constrained bond can
        move the objective appreciably, so by default the rest are skipped.

        This is an approximation, not a free lunch: a skipped atom gets a
        gradient of exactly zero, and since this calculator returns only the
        constraint force (no PES term), such an atom is frozen for the whole
        optimization.  On H2O2 with only the O-O bond constrained, the
        discarded forces on the two spectator hydrogens measured 4.6e-2 --
        the same order as a typical fmax, so they are not noise.

        Hence the default is off.  It is exactly a no-op (verified bitwise)
        when every atom already appears in ``bonds``, which is the case when
        the full connectivity is targeted; turning it on is only worth it
        when constraining a few bonds of a larger molecule and freezing the
        remainder is acceptable.
        """
        if not self.restrict_gradient:
            return None
        return sorted({a for i, j, _ in self.bonds for a in (int(i), int(j))})

    def targets(self, BO: np.ndarray):
        """Target bond-order matrix and its mask; unlisted pairs are free."""
        bmat = np.full_like(BO, -1)
        for i, j, v in self.bonds:
            bmat[i, j] = v
            bmat[j, i] = v
        return bmat, bmat >= 0

    def overlap_correction(self, bmat, bond_mask, ovlp, ovlp_grad, forces):
        """Repulsion that separates fragments whose bonds should break.

        Bond order goes flat once fragments are apart, so a bond targeted to
        zero stops generating force well before the fragments have actually
        separated; the overlap supplies the missing long-range push.  Purely
        analytic -- no density response, so no CPHF either way.
        """
        energy = 0.0
        ovlp_mask = bond_mask & (bmat < self.ovlp_thresh)
        for i, j in zip(*np.where(ovlp_mask)):
            if j < i:
                continue
            # E = S_ij^2 / 2, so the force on *every* atom is -S_ij dS_ij/dR.
            # Applying it to rows i and j with opposite signs instead cancels
            # the antisymmetry of dS_ij/dR (which already satisfies
            # dS/dR_i = -dS/dR_j) and yields a net translation of the pair
            # rather than a separation.
            energy += 0.5 * ovlp[i, j] * ovlp[i, j]
            forces += -ovlp[i, j] * ovlp_grad[:, :, i, j]
        return energy

    def harmonic_constraint(
        self,
        BO: np.ndarray,
        BO_grad: np.ndarray,
        ovlp: np.ndarray,
        ovlp_grad: np.ndarray,
    ):
        """Constraint energy and force.

        The force is returned per **Bohr**, matching the gradients in
        `bondspace.bond` that it is assembled from; `calculate` converts to
        the eV/Ang that ASE expects.
        """
        bmat, bond_mask = self.targets(BO)
        err = bond_mask * (BO - bmat)

        bond_energy = 0.5 * np.sum(np.square(err))
        bond_forces = -np.sum(err * BO_grad, (-2, -1))

        # magnitude, not signed: a bond that is under-formed (BO < target) has
        # negative error, and masked-out pairs contribute 0, so a signed max
        # reports 0.0 for any purely bond-forming target and stops immediately.
        if np.max(np.abs(err)) < self.thresh:
            return 0.0, np.zeros_like(bond_forces)

        bond_energy += self.overlap_correction(
            bmat, bond_mask, ovlp, ovlp_grad, bond_forces
        )
        return bond_energy, bond_forces

    def harmonic_constraint_zvector(
        self,
        mf: scf.hf.SCF,
        BO: np.ndarray,
        ovlp: np.ndarray,
        ovlp_grad: np.ndarray,
    ):
        """Same constraint, but the force comes from one adjoint solve.

        The objective is a single scalar, so its gradient does not need every
        d(bond order)/dR separately.  ``bo_flux_gradient`` contracts first and
        solves once, giving forces on all atoms at a cost independent of atom
        count -- so unlike ``restrict_gradient`` this is exact, and it makes
        that approximation unnecessary.

        As with ``harmonic_constraint``, the force is per **Bohr**; the
        conversion to eV/Ang happens in ``calculate``.
        """
        bmat, bond_mask = self.targets(BO)
        err = bond_mask * (BO - bmat)

        # Check before solving: the direct path builds its gradient first and
        # discards it here, but the adjoint solve is the expensive step.
        if np.max(np.abs(err)) < self.thresh:
            return 0.0, np.zeros((len(BO), 3))

        bond_energy = 0.5 * np.sum(np.square(err))
        bond_forces = -bo_flux_gradient(mf, err)

        bond_energy += self.overlap_correction(
            bmat, bond_mask, ovlp, ovlp_grad, bond_forces
        )
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
        # pes_forces = -pes_gradient

        mf: scf.hf.SCF = self.forces_scanner.base
        self.mo_coeff = mf.mo_coeff if self.reuse_guess else None
        atomlist = self.gradient_atoms()
        bond_order = bo(mf)
        aov, aov_grad = atom_overlap(mf, atomlist=atomlist)

        # save bond/connectivity info
        conn = []
        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                if (bij := bond_order[i, j]) > 0:
                    conn.append((i, j, bij))
        atoms.info["connectivity"] = conn
        atoms.set_array("bond-order", bond_order, bond_order.dtype)

        if self.zvector:
            bond_energy, bond_forces = self.harmonic_constraint_zvector(
                mf,
                bond_order,
                aov,
                aov_grad,
            )
        else:
            bond_energy, bond_forces = self.harmonic_constraint(
                bond_order,
                bo_gradient(mf, atomlist=atomlist),
                aov,
                aov_grad,
            )

        self.results = {
            # The constraint energy is a sum of squared bond orders, taken as
            # eV by fiat -- it is fictitious, so its scale is a convention.
            # Its gradient, though, is not: every derivative in bondspace.bond
            # is with respect to nuclear position in Bohr, because they are
            # built from PySCF integrals.  ASE requires eV/Ang, so convert
            # here, mirroring what PySCFCalculator does with its own results.
            "energy": bond_energy,
            "forces": bond_forces / units.Bohr,
        }
