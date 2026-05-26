from typing import cast
import numpy as np
from pyscf import scf, gto, hessian


def bo(mf: scf.hf.SCF, ov=None, dm=None):
    mol: gto.Mole = mf.mol
    ao_idx = np.asarray([x[0] for x in mol.ao_labels(fmt=False)])

    # density matrix
    if dm is None:
        dm: np.ndarray = mf.make_rdm1()
    if dm.ndim == 2:
        dma, dmb = 0.5 * dm, 0.5 * dm
    else:
        dma, dmb = dm

    # overlap matrix
    if ov is None:
        ov: np.ndarray = mf.get_ovlp()

    # bond order
    DS: np.ndarray = (dma + dmb) @ ov
    RS: np.ndarray = (dma - dmb) @ ov
    Bu = DS * DS.T + RS * RS.T
    B = np.zeros((mol.natm, mol.natm))
    np.add.at(B, (ao_idx[:, None], ao_idx[None, :]), Bu)

    return B


def bo_gradient(
    mf: scf.hf.SCF,
    ov=None,
    ov_grad=None,
    dm=None,
    dm_grad=None,
    atomlist=None,
):
    mol: gto.Mole = mf.mol
    ao_idx = np.asarray([x[0] for x in mol.ao_labels(fmt=False)])

    # overlap matrix
    if ov is None:
        # <i|OVLP|j>
        ov: np.ndarray = mol.intor("int1e_ovlp")
    if ov_grad is None:
        ov_grad: np.ndarray = ov_gradient(mf)

    # density matrix
    if dm is None:
        dm: np.ndarray = mf.make_rdm1()
    if dm.ndim == 2:
        dma, dmb = 0.5 * dm, 0.5 * dm
    else:
        dma, dmb = dm

    if dm_grad is None:
        dm_grad: np.ndarray = dm_gradient(mf, ov_grad=ov_grad, atomlist=atomlist)
    dma_grad, dmb_grad = dm_grad

    # bond order gradient
    DS: np.ndarray = (dma + dmb) @ ov
    RS: np.ndarray = (dma - dmb) @ ov
    GDS: np.ndarray = (dma_grad + dmb_grad) @ ov + (dma + dmb) @ ov_grad
    GRS: np.ndarray = (dma_grad - dmb_grad) @ ov + (dma - dmb) @ ov_grad

    # BG[k, 3, i, j] = gradient of bond order ij w.r.t. position of atom k
    BG = np.zeros((mol.natm, 3, mol.natm, mol.natm))
    for ib, ia in enumerate(ao_idx):
        for jb, ja in enumerate(ao_idx):
            BG[:, :, ia, ja] += (
                DS[ib, jb] * GDS[:, :, jb, ib] + GDS[:, :, ib, jb] * DS[jb, ib]
            )
            BG[:, :, ia, ja] += (
                RS[ib, jb] * GRS[:, :, jb, ib] + GRS[:, :, ib, jb] * RS[jb, ib]
            )
    return BG


def dm_gradient(mf: scf.hf.SCF, ov_grad=None, atomlist=None):
    mol: gto.Mole = mf.mol

    mo_coeff = cast(np.ndarray, mf.mo_coeff)
    mo_occ = cast(np.ndarray, mf.mo_occ)
    mo_energy = cast(np.ndarray, mf.mo_energy)
    if atomlist is None:
        atomlist = range(mol.natm)
    if ov_grad is None:
        ov_grad: np.ndarray = ov_gradient(mf)

    mocca = mo_coeff[0][:, mo_occ[0] > 0]
    moccb = mo_coeff[1][:, mo_occ[1] > 0]

    # occupied-occupied
    dma_oo, dmb_oo = dm_gradient_oo((mocca, moccb), ov_grad)

    # occupied-virtual
    hess_uks = hessian.uks.Hessian(mf)
    h1ao = hess_uks.make_h1(mo_coeff, mo_occ, atmlst=atomlist)
    mo1, mo1e = hess_uks.solve_mo1(mo_energy, mo_coeff, mo_occ, h1ao, atmlst=atomlist)
    mo1a, mo1b = mo1
    mo1a = np.asarray(
        [x if x is not None else np.zeros((3,) + mocca.shape) for x in mo1a]
    )
    mo1b = np.asarray(
        [x if x is not None else np.zeros((3,) + moccb.shape) for x in mo1b]
    )
    dma_ov, dmb_ov = dm_gradient_ov((mocca, moccb), (mo1a, mo1b))

    # return dma_oo, dmb_oo
    # return dma_ov, dmb_ov
    return dma_oo + dma_ov, dmb_oo + dmb_ov


def dm_gradient_ov(mocc, mo1):
    mocca, moccb = mocc
    mo1a, mo1b = mo1

    # dma = (natm, 3, nao, nao)
    dma_ov = np.einsum("nxpi,qi->nxpq", mo1a, mocca) + np.einsum(
        "qi,nxpi->nxqp", mocca, mo1a
    )
    dmb_ov = np.einsum("nxpi,qi->nxpq", mo1b, moccb) + np.einsum(
        "qi,nxpi->nxqp", moccb, mo1b
    )
    return dma_ov, dmb_ov


def dm_gradient_oo(mocc, ov_grad):
    mocca, moccb = mocc

    S_mo_a = np.einsum("pi,nxpq,qj->nxij", mocca, ov_grad, mocca)
    S_mo_b = np.einsum("pi,nxpq,qj->nxij", moccb, ov_grad, moccb)

    # occ-occ contribution to dD/dR
    dma_oo = -np.einsum("nxij,pi,qj->nxpq", S_mo_a, mocca, mocca)
    dma_oo -= np.einsum("nxij,pj,qi->nxpq", S_mo_a, mocca, mocca)
    dmb_oo = -np.einsum("nxij,pi,qj->nxpq", S_mo_b, moccb, moccb)
    dmb_oo -= np.einsum("nxij,pj,qi->nxpq", S_mo_b, moccb, moccb)

    return dma_oo, dmb_oo


def ov_gradient(mf: scf.hf.SCF):
    mol: gto.Mole = mf.mol
    ao_idx = np.asarray([x[0] for x in mol.ao_labels(fmt=False)])

    # <NABLA i|OVLP|j> = (3, nao i, nao j)
    ao_ov_grad: np.ndarray = mol.intor("int1e_ipovlp")

    # <NABLA i|OVLP|j> + <i|OVLP|NABLA j> = (natm, 3, nao i, nao j)
    ov_grad = np.zeros((mol.natm,) + ao_ov_grad.shape)
    for ib, ia in enumerate(ao_idx):
        ov_grad[ia, :, ib, :] -= ao_ov_grad[:, ib, :]
        ov_grad[ia, :, :, ib] -= ao_ov_grad.mT[:, ib, :]

    return ov_grad
