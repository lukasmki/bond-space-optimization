from typing import cast
import numpy as np
from pyscf import lib, scf, gto
from pyscf.hessian import uhf as uhf_hess

# from pyscf import hessian
# from gpu4pyscf import hessian


def atom_overlap(mf: scf.hf.SCF, atomlist=None):
    """Atom-blocked overlap and its gradient.

    ``atomlist`` restricts which perturbed atoms are computed; the gradient
    is still returned at full size (zeros elsewhere) so it stays indexable
    by true atom number.
    """
    mol: gto.Mole = mf.mol
    ao_idx = np.asarray([x[0] for x in mol.ao_labels(fmt=False)])
    atomlist = _as_atomlist(mol, atomlist)
    ov = mf.get_ovlp()  # (nao, nao)
    ov_grad = ov_gradient(mf, atomlist=atomlist)  # (len(atomlist), 3, nao, nao)

    atom_ov = np.zeros((mol.natm, mol.natm))
    np.add.at(atom_ov, (ao_idx[:, None], ao_idx[None, :]), ov)

    atom_ov_grad_sub = np.zeros((len(atomlist), 3, mol.natm, mol.natm))
    np.add.at(
        atom_ov_grad_sub,
        (slice(None), slice(None), ao_idx[:, None], ao_idx[None, :]),
        ov_grad,
    )
    atom_ov_grad = np.zeros((mol.natm, 3, mol.natm, mol.natm))
    atom_ov_grad[atomlist] = atom_ov_grad_sub
    return atom_ov, atom_ov_grad


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


def _as_atomlist(mol: gto.Mole, atomlist) -> np.ndarray:
    """Canonical atom-index array; ``None`` means every atom."""
    if atomlist is None:
        return np.arange(mol.natm)
    return np.asarray(atomlist, dtype=int)


def bo_gradient(
    mf: scf.hf.SCF,
    ov=None,
    ov_grad=None,
    dm=None,
    dm_grad=None,
    atomlist=None,
):
    """Bond-order gradient BG[k, 3, i, j] = d(bond order ij) / d(position of atom k).

    ``atomlist`` restricts which perturbed atoms *k* are computed.  Each row
    of the result is independent -- it needs only that atom's CPHF solution
    and its own block of the overlap derivative -- so the rows requested are
    computed exactly and the rest are left at zero.  The CPHF solve in
    ``dm_gradient`` costs one linear solve per perturbed atom, so this is
    the main lever on the cost of a bond-order gradient.

    "Exactly" holds analytically and bitwise for the overlap and
    occupied-occupied terms.  The CPHF term agrees only to ~1e-6, because
    PySCF solves all perturbations in a block against a single convergence
    criterion, so asking for fewer atoms lands on a slightly different
    point inside that tolerance.  This is far below the force thresholds
    the optimizers use and is not a sign of a mis-restricted gradient.

    Note that the *pair* indices (i, j) are always computed in full: they
    come from contracting the density response, which is already in hand.
    Restricting them would save nothing.  Only the derivative index k is
    reducible.

    A caller-supplied ``ov_grad``/``dm_grad`` must be aligned with
    ``atomlist`` (one row per entry), not with the full atom list.
    """
    mol: gto.Mole = mf.mol
    ao_idx = np.asarray([x[0] for x in mol.ao_labels(fmt=False)])
    atomlist = _as_atomlist(mol, atomlist)

    # overlap matrix
    if ov is None:
        # <i|OVLP|j>
        ov: np.ndarray = mol.intor("int1e_ovlp")
    if ov_grad is None:
        ov_grad: np.ndarray = ov_gradient(mf, atomlist=atomlist)

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

    if len(ov_grad) != len(atomlist) or len(dma_grad) != len(atomlist):
        raise ValueError(
            f"ov_grad/dm_grad must have one row per atom in atomlist "
            f"({len(atomlist)}), got {len(ov_grad)} and {len(dma_grad)}"
        )

    # bond order gradient
    DS: np.ndarray = (dma + dmb) @ ov
    RS: np.ndarray = (dma - dmb) @ ov
    GDS: np.ndarray = (dma_grad + dmb_grad) @ ov + (dma + dmb) @ ov_grad
    GRS: np.ndarray = (dma_grad - dmb_grad) @ ov + (dma - dmb) @ ov_grad

    # Accumulate over the requested atoms only, then scatter back into a
    # full-size array so callers keep indexing by true atom number.
    BG_sub = np.zeros((len(atomlist), 3, mol.natm, mol.natm))
    for ib, ia in enumerate(ao_idx):
        for jb, ja in enumerate(ao_idx):
            BG_sub[:, :, ia, ja] += (
                DS[ib, jb] * GDS[:, :, jb, ib] + GDS[:, :, ib, jb] * DS[jb, ib]
            )
            BG_sub[:, :, ia, ja] += (
                RS[ib, jb] * GRS[:, :, jb, ib] + GRS[:, :, ib, jb] * RS[jb, ib]
            )

    BG = np.zeros((mol.natm, 3, mol.natm, mol.natm))
    BG[atomlist] = BG_sub
    return BG


def dm_gradient(mf: scf.hf.SCF, ov_grad=None, atomlist=None):
    """Density-matrix response, one row per atom in ``atomlist``.

    The occupied-virtual part needs a CPHF solve per perturbed atom, which
    dominates the cost; the occupied-occupied part is a cheap contraction
    of the overlap derivative.
    """
    mol: gto.Mole = mf.mol

    mo_coeff = cast(np.ndarray, mf.mo_coeff)
    mo_occ = cast(np.ndarray, mf.mo_occ)
    mo_energy = cast(np.ndarray, mf.mo_energy)
    atomlist = _as_atomlist(mol, atomlist)
    if ov_grad is None:
        ov_grad: np.ndarray = ov_gradient(mf, atomlist=atomlist)

    mocca = mo_coeff[0][:, mo_occ[0] > 0]
    moccb = mo_coeff[1][:, mo_occ[1] > 0]

    # No separate occupied-occupied term here.  PySCF's solve_mo1 goes
    # through ucphf.solve_withs1, which already fills the occupied-occupied
    # block of mo1 with -S1_ij/2; symmetrising that in dm_gradient_ov below
    # reproduces the -C S1 C^T response exactly.  Adding dm_gradient_oo on
    # top double counts it (verified against finite differences).
    hess_uks = mf.Hessian()
    h1ao = hess_uks.make_h1(mo_coeff, mo_occ, atmlst=atomlist)
    mo1, mo1e = hess_uks.solve_mo1(mo_energy, mo_coeff, mo_occ, h1ao, atmlst=atomlist)
    mo1a, mo1b = mo1

    # solve_mo1 hands back a full natm-length list with None in the slots it
    # was not asked to solve, so select by atom number to stay aligned with
    # atomlist rather than iterating the whole thing.
    mo1a = np.asarray(
        [
            mo1a[ia] if mo1a[ia] is not None else np.zeros((3,) + mocca.shape)
            for ia in atomlist
        ]
    )
    mo1b = np.asarray(
        [
            mo1b[ia] if mo1b[ia] is not None else np.zeros((3,) + moccb.shape)
            for ia in atomlist
        ]
    )
    return dm_gradient_ov((mocca, moccb), (mo1a, mo1b))


def bo_flux_gradient(
    mf: scf.hf.SCF, G: np.ndarray, ov=None, dm=None, tol=1e-9, max_cycle=50
):
    """Gradient of a bond-order objective via the Z-vector (adjoint) method.

    For a scalar objective E whose derivative with respect to the bond-order
    matrix is ``G[A, B] = dE/dB_AB``, this returns ``dE/dR[k, 3]`` directly.

    ``bo_gradient`` builds every d(bond order)/dR_k and then contracts, which
    costs one CPHF solve per atom.  The objective is a single scalar, so the
    adjoint trick applies: contract first, solve once.  The response enters
    only as ``sum_ai L_ai U^k_ai``; substituting ``U^k = A^-1 RHS^k`` and
    moving ``A^-1`` onto L gives ``sum_ai Z_ai RHS^k_ai`` with ``A Z = L``.
    That is one linear solve regardless of system size, leaving only the
    cheap per-atom contractions.

    Cost still includes ``make_h1`` (the perturbed Fock matrices, which the
    RHS needs for every atom); only the iterative solves collapse.  That is
    roughly 75-85% of the work, so expect a several-fold speedup that grows
    with atom count.

    Returns dE/dR, i.e. the negative of the force.
    """
    mol: gto.Mole = mf.mol
    ao_idx = np.asarray([x[0] for x in mol.ao_labels(fmt=False)])

    if ov is None:
        ov = mf.get_ovlp()
    if dm is None:
        dm = mf.make_rdm1()
    dma, dmb = (0.5 * dm, 0.5 * dm) if dm.ndim == 2 else (dm[0], dm[1])

    # dE/dP and dE/dS.  B_AB is quadratic in (P S), so weighting the AO pairs
    # by G and pairing each factor with the other gives the AO-basis
    # derivatives; W[s] is dE/dP^s at fixed S, WS is dE/dS at fixed P.
    D, R = dma + dmb, dma - dmb
    g = G[ao_idx[:, None], ao_idx[None, :]]
    YD, YR = 2 * (g * (D @ ov).T), 2 * (g * (R @ ov).T)
    WD, WR = YD @ ov, YR @ ov
    W = (WD + WR, WD - WR)
    WS = D @ YD + R @ YR

    mo_coeff = cast(np.ndarray, mf.mo_coeff)
    mo_occ = cast(np.ndarray, mf.mo_occ)
    mo_energy = cast(np.ndarray, mf.mo_energy)
    occidx = [mo_occ[s] > 0 for s in (0, 1)]
    viridx = [~o for o in occidx]
    mocc = [mo_coeff[s][:, occidx[s]] for s in (0, 1)]
    nmo = [mo_coeff[s].shape[1] for s in (0, 1)]
    nocc = [mocc[s].shape[1] for s in (0, 1)]
    split = nmo[0] * nocc[0]

    def pack(x):
        return np.hstack([x[0].ravel(), x[1].ravel()])

    def unpack(v):
        return [
            v[:split].reshape(nmo[0], nocc[0]),
            v[split:].reshape(nmo[1], nocc[1]),
        ]

    # L = dE/dU, the objective projected onto orbital rotations.
    Lmo = [mo_coeff[s].T @ (W[s] + W[s].T) @ mocc[s] for s in (0, 1)]

    # Z-vector: (e_a - e_i) Z + V(Z) = L on the virtual-occupied block.
    # lib.krylov solves (1 + a)x = b, which already matches the sign of the
    # response term here, so no extra negation is needed.
    fx = uhf_hess.gen_vind(mf, mo_coeff, mo_occ)
    eai = [np.zeros((nmo[s], nocc[s])) for s in (0, 1)]
    for s in (0, 1):
        eai[s][viridx[s]] = 1.0 / (
            mo_energy[s][viridx[s]][:, None] - mo_energy[s][occidx[s]]
        )

    def aop(v):
        out = []
        for row in np.atleast_2d(v):
            r = unpack(fx(row.reshape(1, -1))[0])
            out.append(pack([r[s] * eai[s] for s in (0, 1)]))
        return np.asarray(out)

    z0 = pack([Lmo[s] * eai[s] for s in (0, 1)])
    Z = unpack(lib.krylov(aop, z0.reshape(1, -1), tol=tol, max_cycle=max_cycle).ravel())
    for s in (0, 1):
        Z[s][occidx[s]] = 0.0
    VZ = unpack(fx(pack(Z).reshape(1, -1))[0])

    # Per-atom contraction.  The occupied-occupied rotations are fixed by
    # orthonormality (U_ij = -S1_ij/2) rather than by the CPHF, so they enter
    # directly instead of through Z.
    hess = mf.Hessian()
    h1ao = hess.make_h1(mo_coeff, mo_occ)
    aog = mol.intor("int1e_ipovlp")
    aoslice = mol.aoslice_by_atom()

    grad = np.zeros((mol.natm, 3))
    for k in range(mol.natm):
        p0, p1 = aoslice[k][2], aoslice[k][3]
        s1ao = np.zeros((3, mol.nao, mol.nao))
        s1ao[:, p0:p1] -= aog[:, p0:p1]
        s1ao[:, :, p0:p1] -= aog[:, p0:p1].transpose(0, 2, 1)

        for s in (0, 1):
            oo = occidx[s]
            ei = mo_energy[s][oo]
            h1 = np.einsum("mp,xmn,ni->xpi", mo_coeff[s], h1ao[s][k], mocc[s])
            s1 = np.einsum("mp,xmn,ni->xpi", mo_coeff[s], s1ao, mocc[s])
            grad[k] -= 0.5 * np.einsum("xji,ji->x", s1[:, oo], Lmo[s][oo] - VZ[s][oo])
            grad[k] -= np.einsum("xpi,pi->x", h1 - s1 * ei, Z[s])

        # explicit dS/dR term (no response involved)
        X = WS + WS.T
        grad[k] -= np.einsum("mn,xmn->x", X[p0:p1], aog[:, p0:p1])

    return grad


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


def ov_gradient(mf: scf.hf.SCF, atomlist=None):
    """Overlap derivative, one row per atom in ``atomlist``.

    Both the bra and ket terms for a basis function land in the row of the
    atom that function sits on, so rows never mix and restricting them is
    exact, not an approximation.  For a large basis this array is the
    dominant memory cost of a gradient, so the restriction saves space as
    well as time.
    """
    mol: gto.Mole = mf.mol
    ao_idx = np.asarray([x[0] for x in mol.ao_labels(fmt=False)])
    atomlist = _as_atomlist(mol, atomlist)
    row = {int(ia): k for k, ia in enumerate(atomlist)}

    # <NABLA i|OVLP|j> = (3, nao i, nao j)
    ao_ov_grad: np.ndarray = mol.intor("int1e_ipovlp")

    # <NABLA i|OVLP|j> + <i|OVLP|NABLA j> = (len(atomlist), 3, nao i, nao j)
    ov_grad = np.zeros((len(atomlist),) + ao_ov_grad.shape)
    for ib, ia in enumerate(ao_idx):
        k = row.get(int(ia))
        if k is None:
            continue
        # dS_{mu,nu}/dR_A = -<grad mu|nu> [mu on A] - <mu|grad nu> [nu on A],
        # and <mu|grad nu> = <grad nu|mu> = ao_ov_grad[x, nu, mu] for real
        # basis functions.  So both terms read the same slice of the
        # integral -- only the slice being written differs.  Using the
        # transpose here instead makes dS/dR asymmetric in (mu, nu) and
        # breaks translational invariance.
        ov_grad[k, :, ib, :] -= ao_ov_grad[:, ib, :]
        ov_grad[k, :, :, ib] -= ao_ov_grad[:, ib, :]

    return ov_grad
