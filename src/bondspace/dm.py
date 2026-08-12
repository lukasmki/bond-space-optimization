import numpy as np
from pyscf import scf, gto


def bo_dm_deriv(mf: scf.hf.SCF, ov=None, dm=None):
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

    # ----------------------------------------------------------------
    # dB_AB / dP^sigma_{lambda, kappa}
    #
    # From the derivation, two cases contribute:
    #   lambda in A:  sum_{nu in B} [ S_{kappa,nu} (DS)_{nu,lambda}
    #                               + s * S_{kappa,nu} (RS)_{nu,lambda} ]
    #   lambda in B:  sum_{mu in A} [ S_{kappa,mu} (DS)_{mu,lambda}
    #                               + s * S_{kappa,mu} (RS)_{mu,lambda} ]
    # where s = +1 for alpha, -1 for beta.
    #
    # Rewriting in matrix form:
    #   (S^T @ DS)_{kappa, lambda} = sum_nu S_{kappa,nu} DS_{nu,lambda}
    #   (S^T @ RS)_{kappa, lambda} = sum_nu S_{kappa,nu} RS_{nu,lambda}
    #
    # So define:
    #   SDS = S^T @ DS   (nao, nao)  [kappa, lambda]
    #   SRS = S^T @ RS   (nao, nao)  [kappa, lambda]
    #
    # Then for a given bond (A, B) and spin sigma:
    #   dB_AB/dP^sigma_{lambda,kappa} =
    #       [lambda in A] * sum_{nu in B} (SDS + s*SRS)_{kappa, nu}  <- Case 1
    #     + [lambda in B] * sum_{mu in A} (SDS + s*SRS)_{kappa, mu}  <- Case 2
    # ----------------------------------------------------------------

    SDS = ov.T @ DS  # (nao, nao): S_{kappa,nu} DS_{nu,lambda} summed over nu
    SRS = ov.T @ RS  # (nao, nao): S_{kappa,nu} RS_{nu,lambda} summed over nu

    # Atom-blocked sums: for each atom X, sum SDS/SRS columns over AOs on X
    # atom_SDS[X, kappa] = sum_{lambda in X} SDS_{kappa, lambda}  -- not needed
    # We need row-sums over atoms for Case 1 and Case 2:
    #
    # Case 1 contribution to dB_{AB}/dP^sigma_{lambda in A, kappa}:
    #   = sum_{nu in B} (SDS + s*SRS)_{kappa, nu}
    #   = atom_col_sum(SDS + s*SRS)_{kappa, B}
    #
    # Case 2 contribution to dB_{AB}/dP^sigma_{lambda in B, kappa}:
    #   = sum_{mu in A} (SDS + s*SRS)_{kappa, mu}
    #   = atom_col_sum(SDS + s*SRS)_{kappa, A}

    # atom_col_SDS[kappa, X] = sum_{lambda in X} SDS_{kappa, lambda}
    atom_col_SDS = np.zeros((mol.nao, mol.natm))
    atom_col_SRS = np.zeros((mol.nao, mol.natm))
    np.add.at(atom_col_SDS, (slice(None), ao_idx), SDS)  # (nao, natm)
    np.add.at(atom_col_SRS, (slice(None), ao_idx), SRS)  # (nao, natm)

    # Result arrays: B_dma/dmb[A, B, lambda, kappa]
    # = dB_{AB} / dP^{alpha,beta}_{lambda, kappa}
    B_dma = np.zeros((mol.natm, mol.natm, mol.nao, mol.nao))
    B_dmb = np.zeros((mol.natm, mol.natm, mol.nao, mol.nao))

    for s, (B_dm, sign) in enumerate([(B_dma, 1.0), (B_dmb, -1.0)]):
        # combined intermediate for this spin: (nao, natm)
        # col X = sum_{lambda in X} (SDS +/- SRS)_{kappa, lambda}
        M = atom_col_SDS + sign * atom_col_SRS  # (nao, natm)

        # Case 1: lambda in A
        # dB_{AB}/dP^sigma_{lambda, kappa} += M_{kappa, B}
        # for all lambda in A, all A, all B
        # B_dm[A, B, lambda, kappa] += M[kappa, B]  for lambda in A
        for A in range(mol.natm):
            lam_mask = ao_idx == A  # AOs on atom A
            # B_dm[A, B, lambda, kappa] += M[kappa, B]
            # shape: (natm, nao_A, nao) += (natm, nao)[None broadcast]
            B_dm[A, :, lam_mask, :] += M.T[
                None, :, :
            ]  # (1, natm, nao) broadcast over lam

        # Case 2: lambda in B
        # dB_{AB}/dP^sigma_{lambda, kappa} += M_{kappa, A}
        # for all lambda in B, all A, all B
        # B_dm[A, B, lambda, kappa] += M[kappa, A]  for lambda in B
        for B in range(mol.natm):
            lam_mask = ao_idx == B
            # B_dm[A, B, lambda, kappa] += M[kappa, A]
            B_dm[:, B, lam_mask, :] += M.T[
                :, None, :
            ]  # (natm, 1, nao) broadcast over lam

    return (B_dma, B_dmb)
