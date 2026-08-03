"""Reaction-network discovery in bond space.

The idea: a reaction is a change in the bond-order matrix, and
``BondFluxCalculator`` can drive an arbitrary such change directly.  So
instead of searching for transition states, enumerate the elementary bond
rearrangements available to a structure (break / form / transfer), drive
each one, and see which species come out.  Repeating this on every newly
discovered species grows a reaction network.

Atoms are conserved along every edge, so the total electron count -- and
hence the spin multiplicity -- is fixed for an entire run.  Every state is
computed at the same ``spin``; species whose true ground state has a
different multiplicity are only approximately described.  This is the main
physical limitation of the approach as implemented here.
"""

from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.optimize import FIRE2
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from bondspace.ase import BondFluxCalculator, PySCFCalculator

# A pair counts as bonded above this Mayer bond order.  0.5 sits in the gap
# between the ~0.02 residual overlap of non-bonded pairs and the ~0.9 of a
# genuine single bond (see the connectivity fields in HCombustion/*.xyz).
BOND_THRESH = 0.5

# Maximum number of bonding partners, used to reject nonsense moves such as
# a five-coordinate hydrogen.  This caps *degree*, not bond order, so O2
# (degree 1, order 2) is fine.
MAX_DEGREE = {"H": 1, "O": 2}


def bond_matrix(atoms: Atoms) -> np.ndarray:
    """Off-diagonal Mayer bond orders; the large atomic diagonal is zeroed."""
    B = np.array(atoms.get_array("bond-order"), dtype=float)
    np.fill_diagonal(B, 0.0)
    return B


def fragments(atoms: Atoms, thresh: float = BOND_THRESH) -> list[np.ndarray]:
    """Connected components of the thresholded bond-order graph."""
    adj = bond_matrix(atoms) > thresh
    n, labels = connected_components(csr_matrix(adj), directed=False)
    return [np.where(labels == k)[0] for k in range(n)]


def species_label(atoms: Atoms, thresh: float = BOND_THRESH) -> str:
    """Canonical name for a state, e.g. ``"H2 + O2"``.

    Sorted so that the same set of fragments always produces the same
    string regardless of atom ordering -- this is the network's node key.
    """
    parts = sorted(
        Atoms(numbers=atoms.get_atomic_numbers()[idx]).get_chemical_formula()
        for idx in fragments(atoms, thresh)
    )
    return " + ".join(parts)


def bond_list(atoms: Atoms, thresh: float = BOND_THRESH) -> list[tuple[int, int, float]]:
    """Current bonds, with orders rounded to the nearest integer."""
    B = bond_matrix(atoms)
    return [
        (int(i), int(j), float(np.round(B[i, j])))
        for i, j in zip(*np.triu_indices(len(atoms), 1))
        if B[i, j] > thresh
    ]


@dataclass(frozen=True)
class Move:
    """An elementary bond rearrangement: pairs and the order they target."""

    kind: str  # "break" | "form" | "transfer"
    changes: tuple[tuple[int, int, float], ...]

    def describe(self, atoms: Atoms) -> str:
        sym = atoms.get_chemical_symbols()
        parts = [f"{sym[i]}{i}-{sym[j]}{j}={v:g}" for i, j, v in self.changes]
        return f"{self.kind}[{','.join(parts)}]"


def _change(i: int, j: int, v: float) -> tuple[int, int, float]:
    """Normalise a pair to i < j so moves dedup correctly."""
    return (min(i, j), max(i, j), v)


def enumerate_moves(atoms: Atoms, thresh: float = BOND_THRESH) -> list[Move]:
    """Elementary rearrangements available to this structure.

    Three families, which between them cover the elementary steps of
    hydrogen combustion:

    - ``break``    dissociation, e.g. H2O2 -> OH + OH
    - ``form``     association, e.g. H + O2 -> HO2
    - ``transfer`` a fragment migrates between heavy atoms in one concerted
      step, e.g. H + O2 -> OH + O.  These are the abstraction reactions
      that carry the chain, and they are the reason a two-change move set
      is needed rather than break-then-form.
    """
    n = len(atoms)
    adj = bond_matrix(atoms) > thresh
    deg = adj.sum(1)
    cap = np.array([MAX_DEGREE.get(s, 4) for s in atoms.get_chemical_symbols()])
    bonds = [(int(i), int(j)) for i, j in zip(*np.triu_indices(n, 1)) if adj[i, j]]

    moves: set[Move] = set()

    for i, j in bonds:
        moves.add(Move("break", (_change(i, j, 0.0),)))

    for i, j in zip(*np.triu_indices(n, 1)):
        if not adj[i, j] and deg[i] < cap[i] and deg[j] < cap[j]:
            moves.add(Move("form", (_change(int(i), int(j), 1.0),)))

    # `mig` leaves `src` and attaches to `dst`; leaving frees a slot on
    # `mig` itself, so only the acceptor's capacity has to be checked.
    for a, b in bonds:
        for src, mig in ((a, b), (b, a)):
            for dst in range(n):
                if dst in (src, mig) or adj[mig, dst] or deg[dst] >= cap[dst]:
                    continue
                moves.add(
                    Move("transfer", (_change(src, mig, 0.0), _change(mig, dst, 1.0)))
                )

    return sorted(moves, key=lambda m: (m.kind, m.changes))


def target_bonds(
    atoms: Atoms, move: Move, thresh: float = BOND_THRESH
) -> list[tuple[int, int, float]]:
    """Constraint list for a move: keep existing bonds, override the changed ones.

    Pairs left out are unconstrained, which is deliberate -- fragments that
    are supposed to separate must be free to drift apart.
    """
    targets = {(i, j): v for i, j, v in bond_list(atoms, thresh)}
    for i, j, v in move.changes:
        targets[(i, j)] = v
    return [(i, j, v) for (i, j), v in sorted(targets.items())]


def preorient(atoms: Atoms, move: Move, sep: float = 2.2, thresh: float = BOND_THRESH) -> Atoms:
    """Bring fragments into bonding range before an association move.

    Bond order decays to zero between distant fragments, taking its
    gradient with it, so a formation constraint on two far-apart fragments
    produces no force at all.  The overlap correction only applies to bonds
    being broken, so it does not help here.  Translating the acceptor
    fragment to ``sep`` gives the constraint something to pull on.
    """
    atoms = atoms.copy()
    frags = fragments(atoms, thresh)
    owner = {int(a): k for k, idx in enumerate(frags) for a in idx}
    pos = atoms.get_positions()

    for i, j, v in move.changes:
        if v <= 0 or owner[i] == owner[j]:
            continue
        vec = pos[j] - pos[i]
        dist = float(np.linalg.norm(vec))
        if dist <= sep:
            continue
        # Slide j's fragment along the i->j axis until the pair sits at `sep`.
        shift = vec / dist * (sep - dist)
        moving = frags[owner[j]]
        pos[moving] += shift
        atoms.set_positions(pos)

    return atoms


def check_spin(atoms: Atoms, charge: int, spin: int) -> None:
    """Fail early if spin and electron count disagree in parity.

    Composition is fixed for a whole run, so this is checkable once up
    front.  Left to PySCF it surfaces as a RuntimeError from deep inside
    ``gto.M`` on the first drive, long after the search has started.
    """
    n_elec = int(atoms.get_atomic_numbers().sum()) - charge
    if (n_elec - spin) % 2:
        raise ValueError(
            f"{atoms.get_chemical_formula()} with charge {charge} has {n_elec} "
            f"electrons, which is inconsistent with spin={spin}. "
            f"Use an {'even' if n_elec % 2 == 0 else 'odd'} spin "
            "(mol.spin = 2S = Nalpha - Nbeta, not 2S+1)."
        )


def relax(
    atoms: Atoms,
    *,
    charge: int,
    spin: int,
    basis: str,
    level_shift: tuple[float, float],
    fmax: float = 0.05,
    steps: int = 30,
) -> Atoms:
    """Relax on the real PES and attach bond orders plus the energy.

    A bond-flux endpoint satisfies its bond-order target but is not a
    stationary structure, so states are relaxed before being used as the
    starting point for further moves.  This also supplies the energies that
    make the resulting network quantitative.
    """
    atoms = atoms.copy()
    atoms.calc = PySCFCalculator(
        charge=charge, spin=spin, basis=basis, level_shift=level_shift
    )
    opt = FIRE2(atoms)
    converged = opt.run(fmax=fmax, steps=steps)

    energy = float(atoms.get_potential_energy())
    atoms = atoms.copy()  # drop the calculator, keep arrays
    atoms.info["energy"] = energy
    atoms.info["relaxed"] = bool(converged)
    return atoms


def drive(
    atoms: Atoms,
    move: Move,
    *,
    charge: int,
    spin: int,
    basis: str,
    level_shift: tuple[float, float],
    thresh: float,
    ovlp_thresh: float,
    fmax: float = 0.1,
    steps: int = 50,
) -> tuple[list[Atoms], bool]:
    """Drive one move in bond space; return the trajectory and convergence."""
    atoms = preorient(atoms, move, thresh=thresh)
    atoms.calc = BondFluxCalculator(
        target_bonds(atoms, move, thresh),
        charge=charge,
        spin=spin,
        basis=basis,
        thresh=thresh,
        ovlp_thresh=ovlp_thresh,
        level_shift=level_shift,
    )

    images: list[Atoms] = []
    opt = FIRE2(atoms)
    opt.attach(lambda: images.append(atoms.copy()), 1)
    converged = bool(opt.run(fmax=fmax, steps=steps))

    return images, converged
