"""Metrics.  Everything here is computed *outside* bond space.

The existing analysis scores a bond-order objective with a bond-order metric,
over reference-bonded pairs only.  That is close to circular, and it cannot
see a spurious bond.  The metrics below are the ones that can actually falsify
the method: geometric distance to a verified saddle, the true PES gradient at
the located structure, the imaginary-mode character, and the barrier.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from ase import Atoms, units

#: kcal/mol per eV.
KCAL = units.mol / units.kcal


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


def kabsch_rmsd(p: np.ndarray, q: np.ndarray) -> float:
    """RMSD after optimal translation and rotation, fixed correspondence."""
    p = p - p.mean(axis=0)
    q = q - q.mean(axis=0)
    u, _, vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag([1.0, 1.0, d]) @ vt
    aligned = p @ rotation
    return float(np.sqrt(np.mean(np.sum((aligned - q) ** 2, axis=1))))


def permutation_rmsd(
    a: Atoms,
    b: Atoms,
    *,
    heavy_only: bool = False,
    max_permutations: int = 20000,
) -> float:
    """RMSD minimised over permutations of chemically equivalent atoms.

    **This is not optional.**  These systems have two to four equivalent
    hydrogens, and a bond-space drive is under no obligation to move "the"
    hydrogen the reference happened to label as transferring.  Index-matched
    RMSD then reports multi-Angstrom errors for structures that are physically
    identical, which would silently inflate every geometric number in the
    paper.  Permuting within element groups is exact and, at six atoms,
    trivially cheap.

    ``heavy_only`` drops hydrogens from the comparison entirely -- the usual
    convention for reporting TS accuracy, since heavy-atom positions carry the
    reaction coordinate.  It falls back to all atoms when fewer than **two**
    heavy atoms survive: Kabsch alignment can superimpose a single point on
    any other exactly, so a one-heavy-atom system (OH + H2, H2O + H, and two
    others here) would report a heavy-atom RMSD of exactly zero for every
    structure ever compared, and T2 would pass for nothing.  Silently
    reporting 0.000 is worse than reporting an all-atom number.
    """
    symbols = np.array(a.get_chemical_symbols())
    if list(symbols) != list(b.get_chemical_symbols()):
        raise ValueError("permutation_rmsd needs the same composition and order")

    mask = np.ones(len(a), dtype=bool)
    if heavy_only:
        heavy = symbols != "H"
        if heavy.sum() >= 2:
            mask = heavy

    pa = a.get_positions()[mask]
    pb = b.get_positions()[mask]
    groups = symbols[mask]

    # Permutations are generated per element group and combined; the product
    # over groups is the full search space.
    index_by_element: dict[str, list[int]] = {}
    for idx, element in enumerate(groups):
        index_by_element.setdefault(element, []).append(idx)

    total = 1
    for indices in index_by_element.values():
        total *= _fact(len(indices))
    if total > max_permutations:
        # Falls back rather than hanging; recorded by the caller if it matters.
        return kabsch_rmsd(pa, pb)

    element_perms = {
        element: list(itertools.permutations(indices))
        for element, indices in index_by_element.items()
    }
    elements = sorted(element_perms)

    best = np.inf
    for combo in itertools.product(*(element_perms[e] for e in elements)):
        order = np.empty(len(groups), dtype=int)
        for element, perm in zip(elements, combo):
            order[index_by_element[element]] = perm
        best = min(best, kabsch_rmsd(pa[order], pb))
    return float(best)


def _fact(n: int) -> int:
    out = 1
    for k in range(2, n + 1):
        out *= k
    return out


# --------------------------------------------------------------------------
# bond space (diagnostic, no longer headline)
# --------------------------------------------------------------------------


def bond_errors(atoms: Atoms, reference: Atoms) -> dict:
    """Bond-order deviation, reported two ways.

    ``max_ref_pairs`` is the existing metric from data/analysis.ipynb, kept for
    continuity.  ``max_all_pairs`` additionally sees bonds the structure
    invented that the reference does not have -- which the existing metric
    cannot, because it iterates the reference connectivity.
    """
    B = atoms.get_array("bond-order")
    Bref = reference.get_array("bond-order")
    n = len(atoms)
    iu = np.triu_indices(n, 1)
    all_err = np.abs(B[iu] - Bref[iu])

    ref_mask = Bref[iu] > 1e-8
    ref_err = all_err[ref_mask] if ref_mask.any() else np.zeros(1)

    return {
        "max_ref_pairs": float(ref_err.max()),
        "max_all_pairs": float(all_err.max()),
        "rms_all_pairs": float(np.sqrt(np.mean(all_err**2))),
    }


def target_error(atoms: Atoms, targets: Sequence[tuple[int, int, float]]) -> float:
    """Largest deviation from the requested targets, in bond-order units."""
    B = atoms.get_array("bond-order")
    if not targets:
        return 0.0
    return float(max(abs(B[i, j] - v) for i, j, v in targets))


# --------------------------------------------------------------------------
# energies
# --------------------------------------------------------------------------


def barrier_kcal(ts_energy: float, r_energy: float) -> float:
    """Forward barrier in kcal/mol from ASE energies in eV."""
    return float((ts_energy - r_energy) * KCAL)


def energy_gap_kcal(a: float, b: float) -> float:
    return float(abs(a - b) * KCAL)


def max_force(forces: np.ndarray) -> float:
    """max |F| per atom, in whatever units the forces came in (eV/Ang here)."""
    return float(np.sqrt((np.asarray(forces) ** 2).sum(axis=1)).max())


# --------------------------------------------------------------------------
# normal modes
# --------------------------------------------------------------------------


def mode_overlap(mode: np.ndarray, direction: np.ndarray) -> float:
    """|cos| between a normal mode and a 3N direction, both mass-weighted.

    E02 uses this against the constrained-bond direction: a high overlap says
    the saddle reached is the one that was asked for, which matching energies
    alone cannot establish -- two different saddles of the same reaction can
    sit at similar energies.
    """
    a = np.asarray(mode, dtype=float).ravel()
    b = np.asarray(direction, dtype=float).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(abs(a @ b) / (na * nb))


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------


def _pairwise_rmsd(path: Sequence[Atoms], reference: Sequence[Atoms],
                   heavy_only: bool = True) -> np.ndarray:
    return np.array(
        [[permutation_rmsd(a, b, heavy_only=heavy_only) for b in reference]
         for a in path]
    )


def tube_distance(path: Sequence[Atoms], reference: Sequence[Atoms]) -> dict:
    """How far a drive strays from the reference IRC.

    For each frame, the distance to the nearest reference frame.  The maximum
    is the honest headline: a path can have a small mean while excursioning
    somewhere chemically absurd.
    """
    if not path or not reference:
        return {"mean": float("nan"), "max": float("nan")}
    d = _pairwise_rmsd(path, reference)
    nearest = d.min(axis=1)
    return {"mean": float(nearest.mean()), "max": float(nearest.max())}


def progress_monotonicity(path: Sequence[Atoms], reference: Sequence[Atoms]) -> dict:
    """Whether the drive advances along the reference coordinate or backtracks.

    Each frame is assigned the index of its nearest reference frame; Spearman
    correlation of that index against step number measures monotone progress,
    and the backtracking count says how often it went the wrong way.
    """
    if len(path) < 3 or not reference:
        return {"spearman": float("nan"), "backtracks": -1}
    d = _pairwise_rmsd(path, reference)
    nearest = d.argmin(axis=1)
    steps = np.arange(len(nearest))

    from scipy.stats import spearmanr

    rho = spearmanr(steps, nearest).statistic
    backtracks = int((np.diff(nearest) < 0).sum())
    return {"spearman": float(rho), "backtracks": backtracks}


def barrier_recovery(path_energies: Sequence[float], reactant_energy: float,
                     reference_barrier_kcal: float) -> dict:
    """How high the drive climbed, against how high it needed to.

    A path that overshoots the true barrier by tens of kcal/mol is not a
    reaction path however good its endpoints are, and this is the number that
    says so.
    """
    if not len(path_energies):
        return {"peak_kcal": float("nan"), "overshoot_kcal": float("nan")}
    peak = (max(path_energies) - reactant_energy) * KCAL
    return {
        "peak_kcal": float(peak),
        "overshoot_kcal": float(peak - reference_barrier_kcal),
    }


# --------------------------------------------------------------------------
# the success ladder
# --------------------------------------------------------------------------


@dataclass
class Tiers:
    """E02's five-tier ladder, from the existing metric to a verified saddle.

    Each tier is strictly harder than the one before, so the attrition down
    the ladder is itself the result: it shows exactly how much of the reported
    success survives leaving bond space.
    """

    t0_ref_bond_error: bool  # the existing metric
    t1_all_bond_error: bool  # ... plus no spurious bonds
    t2_rmsd: bool  # ... plus geometrically close to the verified TS
    t3_one_imaginary: bool  # ... plus actually a first-order saddle
    t4_verified: bool  # ... plus refines and its IRC connects R and P

    def as_dict(self) -> dict:
        return {
            "T0": self.t0_ref_bond_error,
            "T1": self.t1_all_bond_error,
            "T2": self.t2_rmsd,
            "T3": self.t3_one_imaginary,
            "T4": self.t4_verified,
        }

    @property
    def highest(self) -> int:
        for level, passed in enumerate(
            [self.t0_ref_bond_error, self.t1_all_bond_error, self.t2_rmsd,
             self.t3_one_imaginary, self.t4_verified]
        ):
            if not passed:
                return level - 1
        return 4

    @property
    def label(self) -> str:
        """The highest tier as a name.  ``-1`` printed as ``T-1`` reads as a
        tier rather than as failing the first one."""
        highest = self.highest
        return f"T{highest}" if highest >= 0 else "none"


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval -- reported instead of point estimates.

    With n = 19 the normal approximation is not usable, and a bare "12/19"
    invites comparisons between methods whose intervals overlap completely.
    """
    if trials == 0:
        return (float("nan"), float("nan"))
    p = successes / trials
    denom = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denom
    spread = z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denom
    return (float(max(0.0, centre - spread)), float(min(1.0, centre + spread)))
