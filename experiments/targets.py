"""The information ladder: what each rung is allowed to know.

"Single-ended" is a claim about **information**, not about which function was
called.  A run that derives its half-bond targets from the converged
bond-order matrix of the product has been handed the product geometry, and no
amount of starting from the reactant changes that.  This module makes the
distinction explicit and mechanical, so every result in the suite can be
labelled with the rung it was produced at.

    L0  reference R and P *geometries*  ->  0.5 * (B_R + B_P), rounded to halves
        Exactly data/1-run.py.  NOT single-ended; the continuity baseline.

    L1  the chemical equation           ->  0.5 * (n_R + n_P) on changing pairs
        Integer Lewis orders and the atom mapping, nothing geometric.
        THIS IS THE PAPER'S CLAIM.

    L2  the reactant alone              ->  every enumerated move, half-targeted
        Product-agnostic: the move list comes from data/network.py's
        enumerator, which never sees a product.

The gap between L0 and L1 measures what the reference geometries were worth.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from ase import Atoms

from systems import Reaction

DATA = Path(__file__).parent.parent / "data"

Target = tuple[int, int, float]

RUNGS = ("L0", "L1", "L2")

RUNG_DESCRIPTION = {
    "L0": "reference R and P bond-order matrices (not single-ended)",
    "L1": "the chemical equation: integer Lewis orders and the atom mapping",
    "L2": "the reactant alone, plus enumerated bond rearrangements",
}


# --------------------------------------------------------------------------
# L0 -- the existing protocol
# --------------------------------------------------------------------------


def _connectivity(atoms: Atoms, thresh: float = 0.0) -> list[Target]:
    B = atoms.get_array("bond-order")
    out = []
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            if B[i, j] > thresh:
                out.append((i, j, float(B[i, j])))
    return out


def l0_targets(rxn: Reaction, to: str) -> list[Target]:
    """Reproduce data/1-run.py's target construction exactly.

    ``to`` is one of "ts", "r", "p".  TS targets are the R/P midpoint rounded
    to the nearest half and set-differenced against both endpoints; R and P
    targets are their own connectivity rounded to integers and set-differenced
    against the other endpoint.  Pairs absent from the returned list are
    unconstrained, which is what leaves spectators free.
    """
    Bi = rxn.reactant.get_array("bond-order")
    Bf = rxn.product.get_array("bond-order")
    Bm = 0.5 * (Bi + Bf)

    ts_conn = [
        (i, j, float(0.5 * np.round(2 * Bm[i, j])))
        for i in range(len(Bm))
        for j in range(i + 1, len(Bm))
        if Bm[i, j] > 0
    ]
    rs_conn = [(i, j, float(np.round(v))) for i, j, v in _connectivity(rxn.reactant)]
    ps_conn = [(i, j, float(np.round(v))) for i, j, v in _connectivity(rxn.product)]

    if to == "ts":
        return sorted(set(ts_conn) - set(rs_conn) - set(ps_conn))
    if to == "r":
        return sorted(set(rs_conn) - set(ps_conn))
    if to == "p":
        return sorted(set(ps_conn) - set(rs_conn))
    raise ValueError(f"unknown destination {to!r}")


# --------------------------------------------------------------------------
# L1 -- the chemical equation
# --------------------------------------------------------------------------


def l1_targets(rxn: Reaction, to: str, half: float = 0.5) -> list[Target]:
    """Targets a chemist could write down without any reference structure.

    Only the *changing* pairs are constrained; spectator bonds are left out of
    the list entirely, so they are masked to -1 in the calculator and free to
    relax.  For ``to="ts"`` each changing pair is targeted at the midpoint of
    its integer endpoints, which for the usual break-one/form-one case is the
    half bond of the method's slogan.

    ``half`` exists for E05, which asks whether 0.5 is actually special.  It
    rescales the interpolation between the two integer endpoints, so
    ``half=0.5`` is the midpoint and ``half=0.3`` sits nearer the reactant.
    """
    if to == "ts":
        out = []
        for i, j in rxn.changing_pairs():
            n_r = rxn.lewis_order("r", i, j)
            n_p = rxn.lewis_order("p", i, j)
            out.append((i, j, float(n_r + half * (n_p - n_r))))
        return out
    if to == "r":
        return [
            (i, j, float(rxn.lewis_order("r", i, j))) for i, j in rxn.changing_pairs()
        ]
    if to == "p":
        return [
            (i, j, float(rxn.lewis_order("p", i, j))) for i, j in rxn.changing_pairs()
        ]
    raise ValueError(f"unknown destination {to!r}")


def l1_asymmetric_targets(rxn: Reaction, tau: float) -> list[Target]:
    """E05's asymmetric sweep: breaking pairs at tau, forming pairs at 1 - tau.

    Expressed as a fraction of each pair's own integer span, so a 2 -> 0 pair
    (O2 dissociating) scales the same way a 1 -> 0 pair does.
    """
    out = []
    for i, j in rxn.changing_pairs():
        n_r = rxn.lewis_order("r", i, j)
        n_p = rxn.lewis_order("p", i, j)
        frac = tau if n_p < n_r else 1.0 - tau
        out.append((i, j, float(n_r + frac * (n_p - n_r))))
    return out


# --------------------------------------------------------------------------
# L2 -- product-agnostic move enumeration
# --------------------------------------------------------------------------


def _network_module():
    if str(DATA) not in sys.path:
        sys.path.insert(0, str(DATA))
    import network  # type: ignore[import-not-found]

    return network


def enumerate_l2_moves(atoms: Atoms, thresh: float = 0.5) -> list:
    """Every elementary rearrangement available to a structure.

    Delegates to data/network.py's enumerator, which is the same code that
    drives the network discovery in E07 -- so the L2 rung and claim B share
    their definition of "an elementary move" rather than having two.
    """
    return _network_module().enumerate_moves(atoms, thresh=thresh)


def l2_targets(atoms: Atoms, move, half: float = 0.5, thresh: float = 0.5) -> list[Target]:
    """Half-target one enumerated move, leaving every other bond as it is.

    `network.target_bonds` keeps the existing bonds at their rounded integer
    values and overrides the changed pairs; here the override is the midpoint
    between the current integer order and the move's requested order, which is
    the L2 analogue of asking for half bonds.
    """
    network = _network_module()
    current = {(i, j): v for i, j, v in network.bond_list(atoms, thresh=thresh)}
    changed = {}
    for i, j, v in move.changes:
        key = (min(i, j), max(i, j))
        now = current.get(key, 0.0)
        changed[key] = float(now + half * (v - now))

    out = [(i, j, float(v)) for (i, j), v in sorted(changed.items())]
    # Spectator bonds are pinned at their current integer order, matching
    # network.target_bonds; pairs in neither set stay unconstrained.
    for (i, j), v in sorted(current.items()):
        if (i, j) not in changed:
            out.append((i, j, float(v)))
    return sorted(out)


# --------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------


def targets_for(rxn: Reaction, rung: str, to: str = "ts", **kwargs) -> list[Target]:
    if rung == "L0":
        return l0_targets(rxn, to)
    if rung == "L1":
        return l1_targets(rxn, to, **kwargs)
    raise ValueError(
        f"rung {rung!r} has no single target list -- L2 produces one set per "
        "enumerated move; call enumerate_l2_moves/l2_targets"
    )


def bond_direction(
    atoms: Atoms, targets: Sequence[Target], reference: Sequence[Target] | None = None
) -> np.ndarray:
    """Mass-weighted 3N direction that enacts the requested bond changes.

    Forming pairs contribute a compression, breaking pairs an extension, each
    along the pair axis.  E02 compares this against the located structure's
    imaginary normal mode: if the two agree, *the mode you get is the mode you
    asked for*, which no energy or RMSD comparison can establish on its own.

    ``reference`` supplies the starting bond orders when the direction must be
    signed relative to something other than the target itself; when omitted,
    the sign comes from whether the target is above or below the current value.
    """
    B = atoms.get_array("bond-order")
    positions = atoms.get_positions()
    masses = atoms.get_masses()
    vector = np.zeros((len(atoms), 3))
    ref = {(i, j): v for i, j, v in (reference or [])}

    for i, j, target in targets:
        current = ref.get((i, j), float(B[i, j]))
        delta = target - current
        if abs(delta) < 1e-12:
            continue
        axis = positions[j] - positions[i]
        norm = np.linalg.norm(axis)
        if norm < 1e-12:
            continue
        axis = axis / norm
        # delta > 0 means the bond is forming, so the atoms move together.
        vector[i] += np.sign(delta) * axis
        vector[j] -= np.sign(delta) * axis

    weighted = vector * np.sqrt(masses)[:, None]
    norm = np.linalg.norm(weighted)
    return weighted / norm if norm > 1e-12 else weighted


def integer_disagreements(rxn: Reaction) -> list[dict]:
    """Where round(Mayer B) on the reference frames differs from Lewis.

    Reported by E01.  Each disagreement is a place where the L0 and L1 rungs
    genuinely ask for different chemistry, so a difference in their success
    rates on that reaction is explainable rather than mysterious.
    """
    out = []
    for side, frame in (("r", rxn.reactant), ("p", rxn.product)):
        B = frame.get_array("bond-order")
        distances = frame.get_all_distances()
        for i in range(len(frame)):
            for j in range(i + 1, len(frame)):
                rounded = int(np.round(B[i, j]))
                lewis = rxn.lewis_order(side, i, j)
                if rounded != lewis:
                    out.append({
                        "reaction": rxn.id,
                        "side": side,
                        "pair": [i, j],
                        "mayer": float(B[i, j]),
                        "rounded": rounded,
                        "lewis": lewis,
                        "distance_ang": float(distances[i, j]),
                    })
    return out
