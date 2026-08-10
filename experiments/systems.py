"""The benchmark registry, and the hand-audited chemistry behind it.

Adding a benchmark set is one loader function plus one entry in `BENCHMARKS`;
no experiment script changes.  Today the only set is HCombustion.

The reason this module holds a hand-written table rather than deriving
everything from the reference files: the **L1 rung of the information ladder
must not read a reference geometry**, not even indirectly through
`round(Mayer B)`.  Rounding the reference matrices would usually give the same
integers, but not always, and the exceptions are chemically important:

  * rxn_01's reactant has O0-O1 at **2.05 Ang** -- unbound -- yet a Mayer bond
    order of 1.09, because the fixed high-spin reference smears density
    between the fragments.  Rounding says "there is a bond"; chemistry says
    there is not.
  * The six spin-nonconserving reactions dissociate to 3-4 Ang while their
    Mayer orders stay near the bonded value (rxn_05: H-H = 1.00 at 3.0 Ang;
    rxn_06: O-O = 2.00 at 4.0 Ang).  A spin-restricted single determinant
    cannot dissociate, which is a documented caveat of this method, and it
    means the L0 targets for those reactions ask for no change at all.

So `lewis_r` / `lewis_p` below are what a chemist writes from the reaction
equation, and E01 reports every place they disagree with `round(Mayer B)`.
That disagreement table is a result, not bookkeeping.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Sequence

from ase import Atoms, io

HERE = Path(__file__).parent
DATA = HERE.parent / "data"

Bond = tuple[int, int, int]  # (i, j, integer order)


@dataclass(frozen=True)
class Reaction:
    """One benchmark reaction, with everything each rung is allowed to see."""

    id: str
    charge: int
    spin: int
    equation: str
    category: str  # sub | oxygen | hydrogen | ad, from data/analysis.ipynb
    spin_changes: bool  # a single fixed `spin` cannot describe both endpoints
    lewis_r: tuple[Bond, ...]
    lewis_p: tuple[Bond, ...]
    reactant: Atoms = field(repr=False, default=None)  # type: ignore[assignment]
    ts: Atoms = field(repr=False, default=None)  # type: ignore[assignment]
    product: Atoms = field(repr=False, default=None)  # type: ignore[assignment]

    @property
    def natoms(self) -> int:
        return len(self.reactant)

    def changing_pairs(self) -> list[tuple[int, int]]:
        """Pairs whose integer bond order differs between R and P.

        The chemist-level statement of the reaction, and the only structural
        input the L1 rung receives.
        """
        r = {(i, j): v for i, j, v in self.lewis_r}
        p = {(i, j): v for i, j, v in self.lewis_p}
        pairs = sorted(set(r) | set(p))
        return [ij for ij in pairs if r.get(ij, 0) != p.get(ij, 0)]

    def lewis_order(self, side: str, i: int, j: int) -> int:
        table = self.lewis_r if side == "r" else self.lewis_p
        key = (min(i, j), max(i, j))
        for a, b, v in table:
            if (a, b) == key:
                return v
        return 0

    def frame(self, which: str) -> Atoms:
        return {"r": self.reactant, "ts": self.ts, "p": self.product}[which]


# --------------------------------------------------------------------------
# HCombustion
# --------------------------------------------------------------------------

# Assigned by reading the reference geometries *and* their distances -- see the
# module docstring for why the bond orders alone are not enough.  Bond indices
# refer to the atom ordering in data/HCombustion/<id>.xyz.
_HCOMBUSTION: dict[str, dict] = {
    "rxn_01": dict(
        equation="O + OH -> O2 + H",
        lewis_r=((1, 2, 1),),
        lewis_p=((0, 1, 2),),
    ),
    "rxn_02": dict(
        equation="O + H2 -> OH + H",
        lewis_r=((0, 2, 1),),
        lewis_p=((0, 1, 1),),
    ),
    "rxn_03": dict(
        equation="OH + H2 -> H2O + H",
        lewis_r=((0, 1, 1), (2, 3, 1)),
        lewis_p=((1, 3, 1), (2, 3, 1)),
    ),
    "rxn_04": dict(
        equation="H2O2 -> H2O + O",
        lewis_r=((0, 1, 1), (1, 3, 1), (2, 3, 1)),
        lewis_p=((0, 1, 1), (1, 2, 1)),
    ),
    "rxn_05": dict(
        equation="H2 -> H + H",
        lewis_r=((0, 1, 1),),
        lewis_p=(),
    ),
    "rxn_06": dict(
        equation="O2 -> O + O",
        lewis_r=((0, 1, 2),),
        lewis_p=(),
    ),
    "rxn_07": dict(
        equation="OH -> O + H",
        lewis_r=((0, 1, 1),),
        lewis_p=(),
    ),
    "rxn_08": dict(
        equation="H2O -> OH + H",
        lewis_r=((0, 1, 1), (0, 2, 1)),
        lewis_p=((0, 2, 1),),
    ),
    "rxn_09": dict(
        equation="HO2 -> H + O2",
        lewis_r=((0, 1, 1), (0, 2, 1)),
        lewis_p=((0, 1, 2),),
    ),
    "rxn_10": dict(
        equation="HO2 + H -> O2 + H2",
        lewis_r=((0, 1, 1), (1, 2, 1)),
        lewis_p=((0, 1, 2), (2, 3, 1)),
    ),
    "rxn_11": dict(
        equation="H + HO2 -> OH + OH",
        lewis_r=((0, 1, 1), (1, 2, 1)),
        lewis_p=((0, 1, 1), (2, 3, 1)),
    ),
    "rxn_12": dict(
        equation="HO2 + O -> O2 + OH",
        lewis_r=((0, 1, 1), (1, 2, 1)),
        lewis_p=((0, 1, 2), (2, 3, 1)),
    ),
    "rxn_13": dict(
        equation="O2 + H2O -> HO2 + OH",
        lewis_r=((0, 1, 2), (2, 4, 1), (3, 4, 1)),
        lewis_p=((0, 1, 1), (0, 2, 1), (3, 4, 1)),
    ),
    "rxn_14": dict(
        equation="H2O2 + O2 -> HO2 + HO2",
        lewis_r=((0, 1, 1), (1, 2, 1), (2, 3, 1), (4, 5, 2)),
        lewis_p=((0, 1, 1), (1, 2, 1), (3, 4, 1), (4, 5, 1)),
    ),
    "rxn_15": dict(
        equation="H2O2 -> OH + OH",
        lewis_r=((0, 1, 1), (0, 3, 1), (2, 3, 1)),
        lewis_p=((0, 1, 1), (2, 3, 1)),
    ),
    "rxn_16": dict(
        equation="OH + H2O -> H2O2 + H",
        lewis_r=((0, 1, 1), (2, 3, 1), (2, 4, 1)),
        lewis_p=((0, 1, 1), (1, 2, 1), (2, 3, 1)),
    ),
    "rxn_17": dict(
        equation="HO2 + H2 -> H2O2 + H",
        lewis_r=((0, 1, 1), (0, 2, 1), (3, 4, 1)),
        lewis_p=((0, 1, 1), (0, 2, 1), (1, 3, 1)),
    ),
    "rxn_18": dict(
        equation="HO2 + OH -> H2O2 + O",
        lewis_r=((0, 1, 1), (0, 2, 1), (3, 4, 1)),
        lewis_p=((0, 1, 1), (0, 2, 1), (1, 3, 1)),
    ),
    "rxn_19": dict(
        equation="H2O2 + OH -> HO2 + H2O",
        lewis_r=((0, 1, 1), (0, 3, 1), (1, 2, 1), (4, 5, 1)),
        lewis_p=((0, 1, 1), (1, 2, 1), (3, 4, 1), (4, 5, 1)),
    ),
}

#: From data/analysis.ipynb cell 7, copied verbatim so the taxonomy used in
#: the figures is the same one the existing analysis used.
CATEGORY = {
    "rxn_01": "oxygen", "rxn_02": "hydrogen", "rxn_03": "hydrogen",
    "rxn_04": "hydrogen", "rxn_05": "ad", "rxn_06": "ad", "rxn_07": "ad",
    "rxn_08": "ad", "rxn_09": "ad", "rxn_10": "hydrogen", "rxn_11": "oxygen",
    "rxn_12": "oxygen", "rxn_13": "hydrogen", "rxn_14": "hydrogen",
    "rxn_15": "ad", "rxn_16": "sub", "rxn_17": "hydrogen",
    "rxn_18": "hydrogen", "rxn_19": "hydrogen",
}

#: **Pre-registered before any run.**  These reactions' endpoints have
#: different ground-state multiplicities, per the inline comments in
#: data/util.py, so a single fixed `spin` cannot describe both.  Declaring the
#: subset up front is what stops the exclusion from looking post-hoc when the
#: results come in; E10 class 6 makes the consequence quantitative.
SPIN_NONCONSERVING = frozenset(
    {"rxn_05", "rxn_06", "rxn_07", "rxn_08", "rxn_09", "rxn_15"}
)


def _legacy_rxn_data() -> dict:
    """data/util.py's charge/spin table, imported for cross-checking."""
    if str(DATA) not in sys.path:
        sys.path.insert(0, str(DATA))
    from util import rxn_data  # type: ignore[import-not-found]

    return rxn_data


def _load_hcombustion() -> list[Reaction]:
    source = DATA / "HCombustion-bso"
    if not source.exists():
        raise FileNotFoundError(
            f"{source} not found -- run `uv run python data/0-prep.py` first"
        )
    legacy = _legacy_rxn_data()
    reactions = []
    for rid, entry in sorted(_HCOMBUSTION.items()):
        frames = io.read(source / f"{rid}.xyz", index=":")
        assert isinstance(frames, list) and len(frames) == 3, rid
        rs, ts, ps = frames
        reactions.append(
            Reaction(
                id=rid,
                charge=legacy[rid]["charge"],
                spin=legacy[rid]["spin"],
                equation=entry["equation"],
                category=CATEGORY[rid],
                spin_changes=rid in SPIN_NONCONSERVING,
                lewis_r=tuple(entry["lewis_r"]),
                lewis_p=tuple(entry["lewis_p"]),
                reactant=rs,
                ts=ts,
                product=ps,
            )
        )
    return reactions


BENCHMARKS: dict[str, Callable[[], list[Reaction]]] = {
    "hcombustion": _load_hcombustion,
}

DEFAULT_BENCHMARK = "hcombustion"


@lru_cache(maxsize=None)
def load_set(name: str = DEFAULT_BENCHMARK) -> tuple[Reaction, ...]:
    if name not in BENCHMARKS:
        raise KeyError(f"unknown benchmark {name!r}; have {sorted(BENCHMARKS)}")
    reactions = tuple(BENCHMARKS[name]())
    _check_consistency(reactions)
    return reactions


def by_id(name: str = DEFAULT_BENCHMARK) -> dict[str, Reaction]:
    return {r.id: r for r in load_set(name)}


def _check_consistency(reactions: Sequence[Reaction]) -> None:
    """Fail loudly at import if this table and data/util.py have drifted.

    `rxn_data` has already been copied once into data/profiles.ipynb, where a
    comment drifted.  A duplicated table with no cross-check is how the next
    drift goes unnoticed until it is in a figure.
    """
    legacy = _legacy_rxn_data()
    for rxn in reactions:
        want = legacy[rxn.id]
        if (rxn.charge, rxn.spin) != (want["charge"], want["spin"]):
            raise AssertionError(
                f"{rxn.id}: charge/spin disagrees with data/util.py "
                f"({rxn.charge},{rxn.spin}) vs ({want['charge']},{want['spin']})"
            )
        for side, table in (("r", rxn.lewis_r), ("p", rxn.lewis_p)):
            for i, j, order in table:
                if not (0 <= i < len(rxn.reactant) and 0 <= j < len(rxn.reactant)):
                    raise AssertionError(f"{rxn.id}: bad index in lewis_{side}")
                if i >= j:
                    raise AssertionError(
                        f"{rxn.id}: lewis_{side} pairs must be (i<j), got ({i},{j})"
                    )
                if order < 1:
                    raise AssertionError(
                        f"{rxn.id}: lewis_{side} lists only bonds that exist; "
                        f"({i},{j}) has order {order}"
                    )
        if not rxn.changing_pairs():
            raise AssertionError(
                f"{rxn.id}: no integer bond change between R and P -- the "
                "reaction equation is degenerate and L1 would have no targets"
            )


#: Fixed subset for the ablations (E08) and the target sweep (E05), chosen to
#: span the four categories with two reactions each.  Pre-registered so that
#: the subset cannot be picked after seeing which reactions do well.
ABLATION_SUBSET = (
    "rxn_03", "rxn_10",   # hydrogen transfer
    "rxn_11", "rxn_12",   # oxygen transfer
    "rxn_04", "rxn_16",   # substitution-like
    "rxn_09", "rxn_15",   # association/dissociation (both spin-nonconserving)
)
