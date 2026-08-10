"""Alternative ways to get a transition-state guess, for E02 and E04.

A TS result with no baseline is not reviewable.  The first question any reader
has is "would interpolating the two endpoints have done as well?", and the
second is "how does this compare to a method that uses the same information?".
B0a/B0b answer the first; B1 answers the second.

Two rules this module exists to enforce:

  * **Every baseline is scored by the identical ladder** in `quality.Tiers`.
    A baseline scored more leniently than the method is not a baseline.
  * **Every baseline is charged its full cost**, including the cost of
    generating the guess.  CI-NEB's nine images are not free just because the
    saddle refinement afterwards is short.

Baselines are also tagged with the information rung they consume, so the
accuracy-cost figure can distinguish "better" from "better informed".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms

from levels import Level
from systems import Reaction
import spectra


@dataclass
class Guess:
    """A transition-state guess plus what it cost and what it was told."""

    atoms: Atoms
    method: str
    rung: str  # information rung consumed: L0, L1 or L2
    pes_calls: int  # true-PES energy+gradient evaluations spent
    converged: bool = True
    note: str = ""


# --------------------------------------------------------------------------
# B0a / B0b -- the nulls
# --------------------------------------------------------------------------


def _aligned_product(rxn: Reaction) -> np.ndarray:
    """Product positions rotated onto the reactant, so a midpoint is sensible.

    Interpolating between arbitrarily oriented frames produces nonsense that
    would make the null baseline artificially weak.
    """
    p = rxn.reactant.get_positions()
    q = rxn.product.get_positions()
    pc, qc = p.mean(axis=0), q.mean(axis=0)
    u, _, vt = np.linalg.svd((q - qc).T @ (p - pc))
    d = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag([1.0, 1.0, d]) @ vt
    return (q - qc) @ rotation + pc


def midpoint_guess(rxn: Reaction) -> Guess:
    """B0a: the Cartesian midpoint of the aligned endpoints.  Costs nothing."""
    atoms = rxn.reactant.copy()
    atoms.set_positions(0.5 * (rxn.reactant.get_positions() + _aligned_product(rxn)))
    return Guess(atoms, "midpoint", "L0", pes_calls=0)


def idpp_guess(rxn: Reaction, n_images: int = 11) -> Guess:
    """B0b: the middle image of an IDPP interpolation.

    Stronger than a Cartesian midpoint and still essentially free, since IDPP
    optimises an interatomic-distance objective rather than the PES.  If the
    method cannot beat this, the geometric claim is empty.
    """
    from ase.mep import NEB

    product = rxn.product.copy()
    product.set_positions(_aligned_product(rxn))
    images = [rxn.reactant.copy() for _ in range(n_images - 1)] + [product]
    # Only used for its IDPP interpolation, but name the band method anyway:
    # ASE's default changed and the old one is documented as poor.
    neb = NEB(images, method="improvedtangent")
    neb.interpolate(method="idpp")
    return Guess(images[n_images // 2], "idpp", "L0", pes_calls=0)


# --------------------------------------------------------------------------
# B1 / B1r -- Dimer, the like-for-like comparison
# --------------------------------------------------------------------------


def dimer_guess(
    rxn: Reaction,
    level: Level,
    *,
    direction: np.ndarray | None = None,
    steps: int = 60,
    fmax: float = 0.05,
    seed: int | None = None,
) -> Guess:
    """B1/B1r: ASE's Dimer method from the relaxed reactant.

    With ``direction`` set from the constrained-bond vector this consumes
    exactly the L1 information -- the reactant plus which bonds change -- and
    is therefore the honest head-to-head comparison for the paper's claim.
    With a random direction it consumes only the reactant (L2-ish), which
    measures what the bond hint is worth.

    Tuned rather than left at defaults: an under-tuned baseline is a straw
    man, and this is the comparison a referee will scrutinise hardest.
    """
    from ase.mep import DimerControl, MinModeAtoms, MinModeTranslate

    atoms = rxn.reactant.copy()
    atoms.calc = spectra.make_calculator(rxn.charge, rxn.spin, level)
    calls = {"n": 0}
    original = atoms.calc.calculate

    def counted(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    atoms.calc.calculate = counted  # type: ignore[method-assign]

    if direction is None:
        rng = np.random.default_rng(seed)
        direction = rng.normal(size=(len(atoms), 3))
    vector = np.asarray(direction, dtype=float).reshape(len(atoms), 3)
    norm = np.linalg.norm(vector)
    if norm > 1e-12:
        vector = vector / norm

    control = DimerControl(
        initial_eigenmode_method="displacement",
        displacement_method="vector",
        logfile=None,
        maximum_translation=0.1,
        dimer_separation=0.01,
    )
    dimer = MinModeAtoms(atoms, control)
    dimer.displace(displacement_vector=vector * 0.1)
    opt = MinModeTranslate(dimer, logfile=None)
    converged = bool(opt.run(fmax=fmax, steps=steps))

    result = rxn.reactant.copy()
    result.set_positions(dimer.get_positions())
    label = "dimer" if seed is None else f"dimer-random-{seed}"
    rung = "L1" if seed is None else "L2"
    return Guess(result, label, rung, pes_calls=calls["n"], converged=converged)


# --------------------------------------------------------------------------
# B3 -- CI-NEB, the ceiling
# --------------------------------------------------------------------------


def cineb_guess(
    rxn: Reaction,
    level: Level,
    *,
    n_images: int = 9,
    steps: int = 50,
    fmax: float = 0.1,
) -> Guess:
    """B3: the highest image of a climbing-image NEB band.

    The strongest baseline available, and the most expensive: it uses both
    endpoint geometries and roughly an order of magnitude more PES gradient
    evaluations than a bond-space drive.  Charged its full band cost, which is
    the point of putting cost on the x-axis of Figure 3.
    """
    from ase.mep import NEB
    from ase.optimize import FIRE

    product = rxn.product.copy()
    product.set_positions(_aligned_product(rxn))
    images = [rxn.reactant.copy()]
    images += [rxn.reactant.copy() for _ in range(n_images - 2)]
    images += [product]

    neb = NEB(
        images, climb=True, method="improvedtangent",
        allow_shared_calculator=False,
    )
    neb.interpolate(method="idpp")

    calls = {"n": 0}
    # Every image, endpoints included: ASE's NEB reads the endpoint energies
    # on the first force evaluation, so leaving them bare raises "Atoms object
    # has no calculator".  They are charged too -- the band's cost is the whole
    # band's cost, which is the point of putting it on Figure 3's x-axis.
    for image in images:
        image.calc = spectra.make_calculator(rxn.charge, rxn.spin, level)
        original = image.calc.calculate

        def counted(*args, _orig=original, **kwargs):
            calls["n"] += 1
            return _orig(*args, **kwargs)

        image.calc.calculate = counted  # type: ignore[method-assign]

    opt = FIRE(neb, logfile=None)
    converged = bool(opt.run(fmax=fmax, steps=steps))

    energies = [img.get_potential_energy() for img in images[1:-1]]
    peak = images[1:-1][int(np.argmax(energies))]
    return Guess(peak.copy(), "cineb", "L0", pes_calls=calls["n"], converged=converged)


# --------------------------------------------------------------------------
# B2 -- saddle search straight from the reactant
# --------------------------------------------------------------------------


def sella_from_reactant(
    rxn: Reaction, level: Level, *, steps: int = 100, fmax: float = 0.02
) -> Guess:
    """B2: point a saddle optimiser at a minimum and see what happens.

    Expected to fail or wander, because a minimum has no negative curvature to
    follow.  That failure rate is a datum, not a wasted run: it is the
    quantitative statement of why single-ended TS search is hard, and it is
    the thing the method claims to solve.
    """
    result = spectra.refine_saddle(
        rxn.reactant, rxn.charge, rxn.spin, level, fmax=fmax, steps=steps
    )
    return Guess(
        result["atoms"],
        "sella-from-r",
        "L2",
        pes_calls=result["gradient_calls"],
        converged=result["converged"],
    )


# --------------------------------------------------------------------------
# E07's discovery baseline
# --------------------------------------------------------------------------


def constrained_scan(
    atoms: Atoms,
    pair: tuple[int, int],
    charge: int,
    spin: int,
    level: Level,
    *,
    target_distance: float,
    step: float = 0.1,
    relax_steps: int = 20,
) -> dict:
    """Relaxed surface scan along one bond -- what you would do without this.

    Zero new dependencies and entirely standard practice.  Used in E07 as the
    discovery baseline, where its structural limitation is the interesting
    part: a scan drives one internal coordinate, so a concerted two-bond
    transfer is not expressible as a single scan at all.
    """
    from ase.constraints import FixBondLength
    from ase.optimize import BFGS

    work = atoms.copy()
    i, j = pair
    start = work.get_distance(i, j)
    n_steps = max(1, int(abs(target_distance - start) / step))
    frames = []
    calls = {"n": 0}

    for k in range(1, n_steps + 1):
        distance = start + (target_distance - start) * k / n_steps
        work.set_distance(i, j, distance, fix=0.5)
        work.calc = spectra.make_calculator(charge, spin, level)
        original = work.calc.calculate

        def counted(*args, _orig=original, **kwargs):
            calls["n"] += 1
            return _orig(*args, **kwargs)

        work.calc.calculate = counted  # type: ignore[method-assign]
        work.set_constraint(FixBondLength(i, j))
        BFGS(work, logfile=None).run(fmax=0.1, steps=relax_steps)
        work.set_constraint()
        frames.append(work.copy())

    return {"frames": frames, "pes_calls": calls["n"]}


#: Registered so E04 can iterate baselines without a chain of if-statements.
BASELINES: dict[str, dict] = {
    "B0a": {"label": "Cartesian midpoint", "rung": "L0"},
    "B0b": {"label": "IDPP midpoint", "rung": "L0"},
    "B1": {"label": "Dimer (bond direction)", "rung": "L1"},
    "B1r": {"label": "Dimer (random direction)", "rung": "L2"},
    "B2": {"label": "Sella from reactant", "rung": "L2"},
    "B3": {"label": "CI-NEB peak image", "rung": "L0"},
    "B4": {"label": "Bond space L1 (hybrid)", "rung": "L1"},
}
