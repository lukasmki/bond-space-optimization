"""Levels of theory, defined once and imported everywhere.

The split between PRODUCTION and VERIFY is not cosmetic.  Both calculators in
`bondspace.ase` default to ``conv_tol=1e-6``, which is fine for driving a
geometry but far too loose to differentiate twice: the imaginary frequency of a
shallow saddle would be indistinguishable from SCF noise.  Every Hessian,
frequency, IRC and energy comparison in this suite therefore runs at VERIFY,
and every bond-space drive runs at PRODUCTION.

What the two levels *share* is load-bearing.  Functional, basis and grid are
identical, and must stay so: an E01 reference and an E02 drive are compared
directly -- T2 is an RMSD between them, every barrier is a difference of their
energies -- so a basis that differed between them would show up as method error.

`level_shift` shifts nothing at convergence, so VERIFY dropping it is a check
rather than a change: converged results must agree with PRODUCTION's, and any
disagreement is a bug.  E08 measures it.

PRODUCTION no longer reproduces `data/1-run.py`'s settings -- that is cc-pVDZ,
this is cc-pVTZ -- so results here supersede the existing study rather than
extending it.
"""

from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class Level:
    """A complete specification of an electronic-structure method."""

    name: str
    xc: str
    basis: str
    conv_tol: float
    level_shift: float | tuple[float, float]
    density_fit: bool = True
    #: PySCF DFT quadrature grid; None leaves its default (level 3).
    grid_level: int | None = None

    def calc_kwargs(self) -> dict:
        """Keyword arguments common to both calculators in `bondspace.ase`."""
        return {
            "xc": self.xc,
            "basis": self.basis,
            "conv_tol": self.conv_tol,
            "level_shift": self.level_shift,
            "grid_level": self.grid_level,
        }

    def as_dict(self) -> dict:
        return asdict(self)


#: The quadrature grid for both production levels.  PySCF defaults to 3; one
#: step up roughly doubles the point count (33.7k -> 59.7k on water/STO-3G).
#: Not a free knob here: a Mayer bond order is a trace over the density and
#: overlap matrices and `bo_gradient` differentiates it through a CPHF solve, so
#: XC quadrature noise reaches the bond orders, the forces the drives follow,
#: and the shallow imaginary modes E01 must resolve -- rxn_15's is 263i.
GRID_LEVEL = 4

#: Every bond-space drive.
#:
#: This is cc-pVTZ, where `data/1-run.py` is cc-pVDZ, so it no longer
#: reproduces that study's settings and its numbers are not comparable to the
#: existing 12/19 and 8/19 figures.  That comparison was already unavailable
#: (see the README's note on `HCombustion-combined/`); this makes it explicit.
PRODUCTION = Level(
    name="production",
    xc="HYB_GGA_XC_WB97X_D3",
    basis="cc-pvtz",
    conv_tol=1e-6,
    level_shift=(0.3, 0.2),
    grid_level=GRID_LEVEL,
)

#: Every Hessian, frequency, IRC and energy comparison.
#:
#: Functional, basis and grid track PRODUCTION exactly -- only `conv_tol` and
#: `level_shift` differ.  That is what makes an E01 reference and an E02 drive
#: comparable at all: let the bases drift apart and every T2 RMSD and every
#: barrier measures the basis change as much as the method.
VERIFY = Level(
    name="verify",
    xc="HYB_GGA_XC_WB97X_D3",
    basis="cc-pvtz",
    conv_tol=1e-10,
    level_shift=0.0,
    grid_level=GRID_LEVEL,
)

#: Smoke tests and CI.  Cheap enough to exercise every code path in minutes.
CHEAP = Level(
    name="cheap",
    xc="HYB_GGA_XC_WB97X_D3",
    basis="sto-3g",
    conv_tol=1e-9,
    level_shift=(0.3, 0.2),
)

LEVELS = {level.name: level for level in (PRODUCTION, VERIFY, CHEAP)}


def shifted_variant(level: Level) -> Level:
    """`level` with PRODUCTION's level shift, for an SCF that will not converge.

    Not a different method.  A level shift moves nothing at convergence -- which
    is the whole reason VERIFY drops it as a check -- so this is scaffolding for
    getting *through* a stretched open-shell geometry, and whatever it reaches
    must be re-evaluated at the unshifted level before it is recorded.  E01
    reaches for it only after a saddle refinement has already failed: rxn_18
    burned its 300-step budget and then could not converge a Hessian at all.
    """
    # `replace` rather than a field-by-field rebuild: a variant that silently
    # drops a field is a different method wearing the same name, and the
    # enumeration is exactly what stops tracking `Level` when a field is added.
    return replace(
        level,
        name=f"{level.name}-shifted",
        level_shift=PRODUCTION.level_shift,
    )


def smoke_variant(level: Level) -> Level:
    """The CHEAP counterpart of a level, for `--smoke` runs.

    Keeps the level's own convergence discipline (a smoke VERIFY still
    converges tightly and on the same grid, so the frequency code path is
    exercised honestly) but drops to a minimal basis.  See `shifted_variant`
    for why this is `replace` and not a field-by-field rebuild.
    """
    return replace(level, name=f"{level.name}-smoke", basis="sto-3g")
