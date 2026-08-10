"""Levels of theory, defined once and imported everywhere.

The split between PRODUCTION and VERIFY is not cosmetic.  Both calculators in
`bondspace.ase` default to ``conv_tol=1e-6``, which is fine for driving a
geometry but far too loose to differentiate twice: the imaginary frequency of a
shallow saddle would be indistinguishable from SCF noise.  Every Hessian,
frequency, IRC and energy comparison in this suite therefore runs at VERIFY,
and every bond-space drive runs at PRODUCTION -- which reproduces `data/`'s
settings exactly, so the results are comparable to the existing study.

`level_shift` shifts nothing at convergence, so VERIFY dropping it is a check
rather than a change: converged results must agree with PRODUCTION's, and any
disagreement is a bug.  E08 measures it.
"""

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Level:
    """A complete specification of an electronic-structure method."""

    name: str
    xc: str
    basis: str
    conv_tol: float
    level_shift: float | tuple[float, float]
    density_fit: bool = True

    def calc_kwargs(self) -> dict:
        """Keyword arguments common to both calculators in `bondspace.ase`."""
        return {
            "xc": self.xc,
            "basis": self.basis,
            "conv_tol": self.conv_tol,
            "level_shift": self.level_shift,
        }

    def as_dict(self) -> dict:
        return asdict(self)


#: Every bond-space drive.  Matches data/1-run.py exactly.
PRODUCTION = Level(
    name="production",
    xc="HYB_GGA_XC_WB97X_D3",
    basis="cc-pvdz",
    conv_tol=1e-6,
    level_shift=(0.3, 0.2),
)

#: Every Hessian, frequency, IRC and energy comparison.
VERIFY = Level(
    name="verify",
    xc="HYB_GGA_XC_WB97X_D3",
    basis="cc-pvdz",
    conv_tol=1e-10,
    level_shift=0.0,
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


def smoke_variant(level: Level) -> Level:
    """The CHEAP counterpart of a level, for `--smoke` runs.

    Keeps the level's own convergence discipline (a smoke VERIFY still
    converges tightly, so the frequency code path is exercised honestly) but
    drops to a minimal basis.
    """
    return Level(
        name=f"{level.name}-smoke",
        xc=level.xc,
        basis="sto-3g",
        conv_tol=level.conv_tol,
        level_shift=level.level_shift,
        density_fit=level.density_fit,
    )
