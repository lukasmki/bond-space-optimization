"""Hessians, frequencies, and the saddle/IRC machinery that verifies a TS.

Nothing in `src/bondspace` computes a second derivative -- `mf.Hessian()` is
used there only as a vehicle for CPHF.  This module is where the suite leaves
bond space and asks the potential energy surface directly whether a structure
is a first-order saddle point.

Everything here runs at VERIFY, never PRODUCTION.  `conv_tol=1e-6` is fine for
driving a geometry but the imaginary frequency of a shallow saddle would be
indistinguishable from SCF noise at that tolerance.
"""

from __future__ import annotations

import contextlib
import io as _io
from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.optimize import BFGS

from bondspace.ase import PySCFCalculator
from bondspace.util import ase_to_pyscf
from levels import Level

#: Below this magnitude an imaginary frequency is treated as numerical noise
#: rather than a genuine negative curvature.  Reported alongside the raw
#: spectrum so the choice can be re-litigated at analysis time.
IMAGINARY_CUTOFF_CM = 50.0


def make_calculator(rxn_charge: int, rxn_spin: int, level: Level, **extra):
    """A true-PES calculator, with a *fresh* SCF guess at every geometry.

    Everything in this module treats the PES as a single-valued function: a
    minimisation, a Hessian, an IRC.  Seeding each SCF from the previous
    geometry's orbitals breaks that -- the SCF can settle on a different
    solution, and the energy then jumps by ~1 eV between adjacent geometries
    while the forces agree with neither.  Measured on rxn_03's reactant, BFGS
    with orbital reuse oscillates for 150 steps at |F| ~ 3 eV/Ang; with a fresh
    guess the same relaxation converges to |F| < 0.01 in 57.

    Bond-space drives keep the default (reuse), because that is what `data/`
    did and what E08's `fresh_guess` row measures.
    """
    extra.setdefault("reuse_guess", False)
    return PySCFCalculator(
        charge=rxn_charge, spin=rxn_spin, **level.calc_kwargs(), **extra
    )


# --------------------------------------------------------------------------
# single points
# --------------------------------------------------------------------------


def single_point(atoms: Atoms, charge: int, spin: int, level: Level) -> dict:
    """True PES energy, forces and bond orders at one geometry."""
    work = atoms.copy()
    work.calc = make_calculator(charge, spin, level)
    energy = work.get_potential_energy()
    forces = work.get_forces()
    return {
        "energy": float(energy),
        "forces": np.asarray(forces),
        "max_force": float(np.linalg.norm(forces, axis=1).max()),
        "bond_order": np.asarray(work.get_array("bond-order")),
    }


# --------------------------------------------------------------------------
# Hessian and normal modes
# --------------------------------------------------------------------------


@dataclass
class Spectrum:
    """Harmonic analysis of one geometry."""

    wavenumbers: np.ndarray  # cm^-1, negative entries are imaginary
    modes: np.ndarray  # (nmode, natm, 3)
    n_imaginary: int
    omega_imaginary: float  # most negative wavenumber, or nan
    energy: float

    @property
    def imaginary_mode(self) -> np.ndarray | None:
        if self.n_imaginary == 0:
            return None
        return self.modes[int(np.argmin(self.wavenumbers))]

    def as_dict(self) -> dict:
        return {
            "wavenumbers": self.wavenumbers.tolist(),
            "n_imaginary": int(self.n_imaginary),
            "omega_imaginary": float(self.omega_imaginary),
            "energy": float(self.energy),
        }


def hessian_spectrum(atoms: Atoms, charge: int, spin: int, level: Level) -> Spectrum:
    """Analytic DF-UKS Hessian -> harmonic frequencies and normal modes.

    PySCF returns imaginary frequencies as complex numbers; they are flattened
    here to signed real wavenumbers (negative = imaginary), which is the
    convention every downstream consumer expects.
    """
    from pyscf import dft
    from pyscf.hessian import thermo

    mol = ase_to_pyscf(atoms, basis=level.basis, charge=charge, spin=spin)
    mol.verbose = 0
    mf = dft.UKS(mol, xc=level.xc)
    if level.density_fit:
        mf = mf.density_fit()
    mf.conv_tol = level.conv_tol
    mf.level_shift = level.level_shift
    energy = mf.kernel()
    if not mf.converged:
        raise RuntimeError("SCF did not converge; cannot trust a Hessian")

    hess = mf.Hessian().kernel()
    with contextlib.redirect_stdout(_io.StringIO()):
        analysis = thermo.harmonic_analysis(mol, hess)

    freq = np.asarray(analysis["freq_wavenumber"])
    if np.iscomplexobj(freq):
        # PySCF encodes an imaginary frequency as a purely imaginary number.
        signed = np.where(np.abs(freq.imag) > 0, -np.abs(freq.imag), freq.real)
    else:
        signed = freq.astype(float)

    modes = np.asarray(analysis["norm_mode"])
    n_imag = int((signed < -IMAGINARY_CUTOFF_CM).sum())
    omega = float(signed.min()) if signed.size else float("nan")

    from ase import units

    return Spectrum(
        wavenumbers=signed,
        modes=modes,
        n_imaginary=n_imag,
        omega_imaginary=omega if omega < 0 else float("nan"),
        energy=float(energy * units.Hartree),
    )


# --------------------------------------------------------------------------
# minimisation and saddle refinement
# --------------------------------------------------------------------------


def relax(
    atoms: Atoms,
    charge: int,
    spin: int,
    level: Level,
    *,
    fmax: float = 0.01,
    steps: int = 200,
    internal: bool = False,
) -> tuple[Atoms, bool, int]:
    """Relax to a minimum on the true PES.

    Uses BFGS rather than FIRE2: this is a real potential energy surface with
    a meaningful curvature scale, and the tight fmax the reference states need
    is expensive to reach with a damped-dynamics optimiser.

    ``internal=True`` routes through geomeTRIC's internal coordinates instead,
    which converges the fragment-separated products more reliably -- several
    of these reactions end at 3-4 Ang separation, where a Cartesian Hessian
    approximation is poorly conditioned.  It falls back to BFGS rather than
    failing, since geomeTRIC's coordinate system construction can itself fail
    on a dissociated graph, and a reference state that quietly did not relax
    is worse than one that relaxed by a different route.
    """
    work = atoms.copy()
    work.calc = make_calculator(charge, spin, level)
    if internal:
        relaxed = _geometric_relax(work, fmax=fmax, steps=steps)
        if relaxed is not None:
            return relaxed
    opt = BFGS(work, logfile=None)
    converged = bool(opt.run(fmax=fmax, steps=steps))
    return work, converged, opt.get_number_of_steps()


def _geometric_relax(
    atoms: Atoms, *, fmax: float, steps: int
) -> tuple[Atoms, bool, int] | None:
    """geomeTRIC minimisation; None if its coordinate system cannot be built."""
    import tempfile
    from pathlib import Path

    try:
        from geometric.ase_engine import EngineASE
        from geometric.optimize import run_optimizer
    except ImportError:
        return None

    from ase import units

    try:
        with tempfile.TemporaryDirectory() as tmp:
            xyz = Path(tmp) / "start.xyz"
            atoms.write(xyz, format="xyz")
            engine = (
                EngineASE.from_calculator_string(str(xyz), "", calculator=atoms.calc)
                if hasattr(EngineASE, "from_calculator_string")
                else None
            )
            if engine is None:
                from geometric.molecule import Molecule

                engine = EngineASE(Molecule(str(xyz)), atoms.calc)
            result = run_optimizer(
                customengine=engine,
                input=str(xyz),
                converge=["gmax", f"{fmax / (units.Hartree / units.Bohr):.3e}"],
                maxiter=steps,
                check=0,
                prefix=str(Path(tmp) / "opt"),
                logIni=None,
            )
        work = atoms.copy()
        work.set_positions(result.xyzs[-1])
        work.calc = atoms.calc
        forces = work.get_forces()
        converged = bool(np.linalg.norm(forces, axis=1).max() < fmax)
        return work, converged, len(result.xyzs) - 1
    except Exception:
        # geomeTRIC's internal-coordinate construction is the fragile part;
        # falling back is preferable to losing the reference state.
        return None


def refine_saddle(
    atoms: Atoms,
    charge: int,
    spin: int,
    level: Level,
    *,
    fmax: float = 0.02,
    steps: int = 100,
    internal: bool = True,
) -> dict:
    """Refine a guess to a first-order saddle with Sella.

    This is what turns a bond-space endpoint -- which is *not* a stationary
    point, because the calculator returns no PES force -- into something that
    can be called a transition state.  The number of steps it takes is the
    basin-of-attraction measurement that E02 and E04 compare across guesses.
    """
    from sella import Sella

    work = atoms.copy()
    work.calc = make_calculator(charge, spin, level)
    calls = {"n": 0}
    original = work.calc.calculate

    def counted(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    work.calc.calculate = counted  # type: ignore[method-assign]

    opt = Sella(work, internal=internal, logfile=None)
    converged = bool(opt.run(fmax=fmax, steps=steps))
    return {
        "atoms": work,
        "converged": converged,
        "steps": opt.get_number_of_steps(),
        "gradient_calls": calls["n"],
    }


def _curvature_aware_irc(work: Atoms, dx: float):
    """Sella's `IRC`, with its own convergence test reconnected to ASE >= 3.29.

    Sella 2.5.0 overrides `Optimizer.converged`, which used to be what the
    optimisation loop called.  ASE 3.29 inverted that: `Dynamics.irun` now
    calls `gradient_converged` and it is `converged` that delegates to *it*.
    Sella's override is therefore never reached, and the IRC silently falls
    back to ASE's plain `max|force| < fmax`.

    That is not a small difference.  Sella's criterion additionally demands
    `evals[0] > 0` -- positive curvature -- for the express purpose of
    refusing to stop at a saddle, where the gradient vanishes by definition.
    Without it an IRC started from a well-converged transition state halts one
    step off it and reports both directions as the same structure.
    """
    from sella import IRC

    class _IRC(IRC):
        def gradient_converged(self, gradient):
            return IRC.converged(self)

    return _IRC(work, dx=dx, logfile=None, keep_going=True)


def run_irc(
    atoms: Atoms,
    charge: int,
    spin: int,
    level: Level,
    *,
    fmax: float = 0.02,
    steps: int = 60,
    dx: float = 0.1,
) -> dict:
    """Follow the reaction path downhill from a saddle, both directions.

    The endpoints are what decide whether a located saddle belongs to the
    reaction that was asked for.  A structure with one imaginary frequency is
    only the *right* transition state if its IRC arrives at the intended
    reactant and product -- E02's T4 and E07's edge verification both turn on
    this.
    """
    # One IRC object run twice, not two objects: sella diagonalises the mass-
    # weighted Hessian on the first call and restores that same `v0ts` for the
    # second direction, so the two branches are guaranteed to be opposite
    # halves of one mode.  Two independent objects would pay for the
    # diagonalisation twice and rely on its sign convention to agree.
    work = atoms.copy()
    work.calc = make_calculator(charge, spin, level)
    frames: list[Atoms] = []
    opt = _curvature_aware_irc(work, dx)
    opt.attach(lambda: frames.append(work.copy()))

    out: dict = {}
    for direction in ("forward", "reverse"):
        frames.clear()
        converged = bool(opt.run(fmax=fmax, steps=steps, direction=direction))
        # How far the branch actually travelled.  An IRC that stops on the
        # saddle is indistinguishable from one that connects nothing, and it
        # reports success either way -- so record the displacement and let the
        # caller refuse to draw a conclusion from a branch that never moved.
        displacement = float(np.abs(work.get_positions() - atoms.get_positions()).max())
        out[direction] = {
            "atoms": work.copy(),
            "frames": list(frames),
            "converged": converged,
            "displacement": displacement,
            "steps": opt.get_number_of_steps(),
        }
    return out


# --------------------------------------------------------------------------
# species identification
# --------------------------------------------------------------------------


def species_label(atoms: Atoms, thresh: float = 0.5) -> str:
    """Canonical "H2O + OH"-style label, via data/network.py's fragmenter.

    Reusing the network module's labeller means E01, E02 and E07 all agree on
    what counts as the same species -- three independent definitions would be
    three ways to disagree about whether a run succeeded.
    """
    import sys
    from pathlib import Path

    data = Path(__file__).parent.parent / "data"
    if str(data) not in sys.path:
        sys.path.insert(0, str(data))
    import network  # type: ignore[import-not-found]

    return network.species_label(atoms, thresh=thresh)


def irc_connects(
    irc: dict,
    reactant: Atoms,
    product: Atoms,
    charge: int,
    spin: int,
    level: Level,
    *,
    thresh: float = 0.5,
    relax_steps: int = 60,
) -> dict:
    """Do the two IRC endpoints relax to the intended species?

    A bond-flux endpoint is not a stationary structure and neither is an IRC
    terminus at a finite step budget, so both must be relaxed on the real PES
    before their species label means anything -- the same reasoning
    data/network.py already applies before labelling a discovered node.
    """
    want = {species_label(reactant, thresh), species_label(product, thresh)}
    got = {}
    for direction, result in irc.items():
        relaxed, converged, _ = relax(
            result["atoms"], charge, spin, level, fmax=0.05, steps=relax_steps
        )
        got[direction] = {
            "label": species_label(relaxed, thresh),
            "relax_converged": converged,
            "atoms": relaxed,
        }
    labels = {v["label"] for v in got.values()}
    return {
        "expected": sorted(want),
        "observed": sorted(labels),
        "connects": labels == want,
        "endpoints": got,
    }
