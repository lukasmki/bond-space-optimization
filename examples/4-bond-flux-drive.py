"""Drive a reaction by specifying bonds instead of following energy.

`BondFluxCalculator` replaces the potential energy surface with a fictitious
harmonic objective in bond space,

    E = 1/2 sum (B - B_target)^2

and returns its gradient as a force.  Handing that to an ASE optimizer moves
the geometry until the bond-order matrix matches what was asked for.  Nothing
here follows the energy: the reported "energy" is bond-order error, and it is
not physical.

The system is the hydrogen exchange H_a-H_b + H_c  ->  H_a + H_b-H_c: break
one bond, form the other, in a single concerted request.

Run: uv run python examples/4-bond-flux-drive.py
"""

from pathlib import Path

from ase import Atoms, io
from ase.optimize import FIRE2

from bondspace.ase import BondFluxCalculator

OUT = Path(__file__).parent / "out"

# Pairs absent from this list are unconstrained, so the atoms stay free to
# relax however the constrained bonds require.
BONDS = [
    (0, 1, 0.0),  # break H0-H1
    (1, 2, 1.0),  # form  H1-H2
]


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)

    atoms = Atoms("HHH", positions=[[0.0, 0, 0], [0.75, 0, 0], [2.6, 0, 0]])
    atoms.calc = BondFluxCalculator(
        BONDS,
        charge=0,
        spin=1,  # 3 electrons -> one unpaired
        basis="cc-pvdz",
        thresh=0.05,
        ovlp_thresh=0.5,  # only the bond targeted to 0.0 gets the repulsion
        level_shift=(0.3, 0.2),
    )

    print("target:  H0-H1 -> 0.0 (break)   H1-H2 -> 1.0 (form)\n")
    print(" step   d(H0-H1)  d(H1-H2)    B(H0-H1)  B(H1-H2)   bond-order error")

    images: list[Atoms] = []

    def log() -> None:
        B = atoms.get_array("bond-order")
        images.append(atoms.copy())
        print(f"  {len(images) - 1:3d}    {atoms.get_distance(0, 1):7.3f}   "
              f"{atoms.get_distance(1, 2):7.3f}     {B[0, 1]:7.3f}   {B[1, 2]:7.3f}"
              f"      {atoms.get_potential_energy():10.4f}")

    opt = FIRE2(atoms, logfile=None)
    opt.attach(log, 1)
    converged = opt.run(fmax=0.1, steps=40)

    io.write(OUT / "4-bond-flux-drive.xyz", images, format="extxyz")

    B = atoms.get_array("bond-order")
    print(f"\nconverged: {bool(converged)}   ({len(images)} steps)")
    print(f"final     B(H0-H1) = {B[0, 1]:.3f}  (target 0.0)")
    print(f"          B(H1-H2) = {B[1, 2]:.3f}  (target 1.0)")
    print(f"wrote trajectory to {OUT / '4-bond-flux-drive.xyz'}")

    print("""
Note the reported energy falls to zero as the bond orders reach their
targets -- it is the constraint residual, not a physical energy.  The
optimizer stops early once the largest bond-order error drops below
`thresh`, at which point the calculator returns zero energy and force.""")
