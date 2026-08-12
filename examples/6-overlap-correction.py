"""Why `ovlp_thresh` exists.

Bond order is a good reaction coordinate over the range where a bond actually
exists, but it decays to zero once fragments separate -- and its gradient
decays faster still.  A bond targeted to 0.0 therefore stops generating any
force well before the fragments have genuinely come apart, and the optimizer
stalls with the pieces still on top of each other.

`ovlp_thresh` adds a repulsion built from the atom-blocked overlap S_AB,

    E += S_AB^2 / 2

for every bond targeted *below* the threshold.  The overlap decays far more
slowly than the bond order, so it keeps pushing where bond order cannot.

The system is H2 + H as a doublet, scanning the lone hydrogen away.  An
open-shell reference is used deliberately: a closed-shell singlet in a
spin-restricted reference never dissociates, so its bond order would sit at
exactly 1.0 at every distance and this effect would be invisible.

Run: uv run python examples/6-overlap-correction.py
"""

from ase import Atoms

from bondspace.ase import BondFluxCalculator

DISTANCES = [0.9, 1.2, 1.5, 1.8, 2.2, 2.8, 3.5]


def probe(r: float, ovlp_thresh: float) -> tuple[float, float]:
    """Bond order and force on the departing atom, at separation r."""
    atoms = Atoms("HHH", positions=[[-0.74, 0, 0], [0.0, 0, 0], [r, 0, 0]])
    atoms.calc = BondFluxCalculator(
        [(1, 2, 0.0)],  # ask for H1 and H2 to be unbonded
        charge=0,
        spin=1,
        basis="cc-pvdz",
        # The default thresh=0.05 would early-return zero force as soon as the
        # bond order fell below it, hiding exactly what we want to look at.
        thresh=1e-6,
        ovlp_thresh=ovlp_thresh,
        level_shift=(0.3, 0.2),
    )
    force = abs(atoms.get_forces()[2, 0])
    return atoms.get_array("bond-order")[1, 2], force


if __name__ == "__main__":
    print("H2 + H (doublet); the H1-H2 pair is targeted to bond order 0.0.")
    print("ovlp_thresh=0.0 disables the correction (the mask is bmat < ovlp_thresh),")
    print("ovlp_thresh=0.5 enables it for this 0.0-target bond.\n")
    print("   d(H1-H2)   bond order    |force| without   |force| with    ratio")

    rows = []
    for r in DISTANCES:
        B, f_off = probe(r, 0.0)
        _, f_on = probe(r, 0.5)
        rows.append((r, B, f_off, f_on))
        ratio = f_on / f_off if f_off > 1e-9 else float("inf")
        print(f"   {r:6.2f}    {B:9.4f}    {f_off:13.5f}   {f_on:12.5f}  {ratio:9.0f}x")

    r0, B0, off0, on0 = rows[0]
    r1, B1, off1, on1 = rows[3]
    print(
        f"\nFrom {r0} to {r1} A the bond order falls {B0:.3f} -> {B1:.3f}, a factor of"
        f" {B0 / B1:.0f},"
    )
    print(
        f"but the bond-order force falls {off0:.4f} -> {off1:.4f}, a factor of"
        f" {off0 / off1:.0f}."
    )
    print("The gradient vanishes much faster than the quantity itself, which is")
    print("the whole difficulty: past ~1.5 A there is nothing left to optimize.")

    print("""
So bond-breaking targets need `ovlp_thresh` above them, while bond-forming
targets must stay below it -- the correction is repulsive, and applying it to
a bond you are trying to *make* would fight the objective.  With targets of
0.0 and 1.0, any ovlp_thresh in (0, 1] does the right thing.""")
