# Examples

Worked examples for `bondspace`, in reading order. Each is standalone:

```bash
uv run python examples/1-bond-order.py
```

Outputs are written to `examples/out/` (gitignored). Everything uses cc-pVDZ
rather than the calculators' cc-pVTZ default, for speed.

| | Example | Shows | Time |
| --- | --- | --- | --- |
| 1 | [`1-bond-order.py`](1-bond-order.py) | Mayer bond orders from `PySCFCalculator`; the `bond-order` array, `info["connectivity"]`, Mayer valences | ~30 s |
| 2 | [`2-bond-order-gradient.py`](2-bond-order-gradient.py) | `bo_grad=True`, the `(natm, 3*natm*natm)` storage convention, gradients as viewable modes, `bo_grad_atoms` | ~10 s |
| 3 | [`3-validate-gradient.py`](3-validate-gradient.py) | The analytic gradient against finite differences — the correctness gate | ~20 s |
| 4 | [`4-bond-flux-drive.py`](4-bond-flux-drive.py) | `BondFluxCalculator` driving H2 + H → H + H2 by asking for bond orders | ~20 s |
| 5 | [`5-reaction-path.py`](5-reaction-path.py) | **Flagship.** H2 + OH → H + H2O: reactant → transition state and → product | ~3–5 min |
| 6 | [`6-overlap-correction.py`](6-overlap-correction.py) | Why `ovlp_thresh` exists: the bond-order gradient dies at long range | ~40 s |
| 7 | [`7-gradient-methods.py`](7-gradient-methods.py) | `zvector=True` vs `restrict_gradient=True` vs direct — cost and exactness | ~10 s |

`rxn_03.xyz` is input data for example 5 (reactant / TS / product of one
HCombustion reaction). `_old/` holds the previous example set and is not
maintained — two of those scripts no longer resolve their input paths.

## The idea

An ordinary geometry optimization moves atoms downhill on the potential energy
surface. These examples do something different: they define an objective in
*bond space*,

```python
E = 0.5 * sum[(B - B_target)**2]
```

over the Mayer bond-order matrix `B`, and hand its analytic gradient to an ASE
optimizer. The geometry then moves until the bonds are what you asked for. You
specify which bonds should form and break; you do not specify a path, and no
energy is minimized.

The consequence worth internalizing (example 5): a transition state is reached
by asking for *half* bonds. On the energy surface that is a saddle point
needing dedicated search machinery; in bond space it is just another target.

## Things that will bite you

Collected from the examples, in the order you are likely to hit them:

- **`BondFluxCalculator` energies and forces are not physical.** The "energy"
  is bond-order error and it has no PES term at all. Do not compare it across
  geometries as if it were a potential energy.
- **`spin` must match electron-count parity**, and it is a count of unpaired
  electrons, not a multiplicity. It is fixed for a whole run, since atoms are
  conserved.
- **Spin-restricted references do not dissociate.** A closed-shell singlet
  holds its bond order near 1.0 at any distance, so bond-breaking studies need
  an open-shell reference — this is why example 6 uses a doublet.
- **The constraint gradient is small when a bond must form from far away.** A
  bond order of 0.03 at 2.3 Å is almost flat, so a drive that has to form a
  bond across a gap starts with very little force and can converge before it
  has moved. Loosen `fmax` only after checking the force is not simply small.
- **Cap the optimizer's `maxstep`.** The constraint "energy" is a sum of
  squared bond orders, so its scale bears no relation to a potential energy
  surface and FIRE2's 0.2 Å default overshoots into geometries where the SCF
  diverges. Example 5 uses `maxstep=0.05`.
- **`ovlp_thresh` must sit above your break targets and below your form
  targets.** The correction is repulsive; applied to a bond you are forming it
  fights the objective.
- **`thresh` is early stopping.** Below it the calculator returns zero energy
  and force so the optimizer halts — which will hide any effect you are trying
  to measure at small errors (example 6 sets it to `1e-6`).
- **`level_shift`** is usually needed for SCF convergence on stretched or
  radical geometries; `(0.3, 0.2)` is the value used throughout `data/`.

## Beyond these

`data/` holds the full studies: `0-prep.py` and `1-run.py` replay the 19
HCombustion reactions, and `2-network.py` with `network.py` does reaction
network discovery — enumerating elementary bond rearrangements from a seed and
identifying whatever species come out, with nothing told about the products in
advance.

The Z-vector gradient used by `zvector=True` is derived in
[`../src/bondspace/README.md`](../src/bondspace/README.md).
