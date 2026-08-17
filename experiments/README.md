# Bond Space Optimization Experiments

The publication suite. Every headline number here is (a) computed **outside**
bond space, (b) compared against a named baseline, and (c) reported alongside a
stricter version of itself.

---

## 1. The information ladder

Read this before any result. "Single-ended" is a claim about **information**,
not about which function was called: a run whose half-bond targets come from the
converged bond-order matrix of the product has been handed the product geometry,
and starting the optimisation from the reactant does not change that.

| Rung | Information supplied | Targets built from | Honest description |
|---|---|---|---|
| **L0** | reference R **and** P *geometries* | `0.5·(B_R + B_P)`, rounded to halves — exactly `data/1-run.py` | **Not** single-ended. Continuity baseline only. |
| **L1** | the chemical equation: integer Lewis orders + the atom mapping | `0.5·(n_R + n_P)` on changing pairs; spectators unconstrained | **The paper's claim.** No product geometry, no reference DFT. |
| **L2** | the reactant alone, plus `data/network.py`'s move enumerator | every enumerated break/form/transfer, half-targeted | Single-ended **and** product-agnostic. |

`L1` carries the abstract's number. `L0` measures what the reference geometries
were worth. The gap between them is a result in its own right (E03).

The Lewis integers live in [`systems.py`](systems.py) as a hand-audited table
rather than being derived by rounding the reference matrices, because rounding is
sometimes wrong in ways that matter:

- `rxn_01`'s reactant has O–O at **2.05 Å** — unbound — but a Mayer bond order of
  1.09, because a fixed high-spin reference smears density between the fragments.
- The six spin-nonconserving reactions dissociate to 3–4 Å while their Mayer
  orders stay at the bonded value (`rxn_05`: H–H = 1.00 at 3.0 Å; `rxn_06`: O–O =
  2.00 at 4.0 Å). A spin-restricted single determinant cannot dissociate — so the
  **L0 targets for those reactions ask for no change at all.**

E01 tabulates every such disagreement.

---

## 2. The claims, and what verifies them

**Claim A — single-ended transition-state finding.** A bond-flux endpoint is
*not* a stationary point on the PES; the calculator returns no PES force, so
there is no reason for `∇E = 0` there. "We found the TS" is indefensible as
stated. What is defensible, and stronger:

> Bond space is a **transition-state guess generator**: from the reactant plus a
> statement of which bonds change, it lands inside the saddle's basin of
> attraction, so a conventional saddle optimiser converges from it faster and
> more often than from the standard guesses.

**Claim B — reaction path and network finding.** A drive is an *optimisation
trajectory*, not an intrinsic reaction coordinate: it is FIRE2-momentum
dependent, unevenly spaced, and not mass-weighted. It cannot be an IRC and the
paper must not imply it is. What can be asked is whether it stays near one (E06),
and whether move enumeration rediscovers known chemistry (E07).

### Claim → experiment → figure

| Experiment | Question | Figure | Rank |
|---|---|---|---|
| **E01** `01_reference_states` | What is the ground truth at *our* level? | Table S1, Fig S1 | 4 |
| **E02** `02_ts_single_ended` | How close does asking for half bonds get, from the reactant alone? | Fig 1 | **1** |
| **E03** `03_information_ladder` | How much came from the reference product geometry? | Fig 2 | 3 |
| **E04** `04_ts_baselines` | Versus what? | Fig 3 | **2** |
| **E05** `05_target_sharpness` | Is 0.5 special? | Fig 4 | 5 |
| **E06** `06_path_vs_irc` | Is the R→P drive a meaningful path? | Fig 5 | 7 |
| **E07** `07_network_discovery` | Does it rediscover combustion chemistry unprompted? | Fig 6 | 6 |
| **E08** `08_ablations` | How sensitive is it to knobs? | Fig 8 | 8 |
| **E09** `09_scaling` | What does the Z-vector adjoint buy? | Fig 7 | 10 |
| **E10** `10_failure_atlas` | When it fails, why? | Table 4 | 9 |

### The metric ladder (E02, and every baseline in E04)

The same runs scored five ways, each strictly harder than the last:

| Tier | Definition |
|---|---|
| **T0** | `max\|ΔB\| < 0.5` over **reference-bonded pairs** — *the existing metric*, for continuity |
| **T1** | `max\|ΔB\| < 0.5` over **all pairs** — catches invented bonds, which T0 structurally cannot see |
| **T2** | permutation-aware heavy-atom RMSD to the verified TS < 0.25 Å |
| **T3** | exactly one imaginary frequency (\|ω\| > 50i cm⁻¹) |
| **T4** | Sella refines it **and** its IRC connects the correct R and P |

**The paper reports T4.** The attrition from T0 to T4 is the result.

---

## 3. Levels of theory

| Level | Purpose | Settings |
|---|---|---|
| `PRODUCTION` | every bond-space drive | UKS/ωB97X-D3/cc-pVTZ, DF, grid level 4, `conv_tol=1e-6`, `level_shift=(0.3,0.2)` |
| `VERIFY` | every Hessian, frequency, IRC, energy comparison | same functional/basis/grid, DF, **`conv_tol=1e-10`, `level_shift=0`** |
| `CHEAP` | `--smoke` and CI | STO-3G, same grid and `conv_tol` as the level it stands in for |

The split is not cosmetic. Both calculators default to `conv_tol=1e-6`, which is
fine for driving a geometry but far too loose to differentiate twice — a shallow
saddle's imaginary frequency would be indistinguishable from SCF noise. Hence the
`conv_tol` kwarg added to `bondspace.ase`.

`level_shift` shifts nothing at convergence, so VERIFY dropping it is a **check**:
converged results must agree with PRODUCTION's, and a disagreement is a bug, not a
finding. E08 measures it.

**Functional, basis and grid are shared by construction, not by coincidence.**
An E01 reference and an E02 drive are compared directly — T2 is an RMSD between
them and every barrier is a difference of their energies — so a basis that
differed between the two would be measured as method error. Only `conv_tol` and
`level_shift` may differ, and both are checks rather than choices.

**Grid level 4**, one step above PySCF's default (33.7k → 59.7k points on
water/STO-3G). A Mayer bond order is a trace over the density and overlap
matrices and `bo_gradient` differentiates it through a CPHF solve, so XC
quadrature noise reaches the bond orders, the forces the drives follow, and the
shallow imaginary modes E01 has to resolve — `rxn_15`'s is 263i, `rxn_12`'s
489i. Set via the `grid_level` kwarg on both calculators; `None` keeps PySCF's
default.

**cc-pVTZ is not what `data/` ran.** `data/1-run.py` is cc-pVDZ, so the numbers
here do not reproduce that study and are not comparable to its 12/19 and 8/19
figures — which were already not regenerable (see the note under §5). E08's
basis ladder carries a `cc-pvdz` row, so what the move to triple zeta bought is
measured rather than assumed.

### The SCF guess is not free

Both calculators seed each SCF from the previous geometry's orbitals. That makes
the surface **path-dependent**: seeded from a stale guess the SCF can settle on a
different solution, and the energy then jumps by ~1 eV between adjacent
geometries while the forces agree with neither. On `rxn_03`'s reactant a BFGS
relaxation with orbital reuse oscillates for 150 steps at |F| ≈ 3 eV/Å; with a
fresh guess the same relaxation reaches |F| < 0.01 in 57.

So `bondspace.ase` gained a `reuse_guess` kwarg. **Everything in `spectra.py`
sets it False** — a minimisation, a Hessian and an IRC all require the PES to be
single-valued. Bond-space drives keep the default, because that is what `data/`
did and what E08's `fresh_guess` row exists to measure.

---

## 4. Running it

```bash
uv sync --extra bench                # sella + geomeTRIC

# exercise every code path end to end in minutes, locally, before a submission
uv run python experiments/01_reference_states.py --smoke

uv run python experiments/registry.py                      # job counts
uv run python experiments/02_ts_single_ended.py --list      # one count
uv run python experiments/02_ts_single_ended.py --dry-run   # what would run

mkdir -p experiments/logs
sbatch --array=0-18 experiments/run.slurm 01_reference_states.py
sbatch --array=0-18 experiments/run.slurm 02_ts_single_ended.py
# ... see the header of run.slurm for every stage

uv run python experiments/10_failure_atlas.py --spin-check
uv run python experiments/analysis/aggregate.py
uv run python experiments/analysis/figures.py
```

**Run order: E01 → E02 → {E03, E04, E05} → {E06, E07} → E08 → E09 → E10.**
E01 is a hard prerequisite — every other TS experiment skips reactions whose
reference could not be verified, so running them first yields an empty study
rather than a wrong one.

**Gate on E01:** five of the nineteen are barrierless (`systems.BARRIERLESS`)
and have no saddle to find, so **14 is the ceiling**. If fewer than ~12 of those
14 verify, stop and reconsider the level of theory before spending cluster time
on E02–E06.

**E01 endpoints are separated asymptotes.** A reference R or P is fragmented,
pulled apart to 5 Å and *then* relaxed (`spectra.relax_asymptote`), so every
barrier here is an asymptotic barrier. This is not cosmetic. Relaxing a contact
pair as supplied lets it react: `rxn_11`'s dataset product is two OH radicals
with every atom 0.97–0.98 Å from a neighbour, and it relaxes into `H2O + O` —
whose energy comes back bit-identical to `rxn_04`'s product. The same effect on
IRC termini is what made `rxn_11` and `rxn_13` report saddles connecting the
wrong species. Each terminus records `label` (asymptotic, decides T4) and
`label_contact` (what it would have been called in contact), so the difference
is in the record rather than in a footnote.

### Flags

`--index N` / `--chunk K` (array selection; `SLURM_ARRAY_TASK_ID` is picked up
automatically) · `--list` · `--dry-run` · `--force` · `--force-failed` ·
`--smoke` · `--no-cache` · `--verify-cache F` · `--threads N`

`--smoke` normally picks its own job; combined with `--index N` it defers to the
index, so a chosen reaction can be exercised (`--smoke --index 2`). At a minimal
basis several reactions have no product well left, so nothing verifies — a smoke
run therefore accepts an **unverified** E01 reference, or every downstream script
would exercise only its skip path. Those records cannot leak into a result: the
level is named `*-smoke` and `inputs_hash` covers the level, so a production run
recomputes rather than resuming from them.

### Resume and caching

A job is skipped iff its record exists, its `inputs_hash` matches, its
`schema_version` matches, and its status is `ok`. The hash covers the job spec
and the level definition but **deliberately excludes the git commit**, so an
unrelated commit elsewhere in the repo does not invalidate a twelve-hour result;
the commit is still recorded.

Records are written atomically (`.tmp` + `os.replace`). SLURM preemption partway
through a write would otherwise leave a truncated file indistinguishable from a
completed job — the exact failure mode `data/1-run.py`'s skip-if-exists has today.

Trajectories are appended frame by frame, so a preempted job leaves a usable
partial trajectory. **Restarted runs are flagged and excluded from every timing
statistic**: FIRE2's velocity state is lost across a restart.

---

## 5. Results layout

```
experiments/results/
  runs/<experiment>/<key>.json    # RunRecord: the record of truth
  runs/<experiment>/<key>.xyz     # trajectories (extxyz keeps bond-order arrays)
  reference/<rxn>.xyz             # E01: relaxed R, verified TS, relaxed P
  reference/<rxn>-irc.xyz         # E01: the verified IRC
  network/<sector>/network.json   # E07
  failure_atlas.json              # E10
  tables/*.csv                    # aggregate.py output
  cache/*.npz                     # single-point cache, keyed on exact coordinates
experiments/figures/*.pdf
```

**Invariant:** experiment scripts write JSON and extxyz; `analysis/aggregate.py`
writes CSVs; figure scripts read CSVs and never compute anything. That is what
makes every figure regenerable in seconds during revisions, and what stops a
figure quietly disagreeing with its table.

Every record carries the git commit and dirty flag, package versions, hostname,
effective thread count, timings, counters, and — on failure — the **full
traceback**. A failed run writes a record too: silently missing data points are
how a benchmark lies about its denominator.

---

## 6. Known limitations, and what measures each

| Limitation | Measured by |
|---|---|
| **Fixed spin per run.** Six reactions (`rxn_05,06,07,08,09,15`) change multiplicity between endpoints; a single `spin` cannot describe both. Pre-registered in `systems.SPIN_NONCONSERVING` **before any run**, so the exclusion cannot look post-hoc. | E01 exclusions, E07 sectors, E10 class 6 |
| `BondFluxCalculator` energies are fictitious and are never plotted as energies. | E06 re-evaluates true energies at VERIFY |
| **Spectator freezing** under `restrict_gradient`: a skipped atom gets exactly zero gradient and, since no PES force is returned, is frozen for the whole run. | E08, E09 |
| **Mayer bond orders are basis-dependent.** If the target 0.5 means different geometries in different bases, the heuristic is soft. The single biggest scientific risk here. | E08 basis row |
| The dataset's level of theory is unknown and its geometries are not stationary points at ours. | E01 (dataset-vs-refined baseline) |
| A drive is an optimisation trajectory, **not** an IRC. | E06, E08 optimizer row |
| `MAX_DEGREE = {"H": 1, "O": 2}` in `network.py` is a hand-set chemical prior and belongs in the information ledger. | E07 records it |
| **`fmax = 0.1 eV/Å` may be the whole result.** `data/1-run.py`'s stopping criterion is looser than the constraint force on a *relaxed* reactant: measured 0.016 eV/Å on `rxn_03` while max\|ΔB\| was still 0.30, so FIRE2 exits at step **zero** and the endpoint is the start. Every such run is labelled `flat_gradient_stall` and is **not** counted as converged. | E08 `fmax` row, E10 class 3 |
| **Heavy-atom RMSD is degenerate at one heavy atom.** Kabsch superimposes a single point on any other exactly, so four of the nineteen reactions would report 0.000 Å for every structure ever compared. `permutation_rmsd(heavy_only=True)` falls back to all atoms below two heavy atoms. | `tests/test_experiments.py` |
| **The SCF guess is path-dependent** (`mo_coeff` carried between geometries), which makes the PES multi-valued along a trajectory. | E08 `fresh_guess` row; §3 above |
| n = 19. Wilson intervals are reported, never bare point estimates, and settings whose intervals overlap are not ordered. | throughout |

---

## 7. What would falsify this

- If the IDPP midpoint matches bond-space guesses on both RMSD and refinement
  cost (E04/B0b), the geometric claim is **empty** — interpolation was enough.
- If L1 collapses relative to L0 (E03), the method needs the product geometry and
  is **not single-ended**; the claim has to be rewritten.
- If τ\* scatters widely across reactions (E05), "half bonds" is a slogan rather
  than a rule, and the right target is system-dependent.
- If the flat-gradient stall class dominates E10, the method converges without
  reaching its targets and the reported success rate is an artifact of `fmax`.

---

## 8. Deviations from `data/`

- Files are `NN_name.py`, not `data/`'s `N-name.py`: `analysis/` and the tests
  must be able to `import` the experiment modules, and `1-run.py` is not an
  importable module name.
- The array index selects a **registry index**, not a reaction id. Most
  experiments have job counts unrelated to 19 (434 for the ablations). Ask
  `registry.py` for the counts rather than copying them from here.
- `data/HCombustion-paths/` is empty while `data/analysis.ipynb` reads
  `HCombustion-combined/`, so the existing 12/19 and 8/19 numbers are **not
  regenerable from the scripts on disk**. This suite supersedes rather than
  reconciles them.
- `configure_threads` / `available_cpus` moved to `bondspace.threads`;
  `data/util.py` re-exports them, so the `data/` pipeline is unchanged.
