"""Deterministic job lists, one per experiment.

The SLURM array index is the only thing linking a task to its work, so these
lists must be **ordered and stable**: the same call must produce the same job
at the same index on every node and on every rerun.  Everything below sorts
explicitly and never iterates a set or an unordered dict.

`--list` on any experiment script prints `len(jobs)`, so array bounds are
computed rather than hardcoded and cannot drift out of sync with the registry.
"""

from __future__ import annotations


from common import JobSpec
from systems import ABLATION_SUBSET, load_set

#: E05's target sweep.  Symmetric values first, then the asymmetric variant.
TAU_VALUES = (0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70)

#: E08's one-at-a-time ablations.  Each entry is (knob, value) applied to the
#: E02 reference configuration; the reference itself is included as a control
#: so every sweep has a same-code baseline rather than a cross-script one.
ABLATIONS: tuple[tuple[str, object], ...] = (
    ("reference", None),
    ("ovlp_thresh", 0.0), ("ovlp_thresh", 0.25), ("ovlp_thresh", 1.0),
    ("ovlp_thresh", 2.0),
    ("thresh", 0.2), ("thresh", 0.1), ("thresh", 0.01), ("thresh", 1e-6),
    ("basis", "sto-3g"), ("basis", "6-31g*"), ("basis", "def2-svp"),
    ("basis", "cc-pvtz"),
    ("level_shift", 0.0), ("level_shift", 0.2), ("level_shift", (0.5, 0.4)),
    ("optimizer", "FIRE2-0.02"), ("optimizer", "FIRE2-0.1"),
    ("optimizer", "FIRE2-0.2"), ("optimizer", "BFGS"), ("optimizer", "LBFGS"),
    ("fresh_guess", True),
    ("restrict_gradient", True),
    ("zvector", False),
    ("density_fit", False),
    ("full_connectivity", True),
    ("steps", 40), ("steps", 80), ("steps", 150),
    # The knob that turned out to matter most.  data/1-run.py stops at
    # fmax = 0.1 eV/Ang, but the constraint force on a *relaxed* reactant is
    # already below that -- measured 0.016 on rxn_03 with max|dB| = 0.3 --
    # so FIRE2 exits at step zero and the endpoint is the start.  If the
    # success rate moves with this row, E02's headline is an artifact of a
    # convergence threshold and has to be reported as one.
    ("fmax", 0.05), ("fmax", 0.02), ("fmax", 0.01),
    ("perturb", 0.05), ("perturb", 0.10), ("perturb", 0.20),
)

#: Seeds for the perturbation ablation; ten displacements per sigma.
PERTURB_SEEDS = tuple(range(10))

#: E09's cost ladder.  Water clusters give a homologous series at fixed
#: element ratio; the alkanes vary the basis-function count per atom.  No
#: accuracy claim attaches to any of these -- this is a cost curve only.
SCALING_SYSTEMS = (
    "H2O-1", "H2O-2", "H2O-3", "H2O-4", "H2O-5", "H2O-6",
    "CH4", "C2H6", "C3H8", "C6H6", "C4H10",
)
SCALING_BASES = ("cc-pvdz", "cc-pvtz")
SCALING_MODES = ("zvector", "direct", "restrict")

#: E07's network sectors.  Composition -- and therefore spin -- is fixed for a
#: whole run, so covering the benchmark needs several sectors.  That is a
#: first-class finding about the method, not a workaround.
NETWORK_SEEDS = (
    ("h2_o2", "H2 + O2", 0, 2),
    ("h_o2", "H + O2", 0, 1),
    ("h2_o", "H2 + O", 0, 2),
    ("oh_oh", "OH + OH", 0, 0),
    ("h2o2", "H2O2", 0, 0),
)


def _reaction_ids() -> list[str]:
    return sorted(r.id for r in load_set())


def e01_reference_states() -> list[JobSpec]:
    return [
        JobSpec("01_reference_states", rid, {"reaction": rid})
        for rid in _reaction_ids()
    ]


def e02_ts_single_ended() -> list[JobSpec]:
    return [
        JobSpec("02_ts_single_ended", rid, {"reaction": rid, "rung": "L1"})
        for rid in _reaction_ids()
    ]


def e03_information_ladder() -> list[JobSpec]:
    jobs = []
    for rid in _reaction_ids():
        for rung in ("L0", "L1", "L2"):
            jobs.append(
                JobSpec(
                    "03_information_ladder",
                    f"{rid}-{rung}",
                    {"reaction": rid, "rung": rung},
                )
            )
    return jobs


def e04_ts_baselines() -> list[JobSpec]:
    jobs = []
    for rid in _reaction_ids():
        for method in ("B0a", "B0b", "B1", "B1r", "B2", "B3", "B4"):
            jobs.append(
                JobSpec(
                    "04_ts_baselines",
                    f"{rid}-{method}",
                    {"reaction": rid, "method": method},
                )
            )
    return jobs


def e05_target_sharpness() -> list[JobSpec]:
    jobs = []
    for rid in sorted(ABLATION_SUBSET):
        for tau in TAU_VALUES:
            for mode in ("symmetric", "asymmetric"):
                jobs.append(
                    JobSpec(
                        "05_target_sharpness",
                        f"{rid}-{mode}-{tau:.2f}",
                        {"reaction": rid, "tau": tau, "mode": mode},
                    )
                )
    return jobs


def e06_path_vs_irc() -> list[JobSpec]:
    jobs = []
    for rid in _reaction_ids():
        for method in ("bondspace", "linear", "idpp", "cineb"):
            jobs.append(
                JobSpec(
                    "06_path_vs_irc",
                    f"{rid}-{method}",
                    {"reaction": rid, "method": method},
                )
            )
    return jobs


def e07_network() -> list[JobSpec]:
    return [
        JobSpec("07_network_discovery", name,
                {"sector": name, "seed": formula, "charge": charge, "spin": spin})
        for name, formula, charge, spin in NETWORK_SEEDS
    ]


def e08_ablations() -> list[JobSpec]:
    jobs = []
    for rid in sorted(ABLATION_SUBSET):
        for knob, value in ABLATIONS:
            if knob == "perturb":
                for seed in PERTURB_SEEDS:
                    jobs.append(
                        JobSpec(
                            "08_ablations",
                            f"{rid}-perturb-{value:.2f}-{seed}",
                            {"reaction": rid, "knob": knob, "value": value,
                             "seed": seed},
                        )
                    )
            else:
                tag = "reference" if knob == "reference" else f"{knob}-{value}"
                jobs.append(
                    JobSpec(
                        "08_ablations",
                        f"{rid}-{tag}".replace(" ", ""),
                        {"reaction": rid, "knob": knob, "value": value},
                    )
                )
    return jobs


def e09_scaling() -> list[JobSpec]:
    jobs = []
    for system in SCALING_SYSTEMS:
        for basis in SCALING_BASES:
            jobs.append(
                JobSpec(
                    "09_scaling",
                    f"{system}-{basis}",
                    {"system": system, "basis": basis},
                )
            )
    return jobs


REGISTRY = {
    "01_reference_states": e01_reference_states,
    "02_ts_single_ended": e02_ts_single_ended,
    "03_information_ladder": e03_information_ladder,
    "04_ts_baselines": e04_ts_baselines,
    "05_target_sharpness": e05_target_sharpness,
    "06_path_vs_irc": e06_path_vs_irc,
    "07_network_discovery": e07_network,
    "08_ablations": e08_ablations,
    "09_scaling": e09_scaling,
}


def jobs_for(experiment: str) -> list[JobSpec]:
    return REGISTRY[experiment]()


if __name__ == "__main__":
    for name in sorted(REGISTRY):
        print(f"{name:26s} {len(jobs_for(name)):5d} jobs")
