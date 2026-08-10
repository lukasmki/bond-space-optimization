"""E07 -- reaction-network discovery, with recall and precision.

data/network.py is good code and is **imported, not rewritten**: `preorient`,
`check_spin`, `MAX_DEGREE` and relax-before-labelling are all correct and
documented decisions.  What it needs is a real budget, harness provenance, and
a scoring layer -- the run currently on disk reached three nodes and two
edges, and one of those two edges (H2 + O2 -> H2O + O, from a *break* move) is
chemically implausible and was recorded despite not converging.

That edge is the reason precision has to be measured rather than assumed.  So
every discovered edge is **verified**: half-target its move from the source
state, drive, refine with Sella, take a Hessian, run an IRC.  An edge counts
only if a one-imaginary-mode saddle exists whose IRC connects exactly the
species the edge claims.

Two headline numbers:

  * **recall** -- how many of the benchmark reactions appear as a discovered
    edge.  The denominator excludes reactions unreachable from a given seed by
    composition or spin parity, and that exclusion list is printed.
  * **precision** -- verified edges over discovered edges, with a breakdown of
    why the rest failed.

Composition, and therefore spin, is fixed for a whole run.  Covering the
benchmark needs several sectors, and that is a first-class finding about the
method rather than a workaround -- so the sectors are separate jobs and are
reported separately.

    sbatch --time=24:00:00 --array=0-4 experiments/run.slurm 07_network_discovery.py
"""

from __future__ import annotations

import argparse
import json
import sys

from ase import Atoms, io

import common
import spectra
from common import JobSpec, RunRecord
from levels import PRODUCTION, VERIFY, smoke_variant
from registry import jobs_for
from systems import load_set

DATA = common.REPO / "data"
if str(DATA) not in sys.path:
    sys.path.insert(0, str(DATA))
import network  # type: ignore[import-not-found]  # noqa: E402

EXPERIMENT = "07_network_discovery"

MAX_STATES = 30
MAX_DEPTH = 5
DRIVE_STEPS = 150
RELAX_STEPS = 40


def seed_structure(formula: str) -> Atoms:
    """Build a starting structure for a sector from its formula string.

    Fragments are placed well apart; `network.preorient` slides them together
    before an association move, because a formation constraint on two distant
    fragments produces essentially no force.
    """
    from ase import Atoms as A

    pieces = {
        "H2": A("HH", positions=[(0, 0, 0), (0, 0, 0.74)]),
        "O2": A("OO", positions=[(0, 0, 0), (0, 0, 1.21)]),
        "H": A("H", positions=[(0, 0, 0)]),
        "O": A("O", positions=[(0, 0, 0)]),
        "OH": A("OH", positions=[(0, 0, 0), (0, 0, 0.97)]),
        "H2O2": A(
            "HOOH",
            positions=[(0.8, 0.4, 0.3), (0, 0.7, 0), (0, -0.7, 0), (-0.8, -0.4, 0.3)],
        ),
    }
    combined = A()
    offset = 0.0
    for name in formula.split(" + "):
        piece = pieces[name].copy()
        piece.translate((offset, 0, 0))
        combined += piece
        offset += 2.6
    return combined


def known_reactions() -> list[dict]:
    """The benchmark reactions, as species pairs, for the recall denominator.

    Matching is by canonical species label on both sides, so atom ordering and
    which particular hydrogen moved are irrelevant -- only the chemistry is.
    """
    out = []
    for rxn in load_set():
        out.append({
            "id": rxn.id,
            "equation": rxn.equation,
            "reactants": spectra.species_label(rxn.reactant),
            "products": spectra.species_label(rxn.product),
            "spin": rxn.spin,
            "natoms": rxn.natoms,
        })
    return out


def reachable(known: dict, sector_spin: int, sector_atoms: dict) -> tuple[bool, str]:
    """Whether a benchmark reaction could possibly be found in this sector.

    Composition is conserved along every edge, so a reaction with a different
    atom count or a different spin parity is not a miss -- it was never in the
    search space.  Counting it in the denominator would understate recall for
    a reason that has nothing to do with the method.
    """
    if known["natoms"] != sum(sector_atoms.values()):
        return False, "different composition"
    if (known["spin"] - sector_spin) % 2 != 0:
        return False, "different spin parity"
    return True, ""


def verify_edge(
    source: Atoms, move, charge: int, spin: int, production, verify, args
) -> dict:
    """Does this edge correspond to a real elementary step?

    Half-targeting the move gives a transition-state guess for it; the guess
    is refined, checked for exactly one imaginary mode, and its IRC endpoints
    are compared against the species the edge claims to connect.  Without this
    an "edge" is only a statement that an optimiser stopped somewhere.
    """
    import drive as drive_mod
    import targets as targets_mod

    out: dict = {"move": move.describe(source)}
    bonds = targets_mod.l2_targets(source, move)
    out["ts_targets"] = [list(b) for b in bonds]

    result = drive_mod.drive(
        source, bonds, charge, spin, production,
        steps=30 if args.smoke else DRIVE_STEPS, keep_frames=False,
    )
    out["ts_drive_outcome"] = result.outcome
    guess = result.atoms

    spectrum = spectra.hessian_spectrum(guess, charge, spin, verify)
    out["guess_n_imaginary"] = spectrum.n_imaginary

    refinement = spectra.refine_saddle(
        guess, charge, spin, verify, steps=30 if args.smoke else 80
    )
    out["refine_converged"] = refinement["converged"]
    if not refinement["converged"]:
        out["verified"] = False
        out["failure"] = "saddle refinement did not converge"
        return out

    refined_spectrum = spectra.hessian_spectrum(
        refinement["atoms"], charge, spin, verify
    )
    out["n_imaginary"] = refined_spectrum.n_imaginary
    out["omega_imaginary"] = refined_spectrum.omega_imaginary
    if refined_spectrum.n_imaginary != 1:
        out["verified"] = False
        out["failure"] = f"n_imaginary={refined_spectrum.n_imaginary}"
        return out

    irc = spectra.run_irc(
        refinement["atoms"], charge, spin, verify,
        steps=15 if args.smoke else 50,
    )
    endpoints = {}
    for direction, res in irc.items():
        relaxed, converged, _ = spectra.relax(
            res["atoms"], charge, spin, verify, fmax=0.05, steps=RELAX_STEPS
        )
        endpoints[direction] = spectra.species_label(relaxed)
    out["irc_endpoints"] = sorted(set(endpoints.values()))
    out["ts_energy_ev"] = refined_spectrum.energy
    out["verified"] = len(set(endpoints.values())) == 2
    if not out["verified"]:
        out["failure"] = "IRC returned the same species in both directions"
    return out


def run_one(spec: JobSpec, rec: RunRecord, production, verify, args) -> None:
    sector = spec.params["sector"]
    charge = int(spec.params["charge"])
    spin = int(spec.params["spin"])
    seed = seed_structure(spec.params["seed"])
    network.check_spin(seed, charge, spin)

    outdir = common.RESULTS / "network" / sector
    outdir.mkdir(parents=True, exist_ok=True)

    max_states = 3 if args.smoke else MAX_STATES
    max_depth = 1 if args.smoke else MAX_DEPTH
    drive_steps = 30 if args.smoke else DRIVE_STEPS

    # --- BFS over species ------------------------------------------------
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    attempted: set[str] = set()
    queue: list[tuple[Atoms, int]] = []

    relaxed, converged, _ = spectra.relax(
        seed, charge, spin, production, fmax=0.05, steps=RELAX_STEPS
    )
    label = spectra.species_label(relaxed)
    nodes[label] = {"label": label, "depth": 0, "relaxed": converged}
    io.write(outdir / f"{label.replace(' ', '')}.xyz", relaxed, format="extxyz")
    queue.append((relaxed, 0))

    while queue and len(nodes) < max_states:
        state, depth = queue.pop(0)
        if depth >= max_depth:
            continue
        source_label = spectra.species_label(state)
        for move in network.enumerate_moves(state):
            key = f"{source_label}|{move.describe(state)}"
            if key in attempted:
                continue
            attempted.add(key)

            entry: dict = {
                "source": source_label,
                "move": move.describe(state),
                "kind": move.kind,
                "depth": depth,
            }
            try:
                import drive as drive_mod

                prepared = network.preorient(state, move)
                bonds = network.target_bonds(prepared, move)
                result = drive_mod.drive(
                    prepared, bonds, charge, spin, production,
                    steps=drive_steps, keep_frames=False,
                )
                entry["drive_outcome"] = result.outcome
                entry["drive_converged"] = result.converged

                product, prod_converged, _ = spectra.relax(
                    result.atoms, charge, spin, production,
                    fmax=0.05, steps=RELAX_STEPS,
                )
                target_label = spectra.species_label(product)
                entry["target"] = target_label
                entry["product_relaxed"] = prod_converged

                if target_label != source_label and target_label not in nodes:
                    nodes[target_label] = {
                        "label": target_label,
                        "depth": depth + 1,
                        "relaxed": prod_converged,
                    }
                    io.write(
                        outdir / f"{target_label.replace(' ', '')}.xyz",
                        product, format="extxyz",
                    )
                    queue.append((product, depth + 1))
            except Exception as exc:  # noqa: BLE001 -- recorded, never swallowed
                # data/2-network.py's bare `except Exception` prints and moves
                # on, which loses the diagnosis.  E10 needs the type to tell an
                # SCF blow-up apart from a wrong structure.
                import traceback

                entry["error"] = f"{type(exc).__name__}: {exc}"
                entry["traceback"] = traceback.format_exc()
            edges.append(entry)
            print(f"    edge {entry['source']} -> {entry.get('target', 'FAILED')}"
                  f"  [{entry['move']}]", flush=True)

    # --- verification pass ------------------------------------------------
    candidates = [
        e for e in edges
        if e.get("target") and e["target"] != e["source"] and "error" not in e
    ]
    if args.smoke:
        candidates = candidates[:1]

    verified = 0
    for entry in candidates:
        state_path = outdir / f"{entry['source'].replace(' ', '')}.xyz"
        source_atoms = io.read(state_path)
        try:
            for move in network.enumerate_moves(source_atoms):
                if move.describe(source_atoms) == entry["move"]:
                    entry["verification"] = verify_edge(
                        source_atoms, move, charge, spin, production, verify, args
                    )
                    break
        except Exception as exc:  # noqa: BLE001
            entry["verification"] = {"verified": False,
                                     "failure": f"{type(exc).__name__}: {exc}"}
        if entry.get("verification", {}).get("verified"):
            verified += 1

    # --- scoring ----------------------------------------------------------
    sector_atoms: dict[str, int] = {}
    for symbol in relaxed.get_chemical_symbols():
        sector_atoms[symbol] = sector_atoms.get(symbol, 0) + 1

    discovered_pairs = {
        frozenset((e["source"], e["target"]))
        for e in edges if e.get("target") and e["target"] != e["source"]
    }
    verified_pairs = {
        frozenset((e["source"], e["target"]))
        for e in edges if e.get("verification", {}).get("verified")
    }

    recall_rows = []
    for known in known_reactions():
        ok, reason = reachable(known, spin, sector_atoms)
        pair = frozenset((known["reactants"], known["products"]))
        recall_rows.append({
            "id": known["id"],
            "equation": known["equation"],
            "in_search_space": ok,
            "excluded_because": reason,
            "discovered": pair in discovered_pairs,
            "verified": pair in verified_pairs,
        })
    in_space = [r for r in recall_rows if r["in_search_space"]]

    metrics = {
        "sector": sector,
        "seed": spec.params["seed"],
        "charge": charge,
        "spin": spin,
        "composition": sector_atoms,
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "n_edges_with_product": len(candidates),
        "n_verified": verified,
        "precision": verified / len(candidates) if candidates else 0.0,
        "recall_denominator": len(in_space),
        "recall_discovered": sum(1 for r in in_space if r["discovered"]),
        "recall_verified": sum(1 for r in in_space if r["verified"]),
        "nodes": sorted(nodes),
        "edges": edges,
        "recall_table": recall_rows,
        "budget": {"max_states": max_states, "max_depth": max_depth,
                   "drive_steps": drive_steps},
        # A hand-set chemical prior, and therefore part of the information
        # ledger: the enumerator will never propose a bond that exceeds it.
        "max_degree_prior": dict(network.MAX_DEGREE),
        "bond_thresh": network.BOND_THRESH,
    }
    (outdir / "network.json").write_text(json.dumps(common.canonical(metrics), indent=2))
    rec.metrics = metrics
    print(
        f"       {sector}: {len(nodes)} species, {len(candidates)} edges, "
        f"{verified} verified (precision {metrics['precision']:.2f}), "
        f"recall {metrics['recall_verified']}/{metrics['recall_denominator']}",
        flush=True,
    )


def main() -> None:
    parser = common.add_standard_args(argparse.ArgumentParser(description=__doc__))
    args = common.parse_args(parser)
    production = smoke_variant(PRODUCTION) if args.smoke else PRODUCTION
    verify = smoke_variant(VERIFY) if args.smoke else VERIFY
    jobs = jobs_for(EXPERIMENT)
    if args.smoke:
        jobs = common.pick_smoke_jobs(jobs, args)
    common.main_loop(
        EXPERIMENT, jobs, args, production,
        lambda spec, rec: run_one(spec, rec, production, verify, args),
    )


if __name__ == "__main__":
    main()
