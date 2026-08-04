"""Discover the hydrogen-combustion reaction network in bond space.

Breadth-first search over species.  Starting from H2 + O2, every state is
relaxed on the real PES, every elementary bond rearrangement available to
it is enumerated and driven with ``BondFluxCalculator``, and whatever
species the drive produces becomes a new node.  Unlike 0-prep.py/1-run.py,
which replay the known HCombustion reactions, nothing here is told what
the products are -- the network is the output, not the input.

Writes to HCombustion-network/:
    network.json          nodes, edges and run statistics
    states/<label>.xyz    the relaxed geometry adopted for each species
    edges/<n>.xyz         the bond-space trajectory driving each edge

Re-running resumes: states and edges already present in network.json are
not recomputed, so an interrupted search can be continued and the budget
raised incrementally.
"""

import json
from pathlib import Path

from ase import Atoms, io

from network import (
    BOND_THRESH,
    Move,
    check_spin,
    drive,
    enumerate_moves,
    relax,
    species_label,
)
from util import configure_threads

# Composition is fixed for the whole search, so charge and spin are too.
# H2 + O2 has 18 electrons; spin 2 follows triplet O2, which is the ground
# state of the seed and the entry point to the radical chain.
CHARGE = 0
SPIN = 2
BASIS = "cc-pvdz"
LEVEL_SHIFT = (0.3, 0.2)

# Only bonds targeted below this get the overlap repulsion that pushes
# dissociating fragments apart; bonds being formed (target 1.0) must not.
OVLP_THRESH = 0.5

MAX_STATES = 12  # stop after this many distinct species
MAX_DEPTH = 3  # BFS layers from the seed
# `drive` caps FIRE2's step at 0.05 A for stability, so a drive needs a larger
# budget than a PES relaxation does; the adjoint gradient it uses roughly
# halves the per-step cost, which pays for the extra steps.
DRIVE_STEPS = 80
RELAX_STEPS = 30


def seed() -> Atoms:
    """H2 and O2 side by side, far enough apart to be two fragments."""
    return Atoms(
        "HHOO",
        positions=[
            [0.00, 0.00, 0.00],
            [0.74, 0.00, 0.00],
            [0.00, 0.00, 2.60],
            [0.00, 0.00, 3.81],
        ],
    )


def slug(label: str) -> str:
    return label.replace(" + ", "_").replace(" ", "")


class Network:
    """Accumulated nodes and edges, checkpointed to disk after every edge."""

    def __init__(self, root: Path):
        self.root = root
        self.states = root / "states"
        self.edges_dir = root / "edges"
        self.file = root / "network.json"
        for d in (self.root, self.states, self.edges_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.nodes: dict[str, dict] = {}
        self.edges: list[dict] = []
        self.attempted: set[str] = set()
        if self.file.exists():
            data = json.loads(self.file.read_text())
            self.nodes = data["nodes"]
            self.edges = data["edges"]
            self.attempted = set(data["attempted"])

    def save(self) -> None:
        self.file.write_text(
            json.dumps(
                {
                    "nodes": self.nodes,
                    "edges": self.edges,
                    "attempted": sorted(self.attempted),
                    "settings": {
                        "charge": CHARGE,
                        "spin": SPIN,
                        "basis": BASIS,
                        "bond_thresh": BOND_THRESH,
                        "ovlp_thresh": OVLP_THRESH,
                    },
                },
                indent=2,
            )
        )

    def geometry(self, label: str) -> Atoms:
        return io.read(self.states / f"{slug(label)}.xyz")  # type: ignore[return-value]

    def add_node(self, label: str, atoms: Atoms, depth: int) -> bool:
        """Record a species. Returns True if it had not been seen before."""
        if label in self.nodes:
            return False
        io.write(self.states / f"{slug(label)}.xyz", atoms, format="extxyz")
        self.nodes[label] = {
            "label": label,
            "energy": atoms.info.get("energy"),
            "depth": depth,
            "relaxed": atoms.info.get("relaxed", False),
        }
        return True

    def add_edge(
        self,
        src: str,
        dst: str,
        move: Move,
        atoms: Atoms,
        images: list[Atoms],
        converged: bool,
    ) -> None:
        idx = len(self.edges)
        io.write(self.edges_dir / f"{idx:04d}.xyz", images, format="extxyz")
        e_src = self.nodes[src].get("energy")
        e_dst = self.nodes[dst].get("energy")
        self.edges.append(
            {
                "id": idx,
                "reactant": src,
                "product": dst,
                "move": move.describe(atoms),
                "kind": move.kind,
                "converged": converged,
                "steps": len(images),
                "delta_e": None if e_src is None or e_dst is None else e_dst - e_src,
            }
        )


def main() -> None:
    configure_threads()
    net = Network(Path(__file__).parent / "HCombustion-network")
    check_spin(seed(), CHARGE, SPIN)

    start = relax(
        seed(),
        charge=CHARGE,
        spin=SPIN,
        basis=BASIS,
        level_shift=LEVEL_SHIFT,
        steps=RELAX_STEPS,
    )
    root = species_label(start)
    net.add_node(root, start, depth=0)
    net.save()
    print("SEED:", root, flush=True)

    queue: list[tuple[str, int]] = [(root, 0)]
    while queue:
        label, depth = queue.pop(0)
        if depth >= MAX_DEPTH or len(net.nodes) >= MAX_STATES:
            continue

        atoms = net.geometry(label)
        moves = enumerate_moves(atoms)
        print(f"\n=== {label} (depth {depth}): {len(moves)} moves", flush=True)

        for move in moves:
            key = f"{label}|{move.describe(atoms)}"
            if key in net.attempted:
                continue
            net.attempted.add(key)

            print(f"  {move.describe(atoms)}", end=" ", flush=True)
            try:
                images, converged = drive(
                    atoms,
                    move,
                    charge=CHARGE,
                    spin=SPIN,
                    basis=BASIS,
                    level_shift=LEVEL_SHIFT,
                    thresh=BOND_THRESH,
                    ovlp_thresh=OVLP_THRESH,
                    steps=DRIVE_STEPS,
                )
            except Exception as exc:  # SCF blow-ups are expected on some moves
                print(f"-> FAILED ({type(exc).__name__}: {exc})", flush=True)
                net.save()
                continue

            if not images:
                print("-> no motion", flush=True)
                net.save()
                continue

            product = relax(
                images[-1],
                charge=CHARGE,
                spin=SPIN,
                basis=BASIS,
                level_shift=LEVEL_SHIFT,
                steps=RELAX_STEPS,
            )
            plabel = species_label(product)

            if plabel == label:
                print("-> unchanged", flush=True)
                net.save()
                continue

            if net.add_node(plabel, product, depth + 1):
                queue.append((plabel, depth + 1))
                print(f"-> {plabel} (NEW)", flush=True)
            else:
                print(f"-> {plabel}", flush=True)
            net.add_edge(label, plabel, move, atoms, images, converged)
            net.save()

    print(f"\n{len(net.nodes)} species, {len(net.edges)} reactions", flush=True)
    for e in net.edges:
        de = "" if e["delta_e"] is None else f"  dE = {e['delta_e']:+.2f} eV"
        print(f"  {e['reactant']:>16}  ->  {e['product']:<16} [{e['kind']}]{de}")


if __name__ == "__main__":
    main()
