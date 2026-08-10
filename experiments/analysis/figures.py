"""Every figure in the paper.  Reads CSVs only -- never runs a calculation.

That restriction is the point: during paper revisions the figures regenerate
in seconds on a laptop, and no figure can quietly disagree with the table it
was supposed to come from.

    uv run python experiments/analysis/aggregate.py   # first
    uv run python experiments/analysis/figures.py     # then
    uv run python experiments/analysis/figures.py --only fig1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import style  # noqa: E402

TABLES = HERE.parent / "results" / "tables"
RESULTS = HERE.parent / "results"


def read(name: str) -> list[dict]:
    path = TABLES / f"{name}.csv"
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def num(row: dict, key: str, default=np.nan) -> float:
    value = row.get(key, "")
    if value in ("", None, "None"):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def flag(row: dict, key: str) -> bool:
    return str(row.get(key, "")).strip() in ("True", "true", "1")


def _wilson_bars(ax, xs, rates, lows, highs, color, label, marker):
    lower = np.clip(np.array(rates) - np.array(lows), 0, None)
    upper = np.clip(np.array(highs) - np.array(rates), 0, None)
    ax.errorbar(
        xs, rates, yerr=[lower, upper], color=color, marker=marker,
        capsize=3, elinewidth=1.2, label=label, linestyle="-",
    )


# --------------------------------------------------------------------------
# Figure 1 -- the tier ladder (E02)
# --------------------------------------------------------------------------


def fig1() -> None:
    import matplotlib.pyplot as plt

    rows = [r for r in read("02_ts_single_ended") if r.get("status") == "ok"]
    if not rows:
        print("  fig1: no E02 results yet")
        return
    excluded = [r for r in read("02_ts_single_ended") if r.get("status") == "skipped"]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))

    # (a) attrition down the ladder.  The whole argument of the paper is in
    # the shape of this bar chart: how much of the reported success survives
    # leaving bond space.
    tiers = ["T0", "T1", "T2", "T3", "T4"]
    counts = [sum(1 for r in rows if flag(r, f"tier_{t}")) for t in tiers]
    ax = axes[0]
    bars = ax.bar(tiers, counts, color=style.SEQUENTIAL, width=0.7)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, count + 0.15, str(count),
                ha="center", va="bottom", fontsize=8, color=style.INK)
    ax.set_ylabel("reactions passing")
    ax.set_ylim(0, len(rows) + 1.5)
    ax.set_title("(a) attrition down the ladder", loc="left")
    ax.set_xlabel(
        "T0 bond error (ref pairs) · T1 all pairs · T2 RMSD\n"
        "T3 one imaginary mode · T4 IRC connects R and P",
        fontsize=7, color=style.INK_MUTED,
    )

    # (b) is it better than interpolating?  The diagonal is the whole test.
    ax = axes[1]
    bs = [num(r, "rmsd_heavy") for r in rows]
    mid = [num(r, "midpoint_rmsd_heavy") for r in rows]
    saddle = [num(r, "n_imaginary") == 1 for r in rows]
    limit = max([v for v in bs + mid if np.isfinite(v)] + [0.5]) * 1.15
    ax.plot([0, limit], [0, limit], color=style.INK_MUTED, lw=0.8, ls="--", zorder=1)
    for is_saddle, marker, label in (
        (True, "o", "one imaginary mode"),
        (False, "x", "not a first-order saddle"),
    ):
        sel = [i for i, s in enumerate(saddle) if s == is_saddle]
        if sel:
            ax.scatter(
                [mid[i] for i in sel], [bs[i] for i in sel],
                color=style.CATEGORICAL[0], marker=marker, s=34,
                edgecolor=style.SURFACE, linewidth=0.8, label=label, zorder=3,
            )
    ax.set_xlabel("midpoint guess, heavy-atom RMSD to TS (Å)")
    ax.set_ylabel("bond space (Å)")
    ax.set_title("(b) versus the null baseline", loc="left")
    ax.text(limit * 0.55, limit * 0.9, "bond space worse",
            fontsize=7, color=style.INK_MUTED)
    ax.text(limit * 0.5, limit * 0.12, "bond space better",
            fontsize=7, color=style.INK_MUTED)
    ax.legend(loc="lower right")

    # (c) is the mode you got the mode you asked for?
    ax = axes[2]
    overlaps = [num(r, "mode_overlap_bonds") for r in rows]
    overlaps = [v for v in overlaps if np.isfinite(v)]
    if overlaps:
        ax.hist(overlaps, bins=np.linspace(0, 1, 11),
                color=style.CATEGORICAL[1], edgecolor=style.SURFACE, linewidth=1.2)
    ax.set_xlabel("|cos| (imaginary mode, requested bond direction)")
    ax.set_ylabel("reactions")
    ax.set_title("(c) the requested motion?", loc="left")

    fig.text(
        0.0, -0.10, style.caption_exclusions(len(excluded), len(rows) + len(excluded)),
        fontsize=7, color=style.INK_MUTED,
    )
    fig.tight_layout()
    style.save(fig, "fig1_tier_ladder")


# --------------------------------------------------------------------------
# Figure 2 -- the information ladder (E03)
# --------------------------------------------------------------------------


def fig2() -> None:
    import matplotlib.pyplot as plt

    summary = [r for r in read("tier_summary")
               if r.get("experiment") == "03_information_ladder"]
    if not summary:
        print("  fig2: no E03 results yet")
        return

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))
    tiers = ["T0", "T1", "T2", "T3", "T4"]

    ax = axes[0]
    for index, rung in enumerate(("L0", "L1", "L2")):
        rows = {r["tier"]: r for r in summary if r["group"] == rung}
        if not rows:
            continue
        rates = [num(rows[t], "rate") if t in rows else np.nan for t in tiers]
        lows = [num(rows[t], "wilson_low") if t in rows else np.nan for t in tiers]
        highs = [num(rows[t], "wilson_high") if t in rows else np.nan for t in tiers]
        _wilson_bars(
            ax, np.arange(len(tiers)) + (index - 1) * 0.08, rates, lows, highs,
            style.RUNG_COLOR[rung], style.RUNG_LABEL[rung], style.MARKERS[index],
        )
    ax.set_xticks(range(len(tiers)))
    ax.set_xticklabels(tiers)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction of reactions")
    ax.set_title("(a) success by information rung", loc="left")
    ax.legend(loc="upper right")

    # (b) L2's screening behaviour: finding the right saddle among many is
    # only useful if it ranks near the top.
    ax = axes[1]
    l2 = [r for r in read("03_information_ladder")
          if r.get("rung") == "L2" and r.get("status") == "ok"]
    if l2:
        ranks = [num(r, "true_ts_rank_by_barrier") for r in l2]
        found = [r for r in ranks if np.isfinite(r)]
        missed = len(ranks) - len(found)
        if found:
            ax.hist(found, bins=np.arange(0.5, max(found) + 1.5),
                    color=style.RUNG_COLOR["L2"], edgecolor=style.SURFACE,
                    linewidth=1.2)
        ax.set_xlabel("rank of the true TS among discovered saddles (by barrier)")
        ax.set_ylabel("reactions")
        ax.set_title(f"(b) L2 screening — {missed} not found at all", loc="left")
    fig.tight_layout()
    style.save(fig, "fig2_information_ladder")


# --------------------------------------------------------------------------
# Figure 3 -- accuracy vs cost (E04)
# --------------------------------------------------------------------------


def fig3() -> None:
    import matplotlib.pyplot as plt

    rows = [r for r in read("04_ts_baselines") if r.get("status") == "ok"]
    if not rows:
        print("  fig3: no E04 results yet")
        return

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for index, (method, members) in enumerate(sorted(grouped.items())):
        rung = members[0].get("rung", "L0")
        rate = np.mean([1.0 if flag(m, "tier_T4") else 0.0 for m in members])
        cost = np.median([num(m, "total_pes_calls") for m in members])
        if not np.isfinite(cost) or cost <= 0:
            cost = 1.0
        low, high = _wilson(sum(flag(m, "tier_T4") for m in members), len(members))
        ax.errorbar(
            cost, rate, yerr=[[rate - low], [high - rate]],
            color=style.RUNG_COLOR.get(rung, style.INK_MUTED),
            marker=style.MARKERS[index % len(style.MARKERS)],
            markersize=8, capsize=3, elinewidth=1.0, linestyle="none",
        )
        ax.annotate(
            f"{method}  {members[0].get('label', '')}",
            (cost, rate), textcoords="offset points", xytext=(8, 4),
            fontsize=7.5, color=style.INK,
        )

    ax.set_xscale("log")
    ax.set_xlabel("median true-PES energy+gradient evaluations (guess + refinement)")
    ax.set_ylabel("T4 verified success rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Accuracy against cost, coloured by information used", loc="left")

    handles = [
        plt.Line2D([], [], color=style.RUNG_COLOR[r], marker="o", linestyle="none",
                   label=style.RUNG_LABEL[r])
        for r in ("L0", "L1", "L2")
    ]
    ax.legend(handles=handles, loc="lower right")
    fig.tight_layout()
    style.save(fig, "fig3_baseline_pareto")


def _wilson(successes: int, trials: int) -> tuple[float, float]:
    import quality

    return quality.wilson_interval(successes, trials)


# --------------------------------------------------------------------------
# Figure 4 -- is 0.5 special? (E05)
# --------------------------------------------------------------------------


def fig4() -> None:
    import matplotlib.pyplot as plt

    rows = [r for r in read("05_target_sharpness") if r.get("status") == "ok"]
    rows = [r for r in rows if r.get("mode") == "symmetric"]
    if not rows:
        print("  fig4: no E05 results yet")
        return

    by_reaction: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_reaction[row["reaction"]].append(row)

    n = len(by_reaction)
    cols = min(4, n)
    rows_n = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(2.6 * cols, 2.4 * rows_n),
                             squeeze=False)

    for index, (reaction, members) in enumerate(sorted(by_reaction.items())):
        ax = axes[index // cols][index % cols]
        members.sort(key=lambda r: num(r, "tau"))
        taus = [num(r, "tau") for r in members]
        rmsd = [num(r, "rmsd_heavy") for r in members]
        ax.plot(taus, rmsd, color=style.CATEGORICAL[0], marker="o")
        ax.axvline(0.5, color=style.INK_MUTED, lw=0.8, ls=":")
        # Where the verified saddle's own Mayer orders actually sit.  If this
        # rule is far from 0.5, the heuristic is asking for the wrong structure
        # and no optimiser setting will fix it.
        observed = num(members[0], "reference_mayer_mean_fraction")
        if np.isfinite(observed):
            ax.axvline(observed, color=style.CATEGORICAL[2], lw=1.2)
        ax.set_title(reaction, loc="left", fontsize=8)
        if index % cols == 0:
            ax.set_ylabel("RMSD to TS (Å)")
        if index // cols == rows_n - 1:
            ax.set_xlabel("target fraction τ")

    for index in range(n, rows_n * cols):
        axes[index // cols][index % cols].axis("off")

    handles = [
        plt.Line2D([], [], color=style.INK_MUTED, ls=":", label="τ = 0.5 (the heuristic)"),
        plt.Line2D([], [], color=style.CATEGORICAL[2], label="reference TS Mayer order"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    style.save(fig, "fig4_target_sharpness")


# --------------------------------------------------------------------------
# Figure 5 -- path fidelity (E06)
# --------------------------------------------------------------------------


def fig5() -> None:
    import matplotlib.pyplot as plt

    rows = [r for r in read("06_path_vs_irc") if r.get("status") == "ok"]
    if not rows:
        print("  fig5: no E06 results yet")
        return

    methods = ["bondspace", "idpp", "linear", "cineb"]
    colors = dict(zip(methods, style.CATEGORICAL))

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))

    ax = axes[0]
    data, labels, used = [], [], []
    for method in methods:
        values = [num(r, "tube_max") for r in rows if r["method"] == method]
        values = [v for v in values if np.isfinite(v)]
        if values:
            data.append(values)
            labels.append(method)
            used.append(colors[method])
    if data:
        box = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.55)
        for patch, color in zip(box["boxes"], used):
            patch.set_facecolor(color)
            patch.set_edgecolor(style.SURFACE)
            patch.set_linewidth(1.5)
        for element in ("medians", "whiskers", "caps"):
            for item in box[element]:
                item.set_color(style.INK)
                item.set_linewidth(1.0)
    ax.set_ylabel("max distance from reference IRC (Å)")
    ax.set_title("(a) how far the path strays", loc="left")

    ax = axes[1]
    for method in methods:
        values = [num(r, "overshoot_kcal") for r in rows if r["method"] == method]
        values = [v for v in values if np.isfinite(v)]
        if not values:
            continue
        ax.scatter(
            [method] * len(values), values, color=colors[method],
            s=32, edgecolor=style.SURFACE, linewidth=0.8, label=method,
        )
    ax.axhline(0, color=style.INK_MUTED, lw=0.8, ls="--")
    ax.set_ylabel("peak energy − reference barrier (kcal/mol)")
    ax.set_title("(b) how far above the true barrier", loc="left")

    fig.tight_layout()
    style.save(fig, "fig5_path_fidelity")


# --------------------------------------------------------------------------
# Figure 6 -- the discovered network (E07)
# --------------------------------------------------------------------------


def fig6() -> None:
    import matplotlib.pyplot as plt

    sectors = sorted((RESULTS / "network").glob("*/network.json")) if (
        RESULTS / "network"
    ).exists() else []
    if not sectors:
        print("  fig6: no E07 results yet")
        return

    try:
        import networkx as nx
    except ImportError:
        print("  fig6: networkx not available")
        return

    fig, axes = plt.subplots(
        1, len(sectors), figsize=(4.2 * len(sectors), 4.0), squeeze=False
    )
    for index, path in enumerate(sectors):
        data = json.loads(path.read_text())
        ax = axes[0][index]
        graph = nx.Graph()
        known_pairs = {
            frozenset((r["equation"].split(" -> ")[0], r["equation"].split(" -> ")[1]))
            for r in data.get("recall_table", []) if r.get("discovered")
        }
        for node in data.get("nodes", []):
            graph.add_node(node)
        edge_colors = []
        for edge in data.get("edges", []):
            target = edge.get("target")
            if not target or target == edge["source"]:
                continue
            graph.add_edge(edge["source"], target)
            verified = edge.get("verification", {}).get("verified")
            known = frozenset((edge["source"], target)) in known_pairs
            edge_colors.append(
                style.STATUS["good"] if verified and known
                else style.STATUS["warning"] if verified
                else style.STATUS["excluded"]
            )
        pos = nx.spring_layout(graph, seed=0)
        nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=edge_colors, width=1.8)
        nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=style.CATEGORICAL[0],
                               node_size=180, edgecolors=style.SURFACE)
        nx.draw_networkx_labels(graph, pos, ax=ax, font_size=6.5,
                                font_color=style.INK)
        ax.set_title(
            f"{data['sector']}  ·  precision {data['precision']:.2f}  ·  "
            f"recall {data['recall_verified']}/{data['recall_denominator']}",
            loc="left", fontsize=8,
        )
        ax.axis("off")

    handles = [
        plt.Line2D([], [], color=style.STATUS["good"], lw=2,
                   label="verified and in the benchmark"),
        plt.Line2D([], [], color=style.STATUS["warning"], lw=2,
                   label="verified, not in the benchmark"),
        plt.Line2D([], [], color=style.STATUS["excluded"], lw=2,
                   label="discovered but unverified"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    style.save(fig, "fig6_network")


# --------------------------------------------------------------------------
# Figure 7 -- scaling (E09)
# --------------------------------------------------------------------------


def fig7() -> None:
    import matplotlib.pyplot as plt

    rows = [r for r in read("09_scaling") if r.get("status") == "ok"]
    if not rows:
        print("  fig7: no E09 results yet")
        return

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))
    modes = [("zvector_seconds_median", "Z-vector (adjoint)"),
             ("direct_seconds_median", "direct"),
             ("restrict_seconds_median", "restrict_gradient")]

    for basis, axis in zip(sorted({r["basis"] for r in rows}), axes):
        subset = sorted((r for r in rows if r["basis"] == basis),
                        key=lambda r: num(r, "natoms"))
        n = [num(r, "natoms") for r in subset]
        for index, (column, label) in enumerate(modes):
            seconds = [num(r, column) for r in subset]
            pairs = [(a, b) for a, b in zip(n, seconds) if np.isfinite(b)]
            if not pairs:
                continue
            axis.plot(
                [p[0] for p in pairs], [p[1] for p in pairs],
                color=style.CATEGORICAL[index], marker=style.MARKERS[index],
                label=label,
            )
        # The cost of an ordinary DFT gradient, so the reader knows a
        # bond-space step costs k of them and how k grows.
        plain = [(num(r, "natoms"), num(r, "energy_gradient_seconds"))
                 for r in subset]
        plain = [(a, b) for a, b in plain if np.isfinite(b)]
        if plain:
            axis.plot([p[0] for p in plain], [p[1] for p in plain],
                      color=style.INK_MUTED, ls="--", lw=1.2,
                      label="plain DFT energy+gradient")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("atoms")
        axis.set_title(basis, loc="left")
    axes[0].set_ylabel("seconds per force evaluation")
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    style.save(fig, "fig7_scaling")


# --------------------------------------------------------------------------
# Figure 8 -- ablation sensitivity (E08)
# --------------------------------------------------------------------------


def fig8() -> None:
    import matplotlib.pyplot as plt

    rows = [r for r in read("08_ablations") if r.get("status") == "ok"]
    if not rows:
        print("  fig8: no E08 results yet")
        return

    baseline = [num(r, "rmsd_heavy") for r in rows if r.get("knob") == "reference"]
    if not baseline:
        print("  fig8: no reference configuration runs")
        return
    reference = float(np.nanmedian(baseline))

    deltas: dict[str, float] = {}
    for row in rows:
        knob = row.get("knob")
        if knob in (None, "reference"):
            continue
        label = f"{knob} = {row.get('value')}"
        deltas.setdefault(label, [])
        deltas[label].append(num(row, "rmsd_heavy"))
    summary = sorted(
        ((k, float(np.nanmedian(v)) - reference) for k, v in deltas.items()),
        key=lambda kv: abs(kv[1]),
    )

    fig, ax = plt.subplots(figsize=(6.4, max(3.0, 0.22 * len(summary))))
    labels = [k for k, _ in summary]
    values = [v for _, v in summary]
    colors = [
        style.STATUS["critical"] if v > 0 else style.STATUS["good"] for v in values
    ]
    ax.barh(labels, values, color=colors, height=0.65)
    ax.axvline(0, color=style.INK_MUTED, lw=0.9)
    ax.set_xlabel("change in median heavy-atom RMSD to TS vs the reference "
                  "configuration (Å)")
    ax.set_title("Sensitivity to each knob (worse to the right)", loc="left")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    style.save(fig, "fig8_ablation_tornado")


FIGURES = {
    "fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4,
    "fig5": fig5, "fig6": fig6, "fig7": fig7, "fig8": fig8,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default=None, choices=sorted(FIGURES))
    args = parser.parse_args()
    style.apply()
    names = [args.only] if args.only else sorted(FIGURES)
    print("figures:")
    for name in names:
        FIGURES[name]()


if __name__ == "__main__":
    main()
