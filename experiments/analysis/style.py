"""Shared figure style.  One definition, so the figures read as one system.

The categorical palette is fixed and **assigned in order, never cycled**: a
colour means an entity (a rung, a method), so filtering a figure down to fewer
series must not repaint the survivors.  The four hues were checked with an
all-pairs CVD test -- worst pair dE 9.5 (deutan), normal-vision floor 15.3, all
four above 3:1 against the surface -- so identity survives colourblind readers
and greyscale printing.  A fifth series is not a fifth hue: it becomes a facet
or folds into "other".

Every series is also given a marker and, where there are four or fewer, a
direct label.  Identity is never carried by colour alone.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

#: Fixed categorical order.  Do not reorder: figures across the paper rely on
#: a rung or a method keeping its colour.
CATEGORICAL = ("#0072B2", "#009E73", "#D55E00", "#785EF0")

#: Ordinal, for the T0-T4 ladder: one hue, light to dark, because the tiers
#: are ranked rather than distinct kinds.
SEQUENTIAL = ("#cfe3f2", "#9ec7e6", "#5ba3d0", "#2a7db5", "#0a4f7a")

#: Reserved for state, never reused as "series 5".
STATUS = {
    "good": "#009E73",
    "warning": "#E69F00",
    "critical": "#D55E00",
    "excluded": "#8c8c8c",
}

INK = "#1a1a1a"
INK_MUTED = "#6b6b6b"
GRID = "#dcdcdc"
SURFACE = "#ffffff"

MARKERS = ("o", "s", "^", "D", "v", "P", "X")

#: Colour by information rung, so "better" and "better informed" stay visually
#: distinguishable in the baseline comparison.
RUNG_COLOR = {"L0": CATEGORICAL[2], "L1": CATEGORICAL[0], "L2": CATEGORICAL[1]}
RUNG_LABEL = {
    "L0": "L0  reference geometries",
    "L1": "L1  chemical equation",
    "L2": "L2  reactant only",
}


def apply() -> None:
    """Publication defaults: recessive axes, thin marks, no chartjunk."""
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK_MUTED,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "text.color": INK,
    })


def save(fig, name: str) -> None:
    from pathlib import Path

    out = Path(__file__).parent.parent / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = out / f"{name}.{suffix}"
        fig.savefig(path)
    print(f"  wrote figures/{name}.pdf and .png")
    plt.close(fig)


def caption_exclusions(n_excluded: int, total: int) -> str:
    """The sentence every success-rate caption must carry.

    A rate whose denominator is not stated is not a rate.  Reactions excluded
    on the reference side are not method failures and must never be silently
    folded into either the numerator or the denominator.
    """
    return (
        f"n = {total - n_excluded} of {total} reactions; {n_excluded} excluded "
        "because E01 could not verify a reference transition state "
        "(reference-side failure, not a method failure)."
    )
