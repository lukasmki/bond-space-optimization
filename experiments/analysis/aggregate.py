"""Turn run records into tidy CSVs.  The only bridge between runs and figures.

The invariant the whole suite is built around: **experiment scripts write JSON,
this script writes CSVs, figure scripts read CSVs and never compute anything.**
That is what makes every figure regenerable in seconds during paper revisions,
on a laptop, without touching the cluster.

    uv run python experiments/analysis/aggregate.py
    uv run python experiments/analysis/aggregate.py --experiment 02_ts_single_ended
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

import common  # noqa: E402
import quality  # noqa: E402

TABLES = HERE.parent / "results" / "tables"

#: Columns lifted out of every record, whatever the experiment.
BASE_COLUMNS = (
    "experiment",
    "key",
    "status",
    "inputs_hash",
    "restarted",
    "wall_seconds",
    "hostname",
    "git_commit",
)

#: Nested or list-valued metrics that would make a CSV cell unreadable.  They
#: stay in the JSON, which is the record of truth; the CSV is for plotting.
SKIP = {
    "wavenumbers",
    "targets",
    "moves",
    "edges",
    "recall_table",
    "runs",
    "integer_disagreements",
    "profile_energies_ev",
    "reference_mayer",
    "drive_kwargs",
    "level",
    "irc_expected",
    "irc_observed",
    "nodes",
    "zvector_seconds_all",
    "direct_seconds_all",
    "restrict_seconds_all",
    "composition",
    "budget",
    "max_degree_prior",
    "per_pair",
}


def flatten(record: dict) -> dict:
    """One record -> one flat row."""
    provenance = record.get("provenance", {})
    row: dict[str, Any] = {
        "experiment": record.get("experiment"),
        "key": record.get("key"),
        "status": record.get("status"),
        "inputs_hash": record.get("inputs_hash"),
        "restarted": record.get("restarted"),
        "wall_seconds": record.get("timings", {}).get("wall_seconds"),
        "hostname": provenance.get("hostname"),
        "git_commit": (provenance.get("git") or {}).get("commit"),
    }
    for key, value in (record.get("metrics") or {}).items():
        if key in SKIP:
            continue
        if key == "tiers" and isinstance(value, dict):
            row.update({f"tier_{k}": v for k, v in value.items()})
            continue
        if isinstance(value, (dict, list)):
            continue
        row[key] = value
    for key, value in (record.get("counters") or {}).items():
        row[f"count_{key}"] = value
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    columns = list(BASE_COLUMNS)
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path.relative_to(HERE.parent)}  ({len(rows)} rows)")


def summarise_tiers(rows: list[dict]) -> list[dict]:
    """The T0-T4 attrition table, with Wilson intervals.

    Point estimates over 19 reactions invite comparisons that the sample size
    cannot support, so the interval is reported alongside every rate and the
    figure scripts draw it.
    """
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (row.get("experiment"), row.get("rung") or row.get("method") or "-")
        groups.setdefault(key, []).append(row)

    out = []
    for (experiment, group), members in sorted(
        groups.items(), key=lambda kv: str(kv[0])
    ):
        n = len(members)
        for tier in ("T0", "T1", "T2", "T3", "T4"):
            column = f"tier_{tier}"
            passed = sum(1 for m in members if m.get(column) in (True, "True"))
            low, high = quality.wilson_interval(passed, n)
            out.append(
                {
                    "experiment": experiment,
                    "group": group,
                    "tier": tier,
                    "passed": passed,
                    "trials": n,
                    "rate": passed / n if n else float("nan"),
                    "wilson_low": low,
                    "wilson_high": high,
                }
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment", default=None, help="aggregate only this experiment"
    )
    args = common.parse_args(parser)

    runs = common.RESULTS / "runs"
    if not runs.exists():
        print("no results yet -- run an experiment first")
        return

    names = (
        [args.experiment]
        if args.experiment
        else sorted(p.name for p in runs.iterdir() if p.is_dir())
    )

    all_rows: list[dict] = []
    print("aggregating:")
    for name in names:
        records = common.load_records(name)
        rows = [flatten(r) for r in records]
        write_csv(TABLES / f"{name}.csv", rows)
        all_rows.extend(rows)

    if all_rows:
        write_csv(TABLES / "all_runs.csv", all_rows)
        tiers = summarise_tiers(all_rows)
        write_csv(TABLES / "tier_summary.csv", tiers)

    atlas = common.RESULTS / "failure_atlas.json"
    if atlas.exists():
        data = json.loads(atlas.read_text())
        write_csv(
            TABLES / "failure_classes.csv",
            [
                {
                    "experiment": "10_failure_atlas",
                    "key": str(c),
                    "status": "ok",
                    "class_id": c,
                    "class_name": data["class_definitions"][str(c)],
                    "count": n,
                }
                for c, n in data["counts"].items()
            ],
        )


if __name__ == "__main__":
    main()
