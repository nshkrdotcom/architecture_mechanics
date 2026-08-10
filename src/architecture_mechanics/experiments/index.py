"""One command that lists every run in the laboratory.

``04_SCIENCE_GATES.md`` names the gate this program does not have: there is no
mechanical defence against *selection over experiments* — running a candidate
five ways and reporting the one that worked. ``check_prereg.sh`` catches a
post-hoc claim about a given run; nothing catches a drawer of unreported ones.

The partial defence is that ``runs/`` is committed in full and that a reviewer
can see every run against its claim packet in one screen. This is that screen.
Prompts 10, 16, 24 and 29 would otherwise reconstruct it by hand from directory
listings, which is exactly the kind of task that gets done once and then
approximated.

    uv run python -m architecture_mechanics.experiments.index
    uv run python -m architecture_mechanics.experiments.index --json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from architecture_mechanics.experiments.manifest import lab_root

__all__ = ["RunRow", "index_runs", "main"]


class RunRow(dict):
    """One run's row. A dict so ``--json`` is the same object the table prints."""


def index_runs(lab: Path | str | None = None) -> list[RunRow]:
    """Every directory under ``runs/``, whether or not it is well formed.

    Directories with no manifest are listed too, with ``manifest`` false. A run
    that omitted its provenance is precisely the thing a reviewer wants to see,
    and an index that quietly skipped it would be helping to hide it.
    """
    lab = Path(lab or lab_root())
    runs = lab / "runs"
    rows: list[RunRow] = []
    if not runs.is_dir():
        return rows

    for directory in sorted(p for p in runs.iterdir() if p.is_dir()):
        manifest_path = directory / "manifest.json"
        summary_path = directory / "summary.json"
        manifest = _load(manifest_path)
        summary = _load(summary_path)
        metric = (manifest or {}).get("primary_metric") or "associative_recall_accuracy"
        rows.append(
            RunRow(
                run_id=directory.name,
                manifest=manifest is not None,
                rung=(manifest or {}).get("ladder_rung") or (summary or {}).get("config", {}).get("ladder"),
                architecture_id=(manifest or {}).get("architecture_id")
                or (summary or {}).get("config", {}).get("arch", {}).get("arch"),
                condition=(summary or {}).get("config", {}).get("data", {}).get("condition"),
                seed=(manifest or {}).get("seed") or (summary or {}).get("config", {}).get("seed"),
                primary_metric=metric,
                primary_value=(summary or {}).get("final", {}).get(metric),
                passed=(summary or {}).get("passed"),
                claim=(manifest or {}).get("parent_claim_packet"),
                git_commit=((manifest or {}).get("git_commit") or "")[:9] or None,
                dirty=(manifest or {}).get("dirty_tree"),
            )
        )
    return rows


def _load(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def render(rows: Sequence[RunRow]) -> str:
    if not rows:
        return "no runs recorded"
    header = f"{'rung':<5} {'run_id':<52} {'seed':<10} {'primary':>9}  {'ok':<3} claim"
    lines = [header, "-" * len(header)]
    for row in rows:
        value = row["primary_value"]
        shown = "     n/a" if value is None else f"{value:>9.4f}"
        verdict = {True: "yes", False: "NO", None: "—"}[row["passed"]]
        claim = row["claim"] or ("NO MANIFEST" if not row["manifest"] else "NONE")
        lines.append(
            f"{row['rung'] or '?'!s:<5} {row['run_id']:<52} {row['seed']!s:<10} "
            f"{shown}  {verdict:<3} {Path(claim).name if row['claim'] else claim}"
        )
    lines.append("")
    lines.append(f"{len(rows)} runs; primary metric per manifest, value from summary.json")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List every run against its claim packet.")
    parser.add_argument("--lab", default=None, help="laboratory root (default: this source tree's)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    rows = index_runs(args.lab)
    print(json.dumps(rows, indent=2) if args.json else render(rows))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
