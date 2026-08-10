"""Assembling and validating the §8.4 evidence bundle.

"The evidence bundle should be readable without the private notebook history."
That sentence is the requirement, and it has two halves that pull in opposite
directions.

**A screen is not evidence.** R0 to R2 emit a manifest, a metrics stream, and a
summary — and ``reproduce.sh``, because being able to re-run a screen is
provenance rather than a claim about it. They do *not* emit a geometry file or
an interventions file, because a screen measured neither and a directory that
looks like a pilot invites being read as one. ``bin/check_evidence.sh``
distinguishes them by ``ladder_rung`` and asks screens for less.

**A missing file and an empty file mean different things.** For R3 and above
every §8.4 artifact exists, and the ones with nothing in them yet — geometry
before prompt 07, interventions before prompt 19 — are written as valid, loadable,
*self-describing empty structures*. A reader who finds no ``interventions.jsonl``
learns nothing; a reader who finds one containing a schema header and zero
records learns that this run performed no interventions, which is a fact about
the run. Omission would let "we did not measure it" and "we measured it and it
was nothing" wear the same clothes.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "BUNDLE_SCHEMA",
    "FINAL_DIRS",
    "FINAL_FILES",
    "FINAL_RUNGS",
    "SCREEN_FILES",
    "BundleReport",
    "verify_bundle",
    "write_bundle",
    "write_reproduce_script",
]

BUNDLE_SCHEMA = "am.bundle.v1"

FINAL_RUNGS: tuple[str, ...] = ("R3", "R4", "R5")

SCREEN_FILES: tuple[str, ...] = ("manifest.json", "metrics.jsonl", "summary.json", "reproduce.sh")
"""What R0–R2 emit. The first three are what the gate demands of a screen;
``reproduce.sh`` is here because every run should be re-runnable, and a screen
whose result cannot be regenerated is a screen whose result cannot be checked."""

FINAL_FILES: tuple[str, ...] = SCREEN_FILES + (
    "mechanism_activity.json",
    "geometry_metrics.npz",
    "interventions.jsonl",
    "claim_gates.json",
)

FINAL_DIRS: tuple[str, ...] = ("figures", "checkpoint")

MECHANISM_SCHEMA = "am.mechanism_activity.v1"
GEOMETRY_SCHEMA = "am.geometry_metrics.v1"
INTERVENTIONS_SCHEMA = "am.interventions.v1"
FIGURES_SCHEMA = "am.figures_index.v1"
CHECKPOINT_SCHEMA = "am.checkpoint_index.v1"


@dataclass
class BundleReport:
    """What was written, and whether it is complete for the rung."""

    run_dir: Path
    ladder_rung: str
    is_final: bool
    files: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.problems


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #


def write_bundle(
    *,
    result,
    manifest,
    run_dir: Path | str,
    claim_gates: dict | None = None,
    model=None,
    lab_root: Path | str | None = None,
) -> BundleReport:
    """Emit the §8.4 bundle appropriate to this run's rung.

    ``summary.json``, ``metrics.jsonl`` and ``cost.json`` are already on disk —
    the runner writes them — so this adds the rest and then writes the manifest
    last, because the manifest indexes and hashes everything beside it.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    rung = str(manifest.ladder_rung).upper()
    is_final = rung in FINAL_RUNGS

    write_reproduce_script(
        run_dir=run_dir, manifest=manifest, lab_root=lab_root, primary_metric=manifest.primary_metric
    )

    if is_final:
        _write_mechanism_activity(run_dir, result, manifest)
        _write_geometry(run_dir, manifest)
        _write_interventions(run_dir, manifest)
        _write_claim_gates(run_dir, manifest, claim_gates)
        _write_figures_index(run_dir, manifest)
        manifest.checkpoint_hashes = _write_checkpoint(run_dir, manifest, model)

    manifest.finished_utc = manifest.finished_utc or _now()
    manifest.write(run_dir)

    report = BundleReport(run_dir=run_dir, ladder_rung=rung, is_final=is_final)
    report.problems = verify_bundle(run_dir)
    report.files = sorted(
        path.relative_to(run_dir).as_posix() for path in run_dir.rglob("*") if path.is_file()
    )
    return report


def _now() -> str:
    from architecture_mechanics.experiments.manifest import utc_now

    return utc_now()


def _write_mechanism_activity(run_dir: Path, result, manifest) -> None:
    """§6.3 activity, or an explicit statement that this rung captured none."""
    mechanism = getattr(result, "mechanism", None) or {}
    payload = {
        "schema": MECHANISM_SCHEMA,
        "run_id": manifest.run_id,
        "empty": not mechanism,
        "mechanism_activity_metric": "off_diagonal_mass, entropy_ratio, retrieval_lift",
        **({"reason": "this rung captured no mechanism activity"} if not mechanism else {}),
        **mechanism,
    }
    _write_json(run_dir / "mechanism_activity.json", payload)


def _write_geometry(run_dir: Path, manifest) -> None:
    """§6.2 geometry, as an npz that loads whether or not it holds anything.

    Prompt 07 fills this. Until then it is a real ``.npz`` carrying its schema
    and an ``empty`` flag, so ``np.load`` on any run in the laboratory succeeds
    and the answer to "was geometry measured here?" is in the file rather than
    in its absence.
    """
    path = run_dir / "geometry_metrics.npz"
    np.savez(
        path,
        __schema__=np.array(GEOMETRY_SCHEMA),
        __run_id__=np.array(manifest.run_id),
        __empty__=np.array(True),
        __written_by__=np.array("reporting.evidence_bundle; §6.2 metrics arrive in prompt 07"),
    )


def _write_interventions(run_dir: Path, manifest) -> None:
    """§6.4 interventions, as JSONL whose first record declares the schema.

    Zero interventions is written as a header and no records, not as a zero-byte
    file: the header is what tells a reader that the file is the interventions
    file and that it is genuinely empty.
    """
    header = {
        "record": "schema",
        "schema": INTERVENTIONS_SCHEMA,
        "run_id": manifest.run_id,
        "n_records": 0,
        "note": "no interventions were performed; §6.4 interventions arrive in prompt 19",
    }
    (run_dir / "interventions.jsonl").write_text(json.dumps(header) + "\n")


def _write_claim_gates(run_dir: Path, manifest, claim_gates: dict | None) -> None:
    """A copy of ``claims/<id>.gates.json``, so the bundle stands alone.

    The authoritative file lives beside the claim packet; this copy is here
    because §8.4 says a bundle is readable on its own, and "which rungs did this
    run support" is not answerable from a summary.
    """
    if claim_gates is None:
        from architecture_mechanics.experiments.claim_packet import GATES_SCHEMA

        claim_gates = {
            "schema": GATES_SCHEMA,
            "claim_id": Path(manifest.parent_claim_packet).stem,
            "rungs": {},
            "highest_supported_rung": None,
            "empty": True,
        }
    _write_json(run_dir / "claim_gates.json", claim_gates)


def _write_figures_index(run_dir: Path, manifest) -> None:
    """``figures/`` with an index that says how many figures there are.

    The gate requires the directory to be non-empty. It is made non-empty by an
    index listing zero figures, not by a placeholder image — a bundle should
    never contain an artifact that exists only to satisfy a check.
    """
    figures = run_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        path.name for path in figures.iterdir() if path.is_file() and path.name != "INDEX.json"
    )
    _write_json(
        figures / "INDEX.json",
        {
            "schema": FIGURES_SCHEMA,
            "run_id": manifest.run_id,
            "figures": existing,
            "empty": not existing,
        },
    )


def _write_checkpoint(run_dir: Path, manifest, model) -> dict:
    """The trained weights, plus a hashed index of them.

    Weights are gitignored — §8.3 asks for checkpoint *hashes*, and the hash is
    what makes "this manifest describes that checkpoint" checkable without
    putting 250 KB of tensors into git for every run.
    """
    from architecture_mechanics.experiments.manifest import file_digest

    directory = run_dir / "checkpoint"
    directory.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, dict] = {}

    if model is not None:
        import torch

        weights = directory / "model.pt"
        torch.save({"config": model.config.as_dict(), "state_dict": model.state_dict()}, weights)
        hashes["model.pt"] = {"sha256": file_digest(weights), "bytes": weights.stat().st_size}

    _write_json(
        directory / "checkpoint.json",
        {
            "schema": CHECKPOINT_SCHEMA,
            "run_id": manifest.run_id,
            "architecture_id": manifest.architecture_id,
            "parameter_count": manifest.parameter_count,
            "files": hashes,
            "empty": not hashes,
            "note": "weights are gitignored; the hashes here are the committed record",
        },
    )
    return hashes


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_default) + "\n")


def _default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


# --------------------------------------------------------------------------- #
# reproduce.sh
# --------------------------------------------------------------------------- #

REPRODUCE_TEMPLATE = """#!/usr/bin/env bash
# Regenerate {run_id} from its own recorded identity.
#
# Pins three things and re-runs:
#
#   source   git commit {commit}, exported with `git archive` — read-only, no
#            worktree is added and nothing in the laboratory is mutated;
#   config   the `config` block of manifest.json, handed back to the runner
#            verbatim, so options that never had a command-line flag still
#            reproduce;
#   seed     {seed}, which is inside that config.
#
# Then it checks itself: the regenerated summary.json must agree with the
# recorded one on the primary metric, or this script exits non-zero. A
# reproduction script that cannot fail is not evidence of anything.
#
#   ./reproduce.sh [output_dir]

set -euo pipefail

RUN_ID="{run_id}"
COMMIT="{commit}"
LAB="${{AM_LAB_DIR:-{lab}}}"
CLAIM="{claim}"
PRIMARY_METRIC="{primary_metric}"
LOCK_HASH="{lock_hash}"
SOURCE_TREE_HASH="{source_tree_hash}"

HERE="$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")" && pwd)"
OUT="${{1:-$(mktemp -d -t am-reproduce-XXXXXX)}}"
mkdir -p "$OUT"
OUT="$(cd -- "$OUT" && pwd)"

echo "reproducing $RUN_ID"
echo "  commit    $COMMIT"
echo "  lab       $LAB"
echo "  output    $OUT"

if [ ! -d "$LAB/.git" ]; then
  echo "FAIL: $LAB is not a git repository; set AM_LAB_DIR to the laboratory" >&2
  exit 2
fi
if ! git -C "$LAB" cat-file -e "$COMMIT^{{commit}}" 2>/dev/null; then
  echo "FAIL: commit $COMMIT is not in $LAB" >&2
  exit 2
fi

SRC="$OUT/source"
mkdir -p "$SRC"
git -C "$LAB" archive "$COMMIT" | tar -x -C "$SRC"

lock_now="$(sha256sum "$SRC/uv.lock" | cut -d' ' -f1)"
if [ "$lock_now" != "$LOCK_HASH" ]; then
  echo "FAIL: uv.lock at $COMMIT hashes $lock_now, manifest recorded $LOCK_HASH" >&2
  exit 3
fi

python3 - "$HERE/manifest.json" > "$OUT/config.json" <<'PY'
import json, sys
print(json.dumps(json.load(open(sys.argv[1]))["config"], indent=2))
PY

# The pinned source is put ahead of the installed package on PYTHONPATH, and the
# laboratory's own environment supplies the dependencies. Building a fresh
# environment here would reach the network, which no mission of this program is
# authorised to do, and would prove less: uv.lock is verified above, so the
# dependency set is already known to be the recorded one.
export PYTHONPATH="$SRC/src"
export AM_SOURCE_COMMIT="$COMMIT"
( cd "$LAB" && uv run --no-sync python -m architecture_mechanics.experiments.runner \\
    --config-json "$OUT/config.json" \\
    --claim "$CLAIM" \\
    --out "$OUT/runs" \\
    --emit-bundle \\
    --quiet )

NEW="$OUT/runs/$RUN_ID"
if [ ! -d "$NEW" ]; then
  echo "FAIL: expected $NEW; the run identity did not reproduce" >&2
  ls -1 "$OUT/runs" >&2
  exit 4
fi

python3 - "$HERE/summary.json" "$NEW/summary.json" "$PRIMARY_METRIC" "$SOURCE_TREE_HASH" \\
        "$HERE/manifest.json" "$NEW/manifest.json" <<'PY'
import json, sys

original, regenerated, metric, source_hash, m_old, m_new = sys.argv[1:7]
a, b = json.load(open(original)), json.load(open(regenerated))
ma, mb = json.load(open(m_old)), json.load(open(m_new))

rows, bad = [], 0
def row(name, x, y):
    global bad
    same = x == y
    bad += 0 if same else 1
    rows.append(f"  {{'ok  ' if same else 'DIFF'}} {{name}}: {{x}} vs {{y}}")

row(f"final.{{metric}}", a.get("final", {{}}).get(metric), b.get("final", {{}}).get(metric))
row("verdict.passed", a.get("passed"), b.get("passed"))
row("run_id", a.get("run_id"), b.get("run_id"))
row("manifest.source_tree_hash", source_hash, mb.get("source_tree_hash"))
row("manifest.split_hashes", ma.get("split_hashes"), mb.get("split_hashes"))
row("manifest.parameter_count", ma.get("parameter_count"), mb.get("parameter_count"))

print("\\n".join(rows))
if bad:
    print(f"\\nFAIL: {{bad}} field(s) did not reproduce")
    sys.exit(5)
print("\\nok   the primary metric and the run identity reproduced exactly")
PY

echo "regenerated run: $NEW"
"""


def write_reproduce_script(
    *, run_dir: Path, manifest, lab_root: Path | str | None, primary_metric: str
) -> Path:
    """Write an executable ``reproduce.sh`` for this run.

    Generated from the manifest rather than from the command line that happened
    to be typed, so a run launched from a script, a Makefile, or a test all
    produce the same reproduction procedure.
    """
    from architecture_mechanics.experiments import manifest as manifest_module

    lab = Path(lab_root or manifest_module.lab_root())
    text = REPRODUCE_TEMPLATE.format(
        run_id=manifest.run_id,
        commit=manifest.git_commit or "HEAD",
        lab=lab,
        claim=manifest.parent_claim_packet,
        primary_metric=primary_metric,
        lock_hash=manifest.dependency_lock_hash,
        source_tree_hash=manifest.source_tree_hash,
        seed=manifest.seed,
    )
    path = Path(run_dir) / "reproduce.sh"
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


# --------------------------------------------------------------------------- #
# Verification — the same rule bin/check_evidence.sh applies
# --------------------------------------------------------------------------- #


def verify_bundle(run_dir: Path | str) -> list[str]:
    """Every way this bundle falls short, in the gate's own terms.

    Duplicated from ``bin/check_evidence.sh`` deliberately. The gate is the
    authority and this is not allowed to relax it; having it here means an
    incomplete bundle fails at the moment it is written, when the cause is one
    function away, instead of at gate time when it is one mission away.
    ``tests/provenance/test_gate_agreement.py`` holds the two to each other.
    """
    from architecture_mechanics.experiments.manifest import PROVENANCE_FIELDS

    run_dir = Path(run_dir)
    problems: list[str] = []
    path = run_dir / "manifest.json"
    if not path.is_file():
        return ["missing manifest.json"]

    try:
        manifest = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        return [f"manifest does not parse ({error})"]

    rung = str(manifest.get("ladder_rung", "")).upper()
    is_final = rung in FINAL_RUNGS

    problems.extend(
        f"manifest missing provenance field: {name}"
        for name in PROVENANCE_FIELDS
        if manifest.get(name) in (None, "", [], {})
    )
    if manifest.get("dirty_tree") is True and is_final:
        problems.append("final run was produced from a dirty working tree")

    for name in FINAL_FILES if is_final else SCREEN_FILES:
        if not (run_dir / name).is_file():
            problems.append(f"missing {name}")

    if is_final:
        for name in FINAL_DIRS:
            directory = run_dir / name
            if not directory.is_dir():
                problems.append(f"missing {name}/")
            elif not any(directory.iterdir()):
                problems.append(f"{name}/ is empty")

    script = run_dir / "reproduce.sh"
    if script.is_file() and not os.access(script, os.X_OK):
        problems.append("reproduce.sh is not executable")

    return problems
