"""The gate scripts are the specification; this holds the laboratory to them.

``bin/check_prereg.sh``, ``bin/check_claims.sh`` and ``bin/check_evidence.sh``
live outside this repository, in the program that commissioned it. They define
the exact shapes a manifest, a claim packet, and an evidence bundle must have.
Those shapes are restated in Python here — they have to be, because the runner
cannot import a bash script — and a restatement is a copy that will drift.

So this file reads the gates and fails when it does. If a later prompt adds a
required provenance field to ``check_evidence.sh``, the laboratory finds out
from its own test suite rather than from a red gate three missions later.

Skipped, not failed, when the program directory is not present: the laboratory
must remain testable on a machine that only has the laboratory.
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path

import pytest

from architecture_mechanics.experiments.claim_packet import REQUIRED_FIELDS, RUNGS
from architecture_mechanics.experiments.manifest import MACHINE_SIDE_FILES, PROVENANCE_FIELDS
from architecture_mechanics.reporting.evidence_bundle import (
    FINAL_DIRS,
    FINAL_FILES,
    SCREEN_FILES,
)

PROGRAM_DIR = Path(
    os.environ.get(
        "AM_PROGRAM_DIR", "/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260809/ml"
    )
)
BIN = PROGRAM_DIR / "bin"

pytestmark = pytest.mark.skipif(
    not BIN.is_dir(), reason=f"gate scripts not present at {BIN}"
)


def _list_literal(script: str, name: str) -> list[str]:
    """Pull a top-level ``NAME = [...]`` out of the python embedded in a gate."""
    text = (BIN / script).read_text()
    match = re.search(rf"^{name}\s*=\s*(\[[^\]]*\])", text, re.MULTILINE)
    assert match, f"{script} no longer defines {name}; the gate's shape changed"
    return list(ast.literal_eval(match.group(1)))


def test_provenance_fields_match_check_evidence():
    assert list(PROVENANCE_FIELDS) == _list_literal("check_evidence.sh", "PROVENANCE")


def test_full_bundle_matches_check_evidence():
    assert set(FINAL_FILES) == set(_list_literal("check_evidence.sh", "FULL"))
    assert set(FINAL_DIRS) == set(_list_literal("check_evidence.sh", "FULL_DIRS"))


def test_screen_bundle_covers_what_check_evidence_demands_of_a_screen():
    """A superset is allowed and is what we emit: the gate asks a screen for
    three files, and we add reproduce.sh because a screen nobody can re-run is a
    screen nobody can check."""
    demanded = set(_list_literal("check_evidence.sh", "SCREEN"))
    assert demanded <= set(SCREEN_FILES)
    assert set(SCREEN_FILES) - demanded == {"reproduce.sh"}


def test_machine_side_files_match_check_evidence():
    """Both sides skip the same files when verifying the evidence index. A gate
    that skipped one the runner still indexed would fail on every agreeing
    repeat; a runner that indexed one the gate skipped would record a digest
    nobody could ever check."""
    assert list(MACHINE_SIDE_FILES) == _list_literal("check_evidence.sh", "MACHINE_SIDE")


def test_check_evidence_verifies_the_evidence_index():
    """The index is a list of digests; a gate that never compares them records a
    promise it does not keep. Added by prompt 10 after six recorded manifests
    were found disagreeing with the bytes beside them."""
    text = (BIN / "check_evidence.sh").read_text()
    assert 'm.get("evidence_index")' in text
    assert "evidence_index digest does not match" in text


def test_pre_registration_fields_match_check_claims():
    assert list(REQUIRED_FIELDS) == _list_literal("check_claims.sh", "REQUIRED_PREREG_FIELDS")


def test_the_ladder_matches_check_claims():
    assert list(RUNGS) == _list_literal("check_claims.sh", "RUNGS")


def test_always_permitted_differences_match_check_no_rescue():
    """The one key a comparison may differ on without declaring it.

    ``experiments/comparison.py`` mirrors this set so it can refuse an unmatched
    comparison at construction time, before the gate could ever see it. Two
    copies of a rule drift; this is where the drift surfaces."""
    from architecture_mechanics.experiments.comparison import ALWAYS_PERMITTED

    text = (BIN / "check_no_rescue.sh").read_text()
    match = re.search(r"^ALWAYS_PERMITTED\s*=\s*(\{[^}]*\})", text, re.MULTILINE)
    assert match, "check_no_rescue.sh no longer defines ALWAYS_PERMITTED"
    assert set(ALWAYS_PERMITTED) == ast.literal_eval(match.group(1))


def test_check_no_rescue_reads_the_fields_a_declaration_writes():
    """The gate reads three keys out of every comparison file, plus the manifest's
    config block, and all of them are ours to supply. A gate that stopped reading
    ``permitted_differences`` would turn every declared exception into a silent
    one; a resolver that stopped writing it would do the same from the other
    side."""
    from architecture_mechanics.experiments.comparison import DECLARATION_FIELDS

    text = (BIN / "check_no_rescue.sh").read_text()
    read_by_the_gate = {"control_run", "candidate_runs", "permitted_differences"}
    for key in read_by_the_gate:
        assert f'c.get("{key}")' in text, key
    assert 'json.load(open(mp)).get("config")' in text
    assert read_by_the_gate <= set(DECLARATION_FIELDS)


def test_the_gate_directory_is_where_the_resolver_writes():
    """``check_no_rescue.sh`` globs one directory, non-recursively. If that path
    or that glob changed, every declaration this laboratory emits would stop
    being checked while every gate stayed green."""
    from architecture_mechanics.experiments.comparison import COMPARISONS_DIR, PLANNED_DIR

    text = (BIN / "check_no_rescue.sh").read_text()
    assert 'os.path.join(lab, "reports", "comparisons")' in text
    assert 'glob.glob(os.path.join(comparisons_dir, "*.json"))' in text
    assert COMPARISONS_DIR.as_posix() == "reports/comparisons"
    assert PLANNED_DIR.parent == COMPARISONS_DIR


def test_check_prereg_still_reads_the_fields_the_manifest_writes():
    """It reads two keys out of every manifest. Both are ours to supply."""
    text = (BIN / "check_prereg.sh").read_text()
    assert 'm.get("parent_claim_packet")' in text
    assert 'm.get("started_utc")' in text
    assert "parent_claim_packet" in PROVENANCE_FIELDS
    assert "started_utc" in PROVENANCE_FIELDS
