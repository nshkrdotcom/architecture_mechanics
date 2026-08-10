"""The ``--selftest`` gate: it passes, and it is seen to fail.

A gate nobody has watched fail is not known to be a gate. ``--break-invariant``
exists for exactly this: it forces one named check to report failure so the
non-zero exit path is exercised by the suite rather than asserted in prose.
"""

from __future__ import annotations

import re
import subprocess
import sys

import pytest

from architecture_mechanics.data.feature_program import INVARIANTS

MODULE = "architecture_mechanics.data.feature_program"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", MODULE, *args],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )


@pytest.fixture(scope="module")
def passing_run() -> subprocess.CompletedProcess:
    return _run("--selftest")


def test_selftest_passes(passing_run):
    assert passing_run.returncode == 0, passing_run.stdout + passing_run.stderr
    assert "selftest PASSED" in passing_run.stdout


def test_selftest_checks_every_declared_invariant(passing_run):
    reported = set(re.findall(r"\[(?:PASS|FAIL)\] (\w+):", passing_run.stdout))
    assert reported == set(INVARIANTS), (
        "an invariant was declared but not checked, or checked but not declared"
    )


def test_selftest_reports_every_condition_and_its_hash(passing_run):
    for condition in (
        "positive_control",
        "capacity_stressed",
        "negative_control",
        "lexical_decoy",
        "permutation_control",
        "matched_difficulty",
        "T0",
    ):
        assert re.search(rf"^{condition}\s", passing_run.stdout, re.MULTILINE), condition
    assert "oracle bound, negative control" in passing_run.stdout


@pytest.mark.parametrize(
    "invariant", ["negative_control_oracle", "splits_are_disjoint", "decoy_has_no_semantic_effect"]
)
def test_selftest_exits_non_zero_when_an_invariant_is_broken(invariant):
    result = _run("--break-invariant", invariant)
    assert result.returncode == 1
    assert f"[FAIL] {invariant}" in result.stdout
    assert "selftest FAILED (1)" in result.stdout


def test_show_example_prints_the_program_beside_the_tensor():
    result = _run("--show-example", "capacity_stressed", "--index", "0")
    assert result.returncode == 0
    assert "op=recall_by_key" in result.stdout
    assert "<-- SOURCE" in result.stdout and "<-- DEST" in result.stdout
    assert "target == source content bank:   True" in result.stdout
