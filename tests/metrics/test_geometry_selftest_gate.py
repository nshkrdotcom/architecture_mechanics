"""The geometry ``--selftest`` gate: it passes, and it is seen to fail.

Same discipline as the generator's and the capability metrics' gates. A gate
nobody has watched fail is not known to be a gate, so ``--break-invariant``
forces each named check to report failure and the suite exercises the non-zero
exit path rather than asserting it in prose.

This gate carries the same second job the capability gate does: the
retained/diagnostic split lives in ``GEOMETRY_MEASURES`` as source, and the
selftest re-derives it from the constructed cases. A diagnostic measure that
starts passing the rule is as much a failure as a retained one that stops,
because either means the recorded decision and the evidence for it have come
apart.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from architecture_mechanics.metrics.geometry import (
    CONSTRUCTED_CASES,
    DIAGNOSTIC_MEASURES,
    GEOMETRY_MEASURES,
    GEOMETRY_VERSION,
    INVARIANTS,
    RETAINED_MEASURES,
)

MODULE = "architecture_mechanics.metrics.geometry"


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
    assert GEOMETRY_VERSION in passing_run.stdout


def test_selftest_checks_every_declared_invariant(passing_run):
    reported = set(re.findall(r"\[(?:PASS|FAIL)\] (\w+):", passing_run.stdout))
    assert reported == set(INVARIANTS), (
        "an invariant was declared but not checked, or checked but not declared"
    )


def test_selftest_reports_all_five_constructed_cases(passing_run):
    for case in CONSTRUCTED_CASES:
        assert f"{case.name}  —  {case.description}" in passing_run.stdout, case.name
    assert len(CONSTRUCTED_CASES) == 5


def test_selftest_reports_every_measure_and_its_recorded_status(passing_run):
    for spec in GEOMETRY_MEASURES:
        assert re.search(rf"^{spec.name}\s.*\s{spec.status}\s", passing_run.stdout, re.MULTILINE), (
            spec.name
        )


@pytest.mark.parametrize("invariant", INVARIANTS)
def test_selftest_exits_non_zero_when_an_invariant_is_broken(invariant):
    result = _run("--break-invariant", invariant)
    assert result.returncode == 1
    assert f"[FAIL] {invariant}" in result.stdout
    assert "selftest FAILED (1)" in result.stdout


def test_table_writes_a_machine_readable_report():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "geometry_validation.json"
        result = _run("--table", "--json", str(path))
        assert result.returncode == 0, result.stdout + result.stderr
        payload = json.loads(path.read_text())

    assert payload["geometry_version"] == GEOMETRY_VERSION
    assert [case["case"] for case in payload["cases"]] == [c.name for c in CONSTRUCTED_CASES]
    for case in payload["cases"]:
        assert case["all_scalars_finite"] is True, case["case"]
        for row in case["expectations"]:
            assert row["ok"] is True, (case["case"], row)
    for row in payload["measures"]:
        assert row["agrees"] is True, row["name"]


def test_the_register_is_consistent():
    names = [spec.name for spec in GEOMETRY_MEASURES]
    assert len(names) == len(set(names))
    assert {spec.status for spec in GEOMETRY_MEASURES} == {"retained", "diagnostic"}
    assert set(RETAINED_MEASURES) | set(DIAGNOSTIC_MEASURES) == set(names)
    assert not set(RETAINED_MEASURES) & set(DIAGNOSTIC_MEASURES)
    for spec in GEOMETRY_MEASURES:
        assert spec.reason.strip(), spec.name


def test_every_constructed_case_expects_something_of_the_noise_null():
    """The noise case is the one that decides whether a measure is reportable,
    so it must be the most heavily constrained, not the least."""
    by_name = {case.name: case for case in CONSTRUCTED_CASES}
    noise = by_name["pure_noise"]
    measured = {expectation.measure for expectation in noise.expectations}
    assert {spec.name for spec in GEOMETRY_MEASURES} - measured == {"alignment_reference_mean"}, (
        "every measure but the one that needs a true basis must have a declared noise null"
    )
    for case in CONSTRUCTED_CASES:
        assert case.expectations, case.name
        for expectation in case.expectations:
            assert expectation.reason.strip(), (case.name, expectation.measure)
