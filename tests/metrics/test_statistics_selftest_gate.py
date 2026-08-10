"""The statistics ``--selftest`` gate: it passes, and it is seen to fail.

Same discipline as the generator's, the capability metrics' and the geometry
gate's. A gate nobody has watched fail is not known to be a gate, so
``--break-invariant`` forces each named check to report failure and the suite
exercises the non-zero exit path rather than asserting it in prose.

This gate carries the same second job the other two do: the status of every
estimator lives in ``ESTIMATOR_SPECS`` as source, and the selftest re-derives it
from a fresh calibration. An estimator recorded as unusable that starts holding
its level is as much a failure as an adopted one that stops — either means the
record and the evidence have come apart.

Every invocation here runs both calibrations, so the replicate counts are held
down deliberately. The gate's own defaults are three and four times these, and
``reports/statistics_calibration.json`` is ten times them again.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from architecture_mechanics.metrics.statistics import (
    ADOPTED,
    ESTIMATOR_SPECS,
    FORBIDDEN_ESTIMATORS,
    INVARIANTS,
    LEVEL_TOLERANCE,
    STATISTICS_VERSION,
    THRESHOLDS,
)

MODULE = "architecture_mechanics.metrics.statistics"
FAST = ("--null-replicates", "150", "--power-replicates", "100")


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", MODULE, *args],
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )


@pytest.fixture(scope="module")
def passing_run() -> subprocess.CompletedProcess:
    return _run("--selftest", *FAST)


def test_selftest_passes(passing_run):
    assert passing_run.returncode == 0, passing_run.stdout + passing_run.stderr
    assert "selftest PASSED" in passing_run.stdout
    assert STATISTICS_VERSION in passing_run.stdout


def test_selftest_checks_every_declared_invariant(passing_run):
    reported = set(re.findall(r"\[(?:PASS|FAIL)\] (\w+):", passing_run.stdout))
    assert reported == set(INVARIANTS), (
        "an invariant was declared but not checked, or checked but not declared"
    )


def test_selftest_reports_every_estimator_and_its_recorded_status(passing_run):
    for spec in ESTIMATOR_SPECS:
        assert re.search(rf"^{spec.name}\s+{spec.status}\s", passing_run.stdout, re.MULTILINE), (
            spec.name
        )


def test_selftest_reports_the_minimum_detectable_effect(passing_run):
    """The number every later mission reads out of this mission."""
    assert "minimum detectable effect" in passing_run.stdout
    assert re.search(r"n =\s+5 seeds: dz = 1\.[0-9]{2}", passing_run.stdout)


@pytest.mark.parametrize("invariant", INVARIANTS)
def test_selftest_exits_non_zero_when_an_invariant_is_broken(invariant):
    result = _run("--break-invariant", invariant, *FAST)
    assert result.returncode == 1
    assert f"[FAIL] {invariant}" in result.stdout
    assert "selftest FAILED (1)" in result.stdout


def test_calibrate_writes_a_machine_readable_report():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "statistics_calibration.json"
        result = _run("--calibrate", "--null-replicates", "150", "--power-replicates", "100",
                      "--json", str(path))
        assert result.returncode == 0, result.stdout + result.stderr
        payload = json.loads(path.read_text())

    assert payload["statistics_version"] == STATISTICS_VERSION
    assert payload["ok"] is True
    assert {row["name"] for row in payload["estimators"]} == {s.name for s in ESTIMATOR_SPECS}
    for row in payload["estimators"]:
        assert row["agrees"] is True, row["name"]
    assert {row["n"] for row in payload["null_calibration"]["rows"]} >= {3, 5, 10}
    assert payload["power_calibration"]["effect_units"].startswith("standard deviations")


def test_the_recorded_calibration_is_committed_and_agrees_with_the_register():
    """``reports/statistics_calibration.json`` is the recorded run at full
    replicate counts. Its numbers are what ``state/08_statistics.md`` quotes, so
    a register that has drifted away from it fails here rather than in the
    write-up."""
    from architecture_mechanics.experiments.manifest import lab_root

    path = lab_root() / "reports" / "statistics_calibration.json"
    assert path.is_file(), "run `make statistics-calibration`"
    payload = json.loads(path.read_text())

    assert payload["ok"] is True
    assert payload["null_calibration"]["replicates"] >= 200, "the mission's floor"
    assert payload["statistics_version"] == STATISTICS_VERSION
    assert payload["level_tolerance"] == LEVEL_TOLERANCE
    assert payload["adopted"] == ADOPTED

    by_name = {row["name"]: row for row in payload["estimators"]}
    for spec in ESTIMATOR_SPECS:
        row = by_name[spec.name]
        assert row["derived_status"] == spec.status, spec.name
        assert row["measured_fpr"] == pytest.approx(spec.recorded_fpr_at_5, abs=0.005), spec.name

    at_five = next(
        row for row in payload["minimum_detectable_effect"] if row["n_seeds"] == 5
    )
    assert at_five["minimum_detectable_effect_dz"] == pytest.approx(
        THRESHOLDS["minimum_detectable_effect_dz_at_5_seeds"], abs=0.05
    )


def test_the_recorded_calibration_shows_every_forbidden_analysis_still_broken():
    from architecture_mechanics.experiments.manifest import lab_root

    payload = json.loads((lab_root() / "reports" / "statistics_calibration.json").read_text())
    by_name = {row["name"]: row for row in payload["estimators"]}
    for name in FORBIDDEN_ESTIMATORS:
        assert by_name[name]["measured_fpr"] > 4 * payload["alpha"], (
            f"{name} is recorded as a demonstration of a §7.4 error; if it stopped being one, "
            "the calibration stopped measuring what it claims to"
        )
