"""The capability-metric ``--selftest`` gate: it passes, and it is seen to fail.

Same discipline as the generator's gate. A gate nobody has watched fail is not
known to be a gate, so ``--break-invariant`` forces one named check to report
failure and the suite exercises the non-zero exit path rather than asserting it
in prose.

This gate carries a second job the generator's does not: the retirement
decisions live in ``METRIC_SPECS`` as source, and the selftest recomputes the
rule that produced them. A retired metric that starts passing is as much a
failure as a retained one that starts failing, because either means the recorded
decision and the evidence for it have come apart.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from architecture_mechanics.metrics.capability import (
    INVARIANTS,
    METRIC_SPECS,
    RETAINED_METRICS,
    RETIRED_METRICS,
)

MODULE = "architecture_mechanics.metrics.capability"
SMALL = ("--n-examples", "96")


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
    return _run("--selftest", *SMALL)


def test_selftest_passes(passing_run):
    assert passing_run.returncode == 0, passing_run.stdout + passing_run.stderr
    assert "selftest PASSED" in passing_run.stdout


def test_selftest_checks_every_declared_invariant(passing_run):
    reported = set(re.findall(r"\[(?:PASS|FAIL)\] (\w+):", passing_run.stdout))
    assert reported == set(INVARIANTS), (
        "an invariant was declared but not checked, or checked but not declared"
    )


def test_selftest_reports_every_metric_and_its_recorded_status(passing_run):
    for spec in METRIC_SPECS:
        assert re.search(rf"^{spec.name}\s+{spec.status}\s", passing_run.stdout, re.MULTILINE), (
            spec.name
        )


def test_selftest_reports_every_calibration_condition(passing_run):
    for condition in ("T0", "positive_control", "capacity_stressed", "synthetic_overwrite"):
        assert f"condition={condition} " in passing_run.stdout, condition


@pytest.mark.parametrize(
    "invariant",
    [
        "retained_metrics_beat_the_frequency_ceiling",
        "retired_metrics_fail_the_rule",
        "positive_control_threshold_separates",
        "metrics_follow_the_permutation",
    ],
)
def test_selftest_exits_non_zero_when_an_invariant_is_broken(invariant):
    result = _run("--break-invariant", invariant, *SMALL)
    assert result.returncode == 1
    assert f"[FAIL] {invariant}" in result.stdout
    assert "selftest FAILED (1)" in result.stdout


def test_t0_runs_end_to_end_and_prints_the_table():
    result = _run("--t0", *SMALL)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "condition=T0 family=T0" in result.stdout
    assert "No model was involved." in result.stdout
    for name in RETAINED_METRICS:
        assert re.search(rf"^{name}\s", result.stdout, re.MULTILINE), name


def test_calibrate_writes_a_machine_readable_report():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "calibration.json"
        result = _run("--calibrate", "--json", str(path), *SMALL)
        assert result.returncode == 0, result.stdout + result.stderr
        payload = json.loads(path.read_text())

    assert payload["ok"] is True
    assert payload["metric_version"]
    names = {v["name"] for v in payload["verdicts"]}
    assert names == set(RETAINED_METRICS) | set(RETIRED_METRICS) | {
        s.name for s in METRIC_SPECS if s.status == "diagnostic"
    }
    for verdict in payload["verdicts"]:
        assert verdict["agrees_with_recorded_status"] is True, verdict["name"]
        if verdict["status"] == "retained":
            assert verdict["marginal_skill"] == pytest.approx(0.0, abs=1e-12)
    control = payload["positive_control"]
    assert control["values"]["oracle"] > control["threshold"]
    assert control["values"]["marginal"] < control["threshold"]
