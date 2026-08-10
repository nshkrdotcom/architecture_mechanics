"""The four science gates, each fed the artifact it exists to refuse.

`tests/provenance/test_gate_agreement.py` holds the laboratory to the gates'
*shape*. Nothing held the gates to their *behaviour*: every one of them has
exited 0 on every run of this laboratory since it was written, and a gate that
has only ever passed is indistinguishable from a gate that cannot fail.

Each fixture is built under `tmp_path` and the real laboratory is never touched.
Every case is paired with the corrected version of the same fixture, which must
pass — otherwise the assertion is satisfied by a gate that refuses everything.

Skipped, not failed, when the program directory is absent: the laboratory must
remain testable on a machine that only has the laboratory.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[2]
PROGRAM_DIR = Path(
    os.environ.get("AM_PROGRAM_DIR", "/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260809/ml")
)
BIN = PROGRAM_DIR / "bin"

pytestmark = pytest.mark.skipif(not BIN.is_dir(), reason=f"gate scripts not present at {BIN}")


def _gate(name: str, lab: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(BIN / name), str(lab)],
        capture_output=True, text=True, timeout=600, check=False,
    )


def _git(repo: Path, *args: str, when: str | None = None) -> None:
    env = dict(os.environ)
    if when is not None:
        env["GIT_AUTHOR_DATE"] = env["GIT_COMMITTER_DATE"] = when
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, env=env)


def _a_recorded_run(kind: str) -> Path:
    """Any real run directory of the given rung, to copy from."""
    matches = sorted((LAB / "runs").glob(f"{kind}-*"))
    if not matches:
        pytest.skip(f"no recorded {kind} run to build a fixture from")
    return matches[0]


# --------------------------------------------------------------------------- #
# check_prereg.sh — a claim that did not predate its run
# --------------------------------------------------------------------------- #


def _prereg_fixture(tmp_path: Path, *, claim_committed_at: str | None, run_started: str) -> Path:
    lab = tmp_path / "lab"
    (lab / "runs" / "R1-fixture").mkdir(parents=True)
    (lab / "claims").mkdir()
    _git(lab.parent, "init", "-q", str(lab))
    _git(lab, "config", "user.email", "fixture@example.invalid")
    _git(lab, "config", "user.name", "fixture")

    source = _a_recorded_run("R1")
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["parent_claim_packet"] = "claims/fixture.yml"
    manifest["started_utc"] = run_started
    (lab / "runs" / "R1-fixture" / "manifest.json").write_text(json.dumps(manifest, indent=2))
    shutil.copy(LAB / "claims" / "a0-t1-associative-recall.yml", lab / "claims" / "fixture.yml")

    _git(lab, "add", "runs")
    _git(lab, "commit", "-qm", "the run", when="2026-08-01T00:00:00+00:00")
    if claim_committed_at is not None:
        _git(lab, "add", "claims")
        _git(lab, "commit", "-qm", "the claim", when=claim_committed_at)
    return lab


def test_check_prereg_refuses_a_claim_committed_after_the_run(tmp_path):
    lab = _prereg_fixture(
        tmp_path, claim_committed_at="2026-08-10T12:00:00+00:00",
        run_started="2026-08-10T00:00:00Z",
    )
    result = _gate("check_prereg.sh", lab)
    assert result.returncode != 0
    assert "committed AFTER the run started" in result.stdout
    assert "post-hoc claim" in result.stdout


def test_check_prereg_refuses_a_claim_that_is_not_in_history(tmp_path):
    lab = _prereg_fixture(tmp_path, claim_committed_at=None, run_started="2026-08-10T00:00:00Z")
    result = _gate("check_prereg.sh", lab)
    assert result.returncode != 0
    assert "is not committed to git" in result.stdout


def test_check_prereg_accepts_the_same_fixture_committed_in_time(tmp_path):
    """Non-vacuity."""
    lab = _prereg_fixture(
        tmp_path, claim_committed_at="2026-08-02T00:00:00+00:00",
        run_started="2026-08-10T00:00:00Z",
    )
    result = _gate("check_prereg.sh", lab)
    assert result.returncode == 0, result.stdout + result.stderr


# --------------------------------------------------------------------------- #
# check_claims.sh — a rung claimed above the evidence
# --------------------------------------------------------------------------- #


def _claims_fixture(tmp_path: Path, rung: int) -> Path:
    lab = tmp_path / "lab"
    (lab / "claims").mkdir(parents=True)
    text = (LAB / "claims" / "a0-t1-associative-recall.yml").read_text()
    text, n = re.subn(
        r"^claimed_rung:\s*\d+\s*$", f"claimed_rung: {rung}", text, flags=re.MULTILINE
    )
    assert n == 1, "the packet no longer declares claimed_rung on its own line"
    (lab / "claims" / "a0-t1-associative-recall.yml").write_text(text)
    shutil.copy(
        LAB / "claims" / "a0-t1-associative-recall.gates.json",
        lab / "claims" / "a0-t1-associative-recall.gates.json",
    )
    return lab


def test_check_claims_refuses_a_rung_above_the_recorded_evidence(tmp_path):
    result = _gate("check_claims.sh", _claims_fixture(tmp_path, 3))
    assert result.returncode != 0
    assert "is not passed" in result.stdout
    assert "exceeds highest_supported_rung" in result.stdout


def test_check_claims_accepts_the_rung_the_evidence_supports(tmp_path):
    """Non-vacuity."""
    result = _gate("check_claims.sh", _claims_fixture(tmp_path, 1))
    assert result.returncode == 0, result.stdout + result.stderr


# --------------------------------------------------------------------------- #
# check_no_rescue.sh — a candidate tuned more than its control
# --------------------------------------------------------------------------- #


def _comparison_fixture(tmp_path: Path, permitted: dict) -> Path:
    lab = tmp_path / "lab"
    (lab / "reports" / "comparisons").mkdir(parents=True)
    source = _a_recorded_run("R4")
    control = source.name
    candidate = "R4-candidate-fixture"
    for name in (control, candidate):
        (lab / "runs" / name).mkdir(parents=True)
    manifest = json.loads((source / "manifest.json").read_text())
    (lab / "runs" / control / "manifest.json").write_text(json.dumps(manifest, indent=2))
    manifest["config"]["optim"]["learning_rate"] *= 2
    (lab / "runs" / candidate / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (lab / "reports" / "comparisons" / "fixture.json").write_text(
        json.dumps(
            {
                "comparison_id": "fixture",
                "claim": "claims/a0-t1-associative-recall.yml",
                "control_run": control,
                "candidate_runs": [candidate],
                "permitted_differences": permitted,
            },
            indent=2,
        )
    )
    return lab


def test_check_no_rescue_refuses_an_undeclared_configuration_difference(tmp_path):
    result = _gate("check_no_rescue.sh", _comparison_fixture(tmp_path, {}))
    assert result.returncode != 0
    assert "optim.learning_rate" in result.stdout
    assert "undeclared difference" in result.stdout


def test_check_no_rescue_refuses_a_difference_declared_without_a_justification(tmp_path):
    result = _gate("check_no_rescue.sh", _comparison_fixture(tmp_path, {"optim.learning_rate": "  "}))
    assert result.returncode != 0
    assert "empty justification" in result.stdout


def test_check_no_rescue_accepts_a_declared_and_justified_difference(tmp_path):
    """Non-vacuity."""
    lab = _comparison_fixture(tmp_path, {"optim.learning_rate": "fixture: declared deliberately"})
    result = _gate("check_no_rescue.sh", lab)
    assert result.returncode == 0, result.stdout + result.stderr


# --------------------------------------------------------------------------- #
# check_evidence.sh — an incomplete or altered bundle
# --------------------------------------------------------------------------- #


@pytest.fixture
def bundle(tmp_path):
    lab = tmp_path / "lab"
    source = _a_recorded_run("R4")
    shutil.copytree(source, lab / "runs" / source.name)
    return lab, lab / "runs" / source.name


def test_the_intact_bundle_passes(bundle):
    """Non-vacuity, and the baseline every case below is a deviation from."""
    lab, _ = bundle
    result = _gate("check_evidence.sh", lab)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "missing",
    ["geometry_metrics.npz", "mechanism_activity.json", "interventions.jsonl",
     "claim_gates.json", "reproduce.sh", "summary.json", "metrics.jsonl"],
)
def test_check_evidence_refuses_a_bundle_missing_one_file(bundle, missing):
    lab, run_dir = bundle
    (run_dir / missing).unlink()
    result = _gate("check_evidence.sh", lab)
    assert result.returncode != 0
    assert f"missing {missing}" in result.stdout


def test_check_evidence_refuses_results_with_no_manifest(bundle):
    lab, run_dir = bundle
    (run_dir / "manifest.json").unlink()
    result = _gate("check_evidence.sh", lab)
    assert result.returncode != 0
    assert "results with no manifest.json" in result.stdout


def test_check_evidence_refuses_an_empty_figures_directory(bundle):
    lab, run_dir = bundle
    for figure in (run_dir / "figures").iterdir():
        figure.unlink()
    result = _gate("check_evidence.sh", lab)
    assert result.returncode != 0
    assert "figures/ is empty" in result.stdout


def test_check_evidence_refuses_a_blanked_provenance_field(bundle):
    lab, run_dir = bundle
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["split_hashes"] = {}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    result = _gate("check_evidence.sh", lab)
    assert result.returncode != 0
    assert "missing provenance field: split_hashes" in result.stdout


def test_check_evidence_refuses_a_result_that_no_longer_matches_its_own_index(bundle):
    """Added by prompt 10. The manifest carries a sha256 of every artifact and
    nothing compared them, so a summary could be edited after the fact and every
    gate stayed green."""
    lab, run_dir = bundle
    summary = json.loads((run_dir / "summary.json").read_text())
    summary["final"]["associative_recall_accuracy"] = 0.9999
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    result = _gate("check_evidence.sh", lab)
    assert result.returncode != 0
    assert "evidence_index digest does not match the bytes of summary.json" in result.stdout


def test_the_index_check_stays_quiet_about_machine_side_files(bundle):
    """`cost.json` is wall clock and peak VRAM, gitignored, and rewritten by an
    agreeing repeat. Six recorded manifests were found in exactly that state."""
    lab, run_dir = bundle
    (run_dir / "cost.json").write_text('{"tampered": true}')
    result = _gate("check_evidence.sh", lab)
    assert result.returncode == 0, result.stdout + result.stderr
