"""The §8.4 bundle: complete for a pilot, minimal for a screen, empty-not-absent.

The distinction this file exists to protect: *a missing file and an empty file
mean different things*. A run with no ``interventions.jsonl`` says nothing about
whether interventions were done; a run with an empty one says they were not.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from architecture_mechanics.experiments.manifest import RunManifest
from architecture_mechanics.reporting.evidence_bundle import (
    FINAL_DIRS,
    FINAL_FILES,
    SCREEN_FILES,
    verify_bundle,
    write_bundle,
)


@dataclass
class _FakeResult:
    run_id: str = "R3-fake"
    config: dict = field(default_factory=lambda: {"optim": {"max_steps": 10}})
    checks: dict = field(default_factory=lambda: {"shapes": {"ok": True, "detail": ""}})
    history: list = field(default_factory=list)
    final: dict = field(default_factory=lambda: {"associative_recall_accuracy": 0.9})
    mechanism: dict = field(
        default_factory=lambda: {"verdict": {"active": True}, "mechanism_version": "mech-1.0.0"}
    )


def _manifest(run_id: str, rung: str) -> RunManifest:
    return RunManifest(
        schema="am.manifest.v1",
        run_id=run_id,
        ladder_rung=rung,
        git_commit="a" * 40,
        git_branch="main",
        git_source="work_tree",
        dirty_tree=False,
        dirty_paths=[],
        config={"ladder": rung},
        architecture_id="softmax-L2H2d48",
        parameter_count=62520,
        parameter_report={"total": 62520},
        operation_state_summary={"mechanism": "softmax_attention"},
        dataset_generator_version="fpg-1.0.0",
        split_hashes={"train": "aa", "eval": "bb"},
        seed=1,
        device={"resolved": "cuda"},
        precision="fp32",
        numerics={"precision": "fp32"},
        dependency_lock_hash="c" * 64,
        source_tree_hash="d" * 64,
        environment={"python": "3.12.0"},
        parent_claim_packet="claims/k0-test.yml",
        claimed_rung=1,
        primary_metric="associative_recall_accuracy",
        started_utc="2026-08-09T00:00:00+00:00",
    )


def _bundle(tmp_path: Path, rung: str, **kwargs):
    run_dir = tmp_path / "runs" / f"{rung}-fake"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(json.dumps({"run_id": f"{rung}-fake"}) + "\n")
    (run_dir / "metrics.jsonl").write_text("")
    return write_bundle(
        result=_FakeResult(run_id=f"{rung}-fake"),
        manifest=_manifest(f"{rung}-fake", rung),
        run_dir=run_dir,
        lab_root=tmp_path,
        **kwargs,
    )


@pytest.mark.parametrize("rung", ["R0", "R1", "R2"])
def test_a_screen_emits_only_provenance_and_results(tmp_path: Path, rung):
    report = _bundle(tmp_path, rung)
    assert report.complete, report.problems
    assert not report.is_final

    emitted = set(report.files)
    assert set(SCREEN_FILES) <= emitted
    heavy = set(FINAL_FILES) - set(SCREEN_FILES)
    assert not (heavy & emitted), "a screen is not evidence and must not look like it"


@pytest.mark.parametrize("rung", ["R3", "R4", "R5"])
def test_a_pilot_emits_the_whole_bundle(tmp_path: Path, rung):
    report = _bundle(tmp_path, rung)
    assert report.complete, report.problems
    assert report.is_final
    assert set(FINAL_FILES) <= set(report.files)
    for directory in FINAL_DIRS:
        assert any(name.startswith(f"{directory}/") for name in report.files)


def test_empty_artifacts_are_valid_structures_that_say_they_are_empty(tmp_path: Path):
    report = _bundle(tmp_path, "R3")
    run_dir = report.run_dir

    records = [json.loads(line) for line in (run_dir / "interventions.jsonl").read_text().splitlines()]
    assert records == [
        {
            "record": "schema",
            "schema": "am.interventions.v1",
            "run_id": "R3-fake",
            "n_records": 0,
            "note": "no interventions were performed; §6.4 interventions arrive in prompt 19",
        }
    ]

    with np.load(run_dir / "geometry_metrics.npz", allow_pickle=False) as geometry:
        assert str(geometry["__schema__"]) == "am.geometry_metrics.v1"
        assert bool(geometry["__empty__"]) is True

    figures = json.loads((run_dir / "figures" / "INDEX.json").read_text())
    assert figures["figures"] == [] and figures["empty"] is True

    checkpoint = json.loads((run_dir / "checkpoint" / "checkpoint.json").read_text())
    assert checkpoint["files"] == {} and checkpoint["empty"] is True

    gates = json.loads((run_dir / "claim_gates.json").read_text())
    assert gates["rungs"] == {} and gates["highest_supported_rung"] is None


def test_mechanism_activity_records_what_was_measured(tmp_path: Path):
    report = _bundle(tmp_path, "R3")
    payload = json.loads((report.run_dir / "mechanism_activity.json").read_text())
    assert payload["empty"] is False
    assert payload["verdict"] == {"active": True}


def test_the_manifest_indexes_every_other_file_in_the_bundle(tmp_path: Path):
    report = _bundle(tmp_path, "R3")
    manifest = json.loads((report.run_dir / "manifest.json").read_text())
    indexed = {entry["path"] for entry in manifest["evidence_index"]}
    assert indexed == set(report.files) - {"manifest.json"}
    assert all(len(entry["sha256"]) == 64 for entry in manifest["evidence_index"])


def test_reproduce_script_is_executable_and_pins_its_run(tmp_path: Path):
    report = _bundle(tmp_path, "R1")
    script = report.run_dir / "reproduce.sh"
    assert os.access(script, os.X_OK)
    text = script.read_text()
    assert 'RUN_ID="R1-fake"' in text
    assert f'COMMIT="{"a" * 40}"' in text
    assert 'CLAIM="claims/k0-test.yml"' in text
    assert 'PRIMARY_METRIC="associative_recall_accuracy"' in text
    assert "git archive" in text, "the source pin must not mutate the laboratory"


def test_reproduce_script_is_valid_bash(tmp_path: Path):
    import subprocess

    report = _bundle(tmp_path, "R1")
    check = subprocess.run(
        ["bash", "-n", str(report.run_dir / "reproduce.sh")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert check.returncode == 0, check.stderr


def test_verify_reports_a_missing_artifact_rather_than_ignoring_it(tmp_path: Path):
    report = _bundle(tmp_path, "R3")
    (report.run_dir / "interventions.jsonl").unlink()
    assert verify_bundle(report.run_dir) == ["missing interventions.jsonl"]


def test_verify_reports_an_empty_required_directory(tmp_path: Path):
    report = _bundle(tmp_path, "R3")
    for path in (report.run_dir / "figures").iterdir():
        path.unlink()
    assert verify_bundle(report.run_dir) == ["figures/ is empty"]


def test_verify_refuses_a_pilot_built_from_a_dirty_tree(tmp_path: Path):
    run_dir = tmp_path / "runs" / "R3-dirty"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text("{}\n")
    (run_dir / "metrics.jsonl").write_text("")
    manifest = _manifest("R3-dirty", "R3")
    manifest.dirty_tree = True
    manifest.dirty_paths = ["src/architecture_mechanics/models/softmax.py"]
    write_bundle(result=_FakeResult(), manifest=manifest, run_dir=run_dir, lab_root=tmp_path)
    assert "final run was produced from a dirty working tree" in verify_bundle(run_dir)


def test_verify_reports_a_run_with_no_manifest_at_all(tmp_path: Path):
    run_dir = tmp_path / "runs" / "R1-orphan"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text("{}\n")
    assert verify_bundle(run_dir) == ["missing manifest.json"]
