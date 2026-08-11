"""Run identity and the §8.3 manifest.

The property under test is the one §8.3 states and no run can check about
itself: *two identical runs collide by ID, and any change that would make them
different experiments separates them.* A laboratory that fails this accumulates
near-duplicate directories and nobody notices which of them is the result.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from architecture_mechanics.experiments.config import ladder_config, run_config_from_dict
from architecture_mechanics.experiments.manifest import (
    PROVENANCE_FIELDS,
    RunManifest,
    dependency_lock_hash,
    evidence_index,
    git_facts,
    lab_root,
    run_id_for,
    source_tree_hash,
)


def _identity(config, **overrides):
    payload = {
        "config_dict": config.as_dict(),
        "generator_version": "fpg-1.0.0",
        "source_hash": "abc123",
        "seed": config.seed,
        "ladder": config.ladder,
        "arch": config.arch.arch,
        "condition": config.data.condition,
    }
    payload.update(overrides)
    return run_id_for(payload.pop("config_dict"), **payload)


def test_identical_configs_collide_by_id():
    a = ladder_config("R1")
    b = ladder_config("R1")
    assert _identity(a) == _identity(b)


def test_id_has_no_timestamp_component():
    """Called twice a second apart, the same run is the same run."""
    config = ladder_config("R1")
    assert _identity(config) == _identity(config)


@pytest.mark.parametrize(
    "overrides",
    [
        {"seed": 7},
        {"source_hash": "different-source"},
        {"generator_version": "fpg-2.0.0"},
    ],
)
def test_changing_any_identity_input_changes_the_id(overrides):
    config = ladder_config("R1")
    assert _identity(config) != _identity(config, **overrides)


def test_changing_the_config_changes_the_id():
    base = ladder_config("R1")
    wider = ladder_config("R1", d_model=64)
    assert _identity(base) != _identity(wider)


def test_id_is_readable_before_it_is_unique():
    config = ladder_config("R2", seed=11)
    run_id = _identity(config)
    assert run_id.startswith("R2-softmax-capacity_stressed-s11-")


def test_source_tree_hash_is_stable_and_content_sensitive(tmp_path: Path):
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    module = tmp_path / "src" / "pkg" / "a.py"
    module.write_text("x = 1\n")
    first = source_tree_hash(tmp_path)
    assert first == source_tree_hash(tmp_path)

    module.write_text("x = 2\n")
    assert source_tree_hash(tmp_path) != first


def test_source_tree_hash_notices_a_moved_file(tmp_path: Path):
    """Two trees with identical bytes in different places are different trees."""
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "a.py").write_text("x = 1\n")
    first = source_tree_hash(tmp_path)

    (tmp_path / "src" / "pkg" / "a.py").rename(tmp_path / "src" / "pkg" / "b.py")
    assert source_tree_hash(tmp_path) != first


def test_source_tree_hash_ignores_bytecode(tmp_path: Path):
    (tmp_path / "src" / "pkg" / "__pycache__").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "a.py").write_text("x = 1\n")
    first = source_tree_hash(tmp_path)
    (tmp_path / "src" / "pkg" / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"\x00\x01")
    assert source_tree_hash(tmp_path) == first


def test_unidentifiable_source_is_reported_dirty(tmp_path: Path, monkeypatch):
    """No git, no pin: unknown provenance is recorded as unreproducible, not as fine."""
    monkeypatch.delenv("AM_SOURCE_COMMIT", raising=False)
    facts = git_facts(tmp_path)
    assert facts["git_commit"] is None
    assert facts["dirty_tree"] is True
    assert facts["git_source"] == "unavailable"


def test_pinned_export_is_clean_by_construction(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AM_SOURCE_COMMIT", "deadbeef")
    facts = git_facts(tmp_path)
    assert facts == {
        "git_commit": "deadbeef",
        "git_branch": None,
        "git_source": "pinned_export",
        "dirty_tree": False,
        "dirty_paths": [],
    }


def test_the_laboratory_itself_reports_real_git_facts():
    facts = git_facts(lab_root())
    assert facts["git_source"] == "work_tree"
    assert facts["git_commit"] and len(facts["git_commit"]) == 40
    assert dependency_lock_hash() != "unavailable"


def test_evidence_index_hashes_every_emitted_file(tmp_path: Path):
    (tmp_path / "figures").mkdir()
    (tmp_path / "summary.json").write_text("{}\n")
    (tmp_path / "figures" / "one.txt").write_text("hello")
    (tmp_path / "manifest.json").write_text("{}\n")

    index = evidence_index(tmp_path)
    paths = [entry["path"] for entry in index]
    assert paths == ["figures/one.txt", "summary.json"], "the manifest cannot index itself"
    assert all(len(entry["sha256"]) == 64 for entry in index)
    assert index[0]["bytes"] == 5


def _manifest(**overrides) -> RunManifest:
    base = {
        "schema": "am.manifest.v1",
        "run_id": "R1-x",
        "ladder_rung": "R1",
        "git_commit": "a" * 40,
        "git_branch": "main",
        "git_source": "work_tree",
        "dirty_tree": False,
        "dirty_paths": [],
        "config": {"ladder": "R1"},
        "architecture_id": "softmax-L2H2d48",
        "parameter_count": 62520,
        "parameter_report": {"total": 62520},
        "operation_state_summary": {"mechanism": "softmax_attention"},
        "dataset_generator_version": "fpg-1.0.0",
        "split_hashes": {"train": "aa", "eval": "bb"},
        "seed": 1,
        "device": {"resolved": "cuda"},
        "precision": "fp32",
        "numerics": {"precision": "fp32"},
        "dependency_lock_hash": "c" * 64,
        "source_tree_hash": "d" * 64,
        "environment": {"python": "3.12.0"},
        "parent_claim_packet": "claims/x.yml",
        "claimed_rung": 1,
        "primary_metric": "associative_recall_accuracy",
        "started_utc": "2026-08-09T00:00:00+00:00",
        # A complete manifest describes the bytes it sits next to. RunManifest
        # fills this in at write time; the fixture never writes, so it states
        # the finished shape itself.
        "evidence_index": [{"path": "summary.json", "sha256": "e" * 64, "bytes": 1}],
    }
    base.update(overrides)
    return RunManifest(**base)


def test_a_complete_manifest_satisfies_every_gate_field():
    assert _manifest().missing_provenance() == []


@pytest.mark.parametrize("field", PROVENANCE_FIELDS)
def test_every_provenance_field_is_actually_required(field):
    """Blank each §8.3 field in turn; each one must be noticed on its own."""
    blank = {"config": {}, "split_hashes": {}, "device": {}, "dirty_paths": []}.get(field, None)
    if field == "dirty_tree":
        # False is a legitimate value, so the gate's "empty" rule cannot catch
        # it; None is what a manifest that never asked would carry.
        blank = None
    manifest = _manifest(**{field: blank})
    assert manifest.missing_provenance() == [field]


def test_an_incomplete_manifest_refuses_to_be_written(tmp_path: Path):
    manifest = _manifest(parent_claim_packet="")
    with pytest.raises(ValueError, match="parent_claim_packet"):
        manifest.write(tmp_path)
    assert not (tmp_path / "manifest.json").exists()


def test_written_manifest_round_trips_and_indexes_its_neighbours(tmp_path: Path):
    (tmp_path / "summary.json").write_text('{"run_id": "R1-x"}\n')
    path = _manifest().write(tmp_path)
    record = json.loads(path.read_text())
    assert record["schema"] == "am.manifest.v1"
    assert [entry["path"] for entry in record["evidence_index"]] == ["summary.json"]
    assert run_config_from_dict is not None  # imported for the round-trip test below


def test_config_round_trips_through_the_manifest_block():
    """reproduce.sh re-runs from `config`; if it does not round-trip, it re-runs
    something else."""
    for rung in ("R0", "R1", "R2"):
        config = ladder_config(rung, seed=5, device="cpu", d_model=32)
        assert run_config_from_dict(config.as_dict()) == config


def test_a_config_from_another_generator_version_is_refused():
    payload = ladder_config("R1").as_dict() | {"generator_version": "fpg-0.9.0"}
    with pytest.raises(Exception, match="different experiment"):
        run_config_from_dict(payload)
