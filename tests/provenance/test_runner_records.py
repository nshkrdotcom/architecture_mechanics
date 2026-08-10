"""What the runner actually leaves on disk when it records a run.

Written because the first version of this wiring did not. One call site kept the
old signature, so a real R1 trained for eighty seconds and scattered
``summary.json`` into the runs *root* instead of producing a run directory —
which every unit test above passed straight through, because none of them ran
the runner. This file runs it.

R0 on the CPU, so the whole file costs about a second and needs no GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from architecture_mechanics.experiments.claim_packet import (
    ClaimPacket,
    load_gates,
    load_packet,
)
from architecture_mechanics.experiments.config import RunConfigError, ladder_config
from architecture_mechanics.experiments.index import index_runs
from architecture_mechanics.experiments.runner import run
from architecture_mechanics.reporting.evidence_bundle import SCREEN_FILES, verify_bundle


@pytest.fixture
def claim(tmp_path: Path) -> Path:
    packet = ClaimPacket(
        claim_id="t-fixture",
        claimed_rung=1,
        primary_metric_key="associative_recall_accuracy",
        fields={
            "CLAIM": "the instrument stands up",
            "MECHANISM": "softmax attention",
            "STRUCTURALLY_ENFORCED_PROPERTIES": ["causality"],
            "LEARNED_OR_HOPED_PROPERTIES": ["retrieval"],
            "NEAREST_BORING_EXPLANATION": "memorisation",
            "CONTROL_THAT_RULES_IT_OUT": "held-out compositions",
            "PRIMARY_METRIC": "exact answer-set recall",
            "MECHANISM_ACTIVITY_METRIC": "retrieval lift",
            "POSITIVE_CONTROL": "R1",
            "NEGATIVE_CONTROL": "the training marginal must fail the threshold",
            "KILL_CONDITION": "recall below 0.80",
            "REPLICATION_REQUIREMENT": "five seeds at rung 2",
        },
    )
    return packet.write(tmp_path / "claims" / "t-fixture.yml")


def _r0(tmp_path: Path, claim: Path, **kwargs):
    return run(
        ladder_config("R0", device="cpu"),
        out_dir=tmp_path / "runs",
        verbose=False,
        claim=claim,
        claims_dir=claim.parent,
        **kwargs,
    )


def test_a_recorded_run_lands_in_a_directory_named_for_itself(tmp_path: Path, claim: Path):
    result = _r0(tmp_path, claim)
    runs = tmp_path / "runs"
    assert [p.name for p in runs.iterdir()] == [result.run_id]
    assert not [p for p in runs.iterdir() if p.is_file()], "nothing belongs at the run root"


def test_a_recorded_run_emits_a_complete_screen_bundle(tmp_path: Path, claim: Path):
    result = _r0(tmp_path, claim, emit_bundle=True)
    run_dir = tmp_path / "runs" / result.run_id
    assert verify_bundle(run_dir) == []
    for name in SCREEN_FILES:
        assert (run_dir / name).is_file(), name


def test_the_manifest_records_the_claim_and_the_rung(tmp_path: Path, claim: Path):
    result = _r0(tmp_path, claim)
    manifest = json.loads((tmp_path / "runs" / result.run_id / "manifest.json").read_text())
    assert manifest["ladder_rung"] == "R0"
    assert manifest["parent_claim_packet"].endswith("t-fixture.yml")
    assert manifest["claimed_rung"] == 1
    assert manifest["parameter_count"] == 62520
    assert manifest["operation_state_summary"]["mixing"][0]["mechanism"] == "softmax_attention"
    assert manifest["split_hashes"]["train"] and manifest["split_hashes"]["eval"]
    assert manifest["started_utc"] <= manifest["finished_utc"]


def test_the_run_id_is_the_same_on_a_second_identical_run(tmp_path: Path, claim: Path):
    """Two identical runs collide by ID rather than accumulating as two results."""
    first = _r0(tmp_path, claim)
    second = _r0(tmp_path, claim)
    assert first.run_id == second.run_id
    assert len(list((tmp_path / "runs").iterdir())) == 1


def test_an_agreeing_repeat_leaves_the_recorded_run_untouched(tmp_path: Path, claim: Path):
    """Re-running the same thing must not churn the provenance record.

    The science is byte-identical; the manifest's clock and commit are not. A
    rewrite would restate the same result as of a later moment, and the gates
    file would gain a new evaluation timestamp for a measurement nobody remade.
    """
    result = _r0(tmp_path, claim, emit_bundle=True)
    run_dir = tmp_path / "runs" / result.run_id
    before = {
        path.name: path.read_bytes()
        for path in [*run_dir.iterdir(), claim.parent / "t-fixture.gates.json"]
        if path.is_file() and path.name != "cost.json"
    }

    _r0(tmp_path, claim, emit_bundle=True)

    after = {
        path.name: path.read_bytes()
        for path in [*run_dir.iterdir(), claim.parent / "t-fixture.gates.json"]
        if path.is_file() and path.name != "cost.json"
    }
    assert after == before


def test_a_repeat_that_disagrees_is_refused_rather_than_overwriting(tmp_path: Path, claim: Path):
    """Same identity, different result: something outside config+source+seed moved."""
    result = _r0(tmp_path, claim)
    summary = tmp_path / "runs" / result.run_id / "summary.json"
    tampered = json.loads(summary.read_text())
    tampered["passed"] = False
    summary.write_text(json.dumps(tampered, indent=2) + "\n")

    with pytest.raises(RunConfigError, match="already recorded"):
        _r0(tmp_path, claim)
    assert json.loads(summary.read_text())["passed"] is False, "evidence must survive the refusal"


def test_overwrite_replaces_a_disagreeing_recorded_run(tmp_path: Path, claim: Path):
    result = _r0(tmp_path, claim)
    summary = tmp_path / "runs" / result.run_id / "summary.json"
    summary.write_text(json.dumps({"passed": False}, indent=2) + "\n")

    _r0(tmp_path, claim, overwrite=True)
    assert json.loads(summary.read_text())["passed"] is True


def test_recording_a_run_updates_the_claim_gates_from_measurement(tmp_path: Path, claim: Path):
    result = _r0(tmp_path, claim)
    gates = load_gates(claim.parent / "t-fixture.gates.json")
    entry = gates.as_dict()["rungs"]["0_implementation_survives"]

    assert entry["passed"] is True
    assert entry["evidence"] == [str((tmp_path / "runs" / result.run_id).resolve())]
    assert entry["evaluations"][0]["measured"]["invariant_failures"] == []
    # R0 does no forward pass with capture, so rung 1 was never evaluated —
    # which is different from being evaluated and failing.
    assert "1_mechanism_is_active" not in gates.as_dict()["rungs"]
    assert gates.as_dict()["highest_supported_rung"] == 0


def test_a_recorded_run_without_a_claim_is_refused(tmp_path: Path):
    """No packet named, and no packet declaring it covers this run."""
    empty = tmp_path / "claims"
    empty.mkdir()
    with pytest.raises(RunConfigError, match="pre-registration"):
        run(
            ladder_config("R0", device="cpu"),
            out_dir=tmp_path / "runs",
            verbose=False,
            claims_dir=empty,
        )
    assert not (tmp_path / "runs").exists()


def _covering(claim: Path, covers: dict) -> Path:
    """Rewrite the fixture packet with a declared scope."""
    packet = load_packet(claim)
    packet.extra["covers"] = covers
    return packet.write(claim)


R0_SCOPE = {"ladder": ["R0"], "arch": ["softmax"], "condition": ["positive_control"]}


def test_a_run_finds_the_packet_that_declared_it(tmp_path: Path, claim: Path):
    """The whole point: the parent is resolved from the packet's own committed scope."""
    _covering(claim, R0_SCOPE)
    result = run(
        ladder_config("R0", device="cpu"),
        out_dir=tmp_path / "runs",
        verbose=False,
        claims_dir=claim.parent,
    )
    manifest = json.loads((tmp_path / "runs" / result.run_id / "manifest.json").read_text())
    assert manifest["parent_claim_packet"].endswith("t-fixture.yml")
    assert manifest["claimed_rung"] == 1


def test_a_packet_covering_another_rung_does_not_adopt_this_run(tmp_path: Path, claim: Path):
    _covering(claim, {**R0_SCOPE, "ladder": ["R2"]})
    with pytest.raises(RunConfigError, match="no packet in .* declares covers"):
        run(
            ladder_config("R0", device="cpu"),
            out_dir=tmp_path / "runs",
            verbose=False,
            claims_dir=claim.parent,
        )


def test_two_packets_claiming_one_run_are_refused_rather_than_chosen_between(
    tmp_path: Path, claim: Path
):
    """An ambiguous parent is the researcher's problem to resolve, not the runner's."""
    _covering(claim, R0_SCOPE)
    twin = load_packet(claim)
    twin.claim_id = "t-fixture-twin"
    twin.write(claim.parent / "t-fixture-twin.yml")
    with pytest.raises(RunConfigError, match="2 pre-registrations claim"):
        run(
            ladder_config("R0", device="cpu"),
            out_dir=tmp_path / "runs",
            verbose=False,
            claims_dir=claim.parent,
        )


def test_an_unreadable_packet_is_named_rather_than_skipped_silently(tmp_path: Path, claim: Path):
    (claim.parent / "broken.yml").write_text("claim_id: broken\nCLAIM: ''\n")
    with pytest.raises(RunConfigError, match="unreadable packets: broken.yml"):
        run(
            ladder_config("R0", device="cpu"),
            out_dir=tmp_path / "runs",
            verbose=False,
            claims_dir=claim.parent,
        )


def test_an_explicit_claim_overrides_a_declared_scope(tmp_path: Path, claim: Path):
    """--claim still wins, and does not consult covers: at all."""
    _covering(claim, {**R0_SCOPE, "ladder": ["R2"]})
    result = _r0(tmp_path, claim)
    manifest = json.loads((tmp_path / "runs" / result.run_id / "manifest.json").read_text())
    assert manifest["parent_claim_packet"].endswith("t-fixture.yml")


def test_a_scratch_run_needs_no_claim_and_leaves_nothing(tmp_path: Path):
    result = run(ladder_config("R0", device="cpu"), out_dir=None, verbose=False)
    assert result.passed
    assert not list(tmp_path.iterdir())


def test_the_index_finds_the_run_against_its_claim(tmp_path: Path, claim: Path):
    result = _r0(tmp_path, claim)
    rows = index_runs(tmp_path)
    assert [row["run_id"] for row in rows] == [result.run_id]
    assert rows[0]["rung"] == "R0"
    assert rows[0]["manifest"] is True
    assert rows[0]["claim"].endswith("t-fixture.yml")


def test_the_index_shows_a_run_that_hid_its_provenance(tmp_path: Path):
    orphan = tmp_path / "runs" / "R9-orphan"
    orphan.mkdir(parents=True)
    (orphan / "summary.json").write_text('{"passed": true}')
    row = index_runs(tmp_path)[0]
    assert row["manifest"] is False and row["claim"] is None
