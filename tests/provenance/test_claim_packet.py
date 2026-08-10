"""Pre-registration packets, and the one thing a claim may never do.

Two properties are under test. A packet with a blank field is refused — "a
pre-registration with a blank ``KILL_CONDITION`` is not a pre-registration".
And a rung is passed only by code that measured something: there is no path
from a config, a flag, or a hand edit to a supported claim.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import yaml

from architecture_mechanics.experiments.claim_packet import (
    REQUIRED_FIELDS,
    RUNGS,
    ClaimGates,
    ClaimPacket,
    ClaimPacketError,
    RungEvaluation,
    evaluate_rungs,
    load_gates,
    load_packet,
)


def _fields() -> dict:
    return {
        "CLAIM": "the mechanism does the thing",
        "MECHANISM": "attention",
        "STRUCTURALLY_ENFORCED_PROPERTIES": ["causality", "normalisation"],
        "LEARNED_OR_HOPED_PROPERTIES": ["sparse routing"],
        "NEAREST_BORING_EXPLANATION": "it memorised the training split",
        "CONTROL_THAT_RULES_IT_OUT": "held-out compositions",
        "PRIMARY_METRIC": "exact answer-set recall",
        "MECHANISM_ACTIVITY_METRIC": "retrieval lift over chance",
        "POSITIVE_CONTROL": "R1 known-easy control",
        "NEGATIVE_CONTROL": "information-destroyed condition",
        "KILL_CONDITION": "recall below 0.80",
        "REPLICATION_REQUIREMENT": "five seeds",
    }


def _packet(**overrides) -> ClaimPacket:
    packet = ClaimPacket(claim_id="k0-test", claimed_rung=1, fields=_fields())
    for key, value in overrides.items():
        if key in REQUIRED_FIELDS:
            packet.fields[key] = value
        else:
            setattr(packet, key, value)
    return packet


def test_a_complete_packet_validates():
    _packet().validate()


@pytest.mark.parametrize("name", REQUIRED_FIELDS)
def test_every_field_is_refused_when_missing(name):
    packet = _packet()
    del packet.fields[name]
    with pytest.raises(ClaimPacketError, match=name):
        packet.validate()


@pytest.mark.parametrize("name", REQUIRED_FIELDS)
@pytest.mark.parametrize("empty", ["", "   ", [], [""], ["ok", " "]])
def test_every_field_is_refused_when_empty(name, empty):
    with pytest.raises(ClaimPacketError, match=name):
        _packet(**{name: empty}).validate()


def test_a_blank_kill_condition_is_not_a_pre_registration():
    """The named case from the mission brief, spelled out on its own."""
    with pytest.raises(ClaimPacketError, match="KILL_CONDITION is empty"):
        _packet(KILL_CONDITION="").validate()


@pytest.mark.parametrize("rung", [-1, 7, "1", 1.0, True, None])
def test_claimed_rung_must_be_an_integer_on_the_ladder(rung):
    with pytest.raises(ClaimPacketError, match="claimed_rung"):
        _packet(claimed_rung=rung).validate()


def test_unrecognised_fields_are_refused():
    packet = _packet()
    packet.fields["KILL_CONDITIONS"] = "a plural typo silently disarms the real one"
    with pytest.raises(ClaimPacketError, match="unrecognised"):
        packet.validate()


def test_claim_id_must_match_its_filename(tmp_path: Path):
    packet = _packet()
    with pytest.raises(ClaimPacketError, match="does not match filename"):
        packet.write(tmp_path / "something-else.yml")


def test_an_invalid_packet_never_reaches_disk(tmp_path: Path):
    path = tmp_path / "k0-test.yml"
    with pytest.raises(ClaimPacketError):
        _packet(KILL_CONDITION="").write(path)
    assert not path.exists()


def test_packet_round_trips_through_yaml(tmp_path: Path):
    written = _packet().write(tmp_path / "k0-test.yml")
    reloaded = load_packet(written)
    assert reloaded.fields == _fields()
    assert reloaded.claimed_rung == 1
    assert yaml.safe_load(written.read_text())["CLAIM"] == "the mechanism does the thing"


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #


@dataclass
class _FakeResult:
    """The shape :func:`evaluate_rungs` reads. Deliberately not a real run."""

    run_id: str = "R1-fake"
    config: dict = field(default_factory=lambda: {"optim": {"max_steps": 10}})
    checks: dict = field(default_factory=lambda: {"shapes": {"ok": True, "detail": ""}})
    history: list = field(default_factory=lambda: [{"step": 1, "train_loss": 0.5}])
    final: dict = field(default_factory=lambda: {"associative_recall_accuracy": 0.9})
    mechanism: dict = field(
        default_factory=lambda: {
            "verdict": {"active": True, "best_off_diagonal_mass": 0.7, "reasons": []},
            "mechanism_version": "mech-1.0.0",
        }
    )


def test_a_healthy_run_supports_rungs_zero_and_one():
    evaluations = evaluate_rungs(_FakeResult(), run_dir="runs/R1-fake")
    assert [(e.rung, e.passed) for e in evaluations] == [(0, True), (1, True)]
    assert evaluations[1].measured["best_off_diagonal_mass"] == 0.7


def test_a_failed_invariant_takes_down_rung_zero_and_everything_above_it():
    result = _FakeResult(checks={"causal_masking": {"ok": False, "detail": "leak"}})
    evaluations = evaluate_rungs(result, run_dir="runs/R1-fake")
    assert [(e.rung, e.passed) for e in evaluations] == [(0, False), (1, False)]
    assert evaluations[1].measured["active"] is True, "the measurement is still recorded"


def test_a_non_finite_loss_takes_down_rung_zero():
    result = _FakeResult(history=[{"step": 1, "train_loss": float("nan")}])
    assert evaluate_rungs(result, run_dir="runs/x")[0].passed is False


def test_an_inert_mechanism_fails_rung_one_only():
    result = _FakeResult(mechanism={"verdict": {"active": False, "reasons": ["entropy_ratio"]}})
    evaluations = evaluate_rungs(result, run_dir="runs/x")
    assert [(e.rung, e.passed) for e in evaluations] == [(0, True), (1, False)]


def test_rungs_two_and_above_are_never_evaluated_from_one_run():
    """Absent means not evaluated. One run of one architecture cannot support a
    statement about a difference between architectures replicating."""
    assert {e.rung for e in evaluate_rungs(_FakeResult(), run_dir="runs/x")} == {0, 1}


def test_gates_refuse_anything_that_is_not_a_measurement():
    gates = ClaimGates(claim_id="k0-test")
    for impostor in [
        {"rung": 0, "passed": True, "evidence": ["runs/x"]},
        ("0_implementation_survives", True),
        "0_implementation_survives",
    ]:
        with pytest.raises(ClaimPacketError, match="only a RungEvaluation"):
            gates.record(impostor, source="runs/x")


def test_gates_refuse_a_pass_with_no_evidence():
    gates = ClaimGates(claim_id="k0-test")
    forged = RungEvaluation(rung=0, passed=True, evidence=(), measured={}, evaluated_by="hand")
    with pytest.raises(ClaimPacketError, match="no evidence"):
        gates.record(forged, source="runs/x")


def test_highest_supported_rung_stops_at_the_first_gap():
    gates = ClaimGates(claim_id="k0-test")
    assert gates.highest_supported_rung is None

    gates.record(
        RungEvaluation(0, True, ("runs/a",), {}, "test"), source="runs/a"
    )
    assert gates.highest_supported_rung == 0

    # Rung 2 without rung 1 supports nothing: §7.5's ladder has no missing steps.
    gates.rungs["2_capability_difference_replicates"] = {
        "passed": True,
        "evidence": ["runs/b"],
        "evaluations": [],
    }
    assert gates.highest_supported_rung == 0


def test_highest_supported_rung_is_recomputed_not_stored(tmp_path: Path):
    """Raising it by hand survives exactly until the next write."""
    gates = ClaimGates(claim_id="k0-test")
    gates.record(RungEvaluation(0, True, ("runs/a",), {}, "test"), source="runs/a")
    path = gates.write(tmp_path / "k0-test.gates.json")

    tampered = json.loads(path.read_text())
    tampered["highest_supported_rung"] = 5
    path.write_text(json.dumps(tampered))

    assert load_gates(path).as_dict()["highest_supported_rung"] == 0


def test_a_hand_edited_pass_is_refused_on_read(tmp_path: Path):
    path = tmp_path / "k0-test.gates.json"
    path.write_text(
        json.dumps(
            {
                "claim_id": "k0-test",
                "rungs": {"0_implementation_survives": {"passed": True, "evidence": []}},
            }
        )
    )
    with pytest.raises(ClaimPacketError, match="no evidence"):
        load_gates(path)


def test_a_pass_above_a_gap_is_refused_on_read(tmp_path: Path):
    path = tmp_path / "k0-test.gates.json"
    path.write_text(
        json.dumps(
            {
                "claim_id": "k0-test",
                "rungs": {
                    "0_implementation_survives": {"passed": False, "evidence": []},
                    "1_mechanism_is_active": {"passed": True, "evidence": ["runs/x"]},
                },
            }
        )
    )
    with pytest.raises(ClaimPacketError, match="above an unpassed rung"):
        load_gates(path)


def test_a_failing_run_does_not_erase_a_passing_one_but_is_still_recorded():
    gates = ClaimGates(claim_id="k0-test")
    gates.record(RungEvaluation(1, True, ("runs/a",), {"active": True}, "t"), source="runs/a")
    gates.record(RungEvaluation(1, False, (), {"active": False}, "t"), source="runs/b")

    entry = gates.as_dict()["rungs"]["1_mechanism_is_active"]
    assert entry["passed"] is True
    assert entry["evidence"] == ["runs/a"]
    assert [record["passed"] for record in entry["evaluations"]] == [True, False]


def test_re_running_the_same_run_is_not_a_replication():
    gates = ClaimGates(claim_id="k0-test")
    for _ in range(3):
        gates.record(RungEvaluation(0, True, ("runs/a",), {}, "t"), source="runs/a")
    entry = gates.as_dict()["rungs"]["0_implementation_survives"]
    assert entry["evidence"] == ["runs/a"]
    assert len(entry["evaluations"]) == 1


def test_gates_file_matches_the_shape_the_gate_reads(tmp_path: Path):
    """check_claims.sh reads rungs[f'{i}_{RUNGS[i]}'].passed/.evidence and
    highest_supported_rung. Nothing else about the file is contractual."""
    gates = ClaimGates(claim_id="k0-test")
    gates.record(RungEvaluation(0, True, ("runs/a",), {}, "t"), source="runs/a")
    gates.record(RungEvaluation(1, True, ("runs/a",), {}, "t"), source="runs/a")
    record = json.loads(gates.write(tmp_path / "k0-test.gates.json").read_text())

    assert record["highest_supported_rung"] == 1
    for index in (0, 1):
        entry = record["rungs"][f"{index}_{RUNGS[index]}"]
        assert entry["passed"] is True
        assert entry["evidence"]
