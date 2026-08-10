"""Predeclared primary comparisons, and the record they leave in ``summary.json``.

Two things are being checked here, and only the second is about arithmetic.

The first is that the metric a comparison is decided on cannot be chosen at call
time. §7.4 asks for "predeclared primary comparisons"; that is only a discipline
if the declaration and the prediction cannot come apart, and the place the
prediction lives is the committed claim packet whose commit time
``bin/check_prereg.sh`` already compares against the run. So
:func:`primary_comparison` has no ``metric`` parameter at all, and a comparison
file that names a different metric than its packet is refused rather than
reconciled.

The second is that the record survives the round trip. Prompts 15 and 21 write
it into evidence bundles and prompts 22 and 27 read it back, and a record that
lost its per-seed values or its p-value floor somewhere in ``json.dumps`` would
be discovered by whoever was writing the paper.
"""

from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

from architecture_mechanics.experiments.claim_packet import REQUIRED_FIELDS, ClaimPacket
from architecture_mechanics.experiments.runner import RunResult
from architecture_mechanics.metrics.statistics import (
    COMPARISON_SCHEMA,
    SUMMARY_KEY,
    ComparisonRecord,
    RunSummary,
    StatisticsError,
    attach_comparisons,
    comparisons_from_summary,
    load_comparison,
    primary_comparison,
    primary_metric_for,
    secondary_comparison,
)

METRIC = "associative_recall_accuracy"


@pytest.fixture
def claims_dir(tmp_path):
    directory = tmp_path / "claims"
    ClaimPacket(
        claim_id="delta-overwrites-cleanly",
        claimed_rung=2,
        fields={name: f"a real sentence for {name}" for name in REQUIRED_FIELDS},
        primary_metric_key=METRIC,
    ).write(directory / "delta-overwrites-cleanly.yml")
    return directory


@pytest.fixture
def declaration(tmp_path):
    path = tmp_path / "a0-vs-a2.json"
    path.write_text(json.dumps({
        "claim": "delta-overwrites-cleanly",
        "control_run": "R4-softmax-s1",
        "candidate_runs": ["R4-delta-s1"],
        "matching_strategy": "width_matched",
        "permitted_differences": {},
    }))
    return path


def arms(control_values, candidate_values, *, cell: str = "all"):
    control = [
        RunSummary(run_id=f"R4-softmax-{cell}-s{seed}", seed=seed, arm="softmax", cell=cell,
                   metrics={METRIC: value, "brier": 0.01 * (seed + 1)})
        for seed, value in enumerate(control_values)
    ]
    candidate = [
        RunSummary(run_id=f"R4-delta-{cell}-s{seed}", seed=seed, arm="delta_memory", cell=cell,
                   metrics={METRIC: value, "brier": 0.01 * (seed + 1)})
        for seed, value in enumerate(candidate_values)
    ]
    return control, candidate


# --------------------------------------------------------------------------- #
# The metric comes from the packet
# --------------------------------------------------------------------------- #


def test_primary_comparison_has_no_metric_parameter():
    """The mechanism, stated as a signature. Everything else about a comparison
    can be decided when it is run; which number decides it cannot."""
    assert "metric" not in inspect.signature(primary_comparison).parameters


def test_the_metric_is_read_from_the_claim_packet(declaration, claims_dir):
    metric, source = primary_metric_for(load_comparison(declaration), claims_dir=claims_dir)
    assert metric == METRIC
    assert source.endswith("delta-overwrites-cleanly.yml#primary_metric_key")


def test_a_comparison_that_renames_the_metric_is_refused(tmp_path, claims_dir):
    path = tmp_path / "renamed.json"
    path.write_text(json.dumps({
        "claim": "delta-overwrites-cleanly",
        "control_run": "c", "candidate_runs": ["k"],
        "primary_metric": "reconstruction_loss",
    }))
    with pytest.raises(StatisticsError, match="predeclared"):
        primary_metric_for(load_comparison(path), claims_dir=claims_dir)


def test_a_comparison_that_echoes_the_metric_correctly_is_accepted(tmp_path, claims_dir):
    path = tmp_path / "echoed.json"
    path.write_text(json.dumps({
        "claim": "delta-overwrites-cleanly",
        "control_run": "c", "candidate_runs": ["k"],
        "primary_metric": METRIC,
    }))
    metric, _ = primary_metric_for(load_comparison(path), claims_dir=claims_dir)
    assert metric == METRIC


def test_a_comparison_naming_no_claim_is_refused(tmp_path):
    path = tmp_path / "unclaimed.json"
    path.write_text(json.dumps({"control_run": "c", "candidate_runs": ["k"]}))
    with pytest.raises(StatisticsError, match="names no claim"):
        load_comparison(path)


def test_a_claim_that_exists_only_in_prose_is_refused(tmp_path, declaration):
    """``PRIMARY_METRIC`` is a sentence for a human. Without
    ``primary_metric_key`` there is nothing to compare against a number, and the
    fix is a commit, not an argument."""
    directory = tmp_path / "prose-claims"
    ClaimPacket(
        claim_id="delta-overwrites-cleanly",
        claimed_rung=2,
        fields={name: f"a real sentence for {name}" for name in REQUIRED_FIELDS},
    ).write(directory / "delta-overwrites-cleanly.yml")
    with pytest.raises(StatisticsError, match="primary_metric_key"):
        primary_metric_for(load_comparison(declaration), claims_dir=directory)


def test_a_missing_packet_is_refused(tmp_path, declaration):
    with pytest.raises(StatisticsError, match="not in"):
        primary_metric_for(load_comparison(declaration), claims_dir=tmp_path / "nowhere")


# --------------------------------------------------------------------------- #
# The record
# --------------------------------------------------------------------------- #


def test_the_record_carries_everything_section_7_4_asks_for(declaration, claims_dir):
    control, candidate = arms([0.80, 0.82, 0.84, 0.86, 0.88], [0.85, 0.86, 0.90, 0.91, 0.92])
    record = primary_comparison(declaration, control, candidate, claims_dir=claims_dir,
                                resamples=1000)

    assert record.primary is True
    assert record.metric == METRIC
    assert record.claim_id == "delta-overwrites-cleanly"
    assert record.n_seeds == 5
    assert record.seeds == (0, 1, 2, 3, 4)

    # per-seed raw values, both arms
    assert len(record.per_run) == 10
    assert {row["arm"] for row in record.per_run} == {"softmax", "delta_memory"}
    assert len(record.per_seed_difference) == 5

    # effect size, interval, test, seed count
    assert record.effect.estimate == pytest.approx(0.048)
    assert record.effect.ci_low <= record.effect.estimate <= record.effect.ci_high
    assert record.effect.ci_method == "studentized"
    assert record.standardized.name == "cohens_dz"
    assert record.test.test == "paired_t"
    assert 0.0 <= record.test.p_value <= 1.0
    assert record.test.p_value_floor == 0.0


def test_the_record_is_data_and_not_a_formatted_string(declaration, claims_dir):
    control, candidate = arms([0.1, 0.2, 0.3], [0.2, 0.3, 0.4])
    record = primary_comparison(declaration, control, candidate, claims_dir=claims_dir,
                                resamples=500)
    payload = record.as_dict()
    assert payload["schema"] == COMPARISON_SCHEMA
    assert isinstance(payload["per_run"], list)
    assert isinstance(payload["effect"], dict)
    assert all(not isinstance(value, str) or "\n" not in value for value in payload.values())


def test_a_difficulty_matrix_reduces_to_the_seed_and_keeps_its_cells_exploratory(
    declaration, claims_dir
):
    rng = np.random.default_rng(0)
    control, candidate = [], []
    for cell in ("sparse", "dense", "long"):
        control_values = 0.8 + rng.standard_normal(5) * 0.02
        left, right = arms(control_values, control_values + 0.03, cell=cell)
        control += left
        candidate += right
    record = primary_comparison(declaration, control, candidate, claims_dir=claims_dir,
                                resamples=1000)
    assert record.n_seeds == 5
    assert record.n_cells == 3
    assert record.effect.n == 5, "the interval is over seeds, not over the fifteen runs"
    assert len(record.exploratory_cells) == 3
    assert all(row["exploratory"] for row in record.exploratory_cells)
    assert record.variance_components


def test_a_secondary_comparison_says_it_is_secondary():
    control, candidate = arms([0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5])
    record = secondary_comparison("brier-side-look", control, candidate, "brier",
                                  claim_id="delta-overwrites-cleanly", resamples=500)
    assert record.primary is False
    assert record.metric_source == "caller"


# --------------------------------------------------------------------------- #
# summary.json
# --------------------------------------------------------------------------- #


def test_a_record_round_trips_through_summary_json(declaration, claims_dir):
    control, candidate = arms([0.80, 0.82, 0.84, 0.86, 0.88], [0.85, 0.86, 0.90, 0.91, 0.92])
    record = primary_comparison(declaration, control, candidate, claims_dir=claims_dir,
                                resamples=500)
    summary = attach_comparisons({"run_id": "R4-delta-s1", "final": {METRIC: 0.9}}, [record])
    restored = comparisons_from_summary(json.loads(json.dumps(summary)))
    assert len(restored) == 1
    assert restored[0].as_dict() == record.as_dict()
    assert restored[0].test.p_value_floor == record.test.p_value_floor


def test_summary_json_accepts_only_a_record_the_estimators_produced():
    """Same discipline as ``ClaimGates.record`` accepting only a ``RungEvaluation``:
    a dict that merely looks like a measurement could have been written by
    anything."""
    control, candidate = arms([0.1, 0.2, 0.3], [0.2, 0.3, 0.4])
    record = secondary_comparison("s", control, candidate, "brier",
                                  claim_id="c", resamples=200)
    with pytest.raises(StatisticsError, match="ComparisonRecord"):
        attach_comparisons({}, [record.as_dict()])


def test_a_record_with_the_wrong_schema_is_refused():
    with pytest.raises(StatisticsError, match="schema"):
        ComparisonRecord.from_dict({"schema": "am.comparison.v0"})


def test_a_summary_with_no_comparisons_reads_back_as_none():
    assert comparisons_from_summary({"run_id": "x"}) == ()


# --------------------------------------------------------------------------- #
# The runner's slot
# --------------------------------------------------------------------------- #


def _run_result(**extra) -> RunResult:
    return RunResult(
        run_id="R4-delta-s1", config={}, device={}, seeding={}, model={}, parameters={}, **extra
    )


def test_a_run_that_compared_nothing_writes_no_comparisons_key():
    """A single run cannot have a comparison — it is a statement about a set of
    runs — so the key is absent rather than empty, and every summary recorded
    before this module existed is byte-for-byte unchanged."""
    assert SUMMARY_KEY not in _run_result().as_dict()


def test_a_run_carrying_records_writes_them_into_its_summary():
    control, candidate = arms([0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5])
    record = secondary_comparison("s", control, candidate, "brier",
                                  claim_id="c", resamples=200)
    payload = attach_comparisons(_run_result().as_dict(), [record])
    result = _run_result(comparisons=payload[SUMMARY_KEY])

    on_disk = json.loads(json.dumps(result.as_dict()))
    assert comparisons_from_summary(on_disk)[0].as_dict() == record.as_dict()
