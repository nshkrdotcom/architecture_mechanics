"""Every §6.1 metric against a value computed by hand, not by the code.

A metric test that asserts ``metric(oracle) == 1.0`` proves the oracle is the
oracle, not that the metric is right. So the fixture here is four positions and
six features, small enough that every expected number below is worked out in the
comment beside it and can be checked with a pencil.

The fixture, once, so each test can point at it:

    F = 6 content features, T = 4 positions, one example.
    position 0: source, active {0, 2}, values 0.8 and 0.4
    position 1: filler, active {1},    value 0.5
    position 2: distractor
    position 3: query, supervised; the answer is position 0's content
    importance w = (1, 1, 1, 1, 1, 1) unless a test says otherwise
"""

from __future__ import annotations

import numpy as np
import pytest

from architecture_mechanics.data.feature_program import ProgramRecord
from architecture_mechanics.metrics.capability import (
    CapabilityMetricError,
    EvaluationReference,
    OverwriteStep,
    Predictions,
    answer_set_accuracy,
    associative_recall_accuracy,
    associative_recall_jaccard,
    brier_score,
    calibration,
    distance_degradation,
    distractor_sensitivity,
    expected_calibration_error,
    feature_detection,
    heldout_composition_accuracy,
    normalized_skill,
    overwrite_accuracy,
    reconstruction_loss,
    stale_value_error_rate,
    threshold_sweep,
)

N_FEATURES = 6
SEQ_LEN = 4
SOURCE, QUERY = 0, 3


def _record(
    *,
    example_index: int = 0,
    template_id: str = "seen",
    answer: tuple[int, ...] = (0, 2),
    distance: int = 3,
    distractors: tuple[int, ...] = (2,),
    op: str = "recall_by_key",
) -> ProgramRecord:
    return ProgramRecord(
        example_index=example_index,
        family="T1",
        condition="fixture",
        split="test",
        template_id=template_id,
        composition=(op, 0),
        seq_len=SEQ_LEN,
        positions=(),
        steps=(
            OverwriteStep(
                op=op,
                dest=QUERY,
                source=SOURCE,
                key_id=0,
                distance=distance,
                distractors=distractors,
                answer_group=0,
                answer_features=answer,
                information_destroyed=False,
            ),
        ),
        supervised_positions=(QUERY,),
    )


@pytest.fixture
def reference() -> EvaluationReference:
    inputs = np.zeros((1, SEQ_LEN, N_FEATURES), dtype=np.float32)
    active = np.zeros((1, SEQ_LEN, N_FEATURES), dtype=bool)
    inputs[0, SOURCE, 0], inputs[0, SOURCE, 2] = 0.8, 0.4
    active[0, SOURCE, 0] = active[0, SOURCE, 2] = True
    inputs[0, 1, 1] = 0.5
    active[0, 1, 1] = True

    targets = np.zeros_like(inputs)
    target_active = np.zeros_like(active)
    targets[0, QUERY] = inputs[0, SOURCE]
    target_active[0, QUERY] = active[0, SOURCE]

    supervised = np.zeros((1, SEQ_LEN), dtype=bool)
    supervised[0, QUERY] = True

    return EvaluationReference(
        family="T1",
        condition="fixture",
        split="test",
        inputs=inputs,
        input_active=active,
        targets=targets,
        target_active=target_active,
        supervised=supervised,
        importance=np.ones(N_FEATURES, dtype=np.float64),
        content_indices=np.arange(N_FEATURES, dtype=np.int64),
        programs=(_record(),),
        heldout_template_ids=frozenset(),
    )


def _predict(values: dict[int, float], probs: dict[int, float] | None = None) -> Predictions:
    """Predictions at the query position only; everything else is zero."""
    value_array = np.zeros((1, SEQ_LEN, N_FEATURES), dtype=np.float64)
    prob_array = np.zeros((1, SEQ_LEN, N_FEATURES), dtype=np.float64)
    for feature, value in values.items():
        value_array[0, QUERY, feature] = value
    for feature, value in (probs if probs is not None else values).items():
        prob_array[0, QUERY, feature] = min(max(value, 0.0), 1.0)
    return Predictions(values=value_array, active_prob=prob_array)


# --------------------------------------------------------------------------- #
# reconstruction loss
# --------------------------------------------------------------------------- #


def test_reconstruction_loss_matches_a_hand_computed_value(reference):
    # truth at the query is (0.8, 0, 0.4, 0, 0, 0); prediction is (0.5, 0.1, 0.4, 0, 0, 0).
    # residuals    = (-0.3, 0.1, 0.0, 0, 0, 0)
    # squared      = (0.09, 0.01, 0.0, 0, 0, 0), sum = 0.10
    # weights all 1, sum = 6, so the loss is 0.10 / 6 = 0.0166666...
    result = reconstruction_loss(_predict({0: 0.5, 1: 0.1, 2: 0.4}), reference)
    assert result.value == pytest.approx(0.10 / 6)
    assert result.n == 1


def test_reconstruction_loss_honours_unequal_importance(reference):
    # Same prediction, but feature 0 now carries weight 3 and feature 1 weight 0.
    # weighted squared error = 3*0.09 + 0*0.01 + 1*0.0 = 0.27; sum of weights
    # = 3 + 0 + 1 + 1 + 1 + 1 = 7, so the loss is 0.27 / 7.
    weighted = EvaluationReference(
        **{
            **{k: getattr(reference, k) for k in reference.__dataclass_fields__ if k != "content_column"},
            "importance": np.array([3.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
        }
    )
    result = reconstruction_loss(_predict({0: 0.5, 1: 0.1, 2: 0.4}), weighted)
    assert result.value == pytest.approx(0.27 / 7)


def test_reconstruction_loss_is_zero_for_a_perfect_prediction(reference):
    # Built from the reference's own array rather than from 0.8 and 0.4 as
    # Python literals: the tensors are float32, so a float64 literal 0.8 differs
    # from the stored target in the last few bits and the loss would be 3e-17
    # rather than the exact zero the oracle must reach.
    perfect = Predictions(
        values=reference.targets.astype(np.float64),
        active_prob=reference.target_active.astype(np.float64),
    )
    assert reconstruction_loss(perfect, reference).value == 0.0


# --------------------------------------------------------------------------- #
# per-feature precision and recall
# --------------------------------------------------------------------------- #


def test_detection_at_the_rate_matched_point_matches_a_hand_computed_value(reference):
    # Two features are truly active ({0, 2}), so rate matching selects the two
    # highest-scoring cells. Scores are 0.9 (f0), 0.7 (f1), 0.3 (f2): it picks
    # {0, 1}. One of the two selected is correct, and one of the two true is
    # found, so precision = recall = f1 = 1/2.
    result = feature_detection(_predict({0: 0.9, 1: 0.7, 2: 0.3}), reference)
    assert result.n_true == 2 and result.n_selected == 2
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(0.5)
    assert result.f1 == pytest.approx(0.5)


def test_rate_matching_makes_precision_recall_and_f1_coincide(reference):
    for values in ({0: 0.9, 1: 0.7}, {2: 0.4, 4: 0.9, 5: 0.8}, {0: 0.1, 2: 0.2}):
        result = feature_detection(_predict(values), reference)
        assert result.precision == pytest.approx(result.recall)
        assert result.f1 == pytest.approx(result.precision)


def test_macro_averages_drop_features_with_an_empty_denominator(reference):
    # Only features 0 and 2 are ever truly active, so per-feature recall is
    # defined for exactly those two. Predicting {0, 1} gives recall 1.0 on
    # feature 0 and 0.0 on feature 2: the macro recall is 0.5, not 1/6.
    result = feature_detection(_predict({0: 0.9, 1: 0.7, 2: 0.3}), reference)
    defined = ~np.isnan(result.per_feature_recall)
    assert defined.sum() == 2
    assert result.macro_recall == pytest.approx(0.5)


def test_a_fixed_threshold_can_select_nothing_and_says_so(reference):
    result = feature_detection(_predict({0: 0.2, 2: 0.1}), reference, threshold=0.5)
    assert result.n_selected == 0
    assert np.isnan(result.precision)
    assert result.recall == 0.0


# --------------------------------------------------------------------------- #
# associative recall
# --------------------------------------------------------------------------- #


def test_answer_set_accuracy_is_set_equality_not_overlap(reference):
    # Truth is {0, 2}. Rate matching selects two cells.
    assert answer_set_accuracy(_predict({0: 0.9, 2: 0.8}), reference).value == 1.0
    assert answer_set_accuracy(_predict({0: 0.9, 1: 0.8}), reference).value == 0.0


def test_jaccard_grades_a_near_miss_that_exact_accuracy_calls_wrong(reference):
    # Predicted {0, 1} against truth {0, 2}: intersection 1, union 3.
    predictions = _predict({0: 0.9, 1: 0.8})
    assert associative_recall_accuracy(predictions, reference).value == 0.0
    assert associative_recall_jaccard(predictions, reference).value == pytest.approx(1 / 3)


def test_associative_recall_ignores_reconstruction_steps(reference):
    local = EvaluationReference(
        **{
            **{k: getattr(reference, k) for k in reference.__dataclass_fields__ if k != "content_column"},
            "programs": (_record(op="reconstruct"),),
        }
    )
    predictions = _predict({0: 0.9, 2: 0.8})
    assert associative_recall_accuracy(predictions, local).value is None
    assert associative_recall_accuracy(predictions, local).n == 0
    # ...but the general form still scores it.
    assert answer_set_accuracy(predictions, local).value == 1.0


# --------------------------------------------------------------------------- #
# overwrite and stale value
# --------------------------------------------------------------------------- #


@pytest.fixture
def overwrite_reference(reference) -> EvaluationReference:
    """Position 1 becomes the superseded binding: active {1}, value 0.5."""
    step = OverwriteStep(
        op="overwrite_recall",
        dest=QUERY,
        source=SOURCE,
        key_id=0,
        distance=3,
        distractors=(2,),
        answer_group=0,
        answer_features=(0, 2),
        information_destroyed=False,
        stale_source=1,
        stale_answer_features=(1,),
    )
    record = ProgramRecord(
        example_index=0,
        family="T2",
        condition="fixture",
        split="test",
        template_id="seen",
        composition=("overwrite_recall", 0),
        seq_len=SEQ_LEN,
        positions=(),
        steps=(step,),
        supervised_positions=(QUERY,),
    )
    return EvaluationReference(
        **{
            **{k: getattr(reference, k) for k in reference.__dataclass_fields__ if k != "content_column"},
            "programs": (record,),
        }
    )


def test_overwrite_accuracy_scores_the_newest_lawful_value(overwrite_reference):
    assert overwrite_accuracy(_predict({0: 0.9, 2: 0.8}), overwrite_reference).value == 1.0
    # Returning the superseded set {1} instead is wrong, however confident.
    assert overwrite_accuracy(_predict({1: 0.9, 3: 0.1}), overwrite_reference).value == 0.0


def test_stale_value_error_rate_matches_a_hand_computed_verdict(overwrite_reference):
    # current value  = (0.8, 0, 0.4, 0, 0, 0)   [position 0]
    # stale value    = (0, 0.5, 0, 0, 0, 0)     [position 1]
    # A prediction of (0, 0.5, 0, ...) is exactly the stale vector: distance to
    # stale is 0 and to current is 0.8^2 + 0.4^2 = 0.80, so it is a stale error.
    assert stale_value_error_rate(_predict({1: 0.5}), overwrite_reference).value == 1.0
    # A prediction equal to the current value is distance 0 from it: not an error.
    assert stale_value_error_rate(_predict({0: 0.8, 2: 0.4}), overwrite_reference).value == 0.0


def test_an_exact_tie_counts_as_half_an_error(overwrite_reference):
    # The midpoint of the two vectors is equidistant from both by construction.
    # Computed from the stored arrays rather than written out by hand, so the
    # tie is exact in floating point and the branch is really exercised.
    current = overwrite_reference.targets[0, QUERY].astype(np.float64)
    stale = overwrite_reference.inputs[0, 1].astype(np.float64)
    values = np.zeros((1, SEQ_LEN, N_FEATURES), dtype=np.float64)
    values[0, QUERY] = (current + stale) / 2
    midpoint = Predictions(values=values, active_prob=np.clip(values, 0.0, 1.0))
    assert stale_value_error_rate(midpoint, overwrite_reference).value == 0.5


def test_overwrite_metrics_are_absent_rather_than_zero_without_t2_steps(reference):
    result = overwrite_accuracy(_predict({0: 0.9}), reference)
    assert result.value is None and result.n == 0
    assert "prompt 18" in result.detail["reason"]
    assert stale_value_error_rate(_predict({0: 0.9}), reference).value is None


# --------------------------------------------------------------------------- #
# distance, distractors, held-out composition
# --------------------------------------------------------------------------- #


def _multi_example(reference, records, correct: list[bool]) -> tuple[Predictions, EvaluationReference]:
    """Repeat the fixture example once per record, answering right or wrong."""
    n = len(records)
    tile = {
        key: np.repeat(getattr(reference, key), n, axis=0)
        for key in ("inputs", "input_active", "targets", "target_active", "supervised")
    }
    multi = EvaluationReference(
        **{
            **{
                k: getattr(reference, k)
                for k in reference.__dataclass_fields__
                if k not in {"content_column", *tile}
            },
            **tile,
            "programs": tuple(records),
        }
    )
    values = np.zeros((n, SEQ_LEN, N_FEATURES), dtype=np.float64)
    probs = np.zeros((n, SEQ_LEN, N_FEATURES), dtype=np.float64)
    for row, right in enumerate(correct):
        chosen = (0, 2) if right else (1, 3)
        values[row, QUERY, list(chosen)] = 0.9
        probs[row, QUERY, list(chosen)] = 0.9
    return Predictions(values=values, active_prob=probs), multi


def test_distance_degradation_buckets_by_distance_and_slopes_downward(reference):
    records = [
        _record(example_index=i, distance=d)
        for i, d in enumerate([1, 1, 2, 2])
    ]
    predictions, multi = _multi_example(reference, records, [True, True, True, False])
    curve = distance_degradation(predictions, multi)
    # distance 1: both right -> 1.0. distance 2: one of two -> 0.5.
    assert curve.x == (1.0, 2.0) and curve.y == (1.0, 0.5) and curve.n == (2, 2)
    # slope over two equally weighted points one apart is exactly the drop.
    assert curve.slope == pytest.approx(-0.5)


def test_distractor_sensitivity_has_no_slope_from_a_single_bucket(reference):
    records = [_record(example_index=i) for i in range(2)]
    predictions, multi = _multi_example(reference, records, [True, False])
    curve = distractor_sensitivity(predictions, multi)
    assert curve.x == (1.0,) and curve.y == (0.5,)
    assert curve.slope is None, "one bucket must give an undefined slope, never zero"


def test_heldout_composition_accuracy_separates_seen_from_unseen(reference):
    records = [
        _record(example_index=0, template_id="seen"),
        _record(example_index=1, template_id="seen"),
        _record(example_index=2, template_id="unseen"),
        _record(example_index=3, template_id="unseen"),
    ]
    predictions, multi = _multi_example(reference, records, [True, True, True, False])
    multi = EvaluationReference(
        **{
            **{k: getattr(multi, k) for k in multi.__dataclass_fields__ if k != "content_column"},
            "heldout_template_ids": frozenset({"unseen"}),
        }
    )
    result = heldout_composition_accuracy(predictions, multi)
    assert result.value == pytest.approx(0.5)
    assert result.detail["seen_accuracy"] == pytest.approx(1.0)
    assert result.detail["gap"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #


def test_brier_and_ece_match_hand_computed_values(reference):
    # Six scored cells at the query. Truth is (1, 0, 1, 0, 0, 0).
    # Predicted probabilities are (0.9, 0.2, 0.0, 0.0, 0.0, 0.0).
    # Brier = ((0.1)^2 + (0.2)^2 + (1.0)^2 + 0 + 0 + 0) / 6
    #       = (0.01 + 0.04 + 1.00) / 6 = 1.05 / 6 = 0.175
    result = calibration(_predict({0: 0.9, 1: 0.2}, probs={0: 0.9, 1: 0.2}), reference)
    assert result.brier == pytest.approx(1.05 / 6)
    # ECE over ten equal-width bins. Bin 0 [0.0, 0.1) holds four cells: three
    # inactive and feature 2 which is active, so confidence 0.0 and accuracy
    # 0.25 -> |0.25 - 0.0| = 0.25 at weight 4/6. Bin 2 [0.2, 0.3) holds one
    # inactive cell at confidence 0.2 -> 0.2 at weight 1/6. Bin 9 [0.9, 1.0]
    # holds feature 0, active, at confidence 0.9 -> 0.1 at weight 1/6.
    expected = (4 / 6) * 0.25 + (1 / 6) * 0.2 + (1 / 6) * 0.1
    assert result.ece == pytest.approx(expected)
    # The MetricValue wrappers must report the same numbers as the bundle.
    predictions = _predict({0: 0.9, 1: 0.2})
    assert brier_score(predictions, reference).value == pytest.approx(1.05 / 6)
    assert expected_calibration_error(predictions, reference).value == pytest.approx(expected)


def test_a_constant_base_rate_predictor_is_calibrated_while_being_useless(reference):
    # Two of six cells are active, so a predictor emitting 1/3 everywhere is
    # perfectly calibrated: one bin, confidence 1/3, accuracy 1/3, ECE 0.
    values = np.full((1, SEQ_LEN, N_FEATURES), 1 / 3)
    predictions = Predictions(values=values, active_prob=values)
    result = calibration(predictions, reference)
    assert result.ece == pytest.approx(0.0, abs=1e-12)
    # ...and it recovers nothing: this is exactly why ECE is retired.
    assert answer_set_accuracy(predictions, reference).value == 0.0


# --------------------------------------------------------------------------- #
# threshold sweep and skill normalisation
# --------------------------------------------------------------------------- #


def test_recall_is_free_at_a_threshold_of_zero(reference):
    sweep = threshold_sweep(_predict({0: 0.0}), reference)
    assert sweep.best_recall == 1.0, "a threshold of zero selects every cell"
    assert sweep.best_precision < 1.0


def test_normalized_skill_pins_the_marginal_to_zero_and_the_oracle_to_one():
    assert normalized_skill(0.5, 0.5, 1.0) == 0.0
    assert normalized_skill(1.0, 0.5, 1.0) == 1.0
    assert normalized_skill(0.75, 0.5, 1.0) == pytest.approx(0.5)
    # Loss-like metrics need no separate sign convention: the references carry it.
    assert normalized_skill(0.02, 0.04, 0.0) == pytest.approx(0.5)
    # An ill-conditioned normalisation is reported, not divided through.
    assert normalized_skill(0.5, 0.5, 0.5) is None


# --------------------------------------------------------------------------- #
# shape and contract errors
# --------------------------------------------------------------------------- #


def test_mismatched_prediction_shape_is_rejected(reference):
    bad = Predictions(
        values=np.zeros((1, SEQ_LEN, N_FEATURES + 1)),
        active_prob=np.zeros((1, SEQ_LEN, N_FEATURES + 1)),
    )
    with pytest.raises(CapabilityMetricError, match="do not match"):
        reconstruction_loss(bad, reference)


def test_probabilities_outside_the_unit_interval_are_rejected():
    with pytest.raises(CapabilityMetricError, match=r"\[0, 1\]"):
        Predictions(values=np.zeros((1, 2, 2)), active_prob=np.full((1, 2, 2), 1.5))


def test_a_reference_needs_one_program_record_per_row(reference):
    with pytest.raises(CapabilityMetricError, match="positional"):
        EvaluationReference(
            **{
                **{
                    k: getattr(reference, k)
                    for k in reference.__dataclass_fields__
                    if k != "content_column"
                },
                "programs": (),
            }
        )
