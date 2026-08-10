"""The rulers against known answers: oracle, chance, marginal, and the ceiling.

The fixture suite proves each metric computes what it says. This one proves the
metrics are *worth computing*: that the oracle reaches the top of each scale,
that a predictor which knows only feature frequencies reaches the bottom, and
that the positive control's threshold sits between them with room on both sides.

Everything here runs on a small number of examples so the suite stays fast. The
recorded calibration numbers in ``state/03_t0_metrics.md`` come from the default
size; these assertions are deliberately written as inequalities with margins
rather than as exact reproductions of those numbers, because a test that pins a
measured value to four decimals fails for reasons that are not defects.
"""

from __future__ import annotations

import numpy as np
import pytest

from architecture_mechanics.data.feature_program import (
    condition_config,
    generate_dataset,
    t0_config,
)
from architecture_mechanics.metrics.capability import (
    CEILING_DOMINATED_METRICS,
    METRIC_SPEC_BY_NAME,
    METRIC_SPECS,
    POSITIVE_CONTROL_THRESHOLD,
    RETIRED_METRICS,
    EvaluationReference,
    Predictions,
    ProgramOracle,
    RandomBaseline,
    associative_recall_accuracy,
    calibrate,
    evaluate_all,
    fit_frequency_ceiling,
    fit_marginal,
    metric_rule,
    positive_control,
    positive_control_datasets,
    reconstruction_loss,
    score_condition,
    synthetic_overwrite_reference,
    threshold_sweep,
)

N = 96


@pytest.fixture(scope="module")
def t0():
    train = generate_dataset(t0_config(n_examples=N, split="train"))
    evaluation = generate_dataset(t0_config(n_examples=N, split="test"))
    reference = EvaluationReference.from_dataset(evaluation)
    marginal = fit_marginal(train)
    return {
        "train": train,
        "eval": evaluation,
        "reference": reference,
        "marginal": marginal,
        "oracle": ProgramOracle(fallback=marginal).predict(evaluation),
        "marginal_pred": marginal.predict(evaluation),
        "random": RandomBaseline.fit(train).predict(evaluation),
        "ceiling": fit_frequency_ceiling(reference).predict_like(reference),
    }


@pytest.fixture(scope="module")
def report():
    return calibrate(n_examples=N)


# --------------------------------------------------------------------------- #
# The oracle reaches the ceiling
# --------------------------------------------------------------------------- #


def test_the_oracle_reconstructs_the_target_exactly(t0):
    assert reconstruction_loss(t0["oracle"], t0["reference"]).value == 0.0


def test_the_oracle_tops_out_every_retained_score_metric(t0):
    measured = evaluate_all(t0["oracle"], t0["reference"])
    for spec in METRIC_SPECS:
        if spec.status != "retained" or spec.kind != "score":
            continue
        value = measured[spec.name].value
        if value is None:
            continue
        assert value == pytest.approx(1.0), spec.name


def test_the_oracle_reads_the_program_and_not_the_target(t0):
    """The oracle must fail where the program says the route is destroyed.

    An oracle that scored perfectly on the information-destroyed control would
    be reading the answer key rather than reconstructing from the program, and
    every ceiling it established would be a fiction.
    """
    train = generate_dataset(condition_config("negative_control", n_examples=N, split="train"))
    evaluation = generate_dataset(condition_config("negative_control", n_examples=N, split="test"))
    reference = EvaluationReference.from_dataset(evaluation)
    marginal = fit_marginal(train)
    predictions = ProgramOracle(fallback=marginal).predict(evaluation)
    assert associative_recall_accuracy(predictions, reference).value == 0.0
    # ...and it degrades to exactly the marginal it was handed, not to zeros.
    assert reconstruction_loss(predictions, reference).value == pytest.approx(
        reconstruction_loss(marginal.predict(evaluation), reference).value
    )


# --------------------------------------------------------------------------- #
# The marginal is the adversary, and the ceiling is the strong form of it
# --------------------------------------------------------------------------- #


def test_the_frequency_ceiling_is_fitted_on_what_it_is_scored_on(t0):
    ceiling = fit_frequency_ceiling(t0["reference"])
    assert "evaluation split itself" in ceiling.fitted_on
    # Its rates are the evaluation set's own, so it cannot be beaten by any
    # other constant-per-feature predictor on micro detection.
    scored = t0["reference"].target_active[t0["reference"].supervised][
        :, t0["reference"].content_indices
    ]
    assert ceiling.active_rate == pytest.approx(scored.mean(axis=0))


def test_the_ceiling_dominates_the_marginal_where_that_is_provable(t0):
    ceiling = evaluate_all(t0["ceiling"], t0["reference"])
    marginal = evaluate_all(t0["marginal_pred"], t0["reference"])
    for name in CEILING_DOMINATED_METRICS:
        a, b = ceiling[name].value, marginal[name].value
        if a is None or b is None:
            continue
        if METRIC_SPEC_BY_NAME[name].kind == "loss":
            assert a <= b + 1e-9, name
        else:
            assert a >= b - 1e-9, name


def test_no_retained_metric_is_passable_by_the_frequency_ceiling(report):
    for verdict in report.verdicts:
        if verdict.status != "retained":
            continue
        passed, reason = metric_rule(
            METRIC_SPEC_BY_NAME[verdict.name], verdict.oracle, verdict.ceiling
        )
        assert passed, f"{verdict.name}: {reason}"


def test_every_retired_metric_is_still_reachable_without_computing(report):
    assert set(RETIRED_METRICS) == {"ece", "recall_at_free_threshold"}
    for verdict in report.verdicts:
        if verdict.status != "retired":
            continue
        passed, _ = metric_rule(
            METRIC_SPEC_BY_NAME[verdict.name], verdict.oracle, verdict.ceiling
        )
        assert not passed, f"{verdict.name} now passes the rule; revisit its retirement"


def test_ece_is_retired_because_computing_nothing_is_perfectly_calibrated(report):
    t0_scores = next(c for c in report.conditions if c.condition == "T0").scores
    assert t0_scores["ece"]["ceiling"] == pytest.approx(0.0, abs=1e-9)
    assert t0_scores["ece"]["oracle"] == pytest.approx(0.0, abs=1e-9)
    # Brier survives the same comparison, which is why it is the retained form.
    assert t0_scores["brier"]["ceiling"] > 20 * METRIC_SPEC_BY_NAME["brier"].floor


def test_recall_is_retired_because_it_is_free_at_a_low_threshold(t0):
    for kind in ("ceiling", "marginal_pred", "random", "oracle"):
        sweep = threshold_sweep(t0[kind], t0["reference"])
        assert sweep.best_recall == 1.0, kind
    # F1 is not free in the same way: the ceiling cannot reach the oracle's.
    ceiling = threshold_sweep(t0["ceiling"], t0["reference"])
    oracle = threshold_sweep(t0["oracle"], t0["reference"])
    assert oracle.best_f1 == pytest.approx(1.0)
    assert ceiling.best_f1 < 0.5


def test_the_marginal_can_score_below_chance_on_a_compositional_split(report):
    """Recorded because it is the reason the rule is written against the ceiling.

    The coverage-preserving holdout puts exactly the under-represented template
    compositions in the test split, so training frequency is an actively
    misleading predictor of evaluation frequency. A model beating the
    train-fitted marginal on such a split has cleared a bar below chance.
    """
    correlation = report.notes["T0_train_eval_frequency_correlation"]
    assert correlation is not None and correlation < -0.5, correlation
    t0_scores = next(c for c in report.conditions if c.condition == "T0").scores
    assert t0_scores["feature_f1"]["marginal"] < t0_scores["feature_f1"]["random"]
    assert t0_scores["feature_f1"]["ceiling"] > t0_scores["feature_f1"]["random"]


# --------------------------------------------------------------------------- #
# The positive control
# --------------------------------------------------------------------------- #


def test_the_positive_control_threshold_lies_between_oracle_and_marginal(report):
    control = report.positive_control
    assert control["values"]["oracle"] == 1.0
    assert control["values"]["marginal"] == 0.0
    assert control["oracle_margin"] == pytest.approx(1.0 - POSITIVE_CONTROL_THRESHOLD)
    assert control["marginal_margin"] == pytest.approx(POSITIVE_CONTROL_THRESHOLD)
    assert control["ceiling_margin"] > 0.5


def test_the_marginal_baseline_fails_the_positive_control():
    train, _ = positive_control_datasets(n_examples=N)
    marginal = fit_marginal(train)
    result = positive_control(lambda ds: marginal.predict(ds), n_examples=N)
    assert not result.passed
    assert result.instrument_ok, "the failure must be the model's, not the ruler's"
    assert result.value == 0.0


def test_the_oracle_passes_the_positive_control():
    train, _ = positive_control_datasets(n_examples=N)
    marginal = fit_marginal(train)
    result = positive_control(
        lambda ds: ProgramOracle(fallback=marginal).predict(ds), n_examples=N
    )
    assert result.passed and result.instrument_ok
    assert result.value == 1.0
    assert result.skill == pytest.approx(1.0)


def test_chance_fails_the_positive_control():
    train, _ = positive_control_datasets(n_examples=N)
    result = positive_control(
        lambda ds: RandomBaseline.fit(train).predict(ds), n_examples=N
    )
    assert not result.passed


def test_a_result_just_over_the_line_passes_and_just_under_it_fails():
    """The verdict must turn exactly at the recorded threshold, not near it."""
    train, evaluation = positive_control_datasets(n_examples=N)
    marginal = fit_marginal(train)
    oracle = ProgramOracle(fallback=marginal).predict(evaluation)
    marginal_predictions = marginal.predict(evaluation)
    n_steps = len(evaluation.programs)

    def blend(fraction: float) -> Predictions:
        """Answer the first ``fraction`` of examples perfectly, the rest not."""
        cut = round(fraction * n_steps)
        values = marginal_predictions.values.copy()
        probs = marginal_predictions.active_prob.copy()
        values[:cut] = oracle.values[:cut]
        probs[:cut] = oracle.active_prob[:cut]
        return Predictions(values=values, active_prob=probs)

    over = positive_control(lambda _ds: blend(0.85), n_examples=N)
    under = positive_control(lambda _ds: blend(0.75), n_examples=N)
    assert over.value >= POSITIVE_CONTROL_THRESHOLD and over.passed
    assert under.value < POSITIVE_CONTROL_THRESHOLD and not under.passed


def test_the_positive_control_uses_the_known_easy_condition():
    _, evaluation = positive_control_datasets(n_examples=8)
    config = evaluation.config
    assert config.condition == "positive_control"
    assert config.d_recommended > evaluation.n_features, "ample dimension means d > F"
    assert config.n_distractors == 0
    assert config.distance_buckets == ((1, 2),)
    assert config.n_associations == 1
    assert not config.key_collisions


def test_the_positive_control_says_when_the_instrument_rather_than_the_model_failed():
    result = positive_control(
        lambda ds: Predictions(
            values=np.zeros(ds.inputs.shape), active_prob=np.zeros(ds.inputs.shape)
        ),
        n_examples=N,
    )
    assert result.instrument_ok and not result.passed
    assert "R1" in result.detail["note"]
    assert result.as_dict()["threshold"] == POSITIVE_CONTROL_THRESHOLD


# --------------------------------------------------------------------------- #
# Permutation: invariant where it should be, and not where it should not
# --------------------------------------------------------------------------- #


def _oracle_metrics(condition: str):
    train = generate_dataset(condition_config(condition, n_examples=N, split="train"))
    evaluation = generate_dataset(condition_config(condition, n_examples=N, split="test"))
    reference = EvaluationReference.from_dataset(evaluation)
    predictions = ProgramOracle(fallback=fit_marginal(train)).predict(evaluation)
    return reference, predictions, evaluate_all(predictions, reference)


def test_metrics_are_invariant_when_features_are_relabelled():
    """Feature identity is a label. Permuting it must move nothing."""
    _, _, base = _oracle_metrics("capacity_stressed")
    _, _, permuted = _oracle_metrics("permutation_control")
    for name, value in base.items():
        other = permuted[name].value
        if value.value is None or other is None:
            continue
        if value.value != value.value:  # NaN
            continue
        assert other == pytest.approx(value.value, abs=1e-9), name


def test_metrics_are_not_invariant_when_only_the_predictions_are_relabelled():
    """Scrambling which feature is which must destroy the score.

    The mirror of the test above, and the one that has teeth: a metric that
    survived this would not be measuring feature identity at all, only how many
    features were active.
    """
    reference, predictions, base = _oracle_metrics("capacity_stressed")
    content = reference.content_indices
    shuffled = np.random.default_rng(11).permutation(content)
    values = predictions.values.copy()
    probs = predictions.active_prob.copy()
    values[:, :, shuffled] = predictions.values[:, :, content]
    probs[:, :, shuffled] = predictions.active_prob[:, :, content]
    scrambled = evaluate_all(Predictions(values=values, active_prob=probs), reference)

    assert base["associative_recall_accuracy"].value == 1.0
    assert scrambled["associative_recall_accuracy"].value == 0.0
    assert scrambled["feature_f1"].value < 0.10
    assert scrambled["reconstruction_loss"].value > base["reconstruction_loss"].value


def test_shifting_features_by_one_is_a_weaker_scramble_than_permuting_them():
    """A caution for every later mission that builds a label-shuffling control.

    Rolling the feature axis by one position is *not* a label scramble: a
    position draws several features from the same contiguous group, so adjacent
    features co-activate and a shifted prediction still lands on truly active
    neighbours. Measured here, so nobody later writes ``np.roll`` where they
    meant ``rng.permutation`` and reads the residual signal as a real effect.
    """
    reference, predictions, _ = _oracle_metrics("capacity_stressed")
    content = reference.content_indices

    def scramble(order: np.ndarray) -> float:
        values = predictions.values.copy()
        probs = predictions.active_prob.copy()
        values[:, :, order] = predictions.values[:, :, content]
        probs[:, :, order] = predictions.active_prob[:, :, content]
        return evaluate_all(Predictions(values=values, active_prob=probs), reference)[
            "feature_f1"
        ].value

    rolled = scramble(np.roll(content, 1))
    permuted = scramble(np.random.default_rng(11).permutation(content))
    assert rolled > 2 * permuted, (rolled, permuted)


def test_the_answer_group_label_survives_permutation():
    """A generator-space label, not an index: the record's own invariance."""
    base = generate_dataset(condition_config("capacity_stressed", n_examples=32, split="test"))
    permuted = generate_dataset(
        condition_config("permutation_control", n_examples=32, split="test")
    )
    for a, b in zip(base.programs, permuted.programs, strict=True):
        assert [s.answer_group for s in a.steps] == [s.answer_group for s in b.steps]


# --------------------------------------------------------------------------- #
# The synthetic overwrite fixture
# --------------------------------------------------------------------------- #


def test_the_synthetic_overwrite_fixture_always_corrects_something():
    reference = synthetic_overwrite_reference(n_examples=256)
    for record in reference.programs:
        step = record.steps[0]
        assert set(step.answer_features) != set(step.stale_answer_features)
        assert step.stale_source is not None and step.stale_source < step.source


def test_overwrite_metrics_separate_the_oracle_from_a_stale_memory(report):
    scores = next(c for c in report.conditions if c.condition == "synthetic_overwrite").scores
    assert scores["overwrite_accuracy"]["oracle"] == 1.0
    assert scores["stale_value_error_rate"]["oracle"] == 0.0
    assert report.notes["stale_copier_overwrite_accuracy"] == 0.0
    assert report.notes["stale_copier_stale_value_error_rate"] == 1.0


# --------------------------------------------------------------------------- #
# The register and the report agree
# --------------------------------------------------------------------------- #


def test_every_recorded_status_agrees_with_the_measurement(report):
    disagreeing = [v.name for v in report.verdicts if not v.agrees]
    assert not disagreeing, f"recorded status and measured rule disagree for {disagreeing}"
    assert report.ok


def test_every_spec_is_measured_and_every_measurement_has_a_spec(report):
    assert {v.name for v in report.verdicts} == set(METRIC_SPEC_BY_NAME)
    for condition in report.conditions:
        assert set(condition.scores) == set(METRIC_SPEC_BY_NAME)


def test_the_marginal_scores_exactly_zero_skill_on_every_retained_metric(report):
    for verdict in report.verdicts:
        if verdict.status != "retained" or verdict.marginal_skill is None:
            continue
        assert verdict.marginal_skill == pytest.approx(0.0, abs=1e-12), verdict.name


def test_scoring_a_condition_is_deterministic():
    first, _ = score_condition("T0", n_examples=64)
    second, _ = score_condition("T0", n_examples=64)
    assert first.scores == second.scores
