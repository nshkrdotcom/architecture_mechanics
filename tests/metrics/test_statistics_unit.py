"""The experimental-unit rule, checked as a boundary rather than as a convention.

§7.4 names "treating hundreds of tokens as independent samples when the run is
the true experimental unit" as a thing to avoid. A module that avoided it by
documentation would be avoided by whoever is debugging at two in the morning, so
every way of offering the wrong unit has a refusal here and every refusal has a
test.

The last test is the one that says why it matters: the same five runs, analysed
correctly and analysed by pooling their tokens, give standard errors differing by
a factor of five. That factor is not a rounding error in a confidence interval;
it is the difference between a null result and a finding.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from architecture_mechanics.metrics.statistics import (
    MAX_RUNS_PER_CELL,
    RESAMPLING_UNITS,
    ExperimentalUnitError,
    RunSummary,
    StatisticsError,
    bootstrap_ci,
    feature_permutation_test,
    hierarchical_effect,
    paired_effect,
    paired_test,
    run_summary_from_json,
    unpaired_test,
)


def runs(values, *, arm: str, cell: str = "all", first_seed: int = 0):
    return [
        RunSummary(run_id=f"{arm}-{cell}-{index}", seed=first_seed + index, arm=arm, cell=cell,
                   metrics={"accuracy": value})
        for index, value in enumerate(values)
    ]


# --------------------------------------------------------------------------- #
# A metric is one number per run
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "value",
    [
        np.zeros(4096),
        np.zeros((32, 128)),
        np.zeros(1),
        [0.1, 0.2, 0.3],
        (0.1, 0.2),
        torch.zeros(512),
    ],
)
def test_a_metric_that_is_an_array_is_refused(value):
    with pytest.raises(ExperimentalUnitError, match="experimental unit|run-level value"):
        RunSummary(run_id="r", seed=0, arm="a", metrics={"accuracy": value})


@pytest.mark.parametrize("value", [0.5, 1, np.float64(0.5), np.int64(3), np.array(0.5), torch.tensor(0.5)])
def test_a_scalar_metric_is_accepted_however_it_is_spelled(value):
    run = RunSummary(run_id="r", seed=0, arm="a", metrics={"accuracy": value})
    assert isinstance(run.metrics["accuracy"], float)


def test_a_null_metric_becomes_nan_rather_than_zero():
    """``summary.json`` writes ``null`` for a metric that did not apply. Reading it
    as 0.0 would make "not measured" and "measured as nothing" the same number."""
    run = RunSummary(run_id="r", seed=0, arm="a", metrics={"overwrite_accuracy": None})
    assert math.isnan(run.metrics["overwrite_accuracy"])


def test_a_boolean_verdict_is_not_a_metric():
    with pytest.raises(ExperimentalUnitError, match="boolean"):
        RunSummary(run_id="r", seed=0, arm="a", metrics={"passed": True})


def test_metrics_cannot_be_mutated_after_construction():
    run = RunSummary(run_id="r", seed=0, arm="a", metrics={"accuracy": 0.5})
    with pytest.raises(TypeError):
        run.metrics["accuracy"] = 0.9


# --------------------------------------------------------------------------- #
# A comparison is over runs, not over numbers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("estimator", [paired_test, paired_effect, unpaired_test])
def test_bare_arrays_are_refused_as_arms(estimator):
    with pytest.raises(ExperimentalUnitError, match="array of"):
        estimator(np.zeros(4096), np.ones(4096), "accuracy")


@pytest.mark.parametrize("estimator", [paired_test, paired_effect, unpaired_test])
def test_bare_lists_of_numbers_are_refused_as_arms(estimator):
    """Even when the list really is per-run. The module cannot tell a list of five
    run scores from a list of five token scores, so it refuses both and asks for
    the run identity that would have made the difference visible."""
    with pytest.raises(ExperimentalUnitError, match="not RunSummary"):
        estimator([0.1, 0.2, 0.3, 0.4, 0.5], [0.2, 0.3, 0.4, 0.5, 0.6], "accuracy")


def test_a_torch_tensor_of_run_values_is_refused():
    with pytest.raises(ExperimentalUnitError, match="array of"):
        paired_test(torch.zeros(8), torch.ones(8), "accuracy")


def test_one_run_is_not_a_comparison():
    single = RunSummary(run_id="r", seed=0, arm="a", metrics={"accuracy": 0.5})
    with pytest.raises(ExperimentalUnitError, match="single RunSummary"):
        paired_test(single, single, "accuracy")


def test_a_repeated_seed_in_one_arm_is_refused():
    """Two evaluations of one run are one run. Counting them twice is the same
    error as counting tokens, at a scale small enough to look innocent."""
    control = [
        RunSummary(run_id=f"c{index}", seed=7, arm="control", metrics={"accuracy": 0.5})
        for index in range(3)
    ]
    with pytest.raises(ExperimentalUnitError, match="more than once"):
        paired_test(control, control, "accuracy")


def test_repeated_seeds_are_allowed_when_they_are_different_cells():
    control = runs([0.5, 0.6], arm="control", cell="sparse") + runs([0.4, 0.5], arm="control", cell="dense")
    candidate = runs([0.6, 0.7], arm="candidate", cell="sparse") + runs([0.5, 0.6], arm="candidate", cell="dense")
    effect = hierarchical_effect(control, candidate, "accuracy", resamples=200)
    assert effect.n_seeds == 2
    assert effect.n_cells == 2


def test_a_duplicate_run_id_is_refused():
    control = [
        RunSummary(run_id="same", seed=index, arm="control", metrics={"accuracy": 0.5})
        for index in range(3)
    ]
    with pytest.raises(ExperimentalUnitError, match="twice"):
        paired_test(control, control, "accuracy")


def test_more_runs_than_the_laboratory_could_produce_is_refused():
    control = runs(np.linspace(0.0, 1.0, MAX_RUNS_PER_CELL + 1), arm="control")
    with pytest.raises(ExperimentalUnitError, match="per-token or per-example"):
        paired_test(control, control, "accuracy")


def test_unmatched_seed_sets_are_refused():
    """§7.2 freezes the seed set. An arm with a seed the other lacks is an arm
    that was run more, and the extra run would be credited to the architecture."""
    control = runs([0.1, 0.2, 0.3], arm="control")
    candidate = runs([0.2, 0.3, 0.4, 0.5], arm="candidate")
    with pytest.raises(StatisticsError, match="not matched"):
        paired_test(control, candidate, "accuracy")


def test_an_undefined_metric_cannot_carry_a_comparison():
    control = runs([0.1, 0.2, 0.3], arm="control")
    candidate = [
        RunSummary(run_id="k0", seed=0, arm="candidate", metrics={"accuracy": None}),
        *runs([0.3, 0.4], arm="candidate", first_seed=1),
    ]
    with pytest.raises(StatisticsError, match="undefined"):
        paired_test(control, candidate, "accuracy")


# --------------------------------------------------------------------------- #
# A bootstrap has to say what it is resampling
# --------------------------------------------------------------------------- #


def test_bootstrap_requires_a_named_unit():
    with pytest.raises(TypeError):
        bootstrap_ci(np.array([0.1, 0.2, 0.3]))  # type: ignore[call-arg]


def test_bootstrap_refuses_an_unknown_unit():
    with pytest.raises(StatisticsError, match="unit="):
        bootstrap_ci(np.array([0.1, 0.2, 0.3]), unit="token")


@pytest.mark.parametrize("unit", ["run", "seed"])
def test_bootstrap_over_runs_refuses_a_token_sized_input(unit):
    with pytest.raises(ExperimentalUnitError, match="hundreds of tokens"):
        bootstrap_ci(np.zeros(4096), unit=unit)


def test_every_declared_unit_is_usable():
    for unit in RESAMPLING_UNITS:
        low, high, _ = bootstrap_ci(np.linspace(0.0, 1.0, 8), unit=unit, resamples=200)
        assert low <= high


# --------------------------------------------------------------------------- #
# The one function whose unit is the feature says so
# --------------------------------------------------------------------------- #


def test_the_feature_permutation_belongs_to_a_run_and_returns_one_number():
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal(36), rng.standard_normal(36)
    result = feature_permutation_test(a, b, run_id="R1-softmax-s1", resamples=500)
    assert result.run_id == "R1-softmax-s1"
    assert result.as_dict()["unit"] == "feature"
    assert result.as_dict()["scope"] == "within_run"
    assert isinstance(result.as_run_metric(), float)


def test_feature_results_cannot_be_fed_to_a_run_level_estimator():
    """There is no path from a bag of within-run permutation results into a
    comparison across seeds. The reduction has to go through RunSummary, where
    the seed and the run id have to be supplied."""
    rng = np.random.default_rng(1)
    results = [
        feature_permutation_test(rng.standard_normal(36), rng.standard_normal(36),
                                 run_id=f"r{index}", resamples=200)
        for index in range(5)
    ]
    with pytest.raises(ExperimentalUnitError, match="not RunSummary"):
        paired_test(results, results, "statistic")


def test_the_feature_permutation_refuses_a_token_sized_bank():
    rng = np.random.default_rng(2)
    huge = rng.standard_normal(200_000)
    with pytest.raises(ExperimentalUnitError, match="per-token"):
        feature_permutation_test(huge, huge, run_id="r", resamples=10)


# --------------------------------------------------------------------------- #
# The easy path is the correct path
# --------------------------------------------------------------------------- #


def test_a_run_summary_is_read_straight_out_of_summary_json(tmp_path):
    import json

    directory = tmp_path / "R4-softmax-positive_control-s20260809-abcdef"
    directory.mkdir()
    (directory / "summary.json").write_text(json.dumps({
        "run_id": directory.name,
        "config": {"seed": 20260809},
        "final": {"associative_recall_accuracy": 0.9055, "overwrite_accuracy": None},
    }))
    run = run_summary_from_json(directory, arm="softmax")
    assert run.seed == 20260809
    assert run.run_id == directory.name
    assert run.metrics["associative_recall_accuracy"] == pytest.approx(0.9055)
    assert math.isnan(run.metrics["overwrite_accuracy"])


def test_a_summary_without_a_seed_is_refused(tmp_path):
    import json

    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"run_id": "x", "config": {}, "final": {"accuracy": 0.5}}))
    with pytest.raises(StatisticsError, match="no seed"):
        run_summary_from_json(path, arm="softmax")


# --------------------------------------------------------------------------- #
# Why the rule exists
# --------------------------------------------------------------------------- #


def test_pooling_tokens_narrows_the_interval_by_the_square_root_of_the_pooling():
    """The measurement behind the refusal.

    Five runs of sixty-four tokens each, where every token of a run shares that
    run's offset — which is what tokens from one model on one seed do. The
    run-level standard error is the honest one; the token-pooled standard error
    is smaller by a factor that has nothing to do with how much was learned.
    """
    rng = np.random.default_rng(20260810)
    n_runs, n_tokens = 5, 64
    offsets = rng.standard_normal((n_runs, 1))
    tokens = offsets + rng.standard_normal((n_runs, n_tokens))

    honest = float(tokens.mean(axis=1).std(ddof=1)) / math.sqrt(n_runs)
    pooled = float(tokens.std(ddof=1)) / math.sqrt(n_runs * n_tokens)

    assert honest / pooled > 4.0
    with pytest.raises(ExperimentalUnitError):
        bootstrap_ci(tokens.ravel(), unit="run")
