"""The information-destroyed control, fed an oracle.

The §4.4 negative control is the laboratory's hard stop: a trained model that
beats it means the task leaks and every capability number recorded to date is
measuring the leak. Prompt 09 made that a kill condition at 0.05 exact recall.

The control is only worth having if the strongest predictor in the building
cannot beat it. So this file feeds it the program oracle, and — because a test
that everything fails is not a test — a deliberately cheating oracle that reads
the supervised target tensor, which must score exactly 1.0.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from architecture_mechanics.data.feature_program import (
    answer_appears_in_input,
    condition_config,
    generate_dataset,
    perfect_memory_oracle_report,
)
from architecture_mechanics.metrics.capability import (
    EvaluationReference,
    Predictions,
    ProgramOracle,
    RandomBaseline,
    associative_recall_accuracy,
    feature_f1,
    fit_frequency_ceiling,
    fit_marginal,
    reconstruction_loss,
)

EXAMPLES = 512
KILL_THRESHOLD = 0.05
"""`claims/a0-t1-associative-recall.yml`'s second kill condition, restated so the
assertions below are about the bar that was pre-registered."""


@pytest.fixture(scope="module")
def negative():
    train = generate_dataset(condition_config("negative_control", split="train", n_examples=EXAMPLES))
    evaluation = generate_dataset(
        condition_config("negative_control", split="test", n_examples=EXAMPLES)
    )
    return train, evaluation, EvaluationReference.from_dataset(evaluation)


def test_the_program_oracle_is_at_chance_on_the_negative_control(negative):
    """It reads the program, the program says the route is destroyed, and it
    falls back to the marginal. An oracle that scored above chance here would be
    reading the answer key rather than reconstructing from the record."""
    train, evaluation, reference = negative
    marginal = fit_marginal(train)
    oracle = ProgramOracle(fallback=marginal).predict(evaluation)

    assert associative_recall_accuracy(oracle, reference).value == 0.0
    assert associative_recall_accuracy(oracle, reference).value < KILL_THRESHOLD
    # and it pays exactly the marginal's reconstruction loss: no headroom at all
    assert reconstruction_loss(oracle, reference).value == pytest.approx(
        reconstruction_loss(marginal.predict(evaluation), reference).value
    )


def test_a_cheating_oracle_scores_one_so_the_check_is_not_vacuous(negative):
    """The non-vacuity control. If reading the target tensor did not score 1.0,
    the row above would be satisfied by a metric that cannot see any answer."""
    _, _, reference = negative
    cheat = Predictions(
        values=reference.targets.astype(np.float64),
        active_prob=reference.target_active.astype(np.float64),
    )
    assert associative_recall_accuracy(cheat, reference).value == pytest.approx(1.0)
    assert feature_f1(cheat, reference).value == pytest.approx(1.0)
    assert reconstruction_loss(cheat, reference).value == pytest.approx(0.0)


def test_no_input_blind_predictor_clears_the_kill_threshold(negative):
    train, evaluation, reference = negative
    for name, predictions in (
        ("marginal", fit_marginal(train).predict(evaluation)),
        ("frequency ceiling", fit_frequency_ceiling(reference).predict_like(reference)),
        ("chance", RandomBaseline.fit(train).predict(evaluation)),
    ):
        value = associative_recall_accuracy(predictions, reference).value
        assert value < KILL_THRESHOLD, f"{name} scored {value} on an impossible task"


def test_the_boring_strategy_battery_finds_nothing_and_still_works(negative):
    """The one route to the answer that never consults `step.source`. Both halves
    matter: at chance on the impossible condition, and at 1.0 on the possible
    one."""
    _, evaluation, _ = negative
    impossible = perfect_memory_oracle_report(evaluation)
    assert impossible.best_honest_r2 <= 0.05
    assert answer_appears_in_input(evaluation) == 0

    possible = generate_dataset(
        condition_config("capacity_stressed", split="test", n_examples=EXAMPLES)
    )
    assert perfect_memory_oracle_report(possible).scores["key_match_exact"] >= 0.95


def test_the_analytic_chance_rate_is_far_below_the_kill_threshold(negative):
    """Chance on this metric is not zero and is not obvious, so it is derived.

    The answer at a query is drawn over the `G` features of one content group,
    each active independently with probability `p`, conditioned on at least one
    firing. A predictor emitting a constant activity score everywhere gets a
    uniformly random subset of the globally rate-matched budget, so its exact-set
    accuracy is `E_m[q^m (1-q)^(F-m)]` for inclusion rate `q`. Prompt 10
    measured 0.000244 against an analytic 0.000246 at 4096 examples.
    """
    _, evaluation, reference = negative
    config = evaluation.config
    group = config.n_content_features // config.n_content_groups
    p = config.activation_prob
    p0 = (1 - p) ** group
    pmf = {
        m: math.comb(group, m) * p**m * (1 - p) ** (group - m) / (1 - p0)
        for m in range(1, group + 1)
    }
    cells = int(reference.supervised.sum())
    active = int(reference.target_active[reference.supervised][:, reference.content_indices].sum())
    q = active / (cells * config.n_content_features)
    analytic = sum(
        w * q**m * (1 - q) ** (config.n_content_features - m) for m, w in pmf.items()
    )

    measured = associative_recall_accuracy(
        RandomBaseline.fit(evaluation).predict(evaluation), reference
    ).value
    assert analytic < KILL_THRESHOLD / 50, (
        f"chance is {analytic}; a kill threshold of {KILL_THRESHOLD} is no longer a bar"
    )
    assert abs(measured - analytic) < 3 / cells, (
        f"measured chance {measured} is more than three examples away from the analytic "
        f"{analytic}; either the draw law or the selection rule has changed"
    )
