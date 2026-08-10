"""The §7.4 estimators, held to answers known before they were measured.

The calibration in ``statistics --selftest`` checks *behaviour*: how often each
estimator is wrong when the truth is known. This file checks *arithmetic*: that
the t-test agrees with a published critical value, that the exact permutation
enumerates what it claims to, that Benjamini–Hochberg reproduces a worked
example, and that every degenerate input either raises or returns something a
reader can recognise as degenerate.

The two are not interchangeable. A p-value function with a factor of two wrong
would still produce a smooth, plausible calibration curve — just at the wrong
place — and the calibration would faithfully record the wrong number as the
laboratory's operating point.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from architecture_mechanics.metrics.statistics import (
    ALPHA,
    ESTIMATOR_SPEC_BY_NAME,
    ESTIMATOR_SPECS,
    FORBIDDEN_ESTIMATORS,
    THRESHOLDS,
    EffectSize,
    RunSummary,
    StatisticsError,
    _norm_cdf,
    _norm_ppf,
    _paired_t_test,
    _student_t_two_sided_p,
    bootstrap_ci,
    fdr_control,
    fdr_over_tests,
    feature_permutation_test,
    hierarchical_effect,
    paired_effect,
    paired_test,
    standardized_effect,
    unpaired_effect,
    unpaired_test,
)


def arm(values, *, name: str, cell: str = "all", first_seed: int = 0):
    return [
        RunSummary(run_id=f"{name}-{cell}-{index}", seed=first_seed + index, arm=name, cell=cell,
                   metrics={"accuracy": float(value)})
        for index, value in enumerate(values)
    ]


# --------------------------------------------------------------------------- #
# Special functions against published values
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("t", "df", "expected"),
    [
        (2.776445, 4, 0.05),      # two-sided 5% critical value of t(4)
        (2.262157, 9, 0.05),
        (4.604095, 4, 0.01),
        (3.169273, 10, 0.01),
        (0.0, 5, 1.0),
    ],
)
def test_student_t_matches_published_critical_values(t, df, expected):
    assert _student_t_two_sided_p(t, df) == pytest.approx(expected, abs=2e-6)


def test_student_t_converges_on_the_normal():
    """At a hundred thousand degrees of freedom the t tail is 0.0500028 against
    the normal's 0.05 — close, and not equal, which is the correct answer."""
    tails = [_student_t_two_sided_p(1.959964, df) for df in (1e3, 1e4, 1e5, 1e6)]
    assert tails == sorted(tails, reverse=True)
    assert tails[-1] == pytest.approx(0.05, abs=1e-6)


@pytest.mark.parametrize(
    ("p", "expected"),
    [(0.975, 1.959964), (0.025, -1.959964), (0.5, 0.0), (0.995, 2.575829), (0.001, -3.090232)],
)
def test_normal_quantile_matches_published_values(p, expected):
    assert float(_norm_ppf(p)) == pytest.approx(expected, abs=1e-6)


def test_the_normal_quantile_inverts_the_normal_cdf():
    probabilities = np.linspace(1e-6, 1 - 1e-6, 501)
    assert np.max(np.abs(_norm_cdf(_norm_ppf(probabilities)) - probabilities)) < 1e-12


# --------------------------------------------------------------------------- #
# Effect sizes
# --------------------------------------------------------------------------- #


def test_the_paired_effect_is_the_mean_difference_in_the_metric_s_own_units():
    control = arm([0.80, 0.82, 0.84, 0.86, 0.88], name="control")
    candidate = arm([0.83, 0.85, 0.87, 0.89, 0.91], name="candidate")
    effect = paired_effect(control, candidate, "accuracy", resamples=500)
    assert effect.estimate == pytest.approx(0.03)
    assert effect.unit == "accuracy"
    assert effect.n == 5
    assert effect.detail["control_mean"] == pytest.approx(0.84)
    assert effect.detail["candidate_mean"] == pytest.approx(0.87)


def test_the_sign_is_candidate_minus_control():
    """A sign error here would reverse every conclusion this laboratory reaches
    and contradict nothing else."""
    control = arm([0.5, 0.5, 0.5], name="control")
    candidate = arm([0.9, 0.9, 0.9], name="candidate")
    assert paired_effect(control, candidate, "accuracy", resamples=100).estimate > 0
    assert paired_effect(candidate, control, "accuracy", resamples=100).estimate < 0


def test_cohens_dz_is_the_mean_difference_over_its_own_standard_deviation():
    differences = np.array([0.02, 0.04, 0.01, 0.05, 0.03])
    control = arm(np.zeros(5), name="control")
    candidate = arm(differences, name="candidate")
    effect = standardized_effect(control, candidate, "accuracy", resamples=500)
    raw = differences.mean() / differences.std(ddof=1)
    assert effect.estimate == pytest.approx(raw * effect.detail["hedges_correction"], rel=1e-9)
    assert effect.unit == "sd_of_paired_difference"


def test_the_hedges_correction_shrinks_and_approaches_one():
    """At four degrees of freedom J = 0.798, so an uncorrected dz at five seeds
    overstates the effect by a quarter — the size of thing this program looks for."""
    from architecture_mechanics.metrics.statistics import _hedges_correction

    assert _hedges_correction(4) == pytest.approx(0.7979, abs=1e-4)
    assert _hedges_correction(9) == pytest.approx(0.9139, abs=1e-4)
    assert _hedges_correction(200) == pytest.approx(1.0, abs=5e-3)


def test_hedges_g_is_reported_for_the_unpaired_effect():
    control = arm([0.80, 0.82, 0.84], name="control")
    candidate = arm([0.86, 0.88, 0.90], name="candidate", first_seed=100)
    effect = unpaired_effect(control, candidate, "accuracy", resamples=500)
    assert effect.estimate == pytest.approx(0.06)
    assert effect.detail["seeds_matched"] is False
    assert effect.detail["hedges_g"] > 1.0


def test_an_effect_size_round_trips_through_its_dictionary():
    control, candidate = arm([0.1, 0.2, 0.3], name="c"), arm([0.2, 0.3, 0.4], name="k")
    effect = paired_effect(control, candidate, "accuracy", resamples=200)
    assert EffectSize.from_dict(effect.as_dict()).as_dict() == effect.as_dict()


# --------------------------------------------------------------------------- #
# Bootstrap intervals
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", ["percentile", "bca", "studentized"])
def test_a_bootstrap_interval_contains_the_estimate(method):
    values = np.array([0.81, 0.83, 0.79, 0.88, 0.84, 0.86, 0.82, 0.85])
    low, high, _ = bootstrap_ci(values, unit="run", method=method, resamples=2000)
    assert low <= values.mean() <= high


def test_the_bootstrap_is_deterministic_by_default():
    """A reported interval that moves between two readings of the same data is
    not a measurement, and "different random seed" is indistinguishable from
    "wrong number" to whoever reads the record."""
    values = np.linspace(0.1, 0.9, 7)
    first = bootstrap_ci(values, unit="run", resamples=500)[:2]
    second = bootstrap_ci(values, unit="run", resamples=500)[:2]
    assert first == second


def test_the_studentized_interval_is_infinite_when_three_runs_cannot_bound_the_mean():
    """One resample of three in nine draws a single value three times, its
    standard error is zero, and the honest interval is unbounded. NaN would read
    as a defect; an infinite bound reads as the width of the ignorance."""
    low, high, _ = bootstrap_ci(np.array([0.1, 0.2, 0.9]), unit="run", method="studentized",
                                resamples=5000)
    assert not math.isnan(low) and not math.isnan(high)
    assert math.isinf(low) or math.isinf(high)


def test_a_bootstrap_of_one_value_is_refused():
    with pytest.raises(StatisticsError, match="not an interval"):
        bootstrap_ci(np.array([0.5]), unit="run")


def test_an_unknown_bootstrap_method_is_refused():
    with pytest.raises(StatisticsError, match="unknown bootstrap method"):
        bootstrap_ci(np.linspace(0, 1, 6), unit="run", method="jackknife")


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_the_paired_t_test_agrees_with_the_published_critical_value():
    """Five differences whose t is exactly the two-sided 5% critical value of
    t(4) must come back at exactly p = 0.05."""
    spread = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    standard_error = spread.std(ddof=1) / math.sqrt(5)
    differences = spread + 2.776445 * standard_error
    result = _paired_t_test(differences, alpha=ALPHA)
    assert result.statistic == pytest.approx(2.776445)
    assert result.p_value == pytest.approx(0.05, abs=1e-5)
    assert result.detail["df"] == 4


def test_the_exact_paired_permutation_enumerates_every_sign_assignment():
    control = arm(np.zeros(5), name="control")
    candidate = arm([1.0, 2.0, 3.0, 4.0, 5.0], name="candidate")
    result = paired_test(control, candidate, "accuracy", test="paired_permutation")
    assert result.detail["exact"] is True
    assert result.detail["arrangements"] == 32
    assert result.p_value == pytest.approx(2 / 32)


@pytest.mark.parametrize(
    ("n_seeds", "floor", "attainable"),
    [(3, 0.25, False), (4, 0.125, False), (5, 0.0625, False), (6, 0.03125, True), (10, 2 / 1024, True)],
)
def test_the_permutation_floor_is_two_over_two_to_the_n(n_seeds, floor, attainable):
    """The finding that decides how many seeds a permutation-tested claim needs.
    Six is the smallest seed count at which the exact paired permutation can
    produce a significant result at all, whatever the data."""
    control = arm(np.zeros(n_seeds), name="control")
    candidate = arm(np.arange(1.0, n_seeds + 1) * 1000.0, name="candidate")
    result = paired_test(control, candidate, "accuracy", test="paired_permutation")
    assert result.p_value_floor == pytest.approx(floor)
    assert result.p_value == pytest.approx(floor)
    assert result.power_is_attainable is attainable
    assert result.significant is attainable


def test_the_unpaired_permutation_can_reject_at_five_where_the_paired_one_cannot():
    """C(10,5) = 252 label assignments put the floor at 0.008, so the same five
    seeds per arm support a significant result unpaired and cannot paired."""
    control = arm(np.zeros(5), name="control")
    candidate = arm(np.full(5, 1000.0), name="candidate", first_seed=100)
    result = unpaired_test(control, candidate, "accuracy", test="unpaired_permutation")
    assert result.detail["exact"] is True
    assert result.detail["arrangements"] == 252
    assert result.p_value == pytest.approx(2 / 252)
    assert result.significant


def test_welch_degrees_of_freedom_fall_between_the_two_extremes():
    control = arm([0.1, 0.2, 0.3, 0.4], name="control")
    candidate = arm([0.5, 0.9, 0.1, 0.7], name="candidate", first_seed=100)
    result = unpaired_test(control, candidate, "accuracy", test="welch_t")
    assert 3.0 <= result.detail["df"] <= 6.0


def test_a_zero_variance_difference_does_not_divide_by_zero():
    control = arm([0.5, 0.5, 0.5, 0.5], name="control")
    identical = paired_test(control, arm([0.5] * 4, name="candidate"), "accuracy")
    assert identical.statistic == 0.0
    assert identical.p_value == pytest.approx(1.0)

    shifted = paired_test(control, arm([0.9] * 4, name="candidate"), "accuracy")
    assert math.isinf(shifted.statistic)
    assert shifted.p_value == 0.0


def test_an_unknown_test_is_refused():
    control, candidate = arm([0.1, 0.2, 0.3], name="c"), arm([0.2, 0.3, 0.4], name="k")
    with pytest.raises(StatisticsError, match="unknown paired test"):
        paired_test(control, candidate, "accuracy", test="wilcoxon")


# --------------------------------------------------------------------------- #
# Hierarchical
# --------------------------------------------------------------------------- #


def _matrix_arms(matrix: np.ndarray, *, name: str):
    return [
        RunSummary(run_id=f"{name}-s{seed}-c{cell}", seed=seed, arm=name, cell=f"cell{cell}",
                   metrics={"accuracy": float(matrix[seed, cell])})
        for seed in range(matrix.shape[0])
        for cell in range(matrix.shape[1])
    ]


def test_the_hierarchical_effect_averages_cells_into_the_seed():
    """The reduction is the discipline. Twenty runs over five seeds must be five
    numbers, not twenty, or the seed effect gets counted four times."""
    rng = np.random.default_rng(3)
    control = rng.standard_normal((5, 4))
    candidate = control + 0.5
    effect = hierarchical_effect(_matrix_arms(control, name="control"),
                                 _matrix_arms(candidate, name="candidate"),
                                 "accuracy", resamples=500)
    assert effect.n_seeds == 5
    assert effect.n_cells == 4
    assert effect.effect.n == 5, "the interval is over seeds, not over the twenty runs"
    assert effect.test.n == 5
    assert len(effect.per_seed) == 5
    assert effect.effect.estimate == pytest.approx(0.5)


def test_per_cell_results_are_exploratory_and_arrive_with_a_correction():
    """§7.4's last prohibition is upgrading a claim because one exploratory cell
    passed. The cells therefore carry q-values, not just p-values."""
    rng = np.random.default_rng(5)
    control = rng.standard_normal((6, 4))
    candidate = control + rng.standard_normal((6, 4)) * 0.1
    effect = hierarchical_effect(_matrix_arms(control, name="control"),
                                 _matrix_arms(candidate, name="candidate"),
                                 "accuracy", resamples=500)
    assert len(effect.per_cell) == 4
    for row in effect.per_cell:
        assert row["exploratory"] is True
        assert row["q_value"] >= row["p_value"] - 1e-12
    assert effect.cell_fdr.method == "bh"


def test_a_hole_in_the_seed_by_cell_matrix_is_refused():
    control = _matrix_arms(np.zeros((3, 2)), name="control")
    candidate = _matrix_arms(np.zeros((3, 2)), name="candidate")
    paired = hierarchical_effect(control, candidate, "accuracy", resamples=200)
    assert paired.n_cells == 2

    with pytest.raises(StatisticsError, match="not matched"):
        hierarchical_effect(control[:-1], candidate, "accuracy", resamples=200)


# --------------------------------------------------------------------------- #
# False-discovery control
# --------------------------------------------------------------------------- #


def test_benjamini_hochberg_reproduces_a_worked_example():
    """The original 1995 paper's example: fifteen p-values, four rejected at 0.05."""
    p_values = [0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344,
                0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.0000]
    result = fdr_control(p_values, alpha=0.05, method="bh")
    assert result.n_rejected == 4
    assert list(result.rejected[:4]) == [True] * 4
    assert not any(result.rejected[4:])
    assert result.threshold == pytest.approx(0.0095)


def test_benjamini_yekutieli_is_stricter_by_the_harmonic_factor():
    p_values = [0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344,
                0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.0000]
    bh = fdr_control(p_values, alpha=0.05, method="bh")
    by = fdr_control(p_values, alpha=0.05, method="by")
    assert by.n_rejected <= bh.n_rejected
    assert all(q_by >= q_bh - 1e-12 for q_bh, q_by in zip(bh.q_values, by.q_values, strict=True))


def test_q_values_are_monotone_in_the_p_value_ordering():
    rng = np.random.default_rng(7)
    p_values = np.sort(rng.random(40))
    result = fdr_control(p_values, alpha=0.05, method="bh")
    assert list(result.q_values) == sorted(result.q_values)
    assert all(0.0 <= q <= 1.0 for q in result.q_values)


def test_a_family_whose_p_values_cannot_reach_the_threshold_says_so():
    """Twenty-four exact permutation tests at five seeds all floor at 0.0625. No
    BH threshold can be met, and "nothing survived correction" would be a fact
    about arithmetic rather than about mechanism."""
    control = arm(np.zeros(5), name="control")
    tests = [
        paired_test(control, arm(np.arange(1.0, 6.0) * scale, name=f"k{index}"),
                    "accuracy", test="paired_permutation")
        for index, scale in enumerate(np.linspace(1.0, 50.0, 24))
    ]
    result = fdr_over_tests(tests, alpha=0.05, method="bh")
    assert result.p_value_floor == pytest.approx(0.0625)
    assert result.attainable is False
    assert result.n_rejected == 0

    t_tests = [
        paired_test(control, arm(np.arange(1.0, 6.0) * scale, name=f"k{index}"), "accuracy")
        for index, scale in enumerate(np.linspace(1.0, 50.0, 24))
    ]
    assert fdr_over_tests(t_tests, alpha=0.05, method="bh").attainable is True


def test_fdr_refuses_malformed_input():
    with pytest.raises(StatisticsError, match="non-empty"):
        fdr_control([])
    with pytest.raises(StatisticsError, match=r"\[0, 1\]"):
        fdr_control([0.1, 1.4])
    with pytest.raises(StatisticsError, match="unknown FDR method"):
        fdr_control([0.1], method="bonferroni")
    with pytest.raises(StatisticsError, match="labels"):
        fdr_control([0.1, 0.2], labels=["only-one"])


# --------------------------------------------------------------------------- #
# Feature permutation
# --------------------------------------------------------------------------- #


def test_the_feature_permutation_finds_a_shift_it_should_find():
    rng = np.random.default_rng(9)
    baseline = rng.standard_normal(48)
    result = feature_permutation_test(baseline, baseline + 1.0, run_id="r", resamples=2000)
    assert result.p_value < 0.01
    assert result.statistic == pytest.approx(1.0)


def test_the_paired_feature_design_needs_the_same_features_in_both_arms():
    rng = np.random.default_rng(10)
    with pytest.raises(StatisticsError, match="same features"):
        feature_permutation_test(rng.standard_normal(20), rng.standard_normal(30),
                                 run_id="r", resamples=100)


def test_undefined_features_are_refused_rather_than_averaged_over():
    """Geometry records a feature that never varied as NaN. Permuting over it
    would silently drop it from one arm and not the other."""
    values = np.array([1.0, 2.0, np.nan, 4.0])
    with pytest.raises(StatisticsError, match="NaN"):
        feature_permutation_test(values, values, run_id="r", resamples=100)


def test_the_independent_design_shuffles_labels_rather_than_signs():
    rng = np.random.default_rng(12)
    result = feature_permutation_test(rng.standard_normal(12), rng.standard_normal(12) + 3.0,
                                      run_id="r", design="independent", resamples=2000)
    assert result.design == "independent"
    assert result.n_features == 24
    assert result.p_value < 0.01


# --------------------------------------------------------------------------- #
# The register
# --------------------------------------------------------------------------- #


def test_the_register_is_consistent():
    names = [spec.name for spec in ESTIMATOR_SPECS]
    assert len(names) == len(set(names))
    assert set(ESTIMATOR_SPEC_BY_NAME) == set(names)
    assert FORBIDDEN_ESTIMATORS <= set(names)
    for spec in ESTIMATOR_SPECS:
        assert spec.reason.strip(), spec.name
        assert spec.definition.strip(), spec.name
        assert 0.0 <= spec.recorded_fpr_at_5 <= 1.0
        assert (spec.status == "forbidden") == (spec.name in FORBIDDEN_ESTIMATORS)


def test_every_status_is_one_the_rule_can_produce():
    assert {spec.status for spec in ESTIMATOR_SPECS} <= {
        "level_holding", "descriptive_only", "unusable_at_five_seeds", "forbidden",
    }


def test_the_adopted_estimators_all_hold_their_level():
    from architecture_mechanics.metrics.statistics import ADOPTED

    for purpose, name in ADOPTED.items():
        assert ESTIMATOR_SPEC_BY_NAME[name].status == "level_holding", purpose


def test_the_recorded_thresholds_agree_with_the_module_s_defaults():
    from architecture_mechanics.metrics.statistics import CI_LEVEL, PRIMARY_TEST

    assert THRESHOLDS["alpha"] == ALPHA
    assert THRESHOLDS["ci_level"] == CI_LEVEL
    assert ESTIMATOR_SPEC_BY_NAME[PRIMARY_TEST].status == "level_holding"
    assert THRESHOLDS["minimum_seeds_for_the_exact_paired_permutation"] == 6
