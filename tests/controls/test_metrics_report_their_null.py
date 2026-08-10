"""Every family of ruler, fed pure noise, must read its null.

A measure that reports structure in noise reports structure in every
architecture in the quiver, with the same confident decimals as a real result.
Prompt 07 established that seven of eleven §6.2 measures have a null a reader
would mistake for structure — the effective rank of noise is `d`, not zero — and
demoted them to diagnostic on exactly that ground.

The nulls are therefore *values*, not zeros, and they are re-measured here so
that a later change which quietly moves one is caught by the suite rather than
by a reviewer.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from architecture_mechanics.data.feature_program import condition_config, generate_dataset
from architecture_mechanics.metrics import geometry as G
from architecture_mechanics.metrics import statistics as S
from architecture_mechanics.metrics.capability import (
    EvaluationReference,
    Predictions,
    associative_recall_accuracy,
    feature_f1,
)
from architecture_mechanics.metrics.mechanism import (
    ACTIVITY_GATES,
    attention_retrieval,
    mechanism_is_active,
)

ROWS, FEATURES, WIDTH = 4096, 32, 48


def test_capability_metrics_read_chance_on_noise_predictions():
    evaluation = generate_dataset(
        condition_config("capacity_stressed", split="test", n_examples=512)
    )
    reference = EvaluationReference.from_dataset(evaluation)
    rng = np.random.default_rng(20260810)
    noise = Predictions(
        values=rng.random(reference.targets.shape),
        active_prob=rng.random(reference.targets.shape),
    )
    assert associative_recall_accuracy(noise, reference).value == 0.0
    # feature detection is rate-matched, so its null is the base rate, not zero
    base = float(
        reference.target_active[reference.supervised][:, reference.content_indices].mean()
    )
    assert feature_f1(noise, reference).value == pytest.approx(base, abs=0.02)


@pytest.fixture(scope="module")
def noise_geometry():
    rng = np.random.default_rng(20260810)
    values = np.where(rng.random((ROWS, FEATURES)) < 0.2, rng.random((ROWS, FEATURES)), 0.0)
    hidden = rng.standard_normal((ROWS, WIDTH))
    split = G.probe_split(np.arange(ROWS), seed=20260817)
    return G.measure_geometry(hidden, values, split, active=values != 0.0).scalars()


@pytest.mark.parametrize(
    "measure, low, high, why",
    [
        ("probe_macro_r2", -0.10, 0.02, "the null is zero, slightly negative out of sample"),
        ("probe_macro_auc", 0.47, 0.53, "chance"),
        ("mean_purity", 1 / FEATURES - 0.03, 1 / FEATURES + 0.03, "1/F, not zero"),
        ("interference_fraction", 0.35, 0.65, "off-diagonal mass equals the diagonal"),
        ("effective_rank", 0.90 * WIDTH, WIDTH, "the null is d — isotropic noise fills every direction"),
        ("participation_ratio", 0.90 * WIDTH, WIDTH, "as above"),
        ("capacity_total", 10.0, 30.0, "F random directions are about as spread out as F learned ones"),
        ("mean_abs_off_diagonal_cosine", 0.05, 0.20, "sqrt(2/pi d): near-orthogonality IS the null"),
        ("alignment_explained_mean", -0.01, 0.01, "the only alignment whose null is zero"),
        ("alignment_marginal_mean", 0.70, 0.95, "not zero: both estimators share the sampling noise"),
    ],
)
def test_geometry_measures_read_their_recorded_null_on_noise(
    noise_geometry, measure, low, high, why
):
    value = noise_geometry[measure]
    assert low <= value <= high, f"{measure} = {value} left its null band [{low}, {high}] — {why}"


def test_the_four_retained_geometry_measures_are_the_ones_built_on_a_readout():
    retained = {spec.name for spec in G.GEOMETRY_MEASURES if spec.status == "retained"}
    assert retained == {
        "probe_macro_r2",
        "probe_macro_auc",
        "mean_purity",
        "interference_fraction",
    }, "the retained set moved; state/07_geometry.md's reading of the null column is stale"


@pytest.mark.parametrize("pattern", ["uniform", "random"])
def test_the_mechanism_gates_refuse_attention_that_retrieves_nothing(pattern):
    evaluation = generate_dataset(
        condition_config("capacity_stressed", split="test", n_examples=64)
    )
    batch, heads = 64, 2
    length = int(evaluation.inputs.shape[1])
    causal = torch.ones(length, length, dtype=torch.bool).tril()
    if pattern == "uniform":
        weights = causal.double()
    else:
        weights = torch.rand(
            batch, heads, length, length, generator=torch.Generator().manual_seed(7),
            dtype=torch.float64,
        ).masked_fill(causal.logical_not(), 0.0)
    weights = weights / weights.sum(-1, keepdim=True)
    if weights.dim() == 2:
        weights = weights.expand(batch, heads, length, length).contiguous()

    report = attention_retrieval(weights, evaluation.programs[:batch], layer="layers.0.mix")
    assert report.lift < ACTIVITY_GATES["min_retrieval_lift"]
    assert report.argmax_hit_rate < 0.10

    entropy = -(weights.clamp_min(1e-30).log() * weights).sum(-1)
    prefix = torch.arange(1, length + 1).double().log().clamp_min(1e-12)
    stats = {
        "layers.0.mix.entropy_ratio": float((entropy[..., 1:] / prefix[1:]).mean()),
        "layers.0.mix.off_diagonal_mass": 1.0
        - float(weights[..., torch.arange(length), torch.arange(length)].mean()),
    }
    verdict = mechanism_is_active(stats, {"layers.0.mix": report})
    assert not verdict["active"]
    assert any("retrieval lift" in reason for reason in verdict["reasons"])


def test_the_adopted_estimator_holds_its_level_on_two_arms_from_one_distribution():
    """§7.4's whole point: `paired_t` was adopted because its measured
    false-positive rate at five seeds is 0.048, and the percentile bootstrap was
    demoted because its is 0.167. Re-measured at a reduced replicate count, so
    the band is wide; the calibration gate carries the precise numbers."""
    alpha, replicates, seeds = 0.05, 600, 5
    rng = np.random.default_rng(20260810)
    rejections = 0
    for replicate in range(replicates):
        x, y = rng.standard_normal(seeds), rng.standard_normal(seeds)
        control = [
            S.RunSummary(run_id=f"c{replicate}_{i}", seed=i, arm="control", metrics={"m": float(x[i])})
            for i in range(seeds)
        ]
        candidate = [
            S.RunSummary(run_id=f"k{replicate}_{i}", seed=i, arm="candidate", metrics={"m": float(y[i])})
            for i in range(seeds)
        ]
        rejections += S.paired_test(control, candidate, metric="m").p_value < alpha
    rate = rejections / replicates
    assert 0.02 <= rate <= 0.09, (
        f"paired_t rejected a true null {rate:.3f} of the time at {seeds} seeds; "
        f"it was adopted as the primary test on a measured 0.048"
    )
