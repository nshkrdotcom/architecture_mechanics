"""The §6.2 estimators, held to answers known before they were measured.

The constructed-case gate in ``geometry --selftest`` checks the *summary*
numbers. This file checks the pieces underneath them: that the direction
estimator recovers a weight matrix a real model was actually built with, that
measures claiming invariance really are invariant, and that every degenerate
input either raises or returns ``NaN`` rather than a fabricated number.

The recovery test is the important one. Every geometric measure in this module
is a function of the estimated directions, so an estimator that quietly returned
something else would move every number at once and contradict nothing.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from architecture_mechanics.metrics.geometry import (
    GeometryError,
    ProbeSplit,
    alignment,
    constructed_representation,
    effective_rank,
    estimate_feature_directions,
    feature_capacity,
    feature_cosine_similarity,
    feature_reconstruction,
    flatten_site,
    measure_geometry,
    participation_ratio,
    per_feature_purity,
    probe_split,
    representation_similarity,
)
from architecture_mechanics.metrics.geometry import (
    interference_matrix as interference,
)
from architecture_mechanics.models.common import ModelConfig
from architecture_mechanics.models.softmax import build_softmax_model


@pytest.fixture(scope="module")
def orthogonal():
    return constructed_representation("orthogonal_basis", n_rows=2048, seed=11)


def _row_split(n_rows: int, seed: int = 5) -> ProbeSplit:
    return probe_split(np.arange(n_rows), seed=seed)


# --------------------------------------------------------------------------- #
# The estimator recovers a basis it did not see
# --------------------------------------------------------------------------- #


def test_directions_recover_a_constructed_basis(orthogonal):
    directions = estimate_feature_directions(orthogonal.hidden, orthogonal.features)
    # 1e-5, not 1e-9: the ridge shrinks the recovered norms by RIDGE and that is
    # the estimator working, not drifting. A tolerance tight enough to fail here
    # would be a test of the regulariser's size rather than of the estimator.
    assert directions.directions == pytest.approx(orthogonal.basis, abs=1e-5)


def test_directions_recover_a_real_model_encoder():
    """The estimator's answer at the embedding site is the encoder's own weights.

    A position-free model, so the design matrix is not competing with a learned
    position term for the same variance: ``h = W x + b`` exactly, and recovering
    ``W`` from ``(h, x)`` is the whole claim the estimator makes. The trained
    runs keep positions on and record the recovered cosine as a diagnostic; this
    is where it is a check.
    """
    torch.manual_seed(0)
    config = ModelConfig(n_features=20, seq_len=6, d_model=16, n_layers=1, n_heads=2,
                         positional="none")
    model = build_softmax_model(config).eval()

    rng = np.random.default_rng(3)
    active = rng.random((256, 6, 20)) < 0.25
    values = np.where(active, rng.random((256, 6, 20)), 0.0)
    inputs = torch.tensor(values, dtype=torch.float32)
    with torch.no_grad():
        embedded = model.encoder(inputs)

    directions = estimate_feature_directions(
        embedded.reshape(-1, 16).numpy(), values.reshape(-1, 20)
    )
    expected = model.encoder.weight.detach().numpy().T
    assert directions.directions == pytest.approx(expected, abs=1e-4)

    report = alignment(
        embedded.reshape(-1, 16).numpy(), values.reshape(-1, 20), reference=expected
    )
    assert report.reference_mean == pytest.approx(1.0, abs=1e-6)


def test_a_feature_that_never_occurs_has_no_direction():
    rng = np.random.default_rng(0)
    values = np.where(rng.random((512, 4)) < 0.3, rng.random((512, 4)), 0.0)
    values[:, 3] = 0.0  # never active anywhere
    hidden = values @ np.eye(8)[:4]

    directions = estimate_feature_directions(hidden, values)
    assert directions.undefined.tolist() == [False, False, False, True]
    assert directions.directions[3] == pytest.approx(np.zeros(8))
    assert np.isnan(feature_cosine_similarity(directions)[3]).all()
    assert np.isnan(feature_capacity(directions).per_feature[3])
    # ...and the summary averages skip it rather than counting it as zero.
    assert feature_capacity(directions).mean == pytest.approx(1.0, abs=1e-6)


def test_collinear_features_are_reported_not_corrected():
    """Two features that always fire together share one direction arbitrarily."""
    rng = np.random.default_rng(1)
    base = np.where(rng.random((512, 3)) < 0.3, rng.random((512, 3)), 0.0)
    values = np.column_stack([base, base[:, 0]])  # feature 3 is a copy of feature 0
    hidden = values @ np.eye(8)[:4]

    directions = estimate_feature_directions(hidden, values)
    assert directions.collinear[[0, 3]].all()
    assert not directions.collinear[[1, 2]].any()
    assert directions.condition_number > 1e3


# --------------------------------------------------------------------------- #
# Dimensionality measures against hand-computable spectra
# --------------------------------------------------------------------------- #


def test_effective_rank_and_participation_ratio_on_a_flat_spectrum():
    rng = np.random.default_rng(2)
    hidden = np.zeros((4096, 10))
    hidden[:, :4] = rng.standard_normal((4096, 4))
    assert effective_rank(hidden) == pytest.approx(4.0, abs=0.1)
    assert participation_ratio(hidden) == pytest.approx(4.0, abs=0.1)


def test_effective_rank_of_a_rank_one_representation_is_one():
    rng = np.random.default_rng(2)
    hidden = rng.standard_normal((1024, 1)) @ np.ones((1, 12))
    assert effective_rank(hidden) == pytest.approx(1.0, abs=1e-9)
    assert participation_ratio(hidden) == pytest.approx(1.0, abs=1e-9)


def test_a_constant_representation_returns_zero_rather_than_dividing_by_zero():
    hidden = np.full((64, 5), 3.25)
    assert effective_rank(hidden) == 0.0
    assert participation_ratio(hidden) == 0.0


def test_participation_ratio_sees_a_tail_that_effective_rank_weights_differently():
    """The two dimensionality measures are not restatements of each other."""
    rng = np.random.default_rng(4)
    hidden = rng.standard_normal((8192, 20)) * np.concatenate([[6.0, 6.0], np.full(18, 0.35)])
    assert participation_ratio(hidden) < effective_rank(hidden) - 3.0


# --------------------------------------------------------------------------- #
# Invariance, and the things that are not invariant
# --------------------------------------------------------------------------- #


def test_representation_similarity_is_one_for_rotation_and_scale():
    rng = np.random.default_rng(6)
    hidden = rng.standard_normal((512, 12))
    rotation = np.linalg.qr(rng.standard_normal((12, 12)))[0]
    assert representation_similarity(hidden, hidden) == pytest.approx(1.0)
    assert representation_similarity(hidden, hidden @ rotation) == pytest.approx(1.0)
    assert representation_similarity(hidden, 7.5 * hidden) == pytest.approx(1.0)
    assert representation_similarity(hidden, hidden + 4.0) == pytest.approx(1.0)
    assert representation_similarity(hidden, rng.standard_normal((512, 12))) < 0.1


def test_representation_similarity_needs_matched_rows():
    rng = np.random.default_rng(7)
    with pytest.raises(GeometryError, match="matched rows"):
        representation_similarity(rng.standard_normal((10, 3)), rng.standard_normal((11, 3)))


def test_measures_follow_a_permutation_of_the_feature_labels(orthogonal):
    """Feature identity is a label. Permuting it must move nothing."""
    permutation = np.random.default_rng(8).permutation(orthogonal.features.shape[1])
    split = _row_split(orthogonal.hidden.shape[0])
    base = measure_geometry(orthogonal.hidden, orthogonal.features, split).scalars()
    moved = measure_geometry(
        orthogonal.hidden, orthogonal.features[:, permutation], split
    ).scalars()
    for name, value in base.items():
        if isinstance(value, float) and math.isfinite(value):
            assert moved[name] == pytest.approx(value, abs=1e-9), name


# --------------------------------------------------------------------------- #
# Guarded divisions
# --------------------------------------------------------------------------- #


def test_purity_is_nan_where_a_readout_responds_to_nothing():
    rng = np.random.default_rng(9)
    values = np.where(rng.random((512, 3)) < 0.3, rng.random((512, 3)), 0.0)
    hidden = values @ np.eye(6)[:3]
    split = _row_split(512)
    report = interference(hidden, values, split)
    zeroed = report.__class__(
        matrix=np.zeros_like(report.matrix),
        mean_abs_diagonal=0.0,
        mean_abs_off_diagonal=0.0,
        interference_fraction=float("nan"),
        n_train_rows=report.n_train_rows,
        n_eval_rows=report.n_eval_rows,
    )
    purity = per_feature_purity(zeroed, values.var(axis=0))
    assert np.isnan(purity).all()  # not 0.0, and not a ZeroDivisionError


def test_interference_fraction_is_nan_when_the_representation_carries_nothing():
    """A constant hidden state produces denormals, not zeros — and a ratio of
    denormals is a well-formed number with no meaning. The guard is on
    magnitude, so this reports ``NaN`` rather than a confident 0.58."""
    rng = np.random.default_rng(10)
    values = np.where(rng.random((512, 3)) < 0.3, rng.random((512, 3)), 0.0)
    report = interference(np.zeros((512, 6)), values, _row_split(512))
    assert math.isnan(report.interference_fraction)
    assert report.mean_abs_diagonal < 1e-20
    assert np.isnan(per_feature_purity(report, values.var(axis=0))).all()


def test_probe_r2_is_nan_for_a_feature_with_no_evaluation_variance():
    rng = np.random.default_rng(11)
    values = np.where(rng.random((512, 3)) < 0.3, rng.random((512, 3)), 0.0)
    values[:, 2] = 0.0
    hidden = values @ np.eye(6)[:3]
    report = feature_reconstruction(hidden, values, _row_split(512))
    assert math.isnan(report.r2[2])
    assert math.isnan(report.auc[2])
    assert report.n_scored_features == 2
    assert report.macro_r2 == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Input validation and site flattening
# --------------------------------------------------------------------------- #


def test_estimators_refuse_mismatched_or_non_finite_input():
    rng = np.random.default_rng(12)
    hidden = rng.standard_normal((32, 4))
    with pytest.raises(GeometryError, match="rows"):
        estimate_feature_directions(hidden, rng.standard_normal((31, 3)))
    bad = hidden.copy()
    bad[0, 0] = np.nan
    with pytest.raises(GeometryError, match="non-finite"):
        estimate_feature_directions(bad, rng.standard_normal((32, 3)))
    with pytest.raises(GeometryError, match="2-dimensional"):
        effective_rank(rng.standard_normal((4, 5, 6)))


def test_flatten_site_merges_heads_the_way_the_output_projection_reads_them():
    """``(B, H, T, d_head)`` must flatten to what ``out_proj`` is handed."""
    tensor = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
    merged = tensor.transpose(1, 2).reshape(2, 4, 15)  # the model's own merge
    assert flatten_site(tensor) == pytest.approx(merged.reshape(8, 15).numpy())
    assert flatten_site(torch.zeros(2, 4, 7)).shape == (8, 7)
    assert flatten_site(np.zeros((9, 7))).shape == (9, 7)
    with pytest.raises(GeometryError, match="cannot flatten"):
        flatten_site(np.zeros((2, 3, 4, 5, 6)))
