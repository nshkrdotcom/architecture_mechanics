"""Probes are split by example, and this file is the proof that it matters.

§7.4 forbids reporting a probe result without a matched-site baseline; the
reason a probe needs guarding at all is that it is trivially able to memorise.
Twelve positions of one sequence share a program, a key, and an answer, so a
probe fitted on eleven of them and scored on the twelfth has already seen the
answer — and would report a decodability that belongs to the split, not to the
representation.

The demonstration below is the whole argument: a representation in which
*nothing* is honestly decodable scores ``R^2 = 0.996`` under a row-wise split and
``-0.66`` under a by-example one. A discipline that changed no number would not
be a discipline.
"""

from __future__ import annotations

import numpy as np
import pytest

from architecture_mechanics.metrics.geometry import (
    GeometryError,
    ProbeSplit,
    feature_reconstruction,
    probe_split,
)


def _memorisable(n_examples: int = 64, per_example: int = 8, n_features: int = 8,
                 d_model: int = 64, seed: int = 20260810):
    """Features constant within an example; hidden state a code naming the example.

    Nothing linear maps the hidden state to the features except by memorising
    which code belongs to which example, so a probe's honest score here is zero.
    """
    rng = np.random.default_rng(seed)
    example_of_row = np.repeat(np.arange(n_examples), per_example)
    per_example_values = np.where(
        rng.random((n_examples, n_features)) < 0.3, rng.random((n_examples, n_features)), 0.0
    )
    values = per_example_values[example_of_row]
    codes = rng.standard_normal((n_examples, d_model))
    hidden = codes[example_of_row] + 0.01 * rng.standard_normal((values.shape[0], d_model))
    return hidden, values, example_of_row


def test_split_partitions_examples_not_rows():
    _, _, example_of_row = _memorisable()
    split = probe_split(example_of_row, seed=1)

    assert set(split.train.tolist()) & set(split.eval.tolist()) == set()
    assert set(split.train_examples.tolist()) & set(split.eval_examples.tolist()) == set()
    assert split.train.size + split.eval.size == example_of_row.size
    # Every row of an example lands on the same side, which is the property that
    # makes "the probe has not seen this sequence" true.
    for side, examples in ((split.train, split.train_examples), (split.eval, split.eval_examples)):
        assert set(example_of_row[side].tolist()) == set(examples.tolist())


def test_both_halves_come_from_the_same_template_families():
    """The probe answers "is this decodable", not "does this generalise".

    A further compositional holdout inside the probe split would confound the
    two questions, so the two halves are supposed to share templates and the
    split records how many they share as evidence for the design.
    """
    _, _, example_of_row = _memorisable()
    templates = np.asarray([f"t{index % 6}" for index in range(64)])
    split = probe_split(example_of_row, seed=2, template_of_example=templates)
    assert split.n_shared_templates == 6
    assert split.as_dict()["split_by"] == "example"


def test_a_row_wise_split_inflates_a_probe_that_has_learned_nothing():
    hidden, values, example_of_row = _memorisable()
    honest = probe_split(example_of_row, seed=3)

    order = np.random.default_rng(3).permutation(hidden.shape[0])
    half = hidden.shape[0] // 2
    leaky = ProbeSplit(
        train=np.sort(order[:half]),
        eval=np.sort(order[half:]),
        train_examples=np.array([-1]),
        eval_examples=np.array([-2]),
        n_examples=64,
        seed=3,
        train_fraction=0.5,
    )
    honest_r2 = feature_reconstruction(hidden, values, honest).macro_r2
    leaky_r2 = feature_reconstruction(hidden, values, leaky).macro_r2

    assert honest_r2 < 0.2
    assert leaky_r2 > 0.5
    assert leaky_r2 - honest_r2 > 0.5


def test_split_refuses_an_example_on_both_sides():
    with pytest.raises(GeometryError, match="an example appears on both sides"):
        ProbeSplit(
            train=np.array([0, 1]),
            eval=np.array([2, 3]),
            train_examples=np.array([0, 1]),
            eval_examples=np.array([1, 2]),
            n_examples=3,
            seed=0,
            train_fraction=0.5,
        )


def test_split_refuses_a_row_on_both_sides():
    with pytest.raises(GeometryError, match="a row appears on both sides"):
        ProbeSplit(
            train=np.array([0, 1, 2]),
            eval=np.array([2, 3]),
            train_examples=np.array([0]),
            eval_examples=np.array([1]),
            n_examples=2,
            seed=0,
            train_fraction=0.5,
        )


def test_split_refuses_an_empty_side():
    with pytest.raises(GeometryError, match="rows on both sides"):
        ProbeSplit(
            train=np.array([], dtype=np.int64),
            eval=np.array([0, 1]),
            train_examples=np.array([], dtype=np.int64),
            eval_examples=np.array([0]),
            n_examples=1,
            seed=0,
            train_fraction=0.5,
        )


def test_split_refuses_a_single_example():
    with pytest.raises(GeometryError, match="at least two examples"):
        probe_split(np.zeros(32, dtype=np.int64))


def test_split_refuses_a_degenerate_fraction():
    _, _, example_of_row = _memorisable()
    for fraction in (0.0, 1.0, -0.5, 2.0):
        with pytest.raises(GeometryError, match="train_fraction"):
            probe_split(example_of_row, train_fraction=fraction)


def test_split_always_leaves_at_least_one_example_on_each_side():
    """Even an extreme fraction cannot empty a side of a two-example split."""
    rows = np.repeat(np.arange(2), 4)
    for fraction in (0.01, 0.99):
        split = probe_split(rows, train_fraction=fraction)
        assert split.train_examples.size == 1 and split.eval_examples.size == 1


def test_split_is_deterministic_in_its_seed():
    _, _, example_of_row = _memorisable()
    first = probe_split(example_of_row, seed=17)
    again = probe_split(example_of_row, seed=17)
    other = probe_split(example_of_row, seed=18)
    assert first.train.tolist() == again.train.tolist()
    assert first.train.tolist() != other.train.tolist()
