"""R1's positive control, fed things that cannot solve it.

`tests/metrics/test_calibration.py` already checks that the marginal, chance and
the frequency ceiling fail it and that the verdict turns exactly at the
threshold. What was never checked is the case the control actually exists for:
a **model** whose mixing mechanism cannot move information between positions.
R1 is the gate that stands between a broken implementation and sixteen cells of
scientific interpretation, so the thing it must refuse is a broken model.

Prompt 10 established the numbers these assertions bracket by training a real A0
on this condition (0.9055, PASS) and then crippling it two ways (0.0028 and
0.0000, FAIL). Training is not repeated here — it needs a GPU and 42 seconds —
so the cripples used below are untrained or algorithmic and the trained pass is
represented by the program oracle.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from architecture_mechanics.metrics.capability import (
    POSITIVE_CONTROL_THRESHOLD,
    Predictions,
    ProgramOracle,
    fit_marginal,
    positive_control,
    positive_control_datasets,
)
from architecture_mechanics.models.common import FeatureModel, ModelConfig

EXAMPLES = 512


@pytest.fixture(scope="module")
def splits():
    return positive_control_datasets(n_examples=EXAMPLES)


def _verdict(predict):
    return positive_control(predict, n_examples=EXAMPLES)


def test_the_oracle_passes_so_the_bar_is_reachable(splits):
    """Non-vacuity. Without this row, everything below is satisfied by a control
    that refuses every candidate."""
    train, _ = splits
    result = _verdict(ProgramOracle(fallback=fit_marginal(train)).predict)
    assert result.passed and result.instrument_ok
    assert result.value == pytest.approx(1.0)


def test_a_model_whose_mixer_moves_nothing_fails(splits):
    """The cripple the control exists for: an A0 whose mixing branch contributes
    exactly zero, so the trunk is a position-wise MLP and no token can move."""
    train, evaluation = splits
    config = ModelConfig(
        n_features=train.n_features,
        seq_len=int(train.inputs.shape[1]),
        d_model=train.config.d_recommended,
        n_layers=2,
        n_heads=2,
    )
    model = FeatureModel(config)
    for block in model.blocks:
        for parameter in block.mix.parameters():
            torch.nn.init.zeros_(parameter)

    @torch.no_grad()
    def predict(dataset):
        model.eval()
        output = model(dataset.inputs)
        return Predictions(
            values=output.values.numpy().astype(np.float64),
            active_prob=output.active_prob.numpy().astype(np.float64),
        )

    result = _verdict(predict)
    assert not result.passed
    assert result.instrument_ok, "the failure must be attributed to the model, not the ruler"
    assert result.value < POSITIVE_CONTROL_THRESHOLD
    del evaluation


def test_the_strongest_predictor_that_never_reads_the_key_fails(splits):
    """A fixed-offset copier is the strongest strategy available to something
    that performs no addressing at all: on this condition the source is one or
    two positions back, so copying `t-1` is right about half the time.

    It must fail, and it must fail *well below* the bar rather than just under
    it — otherwise the bar is measuring luck. Prompt 10 measured 0.4951 at
    offset 1 and 0.4968 at offset 2 on the full R1 budget.
    """
    _, evaluation = splits
    content = np.asarray(evaluation.content_indices, dtype=np.int64)

    def copier(offset: int):
        def predict(dataset):
            inputs = dataset.inputs.numpy().astype(np.float64)
            active = dataset.active_mask.numpy().astype(np.float64)
            values = np.zeros_like(inputs)
            prob = np.zeros_like(inputs)
            values[:, offset:, content] = inputs[:, :-offset, content]
            prob[:, offset:, content] = active[:, :-offset, content]
            return Predictions(values=values, active_prob=prob)

        return predict

    for offset in (1, 2):
        result = _verdict(copier(offset))
        assert not result.passed, f"offset {offset} cleared the positive control"
        assert result.instrument_ok
        assert result.value < 0.60, (
            f"offset {offset} scored {result.value}: an addressing-free strategy is close "
            f"enough to the bar that the bar no longer separates retrieval from copying"
        )


def test_the_positive_control_is_solved_by_ordinal_addressing(splits):
    """Recorded as a property, not a complaint. `n_associations = 1`, so there is
    exactly one keyed position and "return the value bound to this key" and
    "return the value at the one keyed position" are the same instruction.

    R1 therefore validates that the mixer can *transport* a marked position's
    content. It does not validate content addressing, and no later mission
    should cite a green R1 as evidence that it does. The generator's own
    selftest carries the matching invariant
    (`positive_control_addressing_is_ordinal`).
    """
    _, evaluation = splits
    keyed = []
    for record in evaluation.programs:
        bindings = [p for p in record.positions if p.op_code == "BIND"]
        keyed.append(len(bindings))
    assert set(keyed) == {1}, (
        "the positive control now has more than one binding; R1's scope has changed and "
        "state/10_instrument_review.md's reading of it is stale"
    )
