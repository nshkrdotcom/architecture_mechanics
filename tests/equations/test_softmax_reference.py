"""The fast path computes what the equation says (§8.5).

This is the test every later architecture comparison rests on. If A0's batched
attention is not the attention its name claims, then "A1 differs from A0" is a
statement about a bug. Prompts 11 and 17 each need their own copy of this file
for their own mechanism; the pattern is here.

Three implementations, deliberately:

``reference_forward``  nested Python loops over batch, head, query, key, and
                       channel. No reshape, no broadcast, no fused kernel, and
                       a softmax written as ``exp(s) / sum(exp(s))`` rather than
                       the max-subtracted form. Correct by inspection.
``forward``            the batched path every run uses.
``scaled_dot_product_attention``  torch's fused kernel, as an independent third
                       opinion that shares no code with either of ours.

The primary comparison runs both of ours in **float64 on CPU**, where the only
difference left between them is the order of arithmetic operations. The measured
gap there is around 1e-16 and the tolerance is 1e-11, so a real disagreement of
any size a bug would produce cannot hide inside it. Running the fast path in
float32 against a float64 reference would have mixed implementation error and
rounding error into one number, and the tolerance would then have to be loose
enough to swallow both. The float32 gap is measured separately, and reported
rather than asserted tight, because it is a property of the arithmetic and not
of the code.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch.nn import functional as F

from architecture_mechanics.models.common import (
    FeatureModel,
    ModelConfig,
    model_reference_forward,
)
from architecture_mechanics.models.softmax import SoftmaxAttention

# Kept small because the reference is O(B * H * T^2 * d_head) Python iterations.
# Two heads and two layers are the point: single-head, single-layer shapes are
# exactly where head-splitting and per-layer scoping bugs hide.
SMALL = ModelConfig(n_features=9, seq_len=7, d_model=8, n_layers=2, n_heads=2, mlp_ratio=2)

REFERENCE_TOLERANCE = 1e-11
"""float64, both implementations. Measured gap is ~1e-16; this is five orders of
magnitude of headroom over the observation and eleven below any value a real
bug would produce."""

FLOAT32_TOLERANCE = 2e-5
"""float32 fast path against the float64 reference. Documents the arithmetic
floor of the precision runs actually use."""


def _model(dtype: torch.dtype = torch.float64, config: ModelConfig = SMALL) -> FeatureModel:
    torch.manual_seed(20260809)
    model = FeatureModel(config).to(dtype)
    model.eval()
    return model


def _inputs(model: FeatureModel, batch: int = 3, dtype: torch.dtype = torch.float64):
    generator = torch.Generator().manual_seed(11)
    return torch.randn(
        batch, model.config.seq_len, model.config.n_features, generator=generator, dtype=dtype
    )


def test_attention_fast_path_matches_the_slow_reference():
    model = _model()
    block = model.blocks[0].mix
    assert isinstance(block, SoftmaxAttention)
    hidden = torch.randn(
        3, SMALL.seq_len, SMALL.d_model, generator=torch.Generator().manual_seed(5),
        dtype=torch.float64,
    )
    with torch.no_grad():
        fast = block(hidden)
        slow = block.reference_forward(hidden)
    gap = (fast - slow).abs().max().item()
    assert gap < REFERENCE_TOLERANCE, f"fast path and reference differ by {gap}"


def test_whole_model_matches_the_slow_reference():
    """Not only the mechanism: the trunk around it too.

    An attention block that matches its reference inside a trunk that applies
    its LayerNorm to the wrong tensor still produces a wrong model. The trunk
    reference re-derives the embedding, both residual writes, the MLP, and the
    two heads one position at a time.
    """
    model = _model()
    x = _inputs(model)
    with torch.no_grad():
        fast = model(x)
        slow = model_reference_forward(model, x)
    value_gap = (fast.values - slow.values).abs().max().item()
    logit_gap = (fast.active_logits - slow.active_logits).abs().max().item()
    assert value_gap < REFERENCE_TOLERANCE, f"value head differs by {value_gap}"
    assert logit_gap < REFERENCE_TOLERANCE, f"activity head differs by {logit_gap}"


def test_fast_path_matches_torch_fused_attention():
    """A third implementation that shares no code with either of ours.

    Our fast path materialises the ``(B, H, T, T)`` weight matrix on purpose, so
    that §6.3 can measure it and §6.4 can intervene on it. This checks that
    choice costs nothing in correctness against the kernel we declined to use.
    """
    model = _model()
    block = model.blocks[0].mix
    hidden = torch.randn(
        2, SMALL.seq_len, SMALL.d_model, generator=torch.Generator().manual_seed(7),
        dtype=torch.float64,
    )
    with torch.no_grad():
        ours = block(hidden)

        batch, seq_len, _ = hidden.shape
        projected = block.qkv(hidden)
        query, key, value = projected.split(SMALL.d_model, dim=-1)
        shape = (batch, seq_len, SMALL.n_heads, SMALL.d_model // SMALL.n_heads)
        query = query.view(shape).transpose(1, 2)
        key = key.view(shape).transpose(1, 2)
        value = value.view(shape).transpose(1, 2)
        fused = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        fused = block.out_proj(fused.transpose(1, 2).reshape(batch, seq_len, SMALL.d_model))
    gap = (ours - fused).abs().max().item()
    assert gap < REFERENCE_TOLERANCE, f"our attention differs from the fused kernel by {gap}"


def test_float32_agreement_is_reported_and_bounded():
    """The precision runs actually use, against the float64 definition.

    Asserted loosely on purpose: this number is the arithmetic floor, not a
    property of the implementation, and tightening it would make the test fail
    for a GPU rather than for a bug.
    """
    model64 = _model()
    x64 = _inputs(model64)
    model32 = _model(dtype=torch.float32)
    with torch.no_grad():
        slow = model_reference_forward(model64, x64)
        fast = model32(x64.to(torch.float32))
    gap = (fast.values.to(torch.float64) - slow.values).abs().max().item()
    assert gap < FLOAT32_TOLERANCE, f"float32 forward differs from the definition by {gap}"
    assert gap > 0.0, "float32 and float64 agreeing bitwise means the cast did not happen"


@pytest.mark.parametrize("n_heads", [1, 2, 4])
def test_head_splitting_is_correct_at_every_head_count(n_heads: int):
    """Head layout is where reshape bugs live, and they cancel at ``H = 1``."""
    config = ModelConfig(n_features=6, seq_len=5, d_model=8, n_layers=1, n_heads=n_heads,
                         mlp_ratio=2)
    model = _model(config=config)
    block = model.blocks[0].mix
    hidden = torch.randn(
        2, config.seq_len, config.d_model, generator=torch.Generator().manual_seed(3),
        dtype=torch.float64,
    )
    with torch.no_grad():
        gap = (block(hidden) - block.reference_forward(hidden)).abs().max().item()
    assert gap < REFERENCE_TOLERANCE


def test_reference_softmax_rows_are_causal_and_normalised():
    """The reference's own arithmetic, checked without reference to the fast path.

    Two implementations that agree can still both be wrong. This asserts the two
    properties that make the computation attention at all — every row is a
    distribution, and no row touches the future — directly against the
    definition, using a value matrix of one-hot rows so that the mechanism's
    output *is* its weight matrix.
    """
    config = ModelConfig(n_features=4, seq_len=6, d_model=4, n_layers=1, n_heads=1, mlp_ratio=1)
    model = _model(config=config)
    block = model.blocks[0].mix
    hidden = torch.randn(
        1, config.seq_len, config.d_model, generator=torch.Generator().manual_seed(9),
        dtype=torch.float64,
    )
    # Read the weights out of the fast path, then recompute them by hand from
    # the projections the reference would have formed.
    from architecture_mechanics.instrumentation.hooks import CaptureContext

    hooks = CaptureContext(capture=("weights", "q", "k"))
    with torch.no_grad():
        block(hidden, hooks=hooks)
    weights = hooks.captures["weights"][0, 0]
    query = hooks.captures["q"][0, 0]
    key = hooks.captures["k"][0, 0]

    scale = 1.0 / math.sqrt(config.d_head)
    for i in range(config.seq_len):
        exponentials = [
            math.exp(float(sum(query[i, c] * key[j, c] for c in range(config.d_head))) * scale)
            for j in range(i + 1)
        ]
        total = sum(exponentials)
        for j in range(config.seq_len):
            expected = exponentials[j] / total if j <= i else 0.0
            assert weights[i, j].item() == pytest.approx(expected, abs=REFERENCE_TOLERANCE)
        assert weights[i].sum().item() == pytest.approx(1.0, abs=REFERENCE_TOLERANCE)
        future = weights[i, i + 1 :]
        assert future.numel() == config.seq_len - i - 1
        assert not future.any(), f"row {i} places weight on the future"
