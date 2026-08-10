"""A1's fast path computes what the equation says, and so does its recursion (§8.5).

Prompt 04 built this pattern for A0 with two implementations and a third
independent opinion. A1 needs a third of its own and gets a fourth check that A0
cannot have, because A1 is two algorithms rather than one:

``reference_forward``   nested Python loops over batch, head, query, key and
                        channel, written as the kernel-sum definition. No
                        cumulative sum and no state matrix anywhere, so it shares
                        the equation with the fast forms and nothing else.
``forward``             the parallel form every run uses: form every rank-one
                        write, scan them, read the state.
``recurrent_forward``   the explicit ``O(T)`` recursion, the form prompt 19's
                        state interventions hook.
the quadratic form      ``phi(q) phi(k)^T`` masked and row-normalised, written
                        inline in this file. An independent third opinion that
                        shares no code with any of ours, standing where torch's
                        fused kernel stands for A0.

**The parallel/recurrent comparison is the one that matters here.** They are the
same mathematics. A disagreement is not a tolerance question, it is one of them
being wrong — and since interventions will run through the recurrent form while
every recorded number comes from the parallel one, a disagreement would put the
causal evidence and the capability evidence on two different models.

Everything primary runs in **float64 on CPU**, where the only difference left
between implementations is the order of arithmetic. The float32 gap is measured
and reported rather than asserted tight, because it is a property of the
arithmetic and not of the code.
"""

from __future__ import annotations

import pytest
import torch

from architecture_mechanics.instrumentation.hooks import CaptureContext, capture_all
from architecture_mechanics.models.common import (
    FeatureModel,
    ModelConfig,
    model_reference_forward,
)
from architecture_mechanics.models.linear import (
    NORMALIZER_FLOOR,
    LinearAttention,
    feature_map,
)

# Kept small because the reference is O(B * H * T^2 * d_head) Python iterations.
# Two heads and two layers are the point: single-head, single-layer shapes are
# exactly where head-splitting and per-layer scoping bugs hide.
SMALL = ModelConfig(n_features=9, seq_len=7, d_model=8, n_layers=2, n_heads=2, mlp_ratio=2,
                    arch="linear")

REFERENCE_TOLERANCE = 1e-11
"""float64, both implementations. The same bar A0's equation test uses."""

FLOAT32_TOLERANCE = 2e-5
"""float32 fast path against the float64 reference. The arithmetic floor of the
precision runs actually use."""


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


def _hidden(config: ModelConfig, batch: int, seed: int) -> torch.Tensor:
    return torch.randn(
        batch, config.seq_len, config.d_model,
        generator=torch.Generator().manual_seed(seed), dtype=torch.float64,
    )


# --------------------------------------------------------------------------- #
# The three implementations against each other
# --------------------------------------------------------------------------- #


def test_parallel_form_matches_the_slow_reference():
    block = _model().blocks[0].mix
    assert isinstance(block, LinearAttention)
    hidden = _hidden(SMALL, 3, 5)
    with torch.no_grad():
        gap = (block(hidden) - block.reference_forward(hidden)).abs().max().item()
    assert gap < REFERENCE_TOLERANCE, f"parallel form and reference differ by {gap}"


def test_recurrent_form_matches_the_slow_reference():
    """The recursion is checked against the definition too, not only against
    the parallel form. Two implementations that agree can both be wrong, and
    these two share the idea of accumulating a state; the reference does not."""
    block = _model().blocks[0].mix
    hidden = _hidden(SMALL, 3, 5)
    with torch.no_grad():
        gap = (block.recurrent_forward(hidden) - block.reference_forward(hidden)).abs().max()
    assert gap.item() < REFERENCE_TOLERANCE, f"recurrent form and reference differ by {gap}"


def test_the_parallel_and_recurrent_forms_agree():
    """The equivalence this architecture stands on.

    Every recorded number comes from the parallel form; every state intervention
    will come from the recurrent one. If these disagree, the causal evidence and
    the capability evidence describe two different models.
    """
    block = _model().blocks[0].mix
    for seed in (5, 17, 101):
        hidden = _hidden(SMALL, 4, seed)
        with torch.no_grad():
            gap = (block(hidden) - block.recurrent_forward(hidden)).abs().max().item()
        assert gap < REFERENCE_TOLERANCE, f"parallel and recurrent differ by {gap} at seed {seed}"


def test_the_two_forms_agree_on_the_state_and_not_only_on_the_output():
    """Equal outputs can hide unequal states — and the state is what prompt 19
    intervenes on. Every ``S_t`` from the scan is compared against every ``S_t``
    the recursion carried."""
    block = _model().blocks[0].mix
    hidden = _hidden(SMALL, 2, 23)

    parallel = CaptureContext(capture=("state_post", "state_pre", "write"))
    recurrent = CaptureContext(capture=("state_post", "state_pre", "write"))
    with torch.no_grad():
        block(hidden, hooks=parallel)
        block.recurrent_forward(hidden, hooks=recurrent)

    for name in ("state_pre", "write", "state_post"):
        scanned = parallel.captures[name]
        for t in range(SMALL.seq_len):
            stepped = recurrent.captures[f"step.{t}.{name}"]
            gap = (scanned[:, :, t] - stepped).abs().max().item()
            assert gap < REFERENCE_TOLERANCE, f"{name} differs at step {t} by {gap}"


def test_parallel_form_matches_an_independent_quadratic_implementation():
    """A fourth opinion, written here and sharing no code with the module.

    Linear attention is normalized kernel attention, so it can be computed the
    way softmax attention is: build the ``T x T`` affinity matrix, mask it,
    divide each row by its sum, and mix. This is what A1 declines to do at run
    time — the quadratic object is the thing the architecture exists to avoid —
    which makes it exactly the right independent check.
    """
    model = _model()
    block = model.blocks[0].mix
    hidden = _hidden(SMALL, 2, 7)
    with torch.no_grad():
        ours = block(hidden)

        batch, seq_len, _ = hidden.shape
        d_head = SMALL.d_model // SMALL.n_heads
        projected = block.qkv(hidden)
        query, key, value = projected.split(SMALL.d_model, dim=-1)
        shape = (batch, seq_len, SMALL.n_heads, d_head)
        phi_q = feature_map(query.view(shape).transpose(1, 2))
        phi_k = feature_map(key.view(shape).transpose(1, 2))
        value = value.view(shape).transpose(1, 2)

        affinity = (phi_q @ phi_k.transpose(-2, -1)).tril()
        weights = affinity / affinity.sum(dim=-1, keepdim=True)
        mixed = weights @ value
        theirs = block.out_proj(mixed.transpose(1, 2).reshape(batch, seq_len, SMALL.d_model))
    gap = (ours - theirs).abs().max().item()
    assert gap < REFERENCE_TOLERANCE, f"our linear attention differs from the quadratic form by {gap}"


def test_whole_model_matches_the_slow_reference():
    """Not only the mechanism: the trunk around it too.

    A mixer that matches its reference inside a trunk that applies its LayerNorm
    to the wrong tensor still produces a wrong model.
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


def test_float32_agreement_is_reported_and_bounded():
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
    """Head layout is where reshape bugs live, and they cancel at ``H = 1``.

    A1 is more exposed to them than A0: the state carries a per-head
    ``d_head x (d_head + 1)`` matrix, so a head-axis error changes the *shape* of
    the memory and not only which channels are mixed.
    """
    config = ModelConfig(n_features=6, seq_len=5, d_model=8, n_layers=1, n_heads=n_heads,
                         mlp_ratio=2, arch="linear")
    block = _model(config=config).blocks[0].mix
    hidden = _hidden(config, 2, 3)
    with torch.no_grad():
        assert (block(hidden) - block.reference_forward(hidden)).abs().max().item() \
            < REFERENCE_TOLERANCE
        assert (block(hidden) - block.recurrent_forward(hidden)).abs().max().item() \
            < REFERENCE_TOLERANCE


# --------------------------------------------------------------------------- #
# The induced attention matrix is what §6.3 will read
# --------------------------------------------------------------------------- #


def test_the_induced_matrix_is_causal_and_row_stochastic():
    """Checked against the definition, without reference to any fast path.

    §6.3's activity gates and retrieval lift are only comparable with A0's if
    A1's matrix really is a distribution over the causal prefix. Recomputed here
    from the captured feature maps by hand, one entry at a time.
    """
    config = ModelConfig(n_features=4, seq_len=6, d_model=4, n_layers=1, n_heads=1, mlp_ratio=1,
                         arch="linear")
    block = _model(config=config).blocks[0].mix
    hidden = _hidden(config, 1, 9)

    hooks = CaptureContext(capture=("phi_q", "phi_k"))
    with torch.no_grad():
        block(hidden, hooks=hooks)
    matrix = block.attention_matrix(dict(hooks.captures))[0, 0]
    phi_q = hooks.captures["phi_q"][0, 0]
    phi_k = hooks.captures["phi_k"][0, 0]

    for i in range(config.seq_len):
        affinities = [
            float(sum(phi_q[i, c] * phi_k[j, c] for c in range(config.d_head)))
            for j in range(i + 1)
        ]
        assert all(a > 0.0 for a in affinities), "elu+1 produced a non-positive affinity"
        total = sum(affinities)
        for j in range(config.seq_len):
            expected = affinities[j] / total if j <= i else 0.0
            assert matrix[i, j].item() == pytest.approx(expected, abs=REFERENCE_TOLERANCE)
        assert matrix[i].sum().item() == pytest.approx(1.0, abs=REFERENCE_TOLERANCE)
        assert not matrix[i, i + 1:].any(), f"row {i} places weight on the future"


def test_the_induced_matrix_reproduces_the_readout_the_model_used():
    """The measurement is only honest if mixing by it gives the read back.

    ``attention_matrix`` is derived rather than hooked, so nothing in the forward
    pass forces it to describe the forward pass. This does: ``A @ v`` against the
    ``readout`` site the mechanism actually produced.
    """
    block = _model().blocks[0].mix
    hidden = _hidden(SMALL, 2, 31)
    hooks = capture_all()
    with torch.no_grad():
        block(hidden, hooks=hooks)
    captures = dict(hooks.captures)
    matrix = block.attention_matrix(captures)
    mixed = matrix @ captures["v"].to(torch.float64)
    gap = (mixed - captures["readout"].to(torch.float64)).abs().max().item()
    assert gap < REFERENCE_TOLERANCE, f"the induced matrix does not reproduce the readout ({gap})"


def test_the_normalizer_floor_does_not_bind_at_ordinary_scale():
    """The clamp exists for underflow and for interventions, not for arithmetic.

    If it were binding on ordinary inputs, the induced matrix would stop summing
    to one and every §6.3 number would quietly become something else.
    """
    block = _model().blocks[0].mix
    hidden = _hidden(SMALL, 3, 43)
    hooks = CaptureContext(capture=("phi_q", "phi_k"))
    with torch.no_grad():
        block(hidden, hooks=hooks)
    normalizer = torch.einsum(
        "bhtk,bhtk->bht", hooks.captures["phi_q"], hooks.captures["phi_k"].cumsum(dim=2)
    )
    assert normalizer.min().item() > 1e6 * NORMALIZER_FLOOR
