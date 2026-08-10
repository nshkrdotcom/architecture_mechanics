"""R0 for A0: the §8.5 invariants, each proven rather than assumed.

§7.3's R0 is shapes, causal masking, deterministic state updates, identity
controls, reference-equation comparison, finite gradients, and hook no-op
equivalence. The reference comparison lives in ``tests/equations``; everything
else is here.

Two of these are written as *perturbation* tests rather than as inspections of
the mask, because a mask is easy to build correctly and then apply to the wrong
axis. Changing a later position's input and demanding that no earlier output
moves by a single bit tests the property the mask exists to provide, and it
would still catch the bug if the mask were deleted entirely and replaced by a
comment.
"""

from __future__ import annotations

import pytest
import torch

from architecture_mechanics.instrumentation.hooks import (
    NO_HOOKS,
    CaptureContext,
    HookSiteError,
    capture_all,
)
from architecture_mechanics.models.common import (
    FeatureModel,
    ModelConfig,
    ModelConfigError,
    count_parameters,
    parameter_matched_config,
    parameter_report,
    primitive_names,
)
from architecture_mechanics.models.softmax import (
    A0_VARIANTS,
    SoftmaxAttention,
    build_softmax_model,
    parameter_matched_variant,
)

CONFIG = ModelConfig(n_features=36, seq_len=12, d_model=48, n_layers=2, n_heads=2, mlp_ratio=4)


@pytest.fixture
def model() -> FeatureModel:
    torch.manual_seed(20260809)
    built = FeatureModel(CONFIG)
    built.eval()
    return built


@pytest.fixture
def inputs() -> torch.Tensor:
    generator = torch.Generator().manual_seed(4242)
    return torch.rand(5, CONFIG.seq_len, CONFIG.n_features, generator=generator)


# --------------------------------------------------------------------------- #
# Shapes and registration
# --------------------------------------------------------------------------- #


def test_output_shapes_are_one_prediction_per_feature_per_position(model, inputs):
    with torch.no_grad():
        output = model(inputs)
    expected = (inputs.shape[0], CONFIG.seq_len, CONFIG.n_features)
    assert tuple(output.values.shape) == expected
    assert tuple(output.active_logits.shape) == expected
    assert output.active_prob.min() >= 0.0 and output.active_prob.max() <= 1.0


def test_shorter_sequences_are_accepted_and_longer_ones_refused(model):
    short = torch.rand(2, CONFIG.seq_len - 4, CONFIG.n_features)
    with torch.no_grad():
        assert model(short).values.shape[1] == CONFIG.seq_len - 4
    with pytest.raises(ModelConfigError):
        model(torch.rand(2, CONFIG.seq_len + 1, CONFIG.n_features))


def test_feature_count_mismatch_is_refused_rather_than_broadcast(model):
    with pytest.raises(ModelConfigError):
        model(torch.rand(2, CONFIG.seq_len, CONFIG.n_features + 1))


def test_softmax_is_registered_and_buildable():
    assert "softmax" in primitive_names()
    assert isinstance(build_softmax_model(CONFIG).blocks[0].mix, SoftmaxAttention)
    with pytest.raises(ValueError):
        build_softmax_model(ModelConfig(n_features=4, seq_len=4, arch="not_a_mechanism"))


def test_a0_declares_both_required_variants():
    assert A0_VARIANTS == ("ordinary_residual", "parameter_matched")
    assert CONFIG.residual_write == "ordinary"


# --------------------------------------------------------------------------- #
# Causal masking, by perturbation
# --------------------------------------------------------------------------- #


def test_a_later_position_cannot_change_an_earlier_output(model, inputs):
    with torch.no_grad():
        before = model(inputs)
        perturbed = inputs.clone()
        perturbed[:, -1] = perturbed[:, -1] + 7.5
        after = model(perturbed)
    assert torch.equal(after.values[:, :-1], before.values[:, :-1])
    assert torch.equal(after.active_logits[:, :-1], before.active_logits[:, :-1])
    # The control: the perturbed position itself must move, or the test above
    # would pass for a model that ignores its input entirely.
    assert not torch.equal(after.values[:, -1], before.values[:, -1])


@pytest.mark.parametrize("position", [1, 3, 6, 11])
def test_every_position_is_blind_to_every_later_one(model, inputs, position):
    with torch.no_grad():
        before = model(inputs)
        perturbed = inputs.clone()
        perturbed[:, position] = perturbed[:, position] + 3.0
        after = model(perturbed)
    leak = (after.values[:, :position] - before.values[:, :position]).abs()
    assert leak.numel() == 0 or leak.max().item() == 0.0
    assert not torch.equal(after.values[:, position], before.values[:, position])


def test_attention_weights_are_zero_on_the_future(model, inputs):
    hooks = CaptureContext(capture=("weights",))
    with torch.no_grad():
        model(inputs, hooks=hooks)
    for name, weights in hooks.captures.items():
        upper = torch.triu(weights, diagonal=1)
        assert upper.abs().max().item() == 0.0, f"{name} attends to the future"
        rows = weights.sum(dim=-1)
        assert torch.allclose(rows, torch.ones_like(rows), atol=1e-6), f"{name} rows are not distributions"


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


def test_forward_is_bitwise_deterministic(model, inputs):
    with torch.no_grad():
        first = model(inputs)
        second = model(inputs)
    assert torch.equal(first.values, second.values)
    assert torch.equal(first.active_logits, second.active_logits)


def test_the_same_seed_builds_the_same_weights():
    torch.manual_seed(20260809)
    left = FeatureModel(CONFIG)
    torch.manual_seed(20260809)
    right = FeatureModel(CONFIG)
    for (name, a), (_, b) in zip(left.named_parameters(), right.named_parameters()):
        assert torch.equal(a, b), f"{name} differs between two identically seeded builds"

    torch.manual_seed(20260810)
    other = FeatureModel(CONFIG)
    assert not torch.equal(
        left.encoder.weight, other.encoder.weight
    ), "a different seed produced identical weights, so seeding is not reaching init"


def test_measuring_a_parameter_count_does_not_consume_the_generator():
    """``parameters_for`` builds throwaway models; it must not move the RNG.

    If it did, a run that consulted the parameter-matched control before
    building its model would get different weights from one that did not, and
    the difference would look like an architecture effect.
    """
    torch.manual_seed(7)
    baseline = FeatureModel(CONFIG).encoder.weight.clone()
    torch.manual_seed(7)
    parameter_matched_config(CONFIG, 50_000)
    assert torch.equal(FeatureModel(CONFIG).encoder.weight, baseline)


# --------------------------------------------------------------------------- #
# Gradients
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("magnitude", [1.0, 50.0, 1000.0])
def test_gradients_are_finite_under_large_magnitude_inputs(model, inputs, magnitude):
    model.train()
    model.zero_grad(set_to_none=True)
    output = model(inputs * magnitude)
    (output.values.square().mean() + output.active_logits.square().mean()).backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    for parameter, gradient in zip(model.parameters(), grads):
        assert torch.isfinite(gradient).all(), f"non-finite gradient at {tuple(parameter.shape)}"


def test_masked_positions_do_not_produce_nan_in_the_attention_softmax(model):
    """A row of the mask is all ``-inf`` only if a position has no valid key.

    Causality guarantees every row keeps at least its own position, so the
    softmax never sees an all-``-inf`` row. This holds it to that, including at
    sequence length one where the guarantee is tightest.
    """
    for seq_len in (1, 2, CONFIG.seq_len):
        hooks = CaptureContext(capture=("weights", "scores"))
        with torch.no_grad():
            output = model(torch.rand(2, seq_len, CONFIG.n_features), hooks=hooks)
        assert torch.isfinite(output.values).all()
        for weights in hooks.captures.values():
            assert torch.isfinite(weights[weights > float("-inf")]).all()


# --------------------------------------------------------------------------- #
# Hooks
# --------------------------------------------------------------------------- #


def test_capturing_every_site_does_not_change_one_bit_of_the_output(model, inputs):
    with torch.no_grad():
        clean = model(inputs, hooks=NO_HOOKS)
        hooks = capture_all()
        hooked = model(inputs, hooks=hooks)
    assert torch.equal(clean.values, hooked.values)
    assert torch.equal(clean.active_logits, hooked.active_logits)
    assert len(hooks.captures) == len(model.hook_sites())


def test_declared_hook_sites_are_exactly_the_ones_the_forward_pass_reaches(model, inputs):
    hooks = capture_all()
    with torch.no_grad():
        model(inputs, hooks=hooks)
    assert list(hooks.visited) == list(model.hook_sites())


def test_every_declared_mixing_site_is_reachable_by_its_local_name(model, inputs):
    for name in SoftmaxAttention.SITES:
        hooks = CaptureContext(capture=(name,))
        with torch.no_grad():
            model(inputs, hooks=hooks)
        assert len(hooks.captures) == CONFIG.n_layers, f"site {name!r} not captured in every layer"


def test_no_two_sites_share_a_bare_name(model):
    """Bare-name addressing is supported, so a shared name silently over-captures.

    Found the first time this file ran: the trunk called its post-LayerNorm
    tensor ``readout``, which is also A0's name for the mixed value before the
    output projection. Asking a two-layer model for ``readout`` returned three
    tensors. The trunk site is now ``final_norm``. Prompt 19 addresses sites by
    name across three architectures, so this has to stay true as A1 and A2 add
    their own.
    """
    names = list(model.local_site_names())
    over_counted = {name for name in names if names.count(name) > CONFIG.n_layers}
    assert not over_counted, f"site names shared between the trunk and a mechanism: {over_counted}"


def test_a_captured_tensor_is_a_copy_and_not_a_live_view(model, inputs):
    hooks = CaptureContext(capture=("weights",))
    with torch.no_grad():
        model(inputs, hooks=hooks)
    captured = next(iter(hooks.captures.values()))
    original = captured.clone()
    with torch.no_grad():
        model(inputs * 2.0, hooks=CaptureContext(capture=("weights",)))
    assert torch.equal(captured, original)


def test_a_transform_replaces_the_tensor_and_a_shape_mismatch_is_refused(model, inputs):
    """Interventions are prompt 19's; the substitution contract is established here."""
    with torch.no_grad():
        clean = model(inputs)
        identity = model(inputs, hooks=CaptureContext(transforms={"weights": lambda t: t}))
        zeroed = model(inputs, hooks=CaptureContext(transforms={"output": torch.zeros_like}))
    assert torch.equal(clean.values, identity.values), "an identity transform changed the output"
    assert not torch.equal(clean.values, zeroed.values), "zeroing the mixer changed nothing"

    with pytest.raises(HookSiteError):
        model(inputs, hooks=CaptureContext(transforms={"weights": lambda t: t[..., :1]}))
    with pytest.raises(HookSiteError):
        model(inputs, hooks=CaptureContext(transforms={"weights": lambda t: t.to(torch.float64)}))


# --------------------------------------------------------------------------- #
# Parameter accounting
# --------------------------------------------------------------------------- #


def hand_parameter_count(config: ModelConfig) -> int:
    """The parameter count, derived here from the architecture description.

    Written independently of ``models/common.py`` on purpose: a formula that
    imported the model's own accounting would agree with it by construction and
    would prove nothing. Per layer, with ``d = d_model`` and ``m = mlp_ratio``:

    ``qkv``     ``3d^2 + 3d``     ``out_proj``  ``d^2 + d``
    ``mlp``     ``m d^2 + m d`` up, ``m d^2 + d`` down
    ``norms``   ``2 * 2d``

    and outside the layers: encoder ``dF + d``, positions ``Td``, final norm
    ``2d``, and two heads of ``Fd + F`` each.
    """
    d, f, t, m = config.d_model, config.n_features, config.seq_len, config.mlp_ratio
    per_layer = (
        (3 * d * d + 3 * d)  # qkv
        + (d * d + d)  # out_proj
        + (m * d * d + m * d)  # mlp up
        + (m * d * d + d)  # mlp down
        + 4 * d  # two LayerNorms
    )
    return (
        (d * f + d)  # encoder
        + (t * d)  # learned positions
        + config.n_layers * per_layer
        + 2 * d  # final LayerNorm
        + 2 * (f * d + f)  # value and activity heads
    )


def test_parameter_count_matches_a_hand_calculation(model):
    assert count_parameters(model) == hand_parameter_count(CONFIG) == 62_520


@pytest.mark.parametrize(
    "config",
    [
        ModelConfig(n_features=36, seq_len=12, d_model=48, n_layers=2, n_heads=2),
        ModelConfig(n_features=124, seq_len=48, d_model=16, n_layers=1, n_heads=1, mlp_ratio=2),
        ModelConfig(n_features=64, seq_len=16, d_model=32, n_layers=4, n_heads=4, bias=False),
    ],
)
def test_the_hand_calculation_holds_across_shapes(config):
    expected = hand_parameter_count(config)
    if not config.bias:
        # Every bias term in the formula above, removed.
        d, f, m = config.d_model, config.n_features, config.mlp_ratio
        expected -= (
            d  # encoder
            + config.n_layers * (3 * d + d + m * d + d)
            + 2 * f  # heads
        )
    assert count_parameters(FeatureModel(config)) == expected


def test_the_parameter_report_accounts_for_every_parameter(model):
    report = parameter_report(model)
    assert report["total"] == count_parameters(model)
    assert report["trainable"] == report["total"]
    components = {k: v for k, v in report.items() if k not in ("total", "trainable")}
    assert sum(components.values()) == report["total"]
    assert report["mixing"] == sum(
        count_parameters(block.mix) for block in model.blocks
    )


def test_the_parameter_matched_control_lands_near_its_target():
    """§5 A0's narrower/wider control, for when a candidate adds capacity."""
    target = int(count_parameters(FeatureModel(CONFIG)) * 1.4)
    matched, report = parameter_matched_variant(CONFIG, target)
    assert matched.d_model % CONFIG.n_heads == 0
    assert abs(report["relative_error"]) < 0.10
    assert count_parameters(FeatureModel(matched)) == report["matched_parameters"]
    assert report["narrower"]["parameters"] <= target <= report["wider"]["parameters"]
    assert matched.d_model != CONFIG.d_model, "a 40% larger budget did not change the width"
