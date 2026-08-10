"""R0 for A1: the §8.5 invariants, the same set prompt 04 proved for A0.

Shapes, causal masking by perturbation, determinism, finite gradients under
large inputs, hook no-op equivalence, and parameter accounting against a hand
calculation. The reference-equation comparison lives in
``tests/equations/test_linear_reference.py``; everything else is here.

Three checks exist here that A0's file has no need of.

**Every declared site, zeroed, changes the output.** A hook site is a promise
that a transform will be honoured. A1 computes a state trajectory that a
different implementation of the same mathematics would not compute at all, so
"is this tensor actually consumed" is a live question here in a way it is not for
attention weights. §13.2 is the failure this rules out: a name that promises
enforcement nothing provides.

**The recurrent form declares what it reaches, and its state intervention
propagates.** Prompt 19's causal evidence depends on corrupting a memory at step
``t`` and watching later steps follow. That is a property of the recursion, not
of the equation, and the parallel form does not have it — which is asserted here
too, so the difference is recorded rather than discovered.

**A1 and A0 have identical parameter counts at every width.** Prompt 12 needs a
width-matched and a parameter-matched comparison; for this architecture pair they
coincide, and that is a measurement rather than an argument.
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
    parameter_report,
    primitive_names,
)
from architecture_mechanics.models.linear import (
    A1_VARIANTS,
    FEATURE_MAP,
    LinearAttention,
    build_linear_model,
    parameter_matched_variant,
)

CONFIG = ModelConfig(n_features=36, seq_len=12, d_model=48, n_layers=2, n_heads=2, mlp_ratio=4,
                     arch="linear")
"""Deliberately A0's R0 config with one field changed, so that every number in
this file is directly comparable with ``tests/identity/test_a0_model.py``."""


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


def test_linear_is_registered_and_buildable():
    assert "linear" in primitive_names()
    assert isinstance(build_linear_model(CONFIG).blocks[0].mix, LinearAttention)
    with pytest.raises(ValueError):
        build_linear_model(ModelConfig(n_features=4, seq_len=4, arch="softmax"))


def test_a1_declares_both_variants_and_names_its_feature_map():
    assert A1_VARIANTS == ("ordinary_residual", "parameter_matched")
    assert CONFIG.residual_write == "ordinary"
    assert FEATURE_MAP == "elu_plus_one"


def test_the_state_carries_its_own_normalizer(model):
    """The one non-standard design decision, asserted rather than described.

    The augmented state is ``d_head x (d_head + 1)``: the extra column is the
    running key sum that normalizes the read. It exists so that an intervention
    removing a write removes it from the numerator and the denominator together.
    """
    block = model.blocks[0].mix
    hooks = CaptureContext(capture=("state_post", "v"))
    with torch.no_grad():
        model(torch.rand(2, CONFIG.seq_len, CONFIG.n_features), hooks=hooks)
    state = hooks.captures["layers.0.mix.state_post"]
    assert state.shape[-2:] == (block.d_head, block.d_head + 1)
    # The normalizer column of S_t is the running sum of phi(k), so it is
    # strictly positive and non-decreasing along the sequence.
    normalizer_column = state[..., -1]
    assert normalizer_column.min().item() > 0.0
    assert (normalizer_column.diff(dim=2) >= 0).all()


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


def test_the_state_at_t_contains_no_write_from_after_t(model, inputs):
    """Causality where A1 keeps it: in the accumulated memory, not in a mask.

    A0 gets causality from a triangular mask that a test can inspect. A1 gets it
    from the direction of a prefix sum, so the check is that perturbing position
    ``t`` leaves every earlier state bitwise unmoved.
    """
    hooks_before = CaptureContext(capture=("state_post",))
    hooks_after = CaptureContext(capture=("state_post",))
    perturbed = inputs.clone()
    perturbed[:, 8] = perturbed[:, 8] + 5.0
    with torch.no_grad():
        model(inputs, hooks=hooks_before)
        model(perturbed, hooks=hooks_after)
    for name, state in hooks_before.captures.items():
        later = hooks_after.captures[name]
        assert torch.equal(state[:, :, :8], later[:, :, :8]), f"{name} leaked the future"
    assert not torch.equal(
        hooks_before.captures["layers.0.mix.state_post"][:, :, 8],
        hooks_after.captures["layers.0.mix.state_post"][:, :, 8],
    ), "the perturbed position's own state did not move"


def test_the_induced_attention_places_no_weight_on_the_future(model, inputs):
    hooks = CaptureContext(capture=model.activity_sites())
    with torch.no_grad():
        model(inputs, hooks=hooks)
    matrices = model.attention_matrices(dict(hooks.captures))
    assert len(matrices) == CONFIG.n_layers
    for name, weights in matrices.items():
        upper = torch.triu(weights, diagonal=1)
        assert upper.abs().max().item() == 0.0, f"{name} attends to the future"
        rows = weights.sum(dim=-1)
        assert torch.allclose(rows, torch.ones_like(rows), atol=1e-9), f"{name} rows are not distributions"


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


def test_forward_is_bitwise_deterministic(model, inputs):
    with torch.no_grad():
        first = model(inputs)
        second = model(inputs)
    assert torch.equal(first.values, second.values)
    assert torch.equal(first.active_logits, second.active_logits)


def test_the_state_update_is_deterministic(model, inputs):
    """§7.3 R0 asks for deterministic *state updates* by name, so the state is
    compared and not only the output it produces."""
    first = CaptureContext(capture=("state_pre", "write", "state_post"))
    second = CaptureContext(capture=("state_pre", "write", "state_post"))
    with torch.no_grad():
        model(inputs, hooks=first)
        model(inputs, hooks=second)
    assert set(first.captures) == set(second.captures)
    for name, tensor in first.captures.items():
        assert torch.equal(tensor, second.captures[name]), f"{name} is not deterministic"


def test_the_same_seed_builds_the_same_weights():
    torch.manual_seed(20260809)
    left = FeatureModel(CONFIG)
    torch.manual_seed(20260809)
    right = FeatureModel(CONFIG)
    for (name, a), (_, b) in zip(left.named_parameters(), right.named_parameters()):
        assert torch.equal(a, b), f"{name} differs between two identically seeded builds"

    torch.manual_seed(20260810)
    other = FeatureModel(CONFIG)
    assert not torch.equal(left.encoder.weight, other.encoder.weight)


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


def test_the_read_is_finite_when_the_feature_map_is_driven_to_underflow():
    """``phi(x) = exp(x)`` below zero, so a hard enough push makes the normalizer
    zero in float32 and the read ``0/0``. The clamp is what stops that being a
    NaN, and this is the case it exists for — including at sequence length one,
    where the prefix holds a single key and there is nothing else to normalize
    against.
    """
    torch.manual_seed(3)
    model = FeatureModel(CONFIG)
    with torch.no_grad():
        model.blocks[0].mix.qkv.bias.fill_(-200.0)
    model.eval()
    for seq_len in (1, 2, CONFIG.seq_len):
        hooks = CaptureContext(capture=("readout", "state_post"))
        with torch.no_grad():
            output = model(torch.rand(2, seq_len, CONFIG.n_features), hooks=hooks)
        assert torch.isfinite(output.values).all(), f"non-finite output at T={seq_len}"
        for name, tensor in hooks.captures.items():
            assert torch.isfinite(tensor).all(), f"{name} is not finite at T={seq_len}"


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
    for name in LinearAttention.SITES:
        hooks = CaptureContext(capture=(name,))
        with torch.no_grad():
            model(inputs, hooks=hooks)
        assert len(hooks.captures) == CONFIG.n_layers, f"site {name!r} not captured in every layer"


def test_no_two_sites_share_a_bare_name(model):
    names = list(model.local_site_names())
    over_counted = {name for name in names if names.count(name) > CONFIG.n_layers}
    assert not over_counted, f"site names shared between the trunk and a mechanism: {over_counted}"


def test_a1_and_a0_share_the_site_names_prompt_19_hooks_through():
    """The naming contract, asserted rather than promised.

    Prompt 19 hooks both architectures through one interface. A name that means
    the same tensor in both must be spelled the same in both, and A2's state
    terms must be able to join the list without renaming A1's.
    """
    from architecture_mechanics.models.softmax import SoftmaxAttention

    shared = {"input", "q", "k", "v", "readout", "output"}
    assert shared <= set(SoftmaxAttention.SITES)
    assert shared <= set(LinearAttention.SITES)
    # §5 A2's state terms, in the spelling A1 fixes for them.
    assert {"state_pre", "write", "state_post"} <= set(LinearAttention.SITES)
    # A0 materialises a weight matrix; A1 does not, and must not pretend to.
    assert "weights" not in LinearAttention.SITES
    assert "scores" not in LinearAttention.SITES


@pytest.mark.parametrize("site", LinearAttention.SITES)
def test_zeroing_any_declared_site_changes_the_output(model, inputs, site):
    """No site is decorative.

    A hook site is a promise that a transform will be honoured. This is the only
    check that the promise is kept for all eleven of them, and it is the reason
    the induced attention matrix is *not* a site: nothing consumes it, so it
    could not pass this test, so it is a measurement instead.
    """
    with torch.no_grad():
        clean = model(inputs)
        zeroed = model(inputs, hooks=CaptureContext(transforms={site: torch.zeros_like}))
    assert not torch.equal(clean.values, zeroed.values), f"zeroing {site!r} changed nothing"


def test_a_captured_tensor_is_a_copy_and_not_a_live_view(model, inputs):
    hooks = CaptureContext(capture=("state_post",))
    with torch.no_grad():
        model(inputs, hooks=hooks)
    captured = next(iter(hooks.captures.values()))
    original = captured.clone()
    with torch.no_grad():
        model(inputs * 2.0, hooks=CaptureContext(capture=("state_post",)))
    assert torch.equal(captured, original)


def test_a_transform_replaces_the_tensor_and_a_shape_mismatch_is_refused(model, inputs):
    with torch.no_grad():
        clean = model(inputs)
        identity = model(inputs, hooks=CaptureContext(transforms={"state_post": lambda t: t}))
    assert torch.equal(clean.values, identity.values), "an identity transform changed the output"

    with pytest.raises(HookSiteError):
        model(inputs, hooks=CaptureContext(transforms={"state_post": lambda t: t[..., :1]}))
    with pytest.raises(HookSiteError):
        model(inputs, hooks=CaptureContext(transforms={"write": lambda t: t.to(torch.float64)}))


# --------------------------------------------------------------------------- #
# The recurrent form, which prompt 19 hooks
# --------------------------------------------------------------------------- #


def test_the_recurrent_form_declares_exactly_the_sites_it_reaches(model, inputs):
    block = model.blocks[0].mix
    hidden = torch.randn(2, CONFIG.seq_len, CONFIG.d_model)
    hooks = capture_all()
    with torch.no_grad():
        block.recurrent_forward(hidden, hooks=hooks)
    assert list(hooks.visited) == list(block.recurrent_hook_sites(CONFIG.seq_len))


def test_a_state_intervention_propagates_in_the_recurrent_form(model):
    """The property the recursion exists for.

    Corrupting the memory at step 4 must change what steps 5 onward read, and
    must leave steps 0 to 3 bitwise alone. Prompt 19's state-term interventions
    are built on exactly this.
    """
    block = model.blocks[0].mix
    hidden = torch.randn(2, CONFIG.seq_len, CONFIG.d_model,
                         generator=torch.Generator().manual_seed(77))
    with torch.no_grad():
        clean = block.recurrent_forward(hidden)
        corrupted = block.recurrent_forward(
            hidden, hooks=CaptureContext(transforms={"step.4.state_post": torch.zeros_like})
        )
    assert torch.equal(clean[:, :4], corrupted[:, :4]), "an intervention at step 4 moved step 3"
    for t in range(4, CONFIG.seq_len):
        assert not torch.equal(clean[:, t], corrupted[:, t]), f"step {t} ignored the corruption"


def test_the_parallel_form_does_not_propagate_a_state_intervention(model):
    """The same intervention on the scan is local, and that is recorded here.

    Not a defect: replacing ``S_4`` in the parallel form answers "what does
    position 4 read from a corrupted memory", which is a different and also
    legitimate question. It is asserted so that prompt 19 chooses the form on
    purpose rather than discovering the difference from a null result.
    """
    block = model.blocks[0].mix
    hidden = torch.randn(2, CONFIG.seq_len, CONFIG.d_model,
                         generator=torch.Generator().manual_seed(77))
    zero_one_step = {
        "state_post": lambda t: torch.cat(
            (t[:, :, :4], torch.zeros_like(t[:, :, 4:5]), t[:, :, 5:]), dim=2
        )
    }
    with torch.no_grad():
        clean = block(hidden)
        corrupted = block(hidden, hooks=CaptureContext(transforms=zero_one_step))
    assert not torch.equal(clean[:, 4], corrupted[:, 4])
    assert torch.equal(clean[:, 5:], corrupted[:, 5:]), "the scan propagated, which it cannot"


# --------------------------------------------------------------------------- #
# §6.3 activity
# --------------------------------------------------------------------------- #


def test_mechanism_activity_reports_the_state_and_the_distribution(model, inputs):
    hooks = CaptureContext(capture=model.activity_sites())
    with torch.no_grad():
        model(inputs, hooks=hooks)
    report = model.mechanism_activity(dict(hooks.captures))

    for layer in range(CONFIG.n_layers):
        for name in ("entropy_ratio", "off_diagonal_mass", "self_mass", "max_weight",
                     "state_norm", "write_norm", "write_to_state_ratio",
                     "state_growth_ratio", "normalizer_mean", "readout_magnitude"):
            assert f"layers.{layer}.{name}" in report, f"missing layers.{layer}.{name}"
    assert report["layers.0.state_norm"] > 0.0
    assert report["layers.0.state_growth_ratio"] > 1.0, "the state did not accumulate"
    assert 0.0 < report["layers.0.write_to_state_ratio"] <= 1.0


def test_every_activity_measure_reaches_its_named_degenerate_value(model):
    """A measure whose degenerate value is unreachable is decoration.

    Fed directly rather than provoked through the model, because
    :class:`CaptureContext` records a site's tensor *before* applying a
    transform to it — so intervening on ``write`` and then reading ``write``
    back returns what the mechanism produced, which is the right contract and
    the wrong experiment for this question.
    """
    block = model.blocks[0].mix
    batch, heads, seq_len, width = 2, CONFIG.n_heads, CONFIG.seq_len, block.d_head

    inert = torch.zeros(batch, heads, seq_len, width, width + 1)
    report = block.mechanism_activity(
        {"write": inert, "state_post": inert, "readout": torch.zeros(batch, heads, seq_len, width)}
    )
    assert report["write_norm"] == 0.0, "a mechanism that writes nothing reported a write"
    assert report["state_norm"] == 0.0
    assert report["readout_magnitude"] == 0.0

    # A flat feature map gives every key the same affinity to every query, which
    # is the running-prefix-average degeneracy §6.3's entropy ratio exists to
    # catch: uniform over the causal window, so the ratio is exactly one.
    flat = torch.ones(batch, heads, seq_len, width)
    uniform = block.mechanism_activity({"phi_q": flat, "phi_k": flat})
    assert uniform["entropy_ratio"] == pytest.approx(1.0, abs=1e-12)
    assert uniform["off_diagonal_mass"] == pytest.approx(
        1.0 - sum(1.0 / (t + 1) for t in range(seq_len)) / seq_len, abs=1e-12
    )


# --------------------------------------------------------------------------- #
# Parameter accounting
# --------------------------------------------------------------------------- #


def hand_parameter_count(config: ModelConfig) -> int:
    """Derived here from A1's architecture description, not from the code.

    Per layer, with ``d = d_model`` and ``m = mlp_ratio``:

    ``qkv``     ``3d^2 + 3d``     ``out_proj``  ``d^2 + d``
    ``mlp``     ``m d^2 + m d`` up, ``m d^2 + d`` down
    ``norms``   ``2 * 2d``

    and outside the layers: encoder ``dF + d``, positions ``Td``, final norm
    ``2d``, and two heads of ``Fd + F`` each. The feature map contributes
    nothing: ``elu(x) + 1`` has no parameters, and the augmented state's extra
    column is a constant, not a learned one.
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


@pytest.mark.parametrize("d_model", [16, 24, 32, 48, 64, 96])
def test_a1_and_a0_have_identical_parameter_counts_at_every_width(d_model: int):
    """Reported against A0 as the mission asks, and asserted across widths.

    Both mechanisms hold one fused ``d -> 3d`` projection and one ``d -> d``
    projection and nothing else, so the width-matched and parameter-matched
    comparisons of §7.2 coincide for this architecture pair. Checked at six
    widths rather than argued, because it stops being true the moment a
    mechanism gains a gate — which A2 will.
    """
    from dataclasses import replace

    linear = ModelConfig(n_features=36, seq_len=12, d_model=d_model, n_layers=2, n_heads=2,
                         arch="linear")
    softmax = replace(linear, arch="softmax")
    assert count_parameters(FeatureModel(linear)) == count_parameters(FeatureModel(softmax))
    assert parameter_report(FeatureModel(linear))["mixing"] == \
        parameter_report(FeatureModel(softmax))["mixing"]


def test_the_parameter_report_accounts_for_every_parameter(model):
    report = parameter_report(model)
    assert report["total"] == count_parameters(model)
    assert report["trainable"] == report["total"]
    components = {k: v for k, v in report.items() if k not in ("total", "trainable")}
    assert sum(components.values()) == report["total"]
    assert report["mixing"] == sum(count_parameters(block.mix) for block in model.blocks)


def test_the_parameter_matched_control_lands_near_its_target():
    """§5's narrower/wider control, built for A1 even though A1 does not need it.

    It costs nothing to have and it is the arm prompt 12 reaches for when the
    candidate is A2 rather than A1.
    """
    target = int(count_parameters(FeatureModel(CONFIG)) * 1.4)
    matched, report = parameter_matched_variant(CONFIG, target)
    assert matched.arch == "linear"
    assert matched.d_model % CONFIG.n_heads == 0
    assert abs(report["relative_error"]) < 0.10
    assert count_parameters(FeatureModel(matched)) == report["matched_parameters"]
    assert report["narrower"]["parameters"] <= target <= report["wider"]["parameters"]
    assert matched.d_model != CONFIG.d_model, "a 40% larger budget did not change the width"


def test_the_operation_summary_reports_constant_state_where_a0_reports_growing(model):
    """§8.3's number that names the architectural difference.

    A0 must keep every key and value it has seen; A1 keeps one matrix per head
    whatever the context length. This is the field a manifest diff between two
    runs reads, so it is asserted rather than left to the docstring.

    The assertion is about *scaling*, not about which number is smaller here. At
    this laboratory's shapes A1's constant state is the **larger** of the two —
    2400 scalars against A0's 2304 at ``T = 12``, ``d_head = 24`` — because
    ``d (d_head + 1)`` beats ``2 T d`` until ``T`` passes ``(d_head + 1) / 2``.
    Asserting "A1 is cheaper" would have been asserting an asymptotic argument at
    a shape where it is false, which is precisely the kind of claim §8.3's
    theoretical counts exist to prevent.
    """
    from dataclasses import replace

    torch.manual_seed(1)
    reference = FeatureModel(replace(CONFIG, arch="softmax"))
    a1 = model.operation_state_summary()
    a0 = reference.operation_state_summary()

    assert a1["mixing"][0]["materialises_pairwise_matrix"] is False
    assert a0["mixing"][0]["materialises_pairwise_matrix"] is True
    assert (a1["recurrent_state_scalars"], a0["recurrent_state_scalars"]) == (2400, 2304)

    # Constant in context: doubling the sequence must not change A1's state size,
    # and must double A0's. That is the property, and it is what makes A1 the
    # cheaper of the two at every T above the crossover.
    def state_at(seq_len: int, arch: str) -> int:
        built = FeatureModel(replace(CONFIG, seq_len=seq_len, arch=arch))
        return built.operation_state_summary()["recurrent_state_scalars"]

    assert state_at(2 * CONFIG.seq_len, "linear") == a1["recurrent_state_scalars"]
    assert state_at(2 * CONFIG.seq_len, "softmax") == 2 * a0["recurrent_state_scalars"]
    crossover = (CONFIG.d_head + 1) / 2
    assert state_at(int(crossover) + 1, "softmax") > a1["recurrent_state_scalars"]
