"""A0 — standard causal softmax attention, and its slow reference.

A0 is the primary reference of §5: the known-strong associative-recall control
and the basis for every matched intervention in the program. Every later claim
is stated relative to it, so this file is deliberately the least clever one in
the laboratory.

Two implementations of the same equation live here on purpose.
:meth:`SoftmaxAttention.forward` is the batched path every run uses.
:meth:`SoftmaxAttention.reference_forward` is the same computation written as
nested loops over batch, head, query, key, and channel, with no reshapes, no
broadcasting, and no fused kernels — the textbook definition, correct by
inspection. ``tests/equations/test_softmax_reference.py`` holds them to each
other. Without that test, every architecture comparison in this program rests
on the assumption that the baseline computes what its name says.

The attention weights are materialised rather than delegated to
``scaled_dot_product_attention``. At this scale — ``T`` of 12 to 128 and ``d``
of 16 to 64 — the fused kernel buys nothing measurable, and the ``(B, H, T, T)``
matrix it hides is exactly the §6.3 mechanism-activity evidence and the §6.4
intervention surface. A baseline whose mechanism cannot be observed is not a
useful baseline. The fused path is kept as an independent third opinion in the
equation test.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn

from architecture_mechanics.instrumentation.hooks import NO_HOOKS, HookContext
from architecture_mechanics.models.common import (
    FeatureModel,
    MixingPrimitive,
    ModelConfig,
    parameter_matched_config,
    register_primitive,
)

__all__ = [
    "A0_VARIANTS",
    "SoftmaxAttention",
    "build_softmax_model",
    "parameter_matched_variant",
]

A0_VARIANTS: tuple[str, ...] = ("ordinary_residual", "parameter_matched")
"""The two variants §5 A0 requires.

``ordinary_residual`` is the reference: ``h <- h + Attention(LN(h))``, which is
:class:`~architecture_mechanics.models.common.ModelConfig` with
``residual_write="ordinary"``. ``parameter_matched`` is the narrower/wider
control built by :func:`parameter_matched_variant`, used when a candidate
architecture adds capacity and a width-matched comparison would therefore credit
it for parameters as well as for mechanism.
"""


@register_primitive
class SoftmaxAttention(MixingPrimitive):
    """Multi-head causal self-attention over the residual stream.

    Hook sites, in the order the forward pass declares them:

    ``input``    the post-norm residual stream entering the mechanism;
    ``q``, ``k``, ``v``  head-split projections, ``(B, H, T, d_head)``;
    ``scores``   pre-softmax logits with the causal mask already applied;
    ``weights``  the attention distribution, ``(B, H, T, T)``;
    ``readout``  the mixed values before heads are merged;
    ``output``   the mechanism's contribution to the residual stream.

    ``q``, ``k``, ``v``, ``weights``, ``readout``, and ``output`` are the names
    A1 and A2 reuse, so prompt 19 reaches all three architectures through one
    call. ``scores`` is A0-specific: there is no pre-normalisation logit in a
    linear-attention or delta-rule mechanism.
    """

    kind = "softmax"
    SITES = ("input", "q", "k", "v", "scores", "weights", "readout", "output")

    def __init__(self, config: ModelConfig, layer_index: int) -> None:
        super().__init__(config, layer_index)
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.scale = 1.0 / math.sqrt(self.d_head)

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.out_proj.weight._residual_projection = True

        mask = torch.ones(config.seq_len, config.seq_len, dtype=torch.bool).tril().logical_not()
        # Not persistent: it is a constant derived from seq_len, and a mask in
        # the state dict is a mask that can disagree with the config it is
        # loaded into.
        self.register_buffer("causal_block", mask[None, None], persistent=False)

    # -- fast path --------------------------------------------------------- #

    def forward(self, x: torch.Tensor, *, hooks: HookContext = NO_HOOKS) -> torch.Tensor:
        x = hooks.site("input", x)
        batch, seq_len, _ = x.shape

        projected = self.qkv(x)
        query, key, value = projected.split(self.config.d_model, dim=-1)
        query = hooks.site("q", self._split_heads(query, batch, seq_len))
        key = hooks.site("k", self._split_heads(key, batch, seq_len))
        value = hooks.site("v", self._split_heads(value, batch, seq_len))

        scores = (query @ key.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(self.causal_block[:, :, :seq_len, :seq_len], float("-inf"))
        scores = hooks.site("scores", scores)

        weights = torch.softmax(scores, dim=-1)
        weights = hooks.site("weights", weights)

        readout = hooks.site("readout", weights @ value)
        merged = readout.transpose(1, 2).reshape(batch, seq_len, self.config.d_model)
        return hooks.site("output", self.out_proj(merged))

    def _split_heads(self, tensor: torch.Tensor, batch: int, seq_len: int) -> torch.Tensor:
        return tensor.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)

    # -- slow reference ---------------------------------------------------- #

    def reference_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Explicit loops over batch, head, query, key, and channel.

        No reshape, no broadcast, no fused kernel, and no max-subtraction in the
        softmax — ``exp(s_j) / sum_k exp(s_k)`` is written as the definition. The
        fast path's ``torch.softmax`` subtracts the row maximum, which is
        algebraically the same and numerically better; the equation test runs
        both in float64 at moderate magnitude so that the difference between
        them is far below the tolerance and cannot hide a real disagreement.
        """
        batch, seq_len, d_model = x.shape
        heads, d_head = self.n_heads, self.d_head
        w_qkv, b_qkv = self.qkv.weight, self.qkv.bias
        w_out, b_out = self.out_proj.weight, self.out_proj.bias
        scale = 1.0 / math.sqrt(d_head)

        output = torch.zeros_like(x)
        for b in range(batch):
            query = torch.zeros(seq_len, d_model, dtype=x.dtype, device=x.device)
            key = torch.zeros_like(query)
            value = torch.zeros_like(query)
            for t in range(seq_len):
                projected = w_qkv @ x[b, t]
                if b_qkv is not None:
                    projected = projected + b_qkv
                query[t] = projected[:d_model]
                key[t] = projected[d_model : 2 * d_model]
                value[t] = projected[2 * d_model :]

            mixed = torch.zeros(seq_len, d_model, dtype=x.dtype, device=x.device)
            for h in range(heads):
                lo, hi = h * d_head, (h + 1) * d_head
                for i in range(seq_len):
                    exponentials = []
                    for j in range(i + 1):  # causal: keys after the query do not exist
                        dot = torch.zeros((), dtype=x.dtype, device=x.device)
                        for c in range(lo, hi):
                            dot = dot + query[i, c] * key[j, c]
                        exponentials.append(torch.exp(dot * scale))
                    total = torch.zeros((), dtype=x.dtype, device=x.device)
                    for term in exponentials:
                        total = total + term
                    for c in range(lo, hi):
                        accumulated = torch.zeros((), dtype=x.dtype, device=x.device)
                        for j in range(i + 1):
                            accumulated = accumulated + (exponentials[j] / total) * value[j, c]
                        mixed[i, c] = accumulated

            for t in range(seq_len):
                projected = w_out @ mixed[t]
                if b_out is not None:
                    projected = projected + b_out
                output[b, t] = projected
        return output

    # -- §6.3 mechanism activity ------------------------------------------- #

    def mechanism_activity(self, captures: Mapping[str, torch.Tensor]) -> dict[str, float]:
        """Was the attention actually used, and for what?

        Four numbers, each with a named degenerate value:

        ``entropy_nats``       mean row entropy of the attention distribution.
        ``entropy_ratio``      that entropy divided by the entropy of a uniform
                               distribution over the same causal window. ``1.0``
                               means the mechanism selects nothing — it is a
                               running average, and the model is effectively a
                               position-wise MLP over a prefix mean.
        ``self_mass``          mean weight on the query's own position. ``1.0``
                               means no transport happens at all.
        ``off_diagonal_mass``  ``1 - self_mass``: the fraction of the read that
                               comes from somewhere else. This is the one that
                               answers "did the sequence mixer mix".
        ``max_weight``         mean largest single weight; a sharp retrieval is
                               near ``1.0`` and a diffuse one near ``1/(t+1)``.

        Row ``t = 0`` is excluded from the entropy statistics because its causal
        window holds one key, so its entropy is zero by arithmetic rather than
        by anything the mechanism learned.
        """
        weights = captures.get("weights")
        if weights is None:
            return {}
        weights = weights.detach().to(torch.float64)
        batch, heads, seq_len, _ = weights.shape
        positions = torch.arange(seq_len, device=weights.device)

        safe = weights.clamp_min(1e-30)
        entropy = -(weights * safe.log()).sum(dim=-1)  # (B, H, T)
        uniform = torch.log((positions + 1).to(torch.float64))  # entropy of a flat causal window
        diagonal = weights[..., positions, positions]
        maximum = weights.max(dim=-1).values

        informative = positions >= 1
        ratio = entropy[..., informative] / uniform[informative]
        return {
            "entropy_nats": float(entropy[..., informative].mean()),
            "entropy_ratio": float(ratio.mean()),
            "self_mass": float(diagonal.mean()),
            "off_diagonal_mass": float(1.0 - diagonal.mean()),
            "max_weight": float(maximum.mean()),
            "n_rows": float(batch * heads * seq_len),
        }


# --------------------------------------------------------------------------- #
# The two §5 A0 variants
# --------------------------------------------------------------------------- #


def build_softmax_model(config: ModelConfig) -> FeatureModel:
    """A0 with an ordinary residual write — the reference variant."""
    if config.arch != "softmax":
        raise ValueError(f"build_softmax_model got arch={config.arch!r}")
    if config.residual_write != "ordinary":
        raise ValueError(
            f"the A0 reference variant requires residual_write='ordinary', "
            f"got {config.residual_write!r}"
        )
    return FeatureModel(config)


def parameter_matched_variant(
    config: ModelConfig, target_parameters: int
) -> tuple[ModelConfig, dict]:
    """A0 retuned in width to a candidate's parameter budget.

    §5 A0's second required variant. When a candidate adds capacity, the
    width-matched comparison answers "same ``d``, different mechanism" and this
    one answers "same parameter count, different mechanism". §7.2 asks for both
    because either alone is arguable: the first credits the candidate for extra
    parameters, the second changes the candidate's bottleneck width, and a claim
    that survives only one of the two is not a claim about mechanism.
    """
    if config.arch != "softmax":
        raise ValueError(f"parameter_matched_variant got arch={config.arch!r}")
    return parameter_matched_config(config, target_parameters)
