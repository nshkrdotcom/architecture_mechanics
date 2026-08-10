"""A1 — kernelized linear attention, in a parallel and a recurrent form.

§5 A1's role is precise and narrow: the closest efficient contrast to softmax
attention, testing what removing global softmax normalization does, and
providing **recurrent finite-state behaviour without delta overwrite**. That last
clause is why this file exists before prompt 17's delta rule. Without A1, any
result about A2 is confounded between "carrying a finite recurrent state helps"
and "erasing stale associations helps", and no amount of care in the comparison
between A0 and A2 can separate them. A1 is the middle term.

The mechanism, per head, with ``phi(x) = elu(x) + 1``::

    write_t   = phi(k_t) (x) [v_t, 1]          the rank-one write, augmented
    S_t       = S_{t-1} + write_t              S_{-1} = 0
    r_t       = phi(q_t)^T S_t                 read the whole state
    out_t     = r_t[:d_head] / max(r_t[d_head], eps)

which is algebraically the normalized kernel attention

    out_t = sum_{j<=t} (phi(q_t) . phi(k_j)) v_j / sum_{j<=t} phi(q_t) . phi(k_j)

The **augmented state** is the one design decision here that is not standard, and
it is the one that matters for prompts 17 and 19. The usual presentation carries
two objects: a ``d_head x d_head`` value state and a separate ``d_head``
normalizer. Appending a constant ``1`` to the value makes them one object, so
``write`` is one tensor, ``state`` is one tensor, and an intervention that
removes a write removes it from the numerator *and* the denominator — which is
what "this key was never written" means. With two objects, the coherent
intervention would require hitting two sites, and hitting one would silently
produce a read that is normalized by a history it no longer contains.

**Three implementations, as prompt 04 established for A0.** :meth:`forward` is
the parallel form every run uses: it forms every write term at once and scans
them, so the whole state trajectory is on its critical path.
:meth:`recurrent_forward` is the explicit ``O(T)`` recursion, hooked per step.
:meth:`reference_forward` is the kernel-sum definition in nested loops, correct
by inspection, and it shares no structure with either. The equation tests hold
all three to each other.

**Why the parallel and recurrent forms are not interchangeable for prompt 19.**
They compute the same numbers, and the test proves it. They do not offer the same
intervention surface: replacing ``state_post`` in the parallel form replaces it at
that position only, because the scan has already run, while replacing it in the
recurrent form propagates to every later step. Both are legitimate — "what does
this position read if its memory is corrupted" and "what does the rest of the
sequence do if the memory is corrupted here" are different questions — but they
are different, and prompt 19's state interventions want the second.

**There is no ``weights`` site.** Linear attention induces a row-stochastic
matrix, ``A_tj = phi(q_t).phi(k_j) / z_t``, and §6.3's retrieval measure needs it;
but nothing in the forward pass computes or consumes it, so declaring it as a
hook site would offer an intervention that the mechanism would silently discard.
It is *derived* in :meth:`attention_matrix` from the captured feature maps —
exactly, not approximately — which is a measurement and is named as one. §13.2 is
about exactly this: a name that promises enforcement nothing provides.

**One feature map, not a sweep.** ``elu(x) + 1`` (Katharopoulos et al., 2020) is
strictly positive, needs no parameters, and is the form the linear-attention
literature is written in. Trying several and keeping the best would be §13.1
architecture roulette, and this mission is not about which feature map wins.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from architecture_mechanics.instrumentation.hooks import NO_HOOKS, HookContext
from architecture_mechanics.models.common import (
    FeatureModel,
    MixingPrimitive,
    ModelConfig,
    attention_distribution_statistics,
    parameter_matched_config,
    register_primitive,
)

__all__ = [
    "A1_VARIANTS",
    "FEATURE_MAP",
    "NORMALIZER_FLOOR",
    "LinearAttention",
    "build_linear_model",
    "feature_map",
    "parameter_matched_variant",
]

FEATURE_MAP = "elu_plus_one"
"""The chosen kernel feature map, recorded so the choice is a fact and not a
default. ``phi(x) = elu(x) + 1`` is ``x + 1`` above zero and ``exp(x)`` below it,
hence strictly positive everywhere — which is what makes the induced weights a
distribution rather than a signed score, and therefore what makes §6.3's entropy
and retrieval measures mean the same thing here as they do for A0."""

NORMALIZER_FLOOR = 1e-6
"""Floor under the read normalizer ``z_t = phi(q_t) . sum_{j<=t} phi(k_j)``.

``phi`` is strictly positive, so ``z_t > 0`` holds mathematically and this never
binds on a trained model — measured normalizers on the positive control are of
order ``d_head``. It exists for two cases where the arithmetic can reach zero
anyway: ``exp`` underflowing to zero in float32 when a projection is driven very
negative, which the R0 large-magnitude gradient check deliberately provokes, and
an intervention that zeroes the state, which prompt 19 will deliberately perform.
A clamp rather than an added epsilon, so that wherever it does not bind the
induced weights sum to exactly one."""

A1_VARIANTS: tuple[str, ...] = ("ordinary_residual", "parameter_matched")
"""The same two variants §5 requires of A0, and for the same reason.

``ordinary_residual`` is the reference. ``parameter_matched`` is the
narrower/wider control. It is built here even though A1 turns out to be exactly
parameter-matched to A0 at every width — see
:func:`parameter_matched_variant` — because "the control was unnecessary" is a
measurement, and one that has to be re-checked at every width a comparison uses
rather than assumed from this architecture pair."""


def feature_map(tensor: torch.Tensor) -> torch.Tensor:
    """``phi(x) = elu(x) + 1``. Strictly positive, parameter-free."""
    return F.elu(tensor) + 1.0


@register_primitive
class LinearAttention(MixingPrimitive):
    """Multi-head causal linear attention with an augmented recurrent state.

    Hook sites, in the order the parallel forward pass declares them:

    ``input``      the post-norm residual stream entering the mechanism;
    ``q``, ``k``, ``v``   head-split projections, ``(B, H, T, d_head)``;
    ``phi_q``, ``phi_k``  the feature-mapped query and key, strictly positive;
    ``state_pre``  ``S_{t-1}``, ``(B, H, T, d_head, d_head + 1)``;
    ``write``      ``phi(k_t) (x) [v_t, 1]``, the same shape;
    ``state_post`` ``S_t``, the same shape;
    ``readout``    the normalized read, ``(B, H, T, d_head)``;
    ``output``     the mechanism's contribution to the residual stream.

    ``input``, ``q``, ``k``, ``v``, ``readout`` and ``output`` are A0's names for
    the same tensors, so prompt 19 reaches both architectures through one call.
    ``state_pre``, ``write`` and ``state_post`` are §5 A2's names for the state
    terms, chosen now so that A2 adds ``decay``, ``erase`` and ``write_gate`` to
    a list rather than renaming one. ``phi_q`` and ``phi_k`` are A1-specific, as
    A0's ``scores`` is A0-specific.

    ``state_pre`` is declared before ``write`` even though the parallel form must
    form the write terms first in order to scan them. The order follows §5's
    description of the update rather than this implementation's arithmetic, so
    that A1 and A2 declare their state terms in the same order; what the
    arithmetic costs is that a transform on ``write`` reaches ``state_post`` at
    its own position but not ``state_pre`` at the next one. The recurrent form
    has no such gap.
    """

    kind = "linear"
    SITES = (
        "input", "q", "k", "v", "phi_q", "phi_k",
        "state_pre", "write", "state_post", "readout", "output",
    )
    ACTIVITY_SITES = ("phi_q", "phi_k", "write", "state_post", "readout")

    RECURRENT_STEP_SITES: tuple[str, ...] = ("state_pre", "write", "state_post", "readout")
    """Sites :meth:`recurrent_forward` declares once per position, under the
    scope ``step.<t>``. Everything else it declares is shared with
    :attr:`SITES` and is declared once."""

    def __init__(self, config: ModelConfig, layer_index: int) -> None:
        super().__init__(config, layer_index)
        self.n_heads = config.n_heads
        self.d_head = config.d_head

        # Identical to A0's, down to the fused layout and the residual tagging,
        # so that the two architectures differ in how they mix and in nothing
        # else. §7.2 asks for width and depth to be matched; matching the
        # projection *parameterisation* is what makes the parameter counts equal
        # rather than merely close.
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.out_proj.weight._residual_projection = True

    # -- shared front end -------------------------------------------------- #

    def _split_heads(self, tensor: torch.Tensor, batch: int, seq_len: int) -> torch.Tensor:
        return tensor.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)

    def _project(self, x: torch.Tensor, hooks: HookContext):
        """``input`` through ``phi_k``: the part both fast forms share."""
        x = hooks.site("input", x)
        batch, seq_len, _ = x.shape
        projected = self.qkv(x)
        query, key, value = projected.split(self.config.d_model, dim=-1)
        query = hooks.site("q", self._split_heads(query, batch, seq_len))
        key = hooks.site("k", self._split_heads(key, batch, seq_len))
        value = hooks.site("v", self._split_heads(value, batch, seq_len))
        phi_q = hooks.site("phi_q", feature_map(query))
        phi_k = hooks.site("phi_k", feature_map(key))
        return phi_q, phi_k, value

    def _augment(self, value: torch.Tensor) -> torch.Tensor:
        """``[v, 1]``: the value with the normalizer's own channel appended."""
        ones = value.new_ones(value.shape[:-1] + (1,))
        return torch.cat((value, ones), dim=-1)

    def _read(self, phi_q: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """``phi(q)^T S``, split back into the value read and its normalizer."""
        full = torch.einsum("...k,...kv->...v", phi_q, state)
        numerator, normalizer = full[..., : self.d_head], full[..., self.d_head :]
        return numerator / normalizer.clamp_min(NORMALIZER_FLOOR)

    def _merge(self, readout: torch.Tensor, hooks: HookContext) -> torch.Tensor:
        batch, _, seq_len, _ = readout.shape
        merged = readout.transpose(1, 2).reshape(batch, seq_len, self.config.d_model)
        return hooks.site("output", self.out_proj(merged))

    # -- parallel form: the path every run takes --------------------------- #

    def forward(self, x: torch.Tensor, *, hooks: HookContext = NO_HOOKS) -> torch.Tensor:
        """``(B, T, d) -> (B, T, d)``, causal, with the whole state trajectory.

        The recurrence ``S_t = S_{t-1} + write_t`` is a prefix sum, so it is
        computed as one. ``torch.cumsum`` over the position axis gives every
        ``S_t`` at once; ``state_pre`` is that trajectory *shifted* by one, and
        ``state_post`` is rebuilt as ``state_pre + write`` rather than read off
        the scan so that transforms on either of its two inputs reach it. The
        shift rather than ``cumulative - write``: the two are equal in exact
        arithmetic, and the subtraction cancels catastrophically whenever a
        position's own write nearly balances everything before it.

        Causality is structural: ``state_post[t]`` is a sum over ``j <= t`` and
        the read at ``t`` touches nothing else. There is no mask to get wrong,
        which is why the R0 test checks causality by perturbation and not by
        inspecting one.

        Memory is ``O(B H T d_head^2)`` for the trajectory — linear in ``T``,
        where A0's weight matrix is quadratic. At this laboratory's scale
        (``T <= 128``, ``d_head <= 32``) that is tens of megabytes, and it buys
        the state as a first-class, interventable object rather than an
        implementation detail hidden inside a scan.
        """
        phi_q, phi_k, value = self._project(x, hooks)
        write = phi_k.unsqueeze(-1) * self._augment(value).unsqueeze(-2)

        cumulative = write.cumsum(dim=2)
        shifted = torch.cat((torch.zeros_like(write[:, :, :1]), cumulative[:, :, :-1]), dim=2)
        state_pre = hooks.site("state_pre", shifted)
        write = hooks.site("write", write)
        state_post = hooks.site("state_post", state_pre + write)

        readout = hooks.site("readout", self._read(phi_q, state_post))
        return self._merge(readout, hooks)

    # -- recurrent form: what prompt 19's state interventions hook --------- #

    def recurrent_forward(
        self, x: torch.Tensor, *, hooks: HookContext = NO_HOOKS
    ) -> torch.Tensor:
        """The same mechanism as an explicit ``O(T)`` recursion over positions.

        Batched over examples and heads but not over time: the state is carried
        in a Python loop, and each step's sites are declared under the scope
        ``step.<t>``, so a fully qualified name reads
        ``layers.0.mix.step.7.state_post``.

        The state that enters step ``t + 1`` is the tensor step ``t`` handed
        back *through the hook*, so a transform at ``state_post`` propagates
        forward exactly as a real corruption of the memory would. That is the
        property :meth:`forward` cannot offer and the reason this method is not
        merely a slower duplicate.

        Not used by training runs — it is many small kernel launches where the
        scan is one — but held to :meth:`forward` bit for bit within float
        tolerance by ``tests/equations/test_linear_reference.py``.
        """
        phi_q, phi_k, value = self._project(x, hooks)
        augmented = self._augment(value)
        batch, heads, seq_len, _ = phi_q.shape

        state = phi_q.new_zeros(batch, heads, self.d_head, self.d_head + 1)
        reads = []
        for t in range(seq_len):
            with hooks.scope("step"), hooks.scope(str(t)):
                state_pre = hooks.site("state_pre", state)
                write = hooks.site("write", phi_k[:, :, t].unsqueeze(-1)
                                   * augmented[:, :, t].unsqueeze(-2))
                state = hooks.site("state_post", state_pre + write)
                reads.append(hooks.site("readout", self._read(phi_q[:, :, t], state)))
        readout = torch.stack(reads, dim=2)
        return self._merge(readout, hooks)

    def recurrent_hook_sites(self, seq_len: int) -> tuple[str, ...]:
        """Every local site :meth:`recurrent_forward` declares, in order.

        The recurrent form's site list depends on the sequence length, which is
        why it is a method here rather than the class attribute :attr:`SITES`.
        A test compares this against what an instrumented recurrent pass
        actually visits, for the same reason prompt 04 compared A0's declared
        sites against its visited ones: a site that exists in a docstring and
        not in the code is discovered by prompt 19 at the worst moment.
        """
        sites = ["input", "q", "k", "v", "phi_q", "phi_k"]
        for t in range(seq_len):
            sites.extend(f"step.{t}.{name}" for name in self.RECURRENT_STEP_SITES)
        sites.append("output")
        return tuple(sites)

    # -- §8.3 theoretical cost --------------------------------------------- #

    def operation_state_summary(self) -> dict:
        """What linear attention costs in operations and in carried state.

        Counted against :meth:`forward`, the path runs actually take. The two
        mechanism terms are the state update (one rank-one outer product per
        position, accumulated) and the read (one state-vector product per
        position); each is ``T H d_head (d_head + 1)`` multiply-accumulates,
        which is ``T d (d_head + 1)``.

        ``recurrent_state_scalars`` is the whole point of the architecture:
        ``d (d_head + 1)``, **constant in context**, against A0's ``2 T d`` that
        grows with every position. A manifest diff between an A0 run and an A1
        run at the same width is where that difference becomes a recorded fact
        rather than a claim in a docstring.

        Worth reading beside A0's number rather than instead of it: at this
        laboratory's shapes A1 is not automatically cheaper. The mechanism terms
        scale as ``T d d_head`` where A0's scale as ``T^2 d``, so A0 is the
        smaller count whenever ``T < d_head`` — which on the positive control
        (``T = 12``, ``d_head = 24``) it is. A1's asymptotic advantage is real
        and this laboratory is deliberately below the crossover, because the
        question here is what the mechanism does, not what it costs.
        """
        seq_len, width = self.config.seq_len, self.config.d_model
        qkv = 3 * seq_len * width * width
        state_update = seq_len * width * (self.d_head + 1)
        read = seq_len * width * (self.d_head + 1)
        projection = seq_len * width * width
        return {
            "mechanism": "linear_attention",
            "feature_map": FEATURE_MAP,
            "ops_per_sequence": "O(T d d_head + T d^2) — linear in T",
            "state_growth": (
                "O(d d_head) — constant in context; one d_head x (d_head + 1) "
                "matrix per head, carrying its own normalizer"
            ),
            "multiply_accumulates_per_sequence": int(qkv + state_update + read + projection),
            "recurrent_state_scalars": int(width * (self.d_head + 1)),
            "materialises_pairwise_matrix": False,
            "breakdown": {
                "qkv_projection": int(qkv),
                "state_update": int(state_update),
                "state_read": int(read),
                "output_projection": int(projection),
            },
        }

    # -- slow reference ---------------------------------------------------- #

    def reference_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Explicit loops over batch, head, query, key, and channel.

        Written as the *kernel-sum* definition rather than as a recursion:

            out_i = sum_{j<=i} (phi(q_i) . phi(k_j)) v_j
                    / sum_{j<=i} phi(q_i) . phi(k_j)

        No reshape, no broadcast, no cumulative sum, and no state matrix
        anywhere. That is deliberate: both fast forms accumulate a
        ``d_head x (d_head + 1)`` state, so a reference that also accumulated one
        would share their structure and could share their bug. This one shares
        only the equation.
        """
        batch, seq_len, d_model = x.shape
        heads, d_head = self.n_heads, self.d_head
        w_qkv, b_qkv = self.qkv.weight, self.qkv.bias
        w_out, b_out = self.out_proj.weight, self.out_proj.bias
        floor = torch.tensor(NORMALIZER_FLOOR, dtype=x.dtype, device=x.device)

        output = torch.zeros_like(x)
        for b in range(batch):
            query = torch.zeros(seq_len, d_model, dtype=x.dtype, device=x.device)
            key = torch.zeros_like(query)
            value = torch.zeros_like(query)
            for t in range(seq_len):
                projected = w_qkv @ x[b, t]
                if b_qkv is not None:
                    projected = projected + b_qkv
                query[t] = feature_map(projected[:d_model])
                key[t] = feature_map(projected[d_model : 2 * d_model])
                value[t] = projected[2 * d_model :]

            mixed = torch.zeros(seq_len, d_model, dtype=x.dtype, device=x.device)
            for h in range(heads):
                lo, hi = h * d_head, (h + 1) * d_head
                for i in range(seq_len):
                    affinities = []
                    for j in range(i + 1):  # causal: keys after the query do not exist
                        dot = torch.zeros((), dtype=x.dtype, device=x.device)
                        for c in range(lo, hi):
                            dot = dot + query[i, c] * key[j, c]
                        affinities.append(dot)
                    total = torch.zeros((), dtype=x.dtype, device=x.device)
                    for term in affinities:
                        total = total + term
                    total = torch.maximum(total, floor)
                    for c in range(lo, hi):
                        accumulated = torch.zeros((), dtype=x.dtype, device=x.device)
                        for j in range(i + 1):
                            accumulated = accumulated + affinities[j] * value[j, c]
                        mixed[i, c] = accumulated / total

            for t in range(seq_len):
                projected = w_out @ mixed[t]
                if b_out is not None:
                    projected = projected + b_out
                output[b, t] = projected
        return output

    # -- the induced attention matrix, as a measurement --------------------- #

    def attention_matrix(self, captures: Mapping[str, torch.Tensor]) -> torch.Tensor | None:
        """``A_tj = phi(q_t) . phi(k_j) / z_t`` for ``j <= t``, zero above.

        Exact, not an approximation: the read this mechanism performs *is*
        ``sum_j A_tj v_j``, and ``sum_j A_tj = 1`` wherever the normalizer floor
        does not bind, because ``z_t`` is by definition that same sum of
        affinities. So §6.3's retrieval lift and entropy mean here precisely what
        they mean for A0.

        Derived rather than hooked. Nothing in :meth:`forward` builds this
        matrix — that is the architecture's entire efficiency argument — so it
        cannot be a site, and offering one would be offering an intervention the
        forward pass would ignore.
        """
        phi_q, phi_k = captures.get("phi_q"), captures.get("phi_k")
        if phi_q is None or phi_k is None:
            return None
        phi_q = phi_q.detach().to(torch.float64)
        phi_k = phi_k.detach().to(torch.float64)
        seq_len = phi_q.shape[2]
        affinity = phi_q @ phi_k.transpose(-2, -1)
        causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=affinity.device).tril()
        affinity = affinity * causal
        return affinity / affinity.sum(dim=-1, keepdim=True).clamp_min(NORMALIZER_FLOOR)

    # -- §6.3 mechanism activity ------------------------------------------- #

    def mechanism_activity(self, captures: Mapping[str, torch.Tensor]) -> dict[str, float]:
        """Was the recurrent state actually used, and for what?

        Two groups. The first is the five distribution statistics of the induced
        attention matrix, computed by the code A0 uses, so ``entropy_ratio`` and
        ``off_diagonal_mass`` are directly comparable across the two
        architectures and §6.3's three activity gates apply unchanged.

        The second is the state itself, which A0 does not have. Each has a named
        degenerate value:

        ``state_norm``            mean Frobenius norm of the value block of
                                  ``S_t``. Zero is a mechanism that writes
                                  nothing.
        ``write_norm``            mean norm of the value block of the rank-one
                                  write. Zero is the same failure seen at the
                                  source.
        ``write_to_state_ratio``  ``||write_t|| / ||S_t||``, positions ``t >= 1``.
                                  §6.3's "write norm relative to state norm". Near
                                  ``1`` means each position's read is dominated by
                                  what that position just wrote — a finite state
                                  used as a register and not as a memory. Near
                                  ``0`` means the state is saturated by history and
                                  new writes cannot be heard.
        ``state_growth_ratio``    ``||S_{T-1}|| / ||S_0||``. A state that does not
                                  grow across the sequence is not accumulating.
        ``normalizer_mean``       mean ``z_t``. Its floor is the mechanism's own
                                  scale, not a threshold: it tells a reader
                                  whether :data:`NORMALIZER_FLOOR` was anywhere
                                  near binding.
        ``readout_magnitude``     mean ``||out_t||``. Zero is a mechanism whose
                                  contribution to the residual stream is nothing,
                                  whatever its state does.

        Position ``t = 0`` is excluded from ``write_to_state_ratio`` because
        ``S_0 = write_0`` makes it exactly one by arithmetic, as row ``0`` is
        excluded from A0's entropy for the same kind of reason.
        """
        report: dict[str, float] = {}

        matrix = self.attention_matrix(captures)
        if matrix is not None:
            report.update(attention_distribution_statistics(matrix))
            phi_q, phi_k = captures["phi_q"], captures["phi_k"]
            key_sum = phi_k.detach().to(torch.float64).cumsum(dim=2)
            normalizer = (phi_q.detach().to(torch.float64) * key_sum).sum(dim=-1)
            report["normalizer_mean"] = float(normalizer.mean())

        state = captures.get("state_post")
        write = captures.get("write")
        if state is not None:
            values = state.detach().to(torch.float64)[..., : self.d_head]
            norms = values.flatten(start_dim=3).norm(dim=-1)  # (B, H, T)
            report["state_norm"] = float(norms.mean())
            report["state_growth_ratio"] = float(
                (norms[:, :, -1] / norms[:, :, 0].clamp_min(1e-30)).mean()
            )
            if write is not None:
                write_values = write.detach().to(torch.float64)[..., : self.d_head]
                write_norms = write_values.flatten(start_dim=3).norm(dim=-1)
                report["write_norm"] = float(write_norms.mean())
                if norms.shape[2] > 1:
                    ratio = write_norms[:, :, 1:] / norms[:, :, 1:].clamp_min(1e-30)
                    report["write_to_state_ratio"] = float(ratio.mean())

        readout = captures.get("readout")
        if readout is not None:
            report["readout_magnitude"] = float(
                readout.detach().to(torch.float64).norm(dim=-1).mean()
            )
        return report


# --------------------------------------------------------------------------- #
# The two §5 variants
# --------------------------------------------------------------------------- #


def build_linear_model(config: ModelConfig) -> FeatureModel:
    """A1 with an ordinary residual write — the reference variant."""
    if config.arch != "linear":
        raise ValueError(f"build_linear_model got arch={config.arch!r}")
    if config.residual_write != "ordinary":
        raise ValueError(
            f"the A1 reference variant requires residual_write='ordinary', "
            f"got {config.residual_write!r}"
        )
    return FeatureModel(config)


def parameter_matched_variant(
    config: ModelConfig, target_parameters: int
) -> tuple[ModelConfig, dict]:
    """A1 retuned in width to a parameter budget.

    Provided because §7.2 asks a comparison for both a width-matched and a
    parameter-matched arm, and prompt 12 should not have to discover which
    direction the mismatch runs.

    For A0 against A1 the answer is that there is no mismatch: both mechanisms
    hold exactly one fused ``d -> 3d`` projection and one ``d -> d`` projection,
    the feature map has no parameters, and neither has a mask or a table. The two
    architectures therefore have **identical** parameter counts at every width,
    and the width-matched and parameter-matched comparisons are the same
    comparison. ``tests/identity/test_a1_model.py`` asserts that across widths
    rather than leaving it as an argument, because it is the kind of statement
    that stops being true the moment a mechanism gains a gate.
    """
    if config.arch != "linear":
        raise ValueError(f"parameter_matched_variant got arch={config.arch!r}")
    return parameter_matched_config(config, target_parameters)
