"""The shared tiny-model scaffold and the ``MixingPrimitive`` contract.

Every architecture in §5 is this trunk with one part swapped. The trunk reads a
sparse feature vector per position into a bottleneck of width ``d``, runs ``n``
pre-norm mixing blocks, and reads out two heads over the feature bank: a value
head for "how much of feature f is present here" and an activity head for "is
feature f present here at all". Those are exactly the two channels
:class:`~architecture_mechanics.metrics.capability.Predictions` scores, so no
part of the model has to guess what the ruler wants.

It is a bottleneck autoencoder with a sequence mixer in the middle, not a
language model. There is no tokenizer, no vocabulary, no weight tying, and
``d`` is 16 to 64. Anything the trunk contains is shared identically by every
architecture, which is what makes a §7.2 matched comparison possible: swapping
the mixing primitive is then the only difference, by construction rather than
by care.

The ``MixingPrimitive`` contract is what prompts 11 (A1 linear attention) and
17 (A2 delta memory) implement against. Four requirements:

1. ``forward(x, hooks=...)`` maps ``(B, T, d)`` to ``(B, T, d)`` causally.
2. ``reference_forward(x)`` does the same computation in explicit loops, slowly
   and obviously. The equation test compares the two. This is the part that
   makes a later architecture *comparison* trustworthy, and it is the part
   people skip.
3. ``hook_sites()`` names the tensors the mechanism will expose. The scaffold
   never interprets these names; it only pushes a scope so two layers do not
   collide. Prompt 19 hooks all three architectures through this one interface.
4. ``mechanism_activity(captures)`` turns those captured tensors into §6.3
   scalars. An architecture that trains while ignoring its special branch is a
   failed mechanism experiment even when the loss looks good, so every
   primitive has to be able to answer "were you actually used".
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace

import torch
from torch import nn

from architecture_mechanics.instrumentation.hooks import NO_HOOKS, HookContext

__all__ = [
    "MODEL_VERSION",
    "FeatureModel",
    "MixingBlock",
    "MixingPrimitive",
    "ModelConfig",
    "ModelConfigError",
    "ModelOutput",
    "build_primitive",
    "count_parameters",
    "gelu_reference",
    "layer_norm_reference",
    "model_reference_forward",
    "parameter_matched_config",
    "parameter_report",
    "primitive_names",
    "register_primitive",
]

MODEL_VERSION = "am-model-1.0.0"
"""Bumped when the trunk changes shape or initialisation. Recorded next to
``GENERATOR_VERSION`` and ``METRIC_VERSION`` so a comparison across two runs can
tell "different architecture" from "different scaffold"."""


class ModelConfigError(ValueError):
    """A model configuration that cannot be built, caught before allocation."""


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelConfig:
    """Everything that determines the model. Hashed into the run identity.

    ``n_features`` and ``seq_len`` come from the dataset, never from a default:
    a model whose feature bank silently disagreed with its data would train
    perfectly well on the wrong columns.
    """

    n_features: int
    seq_len: int
    d_model: int = 48
    n_layers: int = 2
    n_heads: int = 2
    mlp_ratio: int = 4
    bias: bool = True
    arch: str = "softmax"

    residual_write: str = "ordinary"
    """§5 A0 requires an ordinary residual write as the reference variant. A4
    (depth routing) will add others; naming the choice here keeps "ordinary" an
    explicit setting rather than an unstated assumption."""

    positional: str = "learned"
    """``learned`` absolute position embeddings, or ``none``. T1's
    content-addressed recall does not need position, but the matched-difficulty
    control requires *ordinal* addressing, so the reference model must be able
    to represent position at all or that control would be unwinnable for reasons
    having nothing to do with the mixing mechanism."""

    init_std: float = 0.02
    """Normal init scale for every projection. §7.2 freezes initialisation
    policy, so it is one number here rather than per-module taste."""

    scale_residual_projections: bool = True
    """Divide residual-write projections by ``sqrt(2 * n_layers)`` at init, so
    the residual stream's variance does not grow with depth. Standard, shared by
    every architecture, and therefore not a confound."""

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.n_features < 1:
            raise ModelConfigError("n_features must be >= 1")
        if self.seq_len < 1:
            raise ModelConfigError("seq_len must be >= 1")
        if self.d_model < 1:
            raise ModelConfigError("d_model must be >= 1")
        if self.n_layers < 1:
            raise ModelConfigError("n_layers must be >= 1")
        if self.n_heads < 1:
            raise ModelConfigError("n_heads must be >= 1")
        if self.d_model % self.n_heads:
            raise ModelConfigError(
                f"d_model={self.d_model} is not divisible by n_heads={self.n_heads}"
            )
        if self.mlp_ratio < 1:
            raise ModelConfigError("mlp_ratio must be >= 1")
        if self.residual_write not in ("ordinary",):
            raise ModelConfigError(f"unknown residual_write {self.residual_write!r}")
        if self.positional not in ("learned", "none"):
            raise ModelConfigError(f"unknown positional {self.positional!r}")

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    def as_dict(self) -> dict:
        record = asdict(self)
        record["model_version"] = MODEL_VERSION
        record["d_head"] = self.d_head
        return record


@dataclass
class ModelOutput:
    """The two prediction channels §6.1 scores, kept separate on purpose.

    ``values`` answers "how much of feature f"; ``active_logits`` answers "is
    feature f present". A single head serving both would force the activity
    ranking to be a monotone function of predicted magnitude, and the
    ground-truth magnitudes are ``Uniform(0, 1)`` — so a genuinely present
    feature with a small magnitude would be indistinguishable from an absent one
    at the rate-matched operating point the metrics use.
    """

    values: torch.Tensor
    active_logits: torch.Tensor

    @property
    def active_prob(self) -> torch.Tensor:
        return torch.sigmoid(self.active_logits)


# --------------------------------------------------------------------------- #
# The mixing primitive contract
# --------------------------------------------------------------------------- #


class MixingPrimitive(nn.Module, ABC):
    """One sequence-mixing mechanism, swappable inside the shared trunk.

    Subclasses set :attr:`kind` and :attr:`SITES` and implement
    :meth:`forward`, :meth:`reference_forward`, and
    :meth:`mechanism_activity`. Everything else the scaffold needs is here.
    """

    kind: str = ""
    """Registry key. ``softmax``, ``linear``, ``delta`` ..."""

    SITES: tuple[str, ...] = ()
    """Local hook-site names, in the order the forward pass declares them.

    Local, not qualified: the primitive does not know it is layer 3. Names are
    shared across architectures wherever the tensors mean the same thing, so
    that prompt 19 can hook ``q``, ``k``, ``v``, ``weights``, and ``readout`` on
    A0, A1, and A2 through the same call.
    """

    def __init__(self, config: ModelConfig, layer_index: int) -> None:
        super().__init__()
        self.config = config
        self.layer_index = int(layer_index)

    # -- required ---------------------------------------------------------- #

    @abstractmethod
    def forward(self, x: torch.Tensor, *, hooks: HookContext = NO_HOOKS) -> torch.Tensor:
        """``(B, T, d) -> (B, T, d)``, causal, batched."""

    @abstractmethod
    def reference_forward(self, x: torch.Tensor) -> torch.Tensor:
        """The same computation in explicit loops, no batching tricks.

        Correct by inspection is the requirement, not speed. This is the
        definition the fast path is tested against.
        """

    @abstractmethod
    def mechanism_activity(self, captures: Mapping[str, torch.Tensor]) -> dict[str, float]:
        """§6.3 activity scalars, computed from this layer's captured sites.

        ``captures`` is keyed by *local* site name; the scaffold strips its own
        scope before calling, so a primitive never has to know its layer index
        to read its own tensors back.
        """

    # -- provided ---------------------------------------------------------- #

    def hook_sites(self) -> tuple[str, ...]:
        return self.SITES

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


_PRIMITIVES: dict[str, type[MixingPrimitive]] = {}


def register_primitive(cls: type[MixingPrimitive]) -> type[MixingPrimitive]:
    """Register a mixing primitive under its ``kind``. Used as a decorator."""
    if not cls.kind:
        raise ModelConfigError(f"{cls.__name__} must set a non-empty `kind`")
    existing = _PRIMITIVES.get(cls.kind)
    if existing is not None and existing is not cls:
        raise ModelConfigError(f"two primitives claim kind {cls.kind!r}: {existing} and {cls}")
    _PRIMITIVES[cls.kind] = cls
    return cls


def build_primitive(config: ModelConfig, layer_index: int) -> MixingPrimitive:
    if config.arch not in _PRIMITIVES:
        raise ModelConfigError(
            f"unknown arch {config.arch!r}; registered: {sorted(_PRIMITIVES)}. "
            "A primitive is registered by importing the module that defines it."
        )
    return _PRIMITIVES[config.arch](config, layer_index)


def primitive_names() -> tuple[str, ...]:
    return tuple(sorted(_PRIMITIVES))


# --------------------------------------------------------------------------- #
# The trunk
# --------------------------------------------------------------------------- #


class MLP(nn.Module):
    """Pre-norm position-wise MLP, identical for every architecture.

    Present because T0 reconstruction of superposed features is not a linear
    problem: with ``F >> d`` the readout has to disentangle overlapping feature
    directions, which needs a nonlinearity somewhere. Putting it here rather
    than in the mixing primitive keeps it out of the architecture contrast.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden = config.d_model * config.mlp_ratio
        self.up = nn.Linear(config.d_model, hidden, bias=config.bias)
        self.down = nn.Linear(hidden, config.d_model, bias=config.bias)
        self.down.weight._residual_projection = True
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.act(self.up(x)))


class MixingBlock(nn.Module):
    """Pre-norm residual block: mix, then position-wise MLP."""

    def __init__(self, config: ModelConfig, layer_index: int) -> None:
        super().__init__()
        self.norm_mix = nn.LayerNorm(config.d_model)
        self.mix = build_primitive(config, layer_index)
        self.norm_mlp = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, *, hooks: HookContext = NO_HOOKS) -> torch.Tensor:
        with hooks.scope("mix"):
            x = x + self.mix(self.norm_mix(x), hooks=hooks)
        x = hooks.site("resid_mid", x)
        x = x + self.mlp(self.norm_mlp(x))
        return hooks.site("resid_out", x)


class FeatureModel(nn.Module):
    """Encoder, ``n`` mixing blocks, and the two-headed feature readout."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

        self.encoder = nn.Linear(config.n_features, config.d_model, bias=config.bias)
        if config.positional == "learned":
            self.position = nn.Parameter(torch.zeros(config.seq_len, config.d_model))
        else:
            self.register_parameter("position", None)

        self.blocks = nn.ModuleList(
            MixingBlock(config, index) for index in range(config.n_layers)
        )
        self.norm_out = nn.LayerNorm(config.d_model)
        self.value_head = nn.Linear(config.d_model, config.n_features, bias=config.bias)
        self.active_head = nn.Linear(config.d_model, config.n_features, bias=config.bias)

        self.apply(self._init_module)
        if config.scale_residual_projections:
            scale = 1.0 / math.sqrt(2 * config.n_layers)
            for parameter in self.parameters():
                if getattr(parameter, "_residual_projection", False):
                    with torch.no_grad():
                        parameter.mul_(scale)

    def _init_module(self, module: nn.Module) -> None:
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, FeatureModel) and module.position is not None:
            nn.init.normal_(module.position, mean=0.0, std=std)

    # -- forward ----------------------------------------------------------- #

    def forward(self, x: torch.Tensor, *, hooks: HookContext = NO_HOOKS) -> ModelOutput:
        if x.dim() != 3 or x.shape[-1] != self.config.n_features:
            raise ModelConfigError(
                f"expected (B, T, {self.config.n_features}) input, got {tuple(x.shape)}"
            )
        seq_len = x.shape[1]
        if seq_len > self.config.seq_len:
            raise ModelConfigError(
                f"sequence of {seq_len} exceeds the model's seq_len={self.config.seq_len}"
            )

        h = self.encoder(x)
        if self.position is not None:
            h = h + self.position[:seq_len]
        h = hooks.site("embed", h)

        for index, block in enumerate(self.blocks):
            with hooks.scope(f"layers.{index}"):
                h = block(h, hooks=hooks)

        h = self.norm_out(h)
        # Named "final_norm", not "readout": "readout" is a *mechanism* site (it
        # is A2's query readout, and A0's post-mixing value), and a hook context
        # may address sites by their bare local name. A trunk site sharing a
        # local name with a mechanism site would silently capture three tensors
        # where the caller asked for two.
        h = hooks.site("final_norm", h)
        values = hooks.site("head_value", self.value_head(h))
        logits = hooks.site("head_active", self.active_head(h))
        return ModelOutput(values=values, active_logits=logits)

    # -- reporting --------------------------------------------------------- #

    def hook_sites(self) -> tuple[str, ...]:
        """Every fully qualified site this model declares, in forward order."""
        sites = ["embed"]
        for index, block in enumerate(self.blocks):
            prefix = f"layers.{index}"
            sites.extend(f"{prefix}.mix.{name}" for name in block.mix.hook_sites())
            sites.append(f"{prefix}.resid_mid")
            sites.append(f"{prefix}.resid_out")
        sites.extend(["final_norm", "head_value", "head_active"])
        return tuple(sites)

    def local_site_names(self) -> tuple[str, ...]:
        """Every bare site name in use, trunk and mechanisms together.

        A test asserts these are unique, because bare-name addressing is a
        supported way to reach a site and it silently over-matches if two
        different tensors share a name.
        """
        names = ["embed", "resid_mid", "resid_out", "final_norm", "head_value", "head_active"]
        for block in self.blocks:
            names.extend(block.mix.hook_sites())
        return tuple(names)

    def mechanism_activity(self, captures: Mapping[str, torch.Tensor]) -> dict[str, float]:
        """Per-layer §6.3 activity, delegated to each primitive.

        The scaffold strips its own scope and hands each primitive only its own
        tensors, so a primitive reads back exactly the local names it declared.
        """
        report: dict[str, float] = {}
        for index, block in enumerate(self.blocks):
            prefix = f"layers.{index}.mix."
            local = {
                key[len(prefix):]: value
                for key, value in captures.items()
                if key.startswith(prefix)
            }
            if not local:
                continue
            for name, value in block.mix.mechanism_activity(local).items():
                report[f"layers.{index}.{name}"] = value
        return report

    def parameter_report(self) -> dict:
        return parameter_report(self)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def parameter_report(model: FeatureModel) -> dict:
    """Parameter count broken down by component.

    Reported per run because §7.2 requires both a width-matched and a
    parameter-matched comparison, and a single total hides which part of a
    candidate grew.
    """
    groups: dict[str, int] = {
        "encoder": count_parameters(model.encoder),
        "position": 0 if model.position is None else model.position.numel(),
        "norm_out": count_parameters(model.norm_out),
        "value_head": count_parameters(model.value_head),
        "active_head": count_parameters(model.active_head),
    }
    mixing = 0
    mlp = 0
    norms = 0
    for block in model.blocks:
        mixing += count_parameters(block.mix)
        mlp += count_parameters(block.mlp)
        norms += count_parameters(block.norm_mix) + count_parameters(block.norm_mlp)
    groups["mixing"] = mixing
    groups["mlp"] = mlp
    groups["block_norms"] = norms
    total = count_parameters(model)
    if sum(groups.values()) != total:
        raise ModelConfigError(
            f"parameter report sums to {sum(groups.values())} but the model has {total}; "
            "a component was added without being accounted for"
        )
    groups["total"] = total
    groups["trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return groups


def parameters_for(config: ModelConfig) -> int:
    """Total parameters a config would produce, by building it and counting.

    Built and counted rather than computed from a formula, because a formula
    here would be a second implementation of the model that could drift from the
    first. The hand formula lives in the R0 test, where disagreement is the
    point.

    Constructed inside a forked RNG so that asking how big a model *would* be
    does not advance the generator and change the model that actually gets
    trained. That is the kind of bug that only shows up as an unreproducible
    run three weeks later.
    """
    with torch.random.fork_rng(devices=[]):
        return count_parameters(FeatureModel(config))


def parameter_matched_config(
    config: ModelConfig,
    target_parameters: int,
    *,
    widths: Sequence[int] | None = None,
) -> tuple[ModelConfig, dict]:
    """The §5 A0 narrower/wider control: same trunk, width retuned to a budget.

    When a candidate architecture adds capacity, comparing it against A0 at
    equal width credits the candidate for parameters as well as for mechanism.
    This returns the A0 config whose parameter count is closest to
    ``target_parameters``, together with the bracketing widths so the report can
    say how tight the match is and in which direction it errs.

    Width is the knob because depth changes the number of mixing operations,
    which is the thing under study.
    """
    if target_parameters < 1:
        raise ModelConfigError("target_parameters must be >= 1")
    step = config.n_heads
    if widths is not None:
        candidates = list(widths)
    else:
        # Anchored to this config's own width, not to a fixed constant: a search
        # grid that stops below the width being matched can only ever return the
        # widest candidate it happens to hold, and would do so silently.
        candidates = list(range(step, max(4 * config.d_model, 16 * step) + 1, step))

    # Parameter count is strictly increasing in width, so the grid is sorted and
    # only the two candidates bracketing the target need to be built. Building
    # all of them would be correct and slower; the bisection keeps this cheap
    # enough that prompt 12 can call it per comparison.
    measured: dict[int, int] = {}

    def count(width: int) -> int:
        if width % step:
            raise ModelConfigError(f"width {width} is not divisible by n_heads={step}")
        if width not in measured:
            measured[width] = parameters_for(replace(config, d_model=width))
        return measured[width]

    low, high = 0, len(candidates) - 1
    while low < high:
        middle = (low + high) // 2
        if count(candidates[middle]) < target_parameters:
            low = middle + 1
        else:
            high = middle
    bracket = [candidates[index] for index in {max(low - 1, 0), low, min(low + 1, len(candidates) - 1)}]
    pairs = sorted((width, count(width)) for width in bracket)

    best_width, best_count = min(pairs, key=lambda item: abs(item[1] - target_parameters))
    below = [item for item in pairs if item[1] <= target_parameters]
    above = [item for item in pairs if item[1] >= target_parameters]
    report = {
        "target_parameters": int(target_parameters),
        "matched_d_model": int(best_width),
        "matched_parameters": int(best_count),
        "relative_error": (best_count - target_parameters) / target_parameters,
        "narrower": None if not below else {"d_model": below[-1][0], "parameters": below[-1][1]},
        "wider": None if not above else {"d_model": above[0][0], "parameters": above[0][1]},
        "searched_widths": sorted(measured),
        "grid": {"min": candidates[0], "max": candidates[-1], "step": step},
    }
    return replace(config, d_model=best_width), report


# --------------------------------------------------------------------------- #
# Slow reference for the whole trunk
# --------------------------------------------------------------------------- #


def layer_norm_reference(vector: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                         eps: float = 1e-5) -> torch.Tensor:
    """LayerNorm over one vector, written as its definition."""
    mean = vector.mean()
    variance = ((vector - mean) ** 2).mean()
    return (vector - mean) / torch.sqrt(variance + eps) * weight + bias


def gelu_reference(vector: torch.Tensor) -> torch.Tensor:
    """Exact GELU: ``0.5 x (1 + erf(x / sqrt(2)))``, matching ``nn.GELU()``."""
    return 0.5 * vector * (1.0 + torch.erf(vector / math.sqrt(2.0)))


def model_reference_forward(model: FeatureModel, x: torch.Tensor) -> ModelOutput:
    """The whole trunk in explicit loops, one position at a time.

    Every batch element and every position is handled separately; the only
    batched call left is each primitive's own :meth:`reference_forward`, which
    is itself loop-written. Slow by design: this is the definition of the model,
    and the fast path is tested against it.
    """
    batch, seq_len, _ = x.shape
    config = model.config
    dtype = x.dtype

    hidden = torch.zeros(batch, seq_len, config.d_model, dtype=dtype, device=x.device)
    for b in range(batch):
        for t in range(seq_len):
            hidden[b, t] = model.encoder.weight @ x[b, t]
            if model.encoder.bias is not None:
                hidden[b, t] = hidden[b, t] + model.encoder.bias
            if model.position is not None:
                hidden[b, t] = hidden[b, t] + model.position[t]

    for block in model.blocks:
        normed = torch.zeros_like(hidden)
        for b in range(batch):
            for t in range(seq_len):
                normed[b, t] = layer_norm_reference(
                    hidden[b, t], block.norm_mix.weight, block.norm_mix.bias, block.norm_mix.eps
                )
        hidden = hidden + block.mix.reference_forward(normed)

        for b in range(batch):
            for t in range(seq_len):
                inner = layer_norm_reference(
                    hidden[b, t], block.norm_mlp.weight, block.norm_mlp.bias, block.norm_mlp.eps
                )
                up = block.mlp.up.weight @ inner
                if block.mlp.up.bias is not None:
                    up = up + block.mlp.up.bias
                down = block.mlp.down.weight @ gelu_reference(up)
                if block.mlp.down.bias is not None:
                    down = down + block.mlp.down.bias
                hidden[b, t] = hidden[b, t] + down

    values = torch.zeros(batch, seq_len, config.n_features, dtype=dtype, device=x.device)
    logits = torch.zeros_like(values)
    for b in range(batch):
        for t in range(seq_len):
            final = layer_norm_reference(
                hidden[b, t], model.norm_out.weight, model.norm_out.bias, model.norm_out.eps
            )
            value = model.value_head.weight @ final
            logit = model.active_head.weight @ final
            if model.value_head.bias is not None:
                value = value + model.value_head.bias
            if model.active_head.bias is not None:
                logit = logit + model.active_head.bias
            values[b, t] = value
            logits[b, t] = logit
    return ModelOutput(values=values, active_logits=logits)
