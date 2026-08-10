"""Typed run configuration, and the §7.2 frozen comparison variables in one place.

§7.2 requires that a direct architecture comparison freeze the generator version
and seed family, the split, the token budget, width and depth where possible,
the optimizer and schedule, the batch size, the initialisation policy, the
evaluation cadence, the stopping rule, the precision, and the seed set. The only
way that survives contact with a dozen later missions is if there is exactly one
place those values live and it is not a call site.

So: every number an architecture comparison must hold constant is a field of
:class:`OptimizationConfig` with an explicit default, and no training code
anywhere is allowed a literal. When prompt 12 builds the matched-comparison
harness it diffs these dataclasses; when a difference is unavoidable it goes in
that comparison's ``permitted_differences`` with a justification, and
``bin/check_no_rescue.sh`` reads it.

The rungs of §7.3's run ladder are presets here, not command-line habits. R1 is
the known-easy positive control; R2 is the capacity-stressed kill screen. What
differs between them is the *task* and the *budget*, and both are declared.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace

from architecture_mechanics.data.feature_program import (
    CONDITION_NAMES,
    GENERATOR_VERSION,
    FeatureProgramConfig,
    condition_config,
)
from architecture_mechanics.metrics.capability import METRIC_VERSION
from architecture_mechanics.models.common import MODEL_VERSION, ModelConfig

R1_EXAMPLES = 32768
R1_STEPS = 4000
"""R1's example and step budget. See :data:`EXAMPLE_BUDGET_CALIBRATION` for how
these two numbers were measured rather than guessed.

Prompt 03's :data:`~architecture_mechanics.metrics.capability.POSITIVE_CONTROL_EXAMPLES`
default of 512 is the size at which the *metric* was calibrated against the
oracle and the frequency ceiling, which needs far fewer examples than training a
model does. It is not a training budget and is not used as one here."""

__all__ = [
    "LADDERS",
    "OPERATING_POINT_EVIDENCE",
    "ArchSpec",
    "DataSpec",
    "OptimizationConfig",
    "RunConfig",
    "config_fingerprint",
    "ladder_config",
    "run_config_from_dict",
]


class RunConfigError(ValueError):
    """A run configuration that would not mean what it says."""


def _canonical(value):
    """What this value looks like after a round trip through JSON.

    A run's identity is a digest of ``RunConfig.as_dict()`` serialised as JSON,
    and ``reproduce.sh`` hands the manifest's recorded config straight back to
    :func:`run_config_from_dict`. A tuple written in source becomes a list in
    the manifest; if the two disagreed the reproduction would compute a
    different run ID for the same experiment. Normalising on the way in makes
    the identity a property of the values rather than of how they were typed.
    """
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


# --------------------------------------------------------------------------- #
# Frozen comparison variables
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class OptimizationConfig:
    """Everything §7.2 freezes about *how* a model is trained.

    Every field is shared identically by every architecture. A candidate that
    needs a different value here is not being compared, it is being rescued, and
    the difference has to be declared and justified rather than typed at a call
    site.
    """

    optimizer: str = "adamw"
    learning_rate: float = 3e-3
    """Large for a language model, ordinary for a 100k-parameter one. Chosen
    once, before any architecture existed to be flattered by it."""

    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.01
    decay_matrices_only: bool = True
    """Decay weights of rank >= 2 only. Biases, LayerNorm gains, and the
    position table are not shrunk toward zero."""

    grad_clip: float = 1.0

    schedule: str = "cosine"
    warmup_fraction: float = 0.1
    min_lr_fraction: float = 0.1

    batch_size: int = 128
    max_steps: int = R1_STEPS
    eval_every: int = 1000

    stopping_rule: str = "fixed_steps"
    """Fixed budget, and the reported number is the *final* evaluation, not the
    best one. Early stopping on the primary metric would let a model be stopped
    at its luckiest evaluation, and "best over evaluations" is a free parameter
    with the same effect. The best-so-far value is recorded as a diagnostic and
    never as the result."""

    precision: str = "fp32"
    float32_matmul_precision: str = "highest"
    """``highest`` disables TF32. On this GPU the tiny matmuls are not
    matmul-bound, so the accuracy is free, and it removes one source of
    cross-machine disagreement from a reproduction."""

    loss: str = "importance_weighted_mse + activity_bce"
    value_loss_weight: float = 1.0
    activity_loss_weight: float = 1.0
    """Two heads, two losses. The value loss is importance-weighted MSE over the
    content bank at supervised positions — the exact quantity
    :func:`~architecture_mechanics.metrics.capability.reconstruction_loss`
    reports, so the training objective and the primary reconstruction metric
    cannot drift apart. The activity loss is BCE against the ground-truth active
    mask, which is what the detection and answer-set metrics score."""

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# What to train, and on what
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ArchSpec:
    """The architecture, without the parts the dataset determines.

    ``n_features`` and ``seq_len`` are properties of the data, so they are not
    here; :meth:`bind` fills them in. That separation is what lets prompt 12
    compare two architectures by diffing specs, and what stops a model from
    being built against a feature bank that does not exist.
    """

    arch: str = "softmax"
    d_model: int | None = None
    """``None`` means "use the condition's ``d_recommended``". The generator
    records the width each condition was designed for — the point of
    ``capacity_stressed`` is ``F >> d`` — and a run that quietly picked its own
    width would have stopped testing what the condition names."""

    n_layers: int = 2
    n_heads: int = 2
    mlp_ratio: int = 4
    bias: bool = True
    init_std: float = 0.02
    residual_write: str = "ordinary"
    positional: str = "learned"
    scale_residual_projections: bool = True

    def bind(self, *, n_features: int, seq_len: int, d_recommended: int) -> tuple[ModelConfig, str]:
        """Produce the concrete model config, and say where ``d`` came from."""
        source = "d_recommended" if self.d_model is None else "explicit"
        width = d_recommended if self.d_model is None else self.d_model
        return (
            ModelConfig(
                n_features=n_features,
                seq_len=seq_len,
                d_model=width,
                n_layers=self.n_layers,
                n_heads=self.n_heads,
                mlp_ratio=self.mlp_ratio,
                bias=self.bias,
                arch=self.arch,
                residual_write=self.residual_write,
                positional=self.positional,
                init_std=self.init_std,
                scale_residual_projections=self.scale_residual_projections,
            ),
            source,
        )

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DataSpec:
    """Which §4.4 condition, at what size, from which seed family."""

    condition: str = "positive_control"
    n_train: int = R1_EXAMPLES
    n_eval: int = R1_EXAMPLES
    data_seed: int | None = None
    """``None`` keeps the generator's own default seed, which is what
    :func:`~architecture_mechanics.metrics.capability.positive_control_datasets`
    uses. Setting it moves the whole seed family, data and all."""

    generator_overrides: dict = field(default_factory=dict)
    """Fields of :class:`~...data.feature_program.FeatureProgramConfig` this run
    moves away from its condition's declared value.

    This exists for exactly one thing: §4.3 names five difficulty axes for T1 —
    source distance, distractor count, feature sparsity, key collisions, and
    simultaneous associations — and a competence envelope is a curve along them.
    Every axis is already a field of the generator config, so the alternative
    was six near-duplicate conditions, and §4.4's six controls are a fixed list
    that means something.

    An override is *declared data*: it goes into ``as_dict``, therefore into the
    run identity, the manifest, and ``bin/check_no_rescue.sh``'s diff. A cell
    that moved an axis cannot be mistaken for the condition it started from, and
    a candidate architecture given a different axis value than its control is an
    undeclared difference the gate names. Values are canonicalised to what JSON
    round-trips, so a config rebuilt from a manifest has the same identity as
    the one that produced it.
    """

    def __post_init__(self) -> None:
        if self.condition not in CONDITION_NAMES:
            raise RunConfigError(f"unknown condition {self.condition!r}; expected {CONDITION_NAMES}")
        object.__setattr__(self, "generator_overrides", _canonical(self.generator_overrides))
        unknown = sorted(
            set(self.generator_overrides) - {f.name for f in fields(FeatureProgramConfig)}
        )
        if unknown:
            raise RunConfigError(f"generator_overrides names no such generator field: {unknown}")
        forbidden = sorted(
            set(self.generator_overrides) & {"family", "condition", "split", "n_examples", "seed"}
        )
        if forbidden:
            # These four are set by the runner from the condition, the split and
            # the DataSpec's own fields. An override would be silently discarded
            # by _datasets() while still changing the run identity, which is the
            # worst of both: a different name for the same experiment.
            raise RunConfigError(
                f"generator_overrides may not set {forbidden}; those come from the condition, "
                "the split, n_train/n_eval and data_seed"
            )
        if self.n_train < 1 or self.n_eval < 1:
            raise RunConfigError("n_train and n_eval must be >= 1")
        if self.condition == "positive_control" and self.n_train != self.n_eval:
            # positive_control_datasets() emits both splits at one size, and the
            # runner checks the dataset hash it is handed. Two different numbers
            # here would be silently ignored, which is worse than refused.
            raise RunConfigError(
                "the positive control draws train and eval at one size; "
                f"got n_train={self.n_train} and n_eval={self.n_eval}"
            )
        if self.generator_overrides and self.condition == "positive_control":
            # positive_control_datasets() owns the R1 split and hashes it. An
            # override here would be discarded there and the run would be named
            # for a dataset it never saw.
            raise RunConfigError(
                "the positive control is the frozen known-easy condition and takes no "
                "generator_overrides; sweep difficulty on capacity_stressed"
            )
        self.generator_config(split="train", n_examples=self.n_train)

    def generator_config(self, *, split: str = "train", n_examples: int = 1) -> FeatureProgramConfig:
        """This run's generator configuration, overrides applied.

        Built once here so that an override combination the generator would
        refuse — three distractors between a source and a destination two apart
        — is refused when the configuration is written rather than after the
        model has been placed on the GPU.
        """
        overrides = dict(self.generator_overrides)
        if self.data_seed is not None:
            overrides["seed"] = self.data_seed
        return condition_config(
            self.condition, split=split, n_examples=n_examples, **overrides
        )

    @property
    def d_recommended(self) -> int:
        return self.generator_config().d_recommended

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RunConfig:
    """One run: a rung, an architecture, a condition, a seed, and a device."""

    ladder: str = "R1"
    seed: int = 20260809
    device: str = "cuda"
    arch: ArchSpec = field(default_factory=ArchSpec)
    data: DataSpec = field(default_factory=DataSpec)
    optim: OptimizationConfig = field(default_factory=OptimizationConfig)
    capture_examples: int = 256
    """How many evaluation examples the mechanism-activity forward pass captures
    attention over. Bounded because the ``(B, H, T, T)`` weight tensor is the one
    object in this laboratory whose size is quadratic in anything."""

    geometry_examples: int = 1024
    """How many evaluation examples the §6.2 geometry pass captures hidden states
    over. Larger than ``capture_examples`` because nothing here is quadratic in
    sequence length, and it needs to be: the probe fits ``F`` regressors on half
    the rows, so ``1024`` examples give roughly 6 000 fitting rows against a
    feature bank of 36 to 124. Frozen in the config rather than chosen at the
    call site because it is a §7.2 comparison variable — a candidate architecture
    measured on more rows than its control would be measured more precisely, and
    a difference in precision reads as a difference in geometry."""

    def __post_init__(self) -> None:
        if self.ladder not in LADDERS:
            raise RunConfigError(f"unknown ladder rung {self.ladder!r}; expected {sorted(LADDERS)}")
        if self.geometry_examples < 2:
            raise RunConfigError("geometry_examples must be >= 2; a probe split needs two examples")

    def as_dict(self) -> dict:
        return {
            "ladder": self.ladder,
            "seed": self.seed,
            "device": self.device,
            "capture_examples": self.capture_examples,
            "geometry_examples": self.geometry_examples,
            "arch": self.arch.as_dict(),
            "data": self.data.as_dict(),
            "optim": self.optim.as_dict(),
            "generator_version": GENERATOR_VERSION,
            "metric_version": METRIC_VERSION,
            "model_version": MODEL_VERSION,
        }


# --------------------------------------------------------------------------- #
# The §7.3 rungs
# --------------------------------------------------------------------------- #

LADDERS: dict[str, dict] = {
    # No training at all: build the model and check the §8.5 invariants. The
    # authoritative versions of these live in tests/; this rung is the gate an
    # operator can run in one command before spending GPU time.
    "R0": {
        "description": "unit and identity checks, no optimisation",
        "data": {"condition": "positive_control", "n_train": 64, "n_eval": 64},
        "optim": {"max_steps": 0, "eval_every": 1},
    },
    # Known-easy positive control: ample dimension, distance one to two, no
    # distractors, one association. A0 must solve this rapidly. Failure here is
    # an implementation or optimisation bug, never a finding.
    #
    # The example budget was calibrated once, before any second architecture
    # existed, by measuring where A0 clears prompt 03's frozen 0.80 threshold —
    # see EXAMPLE_BUDGET_CALIBRATION below. It is not 512: at that size A0
    # reaches exact-set accuracy 1.000 on the examples it trained on and 0.39 on
    # held-out ones, which is a memorisation result and tells us nothing about
    # the mechanism.
    "R1": {
        "description": "known-easy positive control, one seed",
        "data": {"condition": "positive_control", "n_train": R1_EXAMPLES, "n_eval": R1_EXAMPLES},
        "optim": {"max_steps": R1_STEPS, "eval_every": 1000},
    },
    # Kill screen: capacity-stressed, short, one seed, stop on collapse. Half
    # R1's example budget and fewer steps, because the point of a screen is to
    # cost less than the pilot it screens for; the sequence is four times longer
    # and the feature bank three times wider, so this is the larger run in GPU
    # memory even so.
    "R2": {
        "description": "capacity-stressed kill screen, one seed, short",
        "data": {"condition": "capacity_stressed", "n_train": 16384, "n_eval": 4096},
        "optim": {"max_steps": 2000, "eval_every": 500},
    },
    # Full pilot: one seed per cell of the fixed task matrix, complete evidence
    # bundle. The matrix itself is declared in experiments/t1_ladder.py; what is
    # declared here is the operating point every cell shares.
    #
    # d = 64 rather than the condition's d_recommended = 16, and this is the one
    # place in the laboratory where a rung does not honour it. The reason is
    # recorded in OPERATING_POINT_EVIDENCE below and in the run itself, which
    # writes honours_d_recommended: false. It is a choice of *regime*, made once
    # and frozen for every architecture that follows; a pilot run where the
    # baseline scores 0.013 measures the floor and nothing else.
    "R3": {
        "description": "full pilot, one seed per task-matrix cell, complete bundle",
        "data": {"condition": "capacity_stressed", "n_train": 16384, "n_eval": 4096},
        "optim": {"max_steps": 3000, "eval_every": 500},
        "arch": {"d_model": 64},
    },
    # Replication: the R3 base cell at five or more seeds. Identical in every
    # §7.2 variable to R3's base cell except the seed and the rung, so the
    # spread it measures is the spread of the procedure and not of the design.
    "R4": {
        "description": "replication of the R3 base cell across seeds",
        "data": {"condition": "capacity_stressed", "n_train": 16384, "n_eval": 4096},
        "optim": {"max_steps": 3000, "eval_every": 500},
        "arch": {"d_model": 64},
    },
}

OPERATING_POINT_EVIDENCE: tuple[dict, ...] = (
    {"d_model": 16, "note": "the condition's d_recommended", "recall": 0.0129, "feature_f1": 0.3101},
    {"d_model": 32, "recall": 0.0581, "feature_f1": 0.4175},
    {"d_model": 64, "recall": 0.4807, "feature_f1": 0.8318},
)
"""Why R3 and R4 run at ``d = 64``, recorded so the choice is auditable.

Measured by prompt 07's three recorded R2 kill screens — A0, seed 20260809,
``capacity_stressed``, 16384 examples, 2000 steps — and committed to this
laboratory before the pre-registration that cites them. ``recall`` is
``associative_recall_accuracy`` on the held-out split.

At the condition's own ``d_recommended = 16`` A0 answers 1.3% of recall queries
exactly, which is the floor: a comparison run there would measure which
architecture is marginally less useless, and a seed-to-seed spread there is the
spread of a number pinned against zero. At 64 the baseline sits near the middle
of its range, which is where both a difference and a variance are visible.

This is a choice of regime, not a tuning of A0: it is a §7.2 frozen variable,
identical for every architecture measured at this operating point, and it is
declared here rather than passed at a call site so that a candidate cannot
receive a different one."""

EXAMPLE_BUDGET_CALIBRATION: tuple[dict, ...] = (
    {"n_train": 512, "max_steps": 1500, "recall": 0.3867, "feature_f1": 0.8149},
    {"n_train": 2048, "max_steps": 1500, "recall": 0.5869, "feature_f1": 0.8920},
    {"n_train": 8192, "max_steps": 1500, "recall": 0.7957, "feature_f1": 0.9546},
    {"n_train": 8192, "max_steps": 3000, "recall": 0.8010, "feature_f1": 0.9556},
    {"n_train": 16384, "max_steps": 3000, "recall": 0.8589, "feature_f1": 0.9699},
    {"n_train": 32768, "max_steps": 4000, "recall": 0.9055, "feature_f1": 0.9801},
)
"""How R1's example budget was chosen, recorded so it is auditable rather than
asserted. A0, seed 20260809, ``d=48``, two layers, two heads, on the known-easy
positive control; ``recall`` is prompt 03's frozen primary metric.

Two things this table says. Doubling steps at a fixed example count buys almost
nothing (0.7957 to 0.8010) while quadrupling examples buys a lot, so the limit
is samples and not optimisation. And the residual errors are not spread over the
answer: at ``n_train=8192`` the missed features have median true magnitude 0.023
against 0.498 for all active features, and 99.9% of misses are below 0.2. The
generator draws magnitudes ``Uniform(0, 1)`` while the answer set counts a
feature as present at any magnitude, so exact-set accuracy is partly a measure
of magnitude resolution near zero. Prompts 09 and 13 should read it that way.

Chosen: 32768 examples and 4000 steps, the first cell with a margin (0.106) that
should survive a five-seed replication rather than one that lands on the bar.
"""


def ladder_config(
    ladder: str,
    *,
    arch: str = "softmax",
    seed: int = 20260809,
    device: str = "cuda",
    d_model: int | None = None,
    overrides: dict | None = None,
) -> RunConfig:
    """The declared configuration for one rung. Presets, not habits."""
    if ladder not in LADDERS:
        raise RunConfigError(f"unknown ladder rung {ladder!r}; expected {sorted(LADDERS)}")
    preset = LADDERS[ladder]
    # The rung's own architecture block first, then an explicit d_model. R3 and
    # R4 declare their width because it is part of the operating point; passing
    # --d-model still overrides it, and the run identity records which happened.
    arch_spec = replace(ArchSpec(**preset.get("arch", {})), arch=arch)
    if d_model is not None:
        arch_spec = replace(arch_spec, d_model=d_model)
    config = RunConfig(
        ladder=ladder,
        seed=seed,
        device=device,
        arch=arch_spec,
        data=DataSpec(**preset["data"]),
        optim=replace(OptimizationConfig(), **preset["optim"]),
    )
    if overrides:
        config = replace(config, **overrides)
    return config


def run_config_from_dict(payload: dict) -> RunConfig:
    """Rebuild a :class:`RunConfig` from what :meth:`RunConfig.as_dict` wrote.

    This is what makes a run reproducible from its own manifest rather than from
    a remembered command line: ``reproduce.sh`` hands the manifest's recorded
    ``config`` block straight back here, so a run that used an option nobody
    thought to expose on the CLI still reproduces exactly.

    The three version stamps are *verified*, not ignored. ``as_dict`` writes
    them as outputs; if the generator, the metrics, or the model have changed
    semantics since the run, re-executing this config would produce a different
    experiment under the same name, which is worse than refusing. A reproduction
    that must silence this is not a reproduction.
    """
    expected = {
        "generator_version": GENERATOR_VERSION,
        "metric_version": METRIC_VERSION,
        "model_version": MODEL_VERSION,
    }
    for key, current in expected.items():
        recorded = payload.get(key)
        if recorded is not None and recorded != current:
            raise RunConfigError(
                f"config records {key}={recorded!r} but this source tree is {current!r}; "
                "re-running it would be a different experiment under the same name"
            )

    known = set(expected) | {
        "ladder", "seed", "device", "capture_examples", "geometry_examples",
        "arch", "data", "optim",
    }
    unknown = sorted(set(payload) - known)
    if unknown:
        raise RunConfigError(f"unknown configuration keys {unknown}")

    def section(name: str, kind: type) -> dict:
        block = payload.get(name) or {}
        if not isinstance(block, dict):
            raise RunConfigError(f"{name} must be a mapping, got {type(block).__name__}")
        allowed = {f.name for f in fields(kind)}
        extra = sorted(set(block) - allowed)
        if extra:
            raise RunConfigError(f"unknown {name} keys {extra}")
        return block

    return RunConfig(
        ladder=payload.get("ladder", "R1"),
        seed=int(payload.get("seed", 20260809)),
        device=payload.get("device", "cuda"),
        capture_examples=int(payload.get("capture_examples", 256)),
        geometry_examples=int(payload.get("geometry_examples", 1024)),
        arch=ArchSpec(**section("arch", ArchSpec)),
        data=DataSpec(**section("data", DataSpec)),
        optim=OptimizationConfig(**section("optim", OptimizationConfig)),
    )


def config_fingerprint(config: RunConfig) -> str:
    """A short stable digest of the whole configuration.

    Used to name the run directory so that re-running the same configuration
    overwrites its own evidence instead of accumulating near-duplicates that
    differ only in a timestamp. §8.3's full run identity is prompt 05's.
    """
    import hashlib
    import json

    payload = json.dumps(config.as_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:12]
