"""The feature program: a generator whose ground truth is known by construction.

This is the module the rest of the programme rests on. On real text you cannot
measure feature purity, you cannot build an interference matrix, and you cannot
predict what a causal intervention *should* do — you can only measure what it
did and argue. That is exactly why ``attention_lab``'s confirmatory tier stalled
at ``insufficient_evidence``. Here the features are not inferred; they are
sampled, so every downstream measurement has something true to be compared with.

An example is a sequence of positions. Each position carries a sparse subset of
``F`` known latent features with configurable activation probability and unequal
importance. Some positions carry content, some carry an operation or a pointer.
Distractors may sit between a source and its destination. Supervised positions
have a known target feature vector. The model's bottleneck ``d < F`` forces
superposition; ``d`` is a *model* property, so a condition here records the
``d`` it was designed for (:attr:`FeatureProgramConfig.d_recommended`) rather
than pretending the data can set it.

Alongside the tensors the generator emits a :class:`ProgramRecord` per example:
which features were active where, which position was the source for which
destination, and what operation was required. That record is what will later
make an intervention *predictive* — "ablate position 31 and the answer must lose
features {4, 17, 62}" — instead of merely observable. It is designed in now
because adding it retroactively means regenerating every dataset ever produced.

Layout of the feature axis (before any permutation control):

    [ 0 .. n_content )                      content bank — the semantic payload
    [ n_content .. n_content+n_key )        key bank — which association slot
    [ n_content+n_key .. F )                op bank — one index per op code

Everything lives in one ``x_t in R^F`` as §4.2 specifies; content and operations
are distinguished by which bank they occupy, not by a separate channel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass, replace

import numpy as np
import torch

from .splits import ProgramTemplate, SplitPlan, split_templates

# Bump on any change to the *semantics* of generated data. It participates in
# every dataset hash, so a bump invalidates caches and makes stale comparisons
# visible. It deliberately does NOT participate in seed derivation: a version
# bump should not gratuitously reshuffle every draw, because then v1 and v2 can
# never be compared on matched samples.
GENERATOR_VERSION = "fpg-1.0.0"

OP_CODES: tuple[str, ...] = ("CONTENT", "BIND", "QUERY", "NOOP")
"""``NOOP`` is the lexical decoy: an operation-like marker with no semantics."""

_MAX_RESAMPLE = 32
"""Bounded rejection sampling for the "at least k active features" condition."""


class FeatureProgramError(ValueError):
    """Raised when a configuration cannot produce the data it describes."""


# --------------------------------------------------------------------------- #
# Determinism primitives
# --------------------------------------------------------------------------- #


def canonical_json(obj) -> str:
    """One byte-string per value, stable across processes and interpreters."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def derive_seed(base_seed: int, *parts: object) -> int:
    """Derive an independent stream seed from a base seed and a path.

    Streams are addressed by *what they are for* (``"content"``, ``"destroy"``,
    example index, split name) rather than by order of consumption. That is what
    makes generation batch-size independent — example 7 draws the same features
    whether it is produced alone or as part of a run of 4096 — and it is what
    lets a post-hoc transform such as source destruction draw fresh randomness
    without perturbing the base example it is transforming.
    """
    payload = canonical_json([int(base_seed), [str(p) for p in parts]])
    digest = hashlib.sha256(payload.encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


# --------------------------------------------------------------------------- #
# Feature layout
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FeatureBanks:
    """How the ``F`` feature slots are divided into content, key, and op banks."""

    n_content: int
    n_key: int
    op_codes: tuple[str, ...]

    @property
    def n_op(self) -> int:
        return len(self.op_codes)

    @property
    def n_features(self) -> int:
        return self.n_content + self.n_key + self.n_op

    @property
    def content_indices(self) -> tuple[int, ...]:
        return tuple(range(self.n_content))

    @property
    def key_indices(self) -> tuple[int, ...]:
        return tuple(range(self.n_content, self.n_content + self.n_key))

    @property
    def op_indices(self) -> tuple[int, ...]:
        start = self.n_content + self.n_key
        return tuple(range(start, start + self.n_op))

    def key_index(self, local: int) -> int:
        return self.n_content + int(local)

    def op_index(self, code: str) -> int:
        if code not in self.op_codes:
            raise FeatureProgramError(f"op code {code!r} not in bank {self.op_codes}")
        return self.n_content + self.n_key + self.op_codes.index(code)

    def as_dict(self) -> dict:
        return {
            "n_content": self.n_content,
            "n_key": self.n_key,
            "op_codes": list(self.op_codes),
            "n_features": self.n_features,
        }


def group_ranges(n_items: int, n_groups: int) -> tuple[tuple[int, ...], ...]:
    """Partition ``range(n_items)`` into ``n_groups`` contiguous blocks."""
    if n_groups < 1 or n_items < n_groups:
        raise FeatureProgramError(f"cannot split {n_items} items into {n_groups} groups")
    return tuple(tuple(int(i) for i in block) for block in np.array_split(np.arange(n_items), n_groups))


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FeatureProgramConfig:
    """Everything that determines a dataset. Hashed verbatim into its identity."""

    family: str = "T1"
    condition: str = "capacity_stressed"
    split: str = "train"
    seed: int = 20260809
    n_examples: int = 512
    seq_len: int = 48

    n_content_features: int = 96
    n_key_features: int = 24
    n_content_groups: int = 4
    n_key_groups: int = 3
    n_keys: int = 12
    key_bits: int = 3

    activation_prob: float = 0.12
    """Per-feature activation probability *within the group a position draws
    from*. The realised global density per position is much lower and is
    reported in :meth:`FeatureProgramDataset.summary`."""
    activation_profile: str = "uniform"
    activation_alpha: float = 1.0
    min_active_per_position: int = 1
    """A supervised target of all zeros makes retrieval metrics meaningless, so
    content draws are conditioned on at least this many active features."""

    importance_profile: str = "power_law"
    importance_decay: float = 0.97

    n_associations: int = 6
    n_distractors: int = 3
    n_decoys: int = 0
    key_collisions: bool = False
    supervise_content: bool = False

    operations: tuple[str, ...] = ("recall_by_key",)
    distance_buckets: tuple[tuple[int, int], ...] = ((5, 9), (10, 16))

    source_destroyed: bool = False
    permute_features: bool = False

    holdout_fraction: float = 0.25
    d_recommended: int = 16
    """The model width this condition was designed for. Data cannot set ``d``;
    recording it here keeps "ample dimension" and "F >> d" auditable."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "operations", tuple(self.operations))
        object.__setattr__(
            self, "distance_buckets", tuple((int(a), int(b)) for a, b in self.distance_buckets)
        )
        self.validate()

    def validate(self) -> None:
        if self.n_examples < 1:
            raise FeatureProgramError("n_examples must be >= 1")
        if self.seq_len < 2:
            raise FeatureProgramError("seq_len must be >= 2")
        if self.n_content_features < 1:
            raise FeatureProgramError("n_content_features must be >= 1")
        if self.activation_profile not in ("uniform", "power_law"):
            raise FeatureProgramError(f"unknown activation_profile {self.activation_profile!r}")
        if self.importance_profile not in ("uniform", "power_law"):
            raise FeatureProgramError(f"unknown importance_profile {self.importance_profile!r}")
        if not 0.0 < self.activation_prob <= 1.0:
            raise FeatureProgramError("activation_prob must be in (0, 1]")
        if self.min_active_per_position < 1:
            raise FeatureProgramError("min_active_per_position must be >= 1")
        group_ranges(self.n_content_features, self.n_content_groups)

        if self.family == "T0":
            return

        if self.n_key_features < 1 or self.n_keys < 1:
            raise FeatureProgramError("T1 requires a key bank and at least one key")
        key_blocks = group_ranges(self.n_key_features, self.n_key_groups)
        if any(len(block) < self.key_bits + 1 for block in key_blocks):
            raise FeatureProgramError(
                f"each key group needs > key_bits={self.key_bits} indices so that a near-miss "
                f"key exists; group sizes are {[len(b) for b in key_blocks]}"
            )
        if self.n_keys < self.n_key_groups:
            raise FeatureProgramError("n_keys must be at least n_key_groups")
        if self.n_keys <= self.n_associations:
            raise FeatureProgramError(
                "n_keys must exceed n_associations so that an unused key exists for the "
                "information-destroyed negative control"
            )
        if self.n_associations < 1:
            raise FeatureProgramError("n_associations must be >= 1")
        if "recall_first_binding" in self.operations and self.n_associations < 2:
            raise FeatureProgramError(
                "recall_first_binding needs n_associations >= 2, otherwise it is identical "
                "to recall_by_key and stops being a different operation"
            )
        for low, high in self.distance_buckets:
            if not 1 <= low <= high:
                raise FeatureProgramError(f"bad distance bucket ({low}, {high})")
            if high > self.seq_len - 2:
                raise FeatureProgramError(
                    f"distance bucket ({low}, {high}) needs seq_len >= {high + 2}, "
                    f"got {self.seq_len}"
                )
            if self.n_distractors > low - 1:
                raise FeatureProgramError(
                    f"n_distractors={self.n_distractors} cannot fit strictly between a source "
                    f"and a destination {low} apart"
                )
        free_before = self.seq_len - 1 - 1 - self.n_distractors
        if self.n_associations - 1 > free_before - 1:
            raise FeatureProgramError(
                "not enough positions for the requested number of simultaneous associations"
            )

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# The six §4.4 dataset controls, as first-class named conditions
# --------------------------------------------------------------------------- #

_CAPACITY_STRESSED = FeatureProgramConfig(
    family="T1",
    condition="capacity_stressed",
    n_content_features=96,
    n_key_features=24,
    n_content_groups=4,
    n_key_groups=3,
    n_keys=12,
    key_bits=3,
    seq_len=48,
    activation_prob=0.12,
    n_associations=6,
    n_distractors=3,
    distance_buckets=((5, 9), (10, 16)),
    d_recommended=16,
)

_CONDITIONS: dict[str, FeatureProgramConfig] = {
    # Ample dimension (d > F), short distance, no distractors, one association.
    # If a mechanism cannot solve this, the instrument is broken (R1).
    "positive_control": FeatureProgramConfig(
        family="T1",
        condition="positive_control",
        n_content_features=24,
        n_key_features=8,
        n_content_groups=2,
        n_key_groups=2,
        n_keys=4,
        key_bits=2,
        seq_len=12,
        activation_prob=0.20,
        n_associations=1,
        n_distractors=0,
        distance_buckets=((1, 2),),
        d_recommended=48,
    ),
    # F >> d, sparse features, moderate interference. The working condition.
    "capacity_stressed": _CAPACITY_STRESSED,
    # Source removed: the answer exists nowhere in the input and is drawn from a
    # stream the input never touches, so I(input; target) is zero by construction.
    "negative_control": replace(
        _CAPACITY_STRESSED, condition="negative_control", source_destroyed=True
    ),
    # Operation-like markers with no semantic effect anywhere.
    "lexical_decoy": replace(_CAPACITY_STRESSED, condition="lexical_decoy", n_decoys=3),
    # Arbitrary feature IDs and route labels permuted. Results must not move.
    "permutation_control": replace(
        _CAPACITY_STRESSED, condition="permutation_control", permute_features=True
    ),
    # Bitwise-identical inputs, different required operation: ordinal addressing
    # (the earliest binding) instead of content addressing (the matching key).
    "matched_difficulty": replace(
        _CAPACITY_STRESSED,
        condition="matched_difficulty",
        operations=("recall_first_binding",),
    ),
}

CONDITION_NAMES: tuple[str, ...] = (
    "positive_control",
    "capacity_stressed",
    "negative_control",
    "lexical_decoy",
    "permutation_control",
    "matched_difficulty",
)


def condition_config(name: str, **overrides) -> FeatureProgramConfig:
    """Return a §4.4 control condition, optionally overridden.

    Four of the six (negative, decoy, permutation, matched-difficulty) are
    ``replace()`` of ``capacity_stressed``, so each is a matched comparison
    against it by construction rather than by a promise in a comment.
    """
    if name not in _CONDITIONS:
        raise FeatureProgramError(f"unknown condition {name!r}; expected one of {CONDITION_NAMES}")
    base = _CONDITIONS[name]
    return replace(base, **overrides) if overrides else base


def t0_config(**overrides) -> FeatureProgramConfig:
    """T0 local reconstruction: no key bank, no op bank, no transport."""
    base = FeatureProgramConfig(
        family="T0",
        condition="capacity_stressed",
        n_content_features=64,
        n_key_features=0,
        n_content_groups=4,
        n_key_groups=1,
        n_keys=0,
        key_bits=0,
        seq_len=16,
        activation_prob=0.20,
        n_associations=1,
        n_distractors=0,
        operations=("reconstruct",),
        distance_buckets=((0, 0),),
        d_recommended=16,
    )
    return replace(base, **overrides) if overrides else base


# --------------------------------------------------------------------------- #
# Structural plan produced by a task family (see task_families.py)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PositionPlan:
    """What a position *is*, before any feature values are drawn."""

    index: int
    role: str
    op_code: str | None
    key_id: int | None
    key_bits: tuple[int, ...]
    """Key-bank-local indices. Empty for positions with no key. Carried
    explicitly rather than looked up so that near-miss keys (which have no
    ``key_id``) are representable."""
    has_content: bool
    content_groups: tuple[int, ...]


@dataclass(frozen=True)
class StepPlan:
    """One required operation: where the answer must appear and where it lives."""

    op: str
    dest: int
    source: int | None
    key_id: int | None
    distractors: tuple[int, ...]
    answer_group: int


@dataclass(frozen=True)
class ExamplePlan:
    positions: tuple[PositionPlan, ...]
    steps: tuple[StepPlan, ...]


# --------------------------------------------------------------------------- #
# The ground-truth program record
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PositionRecord:
    index: int
    role: str
    op_code: str | None
    key_id: int | None
    key_features: tuple[int, ...]
    content_groups: tuple[int, ...]
    active_features: tuple[int, ...]

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ProgramStep:
    op: str
    dest: int
    source: int | None
    key_id: int | None
    distance: int | None
    distractors: tuple[int, ...]
    answer_group: int
    """Generator-space content-group label the answer was drawn from. A label,
    not a feature index: it is unaffected by the permutation control."""
    answer_features: tuple[int, ...]
    information_destroyed: bool

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ProgramRecord:
    """The ground-truth program behind one example.

    Feature indices are given in the coordinates of the emitted tensors, i.e.
    *after* the permutation control if one was applied. ``feature_permutation``
    maps back: ``base = permuted[..., argsort(feature_permutation)]``.
    """

    example_index: int
    family: str
    condition: str
    split: str
    template_id: str
    composition: tuple
    seq_len: int
    positions: tuple[PositionRecord, ...]
    steps: tuple[ProgramStep, ...]
    supervised_positions: tuple[int, ...]
    feature_permutation: tuple[int, ...] | None = None
    key_permutation: tuple[int, ...] | None = None

    def as_dict(self) -> dict:
        return {
            "example_index": self.example_index,
            "family": self.family,
            "condition": self.condition,
            "split": self.split,
            "template_id": self.template_id,
            "composition": list(self.composition),
            "seq_len": self.seq_len,
            "positions": [p.as_dict() for p in self.positions],
            "steps": [s.as_dict() for s in self.steps],
            "supervised_positions": list(self.supervised_positions),
            "feature_permutation": (
                list(self.feature_permutation) if self.feature_permutation is not None else None
            ),
            "key_permutation": (
                list(self.key_permutation) if self.key_permutation is not None else None
            ),
        }


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #


def activation_probs(cfg: FeatureProgramConfig) -> np.ndarray:
    """Per-content-feature activation probability."""
    n = cfg.n_content_features
    if cfg.activation_profile == "uniform":
        return np.full(n, float(cfg.activation_prob), dtype=np.float64)
    raw = (np.arange(1, n + 1, dtype=np.float64)) ** (-float(cfg.activation_alpha))
    scaled = raw * (cfg.activation_prob * n / raw.sum())
    return np.clip(scaled, 1e-6, 0.95)


def importance_weights(cfg: FeatureProgramConfig, banks: FeatureBanks) -> np.ndarray:
    """Loss weight per feature, over the full ``F`` axis.

    Targets are exactly zero outside the content bank — there is nothing to
    predict there — so key and op features carry weight ``0``.
    """
    weights = np.zeros(banks.n_features, dtype=np.float64)
    n = cfg.n_content_features
    if cfg.importance_profile == "uniform":
        weights[:n] = 1.0
    else:
        weights[:n] = float(cfg.importance_decay) ** np.arange(n, dtype=np.float64)
    return weights


def build_key_table(cfg: FeatureProgramConfig, banks: FeatureBanks) -> tuple[tuple[int, ...], ...]:
    """Key ``k`` is a fixed subset of ``key_bits`` indices inside its key group.

    Seeded from the base seed only — not from the split or the condition — so
    train and test share one key vocabulary and matched conditions share one
    table. Subsets are unique within a group so that "exact key match" is
    unambiguous and every collision is one the config asked for.
    """
    if banks.n_key == 0:
        return ()
    rng = np.random.default_rng(derive_seed(cfg.seed, "keys", cfg.n_keys, cfg.key_bits))
    blocks = group_ranges(cfg.n_key_features, cfg.n_key_groups)
    seen: set[tuple[int, ...]] = set()
    table: list[tuple[int, ...]] = []
    for key_id in range(cfg.n_keys):
        block = np.asarray(blocks[key_id % cfg.n_key_groups])
        for _ in range(_MAX_RESAMPLE):
            bits = tuple(sorted(int(b) for b in rng.choice(block, cfg.key_bits, replace=False)))
            if bits not in seen:
                break
        else:
            raise FeatureProgramError(
                f"could not find a distinct key subset for key {key_id}; widen the key groups"
            )
        seen.add(bits)
        table.append(bits)
    return tuple(table)


def near_miss_bits(
    bits: tuple[int, ...], block: tuple[int, ...], rng: np.random.Generator
) -> tuple[int, ...]:
    """A key that shares all but one index with ``bits`` — a genuine collision."""
    spare = [i for i in block if i not in bits]
    if not spare:
        raise FeatureProgramError("no spare index in the key group for a near-miss key")
    swapped = list(bits)
    swapped[int(rng.integers(0, len(swapped)))] = int(rng.choice(spare))
    return tuple(sorted(swapped))


def draw_content(
    rng: np.random.Generator,
    probs: np.ndarray,
    indices: np.ndarray,
    min_active: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw a sparse feature vector over ``indices``.

    Ported semantics from ``superposition_zoo``'s ``generate_features``
    (``src/superposition_zoo/features.py:52-61``): feature ``i`` is active
    independently with its own probability, and an active feature's magnitude is
    ``Uniform(0, 1)`` while an inactive one is exactly ``0.0``. Two deliberate
    changes: the parameter is a probability of being *active* rather than of
    being sparse, and the draw is conditioned on at least ``min_active``
    features firing (bounded rejection, then a forced activation) because an
    all-zero supervised target makes every retrieval metric undefined.

    Returns ``(values, active)``, both of length ``len(indices)``.
    """
    p = probs[indices]
    active = np.zeros(indices.size, dtype=bool)
    for _ in range(_MAX_RESAMPLE):
        active = rng.random(indices.size) < p
        if int(active.sum()) >= min_active:
            break
    else:
        active = np.zeros(indices.size, dtype=bool)
        forced = rng.choice(indices.size, size=min(min_active, indices.size), replace=False)
        active[forced] = True
    values = rng.random(indices.size) * active
    return values.astype(np.float64), active


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FeatureProgramDataset:
    """Tensors plus the ground-truth program that generated them."""

    config: FeatureProgramConfig
    banks: FeatureBanks
    split_plan: SplitPlan
    inputs: torch.Tensor
    """``(N, T, F)`` float32."""
    targets: torch.Tensor
    """``(N, T, F)`` float32, zero at unsupervised positions and outside the
    content bank."""
    target_mask: torch.Tensor
    """``(N, T)`` bool: which positions are supervised."""
    active_mask: torch.Tensor
    """``(N, T, F)`` bool: ground-truth active features of the *input*."""
    target_active_mask: torch.Tensor
    """``(N, T, F)`` bool: ground-truth active features of the *target*."""
    importance: torch.Tensor
    """``(F,)`` float32 loss weights."""
    programs: tuple[ProgramRecord, ...]
    content_indices: tuple[int, ...]
    key_indices: tuple[int, ...]
    op_indices: tuple[int, ...]
    feature_permutation: tuple[int, ...] | None
    key_permutation: tuple[int, ...] | None
    content_hash: str
    generator_version: str

    @property
    def n_examples(self) -> int:
        return int(self.inputs.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.inputs.shape[2])

    def tensors(self) -> dict[str, torch.Tensor]:
        return {
            "inputs": self.inputs,
            "targets": self.targets,
            "target_mask": self.target_mask,
            "active_mask": self.active_mask,
            "target_active_mask": self.target_active_mask,
            "importance": self.importance,
        }

    def recompute_hash(self, *, generator_version: str | None = None) -> str:
        """Recompute the content hash, optionally under a different version.

        Used to verify a loaded dataset against its manifest, and to demonstrate
        that ``GENERATOR_VERSION`` really does participate in dataset identity.
        """
        return dataset_content_hash(
            cfg=self.config,
            banks=self.banks,
            plan=self.split_plan,
            programs=self.programs,
            tensors=self.tensors(),
            generator_version=generator_version,
        )

    def summary(self) -> dict:
        supervised = int(self.target_mask.sum().item())
        per_position = self.active_mask.float().sum(dim=-1)
        return {
            "generator_version": self.generator_version,
            "family": self.config.family,
            "condition": self.config.condition,
            "split": self.config.split,
            "n_examples": self.n_examples,
            "seq_len": int(self.inputs.shape[1]),
            "n_features": self.n_features,
            "d_recommended": self.config.d_recommended,
            "F_over_d": round(self.n_features / self.config.d_recommended, 3),
            "n_supervised_positions": supervised,
            "mean_active_features_per_position": round(float(per_position.mean().item()), 3),
            "global_density": round(float(per_position.mean().item()) / self.n_features, 4),
            "n_templates_in_split": len(self.split_plan.templates_for(self.config.split)),
            "split_fingerprint": self.split_plan.fingerprint(),
            "content_hash": self.content_hash,
        }


def _permute(perm: np.ndarray, values: np.ndarray) -> np.ndarray:
    return values[..., perm]


def generate_dataset(config: FeatureProgramConfig) -> FeatureProgramDataset:
    """Generate one split of one condition, with its ground-truth program."""
    # Deferred: task_families builds on the plan types defined above, so a
    # module-level import here would be circular. This is the only direction
    # that has to be lazy; splits.py depends on nothing.
    from .task_families import get_family

    cfg = config
    family = get_family(cfg.family)
    banks = family.banks(cfg)
    templates = family.templates(cfg)
    plan = split_templates(
        templates,
        seed=cfg.seed,
        holdout_fraction=cfg.holdout_fraction,
        require_axis_coverage=family.axis_coverage,
    )
    pool = plan.templates_for(cfg.split)
    key_table = build_key_table(cfg, banks)
    probs = activation_probs(cfg)
    importance = importance_weights(cfg, banks)
    content_blocks = group_ranges(cfg.n_content_features, cfg.n_content_groups)

    n, seq_len, n_features = cfg.n_examples, cfg.seq_len, banks.n_features
    inputs = np.zeros((n, seq_len, n_features), dtype=np.float64)
    active = np.zeros((n, seq_len, n_features), dtype=bool)
    targets = np.zeros((n, seq_len, n_features), dtype=np.float64)
    target_active = np.zeros((n, seq_len, n_features), dtype=bool)
    target_mask = np.zeros((n, seq_len), dtype=bool)
    records: list[ProgramRecord] = []

    for i in range(n):
        template = pool[i % len(pool)]
        example_plan = family.plan_example(
            rng=np.random.default_rng(derive_seed(cfg.seed, "layout", cfg.split, i)),
            cfg=cfg,
            banks=banks,
            template=template,
            key_table=key_table,
            example_index=i,
        )
        records.append(
            _fill_example(
                index=i,
                cfg=cfg,
                banks=banks,
                template=template,
                example_plan=example_plan,
                key_table=key_table,
                probs=probs,
                content_blocks=content_blocks,
                inputs=inputs,
                active=active,
                targets=targets,
                target_active=target_active,
                target_mask=target_mask,
            )
        )

    feature_perm: np.ndarray | None = None
    key_perm: np.ndarray | None = None
    content_indices = banks.content_indices
    key_indices = banks.key_indices
    op_indices = banks.op_indices

    if cfg.permute_features:
        perm_rng = np.random.default_rng(derive_seed(cfg.seed, "permutation", cfg.condition))
        feature_perm = perm_rng.permutation(n_features)
        inverse = np.argsort(feature_perm)
        inputs = _permute(feature_perm, inputs)
        active = _permute(feature_perm, active)
        targets = _permute(feature_perm, targets)
        target_active = _permute(feature_perm, target_active)
        importance = importance[feature_perm]
        content_indices = tuple(int(inverse[i]) for i in banks.content_indices)
        key_indices = tuple(int(inverse[i]) for i in banks.key_indices)
        op_indices = tuple(int(inverse[i]) for i in banks.op_indices)
        key_perm = perm_rng.permutation(max(cfg.n_keys, 1))
        records = tuple(_relabel_record(r, inverse, key_perm) for r in records)  # type: ignore[assignment]

    tensors = {
        "inputs": torch.from_numpy(np.ascontiguousarray(inputs, dtype=np.float32)),
        "targets": torch.from_numpy(np.ascontiguousarray(targets, dtype=np.float32)),
        "target_mask": torch.from_numpy(np.ascontiguousarray(target_mask)),
        "active_mask": torch.from_numpy(np.ascontiguousarray(active)),
        "target_active_mask": torch.from_numpy(np.ascontiguousarray(target_active)),
        "importance": torch.from_numpy(np.ascontiguousarray(importance, dtype=np.float32)),
    }
    program_records = tuple(records)
    content_hash = dataset_content_hash(
        cfg=cfg, banks=banks, plan=plan, programs=program_records, tensors=tensors
    )

    return FeatureProgramDataset(
        config=cfg,
        banks=banks,
        split_plan=plan,
        inputs=tensors["inputs"],
        targets=tensors["targets"],
        target_mask=tensors["target_mask"],
        active_mask=tensors["active_mask"],
        target_active_mask=tensors["target_active_mask"],
        importance=tensors["importance"],
        programs=program_records,
        content_indices=content_indices,
        key_indices=key_indices,
        op_indices=op_indices,
        feature_permutation=tuple(int(v) for v in feature_perm) if feature_perm is not None else None,
        key_permutation=tuple(int(v) for v in key_perm) if key_perm is not None else None,
        content_hash=content_hash,
        generator_version=GENERATOR_VERSION,
    )


def _relabel_record(
    record: ProgramRecord, inverse: np.ndarray, key_perm: np.ndarray
) -> ProgramRecord:
    """Move a record into permuted coordinates and permute its route labels."""

    def relabel_key(key_id: int | None) -> int | None:
        return None if key_id is None else int(key_perm[key_id])

    positions = tuple(
        PositionRecord(
            index=p.index,
            role=p.role,
            op_code=p.op_code,
            key_id=relabel_key(p.key_id),
            key_features=tuple(sorted(int(inverse[f]) for f in p.key_features)),
            content_groups=p.content_groups,
            active_features=tuple(sorted(int(inverse[f]) for f in p.active_features)),
        )
        for p in record.positions
    )
    steps = tuple(
        ProgramStep(
            op=s.op,
            dest=s.dest,
            source=s.source,
            key_id=relabel_key(s.key_id),
            distance=s.distance,
            distractors=s.distractors,
            answer_group=s.answer_group,
            answer_features=tuple(sorted(int(inverse[f]) for f in s.answer_features)),
            information_destroyed=s.information_destroyed,
        )
        for s in record.steps
    )
    return replace(
        record,
        positions=positions,
        steps=steps,
        feature_permutation=tuple(int(v) for v in np.argsort(inverse)),
        key_permutation=tuple(int(v) for v in key_perm),
    )


def _fill_example(
    *,
    index: int,
    cfg: FeatureProgramConfig,
    banks: FeatureBanks,
    template: ProgramTemplate,
    example_plan: ExamplePlan,
    key_table: tuple[tuple[int, ...], ...],
    probs: np.ndarray,
    content_blocks: tuple[tuple[int, ...], ...],
    inputs: np.ndarray,
    active: np.ndarray,
    targets: np.ndarray,
    target_active: np.ndarray,
    target_mask: np.ndarray,
) -> ProgramRecord:
    """Draw features for one planned example and apply the condition transforms.

    The order here is load-bearing. The base fill consumes the ``content``
    stream in position order; source destruction and decoy injection then draw
    from their own streams and edit the result in place. That is what makes the
    negative control differ from ``capacity_stressed`` at exactly one position,
    and the decoy condition differ at exactly the decoy positions, rather than
    at every position downstream of a shifted RNG.
    """
    cfg_split = cfg.split
    content_rng = np.random.default_rng(derive_seed(cfg.seed, "content", cfg_split, index))

    positions = {
        p.index: {
            "index": p.index,
            "role": p.role,
            "op_code": p.op_code,
            "key_id": p.key_id,
            "key_features": tuple(banks.key_index(b) for b in p.key_bits),
            "content_groups": p.content_groups,
        }
        for p in example_plan.positions
    }

    for plan_position in example_plan.positions:
        t = plan_position.index
        if plan_position.op_code is not None:
            op_slot = banks.op_index(plan_position.op_code)
            inputs[index, t, op_slot] = 1.0
            active[index, t, op_slot] = True
        for local in plan_position.key_bits:
            slot = banks.key_index(local)
            inputs[index, t, slot] = 1.0
            active[index, t, slot] = True
        if plan_position.has_content:
            indices = np.concatenate(
                [np.asarray(content_blocks[g], dtype=np.int64) for g in plan_position.content_groups]
            )
            values, is_active = draw_content(
                content_rng, probs, indices, cfg.min_active_per_position
            )
            inputs[index, t, indices] = values
            active[index, t, indices] = is_active

    steps: list[ProgramStep] = []
    for step in example_plan.steps:
        target_mask[index, step.dest] = True
        if step.source is not None:
            content = np.asarray(banks.content_indices, dtype=np.int64)
            targets[index, step.dest, content] = inputs[index, step.source, content]
            target_active[index, step.dest, content] = active[index, step.source, content]
        steps.append(
            ProgramStep(
                op=step.op,
                dest=step.dest,
                source=step.source,
                key_id=step.key_id,
                distance=None if step.source is None else step.dest - step.source,
                distractors=step.distractors,
                answer_group=step.answer_group,
                answer_features=(),
                information_destroyed=False,
            )
        )

    if cfg.source_destroyed:
        steps = _destroy_sources(
            index=index,
            cfg=cfg,
            banks=banks,
            template=template,
            steps=steps,
            positions=positions,
            key_table=key_table,
            probs=probs,
            content_blocks=content_blocks,
            inputs=inputs,
            active=active,
            targets=targets,
            target_active=target_active,
        )

    if cfg.n_decoys > 0:
        _inject_decoys(
            index=index,
            cfg=cfg,
            banks=banks,
            example_plan=example_plan,
            steps=steps,
            positions=positions,
            key_table=key_table,
            inputs=inputs,
            active=active,
        )

    steps = [
        replace(
            step,
            answer_features=tuple(int(f) for f in np.nonzero(target_active[index, step.dest])[0]),
        )
        for step in steps
    ]

    position_records = tuple(
        PositionRecord(
            index=meta["index"],
            role=meta["role"],
            op_code=meta["op_code"],
            key_id=meta["key_id"],
            key_features=tuple(int(f) for f in meta["key_features"]),
            content_groups=tuple(int(g) for g in meta["content_groups"]),
            active_features=tuple(int(f) for f in np.nonzero(active[index, meta["index"]])[0]),
        )
        for meta in (positions[t] for t in sorted(positions))
    )

    return ProgramRecord(
        example_index=index,
        family=cfg.family,
        condition=cfg.condition,
        split=cfg_split,
        template_id=template.template_id,
        composition=template.composition,
        seq_len=cfg.seq_len,
        positions=position_records,
        steps=tuple(steps),
        supervised_positions=tuple(int(t) for t in np.nonzero(target_mask[index])[0]),
    )


def _destroy_sources(
    *,
    index: int,
    cfg: FeatureProgramConfig,
    banks: FeatureBanks,
    template: ProgramTemplate,
    steps: list[ProgramStep],
    positions: dict[int, dict],
    key_table: tuple[tuple[int, ...], ...],
    probs: np.ndarray,
    content_blocks: tuple[tuple[int, ...], ...],
    inputs: np.ndarray,
    active: np.ndarray,
    targets: np.ndarray,
    target_active: np.ndarray,
) -> list[ProgramStep]:
    """Make the answer genuinely unrecoverable, not merely hard.

    The source binding keeps its shape — it is still a binding, with a key and a
    value, so surface statistics barely move — but its key is one that nothing
    queries and its content is redrawn. The supervised target is then drawn from
    a stream that never touches the input. No function of the input carries any
    information about it; the best possible predictor is the marginal. That is
    what :func:`perfect_memory_oracle_report` is used to check empirically.
    """
    rng = np.random.default_rng(derive_seed(cfg.seed, "destroy", cfg.split, index))
    used = {meta["key_id"] for meta in positions.values() if meta["key_id"] is not None}
    unused = [k for k in range(cfg.n_keys) if k not in used]
    if not unused:
        raise FeatureProgramError(
            "no unused key available to overwrite a destroyed source; raise n_keys"
        )

    out: list[ProgramStep] = []
    for step in steps:
        # Local reconstruction has no source to remove: its answer is the
        # position itself, so "destroying" it would destroy the input instead of
        # the route. Only transport steps are made impossible.
        if step.source is None or step.op == "reconstruct":
            out.append(step)
            continue
        source = step.source
        meta = positions[source]

        for slot in meta["key_features"]:
            inputs[index, source, slot] = 0.0
            active[index, source, slot] = False
        replacement = int(rng.choice(unused))
        bits = tuple(banks.key_index(b) for b in key_table[replacement])
        for slot in bits:
            inputs[index, source, slot] = 1.0
            active[index, source, slot] = True
        meta["key_id"] = replacement
        meta["key_features"] = bits
        meta["role"] = "destroyed_source"

        content = np.concatenate(
            [np.asarray(content_blocks[g], dtype=np.int64) for g in meta["content_groups"]]
        )
        inputs[index, source, np.asarray(banks.content_indices, dtype=np.int64)] = 0.0
        active[index, source, np.asarray(banks.content_indices, dtype=np.int64)] = False
        values, is_active = draw_content(rng, probs, content, cfg.min_active_per_position)
        inputs[index, source, content] = values
        active[index, source, content] = is_active

        answer_indices = np.asarray(content_blocks[step.answer_group], dtype=np.int64)
        all_content = np.asarray(banks.content_indices, dtype=np.int64)
        targets[index, step.dest, all_content] = 0.0
        target_active[index, step.dest, all_content] = False
        values, is_active = draw_content(rng, probs, answer_indices, cfg.min_active_per_position)
        targets[index, step.dest, answer_indices] = values
        target_active[index, step.dest, answer_indices] = is_active

        out.append(replace(step, source=None, distance=None, information_destroyed=True))
    return out


def _inject_decoys(
    *,
    index: int,
    cfg: FeatureProgramConfig,
    banks: FeatureBanks,
    example_plan: ExamplePlan,
    steps: list[ProgramStep],
    positions: dict[int, dict],
    key_table: tuple[tuple[int, ...], ...],
    inputs: np.ndarray,
    active: np.ndarray,
) -> None:
    """Add operation-like markers that change no answer anywhere.

    A decoy carries a real key and a real op marker, so anything keying on
    surface "there is an operation here" is misled, but it carries no content
    and is never a source, so the ground-truth program is untouched. Decoys only
    ever overwrite filler positions, and they are applied after the base fill,
    so the invariant "same seed, decoys on or off, identical targets" holds.
    """
    rng = np.random.default_rng(derive_seed(cfg.seed, "decoy", cfg.split, index))
    protected = {s.dest for s in steps} | {s.source for s in steps if s.source is not None}
    fillers = sorted(
        p.index for p in example_plan.positions if p.role == "filler" and p.index not in protected
    )
    if not fillers:
        return
    chosen = rng.choice(fillers, size=min(cfg.n_decoys, len(fillers)), replace=False)
    bound_keys = sorted(
        {meta["key_id"] for meta in positions.values() if meta["key_id"] is not None}
    )
    content = np.asarray(banks.content_indices, dtype=np.int64)

    for position in sorted(int(c) for c in chosen):
        meta = positions[position]
        if meta["op_code"] is not None:
            slot = banks.op_index(meta["op_code"])
            inputs[index, position, slot] = 0.0
            active[index, position, slot] = False
        inputs[index, position, content] = 0.0
        active[index, position, content] = False
        noop = banks.op_index("NOOP")
        inputs[index, position, noop] = 1.0
        active[index, position, noop] = True
        key_id = int(rng.choice(bound_keys)) if bound_keys else None
        bits: tuple[int, ...] = ()
        if key_id is not None:
            bits = tuple(banks.key_index(b) for b in key_table[key_id])
            for slot in bits:
                inputs[index, position, slot] = 1.0
                active[index, position, slot] = True
        meta.update(role="decoy_op", op_code="NOOP", key_id=key_id, key_features=bits,
                    content_groups=())


def dataset_content_hash(
    *,
    cfg: FeatureProgramConfig,
    banks: FeatureBanks,
    plan: SplitPlan,
    programs: tuple[ProgramRecord, ...],
    tensors: dict[str, torch.Tensor],
    generator_version: str | None = None,
) -> str:
    """Identity of a dataset: version, config, split, tensors, and program.

    The generator version is hashed first and unconditionally, so a semantics
    change makes every previously recorded hash fail to match rather than
    silently agreeing with data it no longer describes. It is a parameter rather
    than only a module global so that "the version really is in the hash" is a
    property a test can demonstrate instead of read off the source.
    """
    digest = hashlib.sha256()
    digest.update(
        canonical_json(
            {
                "generator_version": generator_version or GENERATOR_VERSION,
                "config": cfg.as_dict(),
                "banks": banks.as_dict(),
                "split_fingerprint": plan.fingerprint(),
                "split_template_ids": [t.template_id for t in plan.templates_for(cfg.split)],
            }
        ).encode()
    )
    for name in sorted(tensors):
        tensor = tensors[name]
        digest.update(name.encode())
        digest.update(
            canonical_json({"dtype": str(tensor.dtype), "shape": list(tensor.shape)}).encode()
        )
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    for record in programs:
        digest.update(canonical_json(record.as_dict()).encode())
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# The perfect-memory oracle
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class OracleReport:
    """What the best boring strategy achieves, per strategy.

    Scores are ``R^2 = 1 - SSE(strategy) / SSE(marginal)`` on a held-out half,
    with the marginal (the best constant predictor) fitted on the other half.
    Chance is therefore exactly ``0.0`` and beating chance means beating the
    marginal. Only supervised positions are scored, and only over the content
    bank, where the targets actually live.
    """

    condition: str
    family: str
    n_supervised: int
    n_fit: int
    n_eval: int
    scores: dict[str, float]
    best_honest_strategy: str
    best_honest_r2: float
    hindsight_upper_bound_r2: float
    """A per-example oracle that picks, with hindsight, the position closest to
    the answer. It is selection-biased upward by construction and is reported
    for context only — never gated on."""

    def as_dict(self) -> dict:
        return asdict(self)

    def table(self) -> str:
        rows = [f"  {'strategy':<26} {'R^2':>9}"]
        for name, value in sorted(self.scores.items(), key=lambda kv: -kv[1]):
            rows.append(f"  {name:<26} {value:>9.4f}")
        rows.append(f"  {'(hindsight, biased)':<26} {self.hindsight_upper_bound_r2:>9.4f}")
        return "\n".join(rows)


def perfect_memory_oracle_report(
    dataset: FeatureProgramDataset,
    *,
    fit_fraction: float = 0.5,
    max_offset: int = 24,
    max_pairs: int = 4096,
) -> OracleReport:
    """Score every boring strategy an oracle with perfect memory could use.

    The oracle sees the entire input sequence — every position, every feature,
    no bottleneck, no forgetting. If it cannot beat the marginal, the answer is
    not in the input. That is the operational meaning of "genuinely impossible",
    and it is why the negative control is checked with this rather than with an
    assertion that some model failed to learn it.

    The strategy family covers everything the task admits: constants, copying
    any fixed offset, averaging the context, exact key match, and nearest key
    match. A real negative control must defeat all of them.
    """
    inputs = dataset.inputs.numpy()
    targets = dataset.targets.numpy()
    mask = dataset.target_mask.numpy()
    content = np.asarray(dataset.content_indices, dtype=np.int64)
    keys = np.asarray(dataset.key_indices, dtype=np.int64)

    example_index, dest = np.nonzero(mask)
    if example_index.size == 0:
        raise FeatureProgramError("dataset has no supervised positions to score")
    if example_index.size > max_pairs:
        stride = int(np.ceil(example_index.size / max_pairs))
        example_index, dest = example_index[::stride], dest[::stride]

    n_pairs = example_index.size
    y = targets[example_index, dest][:, content]
    xc = inputs[example_index][:, :, content]
    xk = inputs[example_index][:, :, keys] if keys.size else np.zeros((n_pairs, inputs.shape[1], 0))
    has_content = xc.any(axis=-1)

    n_examples = dataset.n_examples
    fit_examples = max(1, round(fit_fraction * n_examples))
    is_fit = example_index < fit_examples
    is_eval = ~is_fit
    if not is_fit.any() or not is_eval.any():
        raise FeatureProgramError("fit_fraction leaves one side of the oracle split empty")

    marginal = y[is_fit].mean(axis=0)

    def sse(prediction: np.ndarray) -> float:
        residual = prediction[is_eval] - y[is_eval]
        return float((residual * residual).sum())

    baseline = sse(np.broadcast_to(marginal, y.shape))
    if baseline <= 0.0:
        raise FeatureProgramError("degenerate targets: the marginal predictor is exact")

    def r2(prediction: np.ndarray) -> float:
        return 1.0 - sse(prediction) / baseline

    def with_fallback(prediction: np.ndarray, valid: np.ndarray) -> np.ndarray:
        out = np.broadcast_to(marginal, y.shape).copy()
        out[valid] = prediction[valid]
        return out

    rows = np.arange(n_pairs)
    scores: dict[str, float] = {"marginal_constant": r2(np.broadcast_to(marginal, y.shape))}
    scores["zeros"] = r2(np.zeros_like(y))
    scores["copy_at_dest"] = r2(xc[rows, dest])

    for offset in range(1, min(max_offset, inputs.shape[1] - 1) + 1):
        source = dest - offset
        valid = source >= 0
        picked = xc[rows, np.clip(source, 0, None)]
        scores[f"copy_offset_{offset}"] = r2(with_fallback(picked, valid))

    # Ordinal strategies as well as content-addressed ones. Without them the
    # battery cannot tell a genuinely impossible control from one whose answer
    # is merely addressed by position, and both would look like chance.
    names = (
        "mean_prev_content",
        "copy_first_content",
        "copy_first_keyed",
        "copy_last_keyed",
        "key_match_exact",
        "key_match_nearest",
    )
    predictions = {name: np.zeros_like(y) for name in names}
    valid = {name: np.zeros(n_pairs, dtype=bool) for name in names}

    def take(name: str, p: int, position: int) -> None:
        predictions[name][p] = xc[p, position]
        valid[name][p] = True

    for p in range(n_pairs):
        limit = dest[p]
        candidates = np.nonzero(has_content[p, :limit])[0]
        if candidates.size == 0:
            continue
        predictions["mean_prev_content"][p] = xc[p, candidates].mean(axis=0)
        valid["mean_prev_content"][p] = True
        take("copy_first_content", p, candidates[0])
        if not keys.size:
            continue

        keyed = candidates[xk[p, candidates].any(axis=-1)]
        if keyed.size:
            take("copy_first_keyed", p, keyed[0])
            take("copy_last_keyed", p, keyed[-1])

        query = xk[p, limit]
        candidate_keys = xk[p, candidates]
        hits = candidates[np.all(candidate_keys == query, axis=-1)]
        if hits.size:
            take("key_match_exact", p, hits[-1])
        norms = np.linalg.norm(candidate_keys, axis=-1) * np.linalg.norm(query)
        if norms.max() > 0:
            similarity = np.where(
                norms > 0, (candidate_keys @ query) / np.maximum(norms, 1e-12), -1.0
            )
            take("key_match_nearest", p, candidates[int(np.argmax(similarity))])

    for name in names:
        scores[name] = r2(with_fallback(predictions[name], valid[name]))

    hindsight = np.zeros_like(y)
    for p in range(n_pairs):
        errors = ((xc[p] - y[p]) ** 2).sum(axis=-1)
        hindsight[p] = xc[p, int(np.argmin(errors))]
    hindsight_r2 = r2(hindsight)

    honest = {k: v for k, v in scores.items() if k != "marginal_constant"}
    best = max(honest, key=lambda k: honest[k])
    return OracleReport(
        condition=dataset.config.condition,
        family=dataset.config.family,
        n_supervised=int(mask.sum()),
        n_fit=int(is_fit.sum()),
        n_eval=int(is_eval.sum()),
        scores={k: round(v, 6) for k, v in scores.items()},
        best_honest_strategy=best,
        best_honest_r2=round(honest[best], 6),
        hindsight_upper_bound_r2=round(hindsight_r2, 6),
    )


def answer_appears_in_input(dataset: FeatureProgramDataset) -> int:
    """Count supervised positions whose exact answer vector sits in the input.

    Zero is the structural half of the negative control: the oracle report says
    no strategy recovers the answer, and this says the answer is not physically
    present to be recovered.
    """
    content = np.asarray(dataset.content_indices, dtype=np.int64)
    inputs = dataset.inputs.numpy()[:, :, content]
    targets = dataset.targets.numpy()[:, :, content]
    mask = dataset.target_mask.numpy()
    found = 0
    for example, dest in zip(*np.nonzero(mask), strict=True):
        answer = targets[example, dest]
        if np.any(np.all(np.isclose(inputs[example], answer), axis=-1)):
            found += 1
    return found


# --------------------------------------------------------------------------- #
# Human-readable inspection
# --------------------------------------------------------------------------- #


def format_example(dataset: FeatureProgramDataset, index: int = 0) -> str:
    """Print one example beside its ground-truth program, for eyeballing.

    Automated invariants do not catch an off-by-one in the semantics you
    intended, so every mission that touches the generator should read one of
    these by hand.
    """
    record = dataset.programs[index]
    content = set(dataset.content_indices)
    key_set = set(dataset.key_indices)
    lines = [
        (
            f"condition={record.condition} family={record.family} split={record.split} "
            f"example={record.example_index}"
        ),
        f"template={record.template_id} composition={record.composition}",
        f"seq_len={record.seq_len} F={dataset.n_features} d_recommended={dataset.config.d_recommended}",
        "",
        f"{'pos':>4} {'role':<17} {'op':<8} {'key':>4}  {'key feats':<14} content features (value)",
    ]
    step_by_dest = {s.dest: s for s in record.steps}
    for position in record.positions:
        marks = []
        if position.index in step_by_dest:
            marks.append("<-- DEST")
        for step in record.steps:
            if step.source == position.index:
                marks.append("<-- SOURCE")
            if position.index in step.distractors:
                marks.append("<-- distractor")
        content_features = [f for f in position.active_features if f in content]
        values = ", ".join(
            f"{f}:{dataset.inputs[index, position.index, f].item():.2f}" for f in content_features
        )
        key_features = [f for f in position.active_features if f in key_set]
        lines.append(
            f"{position.index:>4} {position.role:<17} {position.op_code or '-'!s:<8} "
            f"{position.key_id if position.key_id is not None else '-'!s:>4}  "
            f"{key_features!s:<14} {values or '-'}   {' '.join(marks)}"
        )

    lines.append("")
    for step in record.steps:
        lines.append(
            f"step: op={step.op} dest={step.dest} source={step.source} key={step.key_id} "
            f"distance={step.distance} distractors={list(step.distractors)} "
            f"destroyed={step.information_destroyed}"
        )
        lines.append(f"      answer features (ground truth): {list(step.answer_features)}")
        target = dataset.targets[index, step.dest]
        nonzero = [int(f) for f in torch.nonzero(target).flatten().tolist()]
        lines.append(f"      target tensor nonzero indices:   {nonzero}")
        if step.source is not None:
            source_content = dataset.inputs[index, step.source].clone()
            source_content[[f for f in range(dataset.n_features) if f not in content]] = 0.0
            agrees = torch.allclose(source_content, target)
            lines.append(f"      target == source content bank:   {agrees}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Superposition phase-diagram grid (for the prompt-14 figure)
# --------------------------------------------------------------------------- #


def phase_diagram_grid(
    *,
    sparsities: tuple[float, ...] = (0.02, 0.05, 0.10, 0.20, 0.40),
    f_over_d: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0),
    n_content_features: int = 64,
    base: FeatureProgramConfig | None = None,
    **overrides,
) -> tuple[FeatureProgramConfig, ...]:
    """The sparsity x F/d configuration grid for the superposition phase diagram.

    ``sparsity`` here is the per-feature activation probability, and ``F/d`` is
    computed against the full feature axis, so the returned ``d_recommended``
    is what a run must actually use for the cell to mean what it says. Built on
    T0 because the phase diagram is about packing, not transport.
    """
    template = base if base is not None else t0_config(n_content_features=n_content_features)
    total_features = template.n_content_features + template.n_key_features
    grid: list[FeatureProgramConfig] = []
    for ratio in f_over_d:
        width = max(1, round(total_features / ratio))
        for probability in sparsities:
            grid.append(
                replace(
                    template,
                    activation_prob=probability,
                    d_recommended=width,
                    **overrides,
                )
            )
    return tuple(grid)


# --------------------------------------------------------------------------- #
# Selftest
# --------------------------------------------------------------------------- #

INVARIANTS: tuple[str, ...] = (
    "t0_reconstruction_oracle",
    "positive_control_oracle",
    "positive_control_addressing_is_ordinal",
    "capacity_stressed_oracle",
    "negative_control_oracle",
    "negative_control_answer_absent",
    "decoy_has_no_semantic_effect",
    "permutation_is_an_isomorphism",
    "matched_difficulty_shares_inputs",
    "splits_are_disjoint",
    "hash_is_stable_and_version_sensitive",
)

_SELFTEST_EXAMPLES = 768
_NEGATIVE_CONTROL_TOLERANCE = 0.05
_POSITIVE_CONTROL_FLOOR = 0.95


class _Checks:
    """Collects pass/fail with a reason, so the selftest reports all failures."""

    def __init__(self, broken: str | None = None) -> None:
        self.broken = broken
        self.results: list[tuple[str, bool, str]] = []

    def record(self, name: str, ok: bool, detail: str) -> None:
        if name == self.broken:
            ok = False
            detail = f"deliberately broken by --break-invariant: {detail}"
        self.results.append((name, ok, detail))

    @property
    def failed(self) -> list[tuple[str, bool, str]]:
        return [r for r in self.results if not r[1]]


def run_selftest(*, break_invariant: str | None = None, verbose: bool = True) -> int:
    """Generate every control condition and assert the invariants that matter.

    This is the gate for prompt 02 and the smoke test for every later mission:
    if the generator's semantics drift, this fails before any experiment runs.
    """
    checks = _Checks(break_invariant)
    out: list[str] = [f"feature program selftest — {GENERATOR_VERSION}", ""]

    datasets: dict[str, FeatureProgramDataset] = {}
    for name in CONDITION_NAMES:
        datasets[name] = generate_dataset(condition_config(name, n_examples=_SELFTEST_EXAMPLES))
    datasets["T0"] = generate_dataset(t0_config(n_examples=256))

    out.append(
        f"{'condition':<21} {'fam':<4} {'F':>4} {'d*':>4} {'F/d':>6} {'T':>4} {'sup':>6} "
        f"{'act/pos':>8}  hash"
    )
    for name, dataset in datasets.items():
        summary = dataset.summary()
        out.append(
            f"{name:<21} {summary['family']:<4} {summary['n_features']:>4} "
            f"{summary['d_recommended']:>4} {summary['F_over_d']:>6} {summary['seq_len']:>4} "
            f"{summary['n_supervised_positions']:>6} "
            f"{summary['mean_active_features_per_position']:>8}  {summary['content_hash'][:16]}"
        )
    out.append("")

    oracles = {name: perfect_memory_oracle_report(ds) for name, ds in datasets.items()}

    report = oracles["T0"]
    checks.record(
        "t0_reconstruction_oracle",
        report.scores["copy_at_dest"] >= 0.99,
        f"copy_at_dest R^2 = {report.scores['copy_at_dest']:.4f} (want >= 0.99)",
    )

    report = oracles["positive_control"]
    checks.record(
        "positive_control_oracle",
        report.scores["key_match_exact"] >= _POSITIVE_CONTROL_FLOOR,
        f"key_match_exact R^2 = {report.scores['key_match_exact']:.4f} "
        f"(want >= {_POSITIVE_CONTROL_FLOOR})",
    )

    # The positive control carries exactly one binding (`n_associations=1`), so
    # "return the value bound to this key" and "return the value at the one
    # keyed position" are the same instruction. Recorded as a checked property
    # rather than left implicit: R1 validates that the mixer can *move* a marked
    # position's content, not that it addresses by content. If a later mission
    # makes the positive control harder, this check says so instead of letting
    # the scope of R1 change silently.
    ordinal_pc = report.scores["copy_first_keyed"]
    checks.record(
        "positive_control_addressing_is_ordinal",
        ordinal_pc >= _POSITIVE_CONTROL_FLOOR,
        f"copy_first_keyed R^2 = {ordinal_pc:.4f} — with n_associations=1 the ordinal "
        f"strategy solves the positive control outright, so R1 tests transport and not "
        f"content addressing",
    )

    # The condition every recorded T1 run is trained and scored on, held to the
    # same absolute bar as the positive control — and held to it by the *only*
    # route in this laboratory that does not consult `step.source`.
    #
    # Without this the routing variable is untested here: `oracles[
    # "capacity_stressed"]` was computed and used only in the permutation
    # equality check below, which compares two conditions that share the routing
    # and therefore cancels any error in it. A deliberate off-by-one in the
    # source position left the program oracle at 1.0000, both selftests green,
    # and the recorded R1 dataset hash unchanged; `key_match_exact` here reads
    # -0.9479 under the same mutation. See state/10_instrument_review.md.
    report = oracles["capacity_stressed"]
    checks.record(
        "capacity_stressed_oracle",
        report.scores["key_match_exact"] >= _POSITIVE_CONTROL_FLOOR,
        f"key_match_exact R^2 = {report.scores['key_match_exact']:.4f} "
        f"(want >= {_POSITIVE_CONTROL_FLOOR}); this is the one check on the pilot condition "
        f"that never reads step.source",
    )

    report = oracles["negative_control"]
    checks.record(
        "negative_control_oracle",
        report.best_honest_r2 <= _NEGATIVE_CONTROL_TOLERANCE,
        f"best honest strategy '{report.best_honest_strategy}' R^2 = {report.best_honest_r2:.4f} "
        f"(want <= {_NEGATIVE_CONTROL_TOLERANCE})",
    )
    leaks = answer_appears_in_input(datasets["negative_control"])
    checks.record(
        "negative_control_answer_absent",
        leaks == 0,
        f"{leaks} supervised answers found verbatim in their own input",
    )

    base = datasets["capacity_stressed"]
    decoy = datasets["lexical_decoy"]
    decoy_positions = {
        (r.example_index, p.index) for r in decoy.programs for p in r.positions if p.role == "decoy_op"
    }
    targets_match = torch.equal(base.targets, decoy.targets) and torch.equal(
        base.target_mask, decoy.target_mask
    )
    differing = torch.nonzero((base.inputs != decoy.inputs).any(dim=-1)).tolist()
    only_at_decoys = all((int(e), int(t)) in decoy_positions for e, t in differing)
    checks.record(
        "decoy_has_no_semantic_effect",
        targets_match and only_at_decoys and len(decoy_positions) > 0,
        f"{len(decoy_positions)} decoy positions; targets identical={targets_match}; "
        f"{len(differing)} differing positions, all decoys={only_at_decoys}",
    )

    permuted = datasets["permutation_control"]
    inverse = np.argsort(np.asarray(permuted.feature_permutation))
    unpermuted = permuted.inputs[..., inverse]
    isomorphic = torch.equal(unpermuted, base.inputs) and torch.equal(
        permuted.targets[..., inverse], base.targets
    )
    oracle_matches = all(
        abs(oracles["permutation_control"].scores[k] - oracles["capacity_stressed"].scores[k]) < 1e-6
        for k in oracles["capacity_stressed"].scores
    )
    checks.record(
        "permutation_is_an_isomorphism",
        isomorphic and oracle_matches,
        f"unpermuted tensors match={isomorphic}; every oracle score matches={oracle_matches}",
    )

    matched = datasets["matched_difficulty"]
    matched_oracle = oracles["matched_difficulty"]
    inputs_identical = torch.equal(base.inputs, matched.inputs)
    targets_differ = not torch.equal(base.targets, matched.targets)
    ordinal = matched_oracle.scores["copy_first_keyed"]
    content_addressed = matched_oracle.scores["key_match_exact"]
    ops = {s.op for r in matched.programs for s in r.steps}
    checks.record(
        "matched_difficulty_shares_inputs",
        inputs_identical and targets_differ and ordinal >= 0.99 and content_addressed < 0.0,
        f"inputs bitwise identical={inputs_identical}; targets differ={targets_differ}; "
        f"required operation={sorted(ops)}; solvable by ordinal addressing "
        f"(R^2={ordinal:.4f}) and not by content addressing (R^2={content_addressed:.4f})",
    )

    split_report = base.split_plan.report()
    t0_report = datasets["T0"].split_plan.report()
    checks.record(
        "splits_are_disjoint",
        split_report["template_id_overlap"] == 0
        and t0_report["template_id_overlap"] == 0
        and split_report["compositional"]
        and split_report["n_heldout_compositions"] > 0,
        f"T1: {split_report['n_train_templates']} train / {split_report['n_test_templates']} test "
        f"templates, overlap {split_report['template_id_overlap']}, "
        f"{split_report['n_heldout_compositions']} held-out compositions, "
        f"compositional={split_report['compositional']}; "
        f"T0: {t0_report['n_heldout_compositions']} held-out compositions",
    )

    again = generate_dataset(condition_config("capacity_stressed", n_examples=_SELFTEST_EXAMPLES))
    reseeded = generate_dataset(
        condition_config("capacity_stressed", n_examples=_SELFTEST_EXAMPLES, seed=base.config.seed + 1)
    )
    bumped_hash = base.recompute_hash(generator_version=GENERATOR_VERSION + "-probe")
    checks.record(
        "hash_is_stable_and_version_sensitive",
        again.content_hash == base.content_hash
        and base.recompute_hash() == base.content_hash
        and bumped_hash != base.content_hash
        and reseeded.content_hash != base.content_hash
        and not torch.equal(reseeded.inputs, base.inputs),
        f"regenerated={again.content_hash[:16]} same={again.content_hash == base.content_hash}; "
        f"version-bumped differs={bumped_hash != base.content_hash}; "
        f"reseeded differs={reseeded.content_hash != base.content_hash}",
    )

    out.append("invariants")
    for name, ok, detail in checks.results:
        out.append(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    out.append("")
    out.append("oracle bound, negative control (chance == 0.0)")
    out.append(oracles["negative_control"].table())
    out.append("")
    out.append("oracle bound, positive control")
    out.append(oracles["positive_control"].table())
    out.append("")
    verdict = "selftest PASSED" if not checks.failed else f"selftest FAILED ({len(checks.failed)})"
    out.append(verdict)

    if verbose:
        print("\n".join(out))
    return 0 if not checks.failed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="architecture_mechanics.data.feature_program")
    parser.add_argument("--selftest", action="store_true", help="run the invariant gate")
    parser.add_argument(
        "--break-invariant",
        choices=INVARIANTS,
        default=None,
        help="force one invariant to fail; used to prove the gate reports failure",
    )
    parser.add_argument("--show-example", metavar="CONDITION", default=None,
                        help="print one example with its ground-truth program")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args(argv)

    if args.show_example:
        config = (
            t0_config(n_examples=32)
            if args.show_example == "T0"
            else condition_config(args.show_example, n_examples=64)
        )
        print(format_example(generate_dataset(config), args.index))
        return 0

    if args.selftest or args.break_invariant:
        return run_selftest(break_invariant=args.break_invariant)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
