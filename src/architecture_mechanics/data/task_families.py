"""Task families T0–T5 (§4.3). T0 and T1 are implemented; T2–T5 are stubs.

A task family owns *structure* only: which position plays which role, which
position is the source for which destination, and what operation is required.
It never draws a feature value. Feature sampling, the six §4.4 conditions, the
program record, hashing, and the oracle all live in
:mod:`architecture_mechanics.data.feature_program`, which calls into here.

That split is why adding T2 later is a small change: a new family declares its
positions and steps, and inherits the controls, the splits, the determinism, and
the oracle unchanged.
"""

from __future__ import annotations

import numpy as np

from .feature_program import (
    OP_CODES,
    ExamplePlan,
    FeatureBanks,
    FeatureProgramConfig,
    FeatureProgramError,
    PositionPlan,
    StepPlan,
    build_key_table,
    near_miss_bits,
)
from .splits import ProgramTemplate, build_templates


class TaskFamily:
    """Interface every family implements. Subclasses declare structure only."""

    name: str = ""
    axis_coverage: bool = True
    """Whether the family's template grid has enough varying axes to support a
    coverage-preserving (i.e. genuinely compositional) held-out split."""

    def banks(self, cfg: FeatureProgramConfig) -> FeatureBanks:
        raise NotImplementedError

    def templates(self, cfg: FeatureProgramConfig) -> tuple[ProgramTemplate, ...]:
        raise NotImplementedError

    def plan_example(
        self,
        *,
        rng: np.random.Generator,
        cfg: FeatureProgramConfig,
        banks: FeatureBanks,
        template: ProgramTemplate,
        key_table: tuple[tuple[int, ...], ...],
        example_index: int,
    ) -> ExamplePlan:
        raise NotImplementedError


class T0LocalReconstruction(TaskFamily):
    """T0 — local reconstruction. No sequence transport is required.

    Every position carries content drawn from a *pair* of feature groups and is
    supervised on itself. There is no key bank and no op bank, so ``F`` is
    exactly the content bank and the task is the classic superposition
    autoencoder repeated along a sequence axis.

    The group pair is what makes T0's split meaningful: a held-out composition
    is a pair of feature groups that never co-occurred in training, which is
    precisely the interference question the phase diagram is about. §4.3's
    stated purposes — establish the phase diagram, measure base packing, provide
    a known-easy positive control, and catch architecture bugs before any
    sequence behaviour is tested — all need the transport-free case to be exact.
    """

    name = "T0"

    def banks(self, cfg: FeatureProgramConfig) -> FeatureBanks:
        return FeatureBanks(n_content=cfg.n_content_features, n_key=0, op_codes=())

    def templates(self, cfg: FeatureProgramConfig) -> tuple[ProgramTemplate, ...]:
        groups = tuple(range(cfg.n_content_groups))
        return build_templates(
            family=self.name,
            operations=cfg.operations,
            content_groups=groups,
            content_groups_b=groups,
            key_groups=(0,),
            distance_buckets=cfg.distance_buckets,
            n_distractors=0,
            n_associations=1,
            key_collisions=False,
        )

    def plan_example(
        self,
        *,
        rng: np.random.Generator,
        cfg: FeatureProgramConfig,
        banks: FeatureBanks,
        template: ProgramTemplate,
        key_table: tuple[tuple[int, ...], ...],
        example_index: int,
    ) -> ExamplePlan:
        # `is not None`, not `or`: group 0 is a real group and a falsy value.
        second = (
            template.content_group_b
            if template.content_group_b is not None
            else template.content_group
        )
        groups = tuple(sorted({template.content_group, second}))
        positions = tuple(
            PositionPlan(
                index=t,
                role="content",
                op_code=None,
                key_id=None,
                key_bits=(),
                has_content=True,
                content_groups=groups,
            )
            for t in range(cfg.seq_len)
        )
        steps = tuple(
            StepPlan(
                op="reconstruct",
                dest=t,
                source=t,
                key_id=None,
                distractors=(),
                answer_group=template.content_group,
            )
            for t in range(cfg.seq_len)
        )
        return ExamplePlan(positions=positions, steps=steps)


class T1AssociativeRecall(TaskFamily):
    """T1 — associative recall. Earlier positions bind keys to values.

    Layout, with the query pinned to the last position so that distance means
    exactly one thing:

        [ fillers and other bindings ][ SOURCE ][ distractors ][ QUERY ]
                                       ^                          ^
                                       q - distance               q = T-1

    §4.3's five knobs map to config fields directly: ``distance_buckets``
    (source distance, and a template axis so it can be held out),
    ``n_distractors``, ``activation_prob`` (feature sparsity), ``key_collisions``
    (a near-miss key sharing all but one index with the query key), and
    ``n_associations`` (simultaneous associations).

    Two operations are supported, and they differ in *addressing mode* rather
    than in surface statistics:

    ``recall_by_key``
        Return the value bound to the key the query carries. Content-addressed.

    ``recall_first_binding``
        Return the value of the earliest binding in the sequence. Ordinal. This
        is the §4.4 matched-difficulty control: the layout, the draws, and hence
        the inputs are bitwise identical to ``recall_by_key``, and only the
        required function changes. At least one non-source binding is always
        placed strictly before the source, so the two operations never agree.

    Multiple simultaneous *queries* per example are a documented extension, not
    a stub: §4.3 names simultaneous associations, which ``n_associations``
    already provides, and one supervised query per example keeps distance,
    distractor count, and the answer unambiguous per example.
    """

    name = "T1"

    def banks(self, cfg: FeatureProgramConfig) -> FeatureBanks:
        return FeatureBanks(
            n_content=cfg.n_content_features, n_key=cfg.n_key_features, op_codes=OP_CODES
        )

    def templates(self, cfg: FeatureProgramConfig) -> tuple[ProgramTemplate, ...]:
        return build_templates(
            family=self.name,
            operations=cfg.operations,
            content_groups=tuple(range(cfg.n_content_groups)),
            content_groups_b=None,
            key_groups=tuple(range(cfg.n_key_groups)),
            distance_buckets=cfg.distance_buckets,
            n_distractors=cfg.n_distractors,
            n_associations=cfg.n_associations,
            key_collisions=cfg.key_collisions,
        )

    def plan_example(
        self,
        *,
        rng: np.random.Generator,
        cfg: FeatureProgramConfig,
        banks: FeatureBanks,
        template: ProgramTemplate,
        key_table: tuple[tuple[int, ...], ...],
        example_index: int,
    ) -> ExamplePlan:
        query = cfg.seq_len - 1
        distance = int(rng.integers(template.distance_min, template.distance_max + 1))
        source = query - distance
        if source < 1:
            raise FeatureProgramError(
                f"distance {distance} leaves no room before the source in a sequence of "
                f"{cfg.seq_len}"
            )

        between = list(range(source + 1, query))
        n_distractors = min(cfg.n_distractors, len(between))
        distractors = (
            tuple(sorted(int(t) for t in rng.choice(between, n_distractors, replace=False)))
            if n_distractors
            else ()
        )

        taken = {source, query, *distractors}
        free = [t for t in range(query) if t not in taken]
        others: list[int] = []
        if cfg.n_associations > 1:
            before_source = [t for t in free if t < source]
            if not before_source:
                raise FeatureProgramError(
                    "no position before the source is free for another binding; "
                    "recall_first_binding would collapse onto recall_by_key"
                )
            first = int(rng.choice(before_source))
            others.append(first)
            rest = [t for t in free if t != first]
            wanted = cfg.n_associations - 2
            if wanted > len(rest):
                raise FeatureProgramError("not enough free positions for the requested bindings")
            others.extend(int(t) for t in rng.choice(rest, wanted, replace=False))
        others = sorted(others)

        group_keys = [k for k in range(cfg.n_keys) if k % cfg.n_key_groups == template.key_group]
        if not group_keys:
            raise FeatureProgramError(f"no key belongs to key group {template.key_group}")
        query_key = int(rng.choice(group_keys))
        pool = [k for k in range(cfg.n_keys) if k != query_key]
        other_keys = [int(k) for k in rng.choice(pool, len(others), replace=False)]

        near_miss: tuple[int, ...] | None = None
        if cfg.key_collisions and others:
            blocks = np.array_split(np.arange(cfg.n_key_features), cfg.n_key_groups)
            block = tuple(int(i) for i in blocks[query_key % cfg.n_key_groups])
            near_miss = near_miss_bits(key_table[query_key], block, rng)

        fillers = sorted(t for t in range(query) if t not in taken and t not in set(others))
        content_positions = sorted([*others, *distractors, *fillers])
        drawn_groups = rng.integers(0, cfg.n_content_groups, size=len(content_positions))
        group_of = {t: int(g) for t, g in zip(content_positions, drawn_groups, strict=True)}

        positions: list[PositionPlan] = []
        for t in range(cfg.seq_len):
            if t == query:
                positions.append(
                    PositionPlan(
                        index=t,
                        role="query",
                        op_code="QUERY",
                        key_id=query_key,
                        key_bits=key_table[query_key],
                        has_content=False,
                        content_groups=(),
                    )
                )
            elif t == source:
                positions.append(
                    PositionPlan(
                        index=t,
                        role="source_binding",
                        op_code="BIND",
                        key_id=query_key,
                        key_bits=key_table[query_key],
                        has_content=True,
                        content_groups=(template.content_group,),
                    )
                )
            elif t in set(others):
                slot = others.index(t)
                is_collider = near_miss is not None and slot == len(others) - 1
                positions.append(
                    PositionPlan(
                        index=t,
                        role="collision_binding" if is_collider else "binding",
                        op_code="BIND",
                        key_id=None if is_collider else other_keys[slot],
                        key_bits=near_miss if is_collider else key_table[other_keys[slot]],
                        has_content=True,
                        content_groups=(group_of[t],),
                    )
                )
            else:
                positions.append(
                    PositionPlan(
                        index=t,
                        role="distractor" if t in distractors else "filler",
                        op_code="CONTENT",
                        key_id=None,
                        key_bits=(),
                        has_content=True,
                        content_groups=(group_of[t],),
                    )
                )

        if template.operation == "recall_by_key":
            answer_source, answer_key = source, query_key
            answer_group = template.content_group
        elif template.operation == "recall_first_binding":
            bindings = sorted([source, *others])
            answer_source = bindings[0]
            answer_key = query_key if answer_source == source else other_keys[others.index(answer_source)]
            answer_group = (
                template.content_group if answer_source == source else group_of[answer_source]
            )
        else:
            raise FeatureProgramError(f"T1 does not implement operation {template.operation!r}")

        steps = [
            StepPlan(
                op=template.operation,
                dest=query,
                source=answer_source,
                key_id=answer_key,
                distractors=tuple(d for d in distractors if answer_source < d < query),
                answer_group=answer_group,
            )
        ]
        if cfg.supervise_content:
            steps.extend(
                StepPlan(
                    op="reconstruct",
                    dest=p.index,
                    source=p.index,
                    key_id=p.key_id,
                    distractors=(),
                    answer_group=p.content_groups[0],
                )
                for p in positions
                if p.has_content
            )

        return ExamplePlan(positions=tuple(positions), steps=tuple(steps))


class _PlannedFamily(TaskFamily):
    """A documented stub: the design is fixed, the implementation is not here."""

    prompt: str = ""

    def _refuse(self) -> FeatureProgramError:
        return FeatureProgramError(
            f"task family {self.name} is a documented stub; it is implemented by "
            f"prompt {self.prompt}. See §4.3 and this class's docstring for the design."
        )

    def banks(self, cfg: FeatureProgramConfig) -> FeatureBanks:
        raise NotImplementedError(str(self._refuse()))

    def templates(self, cfg: FeatureProgramConfig) -> tuple[ProgramTemplate, ...]:
        raise NotImplementedError(str(self._refuse()))

    def plan_example(self, **kwargs) -> ExamplePlan:
        raise NotImplementedError(str(self._refuse()))


class T2OverwriteAndCorrection(_PlannedFamily):
    """T2 — overwrite and correction. Owned by prompt 18.

    The same key is bound to an old value and later to a corrected value; the
    query must return the newest lawful value. Design already fixed by §4.3:
    two ``BIND`` positions share one ``key_id``, the record's step names the
    *later* binding as source and the earlier one as ``stale_source``, and the
    stale value becomes a named wrong answer that metrics can score separately.
    Distinguishes additive/dilutive memories from genuine erase-and-rewrite, so
    it is the direct test for A2 delta-rule mechanisms (§9 packet K1).
    """

    name = "T2"
    prompt = "18"


class T3SelectiveSuppression(_PlannedFamily):
    """T3 — selective suppression. Owned by prompts 37–42.

    A control token invalidates one feature or association while leaving matched
    neighbouring features intact. The matched neighbour is the whole point: it
    is what separates real suppression from generic context sensitivity, and it
    needs an exact-target and a decoy-target condition. Adds a ``SUPPRESS`` op
    code and a step field naming the suppressed feature indices.
    """

    name = "T3"
    prompt = "37-42"


class T4PendingOperator(_PlannedFamily):
    """T4 — pending operator and delayed binding. Owned by prompts 37–42.

    An operator appears before its target is known; the model must retain it,
    bind it to the correct later structure, and commit at the correct point.
    The record must separate cue recognition, operator retention, target
    binding, and final write so that each can fail independently — that is the
    only way the E005 intuition becomes testable rather than evocative.
    """

    name = "T4"
    prompt = "37-42"


class T5CompositionalRouting(_PlannedFamily):
    """T5 — compositional routing. Owned by wave 3 (prompts 61–90).

    Content, control, and binding signals take different routes and recombine.
    Needs a third bank (route labels) and a per-route ground-truth path in the
    record so that route collapse is measurable rather than inferred. Pairs with
    A6 typed-stream attention; the permutation control already implemented here
    is what will keep a "typed structure helps" result from being an artefact of
    route-label ordering.
    """

    name = "T5"
    prompt = "61-90"


_FAMILIES: dict[str, TaskFamily] = {
    family.name: family
    for family in (
        T0LocalReconstruction(),
        T1AssociativeRecall(),
        T2OverwriteAndCorrection(),
        T3SelectiveSuppression(),
        T4PendingOperator(),
        T5CompositionalRouting(),
    )
}

FAMILY_NAMES: tuple[str, ...] = tuple(_FAMILIES)
IMPLEMENTED_FAMILIES: tuple[str, ...] = ("T0", "T1")


def get_family(name: str) -> TaskFamily:
    if name not in _FAMILIES:
        raise FeatureProgramError(f"unknown task family {name!r}; expected one of {FAMILY_NAMES}")
    return _FAMILIES[name]


__all__ = [
    "FAMILY_NAMES",
    "IMPLEMENTED_FAMILIES",
    "T0LocalReconstruction",
    "T1AssociativeRecall",
    "T2OverwriteAndCorrection",
    "T3SelectiveSuppression",
    "T4PendingOperator",
    "T5CompositionalRouting",
    "TaskFamily",
    "build_key_table",
    "get_family",
]
