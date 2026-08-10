"""Train/eval/probe splits over *program templates*, never over random examples.

A random-example split on synthetic data measures almost nothing: every example
is drawn i.i.d. from the same generator, so "generalisation" reduces to "did the
model see enough samples". The north star (§4.4) is explicit that splits must be
built from generated program templates and held-out compositions, and that the
test set must contain combinations of features, distances, and operations that
were never observed together in training.

So a template here is the *structural skeleton* of an example — which operation
is required, which content-feature group supplies the answer, which key group
the query is drawn from, and which distance bucket separates source from
destination — with the actual feature draws left unspecified. Splitting on
templates and holding out whole *combinations* of axis values, while keeping
every individual axis value present in training, is what makes a test-set
failure interpretable as compositional rather than as never-having-seen-this.

This module deliberately depends on nothing else in the package. It is pure
combinatorics over immutable records, so it can be tested and reasoned about
without importing torch.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np

# Ordered so that a composition tuple always reads the same way. ``None`` on a
# field means "this axis does not apply to this task family" and it is dropped
# from the composition rather than encoded as a distinguished value.
AXIS_FIELDS = (
    "operation",
    "content_group",
    "content_group_b",
    "key_group",
    "distance_bucket",
)


class SplitError(RuntimeError):
    """Raised when a requested split cannot be built as specified."""


@dataclass(frozen=True)
class ProgramTemplate:
    """One structural skeleton. Feature values are not part of a template."""

    family: str
    operation: str
    content_group: int
    key_group: int
    distance_bucket: int
    distance_min: int
    distance_max: int
    n_distractors: int
    n_associations: int
    key_collisions: bool
    content_group_b: int | None = None
    """Second content group, used by T0 to make feature *co-occurrence* an axis
    so that held-out compositions mean "these groups never appeared together"."""

    @property
    def template_id(self) -> str:
        """Stable across processes and runs: sha256 of the canonical field dict.

        Deliberately not Python's ``hash()``, which is salted per interpreter.
        """
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def axis_values(self) -> tuple[tuple[str, object], ...]:
        return tuple(
            (name, getattr(self, name)) for name in AXIS_FIELDS if getattr(self, name) is not None
        )

    @property
    def composition(self) -> tuple:
        """The tuple whose *combination* is held out, not its individual parts."""
        return tuple(value for _, value in self.axis_values())

    def as_dict(self) -> dict:
        record = asdict(self)
        record["template_id"] = self.template_id
        record["composition"] = list(self.composition)
        return record


def build_templates(
    *,
    family: str,
    operations: tuple[str, ...],
    content_groups: tuple[int, ...],
    key_groups: tuple[int, ...],
    distance_buckets: tuple[tuple[int, int], ...],
    n_distractors: int,
    n_associations: int,
    key_collisions: bool,
    content_groups_b: tuple[int, ...] | None = None,
) -> tuple[ProgramTemplate, ...]:
    """Enumerate the full template grid in a fixed, sorted order.

    Order matters: examples are assigned templates round-robin, so a reordering
    here would silently change every dataset while leaving the config identical.
    """
    second = content_groups_b if content_groups_b is not None else (None,)
    templates: list[ProgramTemplate] = []
    for operation in sorted(operations):
        for content_group in sorted(content_groups):
            for content_group_b in sorted(second, key=lambda v: (v is not None, v)):
                for key_group in sorted(key_groups):
                    for bucket_index, (low, high) in enumerate(distance_buckets):
                        templates.append(
                            ProgramTemplate(
                                family=family,
                                operation=operation,
                                content_group=content_group,
                                content_group_b=content_group_b,
                                key_group=key_group,
                                distance_bucket=bucket_index,
                                distance_min=low,
                                distance_max=high,
                                n_distractors=n_distractors,
                                n_associations=n_associations,
                                key_collisions=key_collisions,
                            )
                        )
    return tuple(templates)


@dataclass(frozen=True)
class SplitPlan:
    """Which templates belong to which split, and the evidence that it is sound."""

    train: tuple[ProgramTemplate, ...]
    test: tuple[ProgramTemplate, ...]
    seed: int
    holdout_fraction: float
    axis_coverage_enforced: bool

    def templates_for(self, split: str) -> tuple[ProgramTemplate, ...]:
        if split == "train":
            return self.train
        if split in ("test", "eval"):
            return self.test
        raise SplitError(f"unknown split {split!r}; expected 'train' or 'test'")

    @property
    def heldout_compositions(self) -> tuple[tuple, ...]:
        """Compositions present in test and absent from train."""
        train_compositions = {t.composition for t in self.train}
        return tuple(
            sorted(
                {t.composition for t in self.test} - train_compositions,
                key=repr,
            )
        )

    def report(self) -> dict:
        """Everything a reader needs to believe the split, computed not asserted."""
        train_ids = {t.template_id for t in self.train}
        test_ids = {t.template_id for t in self.test}
        train_axis: dict[str, set] = {}
        test_axis: dict[str, set] = {}
        for template in self.train:
            for name, value in template.axis_values():
                train_axis.setdefault(name, set()).add(value)
        for template in self.test:
            for name, value in template.axis_values():
                test_axis.setdefault(name, set()).add(value)

        uncovered = {
            name: sorted(values - train_axis.get(name, set()), key=repr)
            for name, values in test_axis.items()
            if values - train_axis.get(name, set())
        }
        return {
            "n_train_templates": len(self.train),
            "n_test_templates": len(self.test),
            "template_id_overlap": len(train_ids & test_ids),
            "n_heldout_compositions": len(self.heldout_compositions),
            "heldout_compositions": [list(c) for c in self.heldout_compositions],
            "axis_values_train": {k: sorted(v, key=repr) for k, v in train_axis.items()},
            "axis_values_test": {k: sorted(v, key=repr) for k, v in test_axis.items()},
            "test_axis_values_absent_from_train": uncovered,
            "compositional": not uncovered and len(self.heldout_compositions) > 0,
            "axis_coverage_enforced": self.axis_coverage_enforced,
            "seed": self.seed,
            "holdout_fraction": self.holdout_fraction,
        }

    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "train": [t.template_id for t in self.train],
                "test": [t.template_id for t in self.test],
                "seed": self.seed,
                "holdout_fraction": self.holdout_fraction,
                "axis_coverage_enforced": self.axis_coverage_enforced,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def split_templates(
    templates: tuple[ProgramTemplate, ...],
    *,
    seed: int,
    holdout_fraction: float = 0.25,
    require_axis_coverage: bool = True,
) -> SplitPlan:
    """Hold out whole templates, preserving per-axis coverage in training.

    With ``require_axis_coverage`` (the default) a template only moves to test
    if every axis value it carries still appears somewhere in train afterwards.
    That is what turns the holdout into a *compositional* one: the test set is
    novel combinations of familiar parts, not novel parts.

    Families with a single varying axis cannot satisfy that — removing any
    template would orphan its only axis value — so they pass
    ``require_axis_coverage=False`` and the report says so rather than pretending.
    """
    if not templates:
        raise SplitError("cannot split an empty template grid")
    if not 0.0 < holdout_fraction < 1.0:
        raise SplitError(f"holdout_fraction must be in (0, 1), got {holdout_fraction}")

    order = np.random.default_rng(seed).permutation(len(templates))
    target_test = max(1, round(holdout_fraction * len(templates)))

    counts: dict[str, Counter] = {}
    for template in templates:
        for name, value in template.axis_values():
            counts.setdefault(name, Counter())[value] += 1

    test_indices: list[int] = []
    for index in order:
        if len(test_indices) >= target_test:
            break
        template = templates[index]
        if require_axis_coverage and any(
            counts[name][value] <= 1 for name, value in template.axis_values()
        ):
            continue
        test_indices.append(int(index))
        for name, value in template.axis_values():
            counts[name][value] -= 1

    if not test_indices:
        raise SplitError(
            f"no template could be held out of a grid of {len(templates)} while preserving "
            "per-axis coverage; the grid has too few varying axes for a compositional split "
            "(pass require_axis_coverage=False to take a plain held-out-template split)"
        )

    test_set = set(test_indices)
    train = tuple(t for i, t in enumerate(templates) if i not in test_set)
    test = tuple(templates[i] for i in sorted(test_indices))
    if not train:
        raise SplitError("holdout consumed every template; lower holdout_fraction")

    return SplitPlan(
        train=train,
        test=test,
        seed=seed,
        holdout_fraction=holdout_fraction,
        axis_coverage_enforced=require_axis_coverage,
    )
