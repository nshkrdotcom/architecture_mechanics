"""Task-level capability metrics, calibrated against known answers.

These are the laboratory's rulers, and they are written *before* any
architecture exists in the repository. That ordering is the point. A metric
written after a model exists gets tuned — consciously or not — until it says
something interesting about that model. A metric written against an oracle and
a marginal baseline, with nothing yet to flatter, cannot be.

Every §6.1 metric here is a pure function of ``(predictions, reference)``, where
``reference`` is built from the ground-truth program record. Nothing here trains
anything, touches a GPU, or imports a model.

The adversary
-------------

The reference that decides whether a metric is worth keeping is not the oracle
and not chance. It is the **marginal baseline**: a predictor that emits the
training feature marginals and ignores the input entirely. A metric on which the
marginal scores well is measuring feature frequency, not computation, and it
will make every architecture look competent.

This module therefore carries a *stronger* marginal than the mission requires:
:func:`fit_marginal` can be fitted on the evaluation split itself
(``frequency_ceiling``), which is the best any input-blind predictor could
possibly do. Retirement decisions are made against that ceiling, so a metric
cannot survive by exploiting train/test frequency shift.

Retirement decisions are recorded in :data:`METRIC_SPECS` as data, and
``--selftest`` re-measures them: every ``retained`` metric must still pass the
rule and every ``retired`` metric must still fail it. The decision lives in the
source; the evidence for it is recomputed on every run, so neither can drift
away from the other silently.

Scoring conventions, once, so no metric has to restate them:

- only supervised positions are scored (``target_mask``);
- only the content bank is scored — targets are exactly zero elsewhere;
- feature activity is selected at a **rate-matched** operating point by default
  (the threshold is chosen so the number of predicted-active cells equals the
  number of truly active cells), which removes threshold choice as a free
  parameter and makes micro precision, recall, and F1 coincide;
- feature indices are always tensor coordinates, so everything here is
  permutation-agnostic by construction.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import torch

from ..data.feature_program import (
    FeatureProgramConfig,
    FeatureProgramDataset,
    ProgramRecord,
    ProgramStep,
    condition_config,
    generate_dataset,
    t0_config,
)

METRIC_VERSION = "cap-1.0.0"
"""Bump on any change to the *semantics* of a metric. Recorded beside every
number these functions produce, so a metric change invalidates a comparison
instead of silently redefining it."""

RECALL_OPS: tuple[str, ...] = ("recall_by_key", "recall_first_binding")
"""Operations that require transport. ``reconstruct`` (T0) is deliberately not
one of them: it is scored, but not as *associative* recall."""

OVERWRITE_FIELDS: tuple[str, ...] = ("stale_source", "stale_answer_features")
"""The two step fields T2 must carry. See :class:`OverwriteStep`."""

_TIE_BREAK_SEED = 0x5EED
"""Rate-matched selection needs a tie-break among equal scores. A fixed seeded
permutation keeps it deterministic and unbiased; the alternative (index order)
would systematically favour low-numbered features."""


class CapabilityMetricError(ValueError):
    """Raised when a metric is asked for something the data cannot support."""


# --------------------------------------------------------------------------- #
# The T2 step contract, written before T2 exists
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class OverwriteStep(ProgramStep):
    """A T2 step: one key bound twice, the later binding lawful.

    T2 is prompt 18's task family. Its metrics are written here anyway, against
    this schema, because a metric authored after the mechanism it judges is a
    metric that has already seen what it is supposed to measure.

    ``source`` names the *newest lawful* binding (the one the answer must come
    from) and ``answer_features`` its content, exactly as for T1. The two extra
    fields name the superseded binding:

    ``stale_source``
        Position of the earlier, now-invalid binding of the same key.
    ``stale_answer_features``
        Its content — the named wrong answer. A memory that dilutes rather than
        erases returns this, and :func:`stale_value_error_rate` scores exactly
        that failure.

    Prompt 18 should promote these two fields onto :class:`ProgramStep` itself
    and delete this class; until then this is the contract, and the metrics
    accept any step object carrying the two field names.
    """

    stale_source: int | None = None
    stale_answer_features: tuple[int, ...] = ()


def is_overwrite_step(step: object) -> bool:
    """Does this step carry the T2 contract, filled in?"""
    return all(getattr(step, name, None) is not None for name in OVERWRITE_FIELDS) and bool(
        getattr(step, "stale_answer_features", ())
    )


# --------------------------------------------------------------------------- #
# Predictions and the ground-truth reference
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Predictions:
    """What a predictor emits: a value per feature and an activity probability.

    Two channels because the §6.1 list needs both. ``values`` answers "how much
    of feature f is present", which is what reconstruction loss scores.
    ``active_prob`` answers "is feature f present at all", which is what
    precision/recall, set accuracy, and calibration score. A model with only a
    value head can use :meth:`from_values`; a model with an explicit activity
    head should supply both, because the induced probability from
    :meth:`from_values` ranks well but is not calibrated.
    """

    values: np.ndarray
    """``(N, T, F)`` float."""
    active_prob: np.ndarray
    """``(N, T, F)`` float in ``[0, 1]``."""

    def __post_init__(self) -> None:
        if self.values.shape != self.active_prob.shape:
            raise CapabilityMetricError(
                f"values {self.values.shape} and active_prob {self.active_prob.shape} disagree"
            )
        if self.values.ndim != 3:
            raise CapabilityMetricError(f"predictions must be (N, T, F); got {self.values.shape}")
        if self.active_prob.size and (
            float(self.active_prob.min()) < 0.0 or float(self.active_prob.max()) > 1.0
        ):
            raise CapabilityMetricError("active_prob must lie in [0, 1]")

    @classmethod
    def from_values(cls, values: np.ndarray | torch.Tensor) -> Predictions:
        """Value-only predictor. Activity probability is the clipped value.

        Ground-truth magnitudes are ``Uniform(0, 1)`` when active and exactly
        ``0.0`` otherwise, so the clipped value is a monotone activity score —
        good enough to rank, deliberately not claimed to be calibrated.
        """
        array = _as_numpy(values)
        return cls(values=array, active_prob=np.clip(array, 0.0, 1.0))

    @classmethod
    def concat(cls, parts: Sequence[Predictions]) -> Predictions:
        return cls(
            values=np.concatenate([p.values for p in parts], axis=0),
            active_prob=np.concatenate([p.active_prob for p in parts], axis=0),
        )


@dataclass(frozen=True)
class EvaluationReference:
    """The ground truth a metric scores against, detached from the generator.

    Holds the tensors *and* the program records. Program records are addressed
    **positionally**: ``programs[i]`` describes row ``i`` of the tensors. That is
    what lets two splits be concatenated into one reference so held-out
    composition accuracy can compare seen and unseen templates in one call.
    """

    family: str
    condition: str
    split: str
    inputs: np.ndarray
    input_active: np.ndarray
    targets: np.ndarray
    target_active: np.ndarray
    supervised: np.ndarray
    importance: np.ndarray
    content_indices: np.ndarray
    programs: tuple[ProgramRecord, ...]
    heldout_template_ids: frozenset[str]
    content_column: dict[int, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        n, t, f = self.inputs.shape
        if len(self.programs) != n:
            raise CapabilityMetricError(
                f"{len(self.programs)} program records for {n} examples; records are positional"
            )
        if self.supervised.shape != (n, t):
            raise CapabilityMetricError("target mask shape disagrees with the tensors")
        if self.importance.shape != (f,):
            raise CapabilityMetricError("importance must be one weight per feature")
        object.__setattr__(
            self,
            "content_column",
            {int(feature): column for column, feature in enumerate(self.content_indices)},
        )

    @property
    def n_examples(self) -> int:
        return int(self.inputs.shape[0])

    @property
    def n_supervised(self) -> int:
        return int(self.supervised.sum())

    @classmethod
    def from_dataset(cls, dataset: FeatureProgramDataset) -> EvaluationReference:
        return cls(
            family=dataset.config.family,
            condition=dataset.config.condition,
            split=dataset.config.split,
            inputs=_as_numpy(dataset.inputs),
            input_active=_as_numpy(dataset.active_mask),
            targets=_as_numpy(dataset.targets),
            target_active=_as_numpy(dataset.target_active_mask),
            supervised=_as_numpy(dataset.target_mask),
            importance=_as_numpy(dataset.importance).astype(np.float64),
            content_indices=np.asarray(dataset.content_indices, dtype=np.int64),
            programs=dataset.programs,
            heldout_template_ids=frozenset(t.template_id for t in dataset.split_plan.test),
        )

    @classmethod
    def concat(cls, parts: Sequence[EvaluationReference]) -> EvaluationReference:
        """Join references over the example axis, renumbering the records.

        ``example_index`` is rewritten so it keeps agreeing with the row it
        describes. Everything else about a record is a property of the example,
        not of its position in a split.
        """
        first = parts[0]
        for part in parts[1:]:
            if part.inputs.shape[1:] != first.inputs.shape[1:]:
                raise CapabilityMetricError("cannot concatenate references of different shape")
        programs: list[ProgramRecord] = []
        for part in parts:
            offset = len(programs)
            programs.extend(
                replace(record, example_index=offset + i) for i, record in enumerate(part.programs)
            )
        return cls(
            family=first.family,
            condition=first.condition,
            split="+".join(part.split for part in parts),
            inputs=np.concatenate([p.inputs for p in parts], axis=0),
            input_active=np.concatenate([p.input_active for p in parts], axis=0),
            targets=np.concatenate([p.targets for p in parts], axis=0),
            target_active=np.concatenate([p.target_active for p in parts], axis=0),
            supervised=np.concatenate([p.supervised for p in parts], axis=0),
            importance=first.importance,
            content_indices=first.content_indices,
            programs=tuple(programs),
            heldout_template_ids=frozenset().union(*(p.heldout_template_ids for p in parts)),
        )


def _as_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


@dataclass(frozen=True)
class MetricValue:
    """One number, with the sample size that produced it.

    ``value is None`` means "not applicable to this data" — no overwrite steps,
    no recall operations, one distance bucket — and is never silently coerced to
    zero. A metric that cannot be computed must say so, because a zero here
    would be read as a failing model rather than an absent task.
    """

    name: str
    value: float | None
    n: int
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"name": self.name, "value": self.value, "n": self.n, "detail": self.detail}


@dataclass(frozen=True)
class Curve:
    """Metric-versus-axis data. Deliberately *not* a figure.

    Prompt 14's phase diagram and prompt 23's trajectory figure both consume
    these. Keeping the plotting out of the metric is what lets §8.5's "report
    generated only from recorded artifacts" hold: the curve is recorded, and the
    figure is regenerated from the record.
    """

    name: str
    axis: str
    x: tuple[float, ...]
    y: tuple[float, ...]
    n: tuple[int, ...]
    slope: float | None
    """Count-weighted least-squares slope of ``y`` on ``x``. ``None`` when fewer
    than two populated buckets exist — an undefined slope, never a zero."""

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "axis": self.axis,
            "x": list(self.x),
            "y": list(self.y),
            "n": list(self.n),
            "slope": self.slope,
        }


def _weighted_slope(x: np.ndarray, y: np.ndarray, n: np.ndarray) -> float | None:
    if x.size < 2:
        return None
    weights = n.astype(np.float64)
    if weights.sum() <= 0:
        return None
    mean_x = float((weights * x).sum() / weights.sum())
    mean_y = float((weights * y).sum() / weights.sum())
    denominator = float((weights * (x - mean_x) ** 2).sum())
    if denominator <= 0:
        return None
    return float((weights * (x - mean_x) * (y - mean_y)).sum() / denominator)


# --------------------------------------------------------------------------- #
# Cell-level scoring substrate
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Cells:
    """Supervised positions flattened to rows, restricted to the content bank."""

    example: np.ndarray
    position: np.ndarray
    row_of: np.ndarray
    y: np.ndarray
    a: np.ndarray
    v: np.ndarray
    p: np.ndarray
    w: np.ndarray


def _score_cells(predictions: Predictions, reference: EvaluationReference) -> _Cells:
    if predictions.values.shape != reference.targets.shape:
        raise CapabilityMetricError(
            f"predictions {predictions.values.shape} do not match reference "
            f"{reference.targets.shape}"
        )
    example, position = np.nonzero(reference.supervised)
    if example.size == 0:
        raise CapabilityMetricError("reference has no supervised positions")
    row_of = np.full(reference.supervised.shape, -1, dtype=np.int64)
    row_of[example, position] = np.arange(example.size)
    content = reference.content_indices
    return _Cells(
        example=example,
        position=position,
        row_of=row_of,
        y=reference.targets[example, position][:, content].astype(np.float64),
        a=reference.target_active[example, position][:, content],
        v=predictions.values[example, position][:, content].astype(np.float64),
        p=predictions.active_prob[example, position][:, content].astype(np.float64),
        w=reference.importance[content].astype(np.float64),
    )


def _select_active(scores: np.ndarray, n_true: int, threshold: float | None) -> tuple[np.ndarray, float]:
    """Turn activity scores into a predicted-active set.

    ``threshold=None`` is the rate-matched operating point: take the ``n_true``
    highest-scoring cells, so the predicted-active count equals the true count.
    This is the default because a free threshold is a free parameter, and a free
    parameter is somewhere for a result to hide.
    """
    if threshold is not None:
        return scores >= threshold, float(threshold)
    flat = scores.ravel()
    k = int(n_true)
    if k <= 0:
        return np.zeros_like(scores, dtype=bool), float("inf")
    if k >= flat.size:
        return np.ones_like(scores, dtype=bool), float("-inf")
    jitter = np.random.default_rng(_TIE_BREAK_SEED).permutation(flat.size)
    order = np.lexsort((jitter, -flat))
    picked = np.zeros(flat.size, dtype=bool)
    picked[order[:k]] = True
    return picked.reshape(scores.shape), float(flat[order[k - 1]])


# --------------------------------------------------------------------------- #
# §6.1 metric 1 — reconstruction loss
# --------------------------------------------------------------------------- #


def reconstruction_loss(predictions: Predictions, reference: EvaluationReference) -> MetricValue:
    """Importance-weighted mean squared error over the content bank.

    ``sum_f w_f (yhat_f - y_f)^2 / sum_f w_f``, averaged over supervised
    positions. Features carry unequal importance by §4.2, and the generator
    already sets that weight to exactly zero outside the content bank, so the
    weighting is read from the data rather than chosen here.

    Lower is better; the oracle reaches exactly ``0.0``.
    """
    cells = _score_cells(predictions, reference)
    residual = cells.v - cells.y
    per_position = (residual * residual * cells.w).sum(axis=-1) / cells.w.sum()
    return MetricValue(
        name="reconstruction_loss",
        value=float(per_position.mean()),
        n=int(per_position.size),
        detail={"weighted": True, "n_content_features": int(cells.w.size)},
    )


# --------------------------------------------------------------------------- #
# §6.1 metric 2 — per-feature precision and recall
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DetectionResult:
    """Feature-activity detection at one operating point."""

    threshold: float
    rate_matched: bool
    n_true: int
    n_selected: int
    precision: float
    recall: float
    f1: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_feature_precision: np.ndarray
    per_feature_recall: np.ndarray

    def as_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "rate_matched": self.rate_matched,
            "n_true": self.n_true,
            "n_selected": self.n_selected,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
        }


def feature_detection(
    predictions: Predictions,
    reference: EvaluationReference,
    *,
    threshold: float | None = None,
) -> DetectionResult:
    """Per-feature precision and recall for "is this feature active here".

    At the default rate-matched operating point the predicted-positive count
    equals the true-positive count, so micro precision, recall, and F1 are equal
    by construction. That is intended: it collapses three numbers that differ
    only through threshold choice into one that does not, and it is why raw
    recall — which any predictor can drive to 1.0 by lowering its threshold —
    is retired below rather than reported.

    Per-feature values are ``NaN`` where the denominator is empty, and the macro
    averages ignore those features rather than scoring them zero.
    """
    cells = _score_cells(predictions, reference)
    selected, tau = _select_active(cells.p, int(cells.a.sum()), threshold)

    tp = int((selected & cells.a).sum())
    n_selected = int(selected.sum())
    n_true = int(cells.a.sum())
    precision = tp / n_selected if n_selected else float("nan")
    recall = tp / n_true if n_true else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0

    tp_f = (selected & cells.a).sum(axis=0).astype(np.float64)
    sel_f = selected.sum(axis=0).astype(np.float64)
    true_f = cells.a.sum(axis=0).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_precision = np.where(sel_f > 0, tp_f / sel_f, np.nan)
        per_recall = np.where(true_f > 0, tp_f / true_f, np.nan)
        per_f1 = np.where(
            (per_precision + per_recall) > 0,
            2 * per_precision * per_recall / (per_precision + per_recall),
            0.0,
        )
    return DetectionResult(
        threshold=tau,
        rate_matched=threshold is None,
        n_true=n_true,
        n_selected=n_selected,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        macro_precision=float(np.nanmean(per_precision)) if np.any(~np.isnan(per_precision)) else float("nan"),
        macro_recall=float(np.nanmean(per_recall)) if np.any(~np.isnan(per_recall)) else float("nan"),
        macro_f1=float(np.nanmean(np.where(np.isnan(per_f1), np.nan, per_f1))),
        per_feature_precision=per_precision,
        per_feature_recall=per_recall,
    )


THRESHOLD_GRID: tuple[float, ...] = tuple(round(v, 2) for v in np.linspace(0.0, 1.0, 21))


@dataclass(frozen=True)
class ThresholdSweep:
    """Detection metrics across every operating point a predictor could choose.

    This is the adversarial half of the retirement analysis. Asking what the
    frequency ceiling scores *at one threshold* answers the wrong question: a
    predictor picks its own threshold. Asking what it scores at its **best**
    threshold is what shows whether a metric can be reached without computing
    anything, and it is how ``recall_at_free_threshold`` was retired.
    """

    thresholds: tuple[float, ...]
    precision: tuple[float, ...]
    recall: tuple[float, ...]
    f1: tuple[float, ...]
    best_precision: float
    best_recall: float
    best_f1: float

    def as_dict(self) -> dict:
        return {
            "thresholds": list(self.thresholds),
            "precision": list(self.precision),
            "recall": list(self.recall),
            "f1": list(self.f1),
            "best_precision": self.best_precision,
            "best_recall": self.best_recall,
            "best_f1": self.best_f1,
        }


def threshold_sweep(
    predictions: Predictions,
    reference: EvaluationReference,
    *,
    thresholds: Sequence[float] = THRESHOLD_GRID,
) -> ThresholdSweep:
    """Precision, recall, and F1 at every threshold, plus the best of each.

    Thresholds that select nothing yield an undefined precision and are excluded
    from ``best_precision`` — otherwise "select nothing" would score perfectly by
    vacuity, which is the mirror image of the failure that retires recall.
    """
    cells = _score_cells(predictions, reference)
    n_true = int(cells.a.sum())
    precision: list[float] = []
    recall: list[float] = []
    f1: list[float] = []
    for tau in thresholds:
        selected = cells.p >= tau
        tp = int((selected & cells.a).sum())
        n_selected = int(selected.sum())
        p = tp / n_selected if n_selected else float("nan")
        r = tp / n_true if n_true else float("nan")
        precision.append(float(p))
        recall.append(float(r))
        f1.append(float(2 * p * r / (p + r)) if tp else 0.0)
    finite_p = [v for v in precision if not math.isnan(v)]
    finite_r = [v for v in recall if not math.isnan(v)]
    finite_f = [v for v in f1 if not math.isnan(v)]
    return ThresholdSweep(
        thresholds=tuple(float(t) for t in thresholds),
        precision=tuple(precision),
        recall=tuple(recall),
        f1=tuple(f1),
        best_precision=max(finite_p) if finite_p else float("nan"),
        best_recall=max(finite_r) if finite_r else float("nan"),
        best_f1=max(finite_f) if finite_f else float("nan"),
    )


def recall_at_free_threshold(predictions, reference) -> MetricValue:
    """The best recall a predictor reaches over all thresholds.

    Retired. Every predictor scores exactly 1.0 here, including one that ignores
    the input, because a threshold of zero selects every cell. Recorded as a
    metric so ``--selftest`` keeps re-demonstrating why recall is only ever
    reported at the rate-matched operating point.
    """
    sweep = threshold_sweep(predictions, reference)
    return MetricValue(
        "recall_at_free_threshold",
        sweep.best_recall,
        reference.n_supervised,
        {"n_thresholds": len(sweep.thresholds)},
    )


def feature_precision(predictions, reference, *, threshold=None) -> MetricValue:
    """Micro precision at the (by default rate-matched) operating point."""
    result = feature_detection(predictions, reference, threshold=threshold)
    return MetricValue("feature_precision", result.precision, result.n_selected, result.as_dict())


def feature_recall(predictions, reference, *, threshold=None) -> MetricValue:
    """Micro recall. Retired as a headline; see :data:`METRIC_SPECS`."""
    result = feature_detection(predictions, reference, threshold=threshold)
    return MetricValue("feature_recall", result.recall, result.n_true, result.as_dict())


def feature_f1(predictions, reference, *, threshold=None) -> MetricValue:
    result = feature_detection(predictions, reference, threshold=threshold)
    return MetricValue("feature_f1", result.f1, result.n_true, result.as_dict())


def feature_macro_precision(predictions, reference, *, threshold=None) -> MetricValue:
    result = feature_detection(predictions, reference, threshold=threshold)
    return MetricValue(
        "feature_macro_precision", result.macro_precision, int(result.per_feature_precision.size),
        result.as_dict(),
    )


def feature_macro_recall(predictions, reference, *, threshold=None) -> MetricValue:
    result = feature_detection(predictions, reference, threshold=threshold)
    return MetricValue(
        "feature_macro_recall", result.macro_recall, int(result.per_feature_recall.size),
        result.as_dict(),
    )


# --------------------------------------------------------------------------- #
# Step-level scoring: the substrate for recall, distance, distractors, holdout
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StepScores:
    """Per-step correctness plus the program facts each metric groups by.

    One pass over the program record produces everything the step-level metrics
    need. Associative recall accuracy, the distance curve, the distractor curve,
    and held-out composition accuracy are then all *groupings* of the same
    per-step scores, which is what keeps them from drifting apart.
    """

    exact: np.ndarray
    jaccard: np.ndarray
    distance: np.ndarray
    n_distractors: np.ndarray
    heldout: np.ndarray
    ops: tuple[str, ...]
    threshold: float

    def __len__(self) -> int:
        return int(self.exact.size)


def step_scores(
    predictions: Predictions,
    reference: EvaluationReference,
    *,
    ops: Sequence[str] | None = None,
    threshold: float | None = None,
) -> StepScores:
    """Score each required operation's answer set against the ground truth.

    The predicted answer at a destination is the set of content features chosen
    by the activity rule; the true answer is ``step.answer_features``, which the
    generator writes from the program itself. ``exact`` is set equality;
    ``jaccard`` is the graded version, so a near-miss is distinguishable from a
    miss.
    """
    cells = _score_cells(predictions, reference)
    selected, tau = _select_active(cells.p, int(cells.a.sum()), threshold)
    column = reference.content_column

    exact: list[float] = []
    jaccard: list[float] = []
    distance: list[float] = []
    distractors: list[float] = []
    heldout: list[bool] = []
    seen_ops: list[str] = []

    for row, record in enumerate(reference.programs):
        is_heldout = record.template_id in reference.heldout_template_ids
        for step in record.steps:
            if ops is not None and step.op not in ops:
                continue
            cell = int(cells.row_of[row, step.dest])
            if cell < 0:
                raise CapabilityMetricError(
                    f"step destination {step.dest} of example {row} is not supervised"
                )
            predicted = {
                int(reference.content_indices[c])
                for c in np.nonzero(selected[cell])[0]
            }
            truth = {int(f) for f in step.answer_features if f in column}
            union = predicted | truth
            exact.append(float(predicted == truth))
            jaccard.append(1.0 if not union else len(predicted & truth) / len(union))
            distance.append(np.nan if step.distance is None else float(step.distance))
            distractors.append(float(len(step.distractors)))
            heldout.append(is_heldout)
            seen_ops.append(step.op)

    return StepScores(
        exact=np.asarray(exact, dtype=np.float64),
        jaccard=np.asarray(jaccard, dtype=np.float64),
        distance=np.asarray(distance, dtype=np.float64),
        n_distractors=np.asarray(distractors, dtype=np.float64),
        heldout=np.asarray(heldout, dtype=bool),
        ops=tuple(sorted(set(seen_ops))),
        threshold=tau,
    )


def _mean_or_none(values: np.ndarray) -> float | None:
    return float(values.mean()) if values.size else None


# --------------------------------------------------------------------------- #
# §6.1 metric 3 — associative recall accuracy
# --------------------------------------------------------------------------- #


def answer_set_accuracy(
    predictions: Predictions, reference: EvaluationReference, *, ops=None, threshold=None
) -> MetricValue:
    """Fraction of required operations whose answer set is recovered exactly.

    Set equality, not overlap: recovering four of five features is wrong, and a
    metric that gives it 0.8 lets a diluted memory read as nearly correct.
    :func:`associative_recall_jaccard` is the graded companion for when the
    *degree* of blending is the question.
    """
    scores = step_scores(predictions, reference, ops=ops, threshold=threshold)
    return MetricValue(
        "answer_set_accuracy",
        _mean_or_none(scores.exact),
        len(scores),
        {"ops": scores.ops, "threshold": scores.threshold},
    )


def associative_recall_accuracy(
    predictions: Predictions, reference: EvaluationReference, *, threshold=None
) -> MetricValue:
    """Exact answer-set accuracy over transport operations only.

    T0's ``reconstruct`` is excluded: it requires no transport, so scoring it as
    associative recall would let a model with no mixing at all post a recall
    number. On a T0 reference this returns ``None`` with ``n=0``.
    """
    scores = step_scores(predictions, reference, ops=RECALL_OPS, threshold=threshold)
    return MetricValue(
        "associative_recall_accuracy",
        _mean_or_none(scores.exact),
        len(scores),
        {"ops": scores.ops, "threshold": scores.threshold},
    )


def associative_recall_jaccard(
    predictions: Predictions, reference: EvaluationReference, *, threshold=None
) -> MetricValue:
    """Graded answer-set overlap over transport operations."""
    scores = step_scores(predictions, reference, ops=RECALL_OPS, threshold=threshold)
    return MetricValue(
        "associative_recall_jaccard",
        _mean_or_none(scores.jaccard),
        len(scores),
        {"ops": scores.ops},
    )


# --------------------------------------------------------------------------- #
# §6.1 metrics 4 and 5 — overwrite accuracy and stale-value error rate
# --------------------------------------------------------------------------- #


def _overwrite_steps(reference: EvaluationReference):
    for row, record in enumerate(reference.programs):
        for step in record.steps:
            if is_overwrite_step(step):
                yield row, step


def overwrite_accuracy(
    predictions: Predictions, reference: EvaluationReference, *, threshold=None
) -> MetricValue:
    """Fraction of overwrite queries answered with the newest lawful value.

    Scored on exactly the same rule as :func:`answer_set_accuracy`, restricted
    to steps carrying the T2 contract, so "the model got the right answer" means
    the same thing whether or not a key was rebound.

    Returns ``None`` with ``n=0`` on any dataset without overwrite steps, which
    today is all of them — T2 is prompt 18.
    """
    cells = _score_cells(predictions, reference)
    selected, tau = _select_active(cells.p, int(cells.a.sum()), threshold)
    hits: list[float] = []
    for row, step in _overwrite_steps(reference):
        cell = int(cells.row_of[row, step.dest])
        predicted = {int(reference.content_indices[c]) for c in np.nonzero(selected[cell])[0]}
        hits.append(float(predicted == {int(f) for f in step.answer_features}))
    return MetricValue(
        "overwrite_accuracy",
        _mean_or_none(np.asarray(hits, dtype=np.float64)),
        len(hits),
        {"threshold": tau} if hits else {"reason": "no overwrite steps; T2 is prompt 18"},
    )


def stale_value_error_rate(
    predictions: Predictions, reference: EvaluationReference
) -> MetricValue:
    """Fraction of overwrite queries answered nearer the superseded value.

    The named wrong answer, not a generic error. An additive or dilutive memory
    that never truly erases returns the stale value, and this is the metric that
    catches it; a model that is merely bad scores badly on
    :func:`overwrite_accuracy` without necessarily scoring badly here.

    Distance is the same importance-weighted squared error the reconstruction
    loss uses, so "nearer" means nearer on the scale the task is trained on.
    Exact ties count as half an error rather than being silently forgiven.
    """
    cells = _score_cells(predictions, reference)
    content = reference.content_indices
    errors: list[float] = []
    for row, step in _overwrite_steps(reference):
        cell = int(cells.row_of[row, step.dest])
        prediction = cells.v[cell]
        current = cells.y[cell]
        stale = reference.inputs[row, int(step.stale_source)][content].astype(np.float64)
        to_current = float((cells.w * (prediction - current) ** 2).sum())
        to_stale = float((cells.w * (prediction - stale) ** 2).sum())
        errors.append(0.5 if to_stale == to_current else float(to_stale < to_current))
    return MetricValue(
        "stale_value_error_rate",
        _mean_or_none(np.asarray(errors, dtype=np.float64)),
        len(errors),
        {} if errors else {"reason": "no overwrite steps; T2 is prompt 18"},
    )


# --------------------------------------------------------------------------- #
# §6.1 metrics 6 and 7 — distance degradation and distractor sensitivity
# --------------------------------------------------------------------------- #


def _bucket_curve(name: str, axis: str, key: np.ndarray, value: np.ndarray) -> Curve:
    finite = np.isfinite(key)
    key, value = key[finite], value[finite]
    if key.size == 0:
        return Curve(name=name, axis=axis, x=(), y=(), n=(), slope=None)
    levels = np.unique(key)
    means = np.asarray([value[key == level].mean() for level in levels])
    counts = np.asarray([int((key == level).sum()) for level in levels])
    return Curve(
        name=name,
        axis=axis,
        x=tuple(float(v) for v in levels),
        y=tuple(float(v) for v in means),
        n=tuple(int(v) for v in counts),
        slope=_weighted_slope(levels.astype(np.float64), means, counts),
    )


def distance_degradation(
    predictions: Predictions, reference: EvaluationReference, *, threshold=None
) -> Curve:
    """Answer-set accuracy as a function of source-to-destination distance.

    The slope of this curve is a *diagnostic and never a headline*: a predictor
    that ignores the input entirely degrades not at all, so a flat curve at
    chance is indistinguishable in slope from a flat curve at ceiling. Report it
    only beside the level it was measured at. ``--selftest`` checks that the
    marginal baseline really does achieve the best possible slope while scoring
    at chance, so this warning stays measured rather than remembered.
    """
    scores = step_scores(predictions, reference, ops=RECALL_OPS, threshold=threshold)
    return _bucket_curve("distance_degradation", "distance", scores.distance, scores.exact)


def distractor_sensitivity(
    predictions: Predictions, reference: EvaluationReference, *, threshold=None
) -> Curve:
    """Answer-set accuracy as a function of distractors between source and query.

    Within one generated dataset ``n_distractors`` is a single configured value,
    so this curve has one point and an undefined slope. The across-condition
    curve comes from :func:`sweep_curve`, which is what prompt 14 wants anyway.
    The same slope-is-not-a-headline warning as :func:`distance_degradation`
    applies.
    """
    scores = step_scores(predictions, reference, ops=RECALL_OPS, threshold=threshold)
    return _bucket_curve(
        "distractor_sensitivity", "n_distractors", scores.n_distractors, scores.exact
    )


def sweep_curve(
    *,
    base: FeatureProgramConfig,
    axis: str,
    values: Sequence,
    predict: Callable[[FeatureProgramDataset], Predictions],
    metric: Callable[[Predictions, EvaluationReference], MetricValue],
    n_examples: int = 128,
    split: str = "test",
) -> Curve:
    """Vary one config axis, evaluate a predictor on each cell, return the data.

    Returns a :class:`Curve`, not a figure — prompt 14's phase diagram and
    prompt 23's trajectory both consume curves, and keeping figure code out of
    metric code is what lets §8.5's "report generated only from recorded
    artifacts" hold later.
    """
    xs: list[float] = []
    ys: list[float] = []
    ns: list[int] = []
    for value in values:
        config = replace(base, split=split, n_examples=n_examples, **{axis: value})
        dataset = generate_dataset(config)
        reference = EvaluationReference.from_dataset(dataset)
        result = metric(predict(dataset), reference)
        if result.value is None:
            continue
        xs.append(float(value))
        ys.append(float(result.value))
        ns.append(int(result.n))
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    n = np.asarray(ns, dtype=np.int64)
    return Curve(
        name=f"sweep:{axis}",
        axis=axis,
        x=tuple(xs),
        y=tuple(ys),
        n=tuple(ns),
        slope=_weighted_slope(x, y, n),
    )


# --------------------------------------------------------------------------- #
# §6.1 metric 8 — held-out composition accuracy
# --------------------------------------------------------------------------- #


def heldout_composition_accuracy(
    predictions: Predictions, reference: EvaluationReference, *, threshold=None
) -> MetricValue:
    """Answer-set accuracy on templates whose composition was held out.

    The reference must contain both seen and held-out templates for the gap to
    mean anything, which is why :meth:`EvaluationReference.concat` exists. The
    gap (seen minus held out) is reported in ``detail`` as a diagnostic: like
    the curve slopes, it is best for a predictor that has learned nothing, so it
    is never a headline on its own.
    """
    scores = step_scores(predictions, reference, threshold=threshold)
    heldout = scores.exact[scores.heldout]
    seen = scores.exact[~scores.heldout]
    seen_value = _mean_or_none(seen)
    heldout_value = _mean_or_none(heldout)
    return MetricValue(
        "heldout_composition_accuracy",
        heldout_value,
        int(heldout.size),
        {
            "seen_accuracy": seen_value,
            "n_seen": int(seen.size),
            "gap": None if seen_value is None or heldout_value is None else seen_value - heldout_value,
        },
    )


def heldout_composition_gap(
    predictions: Predictions, reference: EvaluationReference, *, threshold=None
) -> MetricValue:
    """Seen accuracy minus held-out accuracy. Diagnostic only."""
    result = heldout_composition_accuracy(predictions, reference, threshold=threshold)
    return MetricValue(
        "heldout_composition_gap", result.detail["gap"], result.n, {"level": result.value}
    )


# --------------------------------------------------------------------------- #
# §6.1 metric 9 — calibration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CalibrationResult:
    ece: float
    brier: float
    reliability: Curve

    def as_dict(self) -> dict:
        return {"ece": self.ece, "brier": self.brier, "reliability": self.reliability.as_dict()}


def calibration(
    predictions: Predictions, reference: EvaluationReference, *, n_bins: int = 10
) -> CalibrationResult:
    """Expected calibration error and Brier score over supervised content cells.

    Both are computed; only one is retained. The marginal baseline emits the
    true base rate for every feature and is therefore *perfectly calibrated
    while computing nothing*, which makes ECE unusable as a headline and is why
    it is retired below. Brier survives because it charges for the resolution
    the marginal does not have.
    """
    cells = _score_cells(predictions, reference)
    p = cells.p.ravel()
    y = cells.a.ravel().astype(np.float64)
    brier = float(((p - y) ** 2).mean())

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    conf: list[float] = []
    acc: list[float] = []
    counts: list[int] = []
    ece = 0.0
    for b in range(n_bins):
        in_bin = index == b
        count = int(in_bin.sum())
        if count == 0:
            continue
        bin_conf = float(p[in_bin].mean())
        bin_acc = float(y[in_bin].mean())
        ece += (count / p.size) * abs(bin_acc - bin_conf)
        conf.append(bin_conf)
        acc.append(bin_acc)
        counts.append(count)
    reliability = Curve(
        name="reliability",
        axis="confidence",
        x=tuple(conf),
        y=tuple(acc),
        n=tuple(counts),
        slope=_weighted_slope(
            np.asarray(conf), np.asarray(acc), np.asarray(counts, dtype=np.int64)
        ),
    )
    return CalibrationResult(ece=float(ece), brier=brier, reliability=reliability)


def expected_calibration_error(predictions, reference, *, n_bins: int = 10) -> MetricValue:
    """Retired as a headline. Kept as a diagnostic; see :data:`METRIC_SPECS`."""
    result = calibration(predictions, reference, n_bins=n_bins)
    return MetricValue("ece", result.ece, reference.n_supervised, {"n_bins": n_bins})


def brier_score(predictions, reference) -> MetricValue:
    result = calibration(predictions, reference)
    return MetricValue("brier", result.brier, reference.n_supervised, {})


# --------------------------------------------------------------------------- #
# The three references
# --------------------------------------------------------------------------- #


class Baseline:
    """A predictor that needs no training. Every one of these is a ruler mark."""

    name = "baseline"

    def predict(self, dataset: FeatureProgramDataset) -> Predictions:  # pragma: no cover
        raise NotImplementedError


@dataclass(frozen=True)
class ProgramOracle(Baseline):
    """Reads the ground-truth program and answers from it. The ceiling.

    For every supervised destination the program names its source; the oracle
    copies that position's content bank. It is *not* allowed to read the target
    tensor, so this doubles as a check that the program record alone suffices to
    reconstruct the answer.

    Where the program says the route is destroyed (``source is None``, the §4.4
    negative control) the oracle honestly has nothing to read and falls back to
    ``fallback`` — the marginal, if one is supplied. An oracle that scored 1.0
    on an impossible task would be reading the answer, not the program.
    """

    fallback: Baseline | None = None
    name: str = "oracle"

    def predict(self, dataset: FeatureProgramDataset) -> Predictions:
        inputs = _as_numpy(dataset.inputs).astype(np.float64)
        active = _as_numpy(dataset.active_mask)
        content = np.asarray(dataset.content_indices, dtype=np.int64)
        base = (
            self.fallback.predict(dataset)
            if self.fallback is not None
            else Predictions(np.zeros_like(inputs), np.zeros_like(inputs))
        )
        values = base.values.copy()
        prob = base.active_prob.copy()
        for row, record in enumerate(dataset.programs):
            for step in record.steps:
                if step.source is None:
                    continue
                values[row, step.dest, content] = inputs[row, step.source, content]
                prob[row, step.dest, content] = active[row, step.source, content].astype(np.float64)
        return Predictions(values=values, active_prob=prob)


@dataclass(frozen=True)
class RandomBaseline(Baseline):
    """Chance: activity at the task's global base rate, magnitudes uniform.

    The floor, not the adversary. It is here so that a metric's chance level is
    measured rather than assumed — several of the §6.1 metrics have a chance
    level that is not zero and not obvious.
    """

    rate: float
    seed: int = 20260809
    name: str = "random"

    @classmethod
    def fit(cls, dataset: FeatureProgramDataset, *, seed: int = 20260809) -> RandomBaseline:
        content = np.asarray(dataset.content_indices, dtype=np.int64)
        mask = _as_numpy(dataset.target_mask)
        active = _as_numpy(dataset.target_active_mask)[mask][:, content]
        return cls(rate=float(active.mean()), seed=seed)

    def predict(self, dataset: FeatureProgramDataset) -> Predictions:
        shape = tuple(dataset.inputs.shape)
        content = np.asarray(dataset.content_indices, dtype=np.int64)
        rng = np.random.default_rng(self.seed)
        values = np.zeros(shape, dtype=np.float64)
        prob = np.zeros(shape, dtype=np.float64)
        draws = rng.random((shape[0], shape[1], content.size))
        magnitudes = rng.random((shape[0], shape[1], content.size))
        values[:, :, content] = magnitudes * (draws < self.rate)
        prob[:, :, content] = self.rate
        return Predictions(values=values, active_prob=prob)


@dataclass(frozen=True)
class MarginalBaseline(Baseline):
    """Emits the fitted feature marginals and ignores the input entirely.

    **This is the reference that decides whether a metric is worth keeping.** It
    knows how often each feature is active and how large it is on average, and
    knows nothing else — no position, no key, no source. Anything it scores well
    on is a measure of feature frequency.

    Fitted on the training split for the honest baseline; fitting it on the
    evaluation split instead gives the *frequency ceiling*, the best any
    input-blind predictor could do, and that is what the retirement rule uses so
    a metric cannot survive on train/test frequency shift alone.
    """

    mean_value: np.ndarray
    active_rate: np.ndarray
    content_indices: np.ndarray
    fitted_on: str
    name: str = "marginal"

    def _emit(self, shape: tuple[int, int, int], content: np.ndarray) -> Predictions:
        if not np.array_equal(content, self.content_indices):
            raise CapabilityMetricError(
                "marginal baseline was fitted on a different feature layout"
            )
        values = np.zeros(shape, dtype=np.float64)
        prob = np.zeros(shape, dtype=np.float64)
        values[:, :, content] = self.mean_value
        prob[:, :, content] = self.active_rate
        return Predictions(values=values, active_prob=prob)

    def predict(self, dataset: FeatureProgramDataset) -> Predictions:
        return self._emit(
            tuple(dataset.inputs.shape), np.asarray(dataset.content_indices, dtype=np.int64)
        )

    def predict_like(self, reference: EvaluationReference) -> Predictions:
        """Predictions shaped for a reference that may span several splits."""
        return self._emit(reference.inputs.shape, reference.content_indices)


def _fit_marginal_arrays(targets: np.ndarray, active: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if targets.shape[0] == 0:
        raise CapabilityMetricError("cannot fit a marginal with no supervised positions")
    return targets.mean(axis=0), np.clip(active.mean(axis=0), 0.0, 1.0)


def fit_marginal(dataset: FeatureProgramDataset, *, name: str = "marginal") -> MarginalBaseline:
    """Fit per-feature mean value and activation rate on supervised targets."""
    content = np.asarray(dataset.content_indices, dtype=np.int64)
    mask = _as_numpy(dataset.target_mask)
    mean_value, active_rate = _fit_marginal_arrays(
        _as_numpy(dataset.targets)[mask][:, content].astype(np.float64),
        _as_numpy(dataset.target_active_mask)[mask][:, content].astype(np.float64),
    )
    return MarginalBaseline(
        mean_value=mean_value,
        active_rate=active_rate,
        content_indices=content,
        fitted_on=f"{dataset.config.condition}/{dataset.config.split}",
        name=name,
    )


def fit_frequency_ceiling(reference: EvaluationReference) -> MarginalBaseline:
    """Fit the marginal on the very data it will be scored on.

    The best any input-blind predictor can possibly do, because it is handed the
    evaluation set's own feature frequencies. Every retirement decision is made
    against this rather than against the train-fitted marginal: otherwise a
    metric could survive purely on train/test frequency shift, which is not
    computation either.

    It must be fitted on exactly the rows it is scored on. Fitting it on one
    split and scoring it on two is not a ceiling — it is a differently-wrong
    marginal, and it will read as *worse* than the honest baseline.
    """
    mean_value, active_rate = _fit_marginal_arrays(
        reference.targets[reference.supervised][:, reference.content_indices].astype(np.float64),
        reference.target_active[reference.supervised][:, reference.content_indices].astype(
            np.float64
        ),
    )
    return MarginalBaseline(
        mean_value=mean_value,
        active_rate=active_rate,
        content_indices=reference.content_indices,
        fitted_on=f"{reference.condition}/{reference.split} (evaluation split itself)",
        name="ceiling",
    )


# --------------------------------------------------------------------------- #
# Metric register: definition, kind, and the retirement decision
# --------------------------------------------------------------------------- #

MIN_SCORE_GAP = 0.25
"""A score metric must separate oracle from the frequency ceiling by at least
this much to be worth reporting."""

MAX_MARGINAL_SCORE = 0.50
"""...and the frequency ceiling must not itself reach a value a reader would
call success."""

MIN_LOSS_RATIO = 2.0
"""A loss metric must cost the frequency ceiling at least this many times what
it costs the oracle, measured against the metric's resolution floor."""


@dataclass(frozen=True)
class MetricSpec:
    """One ruler: what it means, how it is read, and whether it survived.

    ``status`` is a decision recorded in source, not recomputed at import time.
    ``--selftest`` re-measures the rule and fails if the decision and the
    evidence have come apart — which is what stops a retirement from being
    quietly reversed later by someone whose model does badly on the metric.
    """

    name: str
    kind: str
    definition: str
    floor: float
    status: str
    reason: str
    condition: str

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "definition": self.definition,
            "floor": self.floor,
            "status": self.status,
            "reason": self.reason,
            "calibrated_on": self.condition,
        }


METRIC_SPECS: tuple[MetricSpec, ...] = (
    MetricSpec(
        name="reconstruction_loss",
        kind="loss",
        definition="importance-weighted MSE over the content bank at supervised positions",
        floor=1e-4,
        status="retained",
        reason="the frequency ceiling pays the full target variance; the oracle pays zero",
        condition="T0",
    ),
    MetricSpec(
        name="feature_precision",
        kind="score",
        definition="micro precision of feature activity at the rate-matched operating point",
        floor=0.0,
        status="retained",
        reason="an input-blind predictor can only pick globally frequent features",
        condition="T0",
    ),
    MetricSpec(
        name="feature_recall",
        kind="score",
        definition="micro recall of feature activity at the rate-matched operating point",
        floor=0.0,
        status="retained",
        reason=(
            "retained only at the rate-matched operating point, where it equals precision; "
            "see recall_at_fixed_threshold for why the free-threshold form is retired"
        ),
        condition="T0",
    ),
    MetricSpec(
        name="feature_f1",
        kind="score",
        definition="micro F1 of feature activity at the rate-matched operating point",
        floor=0.0,
        status="retained",
        reason="headline detection number; equals precision and recall by rate-matching",
        condition="T0",
    ),
    MetricSpec(
        name="feature_macro_precision",
        kind="score",
        definition="precision averaged over content features, features with no selection dropped",
        floor=0.0,
        status="retained",
        reason="per-feature form required by §6.1; not reachable by frequency alone",
        condition="T0",
    ),
    MetricSpec(
        name="feature_macro_recall",
        kind="score",
        definition="recall averaged over content features, features never active dropped",
        floor=0.0,
        status="retained",
        reason="per-feature form required by §6.1; not reachable by frequency alone",
        condition="T0",
    ),
    MetricSpec(
        name="answer_set_accuracy",
        kind="score",
        definition="fraction of required operations whose answer set is recovered exactly",
        floor=0.0,
        status="retained",
        reason="exact set equality is unreachable for a predictor emitting one constant set",
        condition="T0",
    ),
    MetricSpec(
        name="associative_recall_accuracy",
        kind="score",
        definition="answer_set_accuracy restricted to transport operations",
        floor=0.0,
        status="retained",
        reason="the primary T1 capability number and the positive control's gate",
        condition="positive_control",
    ),
    MetricSpec(
        name="associative_recall_jaccard",
        kind="score",
        definition="graded answer-set overlap over transport operations",
        floor=0.0,
        status="retained",
        reason="distinguishes a blended retrieval from a missing one",
        condition="positive_control",
    ),
    MetricSpec(
        name="heldout_composition_accuracy",
        kind="score",
        definition="answer_set_accuracy on templates whose composition was held out",
        floor=0.0,
        status="retained",
        reason="same rule as answer_set_accuracy, restricted to unseen compositions",
        condition="T0",
    ),
    MetricSpec(
        name="brier",
        kind="loss",
        definition="mean squared error of activity probability against ground-truth activity",
        floor=1e-3,
        status="retained",
        reason="charges for the resolution the frequency ceiling does not have",
        condition="T0",
    ),
    MetricSpec(
        name="overwrite_accuracy",
        kind="score",
        definition="fraction of overwrite queries answered with the newest lawful value",
        floor=0.0,
        status="retained",
        reason="exact set equality against the new value; the stale set is a named wrong answer",
        condition="synthetic_overwrite",
    ),
    MetricSpec(
        name="stale_value_error_rate",
        kind="loss",
        definition="fraction of overwrite queries answered nearer the superseded value",
        floor=0.02,
        status="retained",
        reason="the frequency ceiling errs near half the time; the oracle never does",
        condition="synthetic_overwrite",
    ),
    MetricSpec(
        name="ece",
        kind="loss",
        definition="expected calibration error of activity probability, 10 equal-width bins",
        floor=0.01,
        status="retired",
        reason=(
            "the frequency ceiling emits each feature's true base rate and is therefore "
            "perfectly calibrated while computing nothing; oracle and ceiling are "
            "indistinguishable, so there is no headroom to measure. Kept as a diagnostic "
            "reported only beside brier, never as a headline and never gated on."
        ),
        condition="T0",
    ),
    MetricSpec(
        name="recall_at_free_threshold",
        kind="score",
        definition="the best micro recall reached over a 21-point threshold grid",
        floor=0.0,
        status="retired",
        reason=(
            "free at every sparsity: a threshold of zero selects every cell, so the "
            "frequency ceiling reaches recall 1.0 and ties the oracle. Retired in favour "
            "of the rate-matched operating point, where the threshold is not a free "
            "parameter and recall coincides with precision."
        ),
        condition="T0",
    ),
    MetricSpec(
        name="distance_degradation_slope",
        kind="diagnostic",
        definition="count-weighted slope of answer-set accuracy against source distance",
        floor=0.0,
        status="diagnostic",
        reason=(
            "a predictor that ignores the input degrades not at all, so the best possible "
            "slope is achieved at chance accuracy. Valid only reported beside its level."
        ),
        condition="capacity_stressed",
    ),
    MetricSpec(
        name="distractor_sensitivity_slope",
        kind="diagnostic",
        definition="count-weighted slope of answer-set accuracy against distractor count",
        floor=0.0,
        status="diagnostic",
        reason=(
            "same failure as the distance slope, and within one dataset the distractor "
            "count is constant, so the within-dataset slope is undefined by construction. "
            "The across-condition curve comes from sweep_curve."
        ),
        condition="capacity_stressed",
    ),
    MetricSpec(
        name="heldout_composition_gap",
        kind="diagnostic",
        definition="seen-composition accuracy minus held-out-composition accuracy",
        floor=0.0,
        status="diagnostic",
        reason=(
            "a predictor that has learned nothing generalises perfectly: its gap is zero. "
            "Valid only reported beside the held-out level."
        ),
        condition="T0",
    ),
)

METRIC_SPEC_BY_NAME: dict[str, MetricSpec] = {spec.name: spec for spec in METRIC_SPECS}
RETAINED_METRICS: tuple[str, ...] = tuple(s.name for s in METRIC_SPECS if s.status == "retained")
RETIRED_METRICS: tuple[str, ...] = tuple(s.name for s in METRIC_SPECS if s.status == "retired")
DIAGNOSTIC_METRICS: tuple[str, ...] = tuple(
    s.name for s in METRIC_SPECS if s.status == "diagnostic"
)

CEILING_DOMINATED_METRICS: tuple[str, ...] = (
    "reconstruction_loss",
    "brier",
    "feature_precision",
    "feature_recall",
    "feature_f1",
)
"""Metrics on which the frequency ceiling provably beats every other input-blind
predictor, so ``ceiling`` at least matching ``marginal`` is an invariant rather
than a hope.

The per-feature mean minimises squared error among per-feature constants; the
base rate minimises the Brier score among constants; and rate-matched micro
detection is maximised by ranking columns by their true activation rate, which
is exactly what the ceiling does.

Deliberately excluded, with reasons, because dominance is *not* provable there:

``feature_macro_precision`` / ``feature_macro_recall``
    Unweighted averages over features. For any column-constant predictor macro
    recall counts little more than *how many* columns were selected, which is
    nearly the same whichever columns those are — measured on T0 the ceiling
    lands a thousandth below the train-fitted marginal. The retention rule is
    untouched (oracle 1.0000 against a ceiling near 0.08), but the ordering is a
    coin flip and is not asserted.
``answer_set_accuracy`` and its restrictions
    Exact set equality is not the quantity the ceiling optimises; both
    references sit at 0.0000 and either could edge the other by chance.
"""

def evaluate_all(predictions: Predictions, reference: EvaluationReference) -> dict[str, MetricValue]:
    """Every §6.1 metric on one (predictions, reference) pair.

    Computed in one place so the T0 table, the calibration report, and the
    positive control cannot drift apart in what they mean by a metric name.
    """
    detection = feature_detection(predictions, reference)
    sweep = threshold_sweep(predictions, reference)
    steps_all = step_scores(predictions, reference)
    steps_recall = step_scores(predictions, reference, ops=RECALL_OPS)
    probs = calibration(predictions, reference)
    holdout = heldout_composition_accuracy(predictions, reference)
    distance = _bucket_curve(
        "distance_degradation", "distance", steps_recall.distance, steps_recall.exact
    )
    distractor = _bucket_curve(
        "distractor_sensitivity", "n_distractors", steps_recall.n_distractors, steps_recall.exact
    )

    n_cells = reference.n_supervised
    values: dict[str, MetricValue] = {
        "reconstruction_loss": reconstruction_loss(predictions, reference),
        "feature_precision": MetricValue(
            "feature_precision", detection.precision, detection.n_selected, detection.as_dict()
        ),
        "feature_recall": MetricValue(
            "feature_recall", detection.recall, detection.n_true, detection.as_dict()
        ),
        "feature_f1": MetricValue("feature_f1", detection.f1, detection.n_true, detection.as_dict()),
        "feature_macro_precision": MetricValue(
            "feature_macro_precision", detection.macro_precision, n_cells, {}
        ),
        "feature_macro_recall": MetricValue(
            "feature_macro_recall", detection.macro_recall, n_cells, {}
        ),
        "answer_set_accuracy": MetricValue(
            "answer_set_accuracy", _mean_or_none(steps_all.exact), len(steps_all),
            {"ops": steps_all.ops},
        ),
        "associative_recall_accuracy": MetricValue(
            "associative_recall_accuracy", _mean_or_none(steps_recall.exact), len(steps_recall),
            {"ops": steps_recall.ops},
        ),
        "associative_recall_jaccard": MetricValue(
            "associative_recall_jaccard", _mean_or_none(steps_recall.jaccard), len(steps_recall), {}
        ),
        "heldout_composition_accuracy": holdout,
        "heldout_composition_gap": MetricValue(
            "heldout_composition_gap", holdout.detail["gap"], holdout.n, {"level": holdout.value}
        ),
        "brier": MetricValue("brier", probs.brier, n_cells, {}),
        "ece": MetricValue("ece", probs.ece, n_cells, {}),
        "recall_at_free_threshold": MetricValue(
            "recall_at_free_threshold", sweep.best_recall, detection.n_true,
            {"best_precision": sweep.best_precision, "best_f1": sweep.best_f1},
        ),
        "overwrite_accuracy": overwrite_accuracy(predictions, reference),
        "stale_value_error_rate": stale_value_error_rate(predictions, reference),
        "distance_degradation_slope": MetricValue(
            "distance_degradation_slope", distance.slope, sum(distance.n),
            {"curve": distance.as_dict()},
        ),
        "distractor_sensitivity_slope": MetricValue(
            "distractor_sensitivity_slope", distractor.slope, sum(distractor.n),
            {"curve": distractor.as_dict()},
        ),
    }
    return values


def normalized_skill(
    raw: float | None, marginal: float | None, oracle: float | None
) -> float | None:
    """Rescale a raw metric so the marginal scores 0 and the oracle scores 1.

    One normaliser for every metric, loss or score, because the sign is carried
    by the two references rather than by a flag. This is the *fix* applied to
    every retained metric: a predictor that knows only feature frequencies
    scores exactly zero by construction, so no retained metric can make an
    architecture look competent for knowing nothing.

    Returns ``None`` when the two references coincide — an ill-conditioned
    normalisation is exactly the signal that the metric has no headroom, and it
    is reported rather than divided through.
    """
    if raw is None or marginal is None or oracle is None:
        return None
    denominator = oracle - marginal
    if abs(denominator) < 1e-12:
        return None
    return float((raw - marginal) / denominator)


def metric_rule(spec: MetricSpec, oracle: float | None, ceiling: float | None) -> tuple[bool, str]:
    """Is this metric worth keeping, given oracle and frequency-ceiling values?

    ``ceiling`` is the best an input-blind predictor can do — the marginal
    fitted on the evaluation split itself. Using the ceiling rather than the
    train-fitted marginal closes the loophole where a metric survives only
    because train and test frequencies differ.
    """
    if spec.kind == "diagnostic":
        return True, "diagnostic; exempt from the rule and never reported alone"
    if oracle is None or ceiling is None:
        return False, "not measurable on the calibration condition"
    if spec.kind == "score":
        if not oracle > ceiling:
            return False, f"frequency ceiling {ceiling:.4f} >= oracle {oracle:.4f}"
        gap = oracle - ceiling
        if gap < MIN_SCORE_GAP:
            return False, f"oracle-ceiling gap {gap:.4f} < {MIN_SCORE_GAP}"
        if ceiling > MAX_MARGINAL_SCORE:
            return False, f"frequency ceiling alone scores {ceiling:.4f} > {MAX_MARGINAL_SCORE}"
        return True, f"gap {gap:.4f}, ceiling {ceiling:.4f}"
    if not ceiling > oracle:
        return False, f"frequency ceiling {ceiling:.5f} <= oracle {oracle:.5f}"
    reference = max(oracle, spec.floor)
    if ceiling < MIN_LOSS_RATIO * reference:
        return False, (
            f"ceiling {ceiling:.5f} < {MIN_LOSS_RATIO}x max(oracle {oracle:.5f}, "
            f"floor {spec.floor})"
        )
    return True, f"ceiling/max(oracle, floor) = {ceiling / reference:.1f}x"


# --------------------------------------------------------------------------- #
# The known-easy positive control
# --------------------------------------------------------------------------- #

POSITIVE_CONTROL_CONDITION = "positive_control"
POSITIVE_CONTROL_METRIC = "associative_recall_accuracy"
POSITIVE_CONTROL_THRESHOLD = 0.80
"""Frozen. Derived from the measured oracle (1.0000) and frequency ceiling
(0.0000) on this condition as ``ceiling + 0.8 * (oracle - ceiling)``, then
written down as a literal so that later missions inherit a fixed bar rather
than one recomputed from whatever data they happen to have. ``--selftest``
re-derives it and fails if the references have moved."""

POSITIVE_CONTROL_EXAMPLES = 512
_POSITIVE_CONTROL_ORACLE_MARGIN = 0.15
_POSITIVE_CONTROL_MARGINAL_MARGIN = 0.50


@dataclass(frozen=True)
class PositiveControlResult:
    """Pass or fail on the known-easy task, with the evidence attached.

    §7.3's R1 has two halves: the model must solve the tiny task, and its
    mechanism must become active. This object is the **capability half only**.
    Mechanism activity is §6.3 and belongs to the prompt that owns the
    mechanism; a green result here is necessary and not sufficient.

    ``instrument_ok`` is the self-check: the oracle and the frequency ceiling
    are recomputed on the same data the candidate saw, and if they no longer
    straddle the threshold then the ruler is broken and ``passed`` says nothing
    about the model.
    """

    passed: bool
    instrument_ok: bool
    metric: str
    value: float | None
    threshold: float
    oracle_value: float | None
    marginal_value: float | None
    random_value: float | None
    oracle_margin: float | None
    marginal_margin: float | None
    skill: float | None
    condition: str
    n_examples: int
    metric_version: str = METRIC_VERSION
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        record = {
            "passed": self.passed,
            "instrument_ok": self.instrument_ok,
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "oracle_value": self.oracle_value,
            "marginal_value": self.marginal_value,
            "random_value": self.random_value,
            "oracle_margin": self.oracle_margin,
            "marginal_margin": self.marginal_margin,
            "skill": self.skill,
            "condition": self.condition,
            "n_examples": self.n_examples,
            "metric_version": self.metric_version,
        }
        record.update(self.detail)
        return record

    def summary(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        if not self.instrument_ok:
            verdict = "INVALID"
        value = "n/a" if self.value is None else f"{self.value:.4f}"
        return (
            f"positive control {verdict}: {self.metric}={value} vs threshold "
            f"{self.threshold:.2f} (oracle {self.oracle_value}, marginal {self.marginal_value})"
        )


def positive_control_datasets(
    *, n_examples: int = POSITIVE_CONTROL_EXAMPLES, seed: int | None = None
) -> tuple[FeatureProgramDataset, FeatureProgramDataset]:
    """The known-easy train and eval splits, as one named object.

    §4.4's known-easy control: ample dimension (``d_recommended=48 > F=36``),
    distance one to two, no distractors, one association, no key collisions.
    Callers train on the first and evaluate on the second; both come from here
    so that no mission can accidentally evaluate the positive control on a task
    it quietly made easier.
    """
    overrides: dict = {"n_examples": n_examples}
    if seed is not None:
        overrides["seed"] = seed
    train = generate_dataset(condition_config(POSITIVE_CONTROL_CONDITION, split="train", **overrides))
    evaluation = generate_dataset(
        condition_config(POSITIVE_CONTROL_CONDITION, split="test", **overrides)
    )
    return train, evaluation


def positive_control(
    predict: Callable[[FeatureProgramDataset], Predictions],
    *,
    n_examples: int = POSITIVE_CONTROL_EXAMPLES,
    seed: int | None = None,
) -> PositiveControlResult:
    """Run the known-easy positive control and return pass or fail.

    Later missions call this and get a verdict, not a number to interpret. The
    threshold is frozen in :data:`POSITIVE_CONTROL_THRESHOLD`; the oracle and
    the frequency ceiling are measured alongside on the same data, so the result
    carries its own evidence that the bar is still in the right place.

    ``predict`` receives the evaluation dataset and returns
    :class:`Predictions`. Use :func:`positive_control_datasets` to obtain the
    matching training split first — the candidate must be trained on the same
    condition it is judged on.
    """
    train, evaluation = positive_control_datasets(n_examples=n_examples, seed=seed)
    reference = EvaluationReference.from_dataset(evaluation)

    marginal = fit_marginal(train)
    oracle = ProgramOracle(fallback=marginal)
    random_baseline = RandomBaseline.fit(train)

    def score(predictions: Predictions) -> float | None:
        return associative_recall_accuracy(predictions, reference).value

    candidate = score(predict(evaluation))
    oracle_value = score(oracle.predict(evaluation))
    marginal_value = score(marginal.predict(evaluation))
    random_value = score(random_baseline.predict(evaluation))

    instrument_ok = (
        oracle_value is not None
        and marginal_value is not None
        and oracle_value >= POSITIVE_CONTROL_THRESHOLD + _POSITIVE_CONTROL_ORACLE_MARGIN
        and marginal_value <= POSITIVE_CONTROL_THRESHOLD - _POSITIVE_CONTROL_MARGINAL_MARGIN
    )
    passed = bool(
        instrument_ok and candidate is not None and candidate >= POSITIVE_CONTROL_THRESHOLD
    )
    return PositiveControlResult(
        passed=passed,
        instrument_ok=bool(instrument_ok),
        metric=POSITIVE_CONTROL_METRIC,
        value=candidate,
        threshold=POSITIVE_CONTROL_THRESHOLD,
        oracle_value=oracle_value,
        marginal_value=marginal_value,
        random_value=random_value,
        oracle_margin=None if oracle_value is None else oracle_value - POSITIVE_CONTROL_THRESHOLD,
        marginal_margin=(
            None if marginal_value is None else POSITIVE_CONTROL_THRESHOLD - marginal_value
        ),
        skill=normalized_skill(candidate, marginal_value, oracle_value),
        condition=POSITIVE_CONTROL_CONDITION,
        n_examples=n_examples,
        detail={
            "d_recommended": evaluation.config.d_recommended,
            "n_features": reference.inputs.shape[2],
            "generator_version": evaluation.generator_version,
            "dataset_hash": evaluation.content_hash,
            "note": "capability half of R1 only; mechanism activity is §6.3",
        },
    )


# --------------------------------------------------------------------------- #
# A synthetic overwrite fixture, so the T2 metrics are calibrated before T2
# --------------------------------------------------------------------------- #


def synthetic_overwrite_reference(
    *, n_examples: int = 256, n_content: int = 12, seed: int = 20260809, split: str = "test"
) -> EvaluationReference:
    """A hand-built T2 reference: one key bound twice, the later binding lawful.

    Six positions. Position 1 holds the stale value, position 3 the corrected
    value, position 5 is the query and the only supervised position. Built by
    hand rather than by the generator because T2 is prompt 18's family, and the
    point of writing these metrics now is that they must be testable before the
    mechanism they judge exists.
    """
    rng = np.random.default_rng(seed)
    seq_len, stale_at, source_at, query_at = 6, 1, 3, 5
    n_features = n_content
    inputs = np.zeros((n_examples, seq_len, n_features), dtype=np.float32)
    active = np.zeros((n_examples, seq_len, n_features), dtype=bool)
    targets = np.zeros_like(inputs)
    target_active = np.zeros_like(active)
    supervised = np.zeros((n_examples, seq_len), dtype=bool)
    programs: list[ProgramRecord] = []

    for i in range(n_examples):
        # The correction must actually correct something. Drawing the two active
        # sets independently lets them coincide (once in ~256 examples at this
        # size), and an "overwrite" whose new value equals its old one is
        # degenerate: a memory that never erases would score correct on it. So
        # the fixture rejects those draws rather than letting them quietly put a
        # floor under the stale-value error rate.
        for _ in range(64):
            stale_on = rng.random(n_content) < 0.25
            new_on = rng.random(n_content) < 0.25
            if not stale_on.any():
                stale_on[rng.integers(n_content)] = True
            if not new_on.any():
                new_on[rng.integers(n_content)] = True
            if not np.array_equal(stale_on, new_on):
                break
        else:  # pragma: no cover - unreachable for any sane n_content
            raise CapabilityMetricError("could not draw distinct stale and corrected values")
        inputs[i, stale_at] = (rng.random(n_content) * stale_on).astype(np.float32)
        active[i, stale_at] = stale_on
        inputs[i, source_at] = (rng.random(n_content) * new_on).astype(np.float32)
        active[i, source_at] = new_on
        targets[i, query_at] = inputs[i, source_at]
        target_active[i, query_at] = new_on
        supervised[i, query_at] = True
        programs.append(
            ProgramRecord(
                example_index=i,
                family="T2",
                condition="synthetic_overwrite",
                split=split,
                template_id="synthetic-overwrite",
                composition=("overwrite_recall",),
                seq_len=seq_len,
                positions=(),
                steps=(
                    OverwriteStep(
                        op="overwrite_recall",
                        dest=query_at,
                        source=source_at,
                        key_id=0,
                        distance=query_at - source_at,
                        distractors=(4,),
                        answer_group=0,
                        answer_features=tuple(int(f) for f in np.nonzero(new_on)[0]),
                        information_destroyed=False,
                        stale_source=stale_at,
                        stale_answer_features=tuple(int(f) for f in np.nonzero(stale_on)[0]),
                    ),
                ),
                supervised_positions=(query_at,),
            )
        )

    return EvaluationReference(
        family="T2",
        condition="synthetic_overwrite",
        split=split,
        inputs=inputs,
        input_active=active,
        targets=targets,
        target_active=target_active,
        supervised=supervised,
        importance=np.ones(n_features, dtype=np.float64),
        content_indices=np.arange(n_content, dtype=np.int64),
        programs=tuple(programs),
        heldout_template_ids=frozenset(),
    )


def _reference_predictions(reference: EvaluationReference, kind: str, seed: int = 20260809):
    """Oracle / marginal / random predictions for a hand-built reference.

    The generator-backed baselines take a :class:`FeatureProgramDataset`; the
    synthetic overwrite fixture has no dataset behind it, so its three
    references are constructed directly here from the same definitions.
    """
    content = reference.content_indices
    shape = reference.inputs.shape
    values = np.zeros(shape, dtype=np.float64)
    prob = np.zeros(shape, dtype=np.float64)
    if kind == "oracle":
        for row, record in enumerate(reference.programs):
            for step in record.steps:
                if step.source is None:
                    continue
                values[row, step.dest, content] = reference.inputs[row, step.source][content]
                prob[row, step.dest, content] = reference.input_active[row, step.source][
                    content
                ].astype(np.float64)
    elif kind in ("marginal", "ceiling"):
        mask = reference.supervised
        mean_value = reference.targets[mask][:, content].astype(np.float64).mean(axis=0)
        rate = reference.target_active[mask][:, content].astype(np.float64).mean(axis=0)
        values[:, :, content] = mean_value
        prob[:, :, content] = rate
    elif kind == "random":
        mask = reference.supervised
        rate = float(reference.target_active[mask][:, content].mean())
        rng = np.random.default_rng(seed)
        draws = rng.random((shape[0], shape[1], content.size))
        magnitudes = rng.random((shape[0], shape[1], content.size))
        values[:, :, content] = magnitudes * (draws < rate)
        prob[:, :, content] = rate
    elif kind == "stale":
        for row, record in enumerate(reference.programs):
            for step in record.steps:
                stale = getattr(step, "stale_source", None)
                if stale is None:
                    continue
                values[row, step.dest, content] = reference.inputs[row, stale][content]
                prob[row, step.dest, content] = reference.input_active[row, stale][content].astype(
                    np.float64
                )
    else:  # pragma: no cover
        raise CapabilityMetricError(f"unknown reference predictor {kind!r}")
    return Predictions(values=values, active_prob=prob)


# --------------------------------------------------------------------------- #
# Calibration: measure every metric against every reference
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ConditionScores:
    """Every metric, under every reference, on one condition."""

    condition: str
    family: str
    n_examples: int
    n_supervised: int
    scores: dict[str, dict[str, float | None]]

    def as_dict(self) -> dict:
        return {
            "condition": self.condition,
            "family": self.family,
            "n_examples": self.n_examples,
            "n_supervised": self.n_supervised,
            "scores": self.scores,
        }


@dataclass(frozen=True)
class MetricVerdict:
    name: str
    condition: str
    status: str
    oracle: float | None
    random: float | None
    marginal: float | None
    ceiling: float | None
    marginal_skill: float | None
    rule_passed: bool
    rule_reason: str
    agrees: bool

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "condition": self.condition,
            "status": self.status,
            "oracle": self.oracle,
            "random": self.random,
            "marginal": self.marginal,
            "frequency_ceiling": self.ceiling,
            "marginal_skill": self.marginal_skill,
            "rule_passed": self.rule_passed,
            "rule_reason": self.rule_reason,
            "agrees_with_recorded_status": self.agrees,
        }


REFERENCE_NAMES: tuple[str, ...] = ("oracle", "random", "marginal", "ceiling")

_CALIBRATION_CONDITIONS: tuple[str, ...] = ("T0", "positive_control", "capacity_stressed")


def _condition_datasets(name: str, n_examples: int):
    if name == "T0":
        train = generate_dataset(t0_config(n_examples=n_examples, split="train"))
        evaluation = generate_dataset(t0_config(n_examples=n_examples, split="test"))
    else:
        train = generate_dataset(condition_config(name, n_examples=n_examples, split="train"))
        evaluation = generate_dataset(condition_config(name, n_examples=n_examples, split="test"))
    return train, evaluation


def score_condition(name: str, *, n_examples: int = 256) -> tuple[ConditionScores, dict]:
    """Measure every metric under all four references on one condition.

    Returns the scores and the raw pieces (reference object and per-reference
    metric dicts) so callers that need more than the table — the selftest's
    invariants, the T0 harness — do not have to regenerate the data.
    """
    if name == "synthetic_overwrite":
        reference = synthetic_overwrite_reference(n_examples=n_examples)
        predictions = {kind: _reference_predictions(reference, kind) for kind in REFERENCE_NAMES}
        predictions["stale"] = _reference_predictions(reference, "stale")
        family = "T2"
    else:
        train, evaluation = _condition_datasets(name, n_examples)
        # Seen and held-out templates must live in one reference for the
        # composition metrics to have anything to compare.
        reference = EvaluationReference.concat(
            [EvaluationReference.from_dataset(train), EvaluationReference.from_dataset(evaluation)]
        )
        marginal = fit_marginal(train)
        both = [train, evaluation]
        predictions = {
            "oracle": Predictions.concat(
                [ProgramOracle(fallback=marginal).predict(ds) for ds in both]
            ),
            "random": Predictions.concat([RandomBaseline.fit(train).predict(ds) for ds in both]),
            "marginal": Predictions.concat([marginal.predict(ds) for ds in both]),
            "ceiling": fit_frequency_ceiling(reference).predict_like(reference),
        }
        family = train.config.family

    measured = {kind: evaluate_all(predictions[kind], reference) for kind in predictions}
    metric_names = tuple(METRIC_SPEC_BY_NAME)
    scores = {
        metric: {kind: measured[kind][metric].value for kind in measured} for metric in metric_names
    }
    condition_scores = ConditionScores(
        condition=name,
        family=family,
        n_examples=reference.n_examples,
        n_supervised=reference.n_supervised,
        scores=scores,
    )
    return condition_scores, {
        "reference": reference,
        "measured": measured,
        "predictions": predictions,
    }


@dataclass(frozen=True)
class CalibrationReport:
    metric_version: str
    conditions: tuple[ConditionScores, ...]
    verdicts: tuple[MetricVerdict, ...]
    positive_control: dict
    notes: dict

    @property
    def ok(self) -> bool:
        return all(v.agrees for v in self.verdicts)

    def as_dict(self) -> dict:
        return {
            "metric_version": METRIC_VERSION,
            "conditions": [c.as_dict() for c in self.conditions],
            "verdicts": [v.as_dict() for v in self.verdicts],
            "positive_control": self.positive_control,
            "notes": self.notes,
            "ok": self.ok,
        }


def calibrate(*, n_examples: int = 256) -> CalibrationReport:
    """Measure every metric against oracle, random, marginal, and ceiling.

    This is the whole point of the mission: the rulers are graduated against
    known answers before there is anything to measure with them. Nothing here
    trains, and the only randomness is the seeded chance baseline.
    """
    conditions: list[ConditionScores] = []
    by_condition: dict[str, dict] = {}
    for name in (*_CALIBRATION_CONDITIONS, "synthetic_overwrite"):
        scores, pieces = score_condition(name, n_examples=n_examples)
        conditions.append(scores)
        by_condition[name] = {"scores": scores, **pieces}

    verdicts: list[MetricVerdict] = []
    for spec in METRIC_SPECS:
        scores = by_condition[spec.condition]["scores"].scores[spec.name]
        passed, reason = metric_rule(spec, scores["oracle"], scores["ceiling"])
        expected = spec.status in ("retained", "diagnostic")
        verdicts.append(
            MetricVerdict(
                name=spec.name,
                condition=spec.condition,
                status=spec.status,
                oracle=scores["oracle"],
                random=scores["random"],
                marginal=scores["marginal"],
                ceiling=scores["ceiling"],
                marginal_skill=normalized_skill(
                    scores["marginal"], scores["marginal"], scores["oracle"]
                ),
                rule_passed=passed,
                rule_reason=reason,
                agrees=passed == expected,
            )
        )

    control = _positive_control_calibration(n_examples=n_examples)
    stale = by_condition["synthetic_overwrite"]
    stale_predictions = _reference_predictions(stale["reference"], "stale")
    t0 = by_condition["T0"]
    ceiling_sweep = threshold_sweep(t0["predictions"]["ceiling"], t0["reference"])
    oracle_sweep = threshold_sweep(t0["predictions"]["oracle"], t0["reference"])
    notes = {
        "stale_copier_overwrite_accuracy": overwrite_accuracy(
            stale_predictions, stale["reference"]
        ).value,
        "stale_copier_stale_value_error_rate": stale_value_error_rate(
            stale_predictions, stale["reference"]
        ).value,
        "distance_slope_marginal_vs_oracle": _slope_note(by_condition["capacity_stressed"]),
        "distractor_sensitivity_sweep": _distractor_sweep(n_examples=min(n_examples, 128)),
        # Why the retirement rule is written against the ceiling and not the
        # train-fitted marginal. On a coverage-preserving compositional split
        # the held-out templates are exactly the ones training under-represents,
        # so training frequency is an actively misleading predictor of
        # evaluation frequency and the honest marginal scores *below* chance.
        "T0_train_eval_frequency_correlation": _train_eval_frequency_correlation(
            by_condition["T0"]
        ),
        # What the frequency ceiling can reach if it is allowed to pick its own
        # operating point. Recall is free; precision and F1 are not.
        "T0_best_over_thresholds": {
            "ceiling": {
                "recall": ceiling_sweep.best_recall,
                "precision": ceiling_sweep.best_precision,
                "f1": ceiling_sweep.best_f1,
            },
            "oracle": {
                "recall": oracle_sweep.best_recall,
                "precision": oracle_sweep.best_precision,
                "f1": oracle_sweep.best_f1,
            },
        },
    }
    return CalibrationReport(
        metric_version=METRIC_VERSION,
        conditions=tuple(conditions),
        verdicts=tuple(verdicts),
        positive_control=control,
        notes=notes,
    )


def _distractor_sweep(*, n_examples: int, values: Sequence[int] = (0, 1, 2, 3)) -> dict:
    """Recall accuracy against distractor count, across conditions.

    Within one dataset ``n_distractors`` is a single configured value, so the
    within-dataset curve has one point. This is the across-condition form, built
    with :func:`sweep_curve`, and it is what makes the "a flat curve proves
    nothing" warning on :func:`distractor_sensitivity` a measurement rather than
    a claim: both references are flat, at opposite ends of the scale.
    """
    base = condition_config("capacity_stressed")

    def oracle_predict(dataset: FeatureProgramDataset) -> Predictions:
        return ProgramOracle().predict(dataset)

    def marginal_predict(dataset: FeatureProgramDataset) -> Predictions:
        train = generate_dataset(replace(dataset.config, split="train"))
        return fit_marginal(train).predict(dataset)

    return {
        kind: sweep_curve(
            base=base,
            axis="n_distractors",
            values=values,
            predict=predictor,
            metric=associative_recall_accuracy,
            n_examples=n_examples,
        ).as_dict()
        for kind, predictor in (("oracle", oracle_predict), ("marginal", marginal_predict))
    }


def _train_eval_frequency_correlation(pieces: dict) -> float | None:
    """Correlation between the training feature marginals and the scored ones.

    Near +1 would mean the marginal baseline is a fair adversary. Strongly
    negative means the split has made training frequency misleading, and that a
    model beating the train-fitted marginal has cleared a bar lower than chance.
    """
    reference = pieces["reference"]
    train_rate = pieces["predictions"]["marginal"].active_prob[0, 0][reference.content_indices]
    scored_rate = fit_frequency_ceiling(reference).active_rate
    if train_rate.std() == 0 or scored_rate.std() == 0:
        return None
    return float(np.corrcoef(train_rate, scored_rate)[0, 1])


def _slope_note(pieces: dict) -> dict:
    measured = pieces["measured"]
    return {
        kind: {
            "slope": measured[kind]["distance_degradation_slope"].value,
            "level": measured[kind]["associative_recall_accuracy"].value,
        }
        for kind in ("oracle", "marginal", "ceiling")
    }


def _positive_control_calibration(*, n_examples: int) -> dict:
    train, evaluation = positive_control_datasets(n_examples=n_examples)
    reference = EvaluationReference.from_dataset(evaluation)
    marginal = fit_marginal(train)
    references = {
        "oracle": ProgramOracle(fallback=marginal).predict(evaluation),
        "random": RandomBaseline.fit(train).predict(evaluation),
        "marginal": marginal.predict(evaluation),
        "ceiling": fit_frequency_ceiling(reference).predict_like(reference),
    }
    values = {
        kind: associative_recall_accuracy(prediction, reference).value
        for kind, prediction in references.items()
    }
    verdicts = {
        kind: positive_control(lambda _ds, p=prediction: p, n_examples=n_examples).passed
        for kind, prediction in references.items()
    }
    oracle, ceiling = values["oracle"], values["ceiling"]
    derived = None if oracle is None or ceiling is None else ceiling + 0.8 * (oracle - ceiling)
    return {
        "metric": POSITIVE_CONTROL_METRIC,
        "threshold": POSITIVE_CONTROL_THRESHOLD,
        "derived_threshold": derived,
        "values": values,
        "verdicts": verdicts,
        "oracle_margin": None if oracle is None else oracle - POSITIVE_CONTROL_THRESHOLD,
        "marginal_margin": (
            None if values["marginal"] is None else POSITIVE_CONTROL_THRESHOLD - values["marginal"]
        ),
        "ceiling_margin": None if ceiling is None else POSITIVE_CONTROL_THRESHOLD - ceiling,
        "n_examples": n_examples,
    }


# --------------------------------------------------------------------------- #
# T0 end to end
# --------------------------------------------------------------------------- #


def run_t0_evaluation(*, n_examples: int = 256) -> dict:
    """T0 from generation to metric table, with no model anywhere.

    This is the measurement path proving itself: generate the transport-free
    task, apply each reference predictor, compute every §6.1 metric, print the
    table. When prompt 04 arrives with an architecture, the only new thing in
    this path is where the predictions come from.
    """
    scores, pieces = score_condition("T0", n_examples=n_examples)
    reference = pieces["reference"]
    predictions = pieces["predictions"]
    skills = {
        metric: {
            kind: normalized_skill(
                scores.scores[metric][kind],
                scores.scores[metric]["marginal"],
                scores.scores[metric]["oracle"],
            )
            for kind in REFERENCE_NAMES
        }
        for metric in scores.scores
    }
    return {
        "metric_version": METRIC_VERSION,
        "condition": scores.as_dict(),
        "skills": skills,
        "detection": {
            kind: feature_detection(predictions[kind], reference).as_dict()
            for kind in REFERENCE_NAMES
        },
        "reliability": {
            kind: calibration(predictions[kind], reference).reliability.as_dict()
            for kind in REFERENCE_NAMES
        },
        "n_supervised": reference.n_supervised,
        "table": format_scores_table(scores),
    }


def format_scores_table(scores: ConditionScores) -> str:
    """One condition, one row per metric, one column per reference."""
    header = (
        f"{'metric':<30} {'status':<11} {'oracle':>9} {'marginal':>9} {'ceiling':>9} "
        f"{'random':>9} {'m-skill':>8}"
    )
    lines = [
        (
            f"condition={scores.condition} family={scores.family} "
            f"examples={scores.n_examples} supervised={scores.n_supervised}"
        ),
        header,
        "-" * len(header),
    ]
    for spec in METRIC_SPECS:
        row = scores.scores[spec.name]
        skill = normalized_skill(row["marginal"], row["marginal"], row["oracle"])
        lines.append(
            f"{spec.name:<30} {spec.status:<11} {_fmt(row['oracle']):>9} "
            f"{_fmt(row['marginal']):>9} {_fmt(row['ceiling']):>9} {_fmt(row['random']):>9} "
            f"{_fmt(skill):>8}"
        )
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return f"{value:.4f}"


def format_verdict_table(report: CalibrationReport) -> str:
    header = f"{'metric':<30} {'status':<11} {'rule':<5} {'agrees':<7} reason"
    lines = [header, "-" * len(header)]
    for verdict in report.verdicts:
        lines.append(
            f"{verdict.name:<30} {verdict.status:<11} "
            f"{'pass' if verdict.rule_passed else 'fail':<5} "
            f"{'yes' if verdict.agrees else 'NO':<7} {verdict.rule_reason}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Selftest
# --------------------------------------------------------------------------- #

INVARIANTS: tuple[str, ...] = (
    "oracle_reaches_the_ceiling",
    "frequency_ceiling_beats_chance",
    "frequency_ceiling_dominates_the_marginal",
    "skill_is_normalized",
    "retained_metrics_beat_the_frequency_ceiling",
    "retired_metrics_fail_the_rule",
    "slope_diagnostics_flatter_for_the_marginal",
    "positive_control_threshold_separates",
    "positive_control_verdicts",
    "overwrite_metrics_name_the_stale_answer",
    "metrics_follow_the_permutation",
)

_SELFTEST_EXAMPLES = 256


class _Checks:
    """Collects pass/fail with a reason, so the selftest reports every failure.

    Deliberately a local copy of the generator selftest's helper rather than an
    import of its private class; fifteen lines duplicated is cheaper than a
    cross-module private dependency between two independent gates.
    """

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


def run_selftest(
    *, break_invariant: str | None = None, verbose: bool = True, n_examples: int = _SELFTEST_EXAMPLES
) -> int:
    """Re-measure every calibration decision and every threshold.

    The retirement decisions live in :data:`METRIC_SPECS`; this recomputes the
    rule that produced them and fails if the source and the evidence have come
    apart in either direction — a retired metric that now passes is as much a
    failure as a retained metric that now fails.
    """
    checks = _Checks(break_invariant)
    report = calibrate(n_examples=n_examples)
    out: list[str] = [f"capability metric selftest — {METRIC_VERSION}", ""]

    by_condition = {c.condition: c for c in report.conditions}

    ceiling_metrics = ("feature_f1", "feature_macro_precision", "answer_set_accuracy")
    t0 = by_condition["T0"].scores
    control = by_condition["positive_control"].scores
    exact = {
        "T0 reconstruction_loss": t0["reconstruction_loss"]["oracle"],
        "T0 feature_f1": t0["feature_f1"]["oracle"],
        "T0 answer_set_accuracy": t0["answer_set_accuracy"]["oracle"],
        "PC associative_recall_accuracy": control["associative_recall_accuracy"]["oracle"],
        "PC brier": control["brier"]["oracle"],
    }
    checks.record(
        "oracle_reaches_the_ceiling",
        exact["T0 reconstruction_loss"] == 0.0
        and exact["T0 feature_f1"] == 1.0
        and exact["T0 answer_set_accuracy"] == 1.0
        and exact["PC associative_recall_accuracy"] == 1.0
        and exact["PC brier"] == 0.0,
        ", ".join(f"{k}={_fmt(v)}" for k, v in exact.items()),
    )

    beats = {metric: (t0[metric]["ceiling"], t0[metric]["random"]) for metric in ceiling_metrics}
    checks.record(
        "frequency_ceiling_beats_chance",
        all(c >= r for c, r in beats.values()),
        "; ".join(f"{k}: ceiling {_fmt(c)} >= random {_fmt(r)}" for k, (c, r) in beats.items()),
    )

    # The train-fitted marginal is deliberately *not* asserted to beat chance.
    # On a compositional split it demonstrably does not — see the recorded
    # train/eval frequency correlation. The ceiling, fitted on the rows it is
    # scored on, is the only input-blind reference with a guaranteed ordering,
    # which is exactly why the retirement rule is written against it.
    dominated = {
        v.name: (v.ceiling, v.marginal)
        for v in report.verdicts
        if v.name in CEILING_DOMINATED_METRICS
        and v.ceiling is not None
        and v.marginal is not None
    }
    better = {
        name: (
            ceiling <= marginal + 1e-9
            if METRIC_SPEC_BY_NAME[name].kind == "loss"
            else ceiling >= marginal - 1e-9
        )
        for name, (ceiling, marginal) in dominated.items()
    }
    checks.record(
        "frequency_ceiling_dominates_the_marginal",
        all(better.values()),
        f"{sum(better.values())}/{len(better)} retained metrics dominated; violations: "
        f"{[n for n, ok in better.items() if not ok] or 'none'}; "
        f"T0 train/eval feature-frequency correlation = "
        f"{_fmt(report.notes['T0_train_eval_frequency_correlation'])}",
    )

    skills = [
        (v.name, v.marginal_skill)
        for v in report.verdicts
        if v.status == "retained"
    ]
    checks.record(
        "skill_is_normalized",
        all(s is not None and abs(s) < 1e-9 for _, s in skills),
        f"{len(skills)} retained metrics; marginal skill max |value| = "
        f"{max((abs(s) for _, s in skills if s is not None), default=float('nan')):.2e}",
    )

    retained = [v for v in report.verdicts if v.status == "retained"]
    checks.record(
        "retained_metrics_beat_the_frequency_ceiling",
        all(v.rule_passed for v in retained),
        "; ".join(
            f"{v.name}: {'pass' if v.rule_passed else 'FAIL — ' + v.rule_reason}" for v in retained
        ),
    )

    retired = [v for v in report.verdicts if v.status == "retired"]
    checks.record(
        "retired_metrics_fail_the_rule",
        all(not v.rule_passed for v in retired),
        "; ".join(f"{v.name}: {v.rule_reason}" for v in retired),
    )

    slopes = report.notes["distance_slope_marginal_vs_oracle"]
    marginal_slope = slopes["marginal"]["slope"]
    oracle_slope = slopes["oracle"]["slope"]
    marginal_level = slopes["marginal"]["level"]
    oracle_level = slopes["oracle"]["level"]
    sweep = report.notes["distractor_sensitivity_sweep"]
    checks.record(
        "slope_diagnostics_flatter_for_the_marginal",
        marginal_slope is not None
        and oracle_slope is not None
        and abs(marginal_slope) <= abs(oracle_slope) + 1e-12
        and marginal_level < 0.05
        and oracle_level > 0.95
        and abs(sweep["marginal"]["slope"]) <= abs(sweep["oracle"]["slope"]) + 1e-12
        and max(sweep["marginal"]["y"]) < 0.05
        and min(sweep["oracle"]["y"]) > 0.95,
        f"distance: marginal slope {_fmt(marginal_slope)} at level {_fmt(marginal_level)}, "
        f"oracle slope {_fmt(oracle_slope)} at level {_fmt(oracle_level)}; "
        f"distractors {list(sweep['oracle']['x'])}: marginal slope "
        f"{_fmt(sweep['marginal']['slope'])} at {sweep['marginal']['y']}, oracle slope "
        f"{_fmt(sweep['oracle']['slope'])} at {sweep['oracle']['y']} — a flat curve at "
        "chance is why a slope is never a headline",
    )

    control_report = report.positive_control
    oracle_value = control_report["values"]["oracle"]
    marginal_value = control_report["values"]["marginal"]
    ceiling_value = control_report["values"]["ceiling"]
    checks.record(
        "positive_control_threshold_separates",
        oracle_value >= POSITIVE_CONTROL_THRESHOLD + _POSITIVE_CONTROL_ORACLE_MARGIN
        and marginal_value <= POSITIVE_CONTROL_THRESHOLD - _POSITIVE_CONTROL_MARGINAL_MARGIN
        and ceiling_value <= POSITIVE_CONTROL_THRESHOLD - _POSITIVE_CONTROL_MARGINAL_MARGIN,
        f"threshold {POSITIVE_CONTROL_THRESHOLD}; oracle {_fmt(oracle_value)} "
        f"(margin +{_fmt(control_report['oracle_margin'])}); marginal {_fmt(marginal_value)} "
        f"(margin -{_fmt(control_report['marginal_margin'])}); ceiling {_fmt(ceiling_value)}",
    )
    checks.record(
        "positive_control_verdicts",
        control_report["verdicts"]["oracle"]
        and not control_report["verdicts"]["marginal"]
        and not control_report["verdicts"]["random"]
        and not control_report["verdicts"]["ceiling"],
        f"oracle passes={control_report['verdicts']['oracle']}, "
        f"marginal passes={control_report['verdicts']['marginal']}, "
        f"random passes={control_report['verdicts']['random']}, "
        f"ceiling passes={control_report['verdicts']['ceiling']}",
    )

    overwrite = by_condition["synthetic_overwrite"].scores
    stale_copier_accuracy = report.notes["stale_copier_overwrite_accuracy"]
    stale_copier_rate = report.notes["stale_copier_stale_value_error_rate"]
    checks.record(
        "overwrite_metrics_name_the_stale_answer",
        overwrite["overwrite_accuracy"]["oracle"] == 1.0
        and overwrite["stale_value_error_rate"]["oracle"] == 0.0
        and stale_copier_accuracy == 0.0
        and stale_copier_rate == 1.0,
        f"oracle: accuracy {_fmt(overwrite['overwrite_accuracy']['oracle'])}, "
        f"stale rate {_fmt(overwrite['stale_value_error_rate']['oracle'])}; "
        f"a predictor that returns the superseded value: accuracy {_fmt(stale_copier_accuracy)}, "
        f"stale rate {_fmt(stale_copier_rate)}",
    )

    base, permuted = _permutation_pair(n_examples=n_examples)
    checks.record(
        "metrics_follow_the_permutation",
        permuted["invariant"] and not permuted["blind"],
        f"permuting dataset and predictions together leaves every metric unchanged="
        f"{permuted['invariant']} (max drift {permuted['drift']:.2e}); permuting only the "
        f"predictions leaves accuracy at {_fmt(permuted['scrambled_accuracy'])} "
        f"(was {_fmt(base)})",
    )

    out.append(format_verdict_table(report))
    out.append("")
    for scores in report.conditions:
        out.append(format_scores_table(scores))
        out.append("")
    out.append("invariants")
    for name, ok, detail in checks.results:
        out.append(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    out.append("")
    verdict = "selftest PASSED" if not checks.failed else f"selftest FAILED ({len(checks.failed)})"
    out.append(verdict)

    if verbose:
        print("\n".join(out))
    return 0 if not checks.failed else 1


def _permutation_pair(*, n_examples: int) -> tuple[float | None, dict]:
    """Permute the feature IDs and check the metrics move only when they should.

    Two halves. Permuting the dataset *and* the predictions together is an
    isomorphism, so every metric must be unchanged — feature identity is a
    label, not a quantity. Permuting only the predictions scrambles which
    feature is which, so accuracy must collapse; a metric that survived that
    would not be measuring feature identity at all.
    """
    base_train = generate_dataset(
        condition_config("capacity_stressed", n_examples=n_examples, split="train")
    )
    base_eval = generate_dataset(
        condition_config("capacity_stressed", n_examples=n_examples, split="test")
    )
    perm_train = generate_dataset(
        condition_config("permutation_control", n_examples=n_examples, split="train")
    )
    perm_eval = generate_dataset(
        condition_config("permutation_control", n_examples=n_examples, split="test")
    )

    def measure(train, evaluation):
        reference = EvaluationReference.from_dataset(evaluation)
        marginal = fit_marginal(train)
        predictions = ProgramOracle(fallback=marginal).predict(evaluation)
        return reference, predictions, evaluate_all(predictions, reference)

    base_reference, base_predictions, base_metrics = measure(base_train, base_eval)
    _, _, perm_metrics = measure(perm_train, perm_eval)

    drift = 0.0
    for name, value in base_metrics.items():
        other = perm_metrics[name].value
        if value.value is None or other is None:
            continue
        if math.isnan(value.value) or math.isnan(other):
            continue
        drift = max(drift, abs(value.value - other))

    scrambled = base_predictions.values.copy()
    scrambled_prob = base_predictions.active_prob.copy()
    content = base_reference.content_indices
    # A full permutation, not a shift by one: a position draws several features
    # from one contiguous group, so adjacent features co-activate and rolling
    # the axis leaves measurable signal behind.
    shuffled = np.random.default_rng(_TIE_BREAK_SEED).permutation(content)
    scrambled[:, :, shuffled] = base_predictions.values[:, :, content]
    scrambled_prob[:, :, shuffled] = base_predictions.active_prob[:, :, content]
    scrambled_accuracy = associative_recall_accuracy(
        Predictions(values=scrambled, active_prob=scrambled_prob), base_reference
    ).value

    return base_metrics["associative_recall_accuracy"].value, {
        "invariant": drift < 1e-9,
        "drift": drift,
        "blind": scrambled_accuracy is not None and scrambled_accuracy > 0.05,
        "scrambled_accuracy": scrambled_accuracy,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="architecture_mechanics.metrics.capability")
    parser.add_argument("--selftest", action="store_true", help="run the calibration gate")
    parser.add_argument(
        "--break-invariant",
        choices=INVARIANTS,
        default=None,
        help="force one invariant to fail; used to prove the gate reports failure",
    )
    parser.add_argument(
        "--t0", action="store_true", help="run T0 end to end and print the metric table"
    )
    parser.add_argument("--calibrate", action="store_true", help="print the calibration report")
    parser.add_argument("--json", metavar="PATH", default=None, help="write the report as JSON")
    parser.add_argument("--n-examples", type=int, default=_SELFTEST_EXAMPLES)
    args = parser.parse_args(argv)

    if args.t0:
        scores, pieces = score_condition("T0", n_examples=args.n_examples)
        print(format_scores_table(scores))
        print()
        reference = pieces["reference"]
        print(
            f"T0 measurement path: generated {reference.n_examples} examples "
            f"({reference.n_supervised} supervised positions), applied "
            f"{len(pieces['measured'])} reference predictors, computed "
            f"{len(METRIC_SPECS)} metrics. No model was involved."
        )
        if args.json:
            _write_json(args.json, scores.as_dict())
        return 0

    if args.calibrate:
        report = calibrate(n_examples=args.n_examples)
        print(format_verdict_table(report))
        print()
        for scores in report.conditions:
            print(format_scores_table(scores))
            print()
        print(json.dumps(report.positive_control, indent=2, sort_keys=True))
        if args.json:
            _write_json(args.json, report.as_dict())
        return 0 if report.ok else 1

    if args.selftest or args.break_invariant:
        return run_selftest(break_invariant=args.break_invariant, n_examples=args.n_examples)

    parser.print_help()
    return 0


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")


def _json_default(value):
    if isinstance(value, np.floating | np.integer):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, frozenset | set):
        return sorted(value)
    raise TypeError(f"not JSON serialisable: {type(value)!r}")


__all__ = [
    "METRIC_SPECS",
    "METRIC_VERSION",
    "POSITIVE_CONTROL_THRESHOLD",
    "RETAINED_METRICS",
    "RETIRED_METRICS",
    "CalibrationReport",
    "Curve",
    "EvaluationReference",
    "MarginalBaseline",
    "MetricValue",
    "OverwriteStep",
    "PositiveControlResult",
    "Predictions",
    "ProgramOracle",
    "RandomBaseline",
    "answer_set_accuracy",
    "associative_recall_accuracy",
    "associative_recall_jaccard",
    "brier_score",
    "calibrate",
    "calibration",
    "distance_degradation",
    "distractor_sensitivity",
    "evaluate_all",
    "expected_calibration_error",
    "feature_detection",
    "feature_f1",
    "feature_macro_precision",
    "feature_macro_recall",
    "feature_precision",
    "feature_recall",
    "fit_frequency_ceiling",
    "fit_marginal",
    "heldout_composition_accuracy",
    "normalized_skill",
    "overwrite_accuracy",
    "positive_control",
    "positive_control_datasets",
    "recall_at_free_threshold",
    "reconstruction_loss",
    "run_selftest",
    "stale_value_error_rate",
    "step_scores",
    "sweep_curve",
    "synthetic_overwrite_reference",
    "threshold_sweep",
]


if __name__ == "__main__":
    sys.exit(main())
