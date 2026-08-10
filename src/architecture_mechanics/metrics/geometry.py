"""§6.2 representation geometry, measured against the known feature basis.

This is the capability the synthetic benchmark exists to unlock. On real text you
cannot measure feature purity because you do not know the features; here the
generator wrote them down, so every measure below is a pure function of
``(hidden states, ground-truth features)`` with no dictionary to learn first and
no sparse autoencoder standing between the question and the answer.

That freedom is also the hazard. A geometry measure is a number computed from a
matrix, and matrices of noise have geometry too. **A measure that reports
structure in noise will report structure in every architecture in the quiver**,
and it will do so with the same confident decimals as a real result. So nothing
here is trusted because it looks reasonable: every measure is run against five
*constructed* representations whose answer is known in advance —

===========================  ====================================================
``orthogonal_basis``         ``d >= F``, one unit direction per feature: the
                             ceiling. Purity 1, interference 0, effective rank
                             exactly ``F``.
``known_superposition``      ``d < F``, features in antipodal pairs sharing a
                             direction: capacity is ``d`` *by construction*, so
                             the measured capacity has an exact target.
``random_rotation``          the orthogonal case times a random orthogonal
                             matrix: everything rotation-invariant must not
                             move at all, and the one thing that should — the
                             cosine to a *fixed external* basis — must.
``degenerate_collapse``      every feature on one direction: floor behaviour,
                             and no NaN or silent division anywhere.
``pure_noise``               hidden states independent of the features: every
                             measure at its null.
===========================  ====================================================

— and the results are recorded in :data:`GEOMETRY_MEASURES` as data. The noise
column is the one that decides whether a measure is reportable at all, and it
retires two of them:

- ``capacity_total`` reads **19.4 of 48 dimensions in use** on pure noise,
  because ``F`` fitted directions in ``d`` dimensions look packed whether or not
  they encode anything. It is a diagnostic, never a headline.
- ``alignment_marginal`` and ``alignment_reference`` both read **1.0 on total
  collapse**: a feature can be perfectly aligned with its own direction while
  every feature shares that direction. Alignment is only meaningful beside the
  cosine matrix.

``--selftest`` re-measures all five cases and re-derives those decisions, so a
measure whose behaviour changes fails the gate rather than quietly changing what
the laboratory believes.

Two disciplines are structural rather than promised.

**Probes are split by example, never by row.** :class:`ProbeSplit` partitions
*examples*; positions from one sequence cannot straddle the split, because they
share a program and a probe that has seen a sequence's other positions is
measuring the probe. Both halves come from the same template families as the
model's own evaluation split — the probe answers "is this feature decodable
here", not "does this generalise", and mixing the two questions would make
neither answerable.

**A mechanism-specific geometry statement carries its matched site with it.**
§6.4's seventh intervention family and §7.4's prohibition on reporting probe
results without a matched-site baseline both require that a number about, say,
an attention readout be comparable against a dimension- and depth-matched
ordinary hidden state. :class:`MatchedSiteComparison` is the shape that makes
that hard to skip: it has no accessor that returns the candidate alone.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from architecture_mechanics.metrics.capability import Curve

__all__ = [
    "CONSTRUCTED_CASES",
    "GEOMETRY_MEASURES",
    "GEOMETRY_VERSION",
    "CapacityReport",
    "ConstructedCase",
    "ConstructedRepresentation",
    "DepthTrajectory",
    "FeatureDirections",
    "GeometryError",
    "GeometryReport",
    "InterferenceReport",
    "MatchedSiteComparison",
    "MatchedSites",
    "ProbeReport",
    "ProbeSplit",
    "RunGeometry",
    "SparsitySample",
    "across_runs",
    "alignment",
    "capacity_versus_sparsity",
    "constructed_representation",
    "depth_trajectory",
    "effective_rank",
    "estimate_feature_directions",
    "feature_capacity",
    "feature_cosine_similarity",
    "feature_reconstruction",
    "flatten_site",
    "interference_matrix",
    "matched_site_baseline",
    "matched_site_comparison",
    "measure_geometry",
    "participation_ratio",
    "per_feature_purity",
    "probe_split",
    "representation_similarity",
    "run_geometry",
    "run_selftest",
    "site_depth",
    "validate_constructed_cases",
]

GEOMETRY_VERSION = "geo-1.0.0"
"""Bump on any change to the *semantics* of a measure. Recorded beside every
number these functions produce, so a redefinition invalidates a comparison
instead of silently replacing it."""

RIDGE = 1e-6
"""Ridge coefficient, expressed as a fraction of the design matrix's mean
diagonal so it is scale-free. Small enough to be invisible on a well-conditioned
fit and large enough that the degenerate case — a rank-one hidden state with 47
exactly-empty directions — solves instead of raising."""

VARIANCE_FLOOR = 1e-12
"""A feature whose value never varies over the scored rows has no direction. It
is recorded as undefined rather than given one, and excluded from every average
by name. This is the alternative to a silent zero."""

RESPONSE_FLOOR = 1e-12
"""A readout response smaller than this is not a small response; it is the
arithmetic noise left over from a fit against a representation that carries
nothing. Guarding on ``> 0`` is not enough — a constant hidden state produces a
response matrix of denormals, and their *ratio* is a perfectly well-formed number
that means nothing at all."""

COLLINEARITY_FLOOR = 0.01
"""Per-feature uniqueness (``1 - R^2`` of a feature regressed on all the others)
below which the split of a shared direction between two features is arbitrary.
Reported, not corrected: the fix is a different feature bank, not a different
estimator."""


class GeometryError(ValueError):
    """Raised when a geometry measure is asked for something the data cannot support."""


# --------------------------------------------------------------------------- #
# Small numerical helpers
# --------------------------------------------------------------------------- #


def _as_float_matrix(value, name: str, *, ndim: int = 2) -> np.ndarray:
    array = np.asarray(_to_numpy(value), dtype=np.float64)
    if array.ndim != ndim:
        raise GeometryError(f"{name} must be {ndim}-dimensional; got shape {array.shape}")
    if not np.isfinite(array).all():
        raise GeometryError(f"{name} contains non-finite values")
    return array


def _to_numpy(value):
    detach = getattr(value, "detach", None)
    if detach is not None:  # a torch.Tensor, without importing torch here
        return detach().cpu().numpy()
    return value


def _ridge_fit(
    x: np.ndarray, y: np.ndarray, *, ridge: float = RIDGE
) -> tuple[np.ndarray, np.ndarray, float]:
    """Centred ridge least squares. Returns ``(coefficients, intercept, cond)``.

    Centring rather than an appended ones column, so the intercept is never
    penalised — a penalised intercept would shrink every prediction toward zero
    and make a sparse feature look less decodable than it is.
    """
    if x.shape[0] != y.shape[0]:
        raise GeometryError(f"{x.shape[0]} design rows for {y.shape[0]} response rows")
    if x.shape[0] < 2:
        raise GeometryError("a ridge fit needs at least two rows")
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    centred = x - x_mean
    gram = centred.T @ centred
    n_columns = gram.shape[0]
    scale = float(np.trace(gram)) / n_columns if n_columns else 0.0
    penalty = ridge * scale if scale > 0.0 else ridge
    regularised = gram + penalty * np.eye(n_columns)
    coefficients = np.linalg.solve(regularised, centred.T @ (y - y_mean))
    return coefficients, y_mean - x_mean @ coefficients, float(np.linalg.cond(regularised))


def _unit_rows(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(matrix, axis=1)
    safe = np.where(norms > 0.0, norms, 1.0)
    return matrix / safe[:, None], norms


def _nanmean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def _off_diagonal(matrix: np.ndarray) -> np.ndarray:
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return matrix[mask]


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUC with average ranks for ties.

    ``NaN`` when one class is absent — a feature that is active everywhere, or
    nowhere, in the evaluation half has no ranking to score, and 0.5 would read
    as "the probe is at chance" rather than "the question was not asked".
    """
    n_positive = int(labels.sum())
    n_negative = int(labels.size - n_positive)
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks_sorted = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < sorted_scores.size:
        stop = start + 1
        while stop < sorted_scores.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks_sorted[start:stop] = 0.5 * (start + stop + 1)  # 1-based average rank
        start = stop
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = ranks_sorted
    positive_rank_sum = float(ranks[labels].sum())
    return (positive_rank_sum - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)


def _weighted_slope(x: np.ndarray, y: np.ndarray, n: np.ndarray) -> float | None:
    """Count-weighted least-squares slope; ``None`` when undefined.

    A deliberate ten-line copy of the capability module's helper rather than an
    import of its private name. Two independent gates should not share a private
    function: the coupling costs more than the duplication.
    """
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
# Probe splits — by example, never by row
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProbeSplit:
    """Disjoint row sets for fitting and scoring a probe, partitioned by example.

    §7.4 forbids reporting a probe result without a matched-site baseline, and
    the reason a probe result needs guarding at all is that a probe is trivially
    able to memorise. Splitting *rows* would not stop it: twelve positions of one
    sequence share a program, a key, and an answer, so a probe fitted on eleven
    of them and scored on the twelfth has seen the answer. Splitting *examples*
    is the only partition under which "the probe decoded this" is a statement
    about the representation.

    Both halves are drawn from the same template families, deliberately. The
    model's own split already holds out compositions; asking the probe to
    generalise across templates as well would confound "the feature is not
    linearly present" with "the feature is present but the probe has not seen
    this composition", and neither would then be measurable.
    """

    train: np.ndarray
    """Row indices used to fit."""
    eval: np.ndarray
    """Row indices used to score. Disjoint from :attr:`train` by example."""
    train_examples: np.ndarray
    eval_examples: np.ndarray
    n_examples: int
    seed: int
    train_fraction: float
    n_shared_templates: int | None = None
    """How many template ids appear on both sides. ``None`` when templates were
    not supplied. A probe split is *supposed* to share templates — that is what
    "from the same template families" means — so this is recorded as evidence
    for the design, not as a violation to be minimised."""

    def __post_init__(self) -> None:
        if np.intersect1d(self.train_examples, self.eval_examples).size:
            raise GeometryError("probe split leaks: an example appears on both sides")
        if np.intersect1d(self.train, self.eval).size:
            raise GeometryError("probe split leaks: a row appears on both sides")
        if self.train.size == 0 or self.eval.size == 0:
            raise GeometryError("a probe split needs rows on both sides")

    def as_dict(self) -> dict:
        return {
            "n_train_rows": int(self.train.size),
            "n_eval_rows": int(self.eval.size),
            "n_train_examples": int(self.train_examples.size),
            "n_eval_examples": int(self.eval_examples.size),
            "n_examples": int(self.n_examples),
            "seed": int(self.seed),
            "train_fraction": float(self.train_fraction),
            "n_shared_templates": self.n_shared_templates,
            "split_by": "example",
        }


def probe_split(
    example_of_row: Sequence[int] | np.ndarray,
    *,
    seed: int = 20260810,
    train_fraction: float = 0.5,
    template_of_example: Sequence[str] | np.ndarray | None = None,
) -> ProbeSplit:
    """Partition rows into probe-train and probe-eval by their owning example."""
    rows = np.asarray(example_of_row, dtype=np.int64)
    if rows.ndim != 1:
        raise GeometryError(f"example_of_row must be one-dimensional; got {rows.shape}")
    examples = np.unique(rows)
    if examples.size < 2:
        raise GeometryError(
            f"a probe split needs at least two examples; got {examples.size}. "
            "Splitting rows within one example would let the probe see the answer."
        )
    if not 0.0 < train_fraction < 1.0:
        raise GeometryError(f"train_fraction must be in (0, 1); got {train_fraction}")

    order = np.random.default_rng(seed).permutation(examples.size)
    # Clamped to leave at least one example on each side: a probe split with an
    # empty half is not a strict split, it is a missing measurement.
    n_train = max(1, min(examples.size - 1, round(train_fraction * examples.size)))
    train_examples = np.sort(examples[order[:n_train]])
    eval_examples = np.sort(examples[order[n_train:]])

    in_train = np.isin(rows, train_examples)
    shared: int | None = None
    if template_of_example is not None:
        templates = np.asarray(template_of_example)
        shared = int(
            np.intersect1d(templates[train_examples], templates[eval_examples]).size
        )
    return ProbeSplit(
        train=np.nonzero(in_train)[0],
        eval=np.nonzero(~in_train)[0],
        train_examples=train_examples,
        eval_examples=eval_examples,
        n_examples=int(examples.size),
        seed=int(seed),
        train_fraction=float(train_fraction),
        n_shared_templates=shared,
    )


# --------------------------------------------------------------------------- #
# §6.2 measure 4 — effective rank; measure 5 — participation ratio
# --------------------------------------------------------------------------- #


def effective_rank(hidden: np.ndarray, *, center: bool = True) -> float:
    """Entropy effective rank of the representation's spectrum (Roy & Vetterli).

    ``exp(H(p))`` for ``p_i = sigma_i / sum(sigma)``. Equal to the true rank when
    the spectrum is flat and to 1 when one direction carries everything, so it
    reads as "how many directions is this representation really using".

    Centred by default: an uncentred spectrum charges a dimension for the mean
    offset, which every position shares and which therefore encodes nothing about
    which features are present.

    Its null is ``d``, not zero — isotropic noise fills every direction. That is
    why a high effective rank is never on its own evidence of structure.
    """
    matrix = _as_float_matrix(hidden, "hidden")
    if center:
        matrix = matrix - matrix.mean(axis=0)
    singular = np.linalg.svd(matrix, compute_uv=False)
    total = float(singular.sum())
    if total <= 0.0:
        return 0.0
    p = singular / total
    p = p[p > 0.0]
    return float(np.exp(-(p * np.log(p)).sum()))


def participation_ratio(hidden: np.ndarray, *, center: bool = True) -> float:
    """``(sum lambda)^2 / sum lambda^2`` over the covariance eigenvalues.

    The second of §6.2's two dimensionality measures, and deliberately not a
    restatement of the first: the participation ratio is quadratic in the
    eigenvalues and so is dominated by the leading directions, while the entropy
    effective rank weights a long tail of small directions much more heavily.
    Reporting both is what makes "the representation has a tail" visible.
    """
    matrix = _as_float_matrix(hidden, "hidden")
    if center:
        matrix = matrix - matrix.mean(axis=0)
    eigenvalues = np.linalg.svd(matrix, compute_uv=False) ** 2
    total = float(eigenvalues.sum())
    if total <= 0.0:
        return 0.0
    return float(total**2 / float((eigenvalues**2).sum()))


# --------------------------------------------------------------------------- #
# Feature directions — the estimator every geometric measure is built on
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FeatureDirections:
    """One estimated direction per ground-truth feature, plus what it cost.

    The estimator is the partial (multiple-regression) coefficient of the hidden
    state on the feature's value, not the marginal mean difference. With
    co-activating features those differ, and the partial estimate is the one that
    answers "which direction does *this* feature write into", which is the
    question §6.2 asks.
    """

    directions: np.ndarray
    """``(F, d)`` raw coefficients."""
    unit: np.ndarray
    """``(F, d)`` unit-normalised; zero rows for undefined features."""
    norms: np.ndarray
    variance: np.ndarray
    """Per-feature variance over the fitted rows. The denominator of purity."""
    n_active: np.ndarray
    undefined: np.ndarray
    """``(F,)`` bool: features that never varied and so have no direction."""
    uniqueness: np.ndarray
    """``(F,)`` ``1 - R^2`` of each feature regressed on all the others. Near zero
    means two features are the same regressor and the split of a shared direction
    between them is arbitrary — a fact about the feature bank, reported rather
    than corrected."""
    condition_number: float
    n_rows: int
    ridge: float

    @property
    def n_features(self) -> int:
        return int(self.directions.shape[0])

    @property
    def d_model(self) -> int:
        return int(self.directions.shape[1])

    @property
    def collinear(self) -> np.ndarray:
        return self.uniqueness < COLLINEARITY_FLOOR


def estimate_feature_directions(
    hidden: np.ndarray,
    features: np.ndarray,
    *,
    rows: np.ndarray | None = None,
    ridge: float = RIDGE,
) -> FeatureDirections:
    """Fit ``h = b + sum_f a_f w_f`` and return the ``w_f``.

    This is the one place the known feature basis is used to *estimate* rather
    than to score, so it is also the one place worth checking against a known
    answer: on a model whose encoder is linear, the directions recovered at the
    embedding site must equal the encoder's own weights, and
    ``tests/metrics/test_geometry_estimators.py`` asserts exactly that.
    """
    h = _as_float_matrix(hidden, "hidden")
    a = _as_float_matrix(features, "features")
    if h.shape[0] != a.shape[0]:
        raise GeometryError(f"{h.shape[0]} hidden rows for {a.shape[0]} feature rows")
    if rows is not None:
        h, a = h[rows], a[rows]

    directions, _intercept, condition = _ridge_fit(a, h, ridge=ridge)  # (F, d)
    variance = a.var(axis=0)
    undefined = variance <= VARIANCE_FLOOR
    directions = np.where(undefined[:, None], 0.0, directions)
    unit, norms = _unit_rows(directions)
    unit = np.where(undefined[:, None], 0.0, unit)
    return FeatureDirections(
        directions=directions,
        unit=unit,
        norms=norms,
        variance=variance,
        n_active=(a != 0.0).sum(axis=0).astype(np.int64),
        undefined=undefined,
        uniqueness=_feature_uniqueness(a, ridge=ridge),
        condition_number=condition,
        n_rows=int(h.shape[0]),
        ridge=float(ridge),
    )


def _feature_uniqueness(features: np.ndarray, *, ridge: float) -> np.ndarray:
    """``1 - R^2`` of each feature on all the others, from one matrix inverse.

    For a correlation matrix ``R``, ``[R^-1]_ff`` is the variance inflation
    factor of feature ``f``, and its reciprocal is exactly ``1 - R^2_f``. One
    inverse answers the question for every feature at once.
    """
    n_features = features.shape[1]
    variance = features.var(axis=0)
    usable = variance > VARIANCE_FLOOR
    uniqueness = np.full(n_features, np.nan)
    if usable.sum() < 2:
        uniqueness[usable] = 1.0
        return uniqueness
    centred = features[:, usable] - features[:, usable].mean(axis=0)
    scale = np.sqrt((centred**2).mean(axis=0))
    correlation = (centred / scale).T @ (centred / scale) / centred.shape[0]
    regularised = correlation + ridge * np.eye(correlation.shape[0])
    inflation = np.diag(np.linalg.inv(regularised))
    uniqueness[usable] = np.clip(1.0 / np.maximum(inflation, 1e-30), 0.0, 1.0)
    return uniqueness


# --------------------------------------------------------------------------- #
# §6.2 measure 2 — cosine similarity between feature directions
# --------------------------------------------------------------------------- #


def feature_cosine_similarity(directions: FeatureDirections) -> np.ndarray:
    """``(F, F)`` cosine matrix between unit feature directions.

    Rows and columns of undefined features are ``NaN``, not zero: "these two
    features are orthogonal" and "one of these features never occurred" are
    different statements and must not share a cell value.
    """
    matrix = directions.unit @ directions.unit.T
    np.clip(matrix, -1.0, 1.0, out=matrix)
    if directions.undefined.any():
        matrix = matrix.copy()
        matrix[directions.undefined, :] = np.nan
        matrix[:, directions.undefined] = np.nan
    return matrix


# --------------------------------------------------------------------------- #
# §6.2 measure 1 — feature-to-direction alignment
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AlignmentReport:
    """How well each feature is carried by a single direction. Three readings.

    ``explained`` is the reference-free one, and the only one with a null at
    zero. For feature ``f`` it is the ``R^2`` of the rank-one model ``a_f w_f``
    against the hidden variation that remains once every *other* feature's fitted
    contribution is removed — literally "is this feature carried by one
    direction". It is exactly 1 whenever the encoding is linear, which is true by
    construction of all four structured cases, and it falls toward zero when the
    hidden state does not move with the feature at all.

    ``marginal`` compares the *partial* direction (the multiple-regression
    coefficient) against the *marginal* one (mean hidden state with the feature
    on, minus with it off). It separates exactly when features co-activate, which
    is the trained-model case. **Its null is not zero**: on pure noise it reads
    0.83, because both estimators are driven by the same sampling noise on the
    same rows and therefore agree with each other about it. Recorded because that
    is a fact worth knowing about the estimator, not because it is reportable.

    ``reference`` compares the estimated direction against a supplied
    ground-truth one. Its real use is as an *estimator check* where the basis is
    known: the constructed cases, and a real model at its embedding site, where
    the encoder weights are the reference the estimator must recover.

    **All three read 1.0 on total collapse.** Every feature can be perfectly
    aligned with its own direction while every feature shares that direction.
    Alignment is only ever reported beside :func:`feature_cosine_similarity`, and
    that is why all three are classified diagnostic in
    :data:`GEOMETRY_MEASURES`.
    """

    explained: np.ndarray
    marginal: np.ndarray
    reference: np.ndarray | None
    explained_mean: float
    marginal_mean: float
    reference_mean: float | None

    def as_dict(self) -> dict:
        return {
            "explained_mean": self.explained_mean,
            "marginal_mean": self.marginal_mean,
            "reference_mean": self.reference_mean,
        }


def alignment(
    hidden: np.ndarray,
    features: np.ndarray,
    *,
    active: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    directions: FeatureDirections | None = None,
    rows: np.ndarray | None = None,
    ridge: float = RIDGE,
) -> AlignmentReport:
    """§6.2's feature-to-direction alignment, in all three of its senses."""
    h = _as_float_matrix(hidden, "hidden")
    a = _as_float_matrix(features, "features")
    mask = (a != 0.0) if active is None else np.asarray(_to_numpy(active), dtype=bool)
    if rows is not None:
        h, a, mask = h[rows], a[rows], mask[rows]
    if directions is None:
        directions = estimate_feature_directions(h, a, ridge=ridge)

    n_rows, n_features = a.shape
    centred_features = a - a.mean(axis=0)
    residual = (h - h.mean(axis=0)) - centred_features @ directions.directions

    # ss_tot for feature f is the squared norm of `residual + a_f w_f`, which
    # expands into three terms. Written out rather than assumed orthogonal: the
    # ridge makes the cross term small but not exactly zero, and a term dropped
    # because it is "usually" zero is how an estimator acquires a silent bias.
    ss_residual = float((residual**2).sum())
    cross = 2.0 * ((centred_features.T @ residual) * directions.directions).sum(axis=1)
    own = a.var(axis=0) * n_rows * directions.norms**2
    ss_total = ss_residual + cross + own
    with np.errstate(invalid="ignore", divide="ignore"):
        explained = np.where(ss_total > 0.0, 1.0 - ss_residual / ss_total, np.nan)

    # The marginal direction for every feature at once: one (F, N) x (N, d)
    # matmul instead of F boolean-indexed reductions over the whole hidden state.
    # At R2 scale that is the difference between forty seconds and one.
    on_counts = mask.sum(axis=0).astype(np.float64)
    on_sums = mask.T.astype(np.float64) @ h
    total = h.sum(axis=0)
    off_counts = n_rows - on_counts
    usable = (on_counts > 0) & (off_counts > 0)
    marginal = np.full(n_features, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        difference = (
            on_sums / np.where(usable, on_counts, 1.0)[:, None]
            - (total - on_sums) / np.where(usable, off_counts, 1.0)[:, None]
        )
    for f in np.nonzero(usable)[0]:
        marginal[f] = _cosine(difference[f], directions.directions[f])

    reference_cosines: np.ndarray | None = None
    if reference is not None:
        basis = _as_float_matrix(reference, "reference")
        if basis.shape != directions.directions.shape:
            raise GeometryError(
                f"reference basis {basis.shape} does not match the estimated directions "
                f"{directions.directions.shape}"
            )
        reference_cosines = np.asarray(
            [_cosine(basis[f], directions.directions[f]) for f in range(n_features)]
        )
        reference_cosines[directions.undefined] = np.nan

    marginal[directions.undefined] = np.nan
    explained[directions.undefined] = np.nan
    return AlignmentReport(
        explained=explained,
        marginal=marginal,
        reference=reference_cosines,
        explained_mean=_nanmean(explained),
        marginal_mean=_nanmean(marginal),
        reference_mean=None if reference_cosines is None else _nanmean(reference_cosines),
    )


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 0.0:
        return float("nan")
    return float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))


# --------------------------------------------------------------------------- #
# §6.2 measure 3 — interference matrix; measure 7 — per-feature purity
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class InterferenceReport:
    """What a reader of feature ``i`` actually sees when feature ``j`` fires.

    Measured functionally and out of sample, not read off the encoding geometry.
    A linear readout is fitted on the probe-train half; its predictions on the
    probe-eval half are then regressed back onto the *true* features, giving
    ``matrix[j, i]`` = the response of the predicted feature ``i`` to the true
    feature ``j``. A perfect representation gives the identity; a superposed one
    gives off-diagonal mass; noise gives a matrix of small random numbers, which
    is why the summary below is a *fraction* rather than a ratio — a ratio's
    denominator is zero exactly when the representation carries nothing.
    """

    matrix: np.ndarray
    mean_abs_diagonal: float
    mean_abs_off_diagonal: float
    interference_fraction: float
    """``mean|off| / (mean|off| + mean|diag|)`` in ``[0, 1]``. Zero when the
    readout is clean, 0.5 when off-diagonal mass equals signal — which is where
    both pure noise and total collapse land."""
    n_train_rows: int
    n_eval_rows: int

    def as_dict(self) -> dict:
        return {
            "mean_abs_diagonal": self.mean_abs_diagonal,
            "mean_abs_off_diagonal": self.mean_abs_off_diagonal,
            "interference_fraction": self.interference_fraction,
            "n_train_rows": self.n_train_rows,
            "n_eval_rows": self.n_eval_rows,
        }


def interference_matrix(
    hidden: np.ndarray,
    features: np.ndarray,
    split: ProbeSplit,
    *,
    ridge: float = RIDGE,
) -> InterferenceReport:
    """§6.2's interference matrix, fitted on one half and measured on the other."""
    h = _as_float_matrix(hidden, "hidden")
    a = _as_float_matrix(features, "features")
    if h.shape[0] != a.shape[0]:
        raise GeometryError(f"{h.shape[0]} hidden rows for {a.shape[0]} feature rows")

    readout, intercept, _ = _ridge_fit(h[split.train], a[split.train], ridge=ridge)
    predicted = h[split.eval] @ readout + intercept
    response, _, _ = _ridge_fit(a[split.eval], predicted, ridge=ridge)

    diagonal = np.abs(np.diag(response))
    off = np.abs(_off_diagonal(response))
    mean_diagonal = float(diagonal.mean())
    mean_off = float(off.mean()) if off.size else 0.0
    total = mean_diagonal + mean_off
    return InterferenceReport(
        matrix=response,
        mean_abs_diagonal=mean_diagonal,
        mean_abs_off_diagonal=mean_off,
        interference_fraction=(mean_off / total) if total > RESPONSE_FLOOR else float("nan"),
        n_train_rows=int(split.train.size),
        n_eval_rows=int(split.eval.size),
    )


def per_feature_purity(interference: InterferenceReport, variance: np.ndarray) -> np.ndarray:
    """Fraction of a feature readout's response that is the feature itself.

    ``purity_i = C[i,i]^2 v_i / sum_j C[j,i]^2 v_j``: the share of the readout's
    variance attributable to the feature it is named after, with each true
    feature weighted by how much it actually varies. Bounded in ``[0, 1]`` with a
    ceiling of 1 for an isolated feature and a floor of ``1/F`` when every
    feature is equally responsible — which is where both total collapse and pure
    noise land, and is the honest chance level rather than zero.

    A readout that responds to nothing at all yields ``NaN`` rather than a
    ratio. The test is on the response column's own magnitude and not on the
    denominator being positive: a column of denormals has a positive denominator
    and a perfectly well-formed ratio, and the ratio means nothing.
    """
    weights = np.asarray(variance, dtype=np.float64)
    squared = interference.matrix**2 * weights[:, None]
    denominator = squared.sum(axis=0)
    column_scale = np.sqrt((interference.matrix**2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        purity = np.where(
            (column_scale > RESPONSE_FLOOR) & (denominator > 0.0),
            np.diag(squared) / denominator,
            np.nan,
        )
    return purity


# --------------------------------------------------------------------------- #
# §6.2 measure 6 — feature reconstruction from hidden states
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProbeReport:
    """A linear probe's out-of-sample recovery of the ground-truth features.

    Fitted on :attr:`ProbeSplit.train` and scored on :attr:`ProbeSplit.eval`,
    with the *training* mean as the null predictor, so ``R^2`` can and does go
    negative on a representation that carries nothing. That is intended: a floor
    pinned at zero would make "no signal" and "slightly worse than the mean"
    indistinguishable, and the second is what overfitting looks like.
    """

    r2: np.ndarray
    auc: np.ndarray
    macro_r2: float
    pooled_r2: float
    macro_auc: float
    n_train_rows: int
    n_eval_rows: int
    n_scored_features: int

    def as_dict(self) -> dict:
        return {
            "macro_r2": self.macro_r2,
            "pooled_r2": self.pooled_r2,
            "macro_auc": self.macro_auc,
            "n_train_rows": self.n_train_rows,
            "n_eval_rows": self.n_eval_rows,
            "n_scored_features": self.n_scored_features,
        }


def feature_reconstruction(
    hidden: np.ndarray,
    features: np.ndarray,
    split: ProbeSplit,
    *,
    active: np.ndarray | None = None,
    ridge: float = RIDGE,
) -> ProbeReport:
    """Fit a linear readout on one half of the examples and score it on the other."""
    h = _as_float_matrix(hidden, "hidden")
    a = _as_float_matrix(features, "features")
    mask = (a != 0.0) if active is None else np.asarray(_to_numpy(active), dtype=bool)
    if h.shape[0] != a.shape[0]:
        raise GeometryError(f"{h.shape[0]} hidden rows for {a.shape[0]} feature rows")

    readout, intercept, _ = _ridge_fit(h[split.train], a[split.train], ridge=ridge)
    predicted = h[split.eval] @ readout + intercept
    truth = a[split.eval]
    null = a[split.train].mean(axis=0)

    residual = ((truth - predicted) ** 2).sum(axis=0)
    total = ((truth - null) ** 2).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = np.where(total > 0.0, 1.0 - residual / total, np.nan)

    labels = mask[split.eval]
    auc = np.asarray([_auc(predicted[:, f], labels[:, f]) for f in range(a.shape[1])])

    pooled_total = float(total.sum())
    return ProbeReport(
        r2=r2,
        auc=auc,
        macro_r2=_nanmean(r2),
        pooled_r2=float(1.0 - residual.sum() / pooled_total) if pooled_total > 0 else float("nan"),
        macro_auc=_nanmean(auc),
        n_train_rows=int(split.train.size),
        n_eval_rows=int(split.eval.size),
        n_scored_features=int(np.isfinite(r2).sum()),
    )


# --------------------------------------------------------------------------- #
# §6.2 measure 8 — superposition capacity versus sparsity
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CapacityReport:
    """Per-feature dimensionality, in the sense of Elhage et al.'s toy models.

    ``D_i = ||w_i||^2 / sum_j (w_hat_i . w_j)^2`` — the share of a dimension
    feature ``i`` has to itself. One when the feature owns an orthogonal
    direction, ``1/k`` when ``k`` features share one, and summing to at most
    ``d`` over the whole bank.

    **Diagnostic, never a headline.** On hidden states that are pure noise the
    estimated directions are random, and ``F`` random directions in ``d``
    dimensions are about as spread out as ``F`` learned ones: the constructed
    noise case reports 19.4 of 48 dimensions in use. Capacity measures the
    geometry of the estimated directions and cannot tell an encoded direction
    from a fitted artifact. Read it beside ``probe_macro_r2``.
    """

    per_feature: np.ndarray
    total: float
    fraction_of_dimensions: float
    mean: float

    def as_dict(self) -> dict:
        return {
            "capacity_total": self.total,
            "capacity_fraction_of_dimensions": self.fraction_of_dimensions,
            "mean_feature_capacity": self.mean,
        }


def feature_capacity(directions: FeatureDirections) -> CapacityReport:
    """Per-feature dimensionality and its total, from the estimated directions."""
    projections = directions.unit @ directions.directions.T  # (F, F): w_hat_i . w_j
    denominator = (projections**2).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_feature = np.where(denominator > 0.0, directions.norms**2 / denominator, np.nan)
    per_feature[directions.undefined] = np.nan
    finite = per_feature[np.isfinite(per_feature)]
    total = float(finite.sum())
    return CapacityReport(
        per_feature=per_feature,
        total=total,
        fraction_of_dimensions=total / directions.d_model if directions.d_model else float("nan"),
        mean=_nanmean(per_feature),
    )


@dataclass(frozen=True)
class SparsitySample:
    """One point on a capacity-versus-sparsity curve."""

    sparsity: float
    hidden: np.ndarray
    features: np.ndarray
    split: ProbeSplit | None = None
    """``None`` builds a row split with ``split_seed``. Legitimate only for
    constructed representations, where a row is not part of a sequence and there
    is nothing for a probe to leak across."""


def capacity_versus_sparsity(
    samples: Sequence[SparsitySample],
    *,
    split_seed: int = 20260810,
    ridge: float = RIDGE,
) -> dict[str, Curve]:
    """Capacity, purity, and probe ``R^2`` as functions of feature sparsity.

    Returns curves rather than a figure, for the same reason the capability
    module does: §8.5 requires a report to be regenerable from recorded
    artifacts, and a curve is recordable where a plot is not.
    """
    if not samples:
        raise GeometryError("capacity_versus_sparsity needs at least one sample")
    axis: list[float] = []
    values: dict[str, list[float]] = {"capacity_total": [], "mean_purity": [], "probe_macro_r2": []}
    counts: list[int] = []
    for sample in sorted(samples, key=lambda s: s.sparsity):
        split = sample.split
        if split is None:
            rows = np.arange(np.asarray(sample.hidden).shape[0])
            split = probe_split(rows, seed=split_seed)
        report = measure_geometry(sample.hidden, sample.features, split, ridge=ridge)
        axis.append(float(sample.sparsity))
        counts.append(int(np.asarray(sample.hidden).shape[0]))
        values["capacity_total"].append(report.capacity.total)
        values["mean_purity"].append(report.mean_purity)
        values["probe_macro_r2"].append(report.probe.macro_r2)

    x = np.asarray(axis)
    n = np.asarray(counts, dtype=np.int64)
    return {
        name: Curve(
            name=f"{name}_vs_sparsity",
            axis="sparsity",
            x=tuple(axis),
            y=tuple(series),
            n=tuple(int(c) for c in counts),
            slope=_weighted_slope(x, np.asarray(series), n),
        )
        for name, series in values.items()
    }


# --------------------------------------------------------------------------- #
# §6.2 measure 9 — representation similarity across seeds and architectures
# --------------------------------------------------------------------------- #


def representation_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Linear centred kernel alignment between two representations of one input.

    ``||A'^T B'||_F^2 / (||A'^T A'||_F ||B'^T B'||_F)`` on column-centred
    matrices. Invariant to orthogonal transforms and to isotropic scaling, which
    is exactly what a cross-seed or cross-architecture comparison needs: two runs
    that learned the same geometry in a different basis must score 1, and the
    ``random_rotation`` constructed case checks that they do.

    Requires the same rows in the same order in both — the comparison is over a
    shared input, not over two independent samples.
    """
    x = _as_float_matrix(a, "a")
    y = _as_float_matrix(b, "b")
    if x.shape[0] != y.shape[0]:
        raise GeometryError(
            f"representation similarity needs matched rows; got {x.shape[0]} and {y.shape[0]}"
        )
    x = x - x.mean(axis=0)
    y = y - y.mean(axis=0)
    cross = float(np.linalg.norm(x.T @ y, ord="fro") ** 2)
    self_x = float(np.linalg.norm(x.T @ x, ord="fro"))
    self_y = float(np.linalg.norm(y.T @ y, ord="fro"))
    if self_x <= 0.0 or self_y <= 0.0:
        return float("nan")
    return cross / (self_x * self_y)


# --------------------------------------------------------------------------- #
# The whole-report object
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GeometryReport:
    """Every §6.2 within-representation measure for one site, in one object."""

    site: str
    n_rows: int
    n_features: int
    d_model: int
    directions: FeatureDirections
    cosine: np.ndarray
    alignment: AlignmentReport
    interference: InterferenceReport
    purity: np.ndarray
    probe: ProbeReport
    capacity: CapacityReport
    effective_rank: float
    participation_ratio: float
    split: ProbeSplit

    @property
    def mean_purity(self) -> float:
        return _nanmean(self.purity)

    @property
    def mean_abs_off_diagonal_cosine(self) -> float:
        return _nanmean(np.abs(_off_diagonal(self.cosine)))

    @property
    def max_abs_off_diagonal_cosine(self) -> float:
        off = np.abs(_off_diagonal(self.cosine))
        finite = off[np.isfinite(off)]
        return float(finite.max()) if finite.size else float("nan")

    def scalars(self) -> dict:
        """The flat record that goes into ``summary.json``."""
        return {
            "site": self.site,
            "geometry_version": GEOMETRY_VERSION,
            "n_rows": self.n_rows,
            "n_features": self.n_features,
            "d_model": self.d_model,
            "n_undefined_features": int(self.directions.undefined.sum()),
            "n_collinear_features": int(self.directions.collinear.sum()),
            "design_condition_number": self.directions.condition_number,
            "effective_rank": self.effective_rank,
            "effective_rank_fraction": self.effective_rank / self.d_model,
            "participation_ratio": self.participation_ratio,
            "mean_abs_off_diagonal_cosine": self.mean_abs_off_diagonal_cosine,
            "max_abs_off_diagonal_cosine": self.max_abs_off_diagonal_cosine,
            "mean_purity": self.mean_purity,
            "min_purity": float(np.nanmin(self.purity)) if np.isfinite(self.purity).any() else float("nan"),
            "probe_macro_r2": self.probe.macro_r2,
            "probe_pooled_r2": self.probe.pooled_r2,
            "probe_macro_auc": self.probe.macro_auc,
            "alignment_explained_mean": self.alignment.explained_mean,
            "alignment_marginal_mean": self.alignment.marginal_mean,
            "alignment_reference_mean": self.alignment.reference_mean,
            **self.interference.as_dict(),
            **self.capacity.as_dict(),
        }

    def arrays(self) -> dict[str, np.ndarray]:
        """The per-feature and per-pair record that goes into the npz."""
        payload = {
            "directions": self.directions.directions,
            "direction_norms": self.directions.norms,
            "feature_variance": self.directions.variance,
            "feature_n_active": self.directions.n_active,
            "feature_uniqueness": self.directions.uniqueness,
            "cosine_matrix": self.cosine,
            "interference_matrix": self.interference.matrix,
            "purity": self.purity,
            "probe_r2": self.probe.r2,
            "probe_auc": self.probe.auc,
            "capacity": self.capacity.per_feature,
            "alignment_explained": self.alignment.explained,
            "alignment_marginal": self.alignment.marginal,
        }
        if self.alignment.reference is not None:
            payload["alignment_reference"] = self.alignment.reference
        return payload


def measure_geometry(
    hidden: np.ndarray,
    features: np.ndarray,
    split: ProbeSplit,
    *,
    active: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    site: str = "",
    ridge: float = RIDGE,
) -> GeometryReport:
    """Every §6.2 measure for one representation, against the known feature basis.

    The directions, alignment, and capacity are estimated on the probe-*train*
    rows only. They could be estimated on everything — they are descriptive, not
    predictive — but then the capacity of a representation would be measured on
    rows its interference matrix was scored on, and two numbers in the same table
    would have different exposure to the same data. One rule for the whole
    report is easier to defend than a per-measure exemption.
    """
    h = _as_float_matrix(hidden, "hidden")
    a = _as_float_matrix(features, "features")
    directions = estimate_feature_directions(h, a, rows=split.train, ridge=ridge)
    interference = interference_matrix(h, a, split, ridge=ridge)
    return GeometryReport(
        site=site,
        n_rows=int(h.shape[0]),
        n_features=int(a.shape[1]),
        d_model=int(h.shape[1]),
        directions=directions,
        cosine=feature_cosine_similarity(directions),
        alignment=alignment(
            h, a, active=active, reference=reference, directions=directions, rows=split.train
        ),
        interference=interference,
        purity=per_feature_purity(interference, directions.variance),
        probe=feature_reconstruction(h, a, split, active=active, ridge=ridge),
        capacity=feature_capacity(directions),
        effective_rank=effective_rank(h),
        participation_ratio=participation_ratio(h),
        split=split,
    )


# --------------------------------------------------------------------------- #
# §6.2 measure 10 — trajectory across depth
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DepthTrajectory:
    """The same measures at every site, in forward order."""

    sites: tuple[str, ...]
    reports: dict[str, GeometryReport]
    curves: dict[str, Curve]

    def as_dict(self) -> dict:
        return {
            "sites": list(self.sites),
            "curves": {name: curve.as_dict() for name, curve in self.curves.items()},
            "per_site": {name: report.scalars() for name, report in self.reports.items()},
        }


TRAJECTORY_MEASURES: tuple[str, ...] = (
    "probe_macro_r2",
    "probe_macro_auc",
    "mean_purity",
    "interference_fraction",
    "mean_abs_off_diagonal_cosine",
    "effective_rank",
    "participation_ratio",
    "capacity_total",
)


def depth_trajectory(
    sites: Mapping[str, np.ndarray],
    features: np.ndarray,
    split: ProbeSplit,
    *,
    active: np.ndarray | None = None,
    references: Mapping[str, np.ndarray] | None = None,
    ridge: float = RIDGE,
) -> DepthTrajectory:
    """Measure every named site and return both the reports and the curves.

    ``sites`` must be ordered as the forward pass declares them; the curve's
    x-axis is depth index, so a reordering here would silently redraw the
    trajectory. Python dictionaries preserve insertion order, which is the
    contract being relied on and the reason the caller builds the mapping in
    forward order rather than sorting it.
    """
    reports = {
        name: measure_geometry(
            hidden,
            features,
            split,
            active=active,
            reference=None if references is None else references.get(name),
            site=name,
            ridge=ridge,
        )
        for name, hidden in sites.items()
    }
    order = tuple(reports)
    x = np.arange(len(order), dtype=np.float64)
    n = np.asarray([reports[name].n_rows for name in order], dtype=np.int64)
    curves = {}
    for measure in TRAJECTORY_MEASURES:
        y = np.asarray([reports[name].scalars()[measure] for name in order], dtype=np.float64)
        curves[measure] = Curve(
            name=f"{measure}_across_depth",
            axis="site_index",
            x=tuple(float(v) for v in x),
            y=tuple(float(v) for v in y),
            n=tuple(int(v) for v in n),
            slope=_weighted_slope(x, y, n),
        )
    return DepthTrajectory(sites=order, reports=reports, curves=curves)


# --------------------------------------------------------------------------- #
# The matched-site baseline — §6.4's seventh family, §7.4's requirement
# --------------------------------------------------------------------------- #

ORDINARY_SITE_SUFFIXES: tuple[str, ...] = ("resid_mid", "resid_out")
"""Sites that exist in every architecture regardless of mechanism. The ordinary
residual stream is the only representation A0, A1, A2 and A4 all have, which is
what makes it the matched baseline every mechanism-specific claim is stated
against."""


@dataclass(frozen=True)
class MatchedSites:
    """A mechanism-specific representation and its dimension- and depth-matched twin.

    §6.4's seventh intervention family in geometric form. The question a
    mechanism claim has to answer is never "is this variable decodable" — a
    residual stream of the same width at the same depth usually is too — but
    "is it *more* decodable, or more selectively so, than an ordinary hidden
    state under the same protocol". That comparison needs the baseline to differ
    in exactly one respect, so this records what had to be adjusted to make the
    two comparable and refuses when they cannot be.
    """

    candidate_site: str
    baseline_site: str
    depth: int
    dim: int
    candidate: np.ndarray
    baseline: np.ndarray
    adjustments: tuple[str, ...]
    projection_seed: int | None

    def as_dict(self) -> dict:
        return {
            "candidate_site": self.candidate_site,
            "baseline_site": self.baseline_site,
            "depth": self.depth,
            "dim": self.dim,
            "adjustments": list(self.adjustments),
            "projection_seed": self.projection_seed,
        }


def flatten_site(tensor) -> np.ndarray:
    """Reduce a captured site to ``(rows, dim)`` without choosing what to keep.

    ``(B, T, d)`` becomes ``(B*T, d)``. ``(B, H, T, d_head)`` — the head-split
    shape A0's ``q``, ``k``, ``v`` and ``readout`` use — becomes
    ``(B*T, H*d_head)``, the heads merged in exactly the order ``out_proj``
    consumes them, so the flattened site is the vector the rest of the model
    actually reads and not a view invented here.
    """
    array = np.asarray(_to_numpy(tensor), dtype=np.float64)
    if array.ndim == 2:
        return array
    if array.ndim == 3:
        return array.reshape(array.shape[0] * array.shape[1], array.shape[2])
    if array.ndim == 4:
        batch, heads, seq_len, head_dim = array.shape
        return array.transpose(0, 2, 1, 3).reshape(batch * seq_len, heads * head_dim)
    raise GeometryError(f"cannot flatten a site of shape {array.shape}")


def site_depth(name: str) -> int | None:
    """Depth index of a site, or ``None`` for a site outside the block stack."""
    parts = name.split(".")
    if len(parts) >= 2 and parts[0] == "layers":
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


def matched_site_baseline(
    states: Mapping[str, np.ndarray],
    candidate_site: str,
    *,
    baseline_site: str | None = None,
    projection_seed: int = 20260810,
) -> MatchedSites:
    """Pair a mechanism site with a depth- and dimension-matched ordinary one.

    The baseline defaults to the ordinary residual stream at the same depth. Where
    the two widths differ the *larger* is reduced by a seeded Gaussian random
    projection rather than by principal components: PCA would hand the reduced
    side its own best subspace, which is a thumb on the scale in whichever
    direction the reduction happened to fall. A random projection is neutral by
    being uninformed, and the seed is recorded so the reduction is reproducible.
    """
    if candidate_site not in states:
        raise GeometryError(f"no captured site named {candidate_site!r}")
    depth = site_depth(candidate_site)
    if depth is None:
        raise GeometryError(
            f"{candidate_site!r} is not inside the block stack, so it has no depth to match. "
            "A matched-site baseline is only defined for a site that sits at a depth an "
            "ordinary hidden state also sits at."
        )

    if baseline_site is None:
        candidates = [
            name
            for name in states
            if site_depth(name) == depth
            and name.rsplit(".", 1)[-1] in ORDINARY_SITE_SUFFIXES
            and name != candidate_site
        ]
        if not candidates:
            raise GeometryError(
                f"no ordinary hidden state was captured at depth {depth}; capture one of "
                f"{ORDINARY_SITE_SUFFIXES} there, or name a baseline explicitly"
            )
        baseline_site = candidates[0]
    elif baseline_site not in states:
        raise GeometryError(f"no captured site named {baseline_site!r}")
    elif site_depth(baseline_site) != depth:
        raise GeometryError(
            f"{baseline_site!r} sits at depth {site_depth(baseline_site)} and "
            f"{candidate_site!r} at depth {depth}; a matched site must match on depth"
        )

    candidate = flatten_site(states[candidate_site])
    baseline = flatten_site(states[baseline_site])
    if candidate.shape[0] != baseline.shape[0]:
        raise GeometryError(
            f"{candidate_site} has {candidate.shape[0]} rows and {baseline_site} has "
            f"{baseline.shape[0]}; a matched site must describe the same positions"
        )

    adjustments: list[str] = []
    dim = min(candidate.shape[1], baseline.shape[1])
    seed_used: int | None = None
    if candidate.shape[1] != baseline.shape[1]:
        seed_used = projection_seed
        rng = np.random.default_rng(projection_seed)
        if candidate.shape[1] > dim:
            candidate = candidate @ (rng.standard_normal((candidate.shape[1], dim)) / math.sqrt(dim))
            adjustments.append(
                f"candidate randomly projected {states[candidate_site].shape[-1]} -> {dim}"
            )
        if baseline.shape[1] > dim:
            baseline = baseline @ (rng.standard_normal((baseline.shape[1], dim)) / math.sqrt(dim))
            adjustments.append(f"baseline randomly projected to {dim} dimensions")

    return MatchedSites(
        candidate_site=candidate_site,
        baseline_site=baseline_site,
        depth=depth,
        dim=int(dim),
        candidate=candidate,
        baseline=baseline,
        adjustments=tuple(adjustments),
        projection_seed=seed_used,
    )


@dataclass(frozen=True)
class MatchedSiteComparison:
    """A geometric statement about a mechanism site, with its baseline attached.

    There is deliberately no accessor that returns the candidate's numbers alone.
    §7.4 forbids reporting a probe result without a matched-site baseline, and
    the cheapest way to keep that rule is to make the record that carries the
    result unable to omit the comparison.
    """

    sites: MatchedSites
    candidate: GeometryReport
    baseline: GeometryReport
    similarity: float
    """Linear CKA between the two sites, so "the mechanism variable differs from
    the ordinary one" is a measured quantity and not an inference from two
    tables."""

    def difference(self) -> dict[str, float]:
        left, right = self.candidate.scalars(), self.baseline.scalars()
        return {
            key: float(left[key]) - float(right[key])
            for key in left
            if isinstance(left.get(key), (int, float)) and isinstance(right.get(key), (int, float))
            and not isinstance(left[key], bool)
        }

    def as_dict(self) -> dict:
        return {
            "sites": self.sites.as_dict(),
            "candidate": self.candidate.scalars(),
            "baseline": self.baseline.scalars(),
            "difference": self.difference(),
            "representation_similarity": self.similarity,
        }


def matched_site_comparison(
    matched: MatchedSites,
    features: np.ndarray,
    split: ProbeSplit,
    *,
    active: np.ndarray | None = None,
    ridge: float = RIDGE,
) -> MatchedSiteComparison:
    """Run the identical measurement protocol on both halves of a matched pair."""
    return MatchedSiteComparison(
        sites=matched,
        candidate=measure_geometry(
            matched.candidate, features, split, active=active,
            site=matched.candidate_site, ridge=ridge,
        ),
        baseline=measure_geometry(
            matched.baseline, features, split, active=active,
            site=matched.baseline_site, ridge=ridge,
        ),
        similarity=representation_similarity(matched.candidate, matched.baseline),
    )


# --------------------------------------------------------------------------- #
# One run's whole §6.2 record
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RunGeometry:
    """Everything §6.2 asks of one trained model, assembled from captured sites.

    Pure in the same sense as the rest of this module: it takes arrays, not a
    model. The runner does the forward pass and the capture; nothing here knows
    what an architecture is, which is what will let A1, A2 and A4 reuse it
    unchanged by naming their own mechanism sites.
    """

    trajectory: DepthTrajectory
    matched: tuple[MatchedSiteComparison, ...]
    split: ProbeSplit
    primary_site: str
    feature_banks: dict[str, tuple[int, ...]]

    @property
    def primary(self) -> GeometryReport:
        return self.trajectory.reports[self.primary_site]

    def summary(self) -> dict:
        """The scalar record that goes into ``summary.json``."""
        return {
            "geometry_version": GEOMETRY_VERSION,
            "primary_site": self.primary_site,
            "primary": self.primary.scalars(),
            "by_bank": self.bank_summary(),
            "split": self.split.as_dict(),
            "per_site": {name: report.scalars() for name, report in self.trajectory.reports.items()},
            "curves": {name: curve.as_dict() for name, curve in self.trajectory.curves.items()},
            "matched_sites": [comparison.as_dict() for comparison in self.matched],
            "measure_status": {spec.name: spec.status for spec in GEOMETRY_MEASURES},
        }

    def bank_summary(self) -> dict:
        """The primary site's per-feature measures, split by feature bank.

        Content, key and operator features are three different kinds of thing —
        one is transported, one addresses, one marks — and averaging a purity
        over all three answers a question nobody asked. Split here rather than
        left to a reader with the npz, because the split is a property of the
        generator and belongs beside the numbers.
        """
        report = self.primary
        out: dict[str, dict] = {}
        for bank, indices in self.feature_banks.items():
            if not indices:
                continue
            index = np.asarray(indices, dtype=np.int64)
            out[bank] = {
                "n_features": int(index.size),
                "mean_purity": _nanmean(report.purity[index]),
                "probe_macro_r2": _nanmean(report.probe.r2[index]),
                "probe_macro_auc": _nanmean(report.probe.auc[index]),
                "mean_capacity": _nanmean(report.capacity.per_feature[index]),
                "mean_direction_norm": float(report.directions.norms[index].mean()),
                "n_undefined": int(report.directions.undefined[index].sum()),
                "n_collinear": int(report.directions.collinear[index].sum()),
            }
        return out

    def arrays(self) -> dict[str, np.ndarray]:
        """The per-feature and per-pair record that goes into the npz.

        Keys are ``"<site>::<array>"``. Every per-feature array is kept at full
        resolution: the scalars in ``summary.json`` are averages, and an average
        is where a bimodal feature bank goes to hide.
        """
        payload: dict[str, np.ndarray] = {}
        for name, report in self.trajectory.reports.items():
            for array_name, array in report.arrays().items():
                payload[f"{name}::{array_name}"] = np.asarray(array, dtype=np.float64)
        for comparison in self.matched:
            for side, report in (
                ("candidate", comparison.candidate),
                ("baseline", comparison.baseline),
            ):
                prefix = f"matched:{comparison.sites.candidate_site}:{side}"
                for array_name in ("purity", "probe_r2", "probe_auc"):
                    payload[f"{prefix}::{array_name}"] = np.asarray(
                        report.arrays()[array_name], dtype=np.float64
                    )
        for bank, indices in self.feature_banks.items():
            payload[f"bank::{bank}"] = np.asarray(indices, dtype=np.int64)
        payload["__sites__"] = np.asarray(list(self.trajectory.sites))
        return payload


def run_geometry(
    states: Mapping[str, np.ndarray],
    features: np.ndarray,
    active: np.ndarray,
    example_of_row: np.ndarray,
    *,
    trajectory_sites: Sequence[str],
    mechanism_sites: Sequence[str] = (),
    references: Mapping[str, np.ndarray] | None = None,
    feature_banks: Mapping[str, Sequence[int]] | None = None,
    primary_site: str | None = None,
    template_of_example: Sequence[str] | np.ndarray | None = None,
    split_seed: int = 20260810,
    projection_seed: int = 20260810,
    ridge: float = RIDGE,
) -> RunGeometry:
    """Measure one run's captured sites, with a matched baseline for each mechanism site.

    ``trajectory_sites`` must be in forward order — it becomes the depth axis.
    ``mechanism_sites`` each get a :class:`MatchedSiteComparison` against the
    ordinary hidden state at the same depth, because a geometric statement about
    a mechanism variable that does not carry one is not reportable under §7.4.
    """
    missing = [name for name in (*trajectory_sites, *mechanism_sites) if name not in states]
    if missing:
        raise GeometryError(f"sites were named but not captured: {missing}")

    split = probe_split(
        example_of_row, seed=split_seed, template_of_example=template_of_example
    )
    trajectory = depth_trajectory(
        {name: states[name] for name in trajectory_sites},
        features,
        split,
        active=active,
        references=references,
        ridge=ridge,
    )
    matched = tuple(
        matched_site_comparison(
            matched_site_baseline(states, site, projection_seed=projection_seed),
            features,
            split,
            active=active,
            ridge=ridge,
        )
        for site in mechanism_sites
    )
    return RunGeometry(
        trajectory=trajectory,
        matched=matched,
        split=split,
        primary_site=primary_site or trajectory_sites[-1],
        feature_banks={k: tuple(int(i) for i in v) for k, v in (feature_banks or {}).items()},
    )


# --------------------------------------------------------------------------- #
# Constructed representations with known answers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ConstructedRepresentation:
    """Hidden states built from a basis that is known rather than estimated."""

    name: str
    hidden: np.ndarray
    features: np.ndarray
    active: np.ndarray
    basis: np.ndarray | None
    """``(F, d)`` true directions, or ``None`` where the construction has no
    basis to compare against (pure noise)."""
    note: str


@dataclass(frozen=True)
class Expectation:
    """What a measure must read on one constructed case, and why.

    ``kind`` is ``exact`` (within ``tolerance``), ``at_most``, ``at_least``, or
    ``between``. Inequalities are here because several expectations are known by
    argument rather than by formula — the cosine between two random directions in
    48 dimensions concentrates near 0.115 but is not a closed-form constant of
    the construction, and asserting a fabricated decimal for it would be worse
    than asserting the bound that the argument actually supports.
    """

    measure: str
    kind: str
    value: float | tuple[float, float]
    tolerance: float = 0.0
    reason: str = ""

    def describe(self) -> str:
        if self.kind == "exact":
            return f"{self.value:g} ±{self.tolerance:g}"
        if self.kind == "at_most":
            return f"<= {self.value:g}"
        if self.kind == "at_least":
            return f">= {self.value:g}"
        low, high = self.value  # type: ignore[misc]
        return f"[{low:g}, {high:g}]"

    def holds(self, measured: float) -> bool:
        if not np.isfinite(measured):
            return False
        if self.kind == "exact":
            return abs(measured - float(self.value)) <= self.tolerance
        if self.kind == "at_most":
            return measured <= float(self.value) + self.tolerance
        if self.kind == "at_least":
            return measured >= float(self.value) - self.tolerance
        if self.kind == "between":
            low, high = self.value  # type: ignore[misc]
            return low - self.tolerance <= measured <= high + self.tolerance
        raise GeometryError(f"unknown expectation kind {self.kind!r}")


@dataclass(frozen=True)
class ConstructedCase:
    """One constructed representation, its build recipe, and its known answers."""

    name: str
    description: str
    n_features: int
    d_model: int
    n_rows: int
    expectations: tuple[Expectation, ...]
    sparsity: float = 0.2
    seed: int = 20260810


_ROWS = 8192
_SPARSITY = 0.2


def _draw_features(rng: np.random.Generator, n_rows: int, n_features: int, sparsity: float):
    """Bernoulli activity times ``Uniform(0, 1)`` magnitude — the generator's shape.

    Independently per feature, deliberately: the constructed cases exist to
    isolate representation geometry, and correlated features would mix "the
    representation superposes them" with "the data superposes them".
    """
    active = rng.random((n_rows, n_features)) < sparsity
    values = np.where(active, rng.random((n_rows, n_features)), 0.0)
    return values, active


def constructed_representation(case: str, *, seed: int = 20260810, n_rows: int = _ROWS,
                               n_features: int = 32, d_model: int = 48,
                               sparsity: float = _SPARSITY) -> ConstructedRepresentation:
    """Build one of the five constructed representations."""
    rng = np.random.default_rng(seed)
    values, active = _draw_features(rng, n_rows, n_features, sparsity)

    if case == "orthogonal_basis":
        if d_model < n_features:
            raise GeometryError("the orthogonal case requires d >= F")
        basis = np.eye(d_model)[:n_features]
        return ConstructedRepresentation(
            case, values @ basis, values, active, basis,
            "one unit axis per feature; the ceiling every other case is read against",
        )

    if case == "known_superposition":
        # Antipodal pairs: features 2k and 2k+1 share axis k with opposite sign.
        # Each feature then owns exactly half a dimension, so the total capacity
        # is d by construction and the measured value has an exact target.
        if n_features != 2 * d_model:
            raise GeometryError("the superposition case is built for F = 2d")
        basis = np.zeros((n_features, d_model))
        for k in range(d_model):
            basis[2 * k, k] = 1.0
            basis[2 * k + 1, k] = -1.0
        return ConstructedRepresentation(
            case, values @ basis, values, active, basis,
            "F = 2d features in antipodal pairs; capacity is exactly d by construction",
        )

    if case == "random_rotation":
        if d_model < n_features:
            raise GeometryError("the rotation case rotates the orthogonal case")
        rotation = np.linalg.qr(np.random.default_rng(seed + 1).standard_normal(
            (d_model, d_model)))[0]
        basis = np.eye(d_model)[:n_features]
        return ConstructedRepresentation(
            case, (values @ basis) @ rotation, values, active, basis @ rotation,
            "the orthogonal case in a rotated basis; every invariant must not move",
        )

    if case == "degenerate_collapse":
        basis = np.zeros((n_features, d_model))
        basis[:, 0] = 1.0
        return ConstructedRepresentation(
            case, values @ basis, values, active, basis,
            "every feature on one axis; the floor, and the case that must not produce NaN",
        )

    if case == "pure_noise":
        hidden = rng.standard_normal((n_rows, d_model))
        return ConstructedRepresentation(
            case, hidden, values, active, None,
            "hidden states independent of the features; every measure at its null",
        )

    raise GeometryError(f"unknown constructed case {case!r}")


CONSTRUCTED_CASES: tuple[ConstructedCase, ...] = (
    ConstructedCase(
        name="orthogonal_basis",
        description="d=48 >= F=32, one unit axis per feature",
        n_features=32,
        d_model=48,
        n_rows=_ROWS,
        expectations=(
            Expectation("probe_macro_r2", "exact", 1.0, 0.01,
                        "an orthogonal readout inverts the encoding exactly"),
            Expectation("probe_macro_auc", "exact", 1.0, 0.01, "as above"),
            Expectation("mean_purity", "exact", 1.0, 0.02,
                        "each readout responds only to its own feature"),
            Expectation("interference_fraction", "exact", 0.0, 0.02,
                        "no off-diagonal response exists to measure"),
            Expectation("mean_abs_off_diagonal_cosine", "exact", 0.0, 0.02,
                        "the directions are orthogonal by construction"),
            Expectation("effective_rank", "exact", 32.0, 0.5,
                        "F equal-variance directions, so the spectrum is flat over F"),
            Expectation("participation_ratio", "exact", 32.0, 0.5, "as above"),
            Expectation("capacity_total", "exact", 32.0, 0.5,
                        "each feature owns a whole dimension"),
            Expectation("alignment_reference_mean", "exact", 1.0, 0.01,
                        "the estimator must recover the basis it was built from"),
            Expectation("alignment_explained_mean", "exact", 1.0, 0.01,
                        "the encoding is linear and noiseless, so one direction explains all"),
            Expectation("alignment_marginal_mean", "exact", 1.0, 0.02,
                        "with independent features the marginal and partial estimators agree"),
        ),
    ),
    ConstructedCase(
        name="known_superposition",
        description="d=16 < F=32, antipodal pairs sharing each axis",
        n_features=32,
        d_model=16,
        n_rows=_ROWS,
        expectations=(
            Expectation("capacity_total", "exact", 16.0, 0.5,
                        "two features per axis, half a dimension each, over 16 axes"),
            Expectation("effective_rank", "exact", 16.0, 0.5, "the space is d-dimensional"),
            Expectation("participation_ratio", "exact", 16.0, 0.5, "as above"),
            Expectation("mean_purity", "exact", 0.5, 0.02,
                        "a readout cannot separate a feature from its antipode"),
            Expectation("probe_macro_r2", "exact", 0.5, 0.02,
                        "predicting a from a-b recovers half its variance"),
            Expectation("mean_abs_off_diagonal_cosine", "exact", 1.0 / 31.0, 0.01,
                        "one partner at cosine -1 among F-1 off-diagonal cells"),
            Expectation("alignment_reference_mean", "exact", 1.0, 0.01,
                        "the estimator recovers the construction despite the sharing"),
            Expectation("alignment_explained_mean", "exact", 1.0, 0.01,
                        "superposition puts the interference in the readout, not the encoding: "
                        "each feature is still carried by exactly one direction"),
            Expectation("probe_macro_auc", "between", (0.75, 0.95), 0.0,
                        "activity is partly recoverable from a signed sum, but not fully"),
        ),
    ),
    ConstructedCase(
        name="random_rotation",
        description="the orthogonal case times a random orthogonal matrix",
        n_features=32,
        d_model=48,
        n_rows=_ROWS,
        expectations=(
            Expectation("probe_macro_r2", "exact", 1.0, 0.01, "a rotation is invertible"),
            Expectation("mean_purity", "exact", 1.0, 0.02, "as above"),
            Expectation("effective_rank", "exact", 32.0, 0.5, "the spectrum is unchanged"),
            Expectation("participation_ratio", "exact", 32.0, 0.5, "as above"),
            Expectation("capacity_total", "exact", 32.0, 0.5, "angles are unchanged"),
            Expectation("mean_abs_off_diagonal_cosine", "exact", 0.0, 0.02, "angles are unchanged"),
            Expectation("alignment_reference_mean", "exact", 1.0, 0.01,
                        "against the rotated basis, which is the true one"),
        ),
    ),
    ConstructedCase(
        name="degenerate_collapse",
        description="every feature written onto one axis",
        n_features=32,
        d_model=48,
        n_rows=_ROWS,
        expectations=(
            Expectation("effective_rank", "exact", 1.0, 0.02, "the hidden state is rank one"),
            Expectation("participation_ratio", "exact", 1.0, 0.02, "as above"),
            Expectation("mean_abs_off_diagonal_cosine", "exact", 1.0, 0.02,
                        "every direction is the same direction"),
            Expectation("mean_purity", "exact", 1.0 / 32.0, 0.02,
                        "each readout is equally responsible for every feature"),
            Expectation("probe_macro_r2", "exact", 1.0 / 32.0, 0.03,
                        "only the sum of the features survives"),
            Expectation("capacity_total", "exact", 1.0, 0.05, "one dimension in use"),
            Expectation("interference_fraction", "between", (0.35, 0.65), 0.0,
                        "off-diagonal response equals the diagonal"),
            Expectation("alignment_reference_mean", "exact", 1.0, 0.01,
                        "alignment is blind to collapse; the cosine matrix is not"),
            Expectation("alignment_explained_mean", "exact", 1.0, 0.01,
                        "every feature is perfectly carried by one direction — the same one. "
                        "This is the cell that retires all three alignments to diagnostic"),
        ),
    ),
    ConstructedCase(
        name="pure_noise",
        description="hidden states drawn independently of the features",
        n_features=32,
        d_model=48,
        n_rows=_ROWS,
        expectations=(
            Expectation("probe_macro_r2", "at_most", 0.02, 0.0,
                        "nothing is decodable out of sample; the null is zero, not positive"),
            Expectation("probe_macro_auc", "exact", 0.5, 0.03, "chance"),
            Expectation("mean_purity", "exact", 1.0 / 32.0, 0.03,
                        "a readout is equally likely to respond to any feature"),
            Expectation("alignment_explained_mean", "at_most", 0.01, 0.0,
                        "the hidden state does not move with the feature, so no direction "
                        "explains it: this is the only alignment whose null is zero"),
            Expectation("alignment_marginal_mean", "between", (0.70, 0.95), 0.0,
                        "NOT zero, and this is the point. The partial and the marginal "
                        "estimator are computed from the same sampling noise on the same "
                        "rows, so on pure noise they agree with each other about it. A "
                        "measure whose null is 0.83 cannot be read as evidence of anything"),
            Expectation("interference_fraction", "between", (0.35, 0.65), 0.0,
                        "diagonal and off-diagonal responses have the same magnitude"),
            Expectation("effective_rank", "between", (44.0, 48.0), 0.0,
                        "isotropic noise fills every direction: the null is d, not zero"),
            Expectation("participation_ratio", "between", (44.0, 48.0), 0.0, "as above"),
            Expectation("mean_abs_off_diagonal_cosine", "between", (0.05, 0.20), 0.0,
                        "the mean absolute cosine of random directions in 48 dimensions"),
            Expectation("capacity_total", "between", (15.0, 24.0), 0.0,
                        "F random directions in d dimensions look packed: this is why "
                        "capacity is a diagnostic and never a headline"),
        ),
    ),
)


# --------------------------------------------------------------------------- #
# The measure register: what each measure reads at the ceiling, the floor, and
# the null — and whether that is good enough to report.
# --------------------------------------------------------------------------- #

MIN_NULL_GAP = 0.25
"""How far the noise null and the collapse floor must each sit from the
orthogonal ceiling, as a fraction of the measure's own observed span. A measure
that fails this cannot tell the reference cases apart at all."""

MAX_NULL_POSITION = 0.5
"""Where the noise null may sit on the axis running from the collapse floor to
the orthogonal ceiling. A null past the midpoint means a reader calibrated on
that scale would see noise and call it structure, which is the failure this whole
validation exists to catch. Prompt 03's ``MAX_MARGINAL_SCORE`` is the same idea
applied to the frequency ceiling: it is not enough for a reference to be
*distinguishable*, it must not itself reach a value that reads as success."""


@dataclass(frozen=True)
class MeasureSpec:
    """One ruler: what it means, where its null is, and whether it survived.

    ``status`` is a decision recorded in source. ``--selftest`` re-derives the
    rule from the constructed cases and fails if the decision and the evidence
    have come apart — in either direction. This is prompt 03's convention applied
    to geometry, and for the same reason: a measure whose retirement can be
    reversed by whoever needs it to pass is not a ruler.

    ``retained`` means the measure can carry a claim on its own. ``diagnostic``
    means report it, but never alone. The split is derived, not chosen, and the
    outcome is worth stating plainly: **the four measures that survive are
    exactly the four built on an out-of-sample readout.** Every purely geometric
    measure here — the cosine matrix, capacity, effective rank, participation
    ratio, all three alignments — has a null a reader would mistake for
    structure, because geometry computed from estimated directions is geometry
    whether or not the directions encode anything.
    """

    name: str
    definition: str
    status: str
    reason: str

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "definition": self.definition,
            "status": self.status,
            "reason": self.reason,
        }


GEOMETRY_MEASURES: tuple[MeasureSpec, ...] = (
    MeasureSpec(
        name="probe_macro_r2",
        definition="out-of-sample R^2 of a linear readout of each feature, macro-averaged",
        status="retained",
        reason="1.00 at the ceiling, -0.01 on noise, 1/F on collapse — the primary §6.2 number",
    ),
    MeasureSpec(
        name="probe_macro_auc",
        definition="out-of-sample AUC for feature activity from the same readout",
        status="retained",
        reason="1.00 at the ceiling and 0.50 on noise: the null is chance, exactly",
    ),
    MeasureSpec(
        name="mean_purity",
        definition="share of each readout's response attributable to its own feature",
        status="retained",
        reason="1.00 at the ceiling, 1/F on both noise and collapse — an honest chance floor",
    ),
    MeasureSpec(
        name="interference_fraction",
        definition="off-diagonal share of the out-of-sample readout response matrix",
        status="retained",
        reason="0.00 at the ceiling and 0.49 on noise; bounded, so it has no divide-by-zero",
    ),
    MeasureSpec(
        name="mean_abs_off_diagonal_cosine",
        definition="mean absolute cosine between estimated feature directions",
        status="diagnostic",
        reason=(
            "the null is not zero. Two random directions in d dimensions have mean absolute "
            "cosine sqrt(2/(pi d)) = 0.115 at d=48, and that is what pure noise reads — 88% "
            "of the way from total collapse to perfect orthogonality. A nearly-orthogonal "
            "cosine matrix is what noise produces; it must be read against the "
            "dimension-specific null and not against zero"
        ),
    ),
    MeasureSpec(
        name="effective_rank",
        definition="entropy effective rank of the centred representation spectrum",
        status="diagnostic",
        reason=(
            "isotropic noise fills every direction, so the null is d — *past* the ceiling of "
            "F rather than below it. A high effective rank is not evidence of structure, and "
            "the measure has no good direction: it is a count, and only a count"
        ),
    ),
    MeasureSpec(
        name="participation_ratio",
        definition="(sum lambda)^2 / sum lambda^2 over the covariance spectrum",
        status="diagnostic",
        reason="as effective_rank, weighting the leading directions more heavily",
    ),
    MeasureSpec(
        name="capacity_total",
        definition="summed per-feature dimensionality of the estimated directions",
        status="diagnostic",
        reason=(
            "pure noise reports 18.5 of 48 dimensions in use — 57% of the way from collapse "
            "to the ceiling — because F random directions in d dimensions are about as "
            "spread out as F learned ones. Capacity measures the geometry of the estimated "
            "directions and cannot tell an encoded direction from a fitted artifact"
        ),
    ),
    MeasureSpec(
        name="alignment_explained_mean",
        definition="R^2 of the rank-one model a_f w_f against feature f's residual variation",
        status="diagnostic",
        reason=(
            "the best-behaved of the three alignments — its null really is zero — but it "
            "reads 1.00 on total collapse, because every feature is perfectly carried by one "
            "direction there and that direction happens to be the same one"
        ),
    ),
    MeasureSpec(
        name="alignment_marginal_mean",
        definition="cosine between each feature's partial and marginal direction estimates",
        status="diagnostic",
        reason=(
            "its null is 0.83, not zero: on pure noise the two estimators are computed from "
            "the same sampling noise on the same rows and therefore agree about it. It is 1.0 "
            "on every constructed case with independent features, where the two estimators "
            "coincide identically, and separates only when features co-activate"
        ),
    ),
    MeasureSpec(
        name="alignment_reference_mean",
        definition="cosine between each estimated direction and a known true direction",
        status="diagnostic",
        reason=(
            "1.0 on total collapse, and undefined on noise, which has no true basis to align "
            "to. Its real use is as an estimator check where the basis is known — the encoder "
            "weights of a linear embedding, which it recovers to 1.0000"
        ),
    ),
)

MEASURE_SPEC_BY_NAME: dict[str, MeasureSpec] = {spec.name: spec for spec in GEOMETRY_MEASURES}
RETAINED_MEASURES: tuple[str, ...] = tuple(
    spec.name for spec in GEOMETRY_MEASURES if spec.status == "retained"
)
DIAGNOSTIC_MEASURES: tuple[str, ...] = tuple(
    spec.name for spec in GEOMETRY_MEASURES if spec.status == "diagnostic"
)


# --------------------------------------------------------------------------- #
# Validation: measure every constructed case and compare against the expectation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CaseResult:
    """One constructed case measured, with every expectation checked."""

    case: str
    description: str
    n_features: int
    d_model: int
    n_rows: int
    measured: dict[str, float]
    rows: tuple[dict, ...]
    finite: bool

    @property
    def failures(self) -> tuple[dict, ...]:
        return tuple(row for row in self.rows if not row["ok"])

    def as_dict(self) -> dict:
        return {
            "case": self.case,
            "description": self.description,
            "n_features": self.n_features,
            "d_model": self.d_model,
            "n_rows": self.n_rows,
            "measured": self.measured,
            "expectations": list(self.rows),
            "all_scalars_finite": self.finite,
        }


@dataclass(frozen=True)
class ValidationReport:
    """The constructed-case table, plus the cross-case invariants."""

    cases: tuple[CaseResult, ...]
    invariants: dict
    measures: tuple[dict, ...]

    @property
    def ok(self) -> bool:
        return not any(case.failures for case in self.cases)

    def as_dict(self) -> dict:
        return {
            "geometry_version": GEOMETRY_VERSION,
            "cases": [case.as_dict() for case in self.cases],
            "invariants": self.invariants,
            "measures": list(self.measures),
        }


def _measure_case(case: ConstructedCase) -> tuple[CaseResult, GeometryReport, ConstructedRepresentation]:
    built = constructed_representation(
        case.name,
        seed=case.seed,
        n_rows=case.n_rows,
        n_features=case.n_features,
        d_model=case.d_model,
        sparsity=case.sparsity,
    )
    split = probe_split(np.arange(case.n_rows), seed=case.seed + 7)
    report = measure_geometry(
        built.hidden,
        built.features,
        split,
        active=built.active,
        reference=built.basis,
        site=case.name,
    )
    scalars = report.scalars()
    rows = []
    for expectation in case.expectations:
        value = scalars.get(expectation.measure)
        rows.append(
            {
                "measure": expectation.measure,
                "expected": expectation.describe(),
                "measured": None if value is None else float(value),
                "ok": value is not None and expectation.holds(float(value)),
                "reason": expectation.reason,
            }
        )
    numeric = [
        v for k, v in scalars.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool) and k != "alignment_reference_mean"
    ]
    return (
        CaseResult(
            case=case.name,
            description=case.description,
            n_features=case.n_features,
            d_model=case.d_model,
            n_rows=case.n_rows,
            measured={k: v for k, v in scalars.items() if isinstance(v, (int, float))},
            rows=tuple(rows),
            finite=all(np.isfinite(v) for v in numeric),
        ),
        report,
        built,
    )


def validate_constructed_cases() -> ValidationReport:
    """Measure all five constructed cases and re-derive every recorded decision."""
    results: list[CaseResult] = []
    reports: dict[str, GeometryReport] = {}
    built: dict[str, ConstructedRepresentation] = {}
    for case in CONSTRUCTED_CASES:
        result, report, representation = _measure_case(case)
        results.append(result)
        reports[case.name] = report
        built[case.name] = representation

    orthogonal = reports["orthogonal_basis"]
    rotated = reports["random_rotation"]
    invariant_measures = (
        "effective_rank",
        "participation_ratio",
        "mean_abs_off_diagonal_cosine",
        "interference_fraction",
        "mean_purity",
        "probe_macro_r2",
        "probe_macro_auc",
        "capacity_total",
    )
    left, right = orthogonal.scalars(), rotated.scalars()
    drift = {name: abs(float(left[name]) - float(right[name])) for name in invariant_measures}

    # The one thing a rotation *must* move: the cosine to a basis fixed outside
    # the representation. Measured by scoring the rotated states against the
    # unrotated basis, which is what an external reference would be.
    rotated_case = next(c for c in CONSTRUCTED_CASES if c.name == "random_rotation")
    rotated_split = probe_split(np.arange(rotated_case.n_rows), seed=rotated_case.seed + 7)
    unrotated_basis = np.eye(rotated_case.d_model)[: rotated_case.n_features]
    against_stale = alignment(
        built["random_rotation"].hidden,
        built["random_rotation"].features,
        active=built["random_rotation"].active,
        reference=unrotated_basis,
        rows=rotated_split.train,
    )

    similarity = {
        "identical": representation_similarity(
            built["orthogonal_basis"].hidden, built["orthogonal_basis"].hidden
        ),
        "rotated": representation_similarity(
            built["orthogonal_basis"].hidden, built["random_rotation"].hidden
        ),
        "independent_noise": representation_similarity(
            built["pure_noise"].hidden,
            constructed_representation("pure_noise", seed=987654321).hidden,
        ),
    }

    def read(case_name: str) -> dict[str, float]:
        scalars = reports[case_name].scalars()
        # ``alignment_reference_mean`` is ``None`` on the noise case, which has
        # no true basis to align to. Read as NaN rather than coerced to a number:
        # a measure that cannot be evaluated against the null has not passed the
        # null test, and the rule below treats it that way.
        return {
            name: float("nan") if scalars.get(name) is None else float(scalars[name])
            for name in MEASURE_SPEC_BY_NAME
        }

    ceiling, noise, collapse = read("orthogonal_basis"), read("pure_noise"), read(
        "degenerate_collapse"
    )
    verdicts = []
    for spec in GEOMETRY_MEASURES:
        noise_gap = abs(ceiling[spec.name] - noise[spec.name])
        collapse_gap = abs(ceiling[spec.name] - collapse[spec.name])
        span = max(noise_gap if np.isfinite(noise_gap) else 0.0,
                   collapse_gap if np.isfinite(collapse_gap) else 0.0,
                   1e-12)
        # Normalised by the measure's own observed span, so a count-valued
        # measure (effective rank) and a fraction-valued one (purity) are held
        # to the same rule without pretending they share units.
        separates_noise = bool(np.isfinite(noise_gap) and noise_gap / span >= MIN_NULL_GAP)
        separates_collapse = bool(
            np.isfinite(collapse_gap) and collapse_gap / span >= MIN_NULL_GAP
        )
        # Where the noise null sits on the collapse -> ceiling axis. Past the
        # midpoint means noise reads as structure on this measure's own scale.
        denominator = ceiling[spec.name] - collapse[spec.name]
        position = (
            (noise[spec.name] - collapse[spec.name]) / denominator
            if abs(denominator) > 1e-12
            else float("nan")
        )
        rule_passed = bool(
            separates_noise
            and separates_collapse
            and np.isfinite(position)
            and position <= MAX_NULL_POSITION
        )
        verdicts.append(
            {
                **spec.as_dict(),
                "ceiling": ceiling[spec.name],
                "noise_null": noise[spec.name],
                "collapse_floor": collapse[spec.name],
                "noise_position": float(position),
                "separates_noise": bool(separates_noise),
                "separates_collapse": bool(separates_collapse),
                "rule_passed": rule_passed,
                "agrees": rule_passed == (spec.status == "retained"),
            }
        )

    capacity_curve = _capacity_ladder_curve()

    return ValidationReport(
        cases=tuple(results),
        invariants={
            "rotation_invariance_max_drift": max(drift.values()),
            "rotation_invariance_drift": drift,
            "rotated_alignment_to_unrotated_basis": against_stale.reference_mean,
            "representation_similarity": similarity,
            "capacity_versus_sparsity": {
                name: curve.as_dict() for name, curve in capacity_curve["curves"].items()
            },
            "capacity_versus_sparsity_expected": capacity_curve["expected"],
            "capacity_versus_sparsity_max_error": capacity_curve["max_error"],
        },
        measures=tuple(verdicts),
    )


def _capacity_ladder_curve() -> dict:
    """A capacity-versus-sparsity family whose answer is fixed in advance.

    At each sparsity only ``k`` of the ``F`` features are given an orthogonal
    direction and the rest are written nowhere, so the measured capacity has an
    exact target at every point on the curve and the curve's shape is a
    prediction rather than an observation. Validating the *curve* and not just
    the point matters because §6.2 asks for capacity as a function of sparsity,
    and a function can be wrong in ways none of its values are.
    """
    sparsities = (0.05, 0.10, 0.20, 0.40)
    represented = (4, 8, 16, 24)
    n_features, d_model, n_rows = 32, 48, 4096
    samples = []
    for index, (sparsity, k) in enumerate(zip(sparsities, represented, strict=True)):
        rng = np.random.default_rng(20260810 + index)
        values, _ = _draw_features(rng, n_rows, n_features, sparsity)
        basis = np.zeros((n_features, d_model))
        basis[:k] = np.eye(d_model)[:k]
        samples.append(SparsitySample(sparsity=sparsity, hidden=values @ basis, features=values))
    curves = capacity_versus_sparsity(samples)
    measured = curves["capacity_total"].y
    errors = [abs(m - k) for m, k in zip(measured, represented, strict=True)]
    return {
        "curves": curves,
        "expected": {
            "sparsity": list(sparsities),
            "capacity_total": [float(k) for k in represented],
        },
        "max_error": float(max(errors)),
    }


# --------------------------------------------------------------------------- #
# §6.2 measure 9, applied: how much does one architecture differ from itself?
# --------------------------------------------------------------------------- #

STABILITY_MEASURES: tuple[str, ...] = (
    "probe_macro_r2",
    "probe_macro_auc",
    "mean_purity",
    "interference_fraction",
    "mean_abs_off_diagonal_cosine",
    "effective_rank",
    "participation_ratio",
    "capacity_total",
)


def across_runs(run_dirs: Sequence[Path | str]) -> dict:
    """Compare recorded geometry across runs. Reads artifacts, runs no model.

    Every later statement of the form "architecture X differs from architecture
    Y in feature geometry" is uninterpretable without knowing how much the *same*
    architecture differs from itself, and that reference has to be measured, not
    assumed small. This is that measurement, and it is deliberately built from
    committed artifacts — the summary's scalars and the npz's arrays — so it can
    be regenerated by anyone holding the run directories and nothing else.

    Similarity is computed between the two runs' **estimated feature direction
    matrices**, rows indexed by ground-truth feature. That is the right object
    rather than the raw hidden states, for two reasons: the feature index is the
    only correspondence two independently-initialised models share, and hidden
    states are dominated by structure both runs hold in common — a position
    table, a LayerNorm offset — which would report a high similarity for reasons
    having nothing to do with the features. Linear CKA on the direction matrices
    is invariant to the change of basis two seeds are free to differ by, which is
    exactly the freedom that should not count as a difference.
    """
    runs = [Path(directory) for directory in run_dirs]
    if len(runs) < 2:
        raise GeometryError("comparing geometry across runs needs at least two runs")

    summaries: list[dict] = []
    arrays: list[dict] = []
    for directory in runs:
        summary = json.loads((directory / "summary.json").read_text())
        if not summary.get("geometry"):
            raise GeometryError(f"{directory.name} recorded no geometry to compare")
        summaries.append(summary)
        with np.load(directory / "geometry_metrics.npz") as loaded:
            arrays.append({name: loaded[name] for name in loaded.files})

    sites = [str(name) for name in arrays[0]["__sites__"]]
    for index, payload in enumerate(arrays[1:], start=1):
        if [str(name) for name in payload["__sites__"]] != sites:
            raise GeometryError(
                f"{runs[index].name} recorded different sites than {runs[0].name}; "
                "a cross-run comparison must be over the same representation"
            )

    similarity: dict[str, dict] = {}
    for site in sites:
        pairs = []
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                pairs.append(
                    {
                        "runs": [runs[i].name, runs[j].name],
                        "direction_similarity": representation_similarity(
                            arrays[i][f"{site}::directions"], arrays[j][f"{site}::directions"]
                        ),
                        "purity_correlation": _correlation(
                            arrays[i][f"{site}::purity"], arrays[j][f"{site}::purity"]
                        ),
                        "probe_r2_correlation": _correlation(
                            arrays[i][f"{site}::probe_r2"], arrays[j][f"{site}::probe_r2"]
                        ),
                    }
                )
        similarity[site] = {
            "pairs": pairs,
            "mean_direction_similarity": float(
                np.mean([p["direction_similarity"] for p in pairs])
            ),
        }

    spread: dict[str, dict] = {}
    primary = summaries[0]["geometry"]["primary_site"]
    for measure in STABILITY_MEASURES:
        values = [float(s["geometry"]["primary"][measure]) for s in summaries]
        spread[measure] = {
            "values": values,
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "range": float(max(values) - min(values)),
        }

    return {
        "geometry_version": GEOMETRY_VERSION,
        "runs": [directory.name for directory in runs],
        "seeds": [int(s["config"]["seed"]) for s in summaries],
        "primary_site": primary,
        "primary_metric": {
            "associative_recall_accuracy": [
                s["final"].get("associative_recall_accuracy") for s in summaries
            ]
        },
        "spread_at_primary_site": spread,
        "similarity_by_site": similarity,
        "note": (
            "the same architecture at different initialisation seeds on identical data; "
            "the seed moves initialisation and batch order, never the generator"
        ),
    }


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation over features defined in both runs."""
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    usable = np.isfinite(x) & np.isfinite(y)
    if usable.sum() < 3:
        return float("nan")
    x, y = x[usable], y[usable]
    x, y = x - x.mean(), y - y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denominator) if denominator > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def format_case_table(report: ValidationReport) -> str:
    """The expected-versus-measured table, exactly as the artifact records it."""
    lines: list[str] = []
    for case in report.cases:
        lines.append("")
        lines.append(f"{case.case}  —  {case.description}  (F={case.n_features}, "
                     f"d={case.d_model}, N={case.n_rows})")
        lines.append(f"  {'measure':<32} {'expected':>16} {'measured':>12}  ok")
        for row in case.rows:
            measured = "n/a" if row["measured"] is None else f"{row['measured']:.4f}"
            lines.append(
                f"  {row['measure']:<32} {row['expected']:>16} {measured:>12}  "
                f"{'yes' if row['ok'] else 'NO'}"
            )
        lines.append(f"  all scalars finite: {'yes' if case.finite else 'NO'}")
    return "\n".join(lines)


def format_stability_table(report: dict) -> str:
    """How far the same architecture moves from itself, measure by measure."""
    lines = [
        (
            f"geometry across {len(report['runs'])} runs at seeds {report['seeds']} "
            f"— {report['geometry_version']}"
        ),
        f"primary site: {report['primary_site']}",
        "",
        f"{'measure':<32} {'mean':>10} {'sd':>10} {'range':>10}   per-seed",
    ]
    for name, entry in report["spread_at_primary_site"].items():
        values = ", ".join(f"{v:.4f}" for v in entry["values"])
        lines.append(
            f"{name:<32} {entry['mean']:>10.4f} {entry['sd']:>10.4f} "
            f"{entry['range']:>10.4f}   {values}"
        )
    lines.append("")
    lines.append(f"{'site':<28} mean cross-seed direction similarity (linear CKA)")
    for site, entry in report["similarity_by_site"].items():
        lines.append(f"{site:<28} {entry['mean_direction_similarity']:.4f}")
    return "\n".join(lines)


def format_measure_table(report: ValidationReport) -> str:
    lines = [
        f"{'measure':<32} {'ceiling':>9} {'noise':>9} {'collapse':>9} {'status':<11} agrees",
    ]
    for row in report.measures:
        lines.append(
            f"{row['name']:<32} {row['ceiling']:>9.4f} {row['noise_null']:>9.4f} "
            f"{row['collapse_floor']:>9.4f} {row['status']:<11} "
            f"{'yes' if row['agrees'] else 'NO'}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Selftest
# --------------------------------------------------------------------------- #

INVARIANTS: tuple[str, ...] = (
    "constructed_cases_meet_their_expectations",
    "degenerate_case_produces_no_nan",
    "rotation_invariant_measures_do_not_move",
    "rotation_moves_what_it_should",
    "representation_similarity_separates",
    "noise_null_is_measured_not_assumed",
    "measure_status_agrees_with_the_rule",
    "capacity_versus_sparsity_matches_the_construction",
    "probe_split_refuses_to_leak",
)


class _Checks:
    """Collects pass/fail with a reason, so the selftest reports every failure.

    A local copy of the capability selftest's helper rather than an import of its
    private class, on the same reasoning: fifteen duplicated lines are cheaper
    than a cross-module private dependency between two independent gates.
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


def run_selftest(*, break_invariant: str | None = None, verbose: bool = True) -> int:
    """Re-measure the five constructed cases and every recorded decision."""
    checks = _Checks(break_invariant)
    report = validate_constructed_cases()
    out: list[str] = [f"geometry selftest — {GEOMETRY_VERSION}", ""]

    failures = {case.case: case.failures for case in report.cases if case.failures}
    checks.record(
        "constructed_cases_meet_their_expectations",
        not failures,
        f"{sum(len(c.rows) for c in report.cases)} expectations over {len(report.cases)} cases; "
        + (
            "all met"
            if not failures
            else "; ".join(
                f"{case}: " + ", ".join(
                    f"{row['measure']} expected {row['expected']} measured {row['measured']:.4f}"
                    for row in rows
                )
                for case, rows in failures.items()
            )
        ),
    )

    collapse = next(c for c in report.cases if c.case == "degenerate_collapse")
    checks.record(
        "degenerate_case_produces_no_nan",
        collapse.finite,
        f"every reported scalar finite on total collapse: {collapse.finite}; "
        f"purity {collapse.measured['mean_purity']:.4f}, "
        f"effective rank {collapse.measured['effective_rank']:.4f}, "
        f"interference fraction {collapse.measured['interference_fraction']:.4f}",
    )

    drift = report.invariants["rotation_invariance_max_drift"]
    checks.record(
        "rotation_invariant_measures_do_not_move",
        drift < 1e-6,
        f"max drift across {len(report.invariants['rotation_invariance_drift'])} "
        f"rotation-invariant measures = {drift:.3e}",
    )

    stale = report.invariants["rotated_alignment_to_unrotated_basis"]
    checks.record(
        "rotation_moves_what_it_should",
        stale is not None and abs(stale) <= 0.35,
        f"alignment of the rotated representation to the *unrotated* basis = "
        f"{stale:.4f} (was 1.0000 before rotation) — a rotation-invariant measure "
        "that also moved here would be measuring the basis, not the geometry",
    )

    similarity = report.invariants["representation_similarity"]
    checks.record(
        "representation_similarity_separates",
        abs(similarity["identical"] - 1.0) < 1e-9
        and abs(similarity["rotated"] - 1.0) < 1e-9
        and similarity["independent_noise"] < 0.05,
        f"identical {similarity['identical']:.6f}, rotated {similarity['rotated']:.6f}, "
        f"independent noise {similarity['independent_noise']:.6f}",
    )

    noise = next(c for c in report.cases if c.case == "pure_noise")
    checks.record(
        "noise_null_is_measured_not_assumed",
        not noise.failures,
        "; ".join(
            f"{row['measure']}={row['measured']:.4f} (expected {row['expected']})"
            for row in noise.rows
        ),
    )

    disagreements = [row["name"] for row in report.measures if not row["agrees"]]
    checks.record(
        "measure_status_agrees_with_the_rule",
        not disagreements,
        f"{len(RETAINED_MEASURES)} retained, {len(DIAGNOSTIC_MEASURES)} diagnostic; "
        f"disagreements: {disagreements or 'none'}",
    )

    error = report.invariants["capacity_versus_sparsity_max_error"]
    expected = report.invariants["capacity_versus_sparsity_expected"]
    checks.record(
        "capacity_versus_sparsity_matches_the_construction",
        error < 0.5,
        f"expected capacity {expected['capacity_total']} at sparsity {expected['sparsity']}; "
        f"measured "
        f"{[round(v, 3) for v in report.invariants['capacity_versus_sparsity']['capacity_total']['y']]}"
        f"; max error {error:.4f}",
    )

    leak_refused = _probe_split_refuses_to_leak()
    checks.record(
        "probe_split_refuses_to_leak",
        leak_refused["refused"] and leak_refused["leak_inflates"],
        f"a split placing one example on both sides is refused: {leak_refused['refused']}; "
        f"a row-wise split of the same data inflates probe R^2 from "
        f"{leak_refused['honest_r2']:.4f} to {leak_refused['leaky_r2']:.4f}",
    )

    out.append(format_measure_table(report))
    out.append(format_case_table(report))
    out.append("")
    out.append("invariants")
    for name, ok, detail in checks.results:
        out.append(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    out.append("")
    out.append("selftest PASSED" if not checks.failed else f"selftest FAILED ({len(checks.failed)})")

    if verbose:
        print("\n".join(out))
    return 0 if not checks.failed else 1


def _probe_split_refuses_to_leak() -> dict:
    """Show that the by-example split is doing work, not just being tidy.

    The construction is the worst case a row-wise split would let through: the
    hidden state is a random code that identifies the *example* and carries no
    image of the features at all, while the features are constant within an
    example. Nothing here is honestly decodable — but a linear readout fitted on
    rows drawn from every example can memorise the map from each code to that
    example's features, and a row-wise split then scores it on rows whose codes
    it has already seen.

    With as many codes as dimensions the memorisation is exact, so the
    difference between the two splits is the whole of the measurement. If they
    scored the same, the discipline would be decoration.
    """
    rng = np.random.default_rng(20260810)
    n_examples, per_example, n_features, d_model = 64, 8, 8, 64
    n_rows = n_examples * per_example
    example_of_row = np.repeat(np.arange(n_examples), per_example)
    per_example_values, _ = _draw_features(rng, n_examples, n_features, 0.3)
    values = per_example_values[example_of_row]
    active = values != 0.0
    per_example_code = rng.standard_normal((n_examples, d_model))
    hidden = per_example_code[example_of_row] + 0.01 * rng.standard_normal((n_rows, d_model))

    honest = probe_split(example_of_row, seed=3)
    leaky_rows = np.random.default_rng(3).permutation(n_rows)
    half = n_rows // 2
    leaky = ProbeSplit(
        train=np.sort(leaky_rows[:half]),
        eval=np.sort(leaky_rows[half:]),
        train_examples=np.array([-1]),
        eval_examples=np.array([-2]),
        n_examples=n_examples,
        seed=3,
        train_fraction=0.5,
    )
    honest_r2 = feature_reconstruction(hidden, values, honest, active=active).macro_r2
    leaky_r2 = feature_reconstruction(hidden, values, leaky, active=active).macro_r2

    refused = False
    try:
        ProbeSplit(
            train=np.array([0, 1]),
            eval=np.array([2, 3]),
            train_examples=np.array([0, 1]),
            eval_examples=np.array([1, 2]),
            n_examples=3,
            seed=0,
            train_fraction=0.5,
        )
    except GeometryError:
        refused = True

    return {
        "refused": refused,
        "honest_r2": honest_r2,
        "leaky_r2": leaky_r2,
        "leak_inflates": leaky_r2 > 0.5 and honest_r2 < 0.2,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="architecture_mechanics.metrics.geometry")
    parser.add_argument("--selftest", action="store_true",
                        help="validate every measure against the five constructed cases")
    parser.add_argument("--break-invariant", choices=INVARIANTS, default=None,
                        help="force one invariant to fail; used to prove the gate reports failure")
    parser.add_argument("--table", action="store_true",
                        help="print the expected-versus-measured table and exit")
    parser.add_argument("--across-runs", nargs="+", metavar="RUN_DIR", default=None,
                        help="compare recorded geometry across two or more run directories; "
                             "reads summary.json and geometry_metrics.npz, runs no model")
    parser.add_argument("--json", metavar="PATH", default=None,
                        help="write the report as JSON")
    args = parser.parse_args(argv)

    if args.across_runs:
        report = across_runs(args.across_runs)
        print(format_stability_table(report))
        if args.json:
            _write_json(Path(args.json), report)
        return 0

    if args.table:
        report = validate_constructed_cases()
        print(format_measure_table(report))
        print(format_case_table(report))
        if args.json:
            _write_json(Path(args.json), report.as_dict())
        return 0 if report.ok else 1

    if args.selftest or args.break_invariant:
        status = run_selftest(break_invariant=args.break_invariant)
        if args.json:
            _write_json(Path(args.json), validate_constructed_cases().as_dict())
        return status

    parser.print_help()
    return 0


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
