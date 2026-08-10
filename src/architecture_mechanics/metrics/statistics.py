"""§7.4 statistical discipline: the estimators, and what they do when nothing is there.

§7.4's "avoid" list is not a list of mistakes. It is a list of *methods for
producing a publishable-looking result from noise*, and every one of them works.
Select the best of five seeds and you can show almost any difference you like.
Treat four thousand tokens as four thousand samples and a confidence interval
narrows by a factor of thirty. Test twenty-four mechanism sites at 0.05 and you
will find one.

The defence is not vigilance. It is having measured what each estimator does
when the answer is known to be nothing, and writing the number down. Everything
in this module is calibrated against two ground truths:

- a **null** — two arms drawn from the same distribution, no effect at all — over
  6000 synthetic replicates per seed count, which says how often each estimator
  cries wolf;
- a **known effect** — the same generator with a difference of declared size —
  which says how large a real effect must be before this laboratory can see it.

The second number is the one every later mission needs. At five seeds, which is
what §10.1 requires, the smallest paired effect this laboratory detects with 80%
power is **1.68 standard deviations of the run-to-run difference**. Below that, a
null result means *underpowered*, not *no effect*, and saying otherwise would be
the quietest of the failures in §13.4. A half-standard-deviation difference —
the smallest this program would call interesting — needs **34 seeds**.

Three findings from the calibration are load-bearing enough to state here.

**The exact paired permutation test cannot reject at five seeds.** Sign-flipping
five differences gives 2^5 = 32 arrangements, so the smallest attainable
two-sided p-value is 2/32 = 0.0625. It is *structurally* above 0.05 — no data
can fix it. At three seeds the floor is 0.25. Every permutation result here
carries its ``p_value_floor`` so this cannot be discovered after the fact, and
six seeds is the smallest number at which the test can produce a significant
result at all.

**The percentile bootstrap is anti-conservative at these sizes.** At five seeds
its 95% interval excludes zero under a true null 16.7% of the time; BCa's does so
18.9% of the time, worse, because the bias correction and the acceleration are
themselves estimated from the same five points. Neither is usable as a *test*
here. The studentized bootstrap is (4.4%), and is the default interval; the
paired t-test holds its level (4.8%), and is the adopted primary test.

**Permutation over features is not protected by any of this.** With independent
features it holds its level (4.6% at thirty-six features); with features merely
equicorrelated at 0.3 — which is what superposition *is* — it rejects a true null
59% of the time, and worse as the bank grows, because the permutation null
assumes an independence the representation does not have. This is the same error
as pooling tokens, wearing different clothes.

The experimental-unit rule is therefore structural rather than advisory.
Every exported test takes a sequence of :class:`RunSummary` — one object per run,
carrying scalars — and :class:`ExperimentalUnitError` is raised on a bare array,
on a metric whose value is a vector, on two runs sharing a seed within a cell,
and on more runs in a cell than this laboratory could have produced. The one
function that accepts per-feature arrays says so in its name and returns a value
that has to be reduced to one number per run before it can enter anything else.

``--selftest`` re-runs both calibrations at reduced replicate counts and fails if
any estimator's false-positive rate has left its recorded tolerance — including
if an estimator recorded as unusable starts behaving, which means the record and
the evidence have come apart.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

import numpy as np

__all__ = [
    "ALPHA",
    "BOOTSTRAP_RESAMPLES",
    "CI_LEVEL",
    "COMPARISON_SCHEMA",
    "ESTIMATOR_SPECS",
    "ESTIMATOR_SPEC_BY_NAME",
    "MAX_RUNS_PER_CELL",
    "NULL_MODELS",
    "PRIMARY_TEST",
    "STATISTICS_VERSION",
    "SUMMARY_KEY",
    "CalibrationReport",
    "ComparisonDeclaration",
    "ComparisonRecord",
    "EffectSize",
    "EstimatorSpec",
    "ExperimentalUnitError",
    "FDRResult",
    "FeaturePermutationResult",
    "HierarchicalEffect",
    "NullCalibration",
    "PowerCalibration",
    "RunSummary",
    "StatisticsError",
    "TestResult",
    "attach_comparisons",
    "bootstrap_ci",
    "calibrate",
    "comparisons_from_summary",
    "fdr_control",
    "fdr_over_tests",
    "feature_permutation_test",
    "hierarchical_effect",
    "load_comparison",
    "minimum_detectable_effect",
    "null_calibration",
    "paired_effect",
    "paired_test",
    "power_calibration",
    "primary_comparison",
    "run_selftest",
    "run_summary_from_json",
    "secondary_comparison",
    "seeds_for_power",
    "standardized_effect",
    "unpaired_effect",
    "unpaired_test",
]

STATISTICS_VERSION = "stat-1.0.0"
"""Bump on any change to the *semantics* of an estimator or a threshold.
Recorded beside every comparison record, so a redefinition invalidates a
comparison instead of silently replacing it."""

ALPHA = 0.05
"""The nominal significance level. Not a threshold to be chosen per comparison —
a level whose *realised* false-positive rate has been measured for every
estimator in :data:`ESTIMATOR_SPECS` at three, five, and ten seeds."""

CI_LEVEL = 0.95

BOOTSTRAP_RESAMPLES = 10000
"""What a reported interval uses. The calibration uses 2000 (recorded in the
report), because coverage is insensitive to B above about a thousand and the
calibration runs a hundred thousand of them."""

DEFAULT_RNG_SEED = 20260810
"""Every resampling estimator here is deterministic by default. A bootstrap
interval that moves between two readings of the same data is not a measurement,
and "it was a different random seed" is indistinguishable from "the number was
wrong" to anyone reading the record."""

MAX_RUNS_PER_CELL = 256
"""More runs than this in one arm of one difficulty cell is not a seed set; it is
per-token or per-example values wearing run clothes. This laboratory has one GPU
and §10.1 asks for five seeds. Two hundred and fifty-six is far above anything a
mission here will legitimately produce and far below the four thousand tokens an
accidental pooling would deliver, so the guard is never in the way of real work
and always in the way of the mistake."""

EXACT_SIGN_FLIP_LIMIT = 1 << 20
"""Enumerate every sign assignment when 2^n does not exceed this. At twenty
seeds the enumeration is a million rows and takes milliseconds; above it the test
falls back to Monte Carlo and its ``p_value_floor`` changes accordingly."""

EXACT_LABEL_SHUFFLE_LIMIT = 100_000

RESAMPLING_UNITS: tuple[str, ...] = ("run", "seed", "feature", "cell")
"""What :func:`bootstrap_ci` will admit as a resampling unit. The argument is
mandatory: a caller who cannot name the unit they are resampling does not know
whether the interval means anything, and the commonest way to get this wrong is
to not have thought about it at all."""

_UNIT_LIMIT = {"run": MAX_RUNS_PER_CELL, "seed": MAX_RUNS_PER_CELL, "feature": 100_000, "cell": 1024}

PRIMARY_TEST = "paired_t"
"""The adopted primary test, chosen by the calibration and not by preference: it
is the only one of the five paired estimators whose false-positive rate is at its
nominal level at five seeds *and* which can produce a significant result there at
all. See ``ESTIMATOR_SPECS`` for the other four and why each was not chosen."""

SUMMARY_KEY = "comparisons"
COMPARISON_SCHEMA = "am.comparison.v1"

CELL_ALL = "all"
"""The cell name a comparison with no difficulty matrix uses. Named rather than
empty so that ``cell`` is always a real value and a one-cell comparison and a
multi-cell one have the same shape."""


class StatisticsError(ValueError):
    """An analysis that would not mean what it says."""


class ExperimentalUnitError(StatisticsError):
    """The wrong thing was offered as an experimental unit.

    §7.4: "treating hundreds of tokens as independent samples when the run is the
    true experimental unit". Every raise of this class is that mistake, caught at
    the boundary rather than reported as a confidence interval thirty times too
    narrow.
    """


# --------------------------------------------------------------------------- #
# Special functions
#
# No scipy in this laboratory's dependency set, and adding one for two
# distribution functions would be a poor trade. Both are checked against
# published values in tests/metrics/test_statistics_estimators.py.
# --------------------------------------------------------------------------- #


_erfc = np.frompyfunc(math.erfc, 1, 1)
"""``math.erfc`` lifted to arrays. numpy carries no error function and the only
places this is called on more than a handful of values are the bias correction
and the Halley step, so the object-array round trip costs nothing measurable."""


def _norm_cdf(x):
    x = np.asarray(x, dtype=float)
    out = 0.5 * np.asarray(_erfc(-x / math.sqrt(2.0)), dtype=float)
    return out if out.ndim else float(out)


_ACKLAM_A = (
    -3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
    1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00,
)
_ACKLAM_B = (
    -5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
    6.680131188771972e01, -1.328068155288572e01,
)
_ACKLAM_C = (
    -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
    -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00,
)
_ACKLAM_D = (
    7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
    3.754408661907416e00,
)


def _norm_ppf(p):
    """Inverse standard normal CDF: Acklam's rational approximation, refined.

    One Halley step against :func:`_norm_cdf` takes the relative error to the
    order of machine epsilon, which matters because BCa feeds its own output back
    through the CDF and a 1e-9 approximation there is visible in the interval.
    """
    p = np.asarray(p, dtype=float)
    out = np.empty_like(p)
    low, high = 0.02425, 1.0 - 0.02425

    lower = p < low
    upper = p > high
    middle = ~(lower | upper)

    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.sqrt(-2.0 * np.log(np.where(lower, p, 0.5)))
        num = ((((_ACKLAM_C[0] * q + _ACKLAM_C[1]) * q + _ACKLAM_C[2]) * q + _ACKLAM_C[3]) * q
               + _ACKLAM_C[4]) * q + _ACKLAM_C[5]
        den = (((_ACKLAM_D[0] * q + _ACKLAM_D[1]) * q + _ACKLAM_D[2]) * q + _ACKLAM_D[3]) * q + 1.0
        out = np.where(lower, num / den, out)

        q = np.sqrt(-2.0 * np.log(np.where(upper, 1.0 - p, 0.5)))
        num = ((((_ACKLAM_C[0] * q + _ACKLAM_C[1]) * q + _ACKLAM_C[2]) * q + _ACKLAM_C[3]) * q
               + _ACKLAM_C[4]) * q + _ACKLAM_C[5]
        den = (((_ACKLAM_D[0] * q + _ACKLAM_D[1]) * q + _ACKLAM_D[2]) * q + _ACKLAM_D[3]) * q + 1.0
        out = np.where(upper, -num / den, out)

        r = np.where(middle, p, 0.5) - 0.5
        s = r * r
        num = (((((_ACKLAM_A[0] * s + _ACKLAM_A[1]) * s + _ACKLAM_A[2]) * s + _ACKLAM_A[3]) * s
                + _ACKLAM_A[4]) * s + _ACKLAM_A[5]) * r
        den = ((((_ACKLAM_B[0] * s + _ACKLAM_B[1]) * s + _ACKLAM_B[2]) * s + _ACKLAM_B[3]) * s
               + _ACKLAM_B[4]) * s + 1.0
        out = np.where(middle, num / den, out)

        error = _norm_cdf(out) - p
        step = error * math.sqrt(2.0 * math.pi) * np.exp(out * out / 2.0)
        refined = out - step / (1.0 + out * step / 2.0)
        out = np.where(np.isfinite(refined), refined, out)

    out = np.where(p <= 0.0, -np.inf, out)
    out = np.where(p >= 1.0, np.inf, out)
    return out if out.ndim else float(out)


def _betacf(a: float, b: float, x: float, *, iterations: int = 400, eps: float = 3e-16) -> float:
    """Lentz's continued fraction for the incomplete beta. Numerical Recipes §6.4."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, iterations + 1):
        m2 = 2 * m
        numerator = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        numerator = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _incomplete_beta(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta ``I_x(a, b)``."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _student_t_two_sided_p(t: float, df: float) -> float:
    """Two-sided p-value of Student's t. ``I_{df/(df+t^2)}(df/2, 1/2)``."""
    if df <= 0:
        return float("nan")
    if not math.isfinite(t):
        return 0.0
    return _incomplete_beta(df / 2.0, 0.5, df / (df + t * t))


def _wilson_interval(successes: int, trials: int, *, level: float = CI_LEVEL) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Reported beside every false-positive rate, because a rate estimated from a
    finite number of replicates is itself an estimate, and "0.058 versus a
    nominal 0.05" means nothing without knowing whether the Monte Carlo error is
    0.005 or 0.05.
    """
    if trials <= 0:
        return (float("nan"), float("nan"))
    z = float(_norm_ppf(1.0 - (1.0 - level) / 2.0))
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def _as_rng(rng) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    if rng is None:
        return np.random.default_rng(DEFAULT_RNG_SEED)
    return np.random.default_rng(int(rng))


# --------------------------------------------------------------------------- #
# The experimental unit
# --------------------------------------------------------------------------- #


def _as_run_scalar(value, *, metric: str, run_id: str) -> float:
    """One number, or a refusal that names the mistake.

    A metric that arrives as an array is the token-pooling error at its source:
    somebody handed over ``per_token_accuracy`` where ``accuracy`` belonged, and
    every interval downstream would be narrower than the truth by the square root
    of the sequence length. It is caught here rather than at the interval,
    because here the error message can still say which run and which metric.
    """
    if value is None:
        return float("nan")
    if isinstance(value, bool):
        raise ExperimentalUnitError(
            f"run {run_id!r} metric {metric!r} is a boolean. A pass/fail verdict is not a "
            "metric; an effect size over booleans is a difference of proportions and needs "
            "to say so."
        )
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if hasattr(value, "detach"):  # torch tensor, without importing torch
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return float(value)
        raise ExperimentalUnitError(
            f"run {run_id!r} metric {metric!r} is an array of shape {tuple(value.shape)}, "
            f"not a run-level value. §7.4 forbids treating {value.size} per-token or "
            "per-example numbers as independent samples: the run is the experimental unit. "
            "Reduce it to one number per run first — and if the reduction is itself the "
            "quantity of interest, that is what belongs here."
        )
    if isinstance(value, (list, tuple)):
        raise ExperimentalUnitError(
            f"run {run_id!r} metric {metric!r} is a {type(value).__name__} of "
            f"{len(value)} values, not a run-level value. See §7.4: the run is the "
            "experimental unit, not the token."
        )
    raise ExperimentalUnitError(
        f"run {run_id!r} metric {metric!r} is a {type(value).__name__}, which is not a number"
    )


@dataclass(frozen=True)
class RunSummary:
    """One run, reduced to the scalars it contributes to an analysis.

    The experimental unit, made into a type. Every exported estimator takes a
    sequence of these and nothing else, so the only way to hand this module
    four thousand token-level numbers is to first claim that each one is a run —
    at which point ``seed`` collides, ``run_id`` collides, or the count exceeds
    :data:`MAX_RUNS_PER_CELL`, and the refusal names the rule that was broken.

    ``cell`` is the §7.3 R5 difficulty cell — a sparsity, a width, a sequence
    length. A run belongs to exactly one, and a seed appears at most once in each.
    """

    run_id: str
    seed: int
    arm: str
    metrics: Mapping[str, float]
    cell: str = CELL_ALL

    def __post_init__(self) -> None:
        for name, value in (("run_id", self.run_id), ("arm", self.arm), ("cell", self.cell)):
            if not isinstance(value, str) or not value.strip():
                raise StatisticsError(f"RunSummary {name} must be a non-empty string, got {value!r}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)):
            raise StatisticsError(
                f"RunSummary seed must be an integer, got {type(self.seed).__name__}. The seed "
                "identifies the replicate; a run without one cannot be matched to its control."
            )
        object.__setattr__(self, "seed", int(self.seed))
        if not isinstance(self.metrics, Mapping):
            raise StatisticsError(
                f"RunSummary metrics must be a mapping of name to scalar, got "
                f"{type(self.metrics).__name__}"
            )
        if not self.metrics:
            raise StatisticsError(f"run {self.run_id!r} carries no metrics")
        reduced = {
            str(name): _as_run_scalar(value, metric=str(name), run_id=self.run_id)
            for name, value in self.metrics.items()
        }
        object.__setattr__(self, "metrics", MappingProxyType(reduced))

    @property
    def key(self) -> tuple[int, str]:
        return (self.seed, self.cell)

    def value(self, metric: str) -> float:
        if metric not in self.metrics:
            raise StatisticsError(
                f"run {self.run_id!r} has no metric {metric!r}; it has "
                f"{sorted(self.metrics)}"
            )
        return float(self.metrics[metric])

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "seed": self.seed,
            "arm": self.arm,
            "cell": self.cell,
            "metrics": dict(self.metrics),
        }


def run_summary_from_json(
    path: Path | str, *, arm: str, cell: str = CELL_ALL, metrics: Sequence[str] | None = None
) -> RunSummary:
    """Read one run's ``summary.json`` into a :class:`RunSummary`.

    The easy path and the correct path are the same path. ``summary.json``'s
    ``final`` block already holds exactly one scalar per §6.1 metric, because the
    runner reduced them when it wrote the run; a caller who starts here cannot
    accidentally pool anything, because there is nothing left to pool.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "summary.json"
    payload = json.loads(path.read_text())
    final = payload.get("final") or {}
    if not final:
        raise StatisticsError(f"{path} has no 'final' block; it recorded no evaluation")
    selected = dict(final) if metrics is None else {
        name: final.get(name) for name in metrics
    }
    missing = [name for name, value in selected.items() if name not in final]
    if missing:
        raise StatisticsError(f"{path} has no metric(s) {missing}; it has {sorted(final)}")
    seed = (payload.get("config") or {}).get("seed")
    if seed is None:
        raise StatisticsError(f"{path} records no seed")
    return RunSummary(
        run_id=str(payload.get("run_id") or path.parent.name),
        seed=int(seed),
        arm=arm,
        cell=cell,
        metrics=selected,
    )


def _require_run_summaries(values, *, what: str) -> tuple[RunSummary, ...]:
    """Refuse anything that is not a list of runs, in the words of the mistake."""
    if isinstance(values, RunSummary):
        raise ExperimentalUnitError(
            f"{what} is a single RunSummary; an effect needs a set of runs, one per seed"
        )
    if isinstance(values, (str, bytes, Mapping)):
        raise ExperimentalUnitError(f"{what} must be a sequence of RunSummary, got {type(values).__name__}")
    if isinstance(values, np.ndarray) or hasattr(values, "detach"):
        # ndarray.size is an int; torch.Tensor.size is a method and numel() is the int.
        size = values.size if isinstance(values, np.ndarray) else int(values.numel())
        raise ExperimentalUnitError(
            f"{what} is an array of {size} numbers. §7.4: the run is the experimental unit. "
            "Pass one RunSummary per run — if these are per-token or per-example values, "
            "reduce them to a run-level number first; if they are already per-run, say which "
            "run and which seed each belongs to."
        )
    try:
        runs = tuple(values)
    except TypeError as error:
        raise ExperimentalUnitError(f"{what} is not iterable: {type(values).__name__}") from error

    if not runs:
        raise ExperimentalUnitError(f"{what} is empty")
    offenders = [index for index, run in enumerate(runs) if not isinstance(run, RunSummary)]
    if offenders:
        kinds = sorted({type(runs[index]).__name__ for index in offenders})
        raise ExperimentalUnitError(
            f"{what} contains {len(offenders)} entries that are not RunSummary ({kinds}). "
            f"A bare sequence of {len(runs)} numbers cannot be an experimental unit: this "
            "module cannot tell a list of per-run scores from a list of per-token scores, so "
            "it refuses both. Wrap each run in a RunSummary carrying its run_id and seed."
        )

    seen_ids: set[str] = set()
    per_cell: dict[tuple[str, str], list[int]] = {}
    for run in runs:
        if run.run_id in seen_ids:
            raise ExperimentalUnitError(
                f"{what} contains run_id {run.run_id!r} twice. Two records of one run are one "
                "run: re-measuring a run does not replicate it."
            )
        seen_ids.add(run.run_id)
        per_cell.setdefault((run.arm, run.cell), []).append(run.seed)

    for (arm, cell), seeds in per_cell.items():
        if len(seeds) > _UNIT_LIMIT["run"]:
            raise ExperimentalUnitError(
                f"{what} has {len(seeds)} runs in arm {arm!r} cell {cell!r}, above the "
                f"{MAX_RUNS_PER_CELL} this laboratory can produce. That is the shape of "
                "per-token or per-example values pooled into a seed set (§7.4), not a seed set."
            )
        duplicates = sorted({seed for seed in seeds if seeds.count(seed) > 1})
        if duplicates:
            raise ExperimentalUnitError(
                f"{what} has seed(s) {duplicates} more than once in arm {arm!r} cell {cell!r}. "
                "Repeated measurements of one seed are not independent replicates; if these are "
                "different difficulty cells, give them different cell names."
            )
    return runs


@dataclass(frozen=True)
class PairedSample:
    """Control and candidate matched on ``(seed, cell)``.

    §7.2 freezes the seed set, so a matched comparison is the normal case and an
    unmatched one is a fact worth refusing over: a candidate with a seed the
    control does not have is a candidate that was run more.
    """

    metric: str
    keys: tuple[tuple[int, str], ...]
    control: np.ndarray
    candidate: np.ndarray
    control_runs: tuple[RunSummary, ...]
    candidate_runs: tuple[RunSummary, ...]

    @property
    def differences(self) -> np.ndarray:
        return self.candidate - self.control

    @property
    def n(self) -> int:
        return len(self.keys)

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(sorted({seed for seed, _ in self.keys}))

    @property
    def cells(self) -> tuple[str, ...]:
        return tuple(sorted({cell for _, cell in self.keys}))


def paired_sample(control, candidate, metric: str) -> PairedSample:
    """Match two arms on ``(seed, cell)`` and read one metric out of each."""
    control_runs = _require_run_summaries(control, what="control")
    candidate_runs = _require_run_summaries(candidate, what="candidate")

    control_by_key = {run.key: run for run in control_runs}
    candidate_by_key = {run.key: run for run in candidate_runs}
    if len(control_by_key) != len(control_runs) or len(candidate_by_key) != len(candidate_runs):
        raise ExperimentalUnitError(
            "an arm has two runs at the same (seed, cell); see the duplicate-seed rule"
        )

    only_control = sorted(set(control_by_key) - set(candidate_by_key))
    only_candidate = sorted(set(candidate_by_key) - set(control_by_key))
    if only_control or only_candidate:
        raise StatisticsError(
            "the arms are not matched: "
            f"{len(only_control)} (seed, cell) in control only {only_control[:4]}, "
            f"{len(only_candidate)} in candidate only {only_candidate[:4]}. §7.2 freezes the "
            "seed set; an arm with cells the other does not have is an arm that was run more, "
            "and comparing their means would credit the difference to the architecture."
        )

    keys = tuple(sorted(control_by_key))
    control_values = np.array([control_by_key[key].value(metric) for key in keys], dtype=float)
    candidate_values = np.array([candidate_by_key[key].value(metric) for key in keys], dtype=float)
    if not np.isfinite(control_values).all() or not np.isfinite(candidate_values).all():
        bad = [
            key for key, ok in zip(
                keys, np.isfinite(control_values) & np.isfinite(candidate_values), strict=True
            ) if not ok
        ]
        raise StatisticsError(
            f"metric {metric!r} is undefined (NaN or null) for (seed, cell) {bad[:4]}. A metric "
            "that does not apply to a condition cannot be the comparison's metric there."
        )
    return PairedSample(
        metric=metric,
        keys=keys,
        control=control_values,
        candidate=candidate_values,
        control_runs=tuple(control_by_key[key] for key in keys),
        candidate_runs=tuple(candidate_by_key[key] for key in keys),
    )


def _arm_values(runs: Sequence[RunSummary], metric: str) -> np.ndarray:
    values = np.array([run.value(metric) for run in runs], dtype=float)
    if not np.isfinite(values).all():
        raise StatisticsError(f"metric {metric!r} is undefined for at least one run")
    return values


# --------------------------------------------------------------------------- #
# Effect sizes and bootstrap intervals
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EffectSize:
    """A point estimate with an interval, and the machinery that produced it."""

    name: str
    estimate: float
    ci_low: float
    ci_high: float
    ci_method: str
    ci_level: float
    unit: str
    n: int
    resamples: int
    detail: dict = field(default_factory=dict)

    @property
    def excludes_zero(self) -> bool:
        if math.isnan(self.ci_low) or math.isnan(self.ci_high):
            return False
        return bool(self.ci_low > 0.0 or self.ci_high < 0.0)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "estimate": self.estimate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "ci_method": self.ci_method,
            "ci_level": self.ci_level,
            "unit": self.unit,
            "n": self.n,
            "resamples": self.resamples,
            "excludes_zero": self.excludes_zero,
            "detail": dict(self.detail),
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> EffectSize:
        return cls(
            name=payload["name"],
            estimate=payload["estimate"],
            ci_low=payload["ci_low"],
            ci_high=payload["ci_high"],
            ci_method=payload["ci_method"],
            ci_level=payload["ci_level"],
            unit=payload["unit"],
            n=payload["n"],
            resamples=payload["resamples"],
            detail=dict(payload.get("detail") or {}),
        )


def _bootstrap_means(values: np.ndarray, resamples: int, rng: np.random.Generator):
    n = values.size
    indices = rng.integers(0, n, size=(resamples, n))
    drawn = values[indices]
    return drawn.mean(axis=1), drawn.std(axis=1, ddof=1)


def bootstrap_ci(
    values,
    *,
    unit: str,
    method: str = "bca",
    level: float = CI_LEVEL,
    resamples: int = BOOTSTRAP_RESAMPLES,
    rng=None,
) -> tuple[float, float, dict]:
    """Bootstrap confidence interval for a mean, over an explicitly named unit.

    ``unit`` is mandatory and has no default. A bootstrap interval is a statement
    about resampling *something*, and the single commonest way to produce a
    confident wrong interval is to resample the wrong thing without noticing.
    Requiring the word costs one keyword argument and makes the mistake visible
    in the call site rather than in the result.

    Three methods, all calibrated in :func:`null_calibration`:

    ``percentile``   the empirical quantiles of the resampled mean. Simple, and
                     anti-conservative at these sample sizes — its 95% interval
                     misses zero 16.7% of the time at five runs.
    ``bca``          bias-corrected and accelerated. Better asymptotics, no
                     better here (18.9% at five runs): the acceleration is itself
                     estimated from five points.
    ``studentized``  bootstrap-t. Honestly wide at small n — at three runs one
                     resample in nine draws a single value three times, giving a
                     zero standard error and an infinite interval, which is the
                     correct report and not a bug.
    """
    if unit not in RESAMPLING_UNITS:
        raise StatisticsError(
            f"bootstrap_ci needs unit= one of {list(RESAMPLING_UNITS)}, got {unit!r}. "
            "Naming the resampling unit is not paperwork: it is the difference between an "
            "interval over runs and an interval over tokens."
        )
    values = np.asarray(values, dtype=float).ravel()
    if values.size < 2:
        raise StatisticsError(f"a bootstrap over {values.size} {unit}(s) is not an interval")
    if values.size > _UNIT_LIMIT[unit]:
        raise ExperimentalUnitError(
            f"bootstrap over {values.size} values declared as unit={unit!r}, above the "
            f"{_UNIT_LIMIT[unit]} plausible here. §7.4: hundreds of tokens are not hundreds of "
            "independent samples."
        )
    if not np.isfinite(values).all():
        raise StatisticsError("bootstrap input contains non-finite values")

    generator = _as_rng(rng)
    n = values.size
    estimate = float(values.mean())
    means, sds = _bootstrap_means(values, resamples, generator)
    tail = (1.0 - level) / 2.0
    detail: dict = {"method": method, "resamples": int(resamples), "n": int(n)}

    if method == "percentile":
        low, high = np.quantile(means, [tail, 1.0 - tail])
    elif method == "bca":
        proportion = float(np.mean(means < estimate))
        proportion = min(max(proportion, 1.0 / (2 * resamples)), 1.0 - 1.0 / (2 * resamples))
        z0 = float(_norm_ppf(proportion))
        total = values.sum()
        jackknife = (total - values) / (n - 1)
        centred = jackknife.mean() - jackknife
        denominator = 6.0 * float(np.sum(centred**2)) ** 1.5
        acceleration = 0.0 if denominator == 0.0 else float(np.sum(centred**3)) / denominator
        z_low, z_high = float(_norm_ppf(tail)), float(_norm_ppf(1.0 - tail))
        adjusted = []
        for z in (z_low, z_high):
            shifted = z0 + z
            adjusted.append(float(_norm_cdf(z0 + shifted / (1.0 - acceleration * shifted))))
        low, high = np.quantile(means, [min(adjusted), max(adjusted)])
        detail |= {"bias_correction_z0": z0, "acceleration": acceleration}
    elif method == "studentized":
        standard_error = float(values.std(ddof=1)) / math.sqrt(n)
        resample_errors = sds / math.sqrt(n)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_star = np.where(
                resample_errors > 0.0,
                (means - estimate) / np.where(resample_errors > 0.0, resample_errors, 1.0),
                np.where(means == estimate, 0.0, np.sign(means - estimate) * np.inf),
            )
        # Order statistics rather than interpolated quantiles: at three runs one
        # resample in nine draws a single value three times, its studentized
        # statistic is infinite, and interpolating between two infinities yields
        # NaN. The honest report is an infinite bound — the bootstrap-t genuinely
        # cannot bound the mean from three points — and a NaN would read as a
        # defect in the estimator rather than as the width of the ignorance.
        t_low = float(np.quantile(t_star, tail, method="lower"))
        t_high = float(np.quantile(t_star, 1.0 - tail, method="higher"))
        low, high = estimate - t_high * standard_error, estimate - t_low * standard_error
        detail |= {
            "standard_error": standard_error,
            "degenerate_resamples": int(np.sum(resample_errors == 0.0)),
        }
    else:
        raise StatisticsError(
            f"unknown bootstrap method {method!r}; expected percentile, bca, or studentized"
        )
    return float(low), float(high), detail


def paired_effect(
    control,
    candidate,
    metric: str,
    *,
    ci_method: str = "studentized",
    ci_level: float = CI_LEVEL,
    resamples: int = BOOTSTRAP_RESAMPLES,
    rng=None,
) -> EffectSize:
    """Mean paired difference (candidate minus control) with a bootstrap interval.

    In the metric's own units, which is what §7.4 means by an effect size: "0.031
    exact recall" is a statement a reader can weigh, and 1.4 standard deviations
    is one they cannot without also knowing the standard deviation.
    :func:`standardized_effect` reports the second beside it.

    The default interval is the studentized one *because of the calibration, not
    despite it*. At five runs the percentile bootstrap's 95% interval misses zero
    under a true null 16.7% of the time and BCa's 18.9%; the studentized
    interval's rate is 4.4%, which is what a 95% interval is supposed to mean.
    The other two remain available and are worth reporting beside it — they are
    narrower and better behaved as *descriptions* — but an interval used to
    decide anything should be the one whose coverage was measured.
    """
    sample = paired_sample(control, candidate, metric)
    differences = sample.differences
    low, high, detail = bootstrap_ci(
        differences, unit="run", method=ci_method, level=ci_level, resamples=resamples, rng=rng
    )
    return EffectSize(
        name="paired_mean_difference",
        estimate=float(differences.mean()),
        ci_low=low,
        ci_high=high,
        ci_method=ci_method,
        ci_level=ci_level,
        unit=metric,
        n=sample.n,
        resamples=resamples,
        detail=detail | {
            "metric": metric,
            "control_mean": float(sample.control.mean()),
            "candidate_mean": float(sample.candidate.mean()),
            "difference_sd": float(differences.std(ddof=1)) if sample.n > 1 else float("nan"),
            "n_seeds": len(sample.seeds),
            "n_cells": len(sample.cells),
        },
    )


def standardized_effect(
    control,
    candidate,
    metric: str,
    *,
    ci_level: float = CI_LEVEL,
    resamples: int = BOOTSTRAP_RESAMPLES,
    rng=None,
) -> EffectSize:
    """Cohen's ``dz`` for the paired difference, bias-corrected, with a bootstrap CI.

    The unit the power calibration speaks in. ``dz = 1`` means the mean difference
    equals the run-to-run standard deviation of that difference, which is the
    scale on which "detectable at five seeds" is a statement about this
    laboratory rather than about one metric.
    """
    sample = paired_sample(control, candidate, metric)
    differences = sample.differences
    if sample.n < 2:
        raise StatisticsError("a standardized effect needs at least two runs")
    correction = _hedges_correction(sample.n - 1)
    estimate = _cohens_dz(differences) * correction

    generator = _as_rng(rng)
    indices = generator.integers(0, sample.n, size=(resamples, sample.n))
    drawn = differences[indices]
    sds = drawn.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        distribution = np.where(sds > 0.0, drawn.mean(axis=1) / np.where(sds > 0, sds, 1.0), 0.0)
    distribution = distribution * correction
    tail = (1.0 - ci_level) / 2.0
    low, high = np.quantile(distribution, [tail, 1.0 - tail])
    return EffectSize(
        name="cohens_dz",
        estimate=float(estimate),
        ci_low=float(low),
        ci_high=float(high),
        ci_method="percentile",
        ci_level=ci_level,
        unit="sd_of_paired_difference",
        n=sample.n,
        resamples=resamples,
        detail={
            "metric": metric,
            "hedges_correction": correction,
            "difference_sd": float(differences.std(ddof=1)),
        },
    )


def _cohens_dz(differences: np.ndarray) -> float:
    sd = float(differences.std(ddof=1))
    if sd == 0.0:
        return 0.0 if float(differences.mean()) == 0.0 else math.copysign(math.inf, differences.mean())
    return float(differences.mean()) / sd


def _hedges_correction(df: float) -> float:
    """Hedges' small-sample correction ``J``. At four degrees of freedom it is
    0.80, so an uncorrected ``dz`` at five seeds overstates the effect by a
    quarter — which is exactly the size of thing this program is looking for."""
    if df <= 1:
        return 1.0
    return math.exp(math.lgamma(df / 2.0) - math.lgamma((df - 1.0) / 2.0)) * math.sqrt(2.0 / df)


def unpaired_effect(
    control,
    candidate,
    metric: str,
    *,
    ci_method: str = "studentized",
    ci_level: float = CI_LEVEL,
    resamples: int = BOOTSTRAP_RESAMPLES,
    rng=None,
) -> EffectSize:
    """Mean difference between two unmatched arms, with Hedges' ``g`` beside it.

    For the case §7.2's frozen seed set does not cover — a candidate whose seeds
    do not mean the same thing as the control's. It costs power; the record says
    so in ``detail["seeds_matched"]`` so that a later reader can see the choice.
    """
    control_runs = _require_run_summaries(control, what="control")
    candidate_runs = _require_run_summaries(candidate, what="candidate")
    a = _arm_values(control_runs, metric)
    b = _arm_values(candidate_runs, metric)
    if a.size < 2 or b.size < 2:
        raise StatisticsError("an unpaired effect needs at least two runs per arm")

    generator = _as_rng(rng)
    estimate = float(b.mean() - a.mean())
    a_drawn = a[generator.integers(0, a.size, size=(resamples, a.size))]
    b_drawn = b[generator.integers(0, b.size, size=(resamples, b.size))]
    resampled = b_drawn.mean(axis=1) - a_drawn.mean(axis=1)
    tail = (1.0 - ci_level) / 2.0
    if ci_method == "percentile":
        low, high = np.quantile(resampled, [tail, 1.0 - tail])
    elif ci_method == "bca":
        proportion = float(np.mean(resampled < estimate))
        proportion = min(max(proportion, 1.0 / (2 * resamples)), 1.0 - 1.0 / (2 * resamples))
        z0 = float(_norm_ppf(proportion))
        z_low, z_high = float(_norm_ppf(tail)), float(_norm_ppf(1.0 - tail))
        adjusted = sorted(float(_norm_cdf(z0 + (z0 + z))) for z in (z_low, z_high))
        low, high = np.quantile(resampled, adjusted)
    elif ci_method == "studentized":
        standard_error = math.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
        resample_errors = np.sqrt(
            a_drawn.var(axis=1, ddof=1) / a.size + b_drawn.var(axis=1, ddof=1) / b.size
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            t_star = np.where(
                resample_errors > 0.0,
                (resampled - estimate) / np.where(resample_errors > 0.0, resample_errors, 1.0),
                np.where(resampled == estimate, 0.0, np.sign(resampled - estimate) * np.inf),
            )
        t_low = float(np.quantile(t_star, tail, method="lower"))
        t_high = float(np.quantile(t_star, 1.0 - tail, method="higher"))
        low, high = estimate - t_high * standard_error, estimate - t_low * standard_error
    else:
        raise StatisticsError(
            f"unpaired_effect supports percentile, bca, and studentized, not {ci_method!r}"
        )

    pooled = math.sqrt(
        ((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / (a.size + b.size - 2)
    )
    correction = _hedges_correction(a.size + b.size - 2)
    hedges = 0.0 if pooled == 0.0 else estimate / pooled * correction
    return EffectSize(
        name="unpaired_mean_difference",
        estimate=estimate,
        ci_low=float(low),
        ci_high=float(high),
        ci_method=ci_method,
        ci_level=ci_level,
        unit=metric,
        n=int(a.size + b.size),
        resamples=resamples,
        detail={
            "metric": metric,
            "n_control": int(a.size),
            "n_candidate": int(b.size),
            "control_mean": float(a.mean()),
            "candidate_mean": float(b.mean()),
            "hedges_g": float(hedges),
            "seeds_matched": sorted({r.seed for r in control_runs})
            == sorted({r.seed for r in candidate_runs}),
        },
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TestResult:
    """One test's verdict, carrying the floor of what it could possibly report.

    ``p_value_floor`` is the field that stops a permutation result from being
    misread. Five paired differences admit thirty-two sign assignments, so the
    smallest two-sided p-value obtainable from them is 0.0625 — above 0.05
    regardless of the data. A test whose floor exceeds its own alpha has zero
    power by construction, ``power_is_attainable`` is False, and a "not
    significant" from it means nothing at all.
    """

    test: str
    statistic: float
    p_value: float
    p_value_floor: float
    alpha: float
    n: int
    detail: dict = field(default_factory=dict)

    @property
    def significant(self) -> bool:
        return bool(self.p_value <= self.alpha)

    @property
    def power_is_attainable(self) -> bool:
        return bool(self.p_value_floor <= self.alpha)

    def as_dict(self) -> dict:
        return {
            "test": self.test,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "p_value_floor": self.p_value_floor,
            "alpha": self.alpha,
            "n": self.n,
            "significant": self.significant,
            "power_is_attainable": self.power_is_attainable,
            "detail": dict(self.detail),
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> TestResult:
        return cls(
            test=payload["test"],
            statistic=payload["statistic"],
            p_value=payload["p_value"],
            p_value_floor=payload["p_value_floor"],
            alpha=payload["alpha"],
            n=payload["n"],
            detail=dict(payload.get("detail") or {}),
        )


@lru_cache(maxsize=32)
def _sign_matrix(n: int) -> np.ndarray:
    """All 2^n sign assignments, as ``(2^n, n)`` of ±1."""
    bits = np.arange(1 << n, dtype=np.int64)[:, None] >> np.arange(n, dtype=np.int64)[None, :]
    return 1.0 - 2.0 * (bits & 1).astype(float)


def _sign_flip_test(differences: np.ndarray, *, alpha: float, resamples: int, rng) -> TestResult:
    n = differences.size
    observed = float(differences.sum())
    if (1 << n) <= EXACT_SIGN_FLIP_LIMIT:
        statistics = _sign_matrix(n) @ differences
        exhaustive = int(1 << n)
        p_value = float(np.mean(np.abs(statistics) >= abs(observed) - 1e-12))
        floor = 2.0 / exhaustive
        detail = {"exact": True, "arrangements": exhaustive}
    else:
        generator = _as_rng(rng)
        signs = generator.integers(0, 2, size=(resamples, n)) * 2.0 - 1.0
        statistics = signs @ differences
        extreme = int(np.sum(np.abs(statistics) >= abs(observed) - 1e-12))
        p_value = (1.0 + extreme) / (resamples + 1.0)
        floor = 1.0 / (resamples + 1.0)
        detail = {"exact": False, "arrangements": int(resamples)}
    return TestResult(
        test="paired_permutation",
        statistic=float(differences.mean()),
        p_value=p_value,
        p_value_floor=floor,
        alpha=alpha,
        n=n,
        detail=detail | {"sum_of_differences": observed},
    )


def _paired_t_test(differences: np.ndarray, *, alpha: float) -> TestResult:
    n = differences.size
    if n < 2:
        raise StatisticsError("a paired t-test needs at least two runs")
    sd = float(differences.std(ddof=1))
    mean = float(differences.mean())
    if sd == 0.0:
        t = 0.0 if mean == 0.0 else math.copysign(math.inf, mean)
    else:
        t = mean / (sd / math.sqrt(n))
    return TestResult(
        test="paired_t",
        statistic=float(t),
        p_value=_student_t_two_sided_p(t, n - 1),
        p_value_floor=0.0,
        alpha=alpha,
        n=n,
        detail={"df": n - 1, "mean_difference": mean, "sd_of_difference": sd},
    )


def paired_test(
    control,
    candidate,
    metric: str,
    *,
    test: str = PRIMARY_TEST,
    alpha: float = ALPHA,
    resamples: int = BOOTSTRAP_RESAMPLES,
    rng=None,
) -> TestResult:
    """Test the paired difference between two arms matched on ``(seed, cell)``.

    ``paired_t`` is the default because it is the only paired estimator whose
    false-positive rate is at its nominal level at five seeds and which can
    reject there at all; ``paired_permutation`` is exact and assumption-free and
    cannot reject below six seeds, which its ``p_value_floor`` records rather
    than hides.
    """
    sample = paired_sample(control, candidate, metric)
    differences = sample.differences
    if test == "paired_t":
        result = _paired_t_test(differences, alpha=alpha)
    elif test == "paired_permutation":
        result = _sign_flip_test(differences, alpha=alpha, resamples=resamples, rng=rng)
    else:
        raise StatisticsError(f"unknown paired test {test!r}; expected paired_t or paired_permutation")
    return TestResult(
        test=result.test,
        statistic=result.statistic,
        p_value=result.p_value,
        p_value_floor=result.p_value_floor,
        alpha=result.alpha,
        n=result.n,
        detail=result.detail | {
            "metric": metric,
            "n_seeds": len(sample.seeds),
            "n_cells": len(sample.cells),
        },
    )


def _welch_t_test(a: np.ndarray, b: np.ndarray, *, alpha: float) -> TestResult:
    va, vb = float(a.var(ddof=1)), float(b.var(ddof=1))
    se_squared = va / a.size + vb / b.size
    difference = float(b.mean() - a.mean())
    if se_squared == 0.0:
        t, df = (0.0 if difference == 0.0 else math.copysign(math.inf, difference)), a.size + b.size - 2
    else:
        t = difference / math.sqrt(se_squared)
        df = se_squared**2 / (
            (va / a.size) ** 2 / (a.size - 1) + (vb / b.size) ** 2 / (b.size - 1)
        )
    return TestResult(
        test="welch_t",
        statistic=float(t),
        p_value=_student_t_two_sided_p(t, df),
        p_value_floor=0.0,
        alpha=alpha,
        n=int(a.size + b.size),
        detail={"df": float(df), "mean_difference": difference},
    )


@lru_cache(maxsize=32)
def _combination_index(total: int, size: int) -> np.ndarray:
    """Every way of choosing ``size`` of ``total`` positions, as an index matrix."""
    return np.array(list(itertools.combinations(range(total), size)), dtype=np.int64)


def _label_shuffle_test(a: np.ndarray, b: np.ndarray, *, alpha: float, resamples: int, rng) -> TestResult:
    pooled = np.concatenate([a, b])
    total = pooled.size
    observed = float(b.mean() - a.mean())
    arrangements = math.comb(total, a.size)
    if arrangements <= EXACT_LABEL_SHUFFLE_LIMIT:
        chosen = _combination_index(total, a.size)
        left = pooled[chosen].mean(axis=1)
        right = (pooled.sum() - pooled[chosen].sum(axis=1)) / b.size
        statistics = right - left
        p_value = float(np.mean(np.abs(statistics) >= abs(observed) - 1e-12))
        floor = (2.0 if a.size == b.size else 1.0) / arrangements
        detail = {"exact": True, "arrangements": int(arrangements)}
    else:
        generator = _as_rng(rng)
        shuffled = np.argsort(generator.random((resamples, total)), axis=1)
        drawn = pooled[shuffled]
        statistics = drawn[:, a.size:].mean(axis=1) - drawn[:, : a.size].mean(axis=1)
        extreme = int(np.sum(np.abs(statistics) >= abs(observed) - 1e-12))
        p_value = (1.0 + extreme) / (resamples + 1.0)
        floor = 1.0 / (resamples + 1.0)
        detail = {"exact": False, "arrangements": int(resamples)}
    return TestResult(
        test="unpaired_permutation",
        statistic=observed,
        p_value=p_value,
        p_value_floor=floor,
        alpha=alpha,
        n=int(total),
        detail=detail,
    )


def unpaired_test(
    control,
    candidate,
    metric: str,
    *,
    test: str = "welch_t",
    alpha: float = ALPHA,
    resamples: int = BOOTSTRAP_RESAMPLES,
    rng=None,
) -> TestResult:
    """Test two unmatched arms. Welch by default; label-shuffle permutation exact."""
    control_runs = _require_run_summaries(control, what="control")
    candidate_runs = _require_run_summaries(candidate, what="candidate")
    a = _arm_values(control_runs, metric)
    b = _arm_values(candidate_runs, metric)
    if a.size < 2 or b.size < 2:
        raise StatisticsError("an unpaired test needs at least two runs per arm")
    if test == "welch_t":
        result = _welch_t_test(a, b, alpha=alpha)
    elif test == "unpaired_permutation":
        result = _label_shuffle_test(a, b, alpha=alpha, resamples=resamples, rng=rng)
    else:
        raise StatisticsError(f"unknown unpaired test {test!r}")
    return TestResult(
        test=result.test,
        statistic=result.statistic,
        p_value=result.p_value,
        p_value_floor=result.p_value_floor,
        alpha=result.alpha,
        n=result.n,
        detail=result.detail | {"metric": metric, "n_control": int(a.size), "n_candidate": int(b.size)},
    )


# --------------------------------------------------------------------------- #
# Hierarchical analysis: seeds above difficulty cells
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HierarchicalEffect:
    """A comparison across a seed × difficulty-cell matrix, reduced to the seed.

    §7.4 asks for "hierarchical analysis across seeds and difficulty cells" and,
    in the same breath, that the run be the experimental unit. Those pull in
    opposite directions if a cell is allowed to be a sample: five seeds over four
    cells is twenty numbers, and treating them as twenty halves the interval
    while the seed effect they share is counted four times.

    So the reduction is fixed and one-directional. Each seed's cells are averaged
    into one number, the primary analysis runs over the *seeds*, and the
    interval comes from resampling seeds — not cells, not runs. Per-cell effects
    exist in ``per_cell`` and are marked exploratory; §7.4's last prohibition is
    "upgrading a claim because one exploratory cell passed", so they arrive with
    a Benjamini–Hochberg correction already applied and no accessor that returns
    a cell's uncorrected verdict alone.
    """

    metric: str
    n_seeds: int
    n_cells: int
    per_seed: tuple[dict, ...]
    per_cell: tuple[dict, ...]
    effect: EffectSize
    test: TestResult
    cluster_bootstrap_ci: tuple[float, float]
    variance_components: dict
    cell_fdr: FDRResult

    def as_dict(self) -> dict:
        return {
            "metric": self.metric,
            "n_seeds": self.n_seeds,
            "n_cells": self.n_cells,
            "per_seed": [dict(row) for row in self.per_seed],
            "per_cell_exploratory": [dict(row) for row in self.per_cell],
            "effect": self.effect.as_dict(),
            "test": self.test.as_dict(),
            "cluster_bootstrap_ci": list(self.cluster_bootstrap_ci),
            "variance_components": dict(self.variance_components),
            "cell_fdr": self.cell_fdr.as_dict(),
        }


def hierarchical_effect(
    control,
    candidate,
    metric: str,
    *,
    alpha: float = ALPHA,
    test: str = PRIMARY_TEST,
    ci_level: float = CI_LEVEL,
    resamples: int = BOOTSTRAP_RESAMPLES,
    rng=None,
) -> HierarchicalEffect:
    """Seed-level effect over a difficulty matrix, with cells kept exploratory."""
    sample = paired_sample(control, candidate, metric)
    seeds, cells = sample.seeds, sample.cells
    if len(seeds) < 2:
        raise StatisticsError("a hierarchical effect needs at least two seeds")

    index_of_seed = {seed: index for index, seed in enumerate(seeds)}
    index_of_cell = {cell: index for index, cell in enumerate(cells)}
    matrix = np.full((len(seeds), len(cells)), np.nan)
    for (seed, cell), difference in zip(sample.keys, sample.differences, strict=True):
        matrix[index_of_seed[seed], index_of_cell[cell]] = difference
    if np.isnan(matrix).any():
        missing = [
            (seeds[i], cells[j])
            for i, j in zip(*np.nonzero(np.isnan(matrix)), strict=True)
        ]
        raise StatisticsError(
            f"the seed × cell matrix has holes at {missing[:4]}; an unbalanced matrix would "
            "weight the seeds that ran everywhere more heavily than the ones that did not"
        )

    per_seed_difference = matrix.mean(axis=1)
    generator = _as_rng(rng)
    low, high, bootstrap_detail = bootstrap_ci(
        per_seed_difference, unit="seed", method="studentized", level=ci_level,
        resamples=resamples, rng=generator,
    )
    effect = EffectSize(
        name="hierarchical_mean_difference",
        estimate=float(per_seed_difference.mean()),
        ci_low=low,
        ci_high=high,
        ci_method="studentized",
        ci_level=ci_level,
        unit=metric,
        n=len(seeds),
        resamples=resamples,
        detail=bootstrap_detail | {"metric": metric, "n_cells": len(cells), "n_runs": sample.n},
    )
    cluster_low, cluster_high, _ = bootstrap_ci(
        per_seed_difference, unit="seed", method="percentile", level=ci_level,
        resamples=resamples, rng=generator,
    )
    if test == "paired_t":
        verdict = _paired_t_test(per_seed_difference, alpha=alpha)
    elif test == "paired_permutation":
        verdict = _sign_flip_test(per_seed_difference, alpha=alpha, resamples=resamples, rng=generator)
    else:
        raise StatisticsError(f"unknown hierarchical test {test!r}")

    cell_tests = [_paired_t_test(matrix[:, index], alpha=alpha) for index in range(len(cells))]
    cell_fdr = fdr_control(
        [result.p_value for result in cell_tests], alpha=alpha, method="bh", labels=cells
    )
    per_cell = tuple(
        {
            "cell": cell,
            "mean_difference": float(matrix[:, index].mean()),
            "sd": float(matrix[:, index].std(ddof=1)),
            "p_value": cell_tests[index].p_value,
            "q_value": cell_fdr.q_values[index],
            "rejected_after_fdr": cell_fdr.rejected[index],
            "exploratory": True,
        }
        for index, cell in enumerate(cells)
    )

    grand = float(matrix.mean())
    between_seed = float(len(cells) * np.var(matrix.mean(axis=1), ddof=1)) if len(seeds) > 1 else float("nan")
    between_cell = float(len(seeds) * np.var(matrix.mean(axis=0), ddof=1)) if len(cells) > 1 else float("nan")
    residual = matrix - matrix.mean(axis=1, keepdims=True) - matrix.mean(axis=0, keepdims=True) + grand
    denominator = (len(seeds) - 1) * (len(cells) - 1)
    return HierarchicalEffect(
        metric=metric,
        n_seeds=len(seeds),
        n_cells=len(cells),
        per_seed=tuple(
            {"seed": int(seed), "mean_difference": float(per_seed_difference[index]),
             "per_cell": {cell: float(matrix[index, j]) for j, cell in enumerate(cells)}}
            for index, seed in enumerate(seeds)
        ),
        per_cell=per_cell,
        effect=effect,
        test=verdict,
        cluster_bootstrap_ci=(cluster_low, cluster_high),
        variance_components={
            "between_seed_mean_square": between_seed,
            "between_cell_mean_square": between_cell,
            "residual_mean_square": float(np.sum(residual**2) / denominator) if denominator > 0 else float("nan"),
            "note": "one-way mean squares over the paired differences; diagnostics, not a "
                    "variance-components model — five seeds cannot identify one",
        },
        cell_fdr=cell_fdr,
    )


# --------------------------------------------------------------------------- #
# Permutation where feature identities are exchangeable
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FeaturePermutationResult:
    """A within-run permutation over features. **Not** a run-level result.

    §7.4 permits permutation "where feature identities are exchangeable", and
    within one run they sometimes are. What they never are is *independent*: the
    whole point of superposition is that features share directions. The
    calibration measures the cost — at 36 equicorrelated features (ρ = 0.3) this
    test rejects a true null 59% of the time against a nominal 5% — and that is
    the same error as pooling tokens, so the type is built to make the next step
    obvious.

    There is no path from a collection of these into any run-level estimator.
    :meth:`as_run_metric` returns one number, which is what a
    :class:`RunSummary` for this run should carry; the p-value is a description
    of *this run* and does not survive being averaged across seeds.
    """

    run_id: str
    design: str
    statistic: float
    p_value: float
    p_value_floor: float
    alpha: float
    n_features: int
    resamples: int
    detail: dict = field(default_factory=dict)

    @property
    def significant(self) -> bool:
        return bool(self.p_value <= self.alpha)

    def as_run_metric(self) -> float:
        """The one number this run contributes upward: the observed statistic."""
        return float(self.statistic)

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "design": self.design,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "p_value_floor": self.p_value_floor,
            "alpha": self.alpha,
            "n_features": self.n_features,
            "resamples": self.resamples,
            "unit": "feature",
            "scope": "within_run",
            "detail": dict(self.detail),
        }


def feature_permutation_test(
    values_a,
    values_b,
    *,
    run_id: str,
    design: str = "paired",
    alpha: float = ALPHA,
    resamples: int = BOOTSTRAP_RESAMPLES,
    rng=None,
) -> FeaturePermutationResult:
    """Permute feature identities within one run.

    ``design="paired"``       one value per feature under each arm, at the same
                              feature — sign-flip over features.
    ``design="independent"``  two disjoint sets of features within one run (say,
                              the ones the mechanism handles and the ones it does
                              not) — shuffle the labels.

    The arrays are per *feature*, and this is the only function here that accepts
    an array at all. It takes ``run_id`` because the result belongs to a run and
    a result that cannot say which run it came from will eventually be pooled
    with one that came from another.
    """
    a = np.asarray(values_a, dtype=float).ravel()
    b = np.asarray(values_b, dtype=float).ravel()
    if a.size < 2 or b.size < 2:
        raise StatisticsError("a permutation over features needs at least two features per arm")
    if a.size > _UNIT_LIMIT["feature"] or b.size > _UNIT_LIMIT["feature"]:
        raise ExperimentalUnitError(
            f"{max(a.size, b.size)} values declared as features exceeds {_UNIT_LIMIT['feature']}; "
            "this is the shape of per-token values, not a feature bank"
        )
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise StatisticsError("feature values contain NaN; drop undefined features by name first")

    generator = _as_rng(rng)
    if design == "paired":
        if a.size != b.size:
            raise StatisticsError(
                f"a paired feature permutation needs the same features in both arms, "
                f"got {a.size} and {b.size}"
            )
        result = _sign_flip_test(b - a, alpha=alpha, resamples=resamples, rng=generator)
    elif design == "independent":
        result = _label_shuffle_test(a, b, alpha=alpha, resamples=resamples, rng=generator)
    else:
        raise StatisticsError(f"unknown feature permutation design {design!r}")

    return FeaturePermutationResult(
        run_id=run_id,
        design=design,
        statistic=float(result.statistic),
        p_value=float(result.p_value),
        p_value_floor=float(result.p_value_floor),
        alpha=alpha,
        n_features=int(a.size if design == "paired" else a.size + b.size),
        resamples=int(result.detail.get("arrangements", resamples)),
        detail=dict(result.detail) | {
            "warning": "features in superposition are not independent; the permutation null "
                       "assumes they are, and is anti-conservative when they are not",
        },
    )


# --------------------------------------------------------------------------- #
# False-discovery control across mechanism-site grids
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FDRResult:
    """Benjamini–Hochberg or Benjamini–Yekutieli over a family of tests.

    §6.3's activity grid is layers × heads × sites; twenty-four of them tested at
    0.05 gives a 70% chance of at least one false positive under a true global
    null, which the calibration measures rather than assumes. ``attainable``
    records whether the family *could* have rejected anything: a grid of exact
    permutation tests at five seeds has every p-value at or above 0.0625, so no
    BH threshold can be met and "nothing survived correction" would be a
    statement about arithmetic rather than about mechanism.
    """

    method: str
    alpha: float
    n_tests: int
    labels: tuple[str, ...]
    p_values: tuple[float, ...]
    q_values: tuple[float, ...]
    rejected: tuple[bool, ...]
    threshold: float | None
    attainable: bool
    p_value_floor: float

    @property
    def n_rejected(self) -> int:
        return int(sum(self.rejected))

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "alpha": self.alpha,
            "n_tests": self.n_tests,
            "labels": list(self.labels),
            "p_values": list(self.p_values),
            "q_values": list(self.q_values),
            "rejected": list(self.rejected),
            "n_rejected": self.n_rejected,
            "threshold": self.threshold,
            "attainable": self.attainable,
            "p_value_floor": self.p_value_floor,
        }


def fdr_control(
    p_values: Sequence[float],
    *,
    alpha: float = ALPHA,
    method: str = "bh",
    labels: Sequence[str] | None = None,
    p_value_floor: float = 0.0,
) -> FDRResult:
    """Control the false-discovery rate over a family of p-values.

    ``bh``  Benjamini–Hochberg. Valid under independence and positive dependence,
            which is the usual case for a mechanism grid whose sites share a
            model and a seed.
    ``by``  Benjamini–Yekutieli, valid under arbitrary dependence at the cost of
            a factor of ``sum 1/i`` — 3.8 at twenty-four sites. Use it when the
            sites' dependence could be negative and you cannot argue otherwise.
    """
    values = np.asarray(list(p_values), dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise StatisticsError("fdr_control needs a non-empty one-dimensional list of p-values")
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise StatisticsError("p-values must be finite and in [0, 1]")
    names = tuple(labels) if labels is not None else tuple(f"test_{i}" for i in range(values.size))
    if len(names) != values.size:
        raise StatisticsError(f"{len(names)} labels for {values.size} p-values")

    m = values.size
    if method == "bh":
        scale = 1.0
    elif method == "by":
        scale = float(np.sum(1.0 / np.arange(1, m + 1)))
    else:
        raise StatisticsError(f"unknown FDR method {method!r}; expected bh or by")

    order = np.argsort(values, kind="stable")
    ranks = np.arange(1, m + 1)
    sorted_values = values[order]
    thresholds = alpha * ranks / (m * scale)
    passing = np.nonzero(sorted_values <= thresholds)[0]
    if passing.size:
        cutoff_index = int(passing[-1])
        threshold = float(sorted_values[cutoff_index])
        rejected_sorted = np.zeros(m, dtype=bool)
        rejected_sorted[: cutoff_index + 1] = True
    else:
        threshold = None
        rejected_sorted = np.zeros(m, dtype=bool)

    q_sorted = np.minimum.accumulate((sorted_values * m * scale / ranks)[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    q_values = np.empty(m)
    rejected = np.empty(m, dtype=bool)
    q_values[order] = q_sorted
    rejected[order] = rejected_sorted

    return FDRResult(
        method=method,
        alpha=alpha,
        n_tests=m,
        labels=names,
        p_values=tuple(float(v) for v in values),
        q_values=tuple(float(v) for v in q_values),
        rejected=tuple(bool(v) for v in rejected),
        threshold=threshold,
        # BH's most generous threshold is the last one, alpha*m/(m*scale) = alpha/scale.
        # A family whose p-values cannot go below that can never reject anything.
        attainable=bool(p_value_floor <= alpha / scale),
        p_value_floor=float(p_value_floor),
    )


def fdr_over_tests(
    tests: Sequence[TestResult],
    *,
    alpha: float = ALPHA,
    method: str = "bh",
    labels: Sequence[str] | None = None,
) -> FDRResult:
    """FDR over a grid of :class:`TestResult`, carrying their p-value floor up.

    The floor is what makes this more than a convenience wrapper. A grid of exact
    permutation tests at five seeds cannot produce a p-value below 0.0625; BH's
    most generous threshold is ``alpha``, so nothing can be rejected and
    ``attainable`` says so instead of the family quietly reporting zero
    discoveries.
    """
    if not tests:
        raise StatisticsError("fdr_over_tests needs at least one test")
    floor = max(float(test.p_value_floor) for test in tests)
    return fdr_control(
        [test.p_value for test in tests],
        alpha=alpha,
        method=method,
        labels=labels,
        p_value_floor=floor,
    )


# --------------------------------------------------------------------------- #
# Predeclared primary comparisons
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ComparisonDeclaration:
    """One ``reports/comparisons/*.json``: what is being compared, against what.

    The file names a ``claim``. It does *not* get to name a metric — or rather,
    it may echo one, and the echo is checked against the packet and refused if it
    disagrees. §7.4's "predeclared primary comparisons" is only meaningful if the
    declaration and the prediction cannot drift apart, and the place the
    prediction lives is the committed claim packet, whose commit time
    ``bin/check_prereg.sh`` compares against the run.
    """

    name: str
    path: Path
    claim_id: str
    control_run: str
    candidate_runs: tuple[str, ...]
    matching_strategy: str
    permitted_differences: dict
    declared_primary_metric: str | None
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "path": str(self.path),
            "claim": self.claim_id,
            "control_run": self.control_run,
            "candidate_runs": list(self.candidate_runs),
            "matching_strategy": self.matching_strategy,
            "permitted_differences": dict(self.permitted_differences),
        }


def load_comparison(path: Path | str) -> ComparisonDeclaration:
    """Read a comparison declaration, refusing one that does not name a claim."""
    path = Path(path)
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise StatisticsError(f"{path} is not a JSON object")
    claim = raw.get("claim") or raw.get("claim_id")
    if not claim:
        raise StatisticsError(
            f"{path} names no claim. A comparison without a pre-registration is not a "
            "predeclared primary comparison; add claim: <claim_id> and commit the packet first."
        )
    control = raw.get("control_run")
    candidates = tuple(raw.get("candidate_runs") or ())
    if not control or not candidates:
        raise StatisticsError(f"{path} must name control_run and a non-empty candidate_runs")
    known = {"claim", "claim_id", "control_run", "candidate_runs", "matching_strategy",
             "permitted_differences", "primary_metric"}
    return ComparisonDeclaration(
        name=path.stem,
        path=path,
        claim_id=str(claim),
        control_run=str(control),
        candidate_runs=tuple(str(run) for run in candidates),
        matching_strategy=str(raw.get("matching_strategy") or "width_matched"),
        permitted_differences=dict(raw.get("permitted_differences") or {}),
        declared_primary_metric=raw.get("primary_metric"),
        extra={key: value for key, value in raw.items() if key not in known},
    )


def primary_metric_for(declaration: ComparisonDeclaration, *, claims_dir: Path | str) -> tuple[str, str]:
    """The primary metric, read out of the committed claim packet.

    Returns ``(metric, source)``. There is deliberately no argument by which a
    caller can supply the metric: §7.4's "predeclared primary comparison" means
    the metric was fixed before the run, and a function that accepted it at call
    time would let the metric be chosen after the numbers were seen. The only
    place it can come from is ``primary_metric_key`` in the packet whose commit
    time the pre-registration gate already checks.
    """
    from architecture_mechanics.experiments.claim_packet import load_packet

    claims_dir = Path(claims_dir)
    candidates = [claims_dir / f"{declaration.claim_id}.yml", claims_dir / f"{declaration.claim_id}.yaml"]
    packet_path = next((path for path in candidates if path.is_file()), None)
    if packet_path is None:
        raise StatisticsError(
            f"comparison {declaration.name!r} names claim {declaration.claim_id!r}, which is not "
            f"in {claims_dir}. The prediction must exist before the comparison reads it."
        )
    packet = load_packet(packet_path)
    metric = packet.primary_metric_key
    if not metric:
        raise StatisticsError(
            f"claim packet {packet_path} states PRIMARY_METRIC in prose but has no "
            "primary_metric_key. Prose cannot be compared against a number: add the machine "
            "name and commit it before the run."
        )
    if declaration.declared_primary_metric and declaration.declared_primary_metric != metric:
        raise StatisticsError(
            f"comparison {declaration.name!r} declares primary_metric "
            f"{declaration.declared_primary_metric!r} but claim {declaration.claim_id!r} "
            f"predeclared {metric!r}. The packet is the prediction; a comparison that renames "
            "the metric after the fact is choosing an outcome."
        )
    return str(metric), f"{packet_path}#primary_metric_key"


@dataclass(frozen=True)
class ComparisonRecord:
    """The structured record §7.4 asks for, as data rather than as a sentence.

    Per-seed raw values, the effect size, its interval, the test used and its
    p-value floor, and the seed count — assembled once, written into
    ``summary.json`` under :data:`SUMMARY_KEY`, and read back by whoever writes
    the paper. It is a record and not a string because a formatted table cannot
    be re-checked, re-plotted, or compared against the claim it belongs to.
    """

    comparison: str
    claim_id: str
    metric: str
    metric_source: str
    primary: bool
    control_arm: str
    candidate_arm: str
    n_seeds: int
    n_cells: int
    seeds: tuple[int, ...]
    per_run: tuple[dict, ...]
    per_seed_difference: tuple[dict, ...]
    effect: EffectSize
    standardized: EffectSize
    test: TestResult
    alpha: float
    ci_level: float
    exploratory_cells: tuple[dict, ...] = ()
    variance_components: dict = field(default_factory=dict)
    statistics_version: str = STATISTICS_VERSION

    def as_dict(self) -> dict:
        return {
            "schema": COMPARISON_SCHEMA,
            "statistics_version": self.statistics_version,
            "comparison": self.comparison,
            "claim_id": self.claim_id,
            "metric": self.metric,
            "metric_source": self.metric_source,
            "primary": self.primary,
            "control_arm": self.control_arm,
            "candidate_arm": self.candidate_arm,
            "n_seeds": self.n_seeds,
            "n_cells": self.n_cells,
            "seeds": list(self.seeds),
            "per_run": [dict(row) for row in self.per_run],
            "per_seed_difference": [dict(row) for row in self.per_seed_difference],
            "effect": self.effect.as_dict(),
            "standardized_effect": self.standardized.as_dict(),
            "test": self.test.as_dict(),
            "alpha": self.alpha,
            "ci_level": self.ci_level,
            "exploratory_cells": [dict(row) for row in self.exploratory_cells],
            "variance_components": dict(self.variance_components),
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> ComparisonRecord:
        if payload.get("schema") != COMPARISON_SCHEMA:
            raise StatisticsError(
                f"comparison record has schema {payload.get('schema')!r}, expected "
                f"{COMPARISON_SCHEMA!r}"
            )
        return cls(
            comparison=payload["comparison"],
            claim_id=payload["claim_id"],
            metric=payload["metric"],
            metric_source=payload["metric_source"],
            primary=payload["primary"],
            control_arm=payload["control_arm"],
            candidate_arm=payload["candidate_arm"],
            n_seeds=payload["n_seeds"],
            n_cells=payload["n_cells"],
            seeds=tuple(payload["seeds"]),
            per_run=tuple(dict(row) for row in payload["per_run"]),
            per_seed_difference=tuple(dict(row) for row in payload["per_seed_difference"]),
            effect=EffectSize.from_dict(payload["effect"]),
            standardized=EffectSize.from_dict(payload["standardized_effect"]),
            test=TestResult.from_dict(payload["test"]),
            alpha=payload["alpha"],
            ci_level=payload["ci_level"],
            exploratory_cells=tuple(dict(row) for row in payload.get("exploratory_cells") or ()),
            variance_components=dict(payload.get("variance_components") or {}),
            statistics_version=payload.get("statistics_version", STATISTICS_VERSION),
        )


def _build_record(
    *,
    comparison: str,
    claim_id: str,
    metric: str,
    metric_source: str,
    primary: bool,
    control,
    candidate,
    alpha: float,
    ci_level: float,
    test: str,
    resamples: int,
    rng,
) -> ComparisonRecord:
    sample = paired_sample(control, candidate, metric)
    generator = _as_rng(rng)
    effect = paired_effect(
        control, candidate, metric, ci_level=ci_level, resamples=resamples, rng=generator
    )
    standardized = standardized_effect(
        control, candidate, metric, ci_level=ci_level, resamples=resamples, rng=generator
    )
    verdict = paired_test(
        control, candidate, metric, test=test, alpha=alpha, resamples=resamples, rng=generator
    )

    exploratory: tuple[dict, ...] = ()
    components: dict = {}
    if len(sample.cells) > 1:
        hierarchical = hierarchical_effect(
            control, candidate, metric, alpha=alpha, test=test, ci_level=ci_level,
            resamples=resamples, rng=generator,
        )
        effect, verdict = hierarchical.effect, hierarchical.test
        exploratory = hierarchical.per_cell
        components = hierarchical.variance_components
        per_seed = hierarchical.per_seed
    else:
        per_seed = tuple(
            {"seed": int(seed), "mean_difference": float(difference), "per_cell": {cell: float(difference)}}
            for (seed, cell), difference in zip(sample.keys, sample.differences, strict=True)
        )

    per_run = tuple(
        row
        for control_run, candidate_run in zip(sample.control_runs, sample.candidate_runs, strict=True)
        for row in (
            {"run_id": control_run.run_id, "seed": control_run.seed, "cell": control_run.cell,
             "arm": control_run.arm, "value": control_run.value(metric)},
            {"run_id": candidate_run.run_id, "seed": candidate_run.seed, "cell": candidate_run.cell,
             "arm": candidate_run.arm, "value": candidate_run.value(metric)},
        )
    )
    arms = {run.arm for run in sample.control_runs}, {run.arm for run in sample.candidate_runs}
    return ComparisonRecord(
        comparison=comparison,
        claim_id=claim_id,
        metric=metric,
        metric_source=metric_source,
        primary=primary,
        control_arm="+".join(sorted(arms[0])),
        candidate_arm="+".join(sorted(arms[1])),
        n_seeds=len(sample.seeds),
        n_cells=len(sample.cells),
        seeds=sample.seeds,
        per_run=per_run,
        per_seed_difference=per_seed,
        effect=effect,
        standardized=standardized,
        test=verdict,
        alpha=alpha,
        ci_level=ci_level,
        exploratory_cells=exploratory,
        variance_components=components,
    )


def primary_comparison(
    declaration,
    control,
    candidate,
    *,
    claims_dir: Path | str,
    alpha: float = ALPHA,
    ci_level: float = CI_LEVEL,
    test: str = PRIMARY_TEST,
    resamples: int = BOOTSTRAP_RESAMPLES,
    rng=None,
) -> ComparisonRecord:
    """The predeclared primary comparison. The metric comes from the packet.

    Note what is missing from the signature: there is no ``metric`` argument.
    That absence is the mechanism. Everything else about a comparison can be
    decided at call time; which number decides it cannot, because §7.4's whole
    point is that the choice was made before the data existed.
    """
    if isinstance(declaration, (str, Path)):
        declaration = load_comparison(declaration)
    if not isinstance(declaration, ComparisonDeclaration):
        raise StatisticsError(
            f"primary_comparison needs a ComparisonDeclaration or a path to one, got "
            f"{type(declaration).__name__}"
        )
    metric, source = primary_metric_for(declaration, claims_dir=claims_dir)
    return _build_record(
        comparison=declaration.name,
        claim_id=declaration.claim_id,
        metric=metric,
        metric_source=source,
        primary=True,
        control=control,
        candidate=candidate,
        alpha=alpha,
        ci_level=ci_level,
        test=test,
        resamples=resamples,
        rng=rng,
    )


def secondary_comparison(
    name: str,
    control,
    candidate,
    metric: str,
    *,
    claim_id: str,
    alpha: float = ALPHA,
    ci_level: float = CI_LEVEL,
    test: str = PRIMARY_TEST,
    resamples: int = BOOTSTRAP_RESAMPLES,
    rng=None,
) -> ComparisonRecord:
    """A comparison on a metric the packet did not predeclare, marked as such.

    Secondary comparisons are legitimate and necessary — §7.4 objects to
    *upgrading a claim* on one, not to computing it. The record carries
    ``primary: False`` and ``metric_source: "caller"`` so no reader can mistake
    it for the predeclared one.
    """
    return _build_record(
        comparison=name,
        claim_id=claim_id,
        metric=metric,
        metric_source="caller",
        primary=False,
        control=control,
        candidate=candidate,
        alpha=alpha,
        ci_level=ci_level,
        test=test,
        resamples=resamples,
        rng=rng,
    )


def attach_comparisons(summary: Mapping, records: Sequence[ComparisonRecord]) -> dict:
    """Put comparison records into a ``summary.json`` payload under one key.

    Accepts only :class:`ComparisonRecord`, on the same reasoning as
    ``ClaimGates.record`` accepting only a ``RungEvaluation``: the record is a
    measurement, and a dict that merely looks like one could have been written by
    anything.
    """
    payload = dict(summary)
    materialised = []
    for index, record in enumerate(records):
        if not isinstance(record, ComparisonRecord):
            raise StatisticsError(
                f"comparison {index} is a {type(record).__name__}; summary.json accepts only a "
                "ComparisonRecord produced by primary_comparison() or secondary_comparison()"
            )
        materialised.append(record.as_dict())
    payload[SUMMARY_KEY] = materialised
    return payload


def comparisons_from_summary(summary: Mapping) -> tuple[ComparisonRecord, ...]:
    """Read comparison records back out of a ``summary.json`` payload."""
    raw = summary.get(SUMMARY_KEY) or []
    if not isinstance(raw, list):
        raise StatisticsError(f"summary {SUMMARY_KEY!r} must be a list, got {type(raw).__name__}")
    return tuple(ComparisonRecord.from_dict(entry) for entry in raw)


# --------------------------------------------------------------------------- #
# The register of estimators and what they do under the null
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EstimatorSpec:
    """One estimator: what it is, and what it did when there was nothing there.

    ``status`` is a decision recorded in source, re-derived by ``--selftest`` from
    a fresh calibration. An estimator recorded as unusable that starts holding
    its level fails the gate exactly as loudly as a primary one that stops,
    because either means the record and the evidence have come apart.
    """

    name: str
    family: str
    design: str
    nominal_alpha: float
    definition: str
    status: str
    reason: str
    max_false_positive_rate: float
    recorded_fpr_at_5: float

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "family": self.family,
            "design": self.design,
            "nominal_alpha": self.nominal_alpha,
            "definition": self.definition,
            "status": self.status,
            "reason": self.reason,
            "max_false_positive_rate": self.max_false_positive_rate,
            "recorded_fpr_at_5": self.recorded_fpr_at_5,
        }


LEVEL_TOLERANCE = 0.075
"""The realised false-positive rate an estimator may have at five units and still
be usable as a test: one and a half times nominal. Not a generous allowance — the
estimators that fail it fail by a factor of three or four, and the ones that pass
it pass at 0.05 or below. It is set with headroom for Monte Carlo error rather
than to admit anything marginal."""

FIVE = 5
"""§10.1's replication requirement, and therefore the operating point every
status in :data:`ESTIMATOR_SPECS` is decided at. An estimator that behaves at
twenty seeds and not at five is, for this laboratory, an estimator that does not
behave."""

FEATURE_OPERATING_POINT = 36
"""Where the feature-level status is decided: §4.5's suggested feature bank."""

FORBIDDEN_ESTIMATORS: frozenset[str] = frozenset(
    {"pooled_cells_t", "pooled_tokens_t", "uncorrected_grid"}
)
"""Three analyses from §7.4's "avoid" list, kept in the calibration on purpose.
Each is a route by which a result can be manufactured from noise, and knowing the
size of each — 4.6×, 14×, and 14× nominal at five seeds — is more use than a
warning comment. The selftest asserts they are *still* broken, because a
forbidden analysis that started behaving would mean the calibration had stopped
measuring what it claims to."""

ESTIMATOR_SPECS: tuple[EstimatorSpec, ...] = (
    EstimatorSpec(
        name="paired_t",
        family="effect_size_and_test",
        design="matched arms, one cell",
        nominal_alpha=ALPHA,
        definition="t-test on the per-seed differences, df = n-1",
        status="level_holding",
        reason="the only paired estimator that both holds its level at five seeds "
               "(0.048 worst of three noise shapes) and can reject there at all; adopted "
               "as PRIMARY_TEST",
        max_false_positive_rate=LEVEL_TOLERANCE,
        recorded_fpr_at_5=0.048,
    ),
    EstimatorSpec(
        name="paired_permutation",
        family="permutation",
        design="matched arms, exact sign flip",
        nominal_alpha=ALPHA,
        definition="exact two-sided sign-flip permutation over the per-seed differences",
        status="unusable_at_five_seeds",
        reason="2^5 = 32 arrangements put the smallest attainable p-value at 0.0625, above "
               "0.05 regardless of the data: measured power is exactly zero at every effect "
               "size. Exact and assumption-free from six seeds up (0.046 at ten)",
        max_false_positive_rate=LEVEL_TOLERANCE,
        recorded_fpr_at_5=0.000,
    ),
    EstimatorSpec(
        name="bootstrap_percentile",
        family="bootstrap_interval",
        design="matched arms, percentile interval on the mean difference",
        nominal_alpha=ALPHA,
        definition="95% interval from the empirical quantiles of the resampled mean; "
                   "significant when it excludes zero",
        status="descriptive_only",
        reason="16.7% false positives at five seeds — a nominal 95% interval with 83% "
               "coverage. Reportable as a description of where the estimate sits, never as "
               "the thing that decides",
        max_false_positive_rate=LEVEL_TOLERANCE,
        recorded_fpr_at_5=0.167,
    ),
    EstimatorSpec(
        name="bootstrap_bca",
        family="bootstrap_interval",
        design="matched arms, bias-corrected and accelerated interval",
        nominal_alpha=ALPHA,
        definition="BCa interval on the mean difference; significant when it excludes zero",
        status="descriptive_only",
        reason="18.9% at five seeds — worse than percentile, because the bias correction and "
               "the acceleration are both estimated from the same five points. Better "
               "asymptotics are not better behaviour at n = 5",
        max_false_positive_rate=LEVEL_TOLERANCE,
        recorded_fpr_at_5=0.189,
    ),
    EstimatorSpec(
        name="bootstrap_studentized",
        family="bootstrap_interval",
        design="matched arms, bootstrap-t interval",
        nominal_alpha=ALPHA,
        definition="interval from the bootstrap distribution of (mean* - mean)/se*; "
                   "significant when it excludes zero",
        status="level_holding",
        reason="0.044 at five seeds — the only bootstrap interval whose coverage is what it "
               "says, and therefore the default for paired_effect. At three seeds it is "
               "infinitely wide and rejects nothing, which is the honest report and not a "
               "defect",
        max_false_positive_rate=LEVEL_TOLERANCE,
        recorded_fpr_at_5=0.044,
    ),
    EstimatorSpec(
        name="welch_t",
        family="effect_size_and_test",
        design="unmatched arms",
        nominal_alpha=ALPHA,
        definition="Welch's unequal-variance t-test with Satterthwaite degrees of freedom",
        status="level_holding",
        reason="0.046 at five seeds per arm; adopted where the seed sets cannot be matched. "
               "Costs roughly half the power of the paired test at the same real effect, "
               "because it pays for the between-seed variation pairing removes",
        max_false_positive_rate=LEVEL_TOLERANCE,
        recorded_fpr_at_5=0.046,
    ),
    EstimatorSpec(
        name="unpaired_permutation",
        family="permutation",
        design="unmatched arms, exact label shuffle",
        nominal_alpha=ALPHA,
        definition="exact two-sided label-shuffle permutation over the pooled arms",
        status="level_holding",
        reason="0.052 at five per arm: C(10,5) = 252 arrangements put the floor at 0.008, so "
               "unlike its paired cousin it can reject at five. At three per arm the floor is "
               "0.1 and it cannot",
        max_false_positive_rate=LEVEL_TOLERANCE,
        recorded_fpr_at_5=0.052,
    ),
    EstimatorSpec(
        name="hierarchical_seed_t",
        family="hierarchical",
        design="seed × 4 difficulty cells, cells averaged into the seed",
        nominal_alpha=ALPHA,
        definition="paired t-test on each seed's cell-averaged difference",
        status="level_holding",
        reason="0.051 at five seeds; adopted for any comparison with a difficulty matrix. "
               "Averaging four cells buys real power (0.88 versus 0.70 at dz = 1.5) without "
               "costing level, because the averaging happens below the experimental unit "
               "rather than beside it",
        max_false_positive_rate=LEVEL_TOLERANCE,
        recorded_fpr_at_5=0.051,
    ),
    EstimatorSpec(
        name="hierarchical_cluster_bootstrap",
        family="hierarchical",
        design="seed × 4 difficulty cells, seeds resampled with replacement",
        nominal_alpha=ALPHA,
        definition="percentile interval from resampling seeds; significant when it excludes zero",
        status="descriptive_only",
        reason="0.175 at five seeds. Clustering on the right unit does not rescue a percentile "
               "bootstrap of five things — the unit was never the problem, the sample size was",
        max_false_positive_rate=LEVEL_TOLERANCE,
        recorded_fpr_at_5=0.175,
    ),
    EstimatorSpec(
        name="pooled_cells_t",
        family="hierarchical",
        design="seed × 4 difficulty cells, every run an independent sample",
        nominal_alpha=ALPHA,
        definition="paired t-test over all 20 seed×cell differences at once",
        status="forbidden",
        reason="0.230 at five seeds — 4.6× nominal. The cells of one seed share that seed; "
               "counting them as twenty independent runs counts the seed effect four times. "
               "This is §7.4's token-pooling error one level up, and there is no route to it "
               "through the exported API",
        max_false_positive_rate=LEVEL_TOLERANCE,
        recorded_fpr_at_5=0.230,
    ),
    EstimatorSpec(
        name="pooled_tokens_t",
        family="forbidden",
        design="64 tokens per run, every token an independent sample",
        nominal_alpha=ALPHA,
        definition="Welch t-test over all pooled per-token values",
        status="forbidden",
        reason="0.710 at five seeds — 14× nominal, and it would be worse at 4096 tokens. "
               "§7.4 names this one directly. It is calibrated here so the cost is a measured "
               "number rather than a caution",
        max_false_positive_rate=LEVEL_TOLERANCE,
        recorded_fpr_at_5=0.710,
    ),
    EstimatorSpec(
        name="fdr_bh",
        family="false_discovery",
        design="24-site mechanism grid",
        nominal_alpha=ALPHA,
        definition="Benjamini–Hochberg at q = 0.05; significant when any site is rejected",
        status="level_holding",
        reason="0.052 with independent sites and 0.051 with sites correlated at 0.5 — BH's "
               "positive-dependence guarantee holds up on exactly the dependence a mechanism "
               "grid has. Adopted for every site grid",
        max_false_positive_rate=LEVEL_TOLERANCE,
        recorded_fpr_at_5=0.052,
    ),
    EstimatorSpec(
        name="fdr_by",
        family="false_discovery",
        design="24-site mechanism grid",
        nominal_alpha=ALPHA,
        definition="Benjamini–Yekutieli at q = 0.05; valid under arbitrary dependence",
        status="level_holding",
        reason="0.016 — conservative by the factor of 3.8 it pays for dropping the dependence "
               "assumption. Correct when the sites' dependence could be negative and nobody "
               "can argue it is not",
        max_false_positive_rate=LEVEL_TOLERANCE,
        recorded_fpr_at_5=0.016,
    ),
    EstimatorSpec(
        name="uncorrected_grid",
        family="forbidden",
        design="24-site mechanism grid, no correction",
        nominal_alpha=ALPHA,
        definition="every site tested at 0.05; significant when any site is",
        status="forbidden",
        reason="0.700 with independent sites: test twenty-four things at 0.05 and you will "
               "find one. This is the row that says what FDR control is worth",
        max_false_positive_rate=LEVEL_TOLERANCE,
        recorded_fpr_at_5=0.700,
    ),
    EstimatorSpec(
        name="feature_permutation",
        family="permutation",
        design="within one run, 36 features, sign flip over feature identities",
        nominal_alpha=ALPHA,
        definition="Monte-Carlo two-sided sign-flip permutation over per-feature differences",
        status="level_holding",
        reason="0.046 when features are independent — and 0.593 when they are merely "
               "equicorrelated at 0.3, which is what superposition is. Valid only where "
               "feature independence can be argued, and the equicorrelated row is calibrated "
               "beside it so the condition cannot be forgotten",
        max_false_positive_rate=LEVEL_TOLERANCE,
        recorded_fpr_at_5=0.046,
    ),
)

ESTIMATOR_SPEC_BY_NAME: dict[str, EstimatorSpec] = {spec.name: spec for spec in ESTIMATOR_SPECS}

ADOPTED: dict[str, str] = {
    "paired_comparison": "paired_t",
    "paired_interval": "bootstrap_studentized",
    "unmatched_comparison": "welch_t",
    "difficulty_matrix": "hierarchical_seed_t",
    "mechanism_site_grid": "fdr_bh",
    "feature_exchangeability": "feature_permutation",
}
"""What this laboratory uses for what, decided by the calibration above. Recorded
as data so ``state/08_statistics.md`` and the selftest read the same thing."""

THRESHOLDS: dict[str, object] = {
    "alpha": ALPHA,
    "ci_level": CI_LEVEL,
    "minimum_seeds_for_a_test": 5,
    "minimum_seeds_for_the_exact_paired_permutation": 6,
    "minimum_detectable_effect_dz_at_5_seeds": 1.68,
    "level_tolerance": LEVEL_TOLERANCE,
    "note": "no estimator's threshold was corrected; the two whose realised level exceeded "
            "nominal at five seeds were demoted to descriptive_only instead, because a "
            "percentile bootstrap re-tuned to 0.014 to buy back a 0.05 level would still be "
            "an interval nobody could interpret.",
}

NULL_MODELS: tuple[str, ...] = ("gaussian", "heavy_tailed", "skewed")
"""Three shapes of run-to-run noise, all with unit variance so the effect scale
means the same thing in each.

``gaussian``      what every parametric test assumes.
``heavy_tailed``  Student-t on three degrees of freedom: one seed in twenty goes
                  badly for reasons that have nothing to do with the
                  architecture.
``skewed``        a centred exponential: a metric against a ceiling, where most
                  seeds cluster and a few trail off. Accuracy near 0.9 looks
                  like this.
"""


def _null_noise(kind: str, rng: np.random.Generator, shape) -> np.ndarray:
    if kind == "gaussian":
        return rng.standard_normal(shape)
    if kind == "heavy_tailed":
        return rng.standard_t(3, size=shape) / math.sqrt(3.0)
    if kind == "skewed":
        return rng.standard_exponential(shape) - 1.0
    raise StatisticsError(f"unknown null model {kind!r}; expected one of {list(NULL_MODELS)}")


# --------------------------------------------------------------------------- #
# Calibration: what each estimator does when the truth is known
# --------------------------------------------------------------------------- #

BASE_METRIC_VALUE = 0.90
"""Where the synthetic metric sits. Arbitrary — every estimator here is
location-invariant — but a number near a real accuracy keeps the printed tables
legible."""

SEED_SD = 1.0
"""Between-seed variation shared by both arms. It cancels in a paired difference
and does not in an unpaired one, which is the whole of what pairing buys and is
visible as the gap between the two power columns."""

RUN_SD = 1.0
"""Within-arm, per-run noise, in units of which every effect below is quoted.
The paired difference therefore has standard deviation ``sqrt(2) * RUN_SD``, and
an effect of ``dz`` is injected as ``dz * sqrt(2) * RUN_SD`` so that ``dz`` means
"this many standard deviations of the run-to-run *difference*" throughout."""

CELL_CORRELATION = 0.5
"""How much of a run's deviation is a property of the seed rather than of the
cell. A seed that trained badly trained badly everywhere; setting this to zero
would make averaging four cells look like a free factor of two in precision,
which is exactly the illusion §7.4's experimental-unit rule exists to stop."""

FEATURE_CORRELATION = 0.3
"""Equicorrelation between features in the calibration's correlated-feature
condition. Superposition is *defined* by features sharing directions, so this is
the realistic case and independence is the special one."""

SITE_CORRELATION = 0.5
"""Shared per-seed component across mechanism sites in the correlated-grid
condition. Sites in one model on one seed are not independent tests."""

GRID_SITES = 24
"""Two layers by two heads by six sites — the shape of §6.3's activity grid."""

TOKENS_PER_RUN = 64
"""How many per-token values the forbidden ``pooled_tokens_t`` estimator pools.
Small; the point is made at any size, and at 64 it is already catastrophic."""

SEED_COUNTS: tuple[int, ...] = (3, 5, 10)
FEATURE_COUNTS: tuple[int, ...] = (16, 36, 64)
"""§4.5's suggested feature banks. The feature permutation's replication unit is
the feature, so this is its size axis and the seed counts are not."""

EFFECT_SIZES: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)

NULL_REPLICATES = 2000
POWER_REPLICATES = 2000
CALIBRATION_RESAMPLES = 2000
"""Bootstrap and Monte-Carlo permutation draws *inside* each calibration
replicate. Coverage is insensitive to this above about a thousand, and the
calibration performs a hundred thousand of them."""

SELFTEST_NULL_REPLICATES = 300
SELFTEST_POWER_REPLICATES = 200

TARGET_POWER = 0.80

INTERESTING_EFFECTS: tuple[float, ...] = (0.3, 0.5, 1.0)
"""Effects small enough to be worth asking how many seeds they would need. 0.5
is the smallest difference this program would call scientifically interesting on
a capability metric; 0.3 is the smallest it could imagine caring about."""


@dataclass(frozen=True)
class _PairedDraw:
    control: tuple[RunSummary, ...]
    candidate: tuple[RunSummary, ...]


def _runs(values: np.ndarray, *, arm: str, seeds, cells) -> tuple[RunSummary, ...]:
    return tuple(
        RunSummary(
            run_id=f"{arm}-s{seed}-{cell}",
            seed=int(seed),
            arm=arm,
            cell=str(cell),
            metrics={"m": float(value)},
        )
        for value, seed, cell in zip(values.ravel(), seeds, cells, strict=True)
    )


def _draw_paired(
    rng: np.random.Generator, *, n_seeds: int, null: str, effect_dz: float, matched: bool = True
) -> _PairedDraw:
    """Two arms over one cell. ``matched`` shares the seed effect, as §7.2 asks."""
    seeds = np.arange(n_seeds)
    shared = rng.standard_normal(n_seeds) * SEED_SD
    other = shared if matched else rng.standard_normal(n_seeds) * SEED_SD
    offset = effect_dz * math.sqrt(2.0) * RUN_SD
    control = BASE_METRIC_VALUE + shared + RUN_SD * _null_noise(null, rng, n_seeds)
    candidate = BASE_METRIC_VALUE + other + RUN_SD * _null_noise(null, rng, n_seeds) + offset
    cells = [CELL_ALL] * n_seeds
    candidate_seeds = seeds if matched else seeds + 1000
    return _PairedDraw(
        control=_runs(control, arm="control", seeds=seeds, cells=cells),
        candidate=_runs(candidate, arm="candidate", seeds=candidate_seeds, cells=cells),
    )


def _draw_hierarchical(
    rng: np.random.Generator, *, n_seeds: int, n_cells: int, null: str, effect_dz: float
) -> tuple[_PairedDraw, np.ndarray]:
    """A seed × cell matrix whose per-run differences share a per-seed component.

    The per-run difference has the same standard deviation as in the flat design,
    ``sqrt(2) * RUN_SD``, split so that :data:`CELL_CORRELATION` of its variance
    belongs to the seed and the rest to the cell. Averaging four cells therefore
    reduces the noise by a factor of ``sqrt(0.5 + 0.5/4) = 0.79`` and not by
    ``sqrt(4)``, which is the difference between an honest hierarchical analysis
    and the pooled one calibrated beside it.
    """
    seeds = np.repeat(np.arange(n_seeds), n_cells)
    cells = [f"cell{index}" for index in range(n_cells)] * n_seeds
    base = (
        BASE_METRIC_VALUE
        + rng.standard_normal(n_seeds)[:, None] * SEED_SD
        + rng.standard_normal(n_cells)[None, :] * SEED_SD
    )
    seed_scale = RUN_SD * math.sqrt(CELL_CORRELATION)
    cell_scale = RUN_SD * math.sqrt(1.0 - CELL_CORRELATION)
    offset = effect_dz * math.sqrt(2.0) * RUN_SD

    control = (
        base
        + _null_noise(null, rng, (n_seeds, 1)) * seed_scale
        + _null_noise(null, rng, (n_seeds, n_cells)) * cell_scale
    )
    candidate = (
        base
        + _null_noise(null, rng, (n_seeds, 1)) * seed_scale
        + _null_noise(null, rng, (n_seeds, n_cells)) * cell_scale
        + offset
    )
    draw = _PairedDraw(
        control=_runs(control, arm="control", seeds=seeds, cells=cells),
        candidate=_runs(candidate, arm="candidate", seeds=seeds, cells=cells),
    )
    return draw, candidate - control


def _draw_feature_arms(
    rng: np.random.Generator, *, n_features: int, correlation: float, effect: float
):
    """Per-feature values for two arms of one run, equicorrelated at ``correlation``.

    The correlation is what the permutation null assumes away. Superposition puts
    features on shared directions, so a per-feature measurement of one arm minus
    the other carries a component common to every feature — and sign-flipping
    feature labels cannot generate that component, so the null it builds is too
    narrow.
    """
    common = rng.standard_normal()
    independent = rng.standard_normal(n_features)
    differences = (
        math.sqrt(correlation) * common + math.sqrt(1.0 - correlation) * independent + effect
    )
    baseline = rng.standard_normal(n_features)
    return baseline, baseline + differences


def _draw_grid(rng: np.random.Generator, *, n_seeds: int, n_sites: int, correlation: float,
               effect_dz: float, n_true: int) -> np.ndarray:
    """``(n_seeds, n_sites)`` paired differences; the first ``n_true`` sites are real."""
    common = rng.standard_normal((n_seeds, 1))
    independent = rng.standard_normal((n_seeds, n_sites))
    differences = math.sqrt(correlation) * common + math.sqrt(1.0 - correlation) * independent
    differences = differences * math.sqrt(2.0) * RUN_SD
    if n_true:
        differences[:, :n_true] += effect_dz * math.sqrt(2.0) * RUN_SD
    return differences


def _draw_tokens(rng: np.random.Generator, *, n_seeds: int, null: str, effect_dz: float):
    """Per-token values for two arms — the input the forbidden estimator pools.

    Each run has a run-level offset shared by all its tokens. That is what makes
    the tokens non-independent, and it is not an artefact of the simulation: two
    tokens from one run share a model, a seed, and a training trajectory.
    """
    shared = rng.standard_normal(n_seeds) * SEED_SD
    run_offset_control = RUN_SD * _null_noise(null, rng, n_seeds)
    run_offset_candidate = RUN_SD * _null_noise(null, rng, n_seeds)
    token_sd = RUN_SD
    control = (
        BASE_METRIC_VALUE + (shared + run_offset_control)[:, None]
        + token_sd * rng.standard_normal((n_seeds, TOKENS_PER_RUN))
    )
    candidate = (
        BASE_METRIC_VALUE + (shared + run_offset_candidate)[:, None]
        + token_sd * rng.standard_normal((n_seeds, TOKENS_PER_RUN))
        + effect_dz * math.sqrt(2.0) * RUN_SD
    )
    return control, candidate


PAIRED_ESTIMATORS: tuple[str, ...] = (
    "paired_t", "paired_permutation",
    "bootstrap_percentile", "bootstrap_bca", "bootstrap_studentized",
)
UNPAIRED_ESTIMATORS: tuple[str, ...] = ("welch_t", "unpaired_permutation")
HIERARCHICAL_ESTIMATORS: tuple[str, ...] = (
    "hierarchical_seed_t", "hierarchical_cluster_bootstrap", "pooled_cells_t",
)
FEATURE_ESTIMATORS: tuple[str, ...] = ("feature_permutation",)
GRID_ESTIMATORS: tuple[str, ...] = ("fdr_bh", "fdr_by", "uncorrected_grid")
TOKEN_ESTIMATORS: tuple[str, ...] = ("pooled_tokens_t",)


def _paired_verdicts(draw: _PairedDraw, *, alpha: float, resamples: int, rng) -> dict[str, bool]:
    """Every paired estimator, through the exported API, on one replicate."""
    verdicts = {
        "paired_t": paired_test(draw.control, draw.candidate, "m", test="paired_t",
                                alpha=alpha).significant,
        "paired_permutation": paired_test(draw.control, draw.candidate, "m",
                                          test="paired_permutation", alpha=alpha,
                                          resamples=resamples, rng=rng).significant,
    }
    for method in ("percentile", "bca", "studentized"):
        effect = paired_effect(draw.control, draw.candidate, "m", ci_method=method,
                               resamples=resamples, rng=rng)
        verdicts[f"bootstrap_{method}"] = effect.excludes_zero
    return verdicts


def _unpaired_verdicts(draw: _PairedDraw, *, alpha: float, resamples: int, rng) -> dict[str, bool]:
    return {
        "welch_t": unpaired_test(draw.control, draw.candidate, "m", test="welch_t",
                                 alpha=alpha).significant,
        "unpaired_permutation": unpaired_test(draw.control, draw.candidate, "m",
                                              test="unpaired_permutation", alpha=alpha,
                                              resamples=resamples, rng=rng).significant,
    }


def _hierarchical_verdicts(
    draw: _PairedDraw, differences: np.ndarray, *, alpha: float, resamples: int, rng
) -> dict[str, bool]:
    effect = hierarchical_effect(draw.control, draw.candidate, "m", alpha=alpha,
                                 resamples=resamples, rng=rng)
    # pooled_cells_t is the forbidden analysis and is deliberately computed with the
    # private kernel: there is no route to it through the exported API, because the
    # exported API reduces cells to seeds before it tests anything. It is calibrated
    # here so that the size of the mistake is on the record rather than in a warning.
    pooled = _paired_t_test(differences.ravel(), alpha=alpha)
    return {
        "hierarchical_seed_t": effect.test.significant,
        "hierarchical_cluster_bootstrap": (
            effect.cluster_bootstrap_ci[0] > 0.0 or effect.cluster_bootstrap_ci[1] < 0.0
        ),
        "pooled_cells_t": pooled.significant,
    }


def _grid_verdicts(differences: np.ndarray, *, alpha: float) -> tuple[dict[str, bool], dict]:
    tests = [_paired_t_test(differences[:, site], alpha=alpha) for site in range(differences.shape[1])]
    labels = [f"site{site}" for site in range(differences.shape[1])]
    bh = fdr_over_tests(tests, alpha=alpha, method="bh", labels=labels)
    by = fdr_over_tests(tests, alpha=alpha, method="by", labels=labels)
    uncorrected = [test.significant for test in tests]
    return (
        {
            "fdr_bh": bh.n_rejected > 0,
            "fdr_by": by.n_rejected > 0,
            "uncorrected_grid": any(uncorrected),
        },
        {"bh": bh, "by": by, "uncorrected": uncorrected},
    )


@dataclass(frozen=True)
class NullCalibration:
    """How often each estimator declares an effect when there is none."""

    replicates: int
    alpha: float
    resamples: int
    seed_counts: tuple[int, ...]
    rows: tuple[dict, ...]

    def rate(self, estimator: str, n: int, *, null: str | None = None,
             condition: str | None = None) -> float:
        matching = [
            row for row in self.rows
            if row["estimator"] == estimator and row["n"] == n
            and (null is None or row["null_model"] == null)
            and (condition is None or row["condition"] == condition)
        ]
        if not matching:
            raise StatisticsError(f"no null calibration row for {estimator} at n={n}")
        return max(row["false_positive_rate"] for row in matching)

    def worst_at(self, estimator: str, n: int) -> dict:
        """The unluckiest primary-condition row: the number the record must survive.

        An estimator is only as trustworthy as its worst noise shape. Reporting
        the gaussian column alone would be choosing the assumption that flatters
        the test, which is §7.4's complaint about selecting the best seed applied
        one level up.
        """
        matching = [
            row for row in self.rows
            if row["estimator"] == estimator and row["n"] == n and row["primary_condition"]
        ]
        if not matching:
            raise StatisticsError(f"no primary-condition null row for {estimator} at n={n}")
        return max(matching, key=lambda row: row["false_positive_rate"])

    def as_dict(self) -> dict:
        return {
            "replicates": self.replicates,
            "alpha": self.alpha,
            "resamples_per_replicate": self.resamples,
            "seed_counts": list(self.seed_counts),
            "rows": [dict(row) for row in self.rows],
        }


def _row(estimator, null_model, condition, n, hits, replicates, alpha, *,
         unit: str = "seed", primary: bool = True) -> dict:
    """One calibration cell.

    ``n`` counts whatever the estimator's own replication unit is, named in
    ``unit``. For everything run-level that is seeds; for the feature permutation
    it is features, because that test replicates over features inside one run and
    calling those "seeds" would be the confusion the whole module exists to
    prevent.
    """
    low, high = _wilson_interval(hits, replicates)
    return {
        "estimator": estimator,
        "null_model": null_model,
        "condition": condition,
        "unit": unit,
        "n": int(n),
        "replicates": replicates,
        "n_significant": int(hits),
        "false_positive_rate": hits / replicates,
        "ci_low": low,
        "ci_high": high,
        "nominal_alpha": alpha,
        "primary_condition": bool(primary),
    }


def null_calibration(
    *,
    replicates: int = NULL_REPLICATES,
    alpha: float = ALPHA,
    seed_counts: Sequence[int] = SEED_COUNTS,
    resamples: int = CALIBRATION_RESAMPLES,
    seed: int = DEFAULT_RNG_SEED,
) -> NullCalibration:
    """Measure every estimator's false-positive rate where the truth is *nothing*.

    Both arms are drawn from the same distribution: no effect, at any seed count,
    under any of the three noise shapes. Anything an estimator declares here it
    declared out of noise, and the fraction of replicates on which it does so is
    the number that decides whether it can be believed at five seeds.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for n_seeds in seed_counts:
        for null in NULL_MODELS:
            paired_hits = dict.fromkeys(PAIRED_ESTIMATORS, 0)
            unpaired_hits = dict.fromkeys(UNPAIRED_ESTIMATORS, 0)
            hierarchical_hits = dict.fromkeys(HIERARCHICAL_ESTIMATORS, 0)
            token_hits = dict.fromkeys(TOKEN_ESTIMATORS, 0)
            for _ in range(replicates):
                draw = _draw_paired(rng, n_seeds=n_seeds, null=null, effect_dz=0.0)
                for name, hit in _paired_verdicts(draw, alpha=alpha, resamples=resamples, rng=rng).items():
                    paired_hits[name] += bool(hit)

                unmatched = _draw_paired(rng, n_seeds=n_seeds, null=null, effect_dz=0.0, matched=False)
                for name, hit in _unpaired_verdicts(unmatched, alpha=alpha, resamples=resamples,
                                                    rng=rng).items():
                    unpaired_hits[name] += bool(hit)

                grid, differences = _draw_hierarchical(rng, n_seeds=n_seeds, n_cells=4, null=null,
                                                       effect_dz=0.0)
                for name, hit in _hierarchical_verdicts(grid, differences, alpha=alpha,
                                                        resamples=resamples, rng=rng).items():
                    hierarchical_hits[name] += bool(hit)

                control, candidate = _draw_tokens(rng, n_seeds=n_seeds, null=null, effect_dz=0.0)
                # The §7.4 sin, performed exactly: every token an independent sample.
                token_hits["pooled_tokens_t"] += bool(
                    _welch_t_test(control.ravel(), candidate.ravel(), alpha=alpha).significant
                )

            for name, hits in paired_hits.items():
                rows.append(_row(name, null, "matched_arms", n_seeds, hits, replicates, alpha))
            for name, hits in unpaired_hits.items():
                rows.append(_row(name, null, "unmatched_arms", n_seeds, hits, replicates, alpha))
            for name, hits in hierarchical_hits.items():
                rows.append(_row(name, null, "seed_x_cell_4", n_seeds, hits, replicates, alpha))
            for name, hits in token_hits.items():
                rows.append(_row(name, null, f"{TOKENS_PER_RUN}_tokens_per_run", n_seeds, hits,
                                 replicates, alpha))

        # A mechanism-site grid's assumption is not about the shape of the noise —
        # the run-level rows already say what that costs — but about whether the
        # sites are independent tests. They are not: they share a model and a seed.
        for label, correlation in (("independent_sites", 0.0),
                                   ("correlated_sites_0.5", SITE_CORRELATION)):
            grid_hits = dict.fromkeys(GRID_ESTIMATORS, 0)
            for _ in range(replicates):
                differences = _draw_grid(rng, n_seeds=n_seeds, n_sites=GRID_SITES,
                                         correlation=correlation, effect_dz=0.0, n_true=0)
                verdicts, _ = _grid_verdicts(differences, alpha=alpha)
                for name, hit in verdicts.items():
                    grid_hits[name] += bool(hit)
            for name, hits in grid_hits.items():
                rows.append(_row(name, "gaussian", label, n_seeds, hits, replicates, alpha,
                                 primary=(correlation == 0.0)))

    # The feature permutation replicates over features inside one run, so its size
    # axis is the feature bank and not the seed set. Its assumption is independence
    # between features, which superposition is the deliberate violation of.
    for n_features in FEATURE_COUNTS:
        for label, correlation in (("independent_features", 0.0),
                                   ("equicorrelated_0.3", FEATURE_CORRELATION)):
            hits = 0
            for _ in range(replicates):
                a, b = _draw_feature_arms(rng, n_features=n_features, correlation=correlation,
                                          effect=0.0)
                hits += bool(
                    feature_permutation_test(a, b, run_id="synthetic", design="paired",
                                             alpha=alpha, resamples=resamples, rng=rng).significant
                )
            rows.append(_row("feature_permutation", "gaussian", label, n_features, hits, replicates,
                             alpha, unit="feature", primary=(correlation == 0.0)))

    return NullCalibration(
        replicates=replicates,
        alpha=alpha,
        resamples=resamples,
        seed_counts=tuple(seed_counts),
        rows=tuple(rows),
    )


@dataclass(frozen=True)
class PowerCalibration:
    """How often each estimator finds an effect that is really there."""

    replicates: int
    alpha: float
    resamples: int
    seed_counts: tuple[int, ...]
    effect_sizes: tuple[float, ...]
    null_model: str
    rows: tuple[dict, ...]
    grid_rows: tuple[dict, ...]

    def power(self, estimator: str, n_seeds: int, effect_dz: float) -> float:
        for row in self.rows:
            if (row["estimator"] == estimator and row["n_seeds"] == n_seeds
                    and math.isclose(row["effect_dz"], effect_dz)):
                return row["power"]
        raise StatisticsError(f"no power row for {estimator} at n={n_seeds}, dz={effect_dz}")

    def as_dict(self) -> dict:
        return {
            "replicates": self.replicates,
            "alpha": self.alpha,
            "resamples_per_replicate": self.resamples,
            "seed_counts": list(self.seed_counts),
            "effect_sizes_dz": list(self.effect_sizes),
            "null_model": self.null_model,
            "effect_units": "standard deviations of the paired run-to-run difference",
            "rows": [dict(row) for row in self.rows],
            "mechanism_grid": [dict(row) for row in self.grid_rows],
        }


def power_calibration(
    *,
    replicates: int = POWER_REPLICATES,
    alpha: float = ALPHA,
    seed_counts: Sequence[int] = SEED_COUNTS,
    effect_sizes: Sequence[float] = EFFECT_SIZES,
    resamples: int = CALIBRATION_RESAMPLES,
    null: str = "gaussian",
    seed: int = DEFAULT_RNG_SEED + 1,
) -> PowerCalibration:
    """Measure detection rates against an effect of declared size.

    The same generator as the null, with a difference injected. ``effect_dz`` is
    in standard deviations of the paired run-to-run difference, so the unpaired
    column is lower at the same ``dz`` for a real reason: it is paying for the
    between-seed variation that pairing removes, not being tested on a smaller
    effect.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    names = PAIRED_ESTIMATORS + UNPAIRED_ESTIMATORS + HIERARCHICAL_ESTIMATORS[:2]

    for n_seeds in seed_counts:
        for effect in effect_sizes:
            hits = dict.fromkeys(names, 0)
            for _ in range(replicates):
                draw = _draw_paired(rng, n_seeds=n_seeds, null=null, effect_dz=effect)
                for name, hit in _paired_verdicts(draw, alpha=alpha, resamples=resamples, rng=rng).items():
                    hits[name] += bool(hit)
                unmatched = _draw_paired(rng, n_seeds=n_seeds, null=null, effect_dz=effect,
                                         matched=False)
                for name, hit in _unpaired_verdicts(unmatched, alpha=alpha, resamples=resamples,
                                                    rng=rng).items():
                    hits[name] += bool(hit)
                grid, differences = _draw_hierarchical(rng, n_seeds=n_seeds, n_cells=4, null=null,
                                                       effect_dz=effect)
                verdicts = _hierarchical_verdicts(grid, differences, alpha=alpha,
                                                  resamples=resamples, rng=rng)
                for name in HIERARCHICAL_ESTIMATORS[:2]:
                    hits[name] += bool(verdicts[name])
            for name, count in hits.items():
                low, high = _wilson_interval(count, replicates)
                rows.append({
                    "estimator": name,
                    "n_seeds": n_seeds,
                    "effect_dz": float(effect),
                    "replicates": replicates,
                    "n_significant": int(count),
                    "power": count / replicates,
                    "ci_low": low,
                    "ci_high": high,
                })

    grid_rows: list[dict] = []
    n_true = 4
    for n_seeds in seed_counts:
        for effect in (1.0, 1.5, 2.0):
            detected = 0
            false_discovery = 0.0
            for _ in range(replicates):
                differences = _draw_grid(rng, n_seeds=n_seeds, n_sites=GRID_SITES,
                                         correlation=SITE_CORRELATION, effect_dz=effect,
                                         n_true=n_true)
                _, detail = _grid_verdicts(differences, alpha=alpha)
                rejected = np.array(detail["bh"].rejected)
                detected += int(rejected[:n_true].sum())
                discoveries = int(rejected.sum())
                false_discovery += (int(rejected[n_true:].sum()) / discoveries) if discoveries else 0.0
            grid_rows.append({
                "method": "fdr_bh",
                "n_seeds": n_seeds,
                "effect_dz": float(effect),
                "n_sites": GRID_SITES,
                "n_true_sites": n_true,
                "per_site_power": detected / (replicates * n_true),
                "realised_fdr": false_discovery / replicates,
                "replicates": replicates,
            })

    return PowerCalibration(
        replicates=replicates,
        alpha=alpha,
        resamples=resamples,
        seed_counts=tuple(seed_counts),
        effect_sizes=tuple(float(effect) for effect in effect_sizes),
        null_model=null,
        rows=tuple(rows),
        grid_rows=tuple(grid_rows),
    )


def _simulate_power(
    *, n_seeds: int, effect_dz: float, replicates: int, alpha: float, null: str, rng
) -> float:
    hits = 0
    for _ in range(replicates):
        draw = _draw_paired(rng, n_seeds=n_seeds, null=null, effect_dz=effect_dz)
        hits += bool(paired_test(draw.control, draw.candidate, "m", test=PRIMARY_TEST,
                                 alpha=alpha).significant)
    return hits / replicates


def minimum_detectable_effect(
    *,
    n_seeds: int = 5,
    target_power: float = TARGET_POWER,
    alpha: float = ALPHA,
    replicates: int = 4000,
    null: str = "gaussian",
    tolerance: float = 0.02,
    seed: int = DEFAULT_RNG_SEED + 2,
) -> dict:
    """The smallest effect the adopted primary test finds at ``target_power``.

    Bisection on ``dz`` against a simulation of the *shipped* test, not against a
    formula for an idealised one. The number this returns at five seeds is the
    honest answer to "does your null result mean no effect": below it, it does
    not.
    """
    rng = np.random.default_rng(seed)
    low, high = 0.0, 6.0
    while high - low > tolerance:
        middle = 0.5 * (low + high)
        if _simulate_power(n_seeds=n_seeds, effect_dz=middle, replicates=replicates,
                           alpha=alpha, null=null, rng=rng) < target_power:
            low = middle
        else:
            high = middle
    effect = 0.5 * (low + high)
    achieved = _simulate_power(n_seeds=n_seeds, effect_dz=effect, replicates=replicates,
                               alpha=alpha, null=null, rng=rng)
    return {
        "test": PRIMARY_TEST,
        "n_seeds": n_seeds,
        "target_power": target_power,
        "alpha": alpha,
        "null_model": null,
        "replicates": replicates,
        "minimum_detectable_effect_dz": float(effect),
        "achieved_power": achieved,
        "units": "standard deviations of the paired run-to-run difference",
    }


def seeds_for_power(
    effect_dz: float,
    *,
    target_power: float = TARGET_POWER,
    alpha: float = ALPHA,
    replicates: int = 2000,
    null: str = "gaussian",
    max_seeds: int = 200,
    seed: int = DEFAULT_RNG_SEED + 3,
) -> dict:
    """How many seeds the adopted primary test needs to see an effect of this size.

    Doubling search then bisection, on the shipped test. If the answer is much
    larger than five, every later mission needs it *before* it designs a
    comparison rather than after it fails to find one.
    """
    rng = np.random.default_rng(seed)

    def power_at(n: int) -> float:
        return _simulate_power(n_seeds=n, effect_dz=effect_dz, replicates=replicates,
                               alpha=alpha, null=null, rng=rng)

    n = 3
    while n <= max_seeds and power_at(n) < target_power:
        n *= 2
    if n > max_seeds:
        return {"effect_dz": float(effect_dz), "target_power": target_power,
                "seeds_required": None, "searched_to": max_seeds}
    low, high = max(3, n // 2), n
    while high - low > 1:
        middle = (low + high) // 2
        if power_at(middle) < target_power:
            low = middle
        else:
            high = middle
    return {
        "effect_dz": float(effect_dz),
        "target_power": target_power,
        "alpha": alpha,
        "null_model": null,
        "replicates": replicates,
        "seeds_required": int(high),
        "power_at_required": power_at(high),
    }


# --------------------------------------------------------------------------- #
# The rule: an estimator's status, re-derived from evidence
# --------------------------------------------------------------------------- #

_POWER_ESTIMATORS = frozenset(PAIRED_ESTIMATORS + UNPAIRED_ESTIMATORS + HIERARCHICAL_ESTIMATORS[:2])


def _operating_point(name: str, null: NullCalibration) -> int:
    units = {row["unit"] for row in null.rows if row["estimator"] == name}
    return FEATURE_OPERATING_POINT if units == {"feature"} else FIVE


def estimator_rule(name: str, null: NullCalibration, power: PowerCalibration | None) -> tuple[str, str]:
    """Re-derive one estimator's status from a calibration, and say why.

    The decision is made on the *interval* around the measured rate rather than on
    the point estimate, so the same rule holds at the selftest's three hundred
    replicates and at the recorded run's two thousand: an estimator is demoted
    only when the evidence puts it confidently above tolerance, and a forbidden
    one is only vindicated when the evidence puts it confidently at level.
    """
    row = null.worst_at(name, _operating_point(name, null))
    rate, low = row["false_positive_rate"], row["ci_low"]
    confidently_bad = low > LEVEL_TOLERANCE
    where = f"{rate:.3f} [{low:.3f}, {row['ci_high']:.3f}] under {row['null_model']}/{row['condition']}"

    if name in FORBIDDEN_ESTIMATORS:
        if confidently_bad:
            return "forbidden", f"still {rate / ALPHA:.1f}x nominal: {where}"
        return "level_holding", (
            f"a forbidden analysis is no longer confidently above tolerance: {where}. Either "
            "the calibration stopped measuring what it claims to, or the demonstration is gone"
        )
    if confidently_bad:
        return "descriptive_only", f"realised level above {LEVEL_TOLERANCE}: {where}"
    if power is not None and name in _POWER_ESTIMATORS:
        largest = max(power.effect_sizes)
        detected = power.power(name, FIVE, largest)
        if detected <= 0.5:
            return "unusable_at_five_seeds", (
                f"power {detected:.3f} at dz = {largest} and five seeds — it cannot reject "
                f"there whatever the data; level {where}"
            )
    return "level_holding", f"level {where}"


@dataclass(frozen=True)
class CalibrationReport:
    """Both calibrations, the derived statuses, and whether they match the record."""

    statistics_version: str
    null: NullCalibration
    power: PowerCalibration
    minimum_detectable_effects: tuple[dict, ...]
    seed_requirements: tuple[dict, ...]
    measures: tuple[dict, ...]

    @property
    def ok(self) -> bool:
        return all(row["agrees"] for row in self.measures)

    def as_dict(self) -> dict:
        return {
            "statistics_version": self.statistics_version,
            "alpha": ALPHA,
            "ci_level": CI_LEVEL,
            "level_tolerance": LEVEL_TOLERANCE,
            "adopted": dict(ADOPTED),
            "thresholds": dict(THRESHOLDS),
            "null_calibration": self.null.as_dict(),
            "power_calibration": self.power.as_dict(),
            "minimum_detectable_effect": [dict(row) for row in self.minimum_detectable_effects],
            "seeds_for_power": [dict(row) for row in self.seed_requirements],
            "estimators": [dict(row) for row in self.measures],
            "ok": self.ok,
        }


def calibrate(
    *,
    null_replicates: int = NULL_REPLICATES,
    power_replicates: int = POWER_REPLICATES,
    mde_replicates: int = 8000,
    seed_search_replicates: int = 2000,
    resamples: int = CALIBRATION_RESAMPLES,
    alpha: float = ALPHA,
    seed_counts: Sequence[int] = SEED_COUNTS,
) -> CalibrationReport:
    """Both calibrations plus the numbers every later mission needs from them."""
    null = null_calibration(replicates=null_replicates, alpha=alpha, seed_counts=seed_counts,
                            resamples=resamples)
    power = power_calibration(replicates=power_replicates, alpha=alpha, seed_counts=seed_counts,
                              resamples=resamples)
    detectable = tuple(
        minimum_detectable_effect(n_seeds=n, alpha=alpha, replicates=mde_replicates)
        for n in seed_counts
    )
    required = tuple(
        seeds_for_power(effect, alpha=alpha, replicates=seed_search_replicates)
        for effect in INTERESTING_EFFECTS
    )

    measures = []
    for spec in ESTIMATOR_SPECS:
        derived, reason = estimator_rule(spec.name, null, power)
        row = null.worst_at(spec.name, _operating_point(spec.name, null))
        measures.append({
            "name": spec.name,
            "family": spec.family,
            "recorded_status": spec.status,
            "derived_status": derived,
            "agrees": derived == spec.status,
            "why": reason,
            "measured_fpr": row["false_positive_rate"],
            "measured_ci": [row["ci_low"], row["ci_high"]],
            "recorded_fpr_at_5": spec.recorded_fpr_at_5,
            "operating_point": row["n"],
            "unit": row["unit"],
        })

    return CalibrationReport(
        statistics_version=STATISTICS_VERSION,
        null=null,
        power=power,
        minimum_detectable_effects=detectable,
        seed_requirements=required,
        measures=tuple(measures),
    )


# --------------------------------------------------------------------------- #
# Demonstrations the selftest turns into numbers
# --------------------------------------------------------------------------- #


def _experimental_unit_is_enforced() -> dict:
    """Show the boundary refuses each shape of the pooling mistake, and its cost.

    The refusals are the mechanism; the width ratio is why it is worth having. The
    same five runs, analysed correctly and analysed by pooling their tokens, give
    intervals differing by a factor this function measures rather than asserts.
    """
    refusals: dict[str, bool] = {}

    try:
        RunSummary(run_id="r", seed=0, arm="a", metrics={"accuracy": np.zeros(4096)})
    except ExperimentalUnitError:
        refusals["array_valued_metric"] = True

    try:
        paired_test(np.zeros(5), np.ones(5), "accuracy")
    except ExperimentalUnitError:
        refusals["bare_array_of_runs"] = True

    try:
        paired_test([0.1] * 5, [0.2] * 5, "accuracy")
    except ExperimentalUnitError:
        refusals["list_of_floats"] = True

    duplicated = [
        RunSummary(run_id=f"r{index}", seed=7, arm="a", metrics={"accuracy": 0.5})
        for index in range(3)
    ]
    try:
        paired_test(duplicated, duplicated, "accuracy")
    except ExperimentalUnitError:
        refusals["repeated_seed"] = True

    try:
        bootstrap_ci(np.zeros(4096), unit="run")
    except ExperimentalUnitError:
        refusals["bootstrap_over_4096_runs"] = True

    try:
        bootstrap_ci(np.zeros(8))  # type: ignore[call-arg]
    except TypeError:
        refusals["bootstrap_without_a_named_unit"] = True

    rng = np.random.default_rng(20260810)
    n_seeds, n_tokens = 5, 64
    run_offset = rng.standard_normal((n_seeds, 1))
    tokens = run_offset + rng.standard_normal((n_seeds, n_tokens))
    per_run = tokens.mean(axis=1)
    honest = float(per_run.std(ddof=1)) / math.sqrt(n_seeds)
    pooled = float(tokens.std(ddof=1)) / math.sqrt(n_seeds * n_tokens)
    return {
        "refusals": refusals,
        "all_refused": len(refusals) == 6,
        "honest_standard_error": honest,
        "token_pooled_standard_error": pooled,
        "narrowing_factor": honest / pooled if pooled else float("inf"),
    }


def _primary_metric_comes_from_the_packet() -> dict:
    """Show the metric can only arrive from a committed packet, never from a caller."""
    import inspect
    import tempfile

    from architecture_mechanics.experiments.claim_packet import REQUIRED_FIELDS, ClaimPacket

    signature = inspect.signature(primary_comparison)
    accepts_metric = "metric" in signature.parameters

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        claims = root / "claims"
        packet = ClaimPacket(
            claim_id="synthetic-claim",
            claimed_rung=2,
            fields={name: f"synthetic {name} for the statistics selftest" for name in REQUIRED_FIELDS},
            primary_metric_key="associative_recall_accuracy",
        )
        packet.write(claims / "synthetic-claim.yml")

        agreeing = root / "agreeing.json"
        agreeing.write_text(json.dumps({
            "claim": "synthetic-claim",
            "control_run": "control",
            "candidate_runs": ["candidate"],
            "primary_metric": "associative_recall_accuracy",
        }))
        disagreeing = root / "disagreeing.json"
        disagreeing.write_text(json.dumps({
            "claim": "synthetic-claim",
            "control_run": "control",
            "candidate_runs": ["candidate"],
            "primary_metric": "reconstruction_loss",
        }))
        unclaimed = root / "unclaimed.json"
        unclaimed.write_text(json.dumps({
            "control_run": "control", "candidate_runs": ["candidate"],
            "primary_metric": "associative_recall_accuracy",
        }))

        metric, source = primary_metric_for(load_comparison(agreeing), claims_dir=claims)

        renamed_refused = False
        try:
            primary_metric_for(load_comparison(disagreeing), claims_dir=claims)
        except StatisticsError:
            renamed_refused = True

        unclaimed_refused = False
        try:
            load_comparison(unclaimed)
        except StatisticsError:
            unclaimed_refused = True

    return {
        "accepts_metric_argument": accepts_metric,
        "metric": metric,
        "source_is_the_packet": source.endswith("#primary_metric_key"),
        "renamed_metric_refused": renamed_refused,
        "comparison_without_a_claim_refused": unclaimed_refused,
    }


def _comparison_record_round_trips() -> dict:
    """A record survives ``summary.json`` unchanged, which is prompts 22 and 27's contract."""
    rng = np.random.default_rng(4)
    draw = _draw_paired(rng, n_seeds=5, null="gaussian", effect_dz=1.5)
    record = secondary_comparison(
        "selftest", draw.control, draw.candidate, "m", claim_id="synthetic-claim", resamples=500
    )
    summary = attach_comparisons({"run_id": "selftest", "final": {"m": 0.9}}, [record])
    text = json.dumps(summary, indent=2, default=_json_default)
    restored = comparisons_from_summary(json.loads(text))
    rejected_dict = False
    try:
        attach_comparisons({}, [record.as_dict()])
    except StatisticsError:
        rejected_dict = True
    return {
        "round_trips": len(restored) == 1 and restored[0].as_dict() == record.as_dict(),
        "carries_per_seed_values": len(record.per_seed_difference) == 5 and len(record.per_run) == 10,
        "carries_effect_ci_test_and_count": all((
            record.effect.ci_method, record.test.test, record.n_seeds == 5,
            math.isfinite(record.effect.estimate),
        )),
        "raw_dict_refused": rejected_dict,
        "json_bytes": len(text),
    }


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #


def format_null_table(null: NullCalibration) -> str:
    title = (f"false-positive rate under the null — {null.replicates} replicates, "
             f"alpha {null.alpha}")
    rows = ["", title, ""]
    header = f"{'estimator':32s} {'condition':22s} {'unit':7s} {'n':>3s}  "
    header += "  ".join(f"{name:>12s}" for name in NULL_MODELS)
    rows.append(header)
    rows.append("-" * len(header))
    seen: set[tuple] = set()
    for row in null.rows:
        key = (row["estimator"], row["condition"], row["n"])
        if key in seen:
            continue
        seen.add(key)
        values = {
            other["null_model"]: other["false_positive_rate"]
            for other in null.rows
            if (other["estimator"], other["condition"], other["n"]) == key
        }
        cells = "  ".join(
            f"{values[name]:12.3f}" if name in values else f"{'—':>12s}" for name in NULL_MODELS
        )
        rows.append(
            f"{row['estimator']:32s} {row['condition']:22s} {row['unit']:7s} {row['n']:3d}  {cells}"
        )
    return "\n".join(rows)


def format_power_table(power: PowerCalibration) -> str:
    title = (f"power against a known effect — {power.replicates} replicates, "
             f"{power.null_model} noise, dz in standard deviations of the paired "
             "run-to-run difference")
    rows = ["", title, ""]
    for n_seeds in power.seed_counts:
        header = f"{'estimator':32s}" + "".join(f"{effect:>8}" for effect in power.effect_sizes)
        rows.append(f"n = {n_seeds} seeds")
        rows.append(header)
        rows.append("-" * len(header))
        names: list[str] = []
        for row in power.rows:
            if row["estimator"] not in names:
                names.append(row["estimator"])
        for name in names:
            line = f"{name:32s}"
            for effect in power.effect_sizes:
                line += f"{power.power(name, n_seeds, effect):8.3f}"
            level = ESTIMATOR_SPEC_BY_NAME[name].status
            rows.append(line + ("" if level == "level_holding" else f"   ({level})"))
        rows.append("")
    return "\n".join(rows)


def format_estimator_table(report: CalibrationReport) -> str:
    rows = ["", "estimators — recorded status against the status the calibration derives", ""]
    header = f"{'estimator':32s} {'recorded':26s} {'derived':26s} {'fpr':>7s} {'agrees':>7s}"
    rows.append(header)
    rows.append("-" * len(header))
    for row in report.measures:
        rows.append(
            f"{row['name']:32s} {row['recorded_status']:26s} {row['derived_status']:26s} "
            f"{row['measured_fpr']:7.3f} {row['agrees']!s:>7s}"
        )
    return "\n".join(rows)


# --------------------------------------------------------------------------- #
# Selftest
# --------------------------------------------------------------------------- #

INVARIANTS: tuple[str, ...] = (
    "special_functions_match_published_values",
    "experimental_unit_is_enforced",
    "null_false_positive_rates_are_within_tolerance",
    "estimator_status_agrees_with_the_rule",
    "permutation_floor_blocks_five_seeds",
    "forbidden_analyses_are_still_forbidden",
    "power_rises_with_effect_and_with_seeds",
    "minimum_detectable_effect_reproduces",
    "primary_metric_comes_from_the_packet",
    "comparison_record_round_trips_through_summary_json",
)


class _Checks:
    """Collects pass/fail with a reason, so the selftest reports every failure.

    A local copy of the capability and geometry gates' helper rather than an
    import of either private class, on the same reasoning: fifteen duplicated
    lines are cheaper than a cross-module private dependency between three
    independent gates.
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
        return [result for result in self.results if not result[1]]


MDE_TOLERANCE = 0.45
"""How far the selftest's low-replicate estimate of the minimum detectable effect
may sit from the recorded one. Bisection against a Monte-Carlo power curve is
noisy at eight hundred replicates; 0.45 dz is wide enough not to fail for that
reason and narrow enough that a real change in the estimator moves it out."""


def run_selftest(
    *,
    break_invariant: str | None = None,
    null_replicates: int = SELFTEST_NULL_REPLICATES,
    power_replicates: int = SELFTEST_POWER_REPLICATES,
    verbose: bool = True,
) -> int:
    """Re-run both calibrations and check every recorded decision against them."""
    checks = _Checks(break_invariant)
    report = calibrate(
        null_replicates=null_replicates,
        power_replicates=power_replicates,
        mde_replicates=800,
        seed_search_replicates=400,
    )
    out: list[str] = [f"statistics selftest — {STATISTICS_VERSION}", ""]

    published = {
        "t(4) two-sided p at 2.776445": (_student_t_two_sided_p(2.776445, 4), 0.05),
        "t(9) two-sided p at 2.262157": (_student_t_two_sided_p(2.262157, 9), 0.05),
        "normal quantile at 0.975": (float(_norm_ppf(0.975)), 1.959964),
        "normal cdf at 1.959964": (float(_norm_cdf(1.959964)), 0.975),
    }
    worst = max(abs(value - target) for value, target in published.values())
    checks.record(
        "special_functions_match_published_values",
        worst < 1e-5,
        "; ".join(f"{name} = {value:.6f} (expected {target})"
                  for name, (value, target) in published.items()),
    )

    unit = _experimental_unit_is_enforced()
    checks.record(
        "experimental_unit_is_enforced",
        unit["all_refused"] and unit["narrowing_factor"] > 4.0,
        f"refused {sorted(unit['refusals'])}; pooling {TOKENS_PER_RUN} tokens per run narrows "
        f"the standard error from {unit['honest_standard_error']:.4f} to "
        f"{unit['token_pooled_standard_error']:.4f}, a factor of {unit['narrowing_factor']:.1f}",
    )

    should_hold = [
        row for row in report.measures
        if row["recorded_status"] in {"level_holding", "unusable_at_five_seeds"}
    ]
    breaches = [
        f"{row['name']} {row['measured_fpr']:.3f} CI[{row['measured_ci'][0]:.3f}, "
        f"{row['measured_ci'][1]:.3f}]"
        for row in should_hold
        if row["measured_ci"][0] > LEVEL_TOLERANCE
    ]
    checks.record(
        "null_false_positive_rates_are_within_tolerance",
        not breaches,
        f"{len(should_hold)} estimators recorded as holding their level at "
        f"{null_replicates} replicates; worst measured "
        f"{max(row['measured_fpr'] for row in should_hold):.3f} against a tolerance of "
        f"{LEVEL_TOLERANCE}; breaches: {breaches or 'none'}",
    )

    disagreements = [
        f"{row['name']}: recorded {row['recorded_status']}, derived {row['derived_status']} "
        f"({row['why']})"
        for row in report.measures if not row["agrees"]
    ]
    checks.record(
        "estimator_status_agrees_with_the_rule",
        not disagreements,
        f"{len(report.measures)} estimators; disagreements: {disagreements or 'none'}",
    )

    rng = np.random.default_rng(11)
    floors = {}
    for n_seeds in (3, 5, 6, 10):
        draw = _draw_paired(rng, n_seeds=n_seeds, null="gaussian", effect_dz=50.0)
        result = paired_test(draw.control, draw.candidate, "m", test="paired_permutation")
        floors[n_seeds] = (result.p_value_floor, result.p_value, result.power_is_attainable)
    checks.record(
        "permutation_floor_blocks_five_seeds",
        floors[3][0] == 0.25 and floors[5][0] == 0.0625 and not floors[5][2]
        and math.isclose(floors[6][0], 2 / 64) and floors[6][2] and floors[10][2],
        "; ".join(
            f"n={n}: floor {floor:.4f}, p on a huge effect {p:.4f}, can reject {attainable}"
            for n, (floor, p, attainable) in floors.items()
        ),
    )

    forbidden = [row for row in report.measures if row["name"] in FORBIDDEN_ESTIMATORS]
    still_broken = [row["measured_ci"][0] > LEVEL_TOLERANCE for row in forbidden]
    checks.record(
        "forbidden_analyses_are_still_forbidden",
        len(forbidden) == len(FORBIDDEN_ESTIMATORS) and all(still_broken),
        "; ".join(
            f"{row['name']} {row['measured_fpr']:.3f} = {row['measured_fpr'] / ALPHA:.1f}x nominal"
            for row in forbidden
        ),
    )

    primary = ADOPTED["paired_comparison"]
    # Three Monte-Carlo standard errors of a proportion. The invariant is that the
    # power curve rises; at eighty replicates a neighbouring pair can cross by
    # chance, and a slack that ignored the replicate count would either fail on
    # noise in the selftest or pass a flat curve in the recorded run.
    slack = 1.5 / math.sqrt(report.power.replicates)
    monotone_in_effect = all(
        report.power.power(primary, n, small) <= report.power.power(primary, n, large) + slack
        for n in report.power.seed_counts
        for small, large in itertools.pairwise(report.power.effect_sizes)
    )
    monotone_in_seeds = all(
        report.power.power(primary, 3, effect) <= report.power.power(primary, 10, effect) + slack
        for effect in report.power.effect_sizes
    )
    checks.record(
        "power_rises_with_effect_and_with_seeds",
        monotone_in_effect and monotone_in_seeds,
        f"{primary} at five seeds: "
        + ", ".join(f"dz {effect} -> {report.power.power(primary, 5, effect):.3f}"
                    for effect in report.power.effect_sizes),
    )

    at_five = next(
        row for row in report.minimum_detectable_effects if row["n_seeds"] == FIVE
    )
    recorded = float(THRESHOLDS["minimum_detectable_effect_dz_at_5_seeds"])
    drift = abs(at_five["minimum_detectable_effect_dz"] - recorded)
    checks.record(
        "minimum_detectable_effect_reproduces",
        drift <= MDE_TOLERANCE,
        f"recorded {recorded:.2f} dz, re-measured "
        f"{at_five['minimum_detectable_effect_dz']:.2f} dz at power "
        f"{at_five['achieved_power']:.3f} ({at_five['replicates']} replicates); "
        f"drift {drift:.2f} against a tolerance of {MDE_TOLERANCE}",
    )

    packet = _primary_metric_comes_from_the_packet()
    checks.record(
        "primary_metric_comes_from_the_packet",
        not packet["accepts_metric_argument"] and packet["source_is_the_packet"]
        and packet["renamed_metric_refused"] and packet["comparison_without_a_claim_refused"],
        f"primary_comparison takes no metric argument: {not packet['accepts_metric_argument']}; "
        f"metric {packet['metric']!r} read from the packet; a comparison renaming it is "
        f"refused: {packet['renamed_metric_refused']}; a comparison naming no claim is "
        f"refused: {packet['comparison_without_a_claim_refused']}",
    )

    record = _comparison_record_round_trips()
    checks.record(
        "comparison_record_round_trips_through_summary_json",
        record["round_trips"] and record["carries_per_seed_values"]
        and record["carries_effect_ci_test_and_count"] and record["raw_dict_refused"],
        f"{record['json_bytes']} bytes of summary.json restore to an identical record; "
        f"per-seed values present: {record['carries_per_seed_values']}; a bare dict is "
        f"refused: {record['raw_dict_refused']}",
    )

    out.append(format_estimator_table(report))
    out.append(format_null_table(report.null))
    out.append("")
    out.append("minimum detectable effect (paired_t, 80% power, alpha 0.05)")
    for row in report.minimum_detectable_effects:
        out.append(f"  n = {row['n_seeds']:2d} seeds: dz = "
                   f"{row['minimum_detectable_effect_dz']:.2f} "
                   f"(achieved power {row['achieved_power']:.3f})")
    out.append("")
    out.append("seeds needed for 80% power")
    for row in report.seed_requirements:
        out.append(f"  dz = {row['effect_dz']:.2f}: {row['seeds_required']} seeds")
    out.append("")
    out.append("invariants")
    for name, ok, detail in checks.results:
        out.append(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    out.append("")
    out.append("selftest PASSED" if not checks.failed else f"selftest FAILED ({len(checks.failed)})")

    if verbose:
        print("\n".join(out))
    return 0 if not checks.failed else 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="architecture_mechanics.metrics.statistics")
    parser.add_argument("--selftest", action="store_true",
                        help="re-run both calibrations at reduced replicate counts and check "
                             "every recorded decision against them")
    parser.add_argument("--break-invariant", choices=INVARIANTS, default=None,
                        help="force one invariant to fail; used to prove the gate reports failure")
    parser.add_argument("--calibrate", action="store_true",
                        help="run the full recorded calibration and print both tables")
    parser.add_argument("--null-replicates", type=int, default=None,
                        help=f"replicates per null cell (default {NULL_REPLICATES} for "
                             f"--calibrate, {SELFTEST_NULL_REPLICATES} for --selftest)")
    parser.add_argument("--power-replicates", type=int, default=None,
                        help=f"replicates per power cell (default {POWER_REPLICATES} for "
                             f"--calibrate, {SELFTEST_POWER_REPLICATES} for --selftest)")
    parser.add_argument("--json", metavar="PATH", default=None, help="write the report as JSON")
    args = parser.parse_args(argv)

    if args.calibrate:
        report = calibrate(
            null_replicates=args.null_replicates or NULL_REPLICATES,
            power_replicates=args.power_replicates or POWER_REPLICATES,
        )
        print(format_estimator_table(report))
        print(format_null_table(report.null))
        print(format_power_table(report.power))
        print("minimum detectable effect (paired_t, 80% power, alpha 0.05)")
        for row in report.minimum_detectable_effects:
            print(f"  n = {row['n_seeds']:2d} seeds: dz = "
                  f"{row['minimum_detectable_effect_dz']:.3f} "
                  f"(achieved power {row['achieved_power']:.3f})")
        print()
        print("seeds needed for 80% power")
        for row in report.seed_requirements:
            print(f"  dz = {row['effect_dz']:.2f}: {row['seeds_required']} seeds "
                  f"(power {row['power_at_required']:.3f})")
        if args.json:
            _write_json(Path(args.json), report.as_dict())
        return 0 if report.ok else 1

    if args.selftest or args.break_invariant:
        status = run_selftest(
            break_invariant=args.break_invariant,
            null_replicates=args.null_replicates or SELFTEST_NULL_REPLICATES,
            power_replicates=args.power_replicates or SELFTEST_POWER_REPLICATES,
        )
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
    if isinstance(value, MappingProxyType):
        return dict(value)
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
