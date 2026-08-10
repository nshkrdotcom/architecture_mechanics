"""The T1 ladder: A0 through §7.3's rungs, and how much A0 differs from itself.

Two things live here and nothing else.

**The declaration.** §7.3 R3 asks for "a fixed task matrix" and §4.3 says what
T1's matrix is made of: source distance, distractor count, feature sparsity, key
collisions, and simultaneous associations. :data:`DIFFICULTY_AXES` is that
matrix written down as data, one cell per level, each moving exactly one axis
away from :data:`BASE_CELL`. A curve whose cells differ in two things at once is
not a curve, so :func:`cells` refuses to build one, and
``tests/experiments/test_t1_ladder.py`` checks the property rather than trusting
the table.

**The reduction.** Recorded runs into two reports: the competence envelope
(:func:`difficulty_curves`) that prompts 13 and 20 choose an operating point
from, and the seed-to-seed spread (:func:`seed_variance`) that every later
"architecture X differs from architecture Y" has to be read against. Both read
``summary.json`` files and run no model.

This is not an experiment framework and must not become one (§13.3). It declares
one task family's matrix and reduces one architecture's runs over it; the runner
underneath is unchanged, and a second architecture needs no new code here — it
needs ``--arch``.

Order is enforced rather than suggested. ``--stage r3`` refuses to start until a
recorded R1 for this claim has passed, because §7.3 is explicit that an R1
failure is an implementation bug and not a finding, and a matrix run on a broken
instrument is sixteen ways of measuring the bug.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from architecture_mechanics.experiments.config import RunConfig, ladder_config
from architecture_mechanics.experiments.manifest import lab_root
from architecture_mechanics.metrics.statistics import (
    ALPHA,
    BOOTSTRAP_RESAMPLES,
    CI_LEVEL,
    STATISTICS_VERSION,
    RunSummary,
    bootstrap_ci,
    minimum_detectable_effect,
    seeds_for_power,
)

__all__ = [
    "BASE_CELL",
    "CAPABILITY_METRICS",
    "CLAIM",
    "DIFFICULTY_AXES",
    "GEOMETRY_METRICS",
    "MECHANISM_METRICS",
    "R2_WIDTHS",
    "R4_SEEDS",
    "R4_SEEDS_EXTENDED",
    "T1_LADDER_VERSION",
    "Cell",
    "cell_config",
    "cells",
    "difficulty_curves",
    "main",
    "run_metrics",
    "seed_variance",
]

T1_LADDER_VERSION = "t1-1.0.0"

CLAIM = "claims/a0-t1-associative-recall.yml"
"""The §7.1 pre-registration every run below is a child of. Named here rather
than passed at a call site so that no stage of this ladder can be run against a
different prediction than the one that was committed."""

BASE_CELL = "base"
"""The R3 matrix's origin: ``capacity_stressed`` exactly as §4.4 declares it,
with no axis moved. Every other cell is this cell plus one override, so a
difference between two cells is attributable to the axis that names them."""

R2_WIDTHS: tuple[int, ...] = (16, 32, 64)
"""The kill screen's width axis. 16 is the condition's own ``d_recommended``
(``F/d = 7.75``) and is the canonical R2; the other two locate where capability
collapses, which is the question "where does the baseline stop being a baseline"
and is not answerable from a single width."""

R4_SEEDS: tuple[int, ...] = (20260809, 20260810, 20260811, 20260812, 20260813)
"""§10.1's five. The first three match the seeds already used for R1, so the
laboratory's seed family stays one family."""

R4_SEEDS_EXTENDED: tuple[int, ...] = R4_SEEDS + (20260814, 20260815, 20260816)
"""Eight. Whether the five-seed interval was honest is checkable only against
seeds it did not see, and at this scale three more runs cost four minutes."""


# --------------------------------------------------------------------------- #
# The matrix
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Cell:
    """One cell of the R3 task matrix: a name, an axis, and one override."""

    name: str
    axis: str
    level: object
    overrides: dict = field(default_factory=dict)
    condition: str = "capacity_stressed"

    def as_dict(self) -> dict:
        record = asdict(self)
        record["level"] = _jsonable(self.level)
        return record


DIFFICULTY_AXES: tuple[dict, ...] = (
    {
        "axis": "source_distance",
        "field": "distance_buckets",
        "why": (
            "§4.3's first knob. The base condition draws from two buckets at once, so no "
            "single-bucket level coincides with it; the curve is read against the base cell "
            "rather than through it."
        ),
        "levels": ([[4, 6]], [[10, 16]], [[20, 26]], [[34, 40]]),
        "label": lambda value: f"d{value[0][0]}-{value[0][1]}",
    },
    {
        "axis": "distractors",
        "field": "n_distractors",
        "why": (
            "§4.3's second knob: content-bearing positions between source and destination. "
            "Bounded above by the shortest distance in the bucket — four distractors cannot "
            "fit strictly inside a gap of five — so this axis stops at 4 and not by choice."
        ),
        "levels": (0, 1, 3, 4),
        "label": lambda value: f"n{value}",
    },
    {
        "axis": "sparsity",
        "field": "activation_prob",
        "why": (
            "§4.3's third knob and the axis the superposition phase diagram is drawn "
            "against. Per-feature activation probability within the group a position draws "
            "from; the realised global density per position is far lower."
        ),
        "levels": (0.06, 0.12, 0.24, 0.40),
        "label": lambda value: f"p{value:.2f}".replace(".", ""),
    },
    {
        "axis": "key_collisions",
        "field": "key_collisions",
        "why": (
            "§4.3's fourth knob: one non-source binding carries a near-miss key sharing all "
            "but one index with the query's. Two levels, because the knob is a boolean."
        ),
        "levels": (False, True),
        "label": lambda value: "on" if value else "off",
    },
    {
        "axis": "associations",
        "field": "n_associations",
        "why": (
            "§4.3's fifth knob: how many key/value bindings are live at once. Bounded above "
            "by the key bank (12 keys, and one must stay unused so the information-destroyed "
            "control has a key nothing queries)."
        ),
        "levels": (2, 4, 6, 10),
        "label": lambda value: f"a{value}",
    },
)
"""§4.3's five T1 difficulty axes, one entry each.

``field`` names the :class:`~...data.feature_program.FeatureProgramConfig` field
the level moves. Levels equal to the base condition's own value collapse onto
:data:`BASE_CELL` rather than being run twice — the axis still passes through
that point, it is just measured once."""


def cells(*, include_base: bool = True) -> tuple[Cell, ...]:
    """Every distinct cell of the matrix, base first, deduplicated.

    A level equal to the base condition's value yields the base cell itself, so
    an axis of four levels through the base costs three runs and not four.
    Refuses a level that would move more than one field, which is the only way
    a curve here could stop meaning what it says.
    """
    from architecture_mechanics.data.feature_program import condition_config

    base = condition_config("capacity_stressed")
    ordered: list[Cell] = [Cell(name=BASE_CELL, axis="base", level=None, overrides={})]
    seen: dict[str, str] = {"{}": BASE_CELL}

    for entry in DIFFICULTY_AXES:
        axis, field_name = entry["axis"], entry["field"]
        if not hasattr(base, field_name):
            raise ValueError(f"axis {axis!r} names no generator field {field_name!r}")
        for level in entry["levels"]:
            overrides = {field_name: _jsonable(level)}
            current = _jsonable(getattr(base, field_name))
            if overrides[field_name] == current:
                continue
            key = json.dumps(overrides, sort_keys=True)
            if key in seen:
                continue
            name = f"{axis}-{entry['label'](level)}"
            seen[key] = name
            ordered.append(Cell(name=name, axis=axis, level=level, overrides=overrides))

    if not include_base:
        return tuple(cell for cell in ordered if cell.name != BASE_CELL)
    return tuple(ordered)


NEGATIVE_CONTROL_CELL = Cell(
    name="negative-control",
    axis="negative_control",
    level="source_destroyed",
    overrides={},
    condition="negative_control",
)
"""§4.4's information-destroyed control at the R3 operating point.

Not a difficulty level: it is the same task with the mutual information removed,
and it belongs to the matrix because a trained model beating it means the task
leaks and every capability number in this laboratory is measuring the leak. The
oracle bound recorded in prompt 02 puts chance at exactly 0.0."""


def cell_config(cell: Cell, *, ladder: str = "R3", seed: int = R4_SEEDS[0], **kwargs) -> RunConfig:
    """The :class:`RunConfig` for one cell at one rung and seed.

    The operating point — condition, sizes, budget, width — comes from the rung
    preset in ``experiments/config.py``; the only thing this function adds is the
    cell's one generator override. That split is deliberate: §7.2's frozen
    variables live in one place and a cell cannot reach them.
    """
    config = ladder_config(ladder, seed=seed, **kwargs)
    data = replace(config.data, condition=cell.condition, generator_overrides=dict(cell.overrides))
    return replace(config, data=data)


def _jsonable(value):
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


# --------------------------------------------------------------------------- #
# What is read back out
# --------------------------------------------------------------------------- #

CAPABILITY_METRICS: tuple[str, ...] = (
    "associative_recall_accuracy",
    "answer_set_accuracy",
    "associative_recall_jaccard",
    "heldout_composition_accuracy",
    "feature_f1",
    "feature_macro_recall",
    "reconstruction_loss",
    "brier",
)
"""§6.1 metrics read from ``summary.json``'s ``final`` block. The first is the
primary and the packet says so; the rest are reported beside it because a spread
measured on one metric says nothing about the spread on another."""

GEOMETRY_METRICS: tuple[str, ...] = (
    "probe_macro_r2",
    "probe_macro_auc",
    "mean_purity",
    "interference_fraction",
    "effective_rank",
    "participation_ratio",
    "mean_abs_off_diagonal_cosine",
    "capacity_total",
    "alignment_marginal_mean",
)
"""§6.2 measures at the run's primary site. Prompt 07 retired several of these
as unable to carry a claim alone; they are still measured, and their seed spread
is exactly as necessary to know before anyone compares two architectures on
them."""

MECHANISM_METRICS: tuple[str, ...] = (
    "best_off_diagonal_mass",
    "best_entropy_ratio",
    "best_retrieval_lift",
)
"""§6.3 activity, from the verdict block. A gate is pass/fail; these are the
numbers behind it, and their spread is what says whether a gate is near its
threshold or far from it."""


def run_metrics(summary: dict) -> dict[str, float | None]:
    """One run's scalars, from every measurement family, under one namespace.

    Geometry and mechanism names are prefixed so that a metric name in a report
    says which ruler produced it. ``None`` is preserved rather than dropped: a
    metric that was not measurable on this cell is a fact about the cell, and a
    silently absent key would read as a metric nobody looked at.
    """
    final = summary.get("final") or {}
    geometry = (summary.get("geometry") or {}).get("primary") or {}
    verdict = (summary.get("mechanism") or {}).get("verdict") or {}
    values: dict[str, float | None] = {name: final.get(name) for name in CAPABILITY_METRICS}
    values |= {f"geometry.{name}": geometry.get(name) for name in GEOMETRY_METRICS}
    values |= {f"mechanism.{name}": verdict.get(name) for name in MECHANISM_METRICS}
    return values


ALL_METRICS: tuple[str, ...] = (
    CAPABILITY_METRICS
    + tuple(f"geometry.{name}" for name in GEOMETRY_METRICS)
    + tuple(f"mechanism.{name}" for name in MECHANISM_METRICS)
)


# --------------------------------------------------------------------------- #
# Finding recorded runs
# --------------------------------------------------------------------------- #


def _runs_root(root: Path | None = None) -> Path:
    return Path(root or lab_root()) / "runs"


def _load(run_dir: Path) -> dict:
    return json.loads((run_dir / "summary.json").read_text())


def recorded_runs(
    *, ladder: str, claim: str = CLAIM, root: Path | None = None
) -> list[tuple[Path, dict]]:
    """Every recorded run at this rung whose manifest names this claim packet.

    Attribution is by manifest and not by directory name, so a run that named a
    different pre-registration cannot be swept into this claim's evidence by a
    glob that happened to match it.
    """
    found: list[tuple[Path, dict]] = []
    runs = _runs_root(root)
    if not runs.is_dir():
        return found
    for directory in sorted(runs.iterdir()):
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("ladder_rung") != ladder:
            continue
        if manifest.get("parent_claim_packet") != claim:
            continue
        found.append((directory, _load(directory)))
    return found


def _matching(summaries: Sequence[tuple[Path, dict]], config_predicate) -> list[tuple[Path, dict]]:
    return [(path, s) for path, s in summaries if config_predicate(s.get("config") or {})]


# --------------------------------------------------------------------------- #
# The competence envelope
# --------------------------------------------------------------------------- #


def difficulty_curves(*, claim: str = CLAIM, root: Path | None = None) -> dict:
    """The five §4.3 curves, assembled from the recorded R3 matrix.

    One point per cell, each point a run that *trained and evaluated* at that
    difficulty. Evaluating one model across difficulties would measure
    extrapolation from its training distribution, which is a different question
    and not the one prompts 13 and 20 need answered.
    """
    recorded = recorded_runs(ladder="R3", claim=claim, root=root)
    by_overrides: dict[str, tuple[Path, dict]] = {}
    for path, summary in recorded:
        data = (summary.get("config") or {}).get("data") or {}
        key = json.dumps(
            {"condition": data.get("condition"), "overrides": data.get("generator_overrides") or {}},
            sort_keys=True,
        )
        by_overrides[key] = (path, summary)

    def point(cell: Cell) -> dict | None:
        key = json.dumps(
            {"condition": cell.condition, "overrides": cell.overrides}, sort_keys=True
        )
        if key not in by_overrides:
            return None
        path, summary = by_overrides[key]
        references = summary.get("references") or {}
        return {
            "cell": cell.name,
            "level": _jsonable(cell.level),
            "run_id": summary.get("run_id"),
            "run_dir": f"runs/{path.name}",
            "overrides": cell.overrides,
            "condition": cell.condition,
            "passed": summary.get("passed"),
            "metrics": run_metrics(summary),
            "skill": (references.get("skill") or {}),
            "oracle_recall": (references.get("oracle") or {}).get(
                "associative_recall_accuracy"
            ),
            "marginal_recall": (references.get("marginal") or {}).get(
                "associative_recall_accuracy"
            ),
            "n_train_templates": ((references.get("train") or {}).get("split") or {}).get(
                "n_train_templates"
            ),
        }

    base = point(cells()[0])
    curves: dict[str, dict] = {}
    for entry in DIFFICULTY_AXES:
        axis = entry["axis"]
        points = []
        for cell in cells(include_base=False):
            if cell.axis != axis:
                continue
            recorded_point = point(cell)
            if recorded_point is not None:
                points.append(recorded_point)
        # A level that collapsed onto the base cell is still a point on this
        # axis; it is simply the base run, cited from every axis it belongs to.
        base_value = _base_level(entry)
        if base_value is not None and base is not None:
            points.append(base | {"level": base_value, "cell": BASE_CELL})
        points.sort(key=lambda p: _sort_key(p["level"]))
        curves[axis] = {
            "field": entry["field"],
            "why": entry["why"],
            "points": points,
            "n_missing": sum(
                1
                for cell in cells(include_base=False)
                if cell.axis == axis and point(cell) is None
            ),
        }

    return {
        "schema": "am.t1_difficulty_curves.v1",
        "t1_ladder_version": T1_LADDER_VERSION,
        "claim": claim,
        "primary_metric": "associative_recall_accuracy",
        "base_cell": base,
        "negative_control": point(NEGATIVE_CONTROL_CELL),
        "axes": curves,
    }


def _base_level(entry: dict):
    """The base condition's own value on this axis, or ``None`` if it is not a level."""
    from architecture_mechanics.data.feature_program import condition_config

    value = _jsonable(getattr(condition_config("capacity_stressed"), entry["field"]))
    return value if any(_jsonable(level) == value for level in entry["levels"]) else None


def _sort_key(level):
    if isinstance(level, list) and level and isinstance(level[0], list):
        return (0.0, float(level[0][0]))
    if isinstance(level, bool):
        return (0.0, float(level))
    if isinstance(level, (int, float)):
        return (0.0, float(level))
    return (1.0, 0.0)


# --------------------------------------------------------------------------- #
# How much A0 differs from itself
# --------------------------------------------------------------------------- #


def _spread(values: Sequence[float], *, name: str) -> dict:
    """Mean, spread, and an interval, over runs — with the spread's own error.

    The standard deviation is the quantity this whole mission exists to produce,
    and a standard deviation from five points is itself noisy: its relative
    standard error is ``1/sqrt(2(n-1))``, which is 35% at five runs. Reporting
    ``s`` without that is how a five-seed variance estimate becomes a number
    later missions trust to two decimal places.
    """
    array = np.asarray(values, dtype=float)
    n = int(array.size)
    finite = bool(np.isfinite(array).all())
    if not finite or n < 2:
        return {
            "metric": name,
            "n": n,
            "values": [None if not math.isfinite(v) else float(v) for v in array],
            "usable": False,
            "reason": (
                "not measurable on every run of this arm"
                if not finite
                else "a spread needs at least two runs"
            ),
        }

    mean = float(array.mean())
    sd = float(array.std(ddof=1))
    low, high, detail = bootstrap_ci(
        array, unit="run", method="studentized", level=CI_LEVEL, resamples=BOOTSTRAP_RESAMPLES
    )
    # Chi-square interval for sigma from a sample standard deviation. The point
    # of quoting it: at five runs the 95% interval for sigma spans a factor of
    # nearly three, so "the seed spread is 0.02" is a statement with a factor-of-
    # three uncertainty attached and should be read as one.
    sd_low, sd_high = _sd_interval(sd, n)
    return {
        "metric": name,
        "n": n,
        "values": [float(v) for v in array],
        "usable": True,
        "mean": mean,
        "sd": sd,
        "se": sd / math.sqrt(n),
        "min": float(array.min()),
        "max": float(array.max()),
        "range": float(array.max() - array.min()),
        "cv": (abs(sd / mean) if mean != 0.0 else None),
        "ci_low": low,
        "ci_high": high,
        "ci_method": "studentized",
        "ci_level": CI_LEVEL,
        "ci_detail": detail,
        "sd_ci_low": sd_low,
        "sd_ci_high": sd_high,
        "sd_relative_standard_error": 1.0 / math.sqrt(2.0 * (n - 1)),
    }


def _sd_interval(sd: float, n: int, level: float = CI_LEVEL) -> tuple[float, float]:
    """A chi-square confidence interval for sigma, from ``s`` on ``n`` runs.

    Implemented here rather than imported: prompt 08's module deliberately
    exports estimators over *differences between arms*, and this is a statement
    about the spread of one arm. Two-sided chi-square quantiles by bisection on
    the regularised lower incomplete gamma, which is fifteen lines and needs no
    dependency.
    """
    if n < 2 or not math.isfinite(sd):
        return (float("nan"), float("nan"))
    tail = (1.0 - level) / 2.0
    df = n - 1
    lower_q = _chi2_ppf(tail, df)
    upper_q = _chi2_ppf(1.0 - tail, df)
    return (sd * math.sqrt(df / upper_q), sd * math.sqrt(df / lower_q))


def _chi2_ppf(probability: float, df: int) -> float:
    """Quantile of the chi-square distribution, by bisection on its CDF."""
    low, high = 1e-9, 1.0
    while _chi2_cdf(high, df) < probability:
        high *= 2.0
        if high > 1e9:
            break
    for _ in range(200):
        middle = 0.5 * (low + high)
        if _chi2_cdf(middle, df) < probability:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def _chi2_cdf(x: float, df: int) -> float:
    """Regularised lower incomplete gamma P(df/2, x/2), series and continued fraction."""
    if x <= 0.0:
        return 0.0
    a, z = df / 2.0, x / 2.0
    log_gamma = math.lgamma(a)
    if z < a + 1.0:
        term = 1.0 / a
        total = term
        for index in range(1, 500):
            term *= z / (a + index)
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return total * math.exp(-z + a * math.log(z) - log_gamma)
    # Lentz's continued fraction for Q(a, z), then P = 1 - Q.
    tiny = 1e-300
    b = z + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for index in range(1, 500):
        an = -index * (index - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return 1.0 - math.exp(-z + a * math.log(z) - log_gamma) * h


def seed_variance(
    *,
    seeds: Sequence[int] = R4_SEEDS,
    claim: str = CLAIM,
    root: Path | None = None,
    mde_replicates: int = 4000,
) -> dict:
    """A0's own seed-to-seed spread at the R3 operating point, and what it buys.

    The number every later architecture comparison is read against. Its use is
    the last block of the return value: prompt 08 measured that the adopted
    paired test needs ``1.68`` standard deviations *of the paired difference* to
    reach 80% power at five seeds, and that is dimensionless until somebody
    supplies a standard deviation in the metric's own units. This does.

    The conversion assumes the candidate's seed noise is of the same size as
    A0's and independent of it, so ``sd(difference) = sqrt(2) * sd(A0)``. The
    seed moves initialisation and batch order and nothing else — the generator,
    the split and the evaluation data are identical across seeds — so the two
    arms share no noise source and the independence half is not a leap. The
    equal-size half is an assumption and is named as one: a candidate that is
    twice as variable makes the detectable effect larger, never smaller.
    """
    recorded = recorded_runs(ladder="R4", claim=claim, root=root)
    wanted = list(seeds)
    by_seed: dict[int, tuple[Path, dict]] = {}
    for path, summary in recorded:
        seed = (summary.get("config") or {}).get("seed")
        if seed in wanted:
            by_seed[int(seed)] = (path, summary)

    missing = [seed for seed in wanted if seed not in by_seed]
    runs: list[RunSummary] = []
    per_run: list[dict] = []
    for seed in wanted:
        if seed not in by_seed:
            continue
        path, summary = by_seed[seed]
        metrics = run_metrics(summary)
        runs.append(
            RunSummary(
                run_id=str(summary["run_id"]),
                seed=seed,
                arm="A0",
                cell=BASE_CELL,
                metrics={k: (float("nan") if v is None else v) for k, v in metrics.items()},
            )
        )
        per_run.append(
            {
                "seed": seed,
                "run_id": summary["run_id"],
                "run_dir": f"runs/{path.name}",
                "passed": summary.get("passed"),
                "verdict": summary.get("verdict"),
                "metrics": metrics,
            }
        )

    spreads = {
        name: _spread([run.value(name) for run in runs], name=name) for name in ALL_METRICS
    } if runs else {}

    n = len(runs)
    detectable = None
    if n >= 2:
        calibration = minimum_detectable_effect(n_seeds=n, replicates=mde_replicates)
        mde_dz = float(calibration["minimum_detectable_effect_dz"])
        detectable = {
            "n_seeds": n,
            "minimum_detectable_dz": mde_dz,
            "calibration": calibration,
            "replicates": mde_replicates,
            "alpha": ALPHA,
            "target_power": 0.80,
            "difference_sd_factor": math.sqrt(2.0),
            "assumption": (
                "sd(paired difference) = sqrt(2) x sd(A0), i.e. the candidate's seed noise "
                "is the same size as A0's and independent of it. The seed moves only "
                "initialisation and batch order, so independence holds by construction; "
                "equal size is an assumption, and a noisier candidate makes the detectable "
                "effect larger."
            ),
            "per_metric": {
                name: {
                    "sd_a0": spread["sd"],
                    "sd_difference_implied": math.sqrt(2.0) * spread["sd"],
                    "minimum_detectable_difference": mde_dz * math.sqrt(2.0) * spread["sd"],
                    "as_fraction_of_mean": (
                        abs(mde_dz * math.sqrt(2.0) * spread["sd"] / spread["mean"])
                        if spread["mean"]
                        else None
                    ),
                }
                for name, spread in spreads.items()
                if spread.get("usable")
            },
            "seeds_for_half_that": {
                "effect_dz": 0.5,
                "seeds_required": seeds_for_power(0.5, replicates=max(500, mde_replicates // 4)),
            },
        }

    return {
        "schema": "am.t1_seed_variance.v1",
        "t1_ladder_version": T1_LADDER_VERSION,
        "statistics_version": STATISTICS_VERSION,
        "claim": claim,
        "arm": "A0",
        "cell": BASE_CELL,
        "ladder": "R4",
        "seeds_requested": wanted,
        "seeds_found": sorted(by_seed),
        "seeds_missing": missing,
        "runs": per_run,
        "spread": spreads,
        "detectable_effect": detectable,
    }


# --------------------------------------------------------------------------- #
# Running the ladder, in order
# --------------------------------------------------------------------------- #


def _passed_r1(*, claim: str, root: Path | None) -> tuple[bool, str]:
    recorded = recorded_runs(ladder="R1", claim=claim, root=root)
    if not recorded:
        return False, "no R1 run for this claim has been recorded"
    passing = [s for _, s in recorded if s.get("passed")]
    if not passing:
        return False, f"{len(recorded)} R1 run(s) recorded, none passed"
    return True, f"{len(passing)} of {len(recorded)} recorded R1 run(s) passed"


def _stage_configs(stage: str, *, seeds: Sequence[int]) -> list[tuple[str, RunConfig, bool]]:
    """``(label, config, assert_pass)`` for one stage, in the order to run them."""
    if stage == "r1":
        return [("R1 positive control", ladder_config("R1", seed=seeds[0]), True)]
    if stage == "r2":
        return [
            (
                f"R2 kill screen d={width}",
                ladder_config("R2", seed=seeds[0], d_model=width),
                False,
            )
            for width in R2_WIDTHS
        ]
    if stage == "r3":
        planned = [
            (f"R3 {cell.name}", cell_config(cell, ladder="R3", seed=seeds[0]), False)
            for cell in cells()
        ]
        planned.append(
            (
                "R3 negative-control",
                cell_config(NEGATIVE_CONTROL_CELL, ladder="R3", seed=seeds[0]),
                False,
            )
        )
        return planned
    if stage == "r4":
        return [
            (
                f"R4 seed {seed}",
                cell_config(cells()[0], ladder="R4", seed=seed),
                False,
            )
            for seed in seeds
        ]
    raise ValueError(f"unknown stage {stage!r}; expected r1, r2, r3, r4 or report")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A0 through the T1 ladder (§7.3), in order.")
    parser.add_argument("--stage", required=True, choices=("r1", "r2", "r3", "r4", "report"))
    parser.add_argument("--arch", default="softmax")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="seeds for this stage; R4 defaults to the five in R4_SEEDS")
    parser.add_argument("--out", default="runs")
    parser.add_argument("--claim", default=CLAIM)
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would run, and the run identity of each, without training")
    args = parser.parse_args(argv)

    root = lab_root()
    if args.stage == "report":
        return _report(root=root, claim=args.claim, reports=root / args.reports)

    seeds = tuple(args.seeds) if args.seeds else (R4_SEEDS if args.stage == "r4" else R4_SEEDS[:1])
    if args.stage in ("r3", "r4"):
        ok, why = _passed_r1(claim=args.claim, root=root)
        if not ok:
            print(
                f"refusing to run {args.stage.upper()}: {why}.\n"
                "§7.3: an R1 failure is an implementation or optimisation bug, not a finding, "
                "and a task matrix run on a broken instrument is sixteen measurements of the "
                "bug. Run --stage r1 first.",
                file=sys.stderr,
            )
            return 2
        print(f"R1 gate: {why}")

    from architecture_mechanics.experiments.runner import run as run_one

    planned = _stage_configs(args.stage, seeds=seeds)
    if args.arch != "softmax":
        planned = [
            (label, replace(config, arch=replace(config.arch, arch=args.arch)), assert_pass)
            for label, config, assert_pass in planned
        ]

    print(f"{args.stage.upper()}: {len(planned)} run(s)")
    failures: list[str] = []
    for index, (label, config, assert_pass) in enumerate(planned, start=1):
        print(f"\n[{index}/{len(planned)}] {label}")
        if args.dry_run:
            print(f"  condition {config.data.condition} overrides {config.data.generator_overrides}")
            print(f"  d_model   {config.arch.d_model} steps {config.optim.max_steps}")
            continue
        result = run_one(
            config,
            out_dir=None if args.out.lower() == "none" else Path(args.out),
            claim=args.claim,
            emit_bundle=True,
        )
        if assert_pass and not result.passed:
            print(f"FAILED: {label}: {result.verdict}", file=sys.stderr)
            return 1
        if not result.passed:
            failures.append(f"{label}: {result.verdict}")

    if failures:
        # Not an error exit: on R3's matrix a cell where A0 fails is the point of
        # having a matrix, and on the negative control an inert mechanism is the
        # expected result. They are printed so nobody has to go looking.
        print("\nruns that did not pass their rung's own check:")
        for line in failures:
            print(f"  {line}")
    return 0


def _report(*, root: Path, claim: str, reports: Path) -> int:
    reports.mkdir(parents=True, exist_ok=True)
    curves = difficulty_curves(claim=claim, root=root)
    found = sorted(
        int((s.get("config") or {}).get("seed"))
        for _, s in recorded_runs(ladder="R4", claim=claim, root=root)
    )
    variance = seed_variance(
        seeds=found or R4_SEEDS, claim=claim, root=root
    )
    for name, payload in (
        ("a0_t1_difficulty_curves.json", curves),
        ("a0_t1_seed_variance.json", variance),
    ):
        (reports / name).write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
        print(f"wrote {reports.name}/{name}")

    print("\ndifficulty curves (associative_recall_accuracy):")
    base = curves.get("base_cell")
    if base:
        print(f"  base  {base['metrics']['associative_recall_accuracy']:.4f}  ({base['run_id']})")
    for axis, block in curves["axes"].items():
        rendered = "  ".join(
            f"{point['level']}={_fmt(point['metrics']['associative_recall_accuracy'])}"
            for point in block["points"]
        )
        print(f"  {axis:<16} {rendered}")
    negative = curves.get("negative_control")
    if negative:
        print(
            f"  negative control  recall="
            f"{_fmt(negative['metrics']['associative_recall_accuracy'])}  "
            f"skill={_fmt(negative['skill'].get('associative_recall_accuracy'))}"
        )

    print(f"\nseed variance over {len(variance['seeds_found'])} seeds:")
    for name in ALL_METRICS:
        spread = variance["spread"].get(name)
        if not spread or not spread.get("usable"):
            continue
        print(
            f"  {name:<40} mean {spread['mean']:>9.4f}  sd {spread['sd']:>8.4f}  "
            f"range {spread['range']:>8.4f}"
        )
    detectable = variance.get("detectable_effect")
    if detectable:
        primary = detectable["per_metric"].get("associative_recall_accuracy")
        print(
            f"\n  minimum detectable dz at {detectable['n_seeds']} seeds: "
            f"{detectable['minimum_detectable_dz']:.3f}"
        )
        if primary:
            print(
                f"  smallest visible difference in associative_recall_accuracy: "
                f"{primary['minimum_detectable_difference']:.4f}"
            )
    return 0


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.4f}"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
