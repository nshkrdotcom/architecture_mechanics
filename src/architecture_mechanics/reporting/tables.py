"""Result tables generated only from recorded artifacts.

One function with one job: turn the resolved declarations of a comparison —
``reports/comparisons/*.json``, written by
:func:`~architecture_mechanics.experiments.comparison.resolve` — plus the
``summary.json`` of every run they name into a table a report can be written
from. It runs no model, opens no checkpoint, and computes nothing that is not
already on disk; if a number is not in a recorded artifact it is not in the
table, and a cell nobody ran is absent rather than zero.

Why it exists at all, given §13.3's warning against building infrastructure: a
comparison is spread across four files per pair — two manifests, two summaries —
and reading a ten-cell screen by hand is where a transcription error becomes a
scientific claim. What it deliberately does *not* do is decide anything. There
is no verdict column, no significance test, no ranking: it reports the control's
number, the candidate's number, their difference, and — beside every difference —
the seed-noise yardstick that says whether a difference of that size is
resolvable at all with the number of seeds that produced it.

That last column is the point. Prompt 09 measured A0's seed-to-seed standard
deviation at the R3/R4 operating point as 0.054 exact recall, so a single pair
resolves nothing below roughly 0.15 and five seeds resolve nothing below 0.128.
A table of single-seed differences without that number beside it invites exactly
the reading §7.4 forbids.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path

from architecture_mechanics.experiments.comparison import COMPARISONS_DIR, DECLARATION_SCHEMA
from architecture_mechanics.experiments.manifest import lab_root
from architecture_mechanics.experiments.t1_ladder import (
    CAPABILITY_METRICS,
    GEOMETRY_METRICS,
    MECHANISM_METRICS,
    cells,
)

__all__ = [
    "A0_SEED_SD",
    "MAX_PILOT_DIFFICULTY_CELLS",
    "MECHANISM_STATE_MEASURES",
    "PHASE_GEOMETRY_METRIC",
    "PHASE_TABLE_SCHEMA",
    "PILOT_CEILING",
    "PILOT_FLOOR",
    "PURITY_FIVE_SEED_MDE",
    "TABLE_SCHEMA",
    "UNCONDITIONAL_PILOT_CELLS",
    "arm_record",
    "comparison_report",
    "main",
    "phase_report",
    "resolved_declarations",
    "surviving_cells",
]

TABLE_SCHEMA = "am.comparison_table.v1"

A0_SEED_SD: float = 0.0540
"""A0's seed-to-seed standard deviation of ``associative_recall_accuracy`` at the
R3/R4 operating point, over the eight seeds of prompt 09's replication
(``reports/a0_t1_seed_variance.json``).

Carried here so that every difference this module reports is printed beside the
noise it has to beat. ``sqrt(2) * A0_SEED_SD = 0.0764`` is the standard deviation
of a *difference* between two single-seed runs, on the assumption — stated by
prompt 09 as an assumption — that the candidate's seed noise is the same size as
the control's and independent of it. Independence holds by construction here (the
seed moves initialisation and batch order only, and the two arms share no noise
source); equal size does not, and a noisier candidate makes the yardstick larger,
never smaller."""

MECHANISM_STATE_MEASURES: tuple[str, ...] = (
    "state_norm",
    "write_norm",
    "write_to_state_ratio",
    "state_growth_ratio",
    "readout_magnitude",
    "normalizer_mean",
)
"""§6.3's "write/erase norm relative to state norm" line, which a recurrent
mechanism has and attention does not. Absent from an A0 run, and absent is what
the table records — a zero would read as a state that writes nothing."""

MECHANISM_DISTRIBUTION_MEASURES: tuple[str, ...] = (
    "entropy_nats",
    "entropy_ratio",
    "self_mass",
    "off_diagonal_mass",
    "max_weight",
)
"""The five §6.3 distribution statistics both architectures have, computed for
both by one copy of the ruler in ``models/common.py``."""


def resolved_declarations(
    comparison: str, ladder: str, *, strategy: str | None = None, lab: Path | None = None
) -> list[dict]:
    """Every resolved declaration of one comparison at one rung, cell order kept.

    Reads the gate's own directory, and only files carrying the declaration
    schema: a plan that found its way in there is not a result and must not be
    reported as one.
    """
    root = Path(lab or lab_root())
    out = []
    for path in sorted((root / COMPARISONS_DIR).glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("schema") != DECLARATION_SCHEMA:
            continue
        if record.get("comparison") != comparison or record.get("ladder") != ladder:
            continue
        if strategy is not None and record.get("matching_strategy") != strategy:
            continue
        out.append(record)
    return out


def _summary(run_id: str, lab: Path) -> dict:
    return json.loads((lab / "runs" / run_id / "summary.json").read_text())


def arm_record(summary: dict) -> dict:
    """One arm's numbers, from its own ``summary.json`` and nothing else."""
    final = summary.get("final") or {}
    mechanism = summary.get("mechanism") or {}
    verdict = mechanism.get("verdict") or {}
    distribution = mechanism.get("distribution") or {}
    geometry = summary.get("geometry") or {}
    primary = geometry.get("primary") or {}
    history = summary.get("history") or []

    layers: dict[str, dict] = {}
    for key, value in distribution.items():
        layer, _, measure = key.rpartition(".")
        if measure in MECHANISM_DISTRIBUTION_MEASURES or measure in MECHANISM_STATE_MEASURES:
            layers.setdefault(layer, {})[measure] = value

    retrieval = {
        layer: {
            "source_mass": report.get("source_mass"),
            "best_head_source_mass": report.get("best_head_source_mass"),
            "chance_mass": report.get("chance_mass"),
            "lift": report.get("lift"),
            "argmax_hit_rate": report.get("argmax_hit_rate"),
        }
        for layer, report in (mechanism.get("retrieval") or {}).items()
        if isinstance(report, dict)
    }

    recalls = [
        entry.get("eval_associative_recall_accuracy")
        for entry in history
        if entry.get("eval_associative_recall_accuracy") is not None
    ]
    return {
        "run_id": summary.get("run_id"),
        "arch": ((summary.get("config") or {}).get("arch") or {}).get("arch"),
        "d_model": (summary.get("model") or {}).get("d_model"),
        "parameters": (summary.get("parameters") or {}).get("total"),
        "capability": {name: final.get(name) for name in CAPABILITY_METRICS},
        "mechanism": {
            "active": verdict.get("active"),
            "reasons": verdict.get("reasons"),
            **{name: verdict.get(name) for name in MECHANISM_METRICS},
            "retrieval_measurable": verdict.get("retrieval_measurable"),
            "by_layer": layers,
            "retrieval": retrieval,
        },
        "geometry": {
            "primary_site": geometry.get("primary_site"),
            **{name: primary.get(name) for name in GEOMETRY_METRICS},
            "interference_fraction": primary.get("interference_fraction"),
            # The same measures split by feature bank, quoted from the run's own
            # summary. Content, key and operator features are three different
            # kinds of thing, and a mean over all three changes meaning when the
            # banks change size relative to one another — which is exactly what
            # prompt 14's phase grid does as it moves F. Carried here so a report
            # that needs the comparable version does not have to reopen the run.
            "by_bank": geometry.get("by_bank") or {},
            "matched_sites": [
                {
                    "candidate_site": (entry.get("sites") or {}).get("candidate_site"),
                    "baseline_site": (entry.get("sites") or {}).get("baseline_site"),
                    "representation_similarity": entry.get("representation_similarity"),
                    "candidate": {
                        name: (entry.get("candidate") or {}).get(name)
                        for name in GEOMETRY_METRICS
                    },
                    "baseline": {
                        name: (entry.get("baseline") or {}).get(name)
                        for name in GEOMETRY_METRICS
                    },
                    "difference": {
                        name: (entry.get("difference") or {}).get(name)
                        for name in GEOMETRY_METRICS
                    },
                }
                for entry in (geometry.get("matched_sites") or [])
            ],
        },
        "training": {
            "evaluations": len(history),
            "first_recall": recalls[0] if recalls else None,
            "final_recall": recalls[-1] if recalls else None,
            "final_step_gain": (recalls[-1] - recalls[-2]) if len(recalls) >= 2 else None,
            "still_rising": (
                bool(recalls[-1] > recalls[-2]) if len(recalls) >= 2 else None
            ),
        },
        "kill": summary.get("kill"),
        "passed": summary.get("passed"),
        "verdict": summary.get("verdict"),
        "references": (summary.get("references") or {}).get("skill"),
        "instrument_ok": (summary.get("positive_control") or {}).get("instrument_ok"),
    }


def _difference(control: dict, candidate: dict, block: str, names: Sequence[str]) -> dict:
    out = {}
    for name in names:
        left, right = control[block].get(name), candidate[block].get(name)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            out[name] = left - right
        else:
            out[name] = None
    return out


def comparison_report(
    comparison: str, ladder: str, *, n_seeds: int = 1, lab: Path | None = None
) -> dict:
    """The whole comparison as one record, control minus candidate, per cell.

    ``n_seeds`` is what the difference column is read against and is not read
    from the runs: it is the number of *pairs* whose difference is being
    reported, and the caller states it because a table that inferred it from the
    files it happened to find would silently rescale its own yardstick.
    """
    lab = Path(lab or lab_root())
    records = resolved_declarations(comparison, ladder, strategy="width_matched", lab=lab)
    all_strategies = sorted(
        {
            record.get("matching_strategy")
            for record in resolved_declarations(comparison, ladder, lab=lab)
        }
    )

    rows = []
    for record in records:
        control = arm_record(_summary(record["control_run"], lab))
        candidate = arm_record(_summary(record["candidate_runs"][0], lab))
        rows.append(
            {
                "cell": record.get("cell"),
                "seed": record.get("seed"),
                "control": control,
                "candidate": candidate,
                "difference": {
                    "capability": _difference(control, candidate, "capability", CAPABILITY_METRICS),
                    "geometry": _difference(
                        control, candidate, "geometry", (*GEOMETRY_METRICS, "interference_fraction")
                    ),
                    "mechanism": _difference(control, candidate, "mechanism", MECHANISM_METRICS),
                },
                "both_alive": _both_alive(control, candidate),
                "both_mechanisms_active": bool(
                    control["mechanism"]["active"] and candidate["mechanism"]["active"]
                ),
                "compute_ledger": record.get("compute_ledger"),
                "checks": record.get("checks"),
                "permitted_differences": record.get("permitted_differences"),
                "parameter_accounting": record.get("parameter_accounting"),
            }
        )

    return {
        "schema": TABLE_SCHEMA,
        "comparison": comparison,
        "ladder": ladder,
        "matching_strategies_resolved": all_strategies,
        "strategies_coincide": len(all_strategies) > 1
        and all(
            record["control_run"] == other["control_run"]
            and record["candidate_runs"] == other["candidate_runs"]
            for record, other in zip(
                records,
                resolved_declarations(
                    comparison, ladder, strategy="parameter_matched", lab=lab
                ),
                strict=False,
            )
        ),
        "resolution": {
            "n_seeds": n_seeds,
            "a0_seed_sd_recall": A0_SEED_SD,
            "sd_of_a_paired_difference": round(math.sqrt(2) * A0_SEED_SD, 4),
            "smallest_resolvable_recall_difference": _minimum_detectable(n_seeds),
            "source": "reports/a0_t1_seed_variance.json — prompt 09, eight seeds",
            "note": (
                "A difference smaller than the last figure is not resolvable with this many "
                "pairs and is reported as an observation, never as an effect. At one pair "
                "there is no interval at all; the figure quoted is what five pairs would "
                "reach, and is therefore a lower bound on what one pair would need."
            ),
        },
        "rows": rows,
    }


PILOT_FLOOR: float = 0.05
PILOT_CEILING: float = 0.95
"""The window prompt 09 pre-registered for a T1 operating point, reused here as
the definition of "alive". Below the floor a run measures the floor; at the
ceiling it cannot carry a difference in either direction."""

UNCONDITIONAL_PILOT_CELLS: tuple[str, ...] = ("base", "negative-control")
"""Piloted whether or not they survive the rule.

``base`` because it is the operating point prompt 12's comparison declared, and a
mission that moved off the inherited operating point without reporting what
happens there would be choosing its own result. ``negative-control`` because it
is the standing control for a capability claim and is not a difficulty cell."""

MAX_PILOT_DIFFICULTY_CELLS: int = 5
"""The budget cap, as a rule rather than a choice. Six pilot cells at both arms
is twelve runs, about eighteen minutes on this machine."""


def surviving_cells(report: dict) -> dict:
    """Apply the declared pilot-selection rule to a screen's table.

    Executable rather than prose, because a rule applied by hand after the
    numbers are visible is indistinguishable from a preference. Every cell the
    screen ran appears in the output with a reason, including the ones the budget
    cap drops — a silently truncated matrix reads as "everything survived".

    The rule: a cell is piloted iff both arms exceed :data:`PILOT_FLOOR`, neither
    reaches :data:`PILOT_CEILING`, no §7.3 R2 kill condition fired for either
    arm, and the pair's measured work matched. If more than
    :data:`MAX_PILOT_DIFFICULTY_CELLS` difficulty cells qualify, one per axis is
    kept — the survivor with the largest ``min(control, candidate)`` recall,
    which is the most-alive point on that axis — and the rest are recorded as
    capped.
    """
    axis_of = {cell.name: cell.axis for cell in cells()}
    verdicts: dict[str, dict] = {}

    for row in report.get("rows") or []:
        cell = row["cell"]
        control = row["control"]["capability"].get("associative_recall_accuracy")
        candidate = row["candidate"]["capability"].get("associative_recall_accuracy")
        reasons: list[str] = []
        for role, value in (("control", control), ("candidate", candidate)):
            if not isinstance(value, (int, float)):
                reasons.append(f"{role} recorded no primary metric")
            elif value <= PILOT_FLOOR:
                reasons.append(f"{role} at or below the floor ({value:.4f} <= {PILOT_FLOOR})")
            elif value >= PILOT_CEILING:
                reasons.append(f"{role} at or above the ceiling ({value:.4f} >= {PILOT_CEILING})")
        for role in ("control", "candidate"):
            fired = ((row[role].get("kill") or {}).get("fired")) or []
            if fired:
                reasons.append(f"{role} kill condition fired: {', '.join(fired)}")
        if not (row.get("checks") or {}).get("measured_work_matched", True):
            reasons.append("the pair's measured work did not match")
        verdicts[cell] = {
            "axis": axis_of.get(cell, "unknown"),
            "control_recall": control,
            "candidate_recall": candidate,
            "min_recall": (
                min(control, candidate)
                if isinstance(control, (int, float)) and isinstance(candidate, (int, float))
                else None
            ),
            "qualifies": not reasons,
            "reasons": reasons,
        }

    qualifying = [name for name, entry in verdicts.items() if entry["qualifies"]]
    difficulty = [name for name in qualifying if name not in UNCONDITIONAL_PILOT_CELLS]
    kept = difficulty
    if len(difficulty) > MAX_PILOT_DIFFICULTY_CELLS:
        best_per_axis: dict[str, str] = {}
        for name in difficulty:
            axis = verdicts[name]["axis"]
            current = best_per_axis.get(axis)
            if current is None or verdicts[name]["min_recall"] > verdicts[current]["min_recall"]:
                best_per_axis[axis] = name
        kept = sorted(best_per_axis.values(), key=difficulty.index)[:MAX_PILOT_DIFFICULTY_CELLS]

    for name in difficulty:
        if name not in kept:
            verdicts[name]["reasons"].append(
                "capped: the budget rule keeps one surviving cell per axis"
            )
            verdicts[name]["qualifies"] = False

    piloted = list(UNCONDITIONAL_PILOT_CELLS[:1]) + kept + list(UNCONDITIONAL_PILOT_CELLS[1:])
    for name in UNCONDITIONAL_PILOT_CELLS:
        entry = verdicts.setdefault(
            name,
            {"axis": axis_of.get(name, "unknown"), "control_recall": None,
             "candidate_recall": None, "min_recall": None, "qualifies": False, "reasons": []},
        )
        entry["piloted_unconditionally"] = True

    return {
        "rule": {
            "floor": PILOT_FLOOR,
            "ceiling": PILOT_CEILING,
            "unconditional": list(UNCONDITIONAL_PILOT_CELLS),
            "max_difficulty_cells": MAX_PILOT_DIFFICULTY_CELLS,
        },
        "piloted": piloted,
        "cells": verdicts,
    }


PHASE_TABLE_SCHEMA = "am.phase_map.v1"

PHASE_GEOMETRY_METRIC = "mean_purity"
"""The §6.2 measure the phase map's geometry panel is drawn from, taken over the
*content* bank. Declared in
``claims/phase-map-a0-a1-sparsity-bottleneck.yml#PRIMARY_METRIC`` with the reason
and with what is not known about its seed noise."""

PURITY_FIVE_SEED_MDE = 0.049
"""Prompt 09's five-seed minimum detectable effect for whole-bank ``mean_purity``
at the R3/R4 operating point, from ``reports/a0_t1_seed_variance.json``.

Used as the yardstick beside the content-bank restriction because it is the
nearest thing that was measured, and labelled as such everywhere it appears: the
content-bank purity has no separately measured seed spread at any operating
point, and quoting this number for it is an approximation and not a measurement.
"""


def _phase_arm(arm: dict) -> dict:
    """One arm of one map point: what the figure draws and what qualifies it."""
    geometry = arm["geometry"]
    content = (geometry.get("by_bank") or {}).get("content") or {}
    kill = arm.get("kill") or {}
    return {
        "run_id": arm["run_id"],
        "arch": arm["arch"],
        "d_model": arm["d_model"],
        "parameters": arm["parameters"],
        "recall": arm["capability"]["associative_recall_accuracy"],
        "feature_f1": arm["capability"]["feature_f1"],
        "reconstruction_loss": arm["capability"]["reconstruction_loss"],
        "content_purity": content.get(PHASE_GEOMETRY_METRIC),
        "content_probe_r2": content.get("probe_macro_r2"),
        "content_features": content.get("n_features"),
        "all_bank_purity": geometry.get("mean_purity"),
        "interference_fraction": geometry.get("interference_fraction"),
        "probe_macro_r2": geometry.get("probe_macro_r2"),
        "effective_rank": geometry.get("effective_rank"),
        "participation_ratio": geometry.get("participation_ratio"),
        "mechanism_active": arm["mechanism"]["active"],
        "mechanism_reasons": arm["mechanism"]["reasons"],
        "kill_fired": kill.get("fired") or [],
        "recall_skill": (arm.get("references") or {}).get(
            "associative_recall_accuracy"
        ),
        "final_step_gain": arm["training"]["final_step_gain"],
        "still_rising": arm["training"]["still_rising"],
    }


def _saturation(recall, *, floor: float = PILOT_FLOOR, ceiling: float = PILOT_CEILING) -> str:
    """Where this arm sits against the pre-registered window.

    Three states and not two. A cell at the floor and a cell at the ceiling are
    both unable to carry a difference, and they are unable to carry it for
    opposite reasons; a figure that marked only "saturated" would lose which. The
    thresholds are prompt 09's, reused unchanged, and they are the same numbers
    ``surviving_cells`` applies.
    """
    if not isinstance(recall, (int, float)):
        return "unmeasured"
    if recall <= floor:
        return "at_chance"
    if recall >= ceiling:
        return "at_ceiling"
    return "interior"


def phase_report(*, lab: Path | None = None) -> dict:
    """§10.2's figure 2 as a record: every point of the sweep, from runs only.

    One row per (cell, width) point of the grid declared in
    ``experiments/phase_grid.py``, assembled from the resolved declarations of
    the comparisons that cover it. Like every other function in this module it
    decides nothing: it reports each arm's number, the paired difference, the
    seed-noise yardstick that says whether a difference of that size is
    resolvable at all — it is not, at one pair — and where each arm sits against
    the pre-registered floor and ceiling. The figure marks; the reader does not
    interpret.
    """
    from architecture_mechanics.experiments.phase_grid import (
        PHASE_COMPARISONS,
        PHASE_CUTS,
        PHASE_GROUP_SIZE,
        PHASE_LADDER,
        PHASE_MAIN_SEQ_LEN,
        PHASE_SPARSITIES,
        PHASE_WIDTH_FEATURE_POINTS,
        cell_axes,
        phase_cost_model,
    )

    lab = Path(lab or lab_root())
    points: list[dict] = []
    controls: dict[str, list[dict]] = {}

    for name, spec in PHASE_COMPARISONS.items():
        ladder = next(iter(spec["rungs"]))
        for record in resolved_declarations(name, ladder, strategy="width_matched", lab=lab):
            control = arm_record(_summary(record["control_run"], lab))
            candidate = arm_record(_summary(record["candidate_runs"][0], lab))
            cell = record.get("cell")
            axes = cell_axes(cell) if cell not in {"positive-control"} else {}
            width = spec["d_model"] or control["d_model"]
            left, right = _phase_arm(control), _phase_arm(candidate)
            point = {
                "comparison": name,
                "ladder": ladder,
                "cell": cell,
                "seed": record.get("seed"),
                "condition": axes.get("condition", "positive_control"),
                "d_model": width,
                "f_content": axes.get("f_content"),
                "f_total": axes.get("f_total"),
                "seq_len": axes.get("seq_len"),
                "activation_prob": axes.get("activation_prob"),
                "expected_active_content_features": axes.get(
                    "expected_active_content_features"
                ),
                "ratio_content": (
                    round(axes["f_content"] / width, 6) if axes.get("f_content") else None
                ),
                "ratio_total": (
                    round(axes["f_total"] / width, 6) if axes.get("f_total") else None
                ),
                "control": left,
                "candidate": right,
                "difference": {
                    "recall": _sub(left["recall"], right["recall"]),
                    "content_purity": _sub(left["content_purity"], right["content_purity"]),
                    "all_bank_purity": _sub(left["all_bank_purity"], right["all_bank_purity"]),
                    "interference_fraction": _sub(
                        left["interference_fraction"], right["interference_fraction"]
                    ),
                    "probe_macro_r2": _sub(left["probe_macro_r2"], right["probe_macro_r2"]),
                },
                "saturation": {
                    "control": _saturation(left["recall"]),
                    "candidate": _saturation(right["recall"]),
                },
                "both_alive": _both_alive(control, candidate),
                "both_mechanisms_active": bool(
                    left["mechanism_active"] and right["mechanism_active"]
                ),
                "compute_ledger": record.get("compute_ledger"),
                "checks": record.get("checks"),
                "permitted_differences": record.get("permitted_differences"),
            }
            if name.startswith("phase_T"):
                points.append(point)
            else:
                controls.setdefault(name, []).append(point)

    points.sort(
        key=lambda row: (
            row["ratio_content"] if row["ratio_content"] is not None else -1.0,
            -(row["d_model"] or 0),
            row["activation_prob"] or 0.0,
        )
    )
    return {
        "schema": PHASE_TABLE_SCHEMA,
        "ladder": PHASE_LADDER,
        "main_seq_len": PHASE_MAIN_SEQ_LEN,
        "sparsities": list(PHASE_SPARSITIES),
        "width_feature_points": [list(pair) for pair in PHASE_WIDTH_FEATURE_POINTS],
        "content_group_size": PHASE_GROUP_SIZE,
        "window": {"floor": PILOT_FLOOR, "ceiling": PILOT_CEILING},
        "resolution": {
            "n_seeds": 1,
            "a0_seed_sd_recall": A0_SEED_SD,
            "sd_of_a_paired_difference": round(math.sqrt(2) * A0_SEED_SD, 4),
            "smallest_resolvable_recall_difference": _minimum_detectable(5),
            "purity_five_seed_mde": PURITY_FIVE_SEED_MDE,
            "source": "reports/a0_t1_seed_variance.json — prompt 09, eight seeds",
            "note": (
                "Every point of this map is ONE PAIR. The figures quoted are what FIVE pairs "
                "would reach at prompt 09's operating point and are therefore lower bounds on "
                "what one pair would need; a single pair has no interval at all. The purity "
                "figure was measured for whole-bank mean_purity and is used beside the "
                "content-bank restriction as the nearest available yardstick, not as a "
                "measurement of it."
            ),
        },
        "cuts": [dict(entry) for entry in PHASE_CUTS],
        "cost_model": phase_cost_model(),
        "controls": controls,
        "points": points,
    }


def _sub(left, right):
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left - right
    return None


def _minimum_detectable(n_seeds: int) -> float:
    """Prompt 09's conversion, for the seed counts it measured. ``None`` elsewhere.

    Deliberately a lookup and not a formula: the numbers came from prompt 08's
    4000-replicate calibration of the paired estimator, and interpolating them
    here would invent a precision nobody measured.
    """
    return {5: 0.128, 8: 0.089, 10: 0.076, 20: 0.050}.get(n_seeds, 0.128)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tabulate a resolved comparison from recorded artifacts only."
    )
    parser.add_argument("--comparison", default=None)
    parser.add_argument("--ladder", default=None)
    parser.add_argument("--seeds", type=int, default=1, help="pairs behind each difference")
    parser.add_argument("--json", default=None, help="write the table here")
    parser.add_argument("--lab", default=None)
    parser.add_argument(
        "--phase",
        action="store_true",
        help="tabulate prompt 14's whole phase-diagram sweep instead of one comparison",
    )
    return parser


def _print_phase(report: dict) -> None:
    """The map as text, in the same order the figure draws its rows."""
    print(
        f"phase map, {report['ladder']}, one seed, T = {report['main_seq_len']} "
        f"(ribbon points excepted)"
    )
    print(
        f"{'F/d':>5}  {'F':>4}  {'d':>3}  {'p_act':>5}  {'A0':>8}  {'A1':>8}  "
        f"{'diff':>8}  {'pur A0':>7}  {'pur A1':>7}  {'d pur':>7}  sat A0/A1  act"
    )
    for row in report["points"]:
        left, right = row["control"], row["candidate"]
        print(
            f"{row['ratio_content']:>5.2f}  {row['f_content']:>4}  {row['d_model']:>3}  "
            f"{row['activation_prob']:>5.2f}  "
            f"{_fmt(left['recall'])}  {_fmt(right['recall'])}  "
            f"{_fmt(row['difference']['recall'])}  "
            f"{_fmt7(left['content_purity'])}  {_fmt7(right['content_purity'])}  "
            f"{_fmt7(row['difference']['content_purity'])}  "
            f"{row['saturation']['control'][:9]:>9}/{row['saturation']['candidate'][:9]:<9}  "
            f"{'yes' if row['both_mechanisms_active'] else 'NO'}"
        )
    for name, rows in report["controls"].items():
        for row in rows:
            print(
                f"\n{name}: {row['cell']}  A0 {_fmt(row['control']['recall'])}  "
                f"A1 {_fmt(row['candidate']['recall'])}  "
                f"diff {_fmt(row['difference']['recall'])}  "
                f"skill A0 {row['control']['recall_skill']} A1 {row['candidate']['recall_skill']}"
            )
    resolution = report["resolution"]
    print(
        f"\nONE SEED PER CELL. Five pairs could not resolve a recall difference below "
        f"{resolution['smallest_resolvable_recall_difference']} or a purity difference below "
        f"{resolution['purity_five_seed_mde']} at prompt 09's operating point; one pair "
        "resolves nothing."
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    lab = Path(args.lab) if args.lab else None

    if args.phase:
        report = phase_report(lab=lab)
        if not report["points"]:
            print("no resolved declarations for the phase sweep")
            return 1
        _print_phase(report)
        if args.json:
            path = Path(args.json)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2) + "\n")
            print(f"wrote {path}")
        return 0

    if not args.comparison or not args.ladder:
        print("--comparison and --ladder are required unless --phase is given")
        return 2
    report = comparison_report(args.comparison, args.ladder, n_seeds=args.seeds, lab=lab)
    if not report["rows"]:
        print(f"no resolved declarations for {args.comparison} at {args.ladder}")
        return 1

    width = max(len(row["cell"]) for row in report["rows"])
    print(f"{report['comparison']} {report['ladder']}  (control minus candidate)")
    print(
        f"{'cell':<{width}}  {'A0':>8}  {'A1':>8}  {'diff':>8}  "
        f"{'A0 act':>7}  {'A1 act':>7}  {'A1 w/S':>7}"
    )
    for row in report["rows"]:
        control, candidate = row["control"], row["candidate"]
        state = candidate["mechanism"]["by_layer"]
        ratios = [
            block.get("write_to_state_ratio")
            for block in state.values()
            if block.get("write_to_state_ratio") is not None
        ]
        print(
            f"{row['cell']:<{width}}  "
            f"{_fmt(control['capability']['associative_recall_accuracy'])}  "
            f"{_fmt(candidate['capability']['associative_recall_accuracy'])}  "
            f"{_fmt(row['difference']['capability']['associative_recall_accuracy'])}  "
            f"{control['mechanism']['active']!s:>7}  "
            f"{candidate['mechanism']['active']!s:>7}  "
            f"{(f'{max(ratios):.3f}' if ratios else '     - '):>7}"
        )
    resolution = report["resolution"]
    print(
        f"\nsmallest resolvable difference at {resolution['n_seeds']} pair(s): "
        f"{resolution['smallest_resolvable_recall_difference']} "
        f"(A0 seed sd {resolution['a0_seed_sd_recall']})"
    )
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {path}")
    return 0


def _both_alive(control: dict, candidate: dict, *, floor: float = 0.05) -> bool:
    """Both arms off the floor on the primary metric.

    ``floor`` is prompt 09's own: the lower edge of the pre-registered window a
    T1 operating point has to sit inside for a comparison run there to be about
    the architectures rather than about the collapse.
    """
    values = [
        arm["capability"].get("associative_recall_accuracy") for arm in (control, candidate)
    ]
    return all(isinstance(value, (int, float)) and value > floor for value in values)


def _fmt(value) -> str:
    return f"{value:>8.4f}" if isinstance(value, (int, float)) else f"{'-':>8}"


def _fmt7(value) -> str:
    return f"{value:>7.4f}" if isinstance(value, (int, float)) else f"{'-':>7}"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
