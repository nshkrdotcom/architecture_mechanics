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
)

__all__ = [
    "A0_SEED_SD",
    "MECHANISM_STATE_MEASURES",
    "TABLE_SCHEMA",
    "arm_record",
    "comparison_report",
    "main",
    "resolved_declarations",
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
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--ladder", required=True)
    parser.add_argument("--seeds", type=int, default=1, help="pairs behind each difference")
    parser.add_argument("--json", default=None, help="write the table here")
    parser.add_argument("--lab", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    lab = Path(args.lab) if args.lab else None
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
