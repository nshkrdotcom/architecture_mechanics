"""The comparison table reads recorded artifacts and decides nothing.

Two properties are worth holding: that the table's numbers come out of the run
summaries unchanged and in the declared direction (control minus candidate), and
that a measure one architecture does not have is *absent* rather than zero — a
state norm of 0.0 is a mechanism that writes nothing, which is a finding, and
printing it for an architecture that has no state at all would be a fiction.
"""

from __future__ import annotations

import json

import pytest

from architecture_mechanics.experiments.comparison import ComparisonError, _cell_by_name
from architecture_mechanics.experiments.config import RunConfigError
from architecture_mechanics.experiments.t1_ladder import (
    NEGATIVE_CONTROL_CELL,
    POSITIVE_CONTROL_CELL,
    cell_config,
    cells,
)
from architecture_mechanics.reporting import tables

# --------------------------------------------------------------------------- #
# The positive-control cell
# --------------------------------------------------------------------------- #


def test_the_positive_control_cell_is_not_part_of_the_difficulty_matrix():
    """It is a cell so that a *comparison* can be run on the known-easy
    condition. Letting it into ``cells()`` would put a point from a different
    condition on every difficulty curve."""
    assert POSITIVE_CONTROL_CELL.name not in {cell.name for cell in cells()}
    assert NEGATIVE_CONTROL_CELL.name not in {cell.name for cell in cells()}
    assert _cell_by_name(POSITIVE_CONTROL_CELL.name) is POSITIVE_CONTROL_CELL


def test_the_positive_control_cell_is_usable_at_r1_and_refused_where_it_would_lie():
    config = cell_config(POSITIVE_CONTROL_CELL, ladder="R1", arch="linear")
    assert config.data.condition == "positive_control"
    assert config.data.generator_overrides == {}
    assert config.data.n_train == config.data.n_eval

    # R2 and R3 draw train and eval at different sizes, and the positive control
    # emits both splits at one size. A cell that produced a config whose data
    # request is silently ignored would name a run for a dataset it never saw.
    for ladder in ("R2", "R3"):
        with pytest.raises(RunConfigError, match="one size"):
            cell_config(POSITIVE_CONTROL_CELL, ladder=ladder)


def test_an_unknown_cell_is_refused_by_name():
    with pytest.raises(ComparisonError, match="unknown cell"):
        _cell_by_name("a-cell-nobody-declared")


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #


def _summary(path):
    return json.loads(path.read_text())


def _recorded(pattern: str):
    """A recorded *trained* run of this architecture, or skip.

    R0 runs match the same glob and carry no ``final`` block at all, because R0
    does not train — reading one here would test the absence of training rather
    than the presence of the measures.
    """
    from architecture_mechanics.experiments.manifest import lab_root

    for path in sorted((lab_root() / "runs").glob(f"{pattern}/summary.json")):
        summary = _summary(path)
        if summary.get("final") and (summary.get("mechanism") or {}).get("distribution"):
            return summary
    pytest.skip(f"no recorded trained run matching {pattern}")
    return None


def test_a_linear_run_carries_the_state_measures_and_a_softmax_run_does_not():
    linear = tables.arm_record(_recorded("*-linear-*"))
    softmax = tables.arm_record(_recorded("*-softmax-*"))

    def measures(record):
        return {
            name
            for block in record["mechanism"]["by_layer"].values()
            for name in block
        }

    assert set(tables.MECHANISM_STATE_MEASURES) <= measures(linear)
    assert not set(tables.MECHANISM_STATE_MEASURES) & measures(softmax)
    # ...and both carry the five distribution statistics, because one copy of the
    # ruler computes them for both.
    for record in (linear, softmax):
        assert set(tables.MECHANISM_DISTRIBUTION_MEASURES) <= measures(record)


def test_the_arm_record_quotes_the_summary_rather_than_recomputing_it():
    summary = _recorded("*-linear-*")
    record = tables.arm_record(summary)
    assert record["run_id"] == summary["run_id"]
    assert (
        record["capability"]["associative_recall_accuracy"]
        == summary["final"]["associative_recall_accuracy"]
    )
    assert (
        record["mechanism"]["best_retrieval_lift"]
        == summary["mechanism"]["verdict"]["best_retrieval_lift"]
    )
    history = [
        entry["eval_associative_recall_accuracy"]
        for entry in summary["history"]
        if entry.get("eval_associative_recall_accuracy") is not None
    ]
    assert record["training"]["final_recall"] == history[-1]
    assert record["training"]["first_recall"] == history[0]


def test_a_difference_is_control_minus_candidate():
    control = {"capability": {"associative_recall_accuracy": 0.5, "feature_f1": 0.8}}
    candidate = {"capability": {"associative_recall_accuracy": 0.1, "feature_f1": None}}
    difference = tables._difference(
        control, candidate, "capability", ("associative_recall_accuracy", "feature_f1")
    )
    assert difference["associative_recall_accuracy"] == pytest.approx(0.4)
    # A metric one arm did not measure has no difference, and None says so where
    # 0.0 would claim the two agreed.
    assert difference["feature_f1"] is None


def test_both_alive_uses_the_pre_registered_floor():
    def arm(value):
        return {"capability": {"associative_recall_accuracy": value}}

    assert tables._both_alive(arm(0.5), arm(0.2))
    assert not tables._both_alive(arm(0.5), arm(0.05)), "0.05 is the floor, not above it"
    assert not tables._both_alive(arm(0.01), arm(0.5))
    assert not tables._both_alive(arm(0.5), arm(None))


def test_the_resolution_yardstick_is_prompt_09s_measurement_and_not_a_formula():
    """A single pair has no interval. The reported figure is what *five* pairs
    would reach, which is a lower bound on what one pair would need, and it is a
    lookup of measured values rather than an interpolation that would invent a
    precision nobody calibrated."""
    assert tables.A0_SEED_SD == pytest.approx(0.0540)
    assert tables._minimum_detectable(5) == 0.128
    assert tables._minimum_detectable(10) == 0.076
    assert tables._minimum_detectable(20) == 0.050
    assert tables._minimum_detectable(1) == 0.128
    assert tables._minimum_detectable(7) == 0.128
