"""The T1 task matrix, and the arithmetic that reads runs back out of it.

The property worth testing is not that the table has fifteen rows. It is that
every row is a *single-axis* move away from the base condition — a curve whose
points differ in two things at once is not a curve, and the failure would be
invisible in a plot.
"""

from __future__ import annotations

import json
import math
from dataclasses import fields

import pytest

from architecture_mechanics.data.feature_program import (
    FeatureProgramConfig,
    condition_config,
)
from architecture_mechanics.experiments import t1_ladder
from architecture_mechanics.experiments.config import (
    LADDERS,
    DataSpec,
    RunConfigError,
    config_fingerprint,
    ladder_config,
    run_config_from_dict,
)
from architecture_mechanics.experiments.t1_ladder import (
    BASE_CELL,
    DIFFICULTY_AXES,
    NEGATIVE_CONTROL_CELL,
    R4_SEEDS,
    cell_config,
    cells,
    difficulty_curves,
    seed_variance,
)

GENERATOR_FIELDS = {f.name for f in fields(FeatureProgramConfig)}


# --------------------------------------------------------------------------- #
# The matrix is single-axis
# --------------------------------------------------------------------------- #


def _differing_fields(a: FeatureProgramConfig, b: FeatureProgramConfig) -> set[str]:
    return {
        name
        for name in GENERATOR_FIELDS
        if name not in {"split", "n_examples"} and getattr(a, name) != getattr(b, name)
    }


def test_every_cell_moves_exactly_one_generator_field():
    base = condition_config("capacity_stressed")
    for cell in cells(include_base=False):
        moved = _differing_fields(base, condition_config(cell.condition, **cell.overrides))
        assert moved == {DIFFICULTY_AXES_BY_NAME[cell.axis]["field"]}, (
            f"cell {cell.name} moves {sorted(moved)}, not one axis"
        )


DIFFICULTY_AXES_BY_NAME = {entry["axis"]: entry for entry in DIFFICULTY_AXES}


def test_base_cell_is_the_condition_untouched_and_comes_first():
    ordered = cells()
    assert ordered[0].name == BASE_CELL
    assert ordered[0].overrides == {}
    assert len({cell.name for cell in ordered}) == len(ordered)


def test_every_axis_named_by_section_4_3_is_present():
    assert {entry["axis"] for entry in DIFFICULTY_AXES} == {
        "source_distance",
        "distractors",
        "sparsity",
        "key_collisions",
        "associations",
    }


def test_a_level_equal_to_the_base_value_is_not_run_twice():
    base = condition_config("capacity_stressed")
    for entry in DIFFICULTY_AXES:
        current = getattr(base, entry["field"])
        for level in entry["levels"]:
            if t1_ladder._jsonable(level) == t1_ladder._jsonable(current):
                assert not any(
                    cell.overrides.get(entry["field"]) == t1_ladder._jsonable(level)
                    for cell in cells(include_base=False)
                ), f"{entry['axis']} level {level!r} duplicates the base cell"


def test_every_cell_generates_both_splits():
    for cell in (*cells(), NEGATIVE_CONTROL_CELL):
        config = cell_config(cell, ladder="R3")
        for split in ("train", "test"):
            generated = config.data.generator_config(split=split, n_examples=8)
            assert generated.condition == cell.condition


def test_the_negative_control_is_the_base_cell_with_information_removed():
    base = condition_config("capacity_stressed")
    negative = condition_config(NEGATIVE_CONTROL_CELL.condition)
    assert _differing_fields(base, negative) == {"condition", "source_destroyed"}
    assert negative.source_destroyed is True


# --------------------------------------------------------------------------- #
# R3 and R4 are matched
# --------------------------------------------------------------------------- #


def test_r4_differs_from_the_r3_base_cell_only_in_rung_and_seed():
    pilot = cell_config(cells()[0], ladder="R3", seed=R4_SEEDS[0]).as_dict()
    replication = cell_config(cells()[0], ladder="R4", seed=R4_SEEDS[1]).as_dict()
    differing = {key for key in pilot if pilot[key] != replication.get(key)}
    assert differing == {"ladder", "seed"}


def test_r3_and_r4_declare_the_operating_point_width_rather_than_taking_the_default():
    for rung in ("R3", "R4"):
        assert LADDERS[rung]["arch"]["d_model"] == 64
        assert ladder_config(rung).arch.d_model == 64
    # And the flag still wins, so the width sweep does not need its own rung.
    assert ladder_config("R3", d_model=16).arch.d_model == 16


def test_r2_still_takes_its_width_from_the_condition_unless_told_otherwise():
    assert ladder_config("R2").arch.d_model is None
    assert ladder_config("R2").data.d_recommended == 16


# --------------------------------------------------------------------------- #
# Generator overrides are part of the run's identity
# --------------------------------------------------------------------------- #


def test_two_cells_are_two_experiments():
    fingerprints = {
        cell.name: config_fingerprint(cell_config(cell, ladder="R3")) for cell in cells()
    }
    assert len(set(fingerprints.values())) == len(fingerprints)


def test_an_override_survives_the_manifest_round_trip_with_its_identity():
    original = cell_config(cells()[3], ladder="R3")
    rebuilt = run_config_from_dict(json.loads(json.dumps(original.as_dict())))
    assert config_fingerprint(rebuilt) == config_fingerprint(original)
    assert rebuilt.data.generator_overrides == original.data.generator_overrides


def test_tuples_and_lists_are_the_same_override():
    from_tuple = DataSpec(condition="capacity_stressed", generator_overrides={"distance_buckets": ((4, 6),)})
    from_list = DataSpec(condition="capacity_stressed", generator_overrides={"distance_buckets": [[4, 6]]})
    assert from_tuple.as_dict() == from_list.as_dict()


def test_an_override_that_names_no_generator_field_is_refused():
    with pytest.raises(RunConfigError, match="no such generator field"):
        DataSpec(condition="capacity_stressed", generator_overrides={"learning_rate": 0.1})


def test_an_override_the_runner_would_discard_is_refused():
    for name, value in (("condition", "negative_control"), ("n_examples", 32), ("split", "test")):
        with pytest.raises(RunConfigError, match="may not set"):
            DataSpec(condition="capacity_stressed", generator_overrides={name: value})


def test_an_override_the_generator_would_reject_is_refused_before_the_gpu():
    # Four distractors cannot fit strictly between a source and a destination
    # two apart. The generator says so; the point is that it says so here.
    with pytest.raises(Exception, match="distractors"):
        DataSpec(
            condition="capacity_stressed",
            generator_overrides={"distance_buckets": [[2, 3]], "n_distractors": 4},
        )


def test_the_positive_control_takes_no_overrides():
    with pytest.raises(RunConfigError, match="takes no generator_overrides"):
        DataSpec(condition="positive_control", generator_overrides={"n_distractors": 1})


# --------------------------------------------------------------------------- #
# The spread arithmetic
# --------------------------------------------------------------------------- #


def test_chi_square_cdf_against_published_values():
    # Critical values from standard tables: P(X <= 3.841 | df=1) = 0.95 and
    # P(X <= 11.070 | df=5) = 0.95.
    assert t1_ladder._chi2_cdf(3.8415, 1) == pytest.approx(0.95, abs=1e-4)
    assert t1_ladder._chi2_cdf(11.0705, 5) == pytest.approx(0.95, abs=1e-4)
    assert t1_ladder._chi2_ppf(0.975, 4) == pytest.approx(11.1433, abs=1e-3)


def test_the_interval_on_a_standard_deviation_is_wide_at_five_runs():
    low, high = t1_ladder._sd_interval(1.0, 5)
    assert low < 1.0 < high
    assert high / low > 2.5, "a five-run sigma is known to within a factor of about three"
    tighter_low, tighter_high = t1_ladder._sd_interval(1.0, 20)
    assert (tighter_high / tighter_low) < (high / low)


def test_spread_reports_the_relative_error_of_its_own_standard_deviation():
    spread = t1_ladder._spread([0.5, 0.52, 0.48, 0.51, 0.49], name="x")
    assert spread["usable"]
    assert spread["n"] == 5
    assert spread["sd_relative_standard_error"] == pytest.approx(1 / math.sqrt(8))
    assert spread["ci_low"] <= spread["mean"] <= spread["ci_high"]


def test_a_metric_missing_on_one_run_is_unusable_rather_than_averaged():
    spread = t1_ladder._spread([0.5, float("nan"), 0.51], name="x")
    assert not spread["usable"]
    assert "not measurable" in spread["reason"]


# --------------------------------------------------------------------------- #
# Reading runs back
# --------------------------------------------------------------------------- #


def _write_run(root, *, run_id, rung, seed, condition, overrides, recall, claim):
    directory = root / "runs" / run_id
    directory.mkdir(parents=True)
    config = cell_config(
        t1_ladder.Cell(name="x", axis="a", level=None, overrides=overrides, condition=condition),
        ladder=rung,
        seed=seed,
    )
    (directory / "manifest.json").write_text(
        json.dumps({"ladder_rung": rung, "parent_claim_packet": claim, "run_id": run_id})
    )
    (directory / "summary.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "config": config.as_dict(),
                "passed": True,
                "verdict": "pilot complete",
                "final": {name: recall for name in t1_ladder.CAPABILITY_METRICS},
                "geometry": {
                    "primary": {name: recall for name in t1_ladder.GEOMETRY_METRICS}
                },
                "mechanism": {
                    "verdict": {name: recall for name in t1_ladder.MECHANISM_METRICS}
                },
                "references": {"skill": {"associative_recall_accuracy": recall}},
            }
        )
    )
    return directory


def test_a_run_naming_another_claim_is_not_this_claim_s_evidence(tmp_path):
    _write_run(
        tmp_path, run_id="R4-a", rung="R4", seed=R4_SEEDS[0], condition="capacity_stressed",
        overrides={}, recall=0.5, claim="claims/somebody-elses.yml",
    )
    variance = seed_variance(seeds=R4_SEEDS[:1], root=tmp_path)
    assert variance["seeds_found"] == []
    assert variance["seeds_missing"] == [R4_SEEDS[0]]


def test_seed_variance_reads_every_family_of_metric(tmp_path):
    for index, seed in enumerate(R4_SEEDS):
        _write_run(
            tmp_path, run_id=f"R4-{seed}", rung="R4", seed=seed, condition="capacity_stressed",
            overrides={}, recall=0.50 + 0.01 * index, claim=t1_ladder.CLAIM,
        )
    variance = seed_variance(seeds=R4_SEEDS, root=tmp_path, mde_replicates=200)
    assert variance["seeds_found"] == sorted(R4_SEEDS)
    assert variance["spread"]["associative_recall_accuracy"]["n"] == 5
    assert variance["spread"]["geometry.mean_purity"]["usable"]
    assert variance["spread"]["mechanism.best_retrieval_lift"]["usable"]
    detectable = variance["detectable_effect"]
    implied = detectable["per_metric"]["associative_recall_accuracy"]
    assert implied["minimum_detectable_difference"] == pytest.approx(
        detectable["minimum_detectable_dz"]
        * math.sqrt(2)
        * variance["spread"]["associative_recall_accuracy"]["sd"]
    )


def test_difficulty_curves_place_each_run_on_the_axis_that_names_it(tmp_path):
    for index, cell in enumerate((*cells(), NEGATIVE_CONTROL_CELL)):
        _write_run(
            tmp_path, run_id=f"R3-{index}", rung="R3", seed=R4_SEEDS[0],
            condition=cell.condition, overrides=cell.overrides,
            recall=0.10 + 0.01 * index, claim=t1_ladder.CLAIM,
        )
    curves = difficulty_curves(root=tmp_path)
    assert curves["base_cell"] is not None
    assert curves["negative_control"] is not None
    for axis, block in curves["axes"].items():
        assert block["n_missing"] == 0, axis
        assert block["points"], axis
        levels = [point["level"] for point in block["points"]]
        assert len(levels) == len(set(map(repr, levels))), axis


def test_the_r1_gate_refuses_a_matrix_on_an_unproven_instrument(tmp_path):
    ok, why = t1_ladder._passed_r1(claim=t1_ladder.CLAIM, root=tmp_path)
    assert not ok and "no R1 run" in why

    directory = tmp_path / "runs" / "R1-failed"
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(
        json.dumps({"ladder_rung": "R1", "parent_claim_packet": t1_ladder.CLAIM})
    )
    (directory / "summary.json").write_text(json.dumps({"run_id": "R1-failed", "passed": False}))
    ok, why = t1_ladder._passed_r1(claim=t1_ladder.CLAIM, root=tmp_path)
    assert not ok and "none passed" in why

    (directory / "summary.json").write_text(json.dumps({"run_id": "R1-failed", "passed": True}))
    ok, _ = t1_ladder._passed_r1(claim=t1_ladder.CLAIM, root=tmp_path)
    assert ok


def test_the_stage_plan_runs_the_rungs_in_the_order_the_ladder_declares():
    assert [config.ladder for _, config, _ in t1_ladder._stage_configs("r1", seeds=R4_SEEDS)] == ["R1"]
    r2 = t1_ladder._stage_configs("r2", seeds=R4_SEEDS)
    assert [config.arch.d_model for _, config, _ in r2] == list(t1_ladder.R2_WIDTHS)
    r3 = t1_ladder._stage_configs("r3", seeds=R4_SEEDS)
    assert len(r3) == len(cells()) + 1
    assert r3[-1][1].data.condition == "negative_control"
    r4 = t1_ladder._stage_configs("r4", seeds=R4_SEEDS)
    assert [config.seed for _, config, _ in r4] == list(R4_SEEDS)
    assert all(config.ladder == "R4" for _, config, _ in r4)


def test_only_r1_asserts_its_own_pass():
    assert t1_ladder._stage_configs("r1", seeds=R4_SEEDS)[0][2] is True
    for stage in ("r2", "r3", "r4"):
        assert not any(assert_pass for _, _, assert_pass in t1_ladder._stage_configs(stage, seeds=R4_SEEDS))


# --------------------------------------------------------------------------- #
# The negative control's verdict rests on a classification made before it
# --------------------------------------------------------------------------- #


def test_only_ceiling_dominated_metrics_can_condemn_the_negative_control():
    from architecture_mechanics.metrics.capability import CEILING_DOMINATED_METRICS

    # A0 above the frequency ceiling is evidence of a leak only where the
    # ceiling provably beats every other input-blind predictor. Prompt 03
    # decided where that is, and excluded the unweighted per-feature averages
    # with a recorded reason; this mission may not widen or narrow the list.
    assert t1_ladder.PRIMARY_METRIC not in CEILING_DOMINATED_METRICS
    assert "feature_macro_recall" not in CEILING_DOMINATED_METRICS
    assert "feature_macro_precision" not in CEILING_DOMINATED_METRICS
    assert set(CEILING_DOMINATED_METRICS) >= {"reconstruction_loss", "brier", "feature_f1"}


def test_the_negative_control_check_says_so_when_there_is_nothing_to_check(tmp_path):
    verdict = t1_ladder.negative_control_check(root=tmp_path)
    assert verdict["measurable"] is False
    assert "no recorded R3 run" in verdict["reason"]


def test_an_axis_narrower_than_the_seed_noise_is_not_resolved():
    points = [
        {"level": 0, "metrics": {t1_ladder.PRIMARY_METRIC: 0.50}},
        {"level": 1, "metrics": {t1_ladder.PRIMARY_METRIC: 0.52}},
    ]
    tight = t1_ladder._axis_resolution(points, 0.001)
    wide = t1_ladder._axis_resolution(points, 0.05)
    assert tight["resolved"] is True
    assert wide["resolved"] is False
    assert wide["range_in_sigma"] < tight["range_in_sigma"]
    # Without an across-seed standard deviation there is no verdict to give,
    # rather than a verdict of "resolved".
    assert t1_ladder._axis_resolution(points, None)["resolved"] is None
