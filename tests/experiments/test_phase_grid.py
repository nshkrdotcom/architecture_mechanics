"""The §10.2 figure 2 sweep grid: what it is, and what it cannot quietly become.

A phase diagram is only readable if the axes mean the same thing at every point
of it. Two of these tests are about exactly that — the sparsity axis and the
bottleneck axis — and they are recomputations rather than assertions about
constants, so a later edit that changes what a cell is fails here rather than in
a figure nobody can check.
"""

from __future__ import annotations

from dataclasses import replace as dataclass_replace

import pytest

import architecture_mechanics.experiments.comparison as comparison_module
import architecture_mechanics.experiments.phase_grid as grid
from architecture_mechanics.data.feature_program import condition_config
from architecture_mechanics.experiments.comparison import (
    DECLARED_COMPARISONS,
    ComparisonError,
    declare,
)
from architecture_mechanics.experiments.config import DEFAULT_SEED, config_fingerprint
from architecture_mechanics.experiments.phase_grid import (
    PHASE_CELLS,
    PHASE_COMPARISONS,
    PHASE_CONTENT_FEATURES,
    PHASE_CUTS,
    PHASE_FIXED_FEATURES,
    PHASE_GROUP_SIZE,
    PHASE_LADDER,
    PHASE_MAIN_CELLS,
    PHASE_MAIN_SEQ_LEN,
    PHASE_NEGATIVE_CONTROL_CELL,
    PHASE_SPARSITIES,
    PHASE_WIDTH_FEATURE_POINTS,
    cell_axes,
    phase_cost_model,
)
from architecture_mechanics.experiments.t1_ladder import cell_config


def test_the_sparsity_axis_means_the_same_thing_at_every_feature_bank_width():
    """The property the whole x-axis rests on.

    ``activation_prob`` is per-feature *within the group a position draws from*,
    so holding the group count fixed while F moves would make one column of the
    figure denser at the bottom than at the top and the map would be unreadable.
    Recomputed from the generator configuration each cell actually builds, not
    from the constant.
    """
    for cell in PHASE_MAIN_CELLS:
        cfg = cell_config(cell, ladder=PHASE_LADDER).data.generator_config()
        group_size = cfg.n_content_features / cfg.n_content_groups
        assert group_size == PHASE_GROUP_SIZE
        assert cfg.n_content_features in PHASE_CONTENT_FEATURES
        assert cfg.activation_prob in PHASE_SPARSITIES


def test_a_feature_bank_that_would_split_unevenly_is_refused():
    """A partial group would silently change the sparsity axis in one row."""
    from architecture_mechanics.experiments import phase_grid

    with pytest.raises(ValueError, match="group size"):
        phase_grid._cell(f_content=40, seq_len=32, activation_prob=0.12, axis="phase")


def test_the_map_spans_the_superposition_transition_in_both_directions():
    """F/d must cross 1, or the figure cannot mark the transition it claims to."""
    ratios = sorted(f / d for f, d in PHASE_WIDTH_FEATURE_POINTS)
    assert min(ratios) < 1.0
    assert max(ratios) > 1.0
    assert 1.0 in ratios


def test_three_bottleneck_ratios_are_realised_twice_so_the_ratio_can_be_tested():
    """The reason the grid stays two-dimensional instead of collapsing onto F/d.

    A map drawn against the ratio alone could not ask whether the ratio is the
    controlling variable. Two (F, d) points at one ratio can.
    """
    by_ratio: dict[float, list[tuple[int, int]]] = {}
    for f_content, width in PHASE_WIDTH_FEATURE_POINTS:
        by_ratio.setdefault(f_content / width, []).append((f_content, width))
    duplicated = {ratio: points for ratio, points in by_ratio.items() if len(points) > 1}
    assert len(duplicated) == 3
    for points in duplicated.values():
        assert len({width for _, width in points}) == len(points)


def test_the_whole_bank_ratio_is_recorded_beside_the_content_ratio():
    """F here is the content bank and the figure says so; both must be derivable."""
    for cell in PHASE_MAIN_CELLS:
        axes = cell_axes(cell.name)
        assert axes["f_total"] == axes["f_content"] + PHASE_FIXED_FEATURES
        cfg = cell_config(cell, ladder=PHASE_LADDER).data.generator_config()
        assert cfg.n_key_features + 4 == PHASE_FIXED_FEATURES


def test_every_cell_is_the_capacity_stressed_condition_with_four_fields_moved():
    """A cell is a dataset. Anything else moving would be an undeclared change."""
    base = condition_config("capacity_stressed")
    moved = {"n_content_features", "n_content_groups", "seq_len", "activation_prob"}
    for cell in PHASE_CELLS:
        assert set(cell.overrides) == moved
        if cell is PHASE_NEGATIVE_CONTROL_CELL:
            assert cell.condition == "negative_control"
            continue
        assert cell.condition == "capacity_stressed"
        cfg = cell_config(cell, ladder=PHASE_LADDER).data.generator_config()
        for field in ("n_associations", "n_distractors", "key_collisions", "distance_buckets"):
            assert getattr(cfg, field) == getattr(base, field)


def test_the_grid_runs_at_the_declared_screening_rung_and_one_seed():
    for name, spec in PHASE_COMPARISONS.items():
        assert spec["claim_id"] == "phase-map-a0-a1-sparsity-bottleneck"
        assert spec["control_arch"] == "softmax"
        assert spec["candidate_archs"] == ("linear",)
        rung, seeds = next(iter(spec["rungs"].items()))
        assert seeds == 1
        assert rung == ("R1" if name == "phase_r1" else PHASE_LADDER)


def test_the_sweep_declares_its_own_positive_and_negative_controls():
    """§7.3's R1 is not optional for a mission, and a map needs its own null."""
    assert PHASE_COMPARISONS["phase_r1"]["cells"] == ("positive-control",)
    assert next(iter(PHASE_COMPARISONS["phase_r1"]["rungs"])) == "R1"
    assert PHASE_NEGATIVE_CONTROL_CELL.condition == "negative_control"
    assert PHASE_NEGATIVE_CONTROL_CELL.name in PHASE_COMPARISONS[
        "phase_negative_control_d32"
    ]["cells"]


def test_every_planned_pair_is_matched_before_any_gpu_is_spent():
    """Construction is the inner gate; it must produce no undeclared difference.

    The count checked at the end is the number of *distinct configurations*, and
    that is what the grid was priced on: both §7.2 matching strategies resolve to
    the same runs here, so the sweep costs one run per arm per cell and not two.
    """
    configurations = set()
    for name in PHASE_COMPARISONS:
        for plan in declare(name, write=False):
            pairs = plan.pairs()
            assert pairs
            for pair in pairs:
                assert pair.permitted_differences == {}
                assert pair.seed == DEFAULT_SEED
                assert set(pair.differences) <= {"arch.arch"}
                configurations.add(config_fingerprint(pair.control))
                configurations.add(config_fingerprint(pair.candidate))
    assert len(configurations) == phase_cost_model()["planned_runs"]


def test_the_two_matching_strategies_coincide_because_the_counts_agree():
    """A0 and A1 have identical parameter counts at every width in this lab, so
    neither strategy can be the flattering one that gets reported."""
    for name in PHASE_COMPARISONS:
        for plan in declare(name, write=False):
            for pair in plan.pairs():
                assert pair.parameters["parameter_difference"] == 0


def test_the_grid_is_merged_into_the_one_comparison_registry():
    for name in PHASE_COMPARISONS:
        assert DECLARED_COMPARISONS[name] == PHASE_COMPARISONS[name]


def test_a_cell_name_collision_between_the_two_registries_is_refused():
    """The cell name is what a resolved declaration and every table are keyed on.

    Exercised by giving the phase registry a name the R3 task matrix already
    owns, which is the only way the two could ever collide in practice.
    """
    saved = grid.PHASE_CELLS
    grid.PHASE_CELLS = (*saved, dataclass_replace(saved[0], name="base"))
    try:
        with pytest.raises(ComparisonError, match="two registries"):
            comparison_module._known_cells()
    finally:
        grid.PHASE_CELLS = saved
    assert comparison_module._known_cells()["base"].axis == "base"


def test_the_cuts_are_recorded_with_their_price_rather_than_implied():
    """A grid cut whose arithmetic is not in the repository is a preference."""
    assert len(PHASE_CUTS) >= 3
    for entry in PHASE_CUTS:
        assert set(entry) == {"cut", "kept", "cost_if_kept", "why_this_axis"}
        for value in entry.values():
            assert value.strip()
    model = phase_cost_model()
    assert len(model["probes"]) >= 4
    for probe in model["probes"]:
        assert probe["estimated_full_run_s"] > probe["wall_at_100_s"]


def test_cell_axes_reads_the_registry_and_not_the_name():
    with pytest.raises(KeyError):
        cell_axes("phase-F999-T32-p012")
    axes = cell_axes(PHASE_MAIN_CELLS[0].name)
    assert axes["seq_len"] == PHASE_MAIN_SEQ_LEN
    assert axes["expected_active_content_features"] == pytest.approx(
        PHASE_GROUP_SIZE * axes["activation_prob"]
    )
