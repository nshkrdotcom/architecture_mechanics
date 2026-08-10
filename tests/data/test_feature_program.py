"""Semantics of the generator, and the six §4.4 controls being what they claim.

The tests that matter most here are the negative-control pair. A "negative
control" that is merely hard silently validates every false result the programme
will ever produce, so it is checked twice and from both directions: an oracle
with perfect memory cannot beat the marginal on it, and the same oracle scores
1.0 on the positive control, which is what rules out "the oracle is just weak".
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from architecture_mechanics.data.feature_program import (
    GENERATOR_VERSION,
    FeatureProgramConfig,
    FeatureProgramError,
    activation_probs,
    answer_appears_in_input,
    build_key_table,
    condition_config,
    draw_content,
    generate_dataset,
    group_ranges,
    perfect_memory_oracle_report,
    phase_diagram_grid,
    t0_config,
)
from architecture_mechanics.data.task_families import get_family

NEGATIVE_CONTROL_TOLERANCE = 0.05
ORACLE_EXAMPLES = 768


@pytest.fixture(scope="module")
def capacity():
    return generate_dataset(condition_config("capacity_stressed", n_examples=ORACLE_EXAMPLES))


# --------------------------------------------------------------------------- #
# Ground truth
# --------------------------------------------------------------------------- #


def test_t0_supervises_every_position_on_itself():
    dataset = generate_dataset(t0_config(n_examples=32))
    assert bool(dataset.target_mask.all())
    assert torch.equal(dataset.inputs, dataset.targets)
    assert dataset.banks.n_key == 0 and dataset.banks.n_op == 0
    assert dataset.n_features == dataset.config.n_content_features


def test_t1_target_is_exactly_the_source_content(capacity):
    content = list(capacity.content_indices)
    for record in capacity.programs[:64]:
        example = record.example_index
        for step in record.steps:
            assert step.source is not None
            expected = capacity.inputs[example, step.source, content]
            assert torch.equal(capacity.targets[example, step.dest, content], expected)
            assert capacity.targets[example, step.dest].nonzero().numel() == len(
                step.answer_features
            )


def test_program_record_agrees_with_the_tensors(capacity):
    """The record must be readable without the tensors, which means it has to
    agree with them everywhere, not approximately."""
    for record in capacity.programs[:32]:
        example = record.example_index
        assert record.seq_len == len(record.positions)
        supervised = tuple(int(t) for t in capacity.target_mask[example].nonzero().flatten())
        assert record.supervised_positions == supervised
        for position in record.positions:
            active = tuple(
                int(f) for f in capacity.active_mask[example, position.index].nonzero().flatten()
            )
            assert position.active_features == active
            assert set(position.key_features) <= set(capacity.key_indices)
        for step in record.steps:
            answer = tuple(
                int(f)
                for f in capacity.target_active_mask[example, step.dest].nonzero().flatten()
            )
            assert step.answer_features == answer


def test_answer_features_lie_in_the_template_content_group(capacity):
    blocks = group_ranges(
        capacity.config.n_content_features, capacity.config.n_content_groups
    )
    for record in capacity.programs[:64]:
        for step in record.steps:
            assert set(step.answer_features) <= set(blocks[step.answer_group])


# --------------------------------------------------------------------------- #
# Control 1 and 2: positive control and capacity stress
# --------------------------------------------------------------------------- #


def test_positive_control_has_ample_dimension_and_no_gauntlet():
    dataset = generate_dataset(condition_config("positive_control", n_examples=64))
    config = dataset.config
    assert config.d_recommended > dataset.n_features, "positive control must have d > F"
    assert config.n_distractors == 0
    assert config.n_associations == 1
    assert config.key_collisions is False
    assert max(high for _, high in config.distance_buckets) <= 2
    for record in dataset.programs:
        for step in record.steps:
            assert step.distractors == ()


def test_capacity_stressed_is_stressed(capacity):
    assert capacity.n_features / capacity.config.d_recommended >= 4.0
    density = capacity.summary()["global_density"]
    assert density < 0.06, f"capacity-stressed features should be sparse, got {density}"
    assert capacity.config.n_associations > 1
    assert capacity.config.n_distractors > 0


# --------------------------------------------------------------------------- #
# Control 3: the negative control must be genuinely impossible
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def negative():
    return generate_dataset(condition_config("negative_control", n_examples=ORACLE_EXAMPLES))


def test_a_perfect_memory_oracle_cannot_exceed_chance_on_the_negative_control(negative):
    report = perfect_memory_oracle_report(negative)
    assert report.best_honest_r2 <= NEGATIVE_CONTROL_TOLERANCE, (
        f"strategy {report.best_honest_strategy!r} reached R^2={report.best_honest_r2} on a "
        f"control that is supposed to carry no information\n{report.table()}"
    )
    for name, score in report.scores.items():
        assert score <= NEGATIVE_CONTROL_TOLERANCE, f"{name} beat chance: {score}"


def test_the_same_oracle_solves_the_positive_control():
    """Without this, the test above is satisfied by an oracle that can do nothing."""
    dataset = generate_dataset(condition_config("positive_control", n_examples=ORACLE_EXAMPLES))
    report = perfect_memory_oracle_report(dataset)
    assert report.scores["key_match_exact"] >= 0.95, report.table()


def test_the_negative_control_answer_is_not_in_its_own_input(negative):
    assert answer_appears_in_input(negative) == 0


def test_the_negative_control_records_the_destruction(negative):
    for record in negative.programs[:64]:
        for step in record.steps:
            assert step.information_destroyed is True
            assert step.source is None
            assert step.distance is None
        roles = {p.role for p in record.positions}
        assert "destroyed_source" in roles


def test_the_negative_control_edits_exactly_one_position(capacity, negative):
    """It is a matched control: it differs from capacity-stress at the destroyed
    source and nowhere else, so a capability gap cannot be blamed on the data
    having been shuffled."""
    differing = (capacity.inputs != negative.inputs).any(dim=-1)
    assert int(differing.sum(dim=1).max()) == 1
    for record in negative.programs[:32]:
        destroyed = [p.index for p in record.positions if p.role == "destroyed_source"]
        changed = differing[record.example_index].nonzero().flatten().tolist()
        assert changed == destroyed


def test_no_position_carries_the_queried_key_after_destruction(negative):
    for record in negative.programs[:64]:
        query = next(p for p in record.positions if p.role == "query")
        matches = [
            p.index
            for p in record.positions
            if p.index != query.index and p.key_features == query.key_features
        ]
        assert matches == []


# --------------------------------------------------------------------------- #
# Control 4: lexical decoy
# --------------------------------------------------------------------------- #


def test_lexical_decoys_look_like_operations_and_change_nothing(capacity):
    decoy = generate_dataset(condition_config("lexical_decoy", n_examples=ORACLE_EXAMPLES))
    assert torch.equal(capacity.targets, decoy.targets)
    assert torch.equal(capacity.target_mask, decoy.target_mask)

    decoy_positions = {
        (r.example_index, p.index)
        for r in decoy.programs
        for p in r.positions
        if p.role == "decoy_op"
    }
    assert len(decoy_positions) == decoy.n_examples * decoy.config.n_decoys
    differing = (capacity.inputs != decoy.inputs).any(dim=-1).nonzero().tolist()
    assert {(int(e), int(t)) for e, t in differing} == decoy_positions

    noop_slot = decoy.op_indices[decoy.banks.op_codes.index("NOOP")]
    for example, position in sorted(decoy_positions)[:32]:
        assert decoy.inputs[example, position, noop_slot] == 1.0
        assert not decoy.inputs[example, position, list(decoy.content_indices)].any()
        assert decoy.inputs[example, position, list(decoy.key_indices)].any()

    for record in decoy.programs[:32]:
        sources = {s.source for s in record.steps}
        for position in record.positions:
            if position.role == "decoy_op":
                assert position.index not in sources
                assert position.index not in record.supervised_positions


# --------------------------------------------------------------------------- #
# Control 5: permutation
# --------------------------------------------------------------------------- #


def test_permuting_feature_ids_is_an_isomorphism(capacity):
    permuted = generate_dataset(
        condition_config("permutation_control", n_examples=ORACLE_EXAMPLES)
    )
    inverse = np.argsort(np.asarray(permuted.feature_permutation))
    assert torch.equal(permuted.inputs[..., inverse], capacity.inputs)
    assert torch.equal(permuted.targets[..., inverse], capacity.targets)
    assert torch.equal(permuted.active_mask[..., inverse], capacity.active_mask)
    assert torch.equal(permuted.importance[inverse], capacity.importance)
    assert permuted.content_hash != capacity.content_hash

    base_report = perfect_memory_oracle_report(capacity)
    permuted_report = perfect_memory_oracle_report(permuted)
    for name, score in base_report.scores.items():
        assert permuted_report.scores[name] == pytest.approx(score, abs=1e-6)


def test_permutation_relabels_route_labels_too():
    permuted = generate_dataset(condition_config("permutation_control", n_examples=64))
    assert permuted.key_permutation is not None
    assert sorted(permuted.key_permutation) == list(range(permuted.config.n_keys))
    recorded = {p.key_id for r in permuted.programs for p in r.positions if p.key_id is not None}
    assert recorded <= set(permuted.key_permutation)


def test_permuted_record_indices_match_permuted_tensors():
    permuted = generate_dataset(condition_config("permutation_control", n_examples=64))
    for record in permuted.programs[:16]:
        for position in record.positions:
            active = tuple(
                int(f)
                for f in permuted.active_mask[record.example_index, position.index]
                .nonzero()
                .flatten()
            )
            assert position.active_features == active


# --------------------------------------------------------------------------- #
# Control 6: matched difficulty
# --------------------------------------------------------------------------- #


def test_matched_difficulty_shares_inputs_and_changes_the_operation(capacity):
    matched = generate_dataset(condition_config("matched_difficulty", n_examples=ORACLE_EXAMPLES))
    assert torch.equal(capacity.inputs, matched.inputs), "marginals must be matched exactly"
    assert torch.equal(capacity.active_mask, matched.active_mask)
    assert torch.equal(capacity.target_mask, matched.target_mask)
    assert not torch.equal(capacity.targets, matched.targets)
    assert {s.op for r in matched.programs for s in r.steps} == {"recall_first_binding"}


def test_matched_difficulty_is_solvable_but_not_by_content_addressing():
    """A matched control that nothing can solve is a second negative control.
    The answer must be recoverable — just by a different operation."""
    matched = generate_dataset(condition_config("matched_difficulty", n_examples=ORACLE_EXAMPLES))
    report = perfect_memory_oracle_report(matched)
    assert report.scores["copy_first_keyed"] >= 0.99, report.table()
    assert report.scores["key_match_exact"] < 0.0, (
        "content addressing must give the wrong answer here, or the two operations agree"
    )


def test_matched_difficulty_answers_the_first_binding_not_the_matching_key(capacity):
    matched = generate_dataset(condition_config("matched_difficulty", n_examples=256))
    disagreements = 0
    for record, base_record in zip(matched.programs, capacity.programs, strict=False):
        bindings = sorted(
            p.index for p in record.positions if p.role in ("binding", "source_binding")
        )
        step = record.steps[0]
        assert step.source == bindings[0]
        if step.source != base_record.steps[0].source:
            disagreements += 1
    assert disagreements == len(matched.programs), (
        "content addressing and ordinal addressing must never agree, or the control "
        "is not testing a different operation"
    )


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #


def test_draw_content_respects_activation_probability():
    rng = np.random.default_rng(0)
    probs = np.full(64, 0.25)
    indices = np.arange(64)
    counts = [int(draw_content(rng, probs, indices, 1)[1].sum()) for _ in range(400)]
    assert np.mean(counts) == pytest.approx(16.0, rel=0.08)


def test_draw_content_never_returns_an_empty_target():
    rng = np.random.default_rng(0)
    probs = np.full(8, 0.01)
    indices = np.arange(8)
    for _ in range(200):
        values, active = draw_content(rng, probs, indices, 1)
        assert int(active.sum()) >= 1
        assert np.all(values[~active] == 0.0)


def test_power_law_activation_profile_is_decreasing_and_normalised():
    config = FeatureProgramConfig(
        family="T0",
        n_content_features=64,
        n_key_features=0,
        n_keys=0,
        key_bits=0,
        n_key_groups=1,
        activation_profile="power_law",
        activation_alpha=0.8,
        activation_prob=0.1,
        operations=("reconstruct",),
        distance_buckets=((0, 0),),
        seq_len=8,
    )
    probs = activation_probs(config)
    assert np.all(np.diff(probs) <= 0)
    assert probs.mean() == pytest.approx(0.1, rel=0.02)


def test_importance_is_power_law_over_the_content_bank_and_zero_elsewhere(capacity):
    weights = capacity.importance.numpy()
    content = np.asarray(capacity.content_indices)
    non_content = np.asarray(capacity.key_indices + capacity.op_indices)
    assert np.all(np.diff(weights[content]) <= 0)
    assert weights[content][0] == pytest.approx(1.0)
    assert weights[content][-1] < 0.1
    assert np.all(weights[non_content] == 0.0)


def test_key_subsets_are_unique_and_stay_inside_their_group():
    config = condition_config("capacity_stressed")
    banks = get_family("T1").banks(config)
    table = build_key_table(config, banks)
    assert len(set(table)) == config.n_keys
    blocks = group_ranges(config.n_key_features, config.n_key_groups)
    for key_id, bits in enumerate(table):
        assert len(bits) == config.key_bits
        assert set(bits) <= set(blocks[key_id % config.n_key_groups])


def test_key_table_does_not_depend_on_the_split():
    config = condition_config("capacity_stressed")
    banks = get_family("T1").banks(config)
    from dataclasses import replace

    assert build_key_table(config, banks) == build_key_table(
        replace(config, split="test"), banks
    )


# --------------------------------------------------------------------------- #
# Configuration guards
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"n_distractors": 9}, "cannot fit strictly between"),
        ({"seq_len": 8}, "needs seq_len"),
        ({"n_keys": 6, "n_associations": 6}, "must exceed n_associations"),
        ({"activation_prob": 0.0}, "activation_prob"),
        ({"activation_profile": "lognormal"}, "activation_profile"),
        ({"key_bits": 8}, "near-miss"),
    ],
)
def test_impossible_geometry_is_refused(overrides, fragment):
    with pytest.raises(FeatureProgramError, match=fragment):
        condition_config("capacity_stressed", **overrides)


def test_unknown_condition_and_family_are_refused():
    with pytest.raises(FeatureProgramError, match="unknown condition"):
        condition_config("wishful_thinking")
    with pytest.raises(FeatureProgramError, match="unknown task family"):
        generate_dataset(condition_config("capacity_stressed", family="T9", n_examples=2))


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


def test_generator_version_participates_in_the_hash(capacity):
    assert capacity.generator_version == GENERATOR_VERSION
    assert capacity.recompute_hash() == capacity.content_hash
    assert capacity.recompute_hash(generator_version="fpg-0.0.0") != capacity.content_hash


def test_train_and_test_splits_hash_differently(capacity):
    from dataclasses import replace

    test_split = generate_dataset(replace(capacity.config, split="test"))
    assert test_split.content_hash != capacity.content_hash
    train_ids = {t.template_id for t in capacity.split_plan.templates_for("train")}
    used = {r.template_id for r in test_split.programs}
    assert not (used & train_ids)


# --------------------------------------------------------------------------- #
# Phase-diagram grid (prompt 14's Figure 2)
# --------------------------------------------------------------------------- #


def test_phase_diagram_grid_covers_sparsity_by_f_over_d():
    from dataclasses import replace

    grid = phase_diagram_grid(sparsities=(0.05, 0.2), f_over_d=(1.0, 4.0), n_content_features=64)
    assert len(grid) == 4
    assert {c.activation_prob for c in grid} == {0.05, 0.2}
    assert {c.d_recommended for c in grid} == {64, 16}
    assert all(c.family == "T0" for c in grid)
    hashes = {generate_dataset(replace(c, n_examples=8)).content_hash for c in grid}
    assert len(hashes) == 4, "every cell of the grid must be a distinct dataset"
