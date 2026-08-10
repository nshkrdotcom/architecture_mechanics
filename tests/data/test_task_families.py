"""T0 and T1 layout semantics, and the T2–T5 stubs refusing loudly.

The layout assertions are the ones an automated invariant in the selftest cannot
express: that the distance really lies in the template's bucket, that the
distractors really sit between source and destination, and that exactly one
position carries the queried key. An off-by-one in any of them would still
produce a dataset that trains.
"""

from __future__ import annotations

import pytest

from architecture_mechanics.data.feature_program import (
    FeatureProgramError,
    condition_config,
    generate_dataset,
    t0_config,
)
from architecture_mechanics.data.task_families import (
    FAMILY_NAMES,
    IMPLEMENTED_FAMILIES,
    get_family,
)

BINDING_ROLES = ("source_binding", "binding", "collision_binding")


@pytest.fixture(scope="module")
def t1():
    return generate_dataset(condition_config("capacity_stressed", n_examples=192))


def test_registry_lists_every_family_and_names_the_implemented_ones():
    assert FAMILY_NAMES == ("T0", "T1", "T2", "T3", "T4", "T5")
    assert IMPLEMENTED_FAMILIES == ("T0", "T1")
    with pytest.raises(FeatureProgramError, match="unknown task family"):
        get_family("T6")


@pytest.mark.parametrize(("name", "prompt"), [("T2", "18"), ("T3", "37-42"), ("T4", "37-42"), ("T5", "61-90")])
def test_unimplemented_families_name_their_owning_prompt(name, prompt):
    family = get_family(name)
    assert family.__doc__ is not None and len(family.__doc__) > 200, (
        "a stub without a design is a TODO, not a documented stub"
    )
    for call in (
        lambda: family.banks(t0_config()),
        lambda: family.templates(t0_config()),
        lambda: family.plan_example(),
    ):
        with pytest.raises(NotImplementedError) as excinfo:
            call()
        assert "documented stub" in str(excinfo.value)
        assert prompt in str(excinfo.value)


def test_t0_has_no_transport(t1):
    dataset = generate_dataset(t0_config(n_examples=32))
    for record in dataset.programs[:8]:
        assert len(record.steps) == record.seq_len
        for step in record.steps:
            assert step.source == step.dest
            assert step.distance == 0
            assert step.distractors == ()
        assert {p.role for p in record.positions} == {"content"}


def test_t0_positions_draw_from_the_template_group_pair():
    dataset = generate_dataset(t0_config(n_examples=32))
    for record in dataset.programs[:8]:
        expected = tuple(sorted({record.composition[1], record.composition[2]}))
        for position in record.positions:
            assert position.content_groups == expected


def test_t1_layout_matches_the_template(t1):
    config = t1.config
    for record in t1.programs:
        query = [p for p in record.positions if p.role == "query"]
        assert len(query) == 1
        assert query[0].index == config.seq_len - 1
        assert not query[0].content_groups

        bindings = [p for p in record.positions if p.role in BINDING_ROLES]
        assert len(bindings) == config.n_associations

        step = record.steps[0]
        assert step.dest == query[0].index
        low, high = config.distance_buckets[record.composition[3]]
        assert low <= step.distance <= high
        assert step.source == step.dest - step.distance

        distractors = {p.index for p in record.positions if p.role == "distractor"}
        assert len(distractors) == config.n_distractors
        assert all(step.source < d < step.dest for d in distractors)
        assert set(step.distractors) == distractors


def test_t1_query_key_matches_exactly_one_binding(t1):
    for record in t1.programs:
        query = next(p for p in record.positions if p.role == "query")
        matching = [
            p
            for p in record.positions
            if p.role in BINDING_ROLES and p.key_features == query.key_features
        ]
        assert len(matching) == 1
        assert matching[0].role == "source_binding"
        assert matching[0].key_id == query.key_id == record.steps[0].key_id
        # The routing assertion, and the only one here that does not go through
        # `step.source`. Everything else in this file — and the program oracle,
        # and every capability metric, and the §6.3 retrieval lift — reads the
        # source position the generator wrote down, so an error in *choosing* it
        # cancels everywhere at once. This locates the source by key match, from
        # the position records, and requires it to be the position the step
        # names. Added by prompt 10 after a deliberate off-by-one in
        # `plan_example` left the oracle at 1.0000 and both selftests green.
        assert matching[0].index == record.steps[0].source


def test_t1_always_places_a_binding_before_the_source(t1):
    """Otherwise recall_first_binding collapses onto recall_by_key and the
    matched-difficulty control stops being a different operation."""
    for record in t1.programs:
        step = record.steps[0]
        earlier = [
            p.index for p in record.positions if p.role in BINDING_ROLES and p.index < step.source
        ]
        assert earlier


def test_key_collisions_produce_a_near_miss_not_a_duplicate():
    dataset = generate_dataset(
        condition_config("capacity_stressed", n_examples=96, key_collisions=True)
    )
    config = dataset.config
    for record in dataset.programs:
        query = next(p for p in record.positions if p.role == "query")
        colliders = [p for p in record.positions if p.role == "collision_binding"]
        assert len(colliders) == 1
        collider = colliders[0]
        assert collider.key_id is None
        shared = set(collider.key_features) & set(query.key_features)
        assert len(shared) == config.key_bits - 1
        assert collider.key_features != query.key_features


def test_supervising_content_adds_reconstruction_steps_without_moving_the_query():
    plain = generate_dataset(condition_config("capacity_stressed", n_examples=32))
    both = generate_dataset(
        condition_config("capacity_stressed", n_examples=32, supervise_content=True)
    )
    assert int(both.target_mask.sum()) > int(plain.target_mask.sum())
    for record in both.programs[:8]:
        recall = [s for s in record.steps if s.op == "recall_by_key"]
        assert len(recall) == 1
        assert all(s.source == s.dest for s in record.steps if s.op == "reconstruct")


def test_t1_refuses_an_operation_it_does_not_implement():
    with pytest.raises(FeatureProgramError, match="does not implement operation"):
        generate_dataset(
            condition_config("capacity_stressed", n_examples=2, operations=("recall_by_vibes",))
        )
