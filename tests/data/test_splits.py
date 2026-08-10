"""Splits are by program template and held-out composition, never by example.

The property under test is not just "train and test are disjoint" — a random
example split satisfies that too and means nothing. It is that the test set is
made of *combinations* the training set never showed while every *part* of those
combinations is familiar. A test failure that only shows up on novel parts is a
different, weaker result than one that shows up on novel compositions.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from architecture_mechanics.data.feature_program import (
    condition_config,
    generate_dataset,
    t0_config,
)
from architecture_mechanics.data.splits import (
    SplitError,
    build_templates,
    split_templates,
)
from architecture_mechanics.data.task_families import get_family


def _t1_templates(**overrides):
    config = condition_config("capacity_stressed", **overrides)
    return get_family("T1").templates(config), config


def test_template_grid_is_the_full_product():
    templates, config = _t1_templates()
    expected = (
        len(config.operations)
        * config.n_content_groups
        * config.n_key_groups
        * len(config.distance_buckets)
    )
    assert len(templates) == expected
    assert len({t.template_id for t in templates}) == expected


@pytest.mark.parametrize("family_name", ["T0", "T1"])
def test_train_and_test_templates_are_disjoint(family_name):
    config = t0_config() if family_name == "T0" else condition_config("capacity_stressed")
    family = get_family(family_name)
    plan = split_templates(
        family.templates(config),
        seed=config.seed,
        holdout_fraction=config.holdout_fraction,
        require_axis_coverage=family.axis_coverage,
    )
    report = plan.report()
    assert report["template_id_overlap"] == 0
    assert report["n_train_templates"] > 0
    assert report["n_test_templates"] > 0
    assert report["n_heldout_compositions"] == report["n_test_templates"]
    assert not set(plan.train) & set(plan.test)


@pytest.mark.parametrize("family_name", ["T0", "T1"])
def test_every_test_axis_value_is_familiar_from_training(family_name):
    """This is what makes the holdout compositional rather than out-of-vocabulary."""
    config = t0_config() if family_name == "T0" else condition_config("capacity_stressed")
    family = get_family(family_name)
    plan = split_templates(
        family.templates(config),
        seed=config.seed,
        holdout_fraction=config.holdout_fraction,
        require_axis_coverage=family.axis_coverage,
    )
    report = plan.report()
    assert report["test_axis_values_absent_from_train"] == {}
    assert report["compositional"] is True


def test_held_out_compositions_are_reported_with_their_count():
    templates, config = _t1_templates()
    plan = split_templates(templates, seed=config.seed, holdout_fraction=0.25)
    report = plan.report()
    assert report["n_heldout_compositions"] == len(report["heldout_compositions"])
    assert report["n_heldout_compositions"] > 0
    train_compositions = {t.composition for t in plan.train}
    for composition in plan.heldout_compositions:
        assert composition not in train_compositions


def test_examples_only_ever_use_their_own_split_templates():
    train = generate_dataset(condition_config("capacity_stressed", n_examples=128))
    from dataclasses import replace

    test = generate_dataset(replace(train.config, split="test"))
    train_ids = {t.template_id for t in train.split_plan.templates_for("train")}
    test_ids = {t.template_id for t in train.split_plan.templates_for("test")}
    assert {r.template_id for r in train.programs} <= train_ids
    assert {r.template_id for r in test.programs} <= test_ids
    assert not ({r.template_id for r in test.programs} & train_ids)


def test_the_split_is_deterministic_and_seed_sensitive():
    templates, _ = _t1_templates()
    first = split_templates(templates, seed=7, holdout_fraction=0.25)
    again = split_templates(templates, seed=7, holdout_fraction=0.25)
    other = split_templates(templates, seed=8, holdout_fraction=0.25)
    assert first.fingerprint() == again.fingerprint()
    assert first.fingerprint() != other.fingerprint()


def test_a_single_axis_grid_cannot_produce_a_compositional_split():
    """Refusing loudly beats quietly returning a split that is not compositional."""
    # Two varying axes, every value appearing twice or more: a compositional
    # holdout exists and is found.
    templates = build_templates(
        family="T1",
        operations=("recall_by_key",),
        content_groups=(0, 1, 2),
        key_groups=(0,),
        distance_buckets=((5, 9), (10, 16)),
        n_distractors=0,
        n_associations=2,
        key_collisions=False,
    )
    plan = split_templates(templates, seed=1, holdout_fraction=0.2)
    assert plan.report()["compositional"] is True
    assert plan.report()["test_axis_values_absent_from_train"] == {}

    tiny = build_templates(
        family="T1",
        operations=("recall_by_key",),
        content_groups=(0,),
        key_groups=(0,),
        distance_buckets=((5, 9), (10, 16)),
        n_distractors=0,
        n_associations=2,
        key_collisions=False,
    )
    # Each distance bucket appears exactly once, so nothing can be held out
    # without orphaning its axis value.
    with pytest.raises(SplitError, match="too few varying axes"):
        split_templates(tiny, seed=1, holdout_fraction=0.5)
    fallback = split_templates(tiny, seed=1, holdout_fraction=0.5, require_axis_coverage=False)
    assert fallback.report()["compositional"] is False
    assert fallback.report()["template_id_overlap"] == 0


def test_bad_holdout_fractions_and_empty_grids_are_refused():
    templates, _ = _t1_templates()
    with pytest.raises(SplitError, match="holdout_fraction"):
        split_templates(templates, seed=1, holdout_fraction=0.0)
    with pytest.raises(SplitError, match="empty template grid"):
        split_templates((), seed=1)
    with pytest.raises(SplitError, match="unknown split"):
        split_templates(templates, seed=1).templates_for("probe")


_ID_SCRIPT = """
import json
from architecture_mechanics.data.feature_program import condition_config
from architecture_mechanics.data.task_families import get_family
config = condition_config("capacity_stressed")
templates = get_family("T1").templates(config)
print(json.dumps([t.template_id for t in templates]))
"""


def test_template_ids_are_stable_across_processes():
    """Python's hash() is salted per interpreter; a template id must not be."""
    runs = [
        subprocess.run(
            [sys.executable, "-c", _ID_SCRIPT], capture_output=True, text=True, timeout=300, check=True
        ).stdout.strip()
        for _ in range(2)
    ]
    assert runs[0] == runs[1]
