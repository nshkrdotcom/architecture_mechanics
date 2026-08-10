"""What a runner command line *means*, checked without a GPU.

Every assertion here is about a refusal or an expansion that happens before the
first gradient step: which seeds an arm is made of, which task family a rung is
allowed to claim, and which flag combinations contradict each other. That is why
``configs_for`` exists as its own function — the alternative is a test that has
to train a model to discover that two flags disagree.

The one that matters for §7.2 is :func:`test_seeds_are_a_prefix_of_the_frozen_family`.
An arm whose seeds were chosen at a command line is not matched against anything,
and the failure is invisible afterwards: two arms of five runs each look
identical in every report whether or not they share a single seed.
"""

from __future__ import annotations

import json

import pytest

from architecture_mechanics.experiments.config import (
    DEFAULT_SEED,
    SEED_FAMILY,
    RunConfigError,
    ladder_config,
    seed_family,
)
from architecture_mechanics.experiments.runner import (
    TASK_FAMILIES,
    build_parser,
    check_task,
    configs_for,
)


def _configs(*argv: str):
    return configs_for(build_parser().parse_args(list(argv)))


# --------------------------------------------------------------------------- #
# Seeds
# --------------------------------------------------------------------------- #


def test_a_bare_command_is_one_run_at_the_default_seed():
    configs = _configs("--ladder", "R1")
    assert [config.seed for config in configs] == [DEFAULT_SEED]


def test_seeds_are_a_prefix_of_the_frozen_family():
    for count in range(1, len(SEED_FAMILY) + 1):
        configs = _configs("--ladder", "R4", "--seeds", str(count))
        assert [config.seed for config in configs] == list(SEED_FAMILY[:count])


def test_the_five_seed_arm_is_the_first_five_of_the_family():
    assert [config.seed for config in _configs("--ladder", "R4", "--seeds", "5")] == [
        seed for seed in seed_family(5)
    ]


def test_seeds_past_the_end_of_the_family_are_refused_not_invented():
    with pytest.raises(RunConfigError, match="frozen family"):
        _configs("--ladder", "R4", "--seeds", str(len(SEED_FAMILY) + 1))


def test_a_zero_or_negative_arm_is_refused():
    with pytest.raises(RunConfigError, match="at least 1"):
        _configs("--ladder", "R4", "--seeds", "0")


def test_naming_one_seed_and_an_arm_of_seeds_is_a_contradiction():
    with pytest.raises(RunConfigError, match="one or the other"):
        _configs("--ladder", "R4", "--seed", "20260809", "--seeds", "5")


def test_an_arm_differs_only_in_the_seed():
    """§7.2, mechanically: five runs, one field between them."""
    configs = _configs("--ladder", "R4", "--seeds", "5")
    payloads = [config.as_dict() for config in configs]
    for payload in payloads[1:]:
        differing = {
            key for key in payload if payload[key] != payloads[0][key]
        }
        assert differing == {"seed"}


def test_the_arm_is_the_rung_preset_at_each_seed():
    for config in _configs("--ladder", "R4", "--seeds", "5"):
        assert config.as_dict() == ladder_config("R4", seed=config.seed).as_dict()


# --------------------------------------------------------------------------- #
# Task family
# --------------------------------------------------------------------------- #


def test_every_declared_task_family_names_a_generator_family():
    from architecture_mechanics.data.feature_program import CONDITION_NAMES, condition_config

    families = {condition_config(name).family for name in CONDITION_NAMES}
    for family in TASK_FAMILIES.values():
        assert family in families


def test_task_t1_accepts_the_rungs_that_are_on_t1():
    for rung in ("R1", "R2", "R3", "R4"):
        configs = _configs("--ladder", rung, "--task", "t1")
        assert check_task(configs[0], "t1") == "T1"


def test_a_rung_on_another_family_is_refused_rather_than_relabelled(monkeypatch):
    """The refusal cannot fire today — T1 is the only family — so it is fired here."""
    monkeypatch.setitem(TASK_FAMILIES, "t2", "T2")
    with pytest.raises(RunConfigError, match="family T2"):
        check_task(ladder_config("R4"), "t2")


def test_an_unknown_task_family_is_refused():
    with pytest.raises(RunConfigError, match="unknown task family"):
        check_task(ladder_config("R4"), "t99")


# --------------------------------------------------------------------------- #
# Reproducing a recorded config
# --------------------------------------------------------------------------- #


def test_a_recorded_config_already_names_its_seed_and_its_task(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(ladder_config("R4").as_dict()))
    assert _configs("--config-json", str(path))[0].as_dict() == ladder_config("R4").as_dict()
    for contradicting in (("--seeds", "5"), ("--task", "t1")):
        with pytest.raises(RunConfigError, match="contradict"):
            _configs("--config-json", str(path), *contradicting)
