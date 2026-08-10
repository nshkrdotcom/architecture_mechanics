"""Deterministic dataset generation, proven across processes (§8.5).

Same-process determinism is nearly free and nearly worthless: two calls share an
interpreter, an already-imported numpy, and whatever global state the previous
test left. A replication three weeks from now has none of that. So every claim
here compares fresh subprocesses, and each is paired with a control that would
catch a digest which is constant regardless of input.

The batch-size test is the one that is easy to omit and expensive to discover
missing: example 7 must draw the same features whether it was produced alone or
inside a run of 512, or two runs that "used the same data" quietly did not.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

GENERATE_SCRIPT = """
import hashlib, json, sys
from architecture_mechanics.data.feature_program import (
    GENERATOR_VERSION, condition_config, generate_dataset,
)

condition, seed, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
dataset = generate_dataset(condition_config(condition, seed=seed, n_examples=n))

digest = hashlib.sha256()
for name in ("inputs", "targets", "target_mask", "active_mask", "target_active_mask", "importance"):
    digest.update(getattr(dataset, name).numpy().tobytes())

print(json.dumps({
    "content_hash": dataset.content_hash,
    "tensor_digest": digest.hexdigest(),
    "program_digest": hashlib.sha256(
        json.dumps([r.as_dict() for r in dataset.programs], sort_keys=True).encode()
    ).hexdigest(),
    "generator_version": GENERATOR_VERSION,
    "split_fingerprint": dataset.split_plan.fingerprint(),
}))
"""


def _generate(condition: str = "capacity_stressed", seed: int = 20260809, n: int = 96) -> dict:
    proc = subprocess.run(
        [sys.executable, "-c", GENERATE_SCRIPT, condition, str(seed), str(n)],
        capture_output=True,
        text=True,
        env={**os.environ},
        timeout=600,
        check=False,
    )
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_same_config_is_bitwise_identical_across_processes():
    first, second = _generate(), _generate()
    assert first["tensor_digest"] == second["tensor_digest"]
    assert first["program_digest"] == second["program_digest"]
    assert first["content_hash"] == second["content_hash"]


def test_a_different_seed_changes_everything():
    """Guards against a digest that is constant for any input, which would make
    the test above pass while proving nothing."""
    first, other = _generate(seed=20260809), _generate(seed=20260810)
    assert first["tensor_digest"] != other["tensor_digest"]
    assert first["content_hash"] != other["content_hash"]


@pytest.mark.parametrize(
    "condition",
    [
        "positive_control",
        "capacity_stressed",
        "negative_control",
        "lexical_decoy",
        "permutation_control",
        "matched_difficulty",
    ],
)
def test_every_control_condition_is_reproducible_and_distinct(condition):
    first, second = _generate(condition), _generate(condition)
    assert first["content_hash"] == second["content_hash"]
    baseline = _generate("capacity_stressed")
    if condition != "capacity_stressed":
        assert first["content_hash"] != baseline["content_hash"]


BATCH_SCRIPT = """
import hashlib, json, sys
import torch
from architecture_mechanics.data.feature_program import condition_config, generate_dataset

small = generate_dataset(condition_config("capacity_stressed", n_examples=32))
large = generate_dataset(condition_config("capacity_stressed", n_examples=128))
print(json.dumps({
    "inputs_prefix_matches": bool(torch.equal(small.inputs, large.inputs[:32])),
    "targets_prefix_matches": bool(torch.equal(small.targets, large.targets[:32])),
    "programs_prefix_matches": [r.as_dict() for r in small.programs]
        == [r.as_dict() for r in large.programs[:32]],
    "hashes_differ": small.content_hash != large.content_hash,
}))
"""


def test_generation_is_independent_of_how_many_examples_were_asked_for():
    proc = subprocess.run(
        [sys.executable, "-c", BATCH_SCRIPT], capture_output=True, text=True, timeout=600, check=False
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["inputs_prefix_matches"]
    assert result["targets_prefix_matches"]
    assert result["programs_prefix_matches"]
    assert result["hashes_differ"], "a 32-example and a 128-example dataset are not the same dataset"


VERSION_SCRIPT = """
import json
from architecture_mechanics.data.feature_program import (
    GENERATOR_VERSION, condition_config, generate_dataset,
)
dataset = generate_dataset(condition_config("capacity_stressed", n_examples=32))
print(json.dumps({
    "recorded": dataset.content_hash,
    "recomputed": dataset.recompute_hash(),
    "under_other_version": dataset.recompute_hash(generator_version=GENERATOR_VERSION + "-x"),
}))
"""


def test_generator_version_is_part_of_dataset_identity():
    proc = subprocess.run(
        [sys.executable, "-c", VERSION_SCRIPT], capture_output=True, text=True, timeout=600, check=False
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["recorded"] == result["recomputed"]
    assert result["under_other_version"] != result["recorded"]


def test_the_hash_is_not_merely_a_hash_of_the_config():
    """A hash that ignores the tensors cannot detect a generator that drifted
    while its config stayed still."""
    from dataclasses import replace

    import torch

    from architecture_mechanics.data.feature_program import (
        condition_config,
        dataset_content_hash,
        generate_dataset,
    )

    dataset = generate_dataset(condition_config("capacity_stressed", n_examples=16))
    tampered = dataset.tensors()
    tampered["inputs"] = dataset.inputs.clone()
    tampered["inputs"][0, 0, 0] += 1.0
    assert (
        dataset_content_hash(
            cfg=dataset.config,
            banks=dataset.banks,
            plan=dataset.split_plan,
            programs=dataset.programs,
            tensors=tampered,
        )
        != dataset.content_hash
    )

    altered = replace(dataset.programs[0], template_id="0" * 16)
    assert (
        dataset_content_hash(
            cfg=dataset.config,
            banks=dataset.banks,
            plan=dataset.split_plan,
            programs=(altered, *dataset.programs[1:]),
            tensors=dataset.tensors(),
        )
        != dataset.content_hash
    )
    assert torch.is_tensor(dataset.inputs)
