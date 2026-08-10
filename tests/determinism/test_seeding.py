"""Determinism is proven across fresh processes, not across two calls in one.

Two calls in one process share an interpreter, an already-imported torch, an
already-created CUDA context, and whatever global state a previous test left
behind. That is exactly the state a replication three weeks from now will not
have, so a same-process check can pass while the property it claims to test is
false. Every assertion here therefore compares subprocesses.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch

# Draws from all three generators, plus a CUDA draw when a device is present,
# hashed bytewise. Printed as a single line so the parent compares strings.
DRAW_SCRIPT = """
import hashlib, json, random, sys
import numpy as np
import torch
from architecture_mechanics.seeding import seed_everything

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 20260809
record = seed_everything(seed)

h = hashlib.sha256()
h.update(torch.randn(64, 64).numpy().tobytes())
h.update(np.random.randn(64, 64).tobytes())
h.update(repr([random.random() for _ in range(16)]).encode())
if torch.cuda.is_available():
    h.update(torch.randn(64, 64, device="cuda").cpu().numpy().tobytes())

print(json.dumps({"digest": h.hexdigest(), "record": record.as_dict()}))
"""


def _draw(seed: int, env: dict[str, str] | None = None) -> dict:
    """Run DRAW_SCRIPT in a fresh interpreter and return its parsed output."""
    import json

    proc = subprocess.run(
        [sys.executable, "-c", DRAW_SCRIPT, str(seed)],
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_same_seed_is_bitwise_identical_across_processes():
    first = _draw(20260809)
    second = _draw(20260809)
    assert first["digest"] == second["digest"]


def test_different_seeds_differ_across_processes():
    """Guards against the failure where the digest is constant for any seed —
    which would make the test above pass while proving nothing."""
    assert _draw(20260809)["digest"] != _draw(20260810)["digest"]


def test_record_reports_what_it_set():
    record = _draw(20260809)["record"]
    assert record["seed"] == 20260809
    assert record["call_index"] == 0
    assert record["determinism_mode"] == "strict"
    assert record["cudnn_deterministic"] is True
    assert record["cudnn_benchmark"] is False
    assert record["cublas_workspace_config"] == ":4096:8"
    assert record["cublas_config_set_before_cuda_init"] is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_draws_are_identical_across_processes():
    """The digest above already covers CUDA when present; this asserts the CUDA
    path was actually exercised rather than skipped inside the subprocess."""
    first, second = _draw(7), _draw(7)
    assert first["record"]["cuda_seeded"] is True
    assert first["digest"] == second["digest"]


def test_seed_rejects_out_of_range_and_non_integers():
    from architecture_mechanics.seeding import seed_everything

    with pytest.raises(ValueError):
        seed_everything(-1)
    with pytest.raises(ValueError):
        seed_everything(2**32)
    with pytest.raises(TypeError):
        seed_everything(1.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        seed_everything(True)  # bool is an int subclass; it is still not a seed
