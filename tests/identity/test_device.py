"""Identity controls for device resolution.

The property under test is a refusal: requesting CUDA when CUDA is unusable
must raise rather than quietly returning a CPU device. It is tested against a
real interpreter with the GPU hidden, not against a monkeypatched flag, because
the monkeypatch only proves that the branch reads the flag it was told to read.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch

from architecture_mechanics.device import CudaUnavailableError, resolve_device

REFUSAL_SCRIPT = """
import torch
from architecture_mechanics.device import CudaUnavailableError, resolve_device

assert not torch.cuda.is_available(), "CUDA_VISIBLE_DEVICES did not hide the GPU"
try:
    device, record = resolve_device("cuda")
except CudaUnavailableError:
    print("RAISED")
else:
    print(f"FELL BACK to {device} silently")
"""

AUTO_SCRIPT = """
from architecture_mechanics.device import resolve_device

device, record = resolve_device("auto")
print(f"{device}|{record.fell_back}")
"""


def _run_without_gpu(script: str) -> str:
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stderr}"
    return proc.stdout.strip().splitlines()[-1]


def test_requesting_cuda_without_cuda_raises():
    assert _run_without_gpu(REFUSAL_SCRIPT) == "RAISED"


def test_auto_falls_back_but_records_it():
    """``auto`` is the only sanctioned fallback, and it is never silent."""
    assert _run_without_gpu(AUTO_SCRIPT) == "cpu|True"


def test_unknown_device_string_raises():
    with pytest.raises(ValueError):
        resolve_device("tpu")


def test_cpu_is_honoured_and_not_marked_a_fallback():
    device, record = resolve_device("cpu")
    assert device.type == "cpu"
    assert record.resolved == "cpu"
    assert record.fell_back is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_absent_ordinal_raises_rather_than_wrapping_around():
    with pytest.raises(CudaUnavailableError):
        resolve_device(f"cuda:{torch.cuda.device_count()}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_record_carries_manifest_fields():
    device, record = resolve_device("cuda")
    assert device.type == "cuda"
    assert record.fell_back is False
    assert record.compute_capability is not None
    assert record.total_memory_bytes and record.total_memory_bytes > 0
    assert record.cuda_version is not None
    assert "+cu" in record.torch_version, "a CPU-only torch build cannot drive this laboratory"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_real_matmul_runs_on_the_gpu_with_finite_output():
    device, _ = resolve_device("cuda")
    x = torch.randn(1024, 1024, device=device)
    y = x @ x
    assert y.device.type == "cuda"
    assert torch.isfinite(y).all()
