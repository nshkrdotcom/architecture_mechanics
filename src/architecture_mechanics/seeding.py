"""Single-point seeding of Python, NumPy, and torch, with a record for the manifest.

Every entry point calls :func:`seed_everything` exactly once, before it builds
data or models, and puts the returned record in the run manifest. The record is
the provenance answer to "was this run actually deterministic, and in what
sense" — it names what was set, what could not be set, and how many times
seeding happened, so a double-seeded run is visible rather than invisible.
"""

from __future__ import annotations

import os
import random
import sys
from dataclasses import asdict, dataclass

import numpy as np
import torch

# cuBLAS needs a fixed workspace for bitwise-reproducible reductions on CUDA.
# torch.use_deterministic_algorithms() raises without one of these values.
CUBLAS_WORKSPACE_CONFIG = ":4096:8"

_SEED_CALLS = 0
_LAST_RECORD: SeedRecord | None = None


@dataclass(frozen=True)
class SeedRecord:
    """What seeding actually did. Serialised verbatim into the run manifest."""

    seed: int
    call_index: int
    """0 for the first call in this process. Anything above 0 means the process
    seeded more than once, which breaks the one-seeding-per-entry-point rule."""

    determinism_mode: str
    """``strict`` (nondeterministic kernels raise), ``warn_only`` (they warn),
    or ``off``."""

    cudnn_deterministic: bool
    cudnn_benchmark: bool
    cublas_workspace_config: str | None
    cublas_config_set_before_cuda_init: bool
    """False means CUDA was already initialised when the variable was set, so
    cuBLAS may have captured the old value and bitwise reproducibility on GPU
    reductions is not guaranteed."""

    cuda_seeded: bool
    cuda_device_count: int
    torch_initial_seed: int
    python_hash_seed: str | None
    hash_randomization_enabled: bool
    """Inherited from interpreter start. PYTHONHASHSEED cannot be changed from
    inside a running process, so this is observed, never set."""

    def as_dict(self) -> dict:
        return asdict(self)


def seed_everything(
    seed: int,
    *,
    deterministic: bool = True,
    warn_only: bool = False,
) -> SeedRecord:
    """Seed Python, NumPy, and torch (CPU and CUDA) from one integer.

    Args:
        seed: non-negative integer below 2**32.
        deterministic: request deterministic algorithms and cuDNN settings.
        warn_only: with ``deterministic``, downgrade "no deterministic kernel"
            from an error to a warning. Defaults to False so that a
            nondeterministic op fails loudly instead of quietly costing a
            reproduction later.

    Returns:
        A :class:`SeedRecord` for the manifest.
    """
    global _SEED_CALLS, _LAST_RECORD

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    if not 0 <= seed < 2**32:
        raise ValueError(f"seed must be in [0, 2**32), got {seed}")

    cuda_available = torch.cuda.is_available()
    cublas_set_early = not torch.cuda.is_initialized()
    cublas_config: str | None = None

    if deterministic:
        # Only set it if absent: an operator who exported a different valid
        # value gets to keep it, and the record shows which one was in force.
        cublas_config = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_available:
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        mode = "warn_only" if warn_only else "strict"
    else:
        torch.use_deterministic_algorithms(False)
        mode = "off"

    record = SeedRecord(
        seed=seed,
        call_index=_SEED_CALLS,
        determinism_mode=mode,
        cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
        cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
        cublas_workspace_config=cublas_config,
        cublas_config_set_before_cuda_init=cublas_set_early,
        cuda_seeded=cuda_available,
        cuda_device_count=torch.cuda.device_count() if cuda_available else 0,
        torch_initial_seed=torch.initial_seed(),
        python_hash_seed=os.environ.get("PYTHONHASHSEED"),
        hash_randomization_enabled=bool(sys.flags.hash_randomization),
    )

    _SEED_CALLS += 1
    _LAST_RECORD = record
    return record


def seeding_calls() -> int:
    """How many times :func:`seed_everything` ran in this process."""
    return _SEED_CALLS


def last_seed_record() -> SeedRecord | None:
    """The most recent record, or None if this process has not seeded."""
    return _LAST_RECORD
