"""Device resolution that refuses to silently fall back to CPU.

A run whose manifest says ``cuda`` but which actually executed on CPU is a lie
in the provenance record, and it is the kind of lie that survives review: the
numbers are real, the timings are merely wrong, and nobody notices until a
replication disagrees. So ``cuda`` here means CUDA or an exception. Callers who
genuinely do not care ask for ``auto``, and the record says which way it went.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import torch


class CudaUnavailableError(RuntimeError):
    """Raised when CUDA was requested explicitly and is not usable."""


@dataclass(frozen=True)
class DeviceRecord:
    """Resolved device and versions, serialised verbatim into the run manifest."""

    requested: str
    resolved: str
    fell_back: bool
    """True only when ``auto`` was asked for and CUDA was unavailable."""

    device_name: str
    compute_capability: str | None
    total_memory_bytes: int | None
    free_memory_bytes: int | None
    multi_processor_count: int | None
    torch_version: str
    cuda_version: str | None
    cudnn_version: int | None
    float32_matmul_precision: str

    def as_dict(self) -> dict:
        return asdict(self)


def resolve_device(requested: str = "cuda") -> tuple[torch.device, DeviceRecord]:
    """Resolve a device string into a device and a manifest record.

    Args:
        requested: ``cuda``, ``cuda:N``, ``cpu``, or ``auto``.

    Raises:
        CudaUnavailableError: ``cuda`` was requested and CUDA is unavailable,
            or the requested ordinal does not exist.
        ValueError: the string names neither CPU nor CUDA.
    """
    requested = requested.strip()
    available = torch.cuda.is_available()

    if requested == "auto":
        target = "cuda:0" if available else "cpu"
        fell_back = not available
    else:
        target = requested
        fell_back = False

    if target == "cpu":
        return torch.device("cpu"), _cpu_record(requested, fell_back)

    if not (target == "cuda" or target.startswith("cuda:")):
        raise ValueError(f"unrecognised device string {requested!r}; expected cuda, cuda:N, cpu, or auto")

    if not available:
        raise CudaUnavailableError(
            f"device {requested!r} was requested but torch.cuda.is_available() is False "
            f"(torch {torch.__version__}, built against CUDA {torch.version.cuda}). "
            "Refusing to fall back to CPU; ask for 'auto' if a CPU run is acceptable."
        )

    index = 0 if target == "cuda" else int(target.split(":", 1)[1])
    count = torch.cuda.device_count()
    if not 0 <= index < count:
        raise CudaUnavailableError(f"device {requested!r} requested but only {count} CUDA device(s) present")

    device = torch.device(f"cuda:{index}")
    props = torch.cuda.get_device_properties(index)
    free, total = torch.cuda.mem_get_info(index)
    record = DeviceRecord(
        requested=requested,
        resolved=str(device),
        fell_back=fell_back,
        device_name=props.name,
        compute_capability=f"{props.major}.{props.minor}",
        total_memory_bytes=total,
        free_memory_bytes=free,
        multi_processor_count=props.multi_processor_count,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version(),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
    )
    return device, record


def _cpu_record(requested: str, fell_back: bool) -> DeviceRecord:
    import platform

    return DeviceRecord(
        requested=requested,
        resolved="cpu",
        fell_back=fell_back,
        device_name=platform.processor() or platform.machine(),
        compute_capability=None,
        total_memory_bytes=None,
        free_memory_bytes=None,
        multi_processor_count=None,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        cudnn_version=None,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
    )


def main() -> None:
    """``make gpu-check``: resolve CUDA, print the record, run a real matmul."""
    device, record = resolve_device("cuda")
    print(json.dumps(record.as_dict(), indent=2))
    x = torch.randn(1024, 1024, device=device)
    y = x @ x
    if not torch.isfinite(y).all():
        raise RuntimeError("GPU matmul produced non-finite values")
    torch.cuda.synchronize()
    print(f"gpu-check ok: {tuple(y.shape)} matmul on {record.device_name}, mean {y.mean().item():.6f}")


if __name__ == "__main__":
    main()
