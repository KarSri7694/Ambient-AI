from __future__ import annotations

import gc
from dataclasses import asdict, dataclass
from typing import Any


SUPPORTED_ROCM_WINDOWS_ARCHES = {"gfx1100", "gfx1101", "gfx1200", "gfx1201"}


@dataclass(frozen=True)
class AcceleratorInfo:
    backend: str
    torch_device: str
    available: bool
    gpu_name: str | None = None
    runtime_version: str | None = None
    gcn_architecture: str | None = None
    supported: bool | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_arch(value: Any) -> str | None:
    if not value:
        return None
    text = str(value).split(":", 1)[0].strip().lower()
    return text or None


def detect_accelerator(
    requested_backend: str = "auto",
    *,
    require_supported_gpu: bool = False,
) -> AcceleratorInfo:
    requested = (requested_backend or "auto").strip().lower()
    if requested in {"amd", "rocm"}:
        requested = "amd_rocm"
    if requested in {"nvidia", "cuda"}:
        requested = "nvidia_cuda"
    if requested not in {"auto", "amd_rocm", "nvidia_cuda", "cpu"}:
        return AcceleratorInfo(
            backend="cpu",
            torch_device="cpu",
            available=False,
            supported=False,
            reason=f"Unknown accelerator backend: {requested_backend}",
        )
    if requested == "cpu":
        return AcceleratorInfo(backend="cpu", torch_device="cpu", available=True, supported=True)

    try:
        import torch
    except Exception as exc:
        return AcceleratorInfo(
            backend="cpu",
            torch_device="cpu",
            available=False,
            supported=False if require_supported_gpu else None,
            reason=f"PyTorch unavailable: {exc}",
        )

    if not torch.cuda.is_available():
        if requested != "auto":
            return AcceleratorInfo(
                backend="cpu",
                torch_device="cpu",
                available=False,
                supported=False,
                reason=f"Requested {requested} but torch.cuda is unavailable",
            )
        return AcceleratorInfo(backend="cpu", torch_device="cpu", available=True, supported=True)

    is_rocm = bool(getattr(torch.version, "hip", None))
    backend = "amd_rocm" if is_rocm else "nvidia_cuda"
    if requested != "auto" and requested != backend:
        return AcceleratorInfo(
            backend=backend,
            torch_device="cuda",
            available=False,
            supported=False,
            reason=f"Requested {requested}, but detected {backend}",
        )

    gpu_name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    arch = _normalize_arch(getattr(props, "gcnArchName", None))
    supported = None
    if backend == "amd_rocm":
        supported = arch in SUPPORTED_ROCM_WINDOWS_ARCHES if arch else None
        if require_supported_gpu and supported is not True:
            return AcceleratorInfo(
                backend=backend,
                torch_device="cuda",
                available=False,
                gpu_name=gpu_name,
                runtime_version=getattr(torch.version, "hip", None),
                gcn_architecture=arch,
                supported=supported,
                reason=(
                    f"ROCm GPU architecture {arch or 'unknown'} is not in the supported "
                    f"Windows set: {', '.join(sorted(SUPPORTED_ROCM_WINDOWS_ARCHES))}"
                ),
            )
    else:
        supported = True

    runtime_version = getattr(torch.version, "hip", None) if is_rocm else getattr(torch.version, "cuda", None)
    return AcceleratorInfo(
        backend=backend,
        torch_device="cuda",
        available=True,
        gpu_name=gpu_name,
        runtime_version=runtime_version,
        gcn_architecture=arch,
        supported=supported,
    )


def resolve_torch_device(preferred: str = "auto", *, workload_default: str = "auto") -> str:
    choice = (preferred or "auto").strip().lower()
    if choice == "auto":
        choice = (workload_default or "auto").strip().lower()
    if choice in {"cpu", "cuda"}:
        return choice
    return "cuda" if detect_accelerator().torch_device == "cuda" else "cpu"


def enable_fast_cuda_math() -> None:
    info = detect_accelerator()
    if info.backend != "nvidia_cuda":
        return
    try:
        import torch

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    except Exception:
        return


def empty_accelerator_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()
