import ctypes
import logging
import subprocess
import threading
import time
from datetime import datetime, timezone
from typing import Optional

from application.ports.resource_monitor_port import ResourceMonitorPort
from core.models import ResourceSnapshot


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class WindowsResourceMonitor(ResourceMonitorPort):
    """Low-overhead RAM telemetry with cached optional NVIDIA VRAM readings."""

    def __init__(self, *, cache_seconds: float = 5.0, logger: logging.Logger | None = None):
        self.cache_seconds = max(1.0, float(cache_seconds))
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._lock = threading.Lock()
        self._cached_at = 0.0
        self._cached: Optional[ResourceSnapshot] = None

    def snapshot(self, *, user_idle: bool = False, force: bool = False) -> ResourceSnapshot:
        now = time.monotonic()
        with self._lock:
            if not force and self._cached is not None and now - self._cached_at < self.cache_seconds:
                return ResourceSnapshot(
                    **{**self._cached.__dict__, "user_idle": bool(user_idle)}
                )
            total_ram, available_ram = self._ram_mb()
            total_vram, free_vram, gpu_info = self._vram_mb()
            available_percent = (available_ram / total_ram * 100.0) if total_ram else 0.0
            current = ResourceSnapshot(
                captured_at=datetime.now(timezone.utc).isoformat(),
                total_ram_mb=total_ram,
                available_ram_mb=available_ram,
                available_ram_percent=available_percent,
                total_vram_mb=total_vram,
                free_vram_mb=free_vram,
                gpu_telemetry_available=total_vram is not None and free_vram is not None,
                user_idle=bool(user_idle),
                gpu_backend=gpu_info.get("backend"),
                gpu_name=gpu_info.get("gpu_name"),
                gpu_runtime_version=gpu_info.get("runtime_version"),
                gpu_architecture=gpu_info.get("gcn_architecture"),
            )
            self._cached = current
            self._cached_at = now
            return current

    @staticmethod
    def _ram_mb() -> tuple[int, int]:
        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not hasattr(ctypes, "windll") or not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return 0, 0
        divisor = 1024 * 1024
        return int(status.ullTotalPhys / divisor), int(status.ullAvailPhys / divisor)

    def _vram_mb(self) -> tuple[Optional[int], Optional[int], dict[str, Optional[str]]]:
        gpu_info: dict[str, Optional[str]] = {}
        try:
            import torch
            from infrastructure.accelerator import detect_accelerator

            info = detect_accelerator()
            gpu_info = {
                "backend": info.backend,
                "gpu_name": info.gpu_name,
                "runtime_version": info.runtime_version,
                "gcn_architecture": info.gcn_architecture,
            }
            if torch.cuda.is_available():
                free_bytes, total_bytes = torch.cuda.mem_get_info(0)
                divisor = 1024 * 1024
                return int(total_bytes / divisor), int(free_bytes / divisor), gpu_info
        except Exception as exc:
            self.logger.debug("PyTorch GPU telemetry unavailable: %s", exc)
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if completed.returncode != 0 or not completed.stdout.strip():
                return None, None, gpu_info
            row = completed.stdout.splitlines()[0].split(",", 1)
            if not gpu_info.get("backend") or gpu_info.get("backend") == "cpu":
                gpu_info["backend"] = "nvidia_cuda"
            return int(float(row[0].strip())), int(float(row[1].strip())), gpu_info
        except (OSError, ValueError, StopIteration, subprocess.TimeoutExpired) as exc:
            self.logger.debug("NVIDIA VRAM telemetry unavailable: %s", exc)
            return None, None, gpu_info
