from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from infrastructure.accelerator import SUPPORTED_ROCM_WINDOWS_ARCHES, detect_accelerator


FORBIDDEN_ROCM_LINUX_PACKAGES = {
    "appopener",
    "pyautogui",
    "pyaudio",
    "pywin32",
    "uiautomation",
}

CUDA_MARKERS = ("+cu", "cuda")


def _llama_version(path: str | None) -> dict[str, str | int | None]:
    if not path:
        return {"path": None, "returncode": None, "version": None}
    try:
        completed = subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        output = (completed.stdout or completed.stderr or "").strip()
        return {"path": path, "returncode": completed.returncode, "version": output.splitlines()[0] if output else ""}
    except Exception as exc:
        return {"path": path, "returncode": -1, "version": str(exc)}


def _installed_packages() -> dict[str, str]:
    packages: dict[str, str] = {}
    for dist in importlib.metadata.distributions():
        name = (dist.metadata.get("Name") or "").strip().lower()
        if name:
            packages[name] = dist.version
    return packages


def _dependency_report(profile: str, asr_backend: str) -> dict[str, object]:
    installed = _installed_packages()
    system = platform.system().lower()
    forbidden: list[str] = []
    cuda_like: list[str] = []

    if profile == "rocm-linux" or (profile == "auto" and system == "linux"):
        forbidden = sorted(name for name in FORBIDDEN_ROCM_LINUX_PACKAGES if name in installed)

    cuda_is_forbidden = profile in {"auto", "safe", "rocm-linux", "rocm-windows", "windows-cpu"}
    if cuda_is_forbidden:
        for name in ("torch", "torchaudio", "torchvision", "ctranslate2", "faster-whisper"):
            version = installed.get(name)
            if not version:
                continue
            normalized = version.lower()
            if any(marker in normalized for marker in CUDA_MARKERS):
                cuda_like.append(f"{name}=={version}")

    asr = asr_backend.strip().lower()
    asr_notes: list[str] = []
    if asr in {"whisper_auto", "auto", "faster_whisper", "faster-whisper", "whisper_cpu", "ctranslate2_rocm"}:
        if "faster-whisper" not in installed:
            asr_notes.append(
                "Whisper ASR selected but faster-whisper is not installed; install a validated "
                "Whisper backend. On ROCm use scripts/install_requirements.py --target rocm-linux."
            )
    if asr in {"ctranslate2_rocm", "ctranslate2_cpu", "whisper_ctranslate2_rocm"} and "ctranslate2" not in installed:
        asr_notes.append("CTranslate2 ASR backend selected but ctranslate2 is not installed.")
    if asr in {"llamacpp_qwen", "qwen", "qwen_asr"}:
        asr_notes.append("Qwen ASR is explicit; Whisper is not being used for ASR in this profile.")

    return {
        "profile": profile,
        "platform": platform.platform(),
        "forbidden_windows_packages": forbidden,
        "cuda_like_packages": sorted(cuda_like),
        "asr_backend": asr,
        "asr_notes": asr_notes,
        "selected_packages": {
            name: installed.get(name)
            for name in [
                "torch",
                "torchaudio",
                "torchvision",
                "ctranslate2",
                "faster-whisper",
                "pywin32",
                "uiautomation",
                "pyautogui",
            ]
            if installed.get(name)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the AMD ROCm Windows runtime for Ambient AI.")
    parser.add_argument("--backend", default="amd_rocm", choices=["auto", "amd_rocm", "nvidia_cuda", "cpu"])
    parser.add_argument("--require-supported-gpu", action="store_true")
    parser.add_argument("--llama-server", default=None, help="Optional path to AMD validated llama-server.exe")
    parser.add_argument("--dependency-profile", default="auto", choices=["auto", "rocm-linux", "rocm-windows", "nvidia-cuda", "windows-cpu", "safe"])
    parser.add_argument("--asr-backend", default="whisper_auto")
    args = parser.parse_args()

    info = detect_accelerator(args.backend, require_supported_gpu=args.require_supported_gpu)
    dependencies = _dependency_report(args.dependency_profile, args.asr_backend)
    dependency_ok = not dependencies["forbidden_windows_packages"] and not dependencies["cuda_like_packages"]
    payload = {
        "ok": info.available and (not args.require_supported_gpu or info.supported is True) and dependency_ok,
        "accelerator": info.to_dict(),
        "supported_rocm_windows_arches": sorted(SUPPORTED_ROCM_WINDOWS_ARCHES),
        "llama_cpp": _llama_version(args.llama_server),
        "dependencies": dependencies,
        "python": sys.version,
    }
    print(json.dumps(payload, indent=2))
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
