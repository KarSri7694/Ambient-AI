from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CTRANSLATE2_ROCM_VERSION = "4.8.1"
FASTER_WHISPER_VERSION = "1.2.1"
CTRANSLATE2_ROCM_ARCHIVES = {
    "rocm-linux": f"https://github.com/OpenNMT/CTranslate2/releases/download/v{CTRANSLATE2_ROCM_VERSION}/rocm-python-wheels-Linux.zip",
    "rocm-windows": f"https://github.com/OpenNMT/CTranslate2/releases/download/v{CTRANSLATE2_ROCM_VERSION}/rocm-python-wheels-Windows.zip",
}

PROFILE_FILES = {
    "safe": ["requirements.txt"],
    "rocm-linux": ["requirements-amd-rocm-linux.txt"],
    "rocm-windows": ["requirements-amd-rocm-windows.txt"],
    "nvidia-cuda": ["requirements-nvidia-cuda.txt"],
    "windows-cpu": ["requirements.txt", "requirements-windows-desktop.txt"],
}


def _has_command(name: str) -> bool:
    return shutil.which(name) is not None


def _detect_target() -> str:
    system = platform.system().lower()
    if system == "linux":
        if _has_command("rocm-smi") or Path("/opt/rocm").exists() or os.environ.get("ROCM_PATH"):
            return "rocm-linux"
        if _has_command("nvidia-smi") or os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH"):
            return "nvidia-cuda"
        return "safe"
    if system == "windows":
        if os.environ.get("ROCM_PATH") or os.environ.get("HIP_PATH"):
            return "rocm-windows"
        if os.environ.get("CUDA_PATH") or _has_command("nvidia-smi"):
            return "nvidia-cuda"
        return "windows-cpu"
    return "safe"


def _requirements_for_target(target: str) -> list[Path]:
    if target == "auto":
        target = _detect_target()
    try:
        names = PROFILE_FILES[target]
    except KeyError as exc:
        raise SystemExit(f"Unknown target {target!r}. Choose one of: auto, {', '.join(PROFILE_FILES)}") from exc
    return [ROOT / name for name in names]


def _python_tag() -> str:
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def _platform_wheel_marker(target: str) -> str:
    return "win_amd64" if target == "rocm-windows" else "linux"


def _find_matching_ctranslate2_wheel(directory: Path, target: str) -> Path:
    py_tag = _python_tag()
    platform_marker = _platform_wheel_marker(target)
    wheels = sorted(directory.rglob(f"ctranslate2-{CTRANSLATE2_ROCM_VERSION}-{py_tag}-*.whl"))
    candidates = [path for path in wheels if platform_marker in path.name.lower()]
    if not candidates:
        available = "\n".join(path.name for path in sorted(directory.rglob("*.whl")))
        raise SystemExit(
            f"No CTranslate2 ROCm wheel found for Python tag {py_tag} and target {target}. "
            f"Available wheels:\n{available}"
        )
    return candidates[0]


def _install_rocm_ctranslate2_and_faster_whisper(
    *,
    target: str,
    python_exe: str,
    dry_run: bool,
    archive_url: str | None,
) -> None:
    url = archive_url or CTRANSLATE2_ROCM_ARCHIVES[target]
    print(f"ROCm CTranslate2 wheel archive: {url}")
    print(f"faster-whisper install mode: faster-whisper=={FASTER_WHISPER_VERSION} --no-deps")
    if dry_run:
        print("+ download", url)
        print("+ install matching ctranslate2 ROCm wheel for", _python_tag())
        print("+", python_exe, "-m pip install", f"faster-whisper=={FASTER_WHISPER_VERSION}", "--no-deps")
        return

    with tempfile.TemporaryDirectory(prefix="ambient-ct2-rocm-") as tmp:
        tmp_path = Path(tmp)
        archive_path = tmp_path / "ctranslate2-rocm-wheels.zip"
        urllib.request.urlretrieve(url, archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(tmp_path)
        wheel = _find_matching_ctranslate2_wheel(tmp_path, target)
        subprocess.run([python_exe, "-m", "pip", "install", "--force-reinstall", str(wheel)], check=True)
        subprocess.run([python_exe, "-m", "pip", "install", f"faster-whisper=={FASTER_WHISPER_VERSION}", "--no-deps"], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Install Ambient AI requirements for a specific hardware/platform profile.")
    parser.add_argument(
        "--target",
        default="auto",
        choices=["auto", *PROFILE_FILES.keys()],
        help="Dependency profile to install. auto detects ROCm/CUDA/platform from the host.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print selected files and pip commands without installing.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use for pip.")
    parser.add_argument(
        "--ctranslate2-rocm-archive",
        default=None,
        help="Override URL/path for rocm-python-wheels-*.zip from the CTranslate2 release.",
    )
    args = parser.parse_args()

    selected_target = _detect_target() if args.target == "auto" else args.target
    files = _requirements_for_target(selected_target)
    missing = [path for path in files if not path.exists()]
    if missing:
        raise SystemExit(f"Missing requirements file(s): {', '.join(str(path) for path in missing)}")

    print(f"Dependency profile: {selected_target}")
    for path in files:
        print(f"  - {path.relative_to(ROOT)}")

    commands = [[args.python, "-m", "pip", "install", "-r", str(path)] for path in files]
    for command in commands:
        print("+", " ".join(command))
        if not args.dry_run:
            subprocess.run(command, check=True)
    if selected_target in {"rocm-linux", "rocm-windows"}:
        _install_rocm_ctranslate2_and_faster_whisper(
            target=selected_target,
            python_exe=args.python,
            dry_run=args.dry_run,
            archive_url=args.ctranslate2_rocm_archive,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
