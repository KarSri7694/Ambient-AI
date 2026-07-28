import builtins
import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))


def _read(name: str) -> str:
    return (REPO_ROOT / name).read_text(encoding="utf-8").lower()


def test_rocm_linux_requirements_exclude_cuda_and_windows_only_packages():
    text = _read("requirements-amd-rocm-linux.txt")
    forbidden = ["cu129", "faster-whisper==", "ctranslate2==", "pywin32", "uiautomation", "pyautogui", "appopener", "pyaudio"]
    for package in forbidden:
        assert package not in text
    assert "rocm" in text
    assert "requirements-faster-whisper-common" in text


def test_safe_default_requirements_do_not_install_cuda_or_windows_desktop_packages():
    text = _read("requirements.txt")
    assert "cu129" not in text
    assert "requirements-windows-desktop" not in text
    assert "requirements-nvidia-cuda" not in text


def test_windows_desktop_requirements_use_platform_markers():
    text = _read("requirements-windows-desktop.txt")
    assert 'platform_system == "windows"' in text
    assert "pywin32" in text
    assert "uiautomation" in text


def test_install_requirements_dry_run_selects_rocm_linux(monkeypatch):
    spec = importlib.util.spec_from_file_location("install_requirements", REPO_ROOT / "scripts" / "install_requirements.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    monkeypatch.setattr(module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(module, "_has_command", lambda name: name == "rocm-smi")
    monkeypatch.setattr(module.Path, "exists", lambda self: False)

    assert module._detect_target() == "rocm-linux"


def test_asr_adapter_imports_without_faster_whisper():
    import infrastructure.adapter.ASR_Adapter as asr_adapter

    assert hasattr(asr_adapter, "WhisperAdapter")
    assert hasattr(asr_adapter, "QwenASRAdapter")


def test_whisper_adapter_reports_clear_error_when_faster_whisper_missing(monkeypatch):
    import infrastructure.adapter.ASR_Adapter as asr_adapter

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "faster_whisper":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(asr_adapter.CONFIG, "get_str", lambda section, key, default="": "whisper_auto" if key == "asr_backend" else default)

    with pytest.raises(RuntimeError, match="Whisper ASR is selected"):
        asr_adapter.WhisperAdapter(model_size="tiny", device="cpu")


def test_rocm_installer_installs_ctranslate2_before_faster_whisper_no_deps(capsys):
    spec = importlib.util.spec_from_file_location("install_requirements", REPO_ROOT / "scripts" / "install_requirements.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    module._install_rocm_ctranslate2_and_faster_whisper(
        target="rocm-linux",
        python_exe="python",
        dry_run=True,
        archive_url="https://example.test/rocm-python-wheels-Linux.zip",
    )
    output = capsys.readouterr().out
    assert "rocm-python-wheels-Linux.zip" in output
    assert "faster-whisper==1.2.1 --no-deps" in output
