import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from infrastructure import accelerator


class _Props:
    def __init__(self, arch):
        self.gcnArchName = arch


class _Cuda:
    def __init__(self, available=True, name="GPU", arch="gfx1100:sramecc+:xnack-"):
        self._available = available
        self._name = name
        self._arch = arch

    def is_available(self):
        return self._available

    def get_device_name(self, index):
        return self._name

    def get_device_properties(self, index):
        return _Props(self._arch)


def _torch(hip=None, cuda="12.9", available=True, arch="gfx1100:sramecc+:xnack-"):
    return types.SimpleNamespace(
        version=types.SimpleNamespace(hip=hip, cuda=cuda),
        cuda=_Cuda(available=available, arch=arch),
    )


def test_detects_rocm_backend_and_keeps_cuda_device(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _torch(hip="7.2.1"))

    info = accelerator.detect_accelerator("amd_rocm", require_supported_gpu=True)

    assert info.available is True
    assert info.backend == "amd_rocm"
    assert info.torch_device == "cuda"
    assert info.gcn_architecture == "gfx1100"
    assert info.supported is True


def test_rejects_unsupported_rocm_architecture_when_required(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _torch(hip="7.2.1", arch="gfx9999"))

    info = accelerator.detect_accelerator("amd_rocm", require_supported_gpu=True)

    assert info.available is False
    assert info.supported is False
    assert "not in the supported" in info.reason


def test_detects_nvidia_cuda(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _torch(hip=None, cuda="12.9", arch=None))

    info = accelerator.detect_accelerator("auto")

    assert info.backend == "nvidia_cuda"
    assert info.torch_device == "cuda"
    assert info.supported is True


def test_returns_cpu_when_cuda_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _torch(available=False))

    info = accelerator.detect_accelerator("auto")

    assert info.backend == "cpu"
    assert info.torch_device == "cpu"
    assert info.available is True
