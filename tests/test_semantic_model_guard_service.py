import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from application.services.semantic_model_guard_service import SemanticModelGuardService


class _FakeMainModelProvider:
    def __init__(self, loaded_model="MainLLM"):
        self.loaded_model = loaded_model
        self.unloaded = []
        self.loaded = []

    def get_current_model(self):
        return self.loaded_model

    def unload_model_sync(self):
        model = self.loaded_model
        self.loaded_model = None
        if model:
            self.unloaded.append(model)
        return model

    def load_model_sync(self, model_name, unload_previous=True):
        self.loaded_model = model_name
        self.loaded.append((model_name, unload_previous))


def test_guard_unloads_for_configured_embedding_without_restore():
    provider = _FakeMainModelProvider()
    guard = SemanticModelGuardService(
        main_model_provider=provider,
        unload_for_embedding=True,
        unload_for_rerank=False,
        restore_after_semantic=False,
    )

    with guard.semantic_operation("embedding"):
        assert provider.loaded_model is None

    assert provider.unloaded == ["MainLLM"]
    assert provider.loaded == []
    assert provider.loaded_model is None


def test_guard_does_not_unload_for_disabled_operation():
    provider = _FakeMainModelProvider()
    guard = SemanticModelGuardService(
        main_model_provider=provider,
        unload_for_embedding=False,
        unload_for_rerank=True,
        restore_after_semantic=False,
    )

    with guard.semantic_operation("embedding"):
        assert provider.loaded_model == "MainLLM"

    assert provider.unloaded == []


def test_guard_can_restore_previous_model_when_configured():
    provider = _FakeMainModelProvider("MainLLM")
    guard = SemanticModelGuardService(
        main_model_provider=provider,
        unload_for_embedding=False,
        unload_for_rerank=True,
        restore_after_semantic=True,
    )

    with guard.semantic_operation("rerank"):
        assert provider.loaded_model is None

    assert provider.unloaded == ["MainLLM"]
    assert provider.loaded == [("MainLLM", True)]
    assert provider.loaded_model == "MainLLM"


def test_guard_noops_when_no_main_model_is_loaded():
    provider = _FakeMainModelProvider(None)
    guard = SemanticModelGuardService(
        main_model_provider=provider,
        unload_for_embedding=True,
        unload_for_rerank=True,
        restore_after_semantic=True,
    )

    with guard.semantic_operation("rerank"):
        assert provider.loaded_model is None

    assert provider.unloaded == []
    assert provider.loaded == []
