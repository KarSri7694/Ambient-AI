import logging
from contextlib import contextmanager
from typing import Iterator, Optional


class SemanticModelGuardService:
    """Optionally evicts the main LLM while semantic models are used."""

    def __init__(
        self,
        *,
        main_model_provider,
        unload_for_embedding: bool = False,
        unload_for_rerank: bool = False,
        restore_after_semantic: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        self.main_model_provider = main_model_provider
        self.unload_for_embedding = bool(unload_for_embedding)
        self.unload_for_rerank = bool(unload_for_rerank)
        self.restore_after_semantic = bool(restore_after_semantic)
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    @contextmanager
    def semantic_operation(self, operation: str) -> Iterator[None]:
        should_unload = (
            (operation == "embedding" and self.unload_for_embedding)
            or (operation == "rerank" and self.unload_for_rerank)
        )
        previous_model = None
        unloaded_model = None
        if should_unload:
            previous_model = self._current_model()
            if previous_model:
                unloaded_model = self._unload_main_model(operation=operation)
        try:
            yield
        finally:
            if self.restore_after_semantic and unloaded_model:
                self._restore_main_model(previous_model or unloaded_model, operation=operation)

    def _current_model(self) -> Optional[str]:
        try:
            getter = getattr(self.main_model_provider, "get_current_model", None)
            if callable(getter):
                return getter()
        except Exception as exc:
            self.logger.warning("Unable to read main LLM residency before semantic call: %s", exc)
        return None

    def _unload_main_model(self, *, operation: str) -> Optional[str]:
        try:
            unload = getattr(self.main_model_provider, "unload_model_sync", None)
            if callable(unload):
                unloaded = unload()
                if unloaded:
                    self.logger.info(
                        "Unloaded main LLM %s before semantic %s to free VRAM.",
                        unloaded,
                        operation,
                    )
                return unloaded
            self.logger.warning("Main LLM provider does not support synchronous semantic eviction.")
        except Exception as exc:
            self.logger.warning("Unable to unload main LLM before semantic %s: %s", operation, exc)
        return None

    def _restore_main_model(self, model_name: str, *, operation: str) -> None:
        if not model_name:
            return
        try:
            load = getattr(self.main_model_provider, "load_model_sync", None)
            if callable(load):
                load(model_name)
                self.logger.info(
                    "Restored main LLM %s after semantic %s.",
                    model_name,
                    operation,
                )
                return
            self.logger.warning("Main LLM provider does not support synchronous semantic restore.")
        except Exception as exc:
            self.logger.warning("Unable to restore main LLM after semantic %s: %s", operation, exc)
