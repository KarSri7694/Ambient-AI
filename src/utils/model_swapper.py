import logging
from typing import TYPE_CHECKING, Optional
from infrastructure.adapter.llamaCppAdapter import LlamaCppAdapter
from infrastructure.adapter.openVinoAdapter import OpenVinoAdapter

if TYPE_CHECKING:
    from application.services.llm_interaction_service import LLMInteractionService

class ModelSwapper:
    def __init__(
        self,
        new_model: str,
        model_adapter: LlamaCppAdapter | OpenVinoAdapter,
        llm_service: Optional["LLMInteractionService"] = None,
    ):
        """
        Args:
            new_model: model to swap to
            model_adapter: adapter managing model loading/unloading
            llm_service: if provided, conversation context is saved on entry
                         and restored on exit so each model has isolated history
        """
        self.new_model = new_model
        self.model_adapter = model_adapter
        self.llm_service = llm_service
        self.previous_model = model_adapter.get_current_model()
        self._saved_context = None
        logging.info(f"Swapping model from {self.previous_model} to {new_model}")

    async def __aenter__(self):
        if self.llm_service is not None:
            self._saved_context = self.llm_service.get_context()
            self.llm_service.reset_conversation()
        await self.model_adapter.unload_model()
        await self.model_adapter.load_model(self.new_model)
        logging.info(f"Successfully swapped model from {self.previous_model} to {self.new_model}")
        return self.new_model

    async def __aexit__(self, exc_type, exc, tb):
        await self.model_adapter.unload_model()
        if self.previous_model is not None:
            await self.model_adapter.load_model(self.previous_model)
        if self.llm_service is not None and self._saved_context is not None:
            self.llm_service.restore_context(self._saved_context)
        logging.info(f"Successfully swapped back to previous model {self.previous_model}")