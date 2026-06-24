from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, List, Dict, Any, Optional


class LLMProvider(ABC):
    """Port for LLM text generation and chat completion."""

    @abstractmethod
    def generate_response(self, prompt: str, image: str = "") -> str:
        """Generate text based on the given prompt using the LLM."""
        pass

    @abstractmethod
    def chat_completion_stream(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        image: str = "",
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 0,
    ) -> Iterator:
        """
        Create a streaming chat completion.
        Yields chunk objects compatible with OpenAI's streaming format.
        """
        pass
    
    # @abstractmethod
    # def get_context(self) -> List[Dict[str, Any]]:
    #     """Return the current message history."""
    #     pass
    
    @abstractmethod
    def load_model(self, model_name: str) -> None:
        """Load a model into the LLM provider."""
        pass
    
    @abstractmethod
    def save_and_unload(self, messages: List[Dict[str, Any]]) -> Optional[Path]:
        """Save the current KV state to disk and unload the model."""
        pass
    
    @abstractmethod
    def load_and_restore(self) -> Path:
        """Load the model and restore the KV state from a given file."""
        pass
    
    # @abstractmethod
    # def restore_kv_state(self, kv_state_file: str) -> None:
    #     """Restore the KV state from a given file."""
    #     pass
