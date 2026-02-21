from abc import ABC, abstractmethod
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
    ) -> Iterator:
        """
        Create a streaming chat completion.
        Yields chunk objects compatible with OpenAI's streaming format.
        """
        pass