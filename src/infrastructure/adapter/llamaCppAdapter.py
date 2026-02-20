import openai
from typing import Optional, List, Dict, Any, Iterator
import requests
import logging

from application.ports.LLMProvider import LLMProvider
from application.ports.modelManager import ModelManager


class LlamaCppAdapter(LLMProvider, ModelManager):
    """Adapter for llama.cpp server — implements both LLMProvider and ModelManager."""

    def __init__(self, base_url: str):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.base_url = base_url
        self.api_uri_v1 = f"{base_url}/v1"
        self.client = openai.OpenAI(
            base_url=self.api_uri_v1,
            api_key="testkey"
        )
        self.currently_loaded_model: Optional[str] = None

    # ── ModelManager ──────────────────────────────────────────

    async def load_model(self, model_name: str, unload_previous: bool = True) -> None:
        if self.currently_loaded_model == model_name:
            self.logger.info(f"Model {model_name} is already loaded.")
            return

        if unload_previous and self.currently_loaded_model is not None:
            await self.unload_model()

        model = {"model": model_name}
        response = requests.post(f"{self.base_url}/models/load", json=model)
        if response.status_code == 200:
            self.logger.info(f"Successfully loaded model: {model_name}")
            self.currently_loaded_model = model_name
        else:
            self.logger.error(f"Failed to load model: {model_name}. Response: {response.text}")

    async def unload_model(self) -> None:
        if self.currently_loaded_model is None:
            return
        model = {"model": self.currently_loaded_model}
        response = requests.post(f"{self.base_url}/models/unload", json=model)
        if response.status_code == 200:
            self.logger.info(f"Successfully unloaded model: {self.currently_loaded_model}")
            self.currently_loaded_model = None
        else:
            self.logger.error(f"Failed to unload model: {self.currently_loaded_model}. Response: {response.text}")

    def get_current_model(self) -> Optional[str]:
        return self.currently_loaded_model

    # ── LLMProvider ───────────────────────────────────────────

    def generate_response(self, prompt: str, image: str = "") -> str:
        """Simple non-streaming generation."""
        completion = self.client.chat.completions.create(
            model=self.currently_loaded_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content or ""

    def chat_completion_stream(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Iterator:
        """
        Create a streaming chat completion against the llama.cpp OpenAI-compatible API.
        Returns the raw streaming iterator of chunk objects.
        """
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools

        return self.client.chat.completions.create(**kwargs)