import openai
from typing import Optional, List, Dict, Any, Iterator
import requests
import logging
from datetime import datetime
from application.ports.LLMProvider import LLMProvider
from application.ports.modelManager import ModelManager
from pathlib import Path
from urllib.parse import urlparse

class LlamaCppAdapter(LLMProvider, ModelManager):
    """Adapter for llama.cpp server — implements both LLMProvider and ModelManager."""
    
    def __init__(self, base_url: str):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.base_url = base_url
        self.api_uri_v1 = f"{base_url}/v1"
        self.client = openai.AsyncOpenAI(
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
        elif response.status_code == 400 and "model is already running" in response.text.lower():
            self.logger.info(f"Specified model: {model_name} is already running.")
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

    def _kv_state_dir(self) -> Path:
        parent_dir = Path(__file__).parent.parent.parent.parent
        kv_state_dir = parent_dir / "model_kv_states"
        kv_state_dir.mkdir(exist_ok=True)
        return kv_state_dir

    def _safe_kv_state_filename(self) -> str:
        model_name = self.currently_loaded_model or "current_model"
        for char in '<>:"/\\|?* ':
            model_name = model_name.replace(char, "_")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{model_name}_{timestamp}_kv_state.bin"

    def _slot_base_url(self) -> str:
        try:
            response = requests.get(f"{self.base_url}/slots", timeout=5)
            if response.status_code == 200:
                return self.base_url
        except requests.RequestException:
            pass

        models_response = requests.get(f"{self.api_uri_v1}/models", timeout=10)
        models_response.raise_for_status()

        models = models_response.json().get("data", [])
        loaded_models = [
            model for model in models
            if model.get("status", {}).get("value") == "loaded"
        ]
        if not loaded_models:
            raise RuntimeError("No loaded llama.cpp model found for slot save/restore.")

        selected_model = next(
            (
                model for model in loaded_models
                if self.currently_loaded_model and model.get("id") == self.currently_loaded_model
            ),
            loaded_models[0],
        )
        args = selected_model.get("status", {}).get("args", [])

        try:
            port = args[args.index("--port") + 1]
        except (ValueError, IndexError) as exc:
            raise RuntimeError(
                f"Loaded model {selected_model.get('id')} does not expose a backend port."
            ) from exc

        parsed_base_url = urlparse(self.base_url)
        host = parsed_base_url.hostname or "localhost"
        scheme = parsed_base_url.scheme or "http"
        slot_base_url = f"{scheme}://{host}:{port}"

        response = requests.get(f"{slot_base_url}/slots", timeout=5)
        response.raise_for_status()
        return slot_base_url

    def save_current_kv_state(self) -> Path | None:
        kv_state_dir = self._kv_state_dir()
        save_file_name = self._safe_kv_state_filename()
        slot_base_url = self._slot_base_url()

        payload = {"filename": save_file_name}
        if self.currently_loaded_model:
            payload["model"] = self.currently_loaded_model

        response = requests.post(
            f"{slot_base_url}/slots/0?action=save",
            json=payload,
            timeout=30,
        )

        if response.status_code == 200:
            save_path = kv_state_dir / save_file_name
            self.logger.info(
                f"Successfully requested KV state save to {save_path}. "
                f"llama.cpp response: {response.text}"
            )
            return save_path
        else:
            self.logger.error(
                f"Failed to save KV state. Status: {response.status_code}. "
                f"Response: {response.text}"
            )
            return None
    
    def restore_kv_state(self, kv_state_file: str) -> None:
        kv_state_path = Path(kv_state_file)
        slot_base_url = self._slot_base_url()
        payload = {'filename': kv_state_path.name}
        if self.currently_loaded_model:
            payload["model"] = self.currently_loaded_model

        response = requests.post(
            f"{slot_base_url}/slots/0?action=restore",
            json=payload,
            timeout=30,
        )
        
        if response.status_code == 200:
            self.logger.info(f"Successfully restored KV state from {kv_state_file}")
        else:
            self.logger.error(f"Failed to restore KV state from {kv_state_file}. Response: {response.text}")
    # ── LLMProvider ───────────────────────────────────────────

    def generate_response(self, prompt: str, image: str = "") -> str:
        """Simple non-streaming generation."""
        completion = self.client.chat.completions.create(
            model=self.currently_loaded_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content or ""

    async def chat_completion_stream(
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
        Create a streaming chat completion against the llama.cpp OpenAI-compatible API.
        Returns the raw streaming iterator of chunk objects.
        """
        # If an image is provided, we need to inject it into the content of the LAST message.
        # This assumes the last message is from the 'user' and text-only.
        if image and messages and messages[-1]["role"] == "user":
            import base64
            with open(image, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode("utf-8")
            
            last_message = messages[-1]
            text_content = last_message["content"]
            
            # format content as list for multimodal
            last_message["content"] = [
                {"type": "text", "text": text_content},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                }
            ]

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "top_p": top_p,
        }
        if top_k > 0:
            kwargs["extra_body"] = {"top_k": top_k}
        if tools:
            kwargs["tools"] = tools

        return await self.client.chat.completions.create(**kwargs)
