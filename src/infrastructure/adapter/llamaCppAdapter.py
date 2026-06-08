import openai
from typing import Optional, List, Dict, Any, Iterator
import requests
import logging
import base64
import re
from datetime import datetime
from application.ports.LLMProvider import LLMProvider
from application.ports.modelManager import ModelManager
from pathlib import Path
from urllib.parse import urlparse

class LlamaCppAdapter(LLMProvider, ModelManager):
    """Adapter for llama.cpp server — implements both LLMProvider and ModelManager."""
    
    def __init__(self, base_url: str):
        """Create an adapter for a llama.cpp-compatible OpenAI API server."""
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
        """Load a model through the llama.cpp server model-management endpoint."""
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
        """Unload the currently tracked model from the llama.cpp server."""
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
        """Return the model name this adapter currently tracks as loaded."""
        return self.currently_loaded_model

    def _kv_state_dir(self) -> Path:
        """Return the local directory used for llama.cpp KV state files."""
        parent_dir = Path(__file__).parent.parent.parent.parent
        kv_state_dir = parent_dir / "model_kv_states"
        kv_state_dir.mkdir(exist_ok=True)
        return kv_state_dir

    def _safe_kv_state_filename(self) -> str:
        """Build a timestamped KV state filename containing a reversible model name."""
        model_name = self.currently_loaded_model or "current_model"
        encoded_model_name = self._encode_model_name_for_filename(model_name)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{encoded_model_name}_{timestamp}_kv_state.bin"

    @staticmethod
    def _encode_model_name_for_filename(model_name: str) -> str:
        """Encode a model name into a filesystem-safe filename component."""
        encoded = base64.urlsafe_b64encode(model_name.encode("utf-8")).decode("ascii")
        return f"model_b64_{encoded.rstrip('=')}"

    @staticmethod
    def _decode_model_name_from_filename(value: str) -> Optional[str]:
        """Decode a filename component created by `_encode_model_name_for_filename`."""
        prefix = "model_b64_"
        if not value.startswith(prefix):
            return None

        encoded = value[len(prefix):]
        encoded += "=" * (-len(encoded) % 4)
        return base64.urlsafe_b64decode(encoded.encode("ascii")).decode("utf-8")

    @classmethod
    def _extract_model_name_from_kv_state_file(cls, kv_state_file: str) -> str:
        """Extract the saved model name from a timestamped KV state filename."""
        kv_state_name = Path(kv_state_file).name
        match = re.fullmatch(
            r"(?P<model>.+)_\d{8}_\d{6}_kv_state\.bin",
            kv_state_name,
        )
        if not match:
            raise ValueError(
                "KV state filename must match '<model>_YYYYMMDD_HHMMSS_kv_state.bin'."
            )

        encoded_or_legacy_name = match.group("model")
        return cls._decode_model_name_from_filename(encoded_or_legacy_name) or encoded_or_legacy_name

    def _slot_base_url(self) -> str:
        """Resolve the server URL that exposes llama.cpp slot save/restore endpoints."""
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
        """Request a KV state save for slot 0 and return the expected local file path."""
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
    
    async def restore_kv_state(self, kv_state_file: str) -> None:
        """Restore slot 0 KV state from the given saved state filename or path."""
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
    
    async def save_and_unload(self) -> Optional[Path]:
        """Save the current KV state, then unload the model if the save succeeds."""
        save_path = self.save_current_kv_state()
        if save_path is not None:
            await self.unload_model()
        return save_path
    
    async def load_and_restore(self, kv_state_file: str) -> None:
        """Load the model named by a KV state file, then restore that KV state."""
        model_name = self._extract_model_name_from_kv_state_file(kv_state_file)
        if self.currently_loaded_model != model_name:
            await self.load_model(model_name)
        await self.restore_kv_state(kv_state_file)

    # ── LLMProvider ───────────────────────────────────────────

    def generate_response(self, prompt: str, image: str = "") -> str:
        """Create a non-streaming chat completion for a single prompt."""
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
        Create a streaming chat completion through the OpenAI-compatible API.

        If `image` is provided, it is attached to the final user message as a
        base64 data URL for multimodal models.
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
