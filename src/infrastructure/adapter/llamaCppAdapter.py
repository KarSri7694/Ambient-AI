import openai
from typing import Optional, List, Dict, Any, Iterator
import requests
import logging
import copy
import json
import time
import re
from datetime import datetime

from application.ports.LLMProvider import LLMProvider
from application.ports.modelManager import ModelManager
from pathlib import Path
from urllib.parse import urlparse
from utils.kv_state_handling import KVStateControl

class LlamaCppAdapter(LLMProvider, ModelManager):
    """Adapter for llama.cpp server — implements both LLMProvider and ModelManager."""
    DEFAULT_MODEL = "Qwen-4b-Thinking-2507-Q4_K_M"
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
        self.kv_state = KVStateControl(self)

    # ── ModelManager ──────────────────────────────────────────

    async def load_model(self, model_name: str, unload_previous: bool = True) -> None:
        """Load a model through the llama.cpp server model-management endpoint."""
        loaded_model = self._sync_loaded_model_state()
        if loaded_model == model_name:
            self.logger.info(f"Model {model_name} is already loaded.")
            self._wait_for_model_status(model_name, expected_status="loaded")
            return

        if unload_previous and loaded_model is not None:
            await self.unload_model()

        model = {"model": model_name}
        response = requests.post(f"{self.base_url}/models/load", json=model)
        if response.status_code == 200:
            self.logger.info(f"Successfully loaded model: {model_name}")
            self.currently_loaded_model = model_name
            self.kv_state.update_shared_state(currently_loaded_model=model_name)
            self._wait_for_model_status(model_name, expected_status="loaded")
        elif response.status_code == 400 and "model is already running" in response.text.lower():
            self.logger.info(f"Specified model: {model_name} is already running.")
            self.currently_loaded_model = model_name
            self.kv_state.update_shared_state(currently_loaded_model=model_name)
            self._wait_for_model_status(model_name, expected_status="loaded")
        else:
            self.logger.error(f"Failed to load model: {model_name}. Response: {response.text}")

    async def unload_model(self) -> None:
        """Unload the currently tracked model from the llama.cpp server."""
        loaded_model = self._sync_loaded_model_state()
        if loaded_model is None:
            return
        model = {"model": loaded_model}
        response = requests.post(f"{self.base_url}/models/unload", json=model)
        if response.status_code == 200:
            self.logger.info(f"Successfully unloaded model: {loaded_model}")
            unloaded_model = loaded_model
            self._set_loaded_model_state(None)
            self._wait_for_model_status(unloaded_model, expected_status="unloaded")
        elif response.status_code == 400 and "model is not running" in response.text.lower():
            self.logger.info(f"Model {loaded_model} is not running.")
            self._set_loaded_model_state(None)
        else:
            self.logger.error(f"Failed to unload model: {loaded_model}. Response: {response.text}")

    def get_current_model(self) -> Optional[str]:
        """Return the model name this adapter currently tracks as loaded."""
        return self.currently_loaded_model or self.kv_state.read_shared_state().get("currently_loaded_model")

    def _discover_loaded_model(self) -> Optional[str]:
        """Query the llama.cpp API for the currently loaded model, if any."""
        return self._sync_loaded_model_state()

    def _fetch_models(self) -> List[Dict[str, Any]]:
        response = requests.get(f"{self.api_uri_v1}/models", timeout=10)
        response.raise_for_status()
        payload = response.json()
        return payload.get("data", [])

    def count_text_tokens(self, text: str, model_name: Optional[str] = None) -> int:
        """Count text tokens using llama.cpp when available, else fall back to a stable estimate."""
        payload: Dict[str, Any] = {
            "content": text or "",
            "add_special": False,
            "with_pieces": False,
        }
        target_model = model_name or self.currently_loaded_model or self.get_current_model()
        if target_model:
            payload["model"] = target_model
        try:
            response = requests.post(f"{self.base_url}/tokenize", json=payload, timeout=15)
            response.raise_for_status()
            data = response.json()
            tokens = data.get("tokens", [])
            if isinstance(tokens, list):
                return len(tokens)
        except requests.RequestException:
            pass
        return max(1, len(text or "") // 4) if text else 0

    def count_message_tokens(
        self,
        messages: List[Dict[str, Any]],
        image: str = "",
        model_name: Optional[str] = None,
    ) -> int:
        """Estimate prompt tokens by normalizing messages, then asking llama.cpp to tokenize the rendered text."""
        rendered_parts: List[str] = []
        for message in messages:
            role = str(message.get("role", "")).strip() or "unknown"
            content = message.get("content", "")
            if isinstance(content, list):
                flattened: List[str] = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        flattened.append(str(item.get("text", "")))
                    elif isinstance(item, dict) and item.get("type") == "image_url":
                        flattened.append("[image]")
                    else:
                        flattened.append(str(item))
                content_text = "\n".join(part for part in flattened if part)
            else:
                content_text = str(content)
            rendered_parts.append(f"{role}: {content_text}")
        if image:
            rendered_parts.append("user: [image-attached]")
        rendered_prompt = "\n".join(rendered_parts)
        return self.count_text_tokens(rendered_prompt, model_name=model_name)

    def _set_loaded_model_state(self, model_name: Optional[str]) -> None:
        self.currently_loaded_model = model_name
        self.kv_state.update_shared_state(currently_loaded_model=model_name)

    def _sync_loaded_model_state(self) -> Optional[str]:
        try:
            models = self._fetch_models()
        except requests.RequestException as exc:
            self.logger.warning("Failed to query loaded llama.cpp models: %s", exc)
            return self.currently_loaded_model or self.kv_state.read_shared_state().get("currently_loaded_model")

        loaded_ids = [
            model.get("id")
            for model in models
            if model.get("status", {}).get("value") == "loaded" and model.get("id")
        ]
        if not loaded_ids:
            self._set_loaded_model_state(None)
            return None

        chosen = None
        if self.currently_loaded_model in loaded_ids:
            chosen = self.currently_loaded_model
        else:
            shared_model = self.kv_state.read_shared_state().get("currently_loaded_model")
            if shared_model in loaded_ids:
                chosen = shared_model
            else:
                chosen = loaded_ids[0]

        self._set_loaded_model_state(chosen)
        return chosen

    def _get_model_metadata(self, model_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        target = model_name or self.currently_loaded_model or self.get_current_model()
        if not target:
            return None
        for model in self._fetch_models():
            if model.get("id") == target:
                return model
        return None

    def _is_multimodal_model(self, model_name: Optional[str] = None) -> bool:
        metadata = self._get_model_metadata(model_name)
        if not metadata:
            return False
        modalities = metadata.get("architecture", {}).get("input_modalities", [])
        normalized = {str(modality).strip().lower() for modality in modalities}
        return len(normalized) > 1 or "image" in normalized or "audio" in normalized

    def _wait_for_model_status(
        self,
        model_name: str,
        *,
        expected_status: str,
        timeout_seconds: float = 30.0,
    ) -> None:
        deadline = time.time() + timeout_seconds
        last_status: Optional[str] = None
        last_error: Optional[Exception] = None

        while time.time() < deadline:
            try:
                metadata = self._get_model_metadata(model_name)
                if metadata is not None:
                    last_status = metadata.get("status", {}).get("value")
                    if last_status == expected_status:
                        return
            except Exception as exc:
                last_error = exc
            time.sleep(0.25)

        if last_error is not None:
            raise RuntimeError(
                f"Timed out waiting for model {model_name} to report status {expected_status}."
            ) from last_error
        raise RuntimeError(
            f"Timed out waiting for model {model_name} to report status {expected_status}. "
            f"Last seen status was {last_status!r}."
        )

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

    def save_current_kv_state(self, messages) -> Optional[Path]:
        """Request a KV state save for slot 0 and return the expected local file path."""
        if self.currently_loaded_model is None:
            self.currently_loaded_model = self.get_current_model() or self._discover_loaded_model()

        save_path = self.kv_state.kv_state_dir() / self.kv_state.safe_kv_state_filename()
        json_path = save_path.with_suffix(".json")
        json_path.write_text(
            json.dumps(
                {
                    "model_name": self.currently_loaded_model,
                    "messages": messages,
                    "saved_at": datetime.now().isoformat(),
                    "kv_cache_saved": False,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        if self._is_multimodal_model(self.currently_loaded_model):
            self.kv_state.update_shared_state(currently_loaded_model=self.currently_loaded_model)
            self.kv_state.push_kv_state(save_path)
            self.logger.info(
                "Skipping KV slot save for multimodal model %s; preserving Python message state only.",
                self.currently_loaded_model,
            )
            return save_path

        slot_base_url = self._slot_base_url()

        payload = {"filename": save_path.name}
        if self.currently_loaded_model:
            payload["model"] = self.currently_loaded_model

        response = requests.post(
            f"{slot_base_url}/slots/0?action=save",
            json=payload,
            timeout=30,
        )
        if response.status_code == 200:
            json_path.write_text(
                json.dumps(
                    {
                        "model_name": self.currently_loaded_model,
                        "messages": messages,
                        "saved_at": datetime.now().isoformat(),
                        "kv_cache_saved": True,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            self.kv_state.update_shared_state(currently_loaded_model=self.currently_loaded_model)
            self.kv_state.push_kv_state(save_path)
            self.logger.info(
                f"Successfully requested KV state save to {save_path}. "
                f"llama.cpp response: {response.text}"
            )
            return save_path

        self.logger.error(
            f"Failed to save KV state. Status: {response.status_code}. "
            f"Response: {response.text}"
        )
        return None

    def restore_kv_state(self, kv_state_file: str) -> None:
        """Restore slot 0 KV state from the given saved state filename or path."""
        kv_state_path = Path(kv_state_file)
        slot_base_url = self._slot_base_url()
        payload = {"filename": kv_state_path.name}
        if self.currently_loaded_model:
            payload["model"] = self.currently_loaded_model

        response = requests.post(
            f"{slot_base_url}/slots/0?action=restore",
            json=payload,
            timeout=30,
        )

        if response.status_code == 200:
            self.logger.info(f"Successfully restored KV state from {kv_state_file}")
            return

        raise RuntimeError(
            f"Failed to restore KV state from {kv_state_file}. Response: {response.text}"
        )
        
    def _wait_for_model_restore_ready(self, model_name: str, timeout_seconds: float = 15.0) -> None:
        """Wait until the loaded model exposes a slot endpoint suitable for KV restore."""
        deadline = time.time() + timeout_seconds
        last_error: Optional[Exception] = None

        while time.time() < deadline:
            try:
                discovered_model = self.get_current_model() or self._discover_loaded_model()
                if discovered_model == model_name:
                    self._slot_base_url()
                    return
            except Exception as exc:
                last_error = exc
            time.sleep(0.25)

        if last_error is not None:
            raise RuntimeError(
                f"Timed out waiting for model {model_name} to become restore-ready."
            ) from last_error
        raise RuntimeError(
            f"Timed out waiting for model {model_name} to become restore-ready."
        )
    
    async def save_and_unload(self, messages) -> Optional[Path]:
        """Save the current KV state, then unload the model if the save succeeds."""
        save_path = self.save_current_kv_state(messages)
        if save_path is not None:
            await self.unload_model()
        return save_path
    
    async def load_and_restore(self) -> Path:
        """Load the model named by a KV state file, then restore that KV state."""
        kv_state_file = self.kv_state.peek_kv_state()
        model_name = self.kv_state.extract_model_name_from_kv_state_file(kv_state_file)
        if self.currently_loaded_model != model_name:
            await self.load_model(model_name)
        kv_state_path = Path(kv_state_file)
        if kv_state_path.exists():
            self._wait_for_model_restore_ready(model_name)
            self.restore_kv_state(kv_state_file)
        else:
            self.logger.info(
                "No KV cache file exists for %s; restoring Python message context only.",
                model_name,
            )
        self.kv_state.pop_kv_state()
        return Path(kv_state_file)

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
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> Iterator:
        """
        Create a streaming chat completion through the OpenAI-compatible API.

        If `image` is provided, it is attached to the final user message as a
        base64 data URL for multimodal models.
        """
        copy_messages = copy.deepcopy(messages)
        if image and copy_messages:
            import base64
            with open(image, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode("utf-8")

            target_message = None
            for message in reversed(copy_messages):
                if message.get("role") == "user":
                    target_message = message
                    break

            if target_message is not None:
                content = target_message.get("content", "")
                image_part = {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                }
                if isinstance(content, list):
                    target_message["content"] = list(content) + [image_part]
                else:
                    target_message["content"] = [
                        {"type": "text", "text": str(content)},
                        image_part,
                    ]

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": copy_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        extra_body: Dict[str, Any] = {}
        if top_k is not None and top_k > 0:
            extra_body["top_k"] = top_k
        if extra_body:
            kwargs["extra_body"] = extra_body
        if tools:
            kwargs["tools"] = tools

        return await self.client.chat.completions.create(**kwargs)
