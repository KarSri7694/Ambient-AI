from __future__ import annotations

import copy
import json
from time import perf_counter
from typing import Any, Optional

from application.ports.LLMProvider import LLMProvider
from benchmarking.metrics import estimate_message_tokens, estimate_text_tokens


class RealWorldTracingLLMProvider(LLMProvider):
    """Capture exact model requests/responses while preserving the provider API."""

    def __init__(self, provider: LLMProvider, emit):
        self.provider = provider
        self.emit = emit

    def __getattr__(self, name: str):
        return getattr(self.provider, name)

    async def load_model(self, model_name: str) -> None:
        self.emit("model", "model_load_started", {"model": model_name}, model=model_name, status="running")
        started = perf_counter()
        try:
            result = await self.provider.load_model(model_name)
        except Exception as exc:
            self.emit("model", "model_load_failed", {"error": str(exc)}, model=model_name,
                      duration_ms=int((perf_counter() - started) * 1000), status="failed")
            raise
        self.emit("model", "model_loaded", {}, model=model_name,
                  duration_ms=int((perf_counter() - started) * 1000))
        return result

    async def unload_model(self) -> None:
        if hasattr(self.provider, "unload_model"):
            return await self.provider.unload_model()
        return None

    async def save_and_unload(self, messages):
        return await self.provider.save_and_unload(messages)

    async def load_and_restore(self):
        return await self.provider.load_and_restore()

    def generate_response(self, prompt: str, image: str = "") -> str:
        started = perf_counter()
        self.emit("model", "model_request", {
            "messages": [{"role": "user", "content": prompt}], "tools": None,
            "image_path": image or None,
        })
        try:
            response = self.provider.generate_response(prompt, image=image)
        except Exception as exc:
            self.emit("model", "model_response", {"error": str(exc)}, status="failed",
                      duration_ms=int((perf_counter() - started) * 1000))
            raise
        self.emit("model", "model_response", {
            "response_text": response,
            "prompt_tokens": estimate_text_tokens(prompt),
            "completion_tokens": estimate_text_tokens(response),
        }, duration_ms=int((perf_counter() - started) * 1000))
        return response

    async def chat_completion_stream(
        self, model: str, messages: list[dict[str, Any]], tools: Optional[list[dict[str, Any]]] = None,
        image: str = "", temperature: Optional[float] = None, top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None, response_format: Optional[dict[str, Any]] = None,
        chat_template_kwargs: Optional[dict[str, Any]] = None,
        request_timeout_seconds: Optional[float] = None,
    ):
        started = perf_counter()
        request_payload = {
            "messages": copy.deepcopy(messages), "tools": copy.deepcopy(tools),
            "image_path": image or None, "temperature": temperature, "top_p": top_p, "top_k": top_k,
            "max_tokens": max_tokens, "response_format": copy.deepcopy(response_format),
            "chat_template_kwargs": copy.deepcopy(chat_template_kwargs),
            "request_timeout_seconds": request_timeout_seconds,
        }
        self.emit("model", "model_request", request_payload, model=model, status="running")
        try:
            provider_kwargs: dict[str, Any] = {
                "model": model, "messages": messages, "tools": tools, "image": image,
                "temperature": temperature, "top_p": top_p, "top_k": top_k,
            }
            if max_tokens is not None:
                provider_kwargs["max_tokens"] = max_tokens
            if response_format is not None:
                provider_kwargs["response_format"] = response_format
            if chat_template_kwargs is not None:
                provider_kwargs["chat_template_kwargs"] = chat_template_kwargs
            if request_timeout_seconds is not None:
                provider_kwargs["request_timeout_seconds"] = request_timeout_seconds
            completion = self.provider.chat_completion_stream(**provider_kwargs)
            if hasattr(completion, "__await__"):
                completion = await completion
        except Exception as exc:
            self.emit("model", "model_response", {"error": str(exc)}, model=model, status="failed",
                      duration_ms=int((perf_counter() - started) * 1000))
            raise

        async def _wrapped():
            response_parts: list[str] = []
            reasoning_parts: list[str] = []
            tool_calls: dict[int, dict[str, Any]] = {}
            usage: dict[str, int] = {}
            error: Exception | None = None
            try:
                async for chunk in completion:
                    chunk_usage = getattr(chunk, "usage", None)
                    if chunk_usage is not None:
                        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
                            value = getattr(chunk_usage, name, None)
                            if value is not None:
                                usage[name] = int(value)
                    if getattr(chunk, "choices", None):
                        delta = chunk.choices[0].delta
                        response_parts.append(getattr(delta, "content", None) or "")
                        reasoning_parts.append(getattr(delta, "reasoning_content", None) or "")
                        for part in getattr(delta, "tool_calls", None) or []:
                            target = tool_calls.setdefault(part.index, {
                                "id": "", "type": "function", "function": {"name": "", "arguments": ""}
                            })
                            target["id"] += getattr(part, "id", None) or ""
                            function = getattr(part, "function", None)
                            target["function"]["name"] += getattr(function, "name", None) or ""
                            target["function"]["arguments"] += getattr(function, "arguments", None) or ""
                    yield chunk
            except Exception as exc:
                error = exc
                raise
            finally:
                response_text = "".join(response_parts)
                prompt_tokens = usage.get("prompt_tokens", estimate_message_tokens(messages, image=image))
                completion_tokens = usage.get("completion_tokens", estimate_text_tokens(response_text))
                payload = {
                    "response_text": response_text,
                    "reasoning_text": "".join(reasoning_parts) or None,
                    "tool_calls": list(tool_calls.values()),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": usage.get("total_tokens", prompt_tokens + completion_tokens),
                    "error": str(error) if error else None,
                }
                self.emit("model", "model_response", payload, model=model,
                          status="failed" if error else "completed",
                          duration_ms=int((perf_counter() - started) * 1000))

        return _wrapped()
