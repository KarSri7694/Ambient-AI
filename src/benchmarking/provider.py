from __future__ import annotations

from time import perf_counter
from typing import Any, Dict, List, Optional

from application.ports.LLMProvider import LLMProvider
from benchmarking.metrics import (
    BenchmarkCallRecord,
    aggregate_metrics,
    elapsed_seconds,
    estimate_message_tokens,
    estimate_text_tokens,
    now_iso,
    safe_rate,
)


class BenchmarkingLLMProvider(LLMProvider):
    """Wrap an LLM provider and capture per-call benchmark metrics."""

    def __init__(self, provider: LLMProvider):
        self.provider = provider
        self._calls: List[BenchmarkCallRecord] = []

    def __getattr__(self, name: str):
        return getattr(self.provider, name)

    def reset_benchmark_metrics(self) -> None:
        self._calls = []

    def benchmark_metrics(self):
        return aggregate_metrics(self._calls)

    async def load_model(self, model_name: str) -> None:
        return await self.provider.load_model(model_name)

    async def save_and_unload(self, messages: List[Dict[str, Any]]):
        return await self.provider.save_and_unload(messages)

    async def load_and_restore(self):
        return await self.provider.load_and_restore()

    def generate_response(self, prompt: str, image: str = "") -> str:
        started_at = now_iso()
        started = perf_counter()
        error_text = None
        response = ""
        try:
            response = self.provider.generate_response(prompt, image=image)
            return response
        except Exception as exc:
            error_text = str(exc)
            raise
        finally:
            ended = perf_counter()
            prompt_tokens, method = self._count_prompt_tokens(
                messages=[{"role": "user", "content": prompt}],
                image=image,
            )
            completion_tokens, completion_method = self._count_completion_tokens(response)
            token_method = method if method == completion_method else "mixed"
            prefill_seconds = elapsed_seconds(started, ended)
            record = BenchmarkCallRecord(
                started_at=started_at,
                completed_at=now_iso(),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                prefill_seconds=prefill_seconds,
                generation_seconds=0.0,
                prefill_tokens_per_second=safe_rate(prompt_tokens, prefill_seconds),
                generation_tokens_per_second=0.0,
                token_count_method=token_method,
                response_text=response,
                error_text=error_text,
            )
            self._calls.append(record)

    async def chat_completion_stream(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        image: str = "",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ):
        started_at = now_iso()
        started = perf_counter()
        first_token_at: Optional[float] = None
        prompt_tokens, prompt_method = self._count_prompt_tokens(messages=messages, image=image)
        response_parts: List[str] = []
        usage_prompt_tokens: Optional[int] = None
        usage_completion_tokens: Optional[int] = None
        usage_total_tokens: Optional[int] = None

        completion = self.provider.chat_completion_stream(
            model=model,
            messages=messages,
            tools=tools,
            image=image,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        if hasattr(completion, "__await__"):
            completion = await completion

        async def _wrapped_stream():
            error_text = None
            nonlocal first_token_at, usage_prompt_tokens, usage_completion_tokens, usage_total_tokens
            try:
                async for chunk in completion:
                    usage = getattr(chunk, "usage", None)
                    if usage is not None:
                        usage_prompt_tokens = self._coerce_usage_value(getattr(usage, "prompt_tokens", None))
                        usage_completion_tokens = self._coerce_usage_value(getattr(usage, "completion_tokens", None))
                        usage_total_tokens = self._coerce_usage_value(getattr(usage, "total_tokens", None))
                    if not getattr(chunk, "choices", None):
                        yield chunk
                        continue
                    delta = chunk.choices[0].delta
                    piece = getattr(delta, "content", None) or ""
                    if piece and first_token_at is None:
                        first_token_at = perf_counter()
                    if piece:
                        response_parts.append(piece)
                    yield chunk
            except Exception as exc:
                error_text = str(exc)
                raise
            finally:
                ended = perf_counter()
                response_text = "".join(response_parts)
                if usage_prompt_tokens is not None:
                    prompt_tokens_local = usage_prompt_tokens
                    prompt_method_local = "server_usage"
                else:
                    prompt_tokens_local = prompt_tokens
                    prompt_method_local = prompt_method
                if usage_completion_tokens is not None:
                    completion_tokens = usage_completion_tokens
                    completion_method = "server_usage"
                else:
                    completion_tokens, completion_method = self._count_completion_tokens(response_text)
                prefill_end = first_token_at if first_token_at is not None else ended
                prefill_seconds = elapsed_seconds(started, prefill_end)
                generation_seconds = elapsed_seconds(prefill_end, ended) if first_token_at is not None else 0.0
                if usage_total_tokens is not None:
                    total_tokens = usage_total_tokens
                else:
                    total_tokens = prompt_tokens_local + completion_tokens
                method = (
                    prompt_method_local
                    if prompt_method_local == completion_method
                    else ("server_usage" if "server_usage" in {prompt_method_local, completion_method} else "mixed")
                )
                self._calls.append(
                    BenchmarkCallRecord(
                        started_at=started_at,
                        completed_at=now_iso(),
                        prompt_tokens=prompt_tokens_local,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        prefill_seconds=prefill_seconds,
                        generation_seconds=generation_seconds,
                        prefill_tokens_per_second=safe_rate(prompt_tokens_local, prefill_seconds),
                        generation_tokens_per_second=safe_rate(completion_tokens, generation_seconds),
                        token_count_method=method,
                        response_text=response_text,
                        error_text=error_text,
                    )
                )

        return _wrapped_stream()

    def _count_prompt_tokens(self, *, messages: List[Dict[str, Any]], image: str) -> tuple[int, str]:
        if hasattr(self.provider, "count_message_tokens"):
            try:
                count = int(self.provider.count_message_tokens(messages, image=image))
                return max(0, count), "provider_tokenize"
            except Exception:
                pass
        return estimate_message_tokens(messages, image=image), "estimated_chars_div4"

    def _count_completion_tokens(self, text: str) -> tuple[int, str]:
        if hasattr(self.provider, "count_text_tokens"):
            try:
                count = int(self.provider.count_text_tokens(text))
                return max(0, count), "provider_tokenize"
            except Exception:
                pass
        return estimate_text_tokens(text), "estimated_chars_div4"

    def _coerce_usage_value(self, value) -> Optional[int]:
        try:
            if value is None:
                return None
            return max(0, int(value))
        except (TypeError, ValueError):
            return None
