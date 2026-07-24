import copy
import json
import uuid
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional

from application.ports.LLMProvider import LLMProvider
from application.services.interaction_trace import current_interaction_metadata, current_interaction_source
from application.services.resource_governor_service import ResourceUnavailableError
from core.models import InteractionLogEntry
from infrastructure.adapter.SQLiteInteractionLogAdapter import SQLiteInteractionLogAdapter


class LoggingLLMProvider(LLMProvider):
    """Wrap an LLM provider and persist every request/response pair."""

    def __init__(
        self,
        provider: LLMProvider,
        log_store: SQLiteInteractionLogAdapter,
        current_response_path: str | None = None,
        capture_store: Any = None,
        residency_manager: Any = None,
    ):
        self.provider = provider
        self.log_store = log_store
        self.current_response_path = Path(current_response_path) if current_response_path else None
        self.capture_store = capture_store
        self.residency_manager = residency_manager
        self._current_response_state: Dict[str, Any] | None = None
        if self.current_response_path is not None:
            self.current_response_path.parent.mkdir(parents=True, exist_ok=True)

    def __getattr__(self, name: str):
        return getattr(self.provider, name)

    def _messages_json(self, messages: Any, *, source: str, interaction_id: str) -> str:
        serialized = json.dumps(copy.deepcopy(messages), ensure_ascii=False, indent=2)
        if self.capture_store is not None and source.startswith(
            ("autonomy", "passive_observer", "audio", "transcript")
        ):
            reference = self.capture_store.store_bytes(
                serialized.encode("utf-8"),
                original_name=f"llm_messages_{interaction_id}.json",
                kind="llm_messages",
                mime_type="application/json",
            )
            return json.dumps({"protected_payload_ref": reference})
        return serialized

    async def load_model(self, model_name: str) -> None:
        if self.residency_manager is not None:
            source = current_interaction_source()
            background = source.startswith(("autonomy", "passive_observer", "audio", "reflection"))
            decision = await self.residency_manager.load_model(
                model_name,
                role=source or "interactive",
                background=background,
                user_active=not bool(current_interaction_metadata().get("user_idle")),
            )
            if not decision.allowed:
                raise ResourceUnavailableError(decision)
            return None
        return await self.provider.load_model(model_name)

    async def unload_model(self) -> None:
        if self.residency_manager is not None:
            return await self.residency_manager.unload_model(reason="LLM interaction released model")
        return await self.provider.unload_model()

    async def save_and_unload(self, messages: List[Dict[str, Any]]):
        if self.residency_manager is not None:
            return await self.residency_manager.save_and_unload(messages)
        return await self.provider.save_and_unload(messages)

    async def load_and_restore(self):
        if self.residency_manager is not None:
            source = current_interaction_source()
            return await self.residency_manager.load_and_restore(
                role=source or "state_restore",
                background=source.startswith(("autonomy", "passive_observer", "audio", "reflection")),
                user_active=not bool(current_interaction_metadata().get("user_idle")),
            )
        return await self.provider.load_and_restore()

    def generate_response(self, prompt: str, image: str = "") -> str:
        started_at = datetime.now().isoformat()
        started = perf_counter()
        interaction_id = uuid.uuid4().hex
        source = current_interaction_source()
        metadata = current_interaction_metadata()
        interaction_run_id = metadata.get("interaction_run_id") or uuid.uuid4().hex
        self._write_current_response(
            source=source,
            model=getattr(self.provider, "currently_loaded_model", "") or "unknown",
            created_at=started_at,
            interaction_run_id=interaction_run_id,
            response_text="",
            reasoning_text="",
            status="streaming",
            metadata=metadata,
        )
        try:
            response = self.provider.generate_response(prompt, image=image)
            self._write_current_response(
                source=source,
                model=getattr(self.provider, "currently_loaded_model", "") or "unknown",
                created_at=started_at,
                interaction_run_id=interaction_run_id,
                response_text=response,
                reasoning_text="",
                status="completed",
                metadata=metadata,
            )
            self._log(
                InteractionLogEntry(
                    interaction_id=interaction_id,
                    interaction_run_id=interaction_run_id,
                    created_at=started_at,
                    completed_at=datetime.now().isoformat(),
                    source=source,
                    model=getattr(self.provider, "currently_loaded_model", "") or "unknown",
                    messages_json=self._messages_json([{"role": "user", "content": prompt}], source=source, interaction_id=interaction_id),
                    image_path=image or None,
                    response_text=response,
                    duration_ms=int((perf_counter() - started) * 1000),
                    metadata_json=json.dumps(metadata, ensure_ascii=False, indent=2) if metadata else None,
                    report_json=None,
                )
            )
            return response
        except Exception as exc:
            self._write_current_response(
                source=source,
                model=getattr(self.provider, "currently_loaded_model", "") or "unknown",
                created_at=started_at,
                interaction_run_id=interaction_run_id,
                response_text="",
                reasoning_text="",
                status="error",
                metadata=metadata,
                error_text=str(exc),
            )
            self._log(
                InteractionLogEntry(
                    interaction_id=interaction_id,
                    interaction_run_id=interaction_run_id,
                    created_at=started_at,
                    completed_at=datetime.now().isoformat(),
                    source=source,
                    model=getattr(self.provider, "currently_loaded_model", "") or "unknown",
                    messages_json=self._messages_json([{"role": "user", "content": prompt}], source=source, interaction_id=interaction_id),
                    image_path=image or None,
                    error_text=str(exc),
                    duration_ms=int((perf_counter() - started) * 1000),
                    metadata_json=json.dumps(metadata, ensure_ascii=False, indent=2) if metadata else None,
                    report_json=None,
                )
            )
            raise

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
        started_at = datetime.now().isoformat()
        started = perf_counter()
        interaction_id = uuid.uuid4().hex
        source = current_interaction_source()
        metadata = current_interaction_metadata()
        interaction_run_id = metadata.get("interaction_run_id") or uuid.uuid4().hex
        response_text_parts: List[str] = []
        reasoning_text_parts: List[str] = []
        streamed_tool_calls: Dict[int, Dict[str, Any]] = {}

        try:
            completion = await self.provider.chat_completion_stream(
                model=model,
                messages=messages,
                tools=tools,
                image=image,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            self._write_current_response(
                source=source,
                model=model,
                created_at=started_at,
                interaction_run_id=interaction_run_id,
                response_text="",
                reasoning_text="",
                status="streaming",
                metadata=metadata,
            )

            async def _wrapped_stream():
                try:
                    async for chunk in completion:
                        if not getattr(chunk, "choices", None):
                            yield chunk
                            continue
                        delta = chunk.choices[0].delta
                        if getattr(delta, "content", None):
                            response_text_parts.append(delta.content)
                        reasoning = getattr(delta, "reasoning_content", None)
                        if reasoning:
                            reasoning_text_parts.append(reasoning)
                        if getattr(delta, "tool_calls", None):
                            for tc_delta in delta.tool_calls:
                                idx = tc_delta.index
                                existing = streamed_tool_calls.setdefault(
                                    idx,
                                    {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                                )
                                if tc_delta.id:
                                    existing["id"] = tc_delta.id
                                if tc_delta.function.name:
                                    existing["function"]["name"] += tc_delta.function.name
                                if tc_delta.function.arguments:
                                    existing["function"]["arguments"] += tc_delta.function.arguments
                        self._write_current_response(
                            source=source,
                            model=model,
                            created_at=started_at,
                            interaction_run_id=interaction_run_id,
                            response_text="".join(response_text_parts),
                            reasoning_text="".join(reasoning_text_parts),
                            status="streaming",
                            metadata=metadata,
                        )
                        yield chunk
                except Exception as exc:
                    self._write_current_response(
                        source=source,
                        model=model,
                        created_at=started_at,
                        interaction_run_id=interaction_run_id,
                        response_text="".join(response_text_parts),
                        reasoning_text="".join(reasoning_text_parts),
                        status="error",
                        metadata=metadata,
                        error_text=str(exc),
                    )
                    self._log(
                        InteractionLogEntry(
                            interaction_id=interaction_id,
                            interaction_run_id=interaction_run_id,
                            created_at=started_at,
                            completed_at=datetime.now().isoformat(),
                            source=source,
                            model=model,
                            messages_json=self._messages_json(messages, source=source, interaction_id=interaction_id),
                            tools_json=json.dumps(tools, ensure_ascii=False, indent=2) if tools is not None else None,
                            image_path=image or None,
                            response_text="".join(response_text_parts) or None,
                            reasoning_text="".join(reasoning_text_parts) or None,
                            tool_calls_json=json.dumps(list(streamed_tool_calls.values()), ensure_ascii=False, indent=2)
                            if streamed_tool_calls
                            else None,
                            error_text=str(exc),
                            duration_ms=int((perf_counter() - started) * 1000),
                            metadata_json=json.dumps(metadata, ensure_ascii=False, indent=2) if metadata else None,
                            report_json=None,
                        )
                    )
                    raise
                else:
                    self._write_current_response(
                        source=source,
                        model=model,
                        created_at=started_at,
                        interaction_run_id=interaction_run_id,
                        response_text="".join(response_text_parts),
                        reasoning_text="".join(reasoning_text_parts),
                        status="completed",
                        metadata=metadata,
                    )
                    self._log(
                        InteractionLogEntry(
                            interaction_id=interaction_id,
                            interaction_run_id=interaction_run_id,
                            created_at=started_at,
                            completed_at=datetime.now().isoformat(),
                            source=source,
                            model=model,
                            messages_json=self._messages_json(messages, source=source, interaction_id=interaction_id),
                            tools_json=json.dumps(tools, ensure_ascii=False, indent=2) if tools is not None else None,
                            image_path=image or None,
                            response_text="".join(response_text_parts) or None,
                            reasoning_text="".join(reasoning_text_parts) or None,
                            tool_calls_json=json.dumps(list(streamed_tool_calls.values()), ensure_ascii=False, indent=2)
                            if streamed_tool_calls
                            else None,
                            duration_ms=int((perf_counter() - started) * 1000),
                            metadata_json=json.dumps(metadata, ensure_ascii=False, indent=2) if metadata else None,
                            report_json=None,
                        )
                    )

            return _wrapped_stream()
        except Exception as exc:
            self._write_current_response(
                source=source,
                model=model,
                created_at=started_at,
                interaction_run_id=interaction_run_id,
                response_text="".join(response_text_parts),
                reasoning_text="".join(reasoning_text_parts),
                status="error",
                metadata=metadata,
                error_text=str(exc),
            )
            self._log(
                InteractionLogEntry(
                    interaction_id=interaction_id,
                    interaction_run_id=interaction_run_id,
                    created_at=started_at,
                    completed_at=datetime.now().isoformat(),
                    source=source,
                    model=model,
                    messages_json=self._messages_json(messages, source=source, interaction_id=interaction_id),
                    tools_json=json.dumps(tools, ensure_ascii=False, indent=2) if tools is not None else None,
                    image_path=image or None,
                    error_text=str(exc),
                    duration_ms=int((perf_counter() - started) * 1000),
                    metadata_json=json.dumps(metadata, ensure_ascii=False, indent=2) if metadata else None,
                    report_json=None,
                )
            )
            raise

    def _log(self, entry: InteractionLogEntry) -> None:
        self.log_store.insert(entry)

    def attach_report(self, interaction_run_id: str, report: Dict[str, Any]) -> None:
        report_json = json.dumps(report, ensure_ascii=False, indent=2)
        self.log_store.attach_report(interaction_run_id, report_json)
        if self._current_response_state and self._current_response_state.get("interaction_run_id") == interaction_run_id:
            self._current_response_state["report"] = report
            self._flush_current_response_state()

    def _write_current_response(
        self,
        *,
        source: str,
        model: str,
        created_at: str,
        interaction_run_id: str | None,
        response_text: str,
        reasoning_text: str,
        status: str,
        metadata: Dict[str, Any],
        error_text: str | None = None,
    ) -> None:
        if self.current_response_path is None:
            return
        self._current_response_state = {
            "status": status,
            "created_at": created_at,
            "source": source,
            "model": model,
            "metadata": dict(metadata),
            "error_text": error_text,
            "reasoning_text": reasoning_text,
            "response_text": response_text,
            "interaction_run_id": interaction_run_id,
            "report": None,
            "protected": source.startswith(("autonomy", "passive_observer", "audio", "transcript")),
        }
        self._flush_current_response_state()

    def _flush_current_response_state(self) -> None:
        if self.current_response_path is None or self._current_response_state is None:
            return
        state = self._current_response_state
        if state.get("protected"):
            return
        lines = [
            f"status: {state['status']}",
            f"created_at: {state['created_at']}",
            f"source: {state['source']}",
            f"model: {state['model']}",
        ]
        if state.get("interaction_run_id"):
            lines.append(f"interaction_run_id: {state['interaction_run_id']}")
        if state["metadata"]:
            lines.append(f"metadata: {json.dumps(state['metadata'], ensure_ascii=False)}")
        report = state.get("report")
        if report:
            if report.get("title"):
                lines.extend(["", "## Report Title", report["title"]])
            if report.get("summary"):
                lines.extend(["", "## Report Summary", report["summary"]])
            if report.get("artifact_path"):
                lines.extend(["", "## Report Artifact", report["artifact_path"]])
        if state.get("error_text"):
            lines.extend(["", "## Error", state["error_text"]])
        if state.get("reasoning_text"):
            lines.extend(["", "## Reasoning", state["reasoning_text"]])
        lines.extend(["", "## Response", state.get("response_text") or ""])
        self.current_response_path.write_text("\n".join(lines), encoding="utf-8")
