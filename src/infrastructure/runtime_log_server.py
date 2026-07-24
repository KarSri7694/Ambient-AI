import asyncio
import hashlib
import json
import logging
import mimetypes
import queue
import threading
import uuid
from collections import deque
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles

from application.services.training_data_service import TrainingDataService
from infrastructure.adapter.SQLiteBenchmarkAdapter import SQLiteBenchmarkAdapter
from infrastructure.adapter.SQLiteChatAdapter import ChatEventBroker, SQLiteChatAdapter
from infrastructure.adapter.SQLiteInteractionLogAdapter import SQLiteInteractionLogAdapter
from infrastructure.adapter.SQLiteTaskQueueAdapter import SQLiteTaskQueueAdapter
from core.models import AmbientEvent
from infrastructure.adapter.SQLiteTrainingDataAdapter import SQLiteTrainingDataAdapter


UI_SOURCE_ROOT = Path(__file__).resolve().parent / "runtime_ui"
UI_ROOT = UI_SOURCE_ROOT / "dist"
UI_INDEX_PATH = UI_ROOT / "index.html"


class RuntimeLogBuffer:
    def __init__(self, max_entries: int = 2000):
        self.max_entries = max(100, max_entries)
        self._entries: deque[dict[str, Any]] = deque(maxlen=self.max_entries)
        self._lock = threading.Lock()
        self._next_id = 1

    def append(self, record: logging.LogRecord, rendered_message: str) -> dict[str, Any]:
        entry = {
            "id": 0,
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "logger": record.name,
            "level": record.levelname,
            "message": rendered_message,
            "pathname": record.pathname,
            "lineno": record.lineno,
            "thread": record.threadName,
        }
        with self._lock:
            entry["id"] = self._next_id
            self._next_id += 1
            self._entries.append(entry)
        return entry

    def snapshot(self, *, after_id: int = 0, limit: int = 200) -> list[dict[str, Any]]:
        normalized_limit = max(1, min(limit, self.max_entries))
        with self._lock:
            rows = [entry for entry in self._entries if entry["id"] > after_id]
        if len(rows) > normalized_limit:
            return rows[-normalized_limit:]
        return rows

    def latest_id(self) -> int:
        with self._lock:
            if not self._entries:
                return 0
            return self._entries[-1]["id"]


class RuntimeLogBufferHandler(logging.Handler):
    def __init__(self, buffer: RuntimeLogBuffer):
        super().__init__()
        self.buffer = buffer

    def emit(self, record: logging.LogRecord) -> None:
        try:
            rendered = self.format(record)
        except Exception:
            rendered = record.getMessage()
        self.buffer.append(record, rendered)


_LOG_BUFFER: RuntimeLogBuffer | None = None
_LOG_HANDLER: RuntimeLogBufferHandler | None = None
_SERVER_THREAD: threading.Thread | None = None
_SERVER: uvicorn.Server | None = None
_SERVER_LOCK = threading.Lock()


def configure_runtime_log_streaming(max_entries: int = 2000, debug_enabled: bool = False) -> RuntimeLogBuffer:
    global _LOG_BUFFER, _LOG_HANDLER
    if _LOG_BUFFER is not None and _LOG_HANDLER is not None:
        return _LOG_BUFFER

    buffer = RuntimeLogBuffer(max_entries=max_entries)
    handler = RuntimeLogBufferHandler(buffer)
    handler.setLevel(logging.DEBUG if debug_enabled else logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    _LOG_BUFFER = buffer
    _LOG_HANDLER = handler
    return buffer


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value.__dict__)


def _safe_json(value: Optional[str], fallback: Any) -> Any:
    if value is None or not str(value).strip():
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


def _message_content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            text_value = item.get("text") or item.get("content")
            if isinstance(text_value, str):
                parts.append(text_value)
            elif str(item.get("type") or "").lower() in {"image", "image_url", "input_image"}:
                parts.append("[image]")
            else:
                parts.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False, indent=2)
    return str(content)


def _normalize_messages(messages: Any) -> dict[str, Any]:
    if not isinstance(messages, list):
        return {"request": None, "context_messages": [], "malformed": True}
    normalized: list[dict[str, str]] = []
    for message in messages:
        if isinstance(message, dict):
            normalized.append(
                {
                    "role": str(message.get("role") or "unknown"),
                    "content": _message_content_text(message.get("content")),
                }
            )
        else:
            normalized.append({"role": "unknown", "content": _message_content_text(message)})
    request_index = next(
        (index for index in range(len(normalized) - 1, -1, -1) if normalized[index]["role"] == "user"),
        len(normalized) - 1,
    )
    if request_index < 0:
        return {"request": None, "context_messages": [], "malformed": False}
    return {
        "request": normalized[request_index],
        "context_messages": [message for index, message in enumerate(normalized) if index != request_index],
        "malformed": False,
    }


def _interaction_input(
    row: Any,
    *,
    capture_store: Any = None,
    reveal_protected: bool = False,
) -> dict[str, Any]:
    try:
        payload = json.loads(row.messages_json)
    except (json.JSONDecodeError, TypeError):
        return {
            "protected": False,
            "request": None,
            "context_messages": [],
            "malformed": True,
        }
    protected_ref = payload.get("protected_payload_ref") if isinstance(payload, dict) else None
    if protected_ref:
        if not reveal_protected:
            return {
                "protected": True,
                "request": None,
                "context_messages": [],
                "malformed": False,
            }
        if capture_store is None:
            raise HTTPException(status_code=503, detail="protected_input_unavailable")
        try:
            raw, metadata = capture_store.read_bytes(str(protected_ref))
            if str(metadata.get("kind") or "") != "llm_messages":
                raise ValueError("capture is not an LLM message payload")
            payload = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=404, detail="protected_input_not_found") from exc
    normalized = _normalize_messages(payload)
    return {"protected": bool(protected_ref), **normalized}


def _serialize_interaction(row: Any) -> dict[str, Any]:
    return {
        "interaction_id": row.interaction_id,
        "interaction_run_id": row.interaction_run_id,
        "created_at": row.created_at,
        "completed_at": row.completed_at,
        "source": row.source,
        "model": row.model,
        "duration_ms": row.duration_ms,
        "input": _interaction_input(row),
        "response_text": row.response_text,
        "error_text": row.error_text,
        "reasoning_text": row.reasoning_text,
        "tools": _safe_json(row.tools_json, None),
        "tool_calls": _safe_json(row.tool_calls_json, None),
        "metadata": _safe_json(row.metadata_json, {}),
        "report": _safe_json(row.report_json, None),
        "has_image": bool(row.image_path),
        "image_url": f"/api/interactions/{row.interaction_id}/image" if row.image_path else None,
    }


def _resolve_media_path(path_value: str, media_roots: list[Path]) -> Path:
    candidate = Path(path_value).expanduser()
    if not candidate.is_absolute():
        candidate = candidate.resolve()
    else:
        candidate = candidate.resolve(strict=False)
    normalized_roots = [root.resolve(strict=False) for root in media_roots]
    for root in normalized_roots:
        try:
            candidate.relative_to(root)
            if candidate.exists() and candidate.is_file():
                return candidate
        except ValueError:
            continue
    raise HTTPException(status_code=404, detail="media_not_found")


def _load_dashboard_html() -> str:
    try:
        return UI_INDEX_PATH.read_text(encoding="utf-8")
    except OSError:
        return (
            "<!doctype html><html><head><title>Ambient Agent Dashboard</title></head>"
            "<body><h1>Ambient Agent Dashboard</h1><p>Runtime UI assets are missing.</p></body></html>"
        )


def create_runtime_log_app(
    log_buffer: RuntimeLogBuffer,
    report_store: SQLiteInteractionLogAdapter | None = None,
    task_store: SQLiteTaskQueueAdapter | None = None,
    benchmark_store: SQLiteBenchmarkAdapter | None = None,
    training_store: SQLiteTrainingDataAdapter | None = None,
    training_service: TrainingDataService | None = None,
    media_roots: Optional[list[str]] = None,
    chat_store: SQLiteChatAdapter | None = None,
    chat_event_broker: ChatEventBroker | None = None,
    autonomy_store: Any = None,
    capture_store: Any = None,
    capture_control: Any = None,
    resource_governor: Any = None,
) -> FastAPI:
    app = FastAPI(title="Ambient Runtime Logs")
    normalized_media_roots = [Path(root) for root in (media_roots or [])]
    if UI_ROOT.exists():
        app.mount("/runtime-ui", StaticFiles(directory=str(UI_ROOT)), name="runtime_ui")

    @app.middleware("http")
    async def enforce_same_origin(request: Request, call_next):
        path = request.url.path
        state_changing = request.method.upper() in {"POST", "PUT", "PATCH", "DELETE"}
        if path.startswith("/api/") and state_changing:
            from urllib.parse import urlsplit

            request_host = (request.headers.get("host") or "").lower()
            if not request_host:
                return JSONResponse(status_code=400, content={"detail": "host header required"})
            origin = request.headers.get("origin")
            if origin and (urlsplit(origin).netloc or "").lower() != request_host:
                return JSONResponse(status_code=401, content={"detail": "origin does not match host"})
        return await call_next(request)

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"status": "ok", "latest_id": log_buffer.latest_id()}

    @app.get("/api/logs")
    def get_logs(
        after_id: int = Query(default=0, ge=0),
        limit: int = Query(default=200, ge=1, le=2000),
    ) -> dict[str, Any]:
        entries = log_buffer.snapshot(after_id=after_id, limit=limit)
        latest_id = log_buffer.latest_id()
        return {
            "entries": entries,
            "latest_id": latest_id,
            "count": len(entries),
        }

    @app.get("/api/reports")
    def get_reports(limit: int = Query(default=50, ge=1, le=200)) -> dict[str, Any]:
        rows = report_store.list_recent_reports(limit=limit) if report_store is not None else []
        reports: list[dict[str, Any]] = []
        for row in rows:
            try:
                report = json.loads(row.report_json or "{}")
            except json.JSONDecodeError:
                continue
            reports.append(
                {
                    "interaction_id": row.interaction_id,
                    "interaction_run_id": row.interaction_run_id,
                    "created_at": row.created_at,
                    "completed_at": row.completed_at,
                    "source": row.source,
                    "model": row.model,
                    "report": report,
                }
            )
        queued_tasks = []
        if task_store is not None:
            for task in task_store.get_all_pending_tasks():
                metadata = {}
                if getattr(task, "metadata_json", None):
                    try:
                        metadata = json.loads(task.metadata_json)
                    except json.JSONDecodeError:
                        metadata = {}
                queued_tasks.append(
                    {
                        "id": task.id,
                        "description": task.description,
                        "priority": task.priority,
                        "created_at": task.created_at,
                        "status": task.status,
                        "metadata": metadata,
                        "run_at_utc": task.run_at_utc,
                    }
                )
        return {"reports": reports, "queued_tasks": queued_tasks, "count": len(reports)}

    @app.get("/api/interactions")
    def get_interactions(
        date_from: date | None = Query(default=None),
        date_to: date | None = Query(default=None),
        sort: Literal["newest", "oldest"] = Query(default="newest"),
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        if report_store is None:
            return {
                "items": [],
                "pagination": {"limit": limit, "offset": offset, "total": 0, "has_more": False},
                "sort": sort,
                "date_from": date_from.isoformat() if date_from else None,
                "date_to": date_to.isoformat() if date_to else None,
            }
        if date_from and date_to and date_from > date_to:
            raise HTTPException(status_code=422, detail="date_from must not be later than date_to")
        from_value = date_from.isoformat() if date_from else None
        to_value = date_to.isoformat() if date_to else None
        rows = report_store.list_entries(
            limit=limit,
            offset=offset,
            date_from=from_value,
            date_to=to_value,
            sort_order="asc" if sort == "oldest" else "desc",
        )
        total = report_store.count_entries(date_from=from_value, date_to=to_value)
        return {
            "items": [_serialize_interaction(row) for row in rows],
            "pagination": {
                "limit": limit,
                "offset": offset,
                "total": total,
                "has_more": offset + len(rows) < total,
            },
            "sort": sort,
            "date_from": from_value,
            "date_to": to_value,
        }

    @app.get("/api/interactions/{interaction_id}/input")
    def get_interaction_input(interaction_id: str) -> dict[str, Any]:
        if report_store is None:
            raise HTTPException(status_code=503, detail="interaction_store_unavailable")
        row = report_store.get_by_interaction_id(interaction_id)
        if row is None:
            raise HTTPException(status_code=404, detail="interaction_not_found")
        input_payload = _interaction_input(row, capture_store=capture_store, reveal_protected=True)
        if input_payload.get("protected") and autonomy_store is not None:
            autonomy_store.audit(
                "local_user",
                "interaction.input_viewed",
                interaction_id,
                {"source": row.source},
            )
        return {"interaction_id": interaction_id, "input": input_payload}

    @app.get("/api/interactions/{interaction_id}/image")
    def get_interaction_image(interaction_id: str) -> Response:
        if report_store is None:
            raise HTTPException(status_code=503, detail="interaction_store_unavailable")
        row = report_store.get_by_interaction_id(interaction_id)
        if row is None:
            raise HTTPException(status_code=404, detail="interaction_not_found")
        image_path = str(row.image_path or "").strip()
        if not image_path:
            raise HTTPException(status_code=404, detail="interaction_image_not_found")
        if image_path.startswith("capture://"):
            if capture_store is None:
                raise HTTPException(status_code=404, detail="interaction_image_not_found")
            try:
                data, metadata = capture_store.read_bytes(image_path)
            except (OSError, ValueError) as exc:
                raise HTTPException(status_code=404, detail="interaction_image_not_found") from exc
            response: Response = Response(
                content=data,
                media_type=str(metadata.get("mime_type") or "application/octet-stream"),
                headers={"Content-Disposition": "inline"},
            )
        else:
            if not normalized_media_roots:
                raise HTTPException(status_code=404, detail="interaction_image_not_found")
            resolved = _resolve_media_path(image_path, normalized_media_roots)
            mime_type, _ = mimetypes.guess_type(str(resolved))
            response = FileResponse(
                path=str(resolved),
                media_type=mime_type or "application/octet-stream",
                headers={"Content-Disposition": "inline"},
            )
        if autonomy_store is not None:
            autonomy_store.audit(
                "local_user",
                "interaction.image_viewed",
                interaction_id,
                {"source": row.source},
            )
        return response

    @app.get("/api/chat/sessions")
    def get_chat_sessions(limit: int = Query(default=100, ge=1, le=500)) -> dict[str, Any]:
        if chat_store is None:
            return {"sessions": [], "count": 0}
        sessions = chat_store.list_sessions(limit=limit)
        return {"sessions": sessions, "count": len(sessions)}

    @app.post("/api/chat/sessions")
    async def create_chat_session(request: Request) -> dict[str, Any]:
        if chat_store is None:
            raise HTTPException(status_code=503, detail="chat_unavailable")
        body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
        return {"session": chat_store.create_session(str(body.get("title") or "New conversation"))}

    @app.patch("/api/chat/sessions/{session_id}")
    async def rename_chat_session(session_id: str, request: Request) -> dict[str, Any]:
        if chat_store is None:
            raise HTTPException(status_code=503, detail="chat_unavailable")
        body = await request.json()
        try:
            session = chat_store.rename_session(session_id, str(body.get("title") or ""))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if session is None:
            raise HTTPException(status_code=404, detail="session_not_found")
        return {"session": session}

    @app.get("/api/chat/sessions/{session_id}/messages")
    def get_chat_messages(
        session_id: str,
        limit: int = Query(default=200, ge=1, le=1000),
    ) -> dict[str, Any]:
        if chat_store is None:
            raise HTTPException(status_code=503, detail="chat_unavailable")
        session = chat_store.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="session_not_found")
        messages = chat_store.list_messages(session_id, limit=limit)
        return {"session": session, "messages": messages, "count": len(messages)}

    @app.post("/api/chat/sessions/{session_id}/messages")
    async def submit_chat_message(session_id: str, request: Request) -> dict[str, Any]:
        if chat_store is None:
            raise HTTPException(status_code=503, detail="chat_unavailable")
        body = await request.json()
        try:
            turn = chat_store.enqueue_turn(session_id, str(body.get("content") or ""))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="session_not_found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if chat_event_broker is not None:
            chat_event_broker.notify_turn_enqueued()
        assistant = turn["assistant_message"]
        return {
            **turn,
            "stream_url": f"/api/chat/messages/{assistant['id']}/events",
        }

    @app.get("/api/chat/messages/{message_id}/events")
    async def stream_chat_message(message_id: str, request: Request):
        if chat_store is None or chat_event_broker is None:
            raise HTTPException(status_code=503, detail="chat_stream_unavailable")
        initial = chat_store.get_message(message_id)
        if initial is None or initial.get("role") != "assistant":
            raise HTTPException(status_code=404, detail="message_not_found")
        subscriber = chat_event_broker.subscribe(message_id)

        async def event_stream():
            try:
                snapshot = chat_store.get_message(message_id)
                yield f"event: snapshot\ndata: {json.dumps(snapshot, ensure_ascii=False)}\n\n"
                if snapshot and snapshot.get("status") in {"completed", "failed"}:
                    terminal_type = "done" if snapshot["status"] == "completed" else "error"
                    yield f"event: {terminal_type}\ndata: {json.dumps(snapshot, ensure_ascii=False)}\n\n"
                    return
                while not await request.is_disconnected():
                    try:
                        event = await asyncio.to_thread(subscriber.get, True, 15.0)
                    except queue.Empty:
                        yield ": heartbeat\n\n"
                        continue
                    if event.get("type") == "snapshot_required":
                        event = {"type": "snapshot", "message": chat_store.get_message(message_id)}
                    event_type = str(event.get("type") or "message")
                    yield f"event: {event_type}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
                    if event_type in {"done", "error"}:
                        return
            finally:
                chat_event_broker.unsubscribe(message_id, subscriber)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/chat/scheduled/{task_id}/cancel")
    def cancel_scheduled_task(task_id: int) -> dict[str, Any]:
        if task_store is None:
            raise HTTPException(status_code=503, detail="task_store_unavailable")
        if not task_store.cancel_task(task_id):
            raise HTTPException(status_code=409, detail="task_not_pending")
        return {"ok": True, "task_id": task_id, "status": "cancelled"}

    @app.get("/api/benchmarks/runs")
    def get_benchmark_runs(
        limit: int = Query(default=50, ge=1, le=200),
        service_name: str | None = Query(default=None),
    ) -> dict[str, Any]:
        rows = benchmark_store.list_runs(limit=limit, service_name=service_name) if benchmark_store is not None else []
        return {"runs": [_as_dict(row) for row in rows], "count": len(rows)}

    @app.get("/api/benchmarks/results")
    def get_benchmark_results(
        limit: int = Query(default=200, ge=1, le=500),
        run_id: str | None = Query(default=None),
        service_name: str | None = Query(default=None),
        model_name: str | None = Query(default=None),
        case_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        rows = (
            benchmark_store.list_results(
                limit=limit,
                run_id=run_id,
                service_name=service_name,
                model_name=model_name,
                case_id=case_id,
            )
            if benchmark_store is not None
            else []
        )
        payload: list[dict[str, Any]] = []
        for row in rows:
            item = _as_dict(row)
            review = benchmark_store.get_manual_review(row.result_id) if benchmark_store is not None else None
            item["manual_review"] = _as_dict(review) if review is not None else None
            payload.append(item)
        return {"results": payload, "count": len(payload)}

    @app.get("/api/benchmarks/results/{result_id}")
    def get_benchmark_result(result_id: str) -> dict[str, Any]:
        if benchmark_store is None:
            return {"result": None}
        row = benchmark_store.get_result(result_id)
        if row is None:
            return {"result": None}
        payload = _as_dict(row)
        review = benchmark_store.get_manual_review(result_id)
        payload["manual_review"] = _as_dict(review) if review is not None else None
        return {"result": payload}

    @app.post("/api/benchmarks/results/{result_id}/review")
    async def upsert_benchmark_review(result_id: str, request: Request) -> dict[str, Any]:
        if benchmark_store is None:
            return {"ok": False, "error": "benchmark_store_unavailable"}
        body = await request.json()
        now = datetime.now().isoformat()
        score = body.get("score")
        review = benchmark_store.upsert_manual_review(
            result_id=result_id,
            reviewer=str(body.get("reviewer") or "local-user"),
            score=float(score) if score not in (None, "") else None,
            notes=str(body.get("notes") or "").strip() or None,
            created_at=now,
            updated_at=now,
        )
        return {"ok": True, "review": _as_dict(review)}

    @app.post("/api/training/sync/llm")
    async def sync_training_llm() -> dict[str, Any]:
        if training_service is None:
            return {"ok": False, "error": "training_service_unavailable"}
        return {"ok": True, **training_service.sync_llm_records()}

    @app.post("/api/training/sync/asr")
    async def sync_training_asr() -> dict[str, Any]:
        if training_service is None:
            return {"ok": False, "error": "training_service_unavailable"}
        return {"ok": True, **training_service.sync_asr_records()}

    @app.get("/api/training/llm")
    def get_training_llm_records(
        limit: int = Query(default=100, ge=1, le=500),
        source: str | None = Query(default=None),
        model: str | None = Query(default=None),
        review_status: str | None = Query(default=None),
    ) -> dict[str, Any]:
        if training_store is None:
            return {"records": [], "count": 0}
        rows = training_store.list_llm_records(limit=limit, source=source, model=model, review_status=review_status)
        payload = []
        for row in rows:
            item = _as_dict(row)
            review = training_store.get_llm_review(row.record_id)
            item["review"] = _as_dict(review) if review is not None else None
            payload.append(item)
        return {"records": payload, "count": len(payload)}

    @app.get("/api/training/llm/{record_id}")
    def get_training_llm_record(record_id: str) -> dict[str, Any]:
        if training_store is None:
            return {"record": None}
        row = training_store.get_llm_record(record_id)
        if row is None:
            return {"record": None}
        payload = _as_dict(row)
        review = training_store.get_llm_review(record_id)
        payload["review"] = _as_dict(review) if review is not None else None
        payload["messages"] = _safe_json(row.messages_json, [])
        payload["tools"] = _safe_json(row.tools_json, None)
        payload["tool_calls"] = _safe_json(row.tool_calls_json, None)
        payload["metadata"] = _safe_json(row.metadata_json, {})
        payload["report"] = _safe_json(row.report_json, None)
        return {"record": payload}

    @app.post("/api/training/llm/{record_id}/review")
    async def upsert_training_llm_review(record_id: str, request: Request) -> dict[str, Any]:
        if training_store is None:
            return {"ok": False, "error": "training_store_unavailable"}
        body = await request.json()
        now = datetime.now().isoformat()
        review = training_store.upsert_llm_review(
            record_id=record_id,
            reviewer=str(body.get("reviewer") or "local-user"),
            status=str(body.get("status") or "pending"),
            corrected_response_text=str(body.get("corrected_response_text") or "").strip() or None,
            corrected_reasoning_text=str(body.get("corrected_reasoning_text") or "").strip() or None,
            corrected_messages_json=str(body.get("corrected_messages_json") or "").strip() or None,
            notes=str(body.get("notes") or "").strip() or None,
            created_at=now,
            updated_at=now,
        )
        return {"ok": True, "review": _as_dict(review)}

    @app.get("/api/training/asr")
    def get_training_asr_records(
        limit: int = Query(default=100, ge=1, le=500),
        review_status: str | None = Query(default=None),
    ) -> dict[str, Any]:
        if training_store is None:
            return {"records": [], "count": 0}
        rows = training_store.list_asr_records(limit=limit, review_status=review_status)
        payload = []
        for row in rows:
            item = _as_dict(row)
            review = training_store.get_asr_review(row.record_id)
            item["review"] = _as_dict(review) if review is not None else None
            payload.append(item)
        return {"records": payload, "count": len(payload)}

    @app.get("/api/training/asr/{record_id}")
    def get_training_asr_record(record_id: str) -> dict[str, Any]:
        if training_store is None:
            return {"record": None}
        row = training_store.get_asr_record(record_id)
        if row is None:
            return {"record": None}
        payload = _as_dict(row)
        review = training_store.get_asr_review(record_id)
        payload["review"] = _as_dict(review) if review is not None else None
        payload["metadata"] = _safe_json(row.metadata_json, {})
        return {"record": payload}

    @app.post("/api/training/asr/{record_id}/review")
    async def upsert_training_asr_review(record_id: str, request: Request) -> dict[str, Any]:
        if training_store is None:
            return {"ok": False, "error": "training_store_unavailable"}
        body = await request.json()
        now = datetime.now().isoformat()
        review = training_store.upsert_asr_review(
            record_id=record_id,
            reviewer=str(body.get("reviewer") or "local-user"),
            status=str(body.get("status") or "pending"),
            corrected_transcript_text=str(body.get("corrected_transcript_text") or "").strip() or None,
            notes=str(body.get("notes") or "").strip() or None,
            created_at=now,
            updated_at=now,
        )
        return {"ok": True, "review": _as_dict(review)}

    @app.post("/api/training/export/llm")
    async def export_training_llm(request: Request) -> dict[str, Any]:
        if training_service is None:
            return {"ok": False, "error": "training_service_unavailable"}
        body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
        return {"ok": True, **training_service.export_llm_dataset(statuses=body.get("statuses"))}

    @app.post("/api/training/export/asr")
    async def export_training_asr(request: Request) -> dict[str, Any]:
        if training_service is None:
            return {"ok": False, "error": "training_service_unavailable"}
        body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
        return {"ok": True, **training_service.export_asr_dataset(statuses=body.get("statuses"))}

    @app.get("/api/training/exports")
    def get_training_exports(
        dataset_kind: str | None = Query(default=None),
        limit: int = Query(default=50, ge=1, le=200),
    ) -> dict[str, Any]:
        if training_store is None:
            return {"exports": [], "count": 0}
        rows = training_store.list_exports(dataset_kind=dataset_kind, limit=limit)
        return {"exports": [_as_dict(row) for row in rows], "count": len(rows)}

    @app.get("/api/training/media")
    def get_training_media(path: str = Query(..., min_length=1)) -> FileResponse:
        if not normalized_media_roots:
            raise HTTPException(status_code=404, detail="media_unavailable")
        resolved = _resolve_media_path(path, normalized_media_roots)
        mime_type, _ = mimetypes.guess_type(str(resolved))
        return FileResponse(path=str(resolved), media_type=mime_type or "application/octet-stream")

    @app.get("/api/autonomy/opportunities")
    def list_opportunities(
        limit: int = Query(default=50, ge=1, le=500),
        status: str | None = Query(default=None),
    ) -> dict[str, Any]:
        rows = autonomy_store.list_opportunities(limit=limit, status=status) if autonomy_store is not None else []
        return {"opportunities": [_as_dict(row) for row in rows], "count": len(rows)}

    @app.get("/api/autonomy/inbox")
    def list_proactive_inbox(
        limit: int = Query(default=50, ge=1, le=500),
        status: str | None = Query(default=None),
    ) -> dict[str, Any]:
        rows = autonomy_store.list_inbox_items(limit=limit, status=status) if autonomy_store is not None else []
        return {"items": [_as_dict(row) for row in rows], "count": len(rows)}

    @app.post("/api/autonomy/inbox/{inbox_id}/feedback")
    async def proactive_inbox_feedback(inbox_id: str, request: Request) -> dict[str, Any]:
        if autonomy_store is None:
            raise HTTPException(status_code=503, detail="autonomy_store_unavailable")
        body = await request.json()
        try:
            updated = autonomy_store.record_feedback(inbox_id, str(body.get("feedback") or ""))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": updated}

    @app.get("/api/autonomy/policies")
    def list_capability_policies() -> dict[str, Any]:
        rows = autonomy_store.list_policies() if autonomy_store is not None else []
        return {"policies": rows, "count": len(rows)}

    @app.put("/api/autonomy/policies/{capability}")
    async def update_capability_policy(capability: str, request: Request) -> dict[str, Any]:
        if autonomy_store is None:
            raise HTTPException(status_code=503, detail="autonomy_store_unavailable")
        body = await request.json()
        try:
            policy = autonomy_store.set_policy(
                capability,
                str(body.get("decision") or ""),
                body.get("constraints") if isinstance(body.get("constraints"), dict) else None,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "policy": policy}

    @app.get("/api/autonomy/calibration/{capability}")
    def get_calibration(capability: str) -> dict[str, Any]:
        if autonomy_store is None:
            raise HTTPException(status_code=503, detail="autonomy_store_unavailable")
        return autonomy_store.calibration_stats(capability)

    @app.post("/api/autonomy/calibration/{capability}/outcomes")
    async def add_calibration_outcome(capability: str, request: Request) -> dict[str, Any]:
        if autonomy_store is None:
            raise HTTPException(status_code=503, detail="autonomy_store_unavailable")
        body = await request.json()
        autonomy_store.record_calibration_outcome(
            capability,
            correct=bool(body.get("correct")),
            source_ref=str(body.get("source_ref") or "") or None,
        )
        return {"ok": True, **autonomy_store.calibration_stats(capability)}

    @app.get("/api/autonomy/approvals")
    def list_autonomy_approvals(
        status: str | None = Query(default=None),
        limit: int = Query(default=100, ge=1, le=500),
    ) -> dict[str, Any]:
        rows = autonomy_store.list_approvals(status=status, limit=limit) if autonomy_store is not None else []
        return {"approvals": [_as_dict(row) for row in rows], "count": len(rows)}

    @app.post("/api/autonomy/approvals/{approval_id}/decision")
    async def decide_autonomy_approval(approval_id: str, request: Request) -> dict[str, Any]:
        if autonomy_store is None:
            raise HTTPException(status_code=503, detail="autonomy_store_unavailable")
        body = await request.json()
        approved = bool(body.get("approved"))
        changed = autonomy_store.decide_approval(
            approval_id, approved=approved, approver="local_user"
        )
        if changed and approved:
            approval = next(
                (item for item in autonomy_store.list_approvals(limit=500) if item.approval_id == approval_id),
                None,
            )
            if approval is not None:
                details = _safe_json(approval.constraints_json, {})
                now = datetime.now().astimezone().isoformat()
                autonomy_store.enqueue_event(
                    AmbientEvent(
                        event_id=uuid.uuid4().hex,
                        event_type="approval_granted",
                        source_kind="local_approval",
                        source_ref=approval_id,
                        occurred_at=now,
                        payload_json=json.dumps(details, ensure_ascii=False),
                        confidence=1.0,
                        privacy_label="private",
                        fingerprint=hashlib.sha256(f"approval|{approval_id}".encode("utf-8")).hexdigest(),
                        priority=1.0,
                        available_at=now,
                    )
                )
        return {"ok": changed, "approved": approved}

    @app.get("/api/privacy/status")
    def privacy_status() -> dict[str, Any]:
        storage = (
            capture_store.storage_status()
            if capture_store is not None and hasattr(capture_store, "storage_status")
            else {"capture_size_bytes": capture_store.size_bytes() if capture_store is not None else 0}
        )
        return {
            "capture": capture_control.status() if capture_control is not None else {"paused": False},
            "raw_retention": (
                "indefinite_plain"
            ),
            **storage,
        }

    @app.get("/api/runtime/resources")
    def runtime_resources() -> dict[str, Any]:
        if resource_governor is None:
            raise HTTPException(status_code=503, detail="resource_governor_unavailable")
        payload = resource_governor.status()
        if autonomy_store is not None and hasattr(autonomy_store, "event_counts"):
            payload["event_counts"] = autonomy_store.event_counts()
        return payload

    @app.put("/api/runtime/resource-policy")
    async def update_runtime_resource_policy(request: Request) -> dict[str, Any]:
        if resource_governor is None:
            raise HTTPException(status_code=503, detail="resource_governor_unavailable")
        body = await request.json()
        try:
            preset = resource_governor.set_preset(str(body.get("preset") or ""))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "preset": preset, **resource_governor.status()}

    @app.post("/api/privacy/capture/{action}")
    def privacy_capture_action(action: str, request: Request) -> dict[str, Any]:
        if capture_control is None:
            raise HTTPException(status_code=503, detail="capture_control_unavailable")
        if action == "pause":
            capture_control.pause()
        elif action == "resume":
            capture_control.resume()
        else:
            raise HTTPException(status_code=400, detail="action_must_be_pause_or_resume")
        if autonomy_store is not None:
            autonomy_store.audit("local_user", f"capture.{action}", "global", {})
        return {"ok": True, **capture_control.status()}

    @app.put("/api/privacy/capture/exclusions")
    async def privacy_capture_exclusions(request: Request) -> dict[str, Any]:
        if capture_control is None:
            raise HTTPException(status_code=503, detail="capture_control_unavailable")
        body = await request.json()
        apps = body.get("apps")
        domains = body.get("domains")
        if apps is not None and not isinstance(apps, list):
            raise HTTPException(status_code=400, detail="apps_must_be_a_list")
        if domains is not None and not isinstance(domains, list):
            raise HTTPException(status_code=400, detail="domains_must_be_a_list")
        capture_control.set_exclusions(apps=apps, domains=domains)
        if autonomy_store is not None:
            autonomy_store.audit(
                "local_user",
                "capture.exclusions_updated",
                "global",
                capture_control.status(),
            )
        return {"ok": True, **capture_control.status()}

    @app.get("/api/privacy/captures")
    def read_capture_file(request: Request, uri: str = Query(..., min_length=10)) -> Response:
        if capture_store is None:
            raise HTTPException(status_code=503, detail="capture_store_unavailable")
        try:
            data, metadata = capture_store.read_bytes(uri)
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="capture_not_found") from exc
        if autonomy_store is not None:
            autonomy_store.audit(
                "local_user", "capture.viewed", uri, {"kind": metadata.get("kind")}
            )
        headers = {"Content-Disposition": f'attachment; filename="{metadata.get("original_name", "ambient.bin")}"'}
        return Response(content=data, media_type=metadata.get("mime_type") or "application/octet-stream", headers=headers)

    @app.get("/api/privacy/captures/export")
    def export_capture_file(request: Request, uri: str = Query(..., min_length=10)) -> Response:
        if capture_store is None:
            raise HTTPException(status_code=503, detail="capture_store_unavailable")
        try:
            data, filename = capture_store.read_stored_file(uri)
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="capture_not_found") from exc
        if autonomy_store is not None:
            autonomy_store.audit(
                "local_user", "capture.exported", uri, {}
            )
        return Response(
            content=data,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.delete("/api/privacy/captures")
    def delete_capture_file(request: Request, uri: str = Query(..., min_length=10)) -> dict[str, Any]:
        if capture_store is None:
            raise HTTPException(status_code=503, detail="capture_store_unavailable")
        deleted = capture_store.delete(uri)
        if autonomy_store is not None:
            autonomy_store.audit("local_user", "capture.deleted", uri, {"deleted": deleted})
        return {"ok": deleted}

    @app.get("/reports", response_class=HTMLResponse)
    @app.get("/inbox", response_class=HTMLResponse)
    @app.get("/chat", response_class=HTMLResponse)
    @app.get("/interactions", response_class=HTMLResponse)
    @app.get("/", response_class=HTMLResponse)
    @app.get("/logs", response_class=HTMLResponse)
    @app.get("/benchmarks", response_class=HTMLResponse)
    @app.get("/training", response_class=HTMLResponse)
    def view_logs() -> str:
        return _load_dashboard_html()

    return app


def start_runtime_log_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    max_entries: int = 2000,
    report_store: SQLiteInteractionLogAdapter | None = None,
    task_store: SQLiteTaskQueueAdapter | None = None,
    benchmark_store: SQLiteBenchmarkAdapter | None = None,
    training_store: SQLiteTrainingDataAdapter | None = None,
    training_service: TrainingDataService | None = None,
    media_roots: Optional[list[str]] = None,
    chat_store: SQLiteChatAdapter | None = None,
    chat_event_broker: ChatEventBroker | None = None,
    autonomy_store: Any = None,
    capture_store: Any = None,
    capture_control: Any = None,
    resource_governor: Any = None,
) -> RuntimeLogBuffer:
    global _SERVER_THREAD, _SERVER
    log_buffer = configure_runtime_log_streaming(max_entries=max_entries)
    with _SERVER_LOCK:
        if _SERVER_THREAD is not None and _SERVER_THREAD.is_alive():
            return log_buffer

        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise RuntimeError("The unauthenticated runtime API may only bind to loopback.")

        app = create_runtime_log_app(
            log_buffer,
            report_store=report_store,
            task_store=task_store,
            benchmark_store=benchmark_store,
            training_store=training_store,
            training_service=training_service,
            media_roots=media_roots,
            chat_store=chat_store,
            chat_event_broker=chat_event_broker,
            autonomy_store=autonomy_store,
            capture_store=capture_store,
            capture_control=capture_control,
            resource_governor=resource_governor,
        )

        def _serve() -> None:
            global _SERVER
            config = uvicorn.Config(
                app=app,
                host=host,
                port=port,
                log_level="warning",
                access_log=False,
            )
            server = uvicorn.Server(config)
            _SERVER = server
            server.run()
            _SERVER = None

        thread = threading.Thread(
            target=_serve,
            name="RuntimeLogServerThread",
        )
        thread.start()
        _SERVER_THREAD = thread
    return log_buffer


def shutdown_runtime_log_server(*, join_timeout: float = 5.0, remove_log_handler: bool = False) -> None:
    global _SERVER_THREAD, _SERVER, _LOG_BUFFER, _LOG_HANDLER
    with _SERVER_LOCK:
        server = _SERVER
        thread = _SERVER_THREAD
        if server is not None:
            server.should_exit = True
        if thread is not None:
            thread.join(timeout=join_timeout)
        _SERVER_THREAD = None
        _SERVER = None
    if remove_log_handler and _LOG_HANDLER is not None:
        root_logger = logging.getLogger()
        root_logger.removeHandler(_LOG_HANDLER)
        _LOG_HANDLER.close()
        _LOG_HANDLER = None
        _LOG_BUFFER = None
