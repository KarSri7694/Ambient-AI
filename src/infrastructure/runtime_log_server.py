import asyncio
import hashlib
import json
import logging
import mimetypes
import queue
import threading
import uuid
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import date, datetime, timezone
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
_SERVER_SHUTDOWN_EVENT = threading.Event()


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


_RAG_CONTEXT_KEYS = {
    "personalization_context",
    "temporal_context",
    "temporal_work_context",
    "recent_context",
    "relevant_user_context",
    "relevant_user_memory",
    "user_context",
}


def _json_object_from_content(content: Any) -> dict[str, Any] | None:
    """Best-effort extraction of the JSON payload actually sent to the model."""
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        return None
    text = content.strip()
    if not text:
        return None
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _extract_injected_rag_context(messages: Any) -> list[dict[str, Any]]:
    """Return retrieval/personalization blocks that were truly present in a prompt.

    This deliberately reads the saved request instead of reconstructing RAG from
    current memory.  The audit trail therefore remains accurate after memory
    changes, reranking changes, or a runtime restart.
    """
    if not isinstance(messages, list):
        return []
    injected: list[dict[str, Any]] = []

    def visit(value: Any, *, message_index: int, path: str = "") -> None:
        if not isinstance(value, dict):
            return
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if key in _RAG_CONTEXT_KEYS and child not in (None, "", [], {}):
                injected.append(
                    {
                        "message_index": message_index,
                        "field": child_path,
                        "value": child,
                    }
                )
            if isinstance(child, dict):
                visit(child, message_index=message_index, path=child_path)

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        parsed = _json_object_from_content(message.get("content"))
        if parsed is not None:
            visit(parsed, message_index=index)
    return injected


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
            "rag_context": [],
            "malformed": True,
        }
    protected_ref = payload.get("protected_payload_ref") if isinstance(payload, dict) else None
    if protected_ref:
        if not reveal_protected:
            return {
                "protected": True,
                "request": None,
                "context_messages": [],
                "rag_context": [],
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
    messages = payload if isinstance(payload, list) else []
    return {
        "protected": bool(protected_ref),
        **normalized,
        "rag_context": _extract_injected_rag_context(messages),
    }


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
    runtime_control: Any = None,
    real_world_lab: Any = None,
    rocm_tuning_service: Any = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            if real_world_lab is not None:
                real_world_lab.shutdown()

    app = FastAPI(title="Ambient Runtime Logs", lifespan=lifespan)
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
        return {"status": "ok", "latest_id": log_buffer.latest_id(), "real_world_lab": real_world_lab is not None}

    @app.get("/api/home")
    def home_dashboard(
        date_value: date | None = Query(default=None, alias="date"),
        since: str | None = Query(default=None),
    ) -> dict[str, Any]:
        if runtime_control is None or not hasattr(runtime_control, "home_snapshot"):
            raise HTTPException(status_code=503, detail="daily_briefing_unavailable")
        try:
            return runtime_control.home_snapshot(
                date_value=date_value.isoformat() if date_value else None,
                since=since,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid_date_or_since") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/real-world/suites")
    def get_real_world_suites() -> dict[str, Any]:
        if real_world_lab is None:
            return {"suites": [], "available": False}
        suites = real_world_lab.suites()
        return {"suites": suites, "available": True}

    @app.get("/api/rocm-tuning/status")
    def get_rocm_tuning_status() -> dict[str, Any]:
        if rocm_tuning_service is None:
            return {"available": False, "accelerator": None, "latest_profile": None, "recent_runs": []}
        return rocm_tuning_service.status()

    @app.get("/api/rocm-tuning/runs")
    def get_rocm_tuning_runs(limit: int = Query(default=20, ge=1, le=200)) -> dict[str, Any]:
        if rocm_tuning_service is None:
            return {"available": False, "runs": []}
        return {"available": True, "runs": rocm_tuning_service.store.list_runs(limit=limit)}

    @app.get("/api/rocm-tuning/runs/{run_id}")
    def get_rocm_tuning_run(run_id: str) -> dict[str, Any]:
        if rocm_tuning_service is None:
            raise HTTPException(status_code=503, detail="rocm_tuning_unavailable")
        run = rocm_tuning_service.store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="rocm_tuning_run_not_found")
        return {"run": run}

    @app.post("/api/rocm-tuning/runs")
    def start_rocm_tuning_run() -> dict[str, Any]:
        if rocm_tuning_service is None:
            raise HTTPException(status_code=503, detail="rocm_tuning_unavailable")
        try:
            return {"run": rocm_tuning_service.run()}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/api/rocm-tuning/latest")
    def get_rocm_tuning_latest() -> dict[str, Any]:
        if rocm_tuning_service is None:
            return {"profile": None}
        return {"profile": rocm_tuning_service.store.latest_profile()}

    @app.get("/api/rocm-tuning/export.json")
    def export_rocm_tuning_json() -> dict[str, Any]:
        if rocm_tuning_service is None:
            raise HTTPException(status_code=503, detail="rocm_tuning_unavailable")
        return rocm_tuning_service.store.export_json()

    @app.get("/api/rocm-tuning/export.csv")
    def export_rocm_tuning_csv() -> Response:
        if rocm_tuning_service is None:
            raise HTTPException(status_code=503, detail="rocm_tuning_unavailable")
        return Response(
            content=rocm_tuning_service.store.export_csv(),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="rocm-tuning.csv"'},
        )

    @app.get("/api/real-world/models")
    def get_real_world_models() -> dict[str, Any]:
        if real_world_lab is None:
            raise HTTPException(status_code=503, detail="real_world_lab_unavailable")
        return real_world_lab.models()

    @app.get("/api/real-world/runs")
    def get_real_world_runs(limit: int = Query(default=50, ge=1, le=200)) -> dict[str, Any]:
        if real_world_lab is None:
            return {"runs": [], "available": False}
        rows = real_world_lab.store.list_runs(limit=limit)
        return {"runs": rows, "available": True, "count": len(rows)}

    @app.get("/api/real-world/runs/{run_id}")
    def get_real_world_run(run_id: str) -> dict[str, Any]:
        if real_world_lab is None:
            raise HTTPException(status_code=503, detail="real_world_lab_unavailable")
        row = real_world_lab.store.get_run(run_id)
        if row is None:
            raise HTTPException(status_code=404, detail="real_world_run_not_found")
        return {"run": row}

    @app.get("/api/real-world/runs/{run_id}/trace")
    def get_real_world_trace(run_id: str, after_sequence: int = Query(default=0, ge=0),
                             limit: int = Query(default=1000, ge=1, le=5000)) -> dict[str, Any]:
        if real_world_lab is None:
            raise HTTPException(status_code=503, detail="real_world_lab_unavailable")
        if real_world_lab.store.get_run(run_id) is None:
            raise HTTPException(status_code=404, detail="real_world_run_not_found")
        events = real_world_lab.store.list_events(run_id, after_sequence=after_sequence, limit=limit)
        return {"events": events, "count": len(events)}

    @app.get("/api/real-world/runs/{run_id}/export.json")
    def export_real_world_run_json(run_id: str) -> dict[str, Any]:
        if real_world_lab is None:
            raise HTTPException(status_code=503, detail="real_world_lab_unavailable")
        try:
            return real_world_lab.export_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="real_world_run_not_found") from exc

    @app.get("/api/real-world/runs/{run_id}/export.csv")
    def export_real_world_run_csv(run_id: str) -> Response:
        if real_world_lab is None:
            raise HTTPException(status_code=503, detail="real_world_lab_unavailable")
        try:
            csv_text = real_world_lab.export_run_csv(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="real_world_run_not_found") from exc
        return Response(
            content=csv_text,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="real-world-{run_id}.csv"'},
        )

    @app.get("/api/real-world/runs/{run_id}/events")
    async def stream_real_world_events(run_id: str, request: Request):
        if real_world_lab is None or real_world_lab.store.get_run(run_id) is None:
            raise HTTPException(status_code=404, detail="real_world_run_not_found")

        async def event_stream():
            sequence = 0
            while not await request.is_disconnected():
                events = real_world_lab.store.list_events(run_id, after_sequence=sequence, limit=500)
                for event in events:
                    sequence = max(sequence, int(event["sequence"]))
                    yield f"event: trace\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
                run = real_world_lab.store.get_run(run_id)
                if run and run["status"] in {"completed", "completed_with_errors", "failed", "cancelled", "interrupted"}:
                    yield f"event: done\ndata: {json.dumps({'status': run['status']})}\n\n"
                    return
                if not events:
                    yield ": heartbeat\n\n"
                await asyncio.sleep(0.5)
        return StreamingResponse(event_stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/real-world/runs")
    async def start_real_world_run(request: Request) -> dict[str, Any]:
        if real_world_lab is None:
            raise HTTPException(status_code=503, detail="real_world_lab_unavailable")
        try:
            run = real_world_lab.start_run(await request.json())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"run": run}

    @app.post("/api/real-world/runs/{run_id}/cancel")
    def cancel_real_world_run(run_id: str) -> dict[str, Any]:
        if real_world_lab is None:
            raise HTTPException(status_code=503, detail="real_world_lab_unavailable")
        if not real_world_lab.cancel_run(run_id):
            raise HTTPException(status_code=409, detail="run_is_not_active")
        return {"ok": True, "run_id": run_id}

    @app.post("/api/real-world/uploads")
    async def upload_real_world_media(request: Request, filename: str = Query(...),
                                      kind: Literal["image", "audio"] = Query(...)) -> dict[str, Any]:
        if real_world_lab is None:
            raise HTTPException(status_code=503, detail="real_world_lab_unavailable")
        try:
            media = real_world_lab.upload_media(filename=filename, kind=kind, data=await request.body())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"media": media}

    @app.get("/api/real-world/media/{media_id}")
    def get_real_world_media(media_id: str):
        if real_world_lab is None:
            raise HTTPException(status_code=503, detail="real_world_lab_unavailable")
        path = real_world_lab.media_path(media_id)
        if path is None:
            raise HTTPException(status_code=404, detail="media_not_found")
        mime_type, _ = mimetypes.guess_type(str(path))
        return FileResponse(str(path), media_type=mime_type or "application/octet-stream")

    @app.get("/api/real-world/runs/{run_id}/results/{result_id}/media/{index}")
    def get_real_world_result_media(run_id: str, result_id: str, index: int):
        if real_world_lab is None:
            raise HTTPException(status_code=503, detail="real_world_lab_unavailable")
        path = real_world_lab.scenario_media_path(run_id, result_id, index)
        if path is None:
            raise HTTPException(status_code=404, detail="media_not_found")
        mime_type, _ = mimetypes.guess_type(str(path))
        return FileResponse(str(path), media_type=mime_type or "application/octet-stream")

    @app.post("/api/real-world/results/{result_id}/review")
    async def review_real_world_result(result_id: str, request: Request) -> dict[str, Any]:
        if real_world_lab is None:
            raise HTTPException(status_code=503, detail="real_world_lab_unavailable")
        try:
            review = real_world_lab.store.upsert_review(result_id, await request.json())
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="real_world_result_not_found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"review": review}

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
                while (
                    not _SERVER_SHUTDOWN_EVENT.is_set()
                    and not await request.is_disconnected()
                ):
                    event = None
                    for _ in range(30):
                        try:
                            event = subscriber.get_nowait()
                            break
                        except queue.Empty:
                            if (
                                _SERVER_SHUTDOWN_EVENT.is_set()
                                or await request.is_disconnected()
                            ):
                                return
                            await asyncio.sleep(0.5)
                    if event is None:
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
        items = []
        for row in rows:
            item = _as_dict(row)
            if hasattr(autonomy_store, "list_feedback_for_inbox"):
                item["feedback_history"] = autonomy_store.list_feedback_for_inbox(row.inbox_id)
            items.append(item)
        return {"items": items, "count": len(items)}

    @app.get("/api/recurring-tasks")
    def list_recurring_tasks(
        status: str | None = Query(default=None),
        limit: int = Query(default=100, ge=1, le=500),
    ) -> dict[str, Any]:
        if autonomy_store is None or not hasattr(autonomy_store, "list_recurring_tasks"):
            return {"tasks": [], "count": 0, "available": False}
        rows = autonomy_store.list_recurring_tasks(status=status, limit=limit)
        payload = []
        for row in rows:
            item = _as_dict(row)
            item["monitor_state"] = autonomy_store.get_recurring_monitor_state(row.task_id)
            item["source_scope"] = _safe_json(row.source_scope_json, {})
            item["safe_actions"] = _safe_json(row.safe_actions_json, [])
            item["last_result"] = _safe_json(row.last_result_json, {})
            payload.append(item)
        return {"tasks": payload, "count": len(payload), "available": True}

    @app.post("/api/recurring-tasks/{task_id}/{action}")
    def recurring_task_action(task_id: str, action: str) -> dict[str, Any]:
        if autonomy_store is None or not hasattr(autonomy_store, "update_recurring_task_status"):
            raise HTTPException(status_code=503, detail="recurring_tasks_unavailable")
        if action == "run":
            task = autonomy_store.run_recurring_task_now(task_id) if hasattr(autonomy_store, "run_recurring_task_now") else None
            if task is None:
                raise HTTPException(status_code=404, detail="recurring_task_not_runnable")
            return {"ok": True, "task": _as_dict(task)}
        mapping = {"pause": "paused", "resume": "active", "cancel": "cancelled"}
        if action not in mapping:
            raise HTTPException(status_code=400, detail="unsupported_action")
        task = autonomy_store.update_recurring_task_status(task_id, mapping[action])
        if task is None:
            raise HTTPException(status_code=404, detail="recurring_task_not_found")
        return {"ok": True, "task": _as_dict(task)}

    @app.post("/api/autonomy/inbox/{inbox_id}/feedback")
    async def proactive_inbox_feedback(inbox_id: str, request: Request) -> dict[str, Any]:
        if autonomy_store is None:
            raise HTTPException(status_code=503, detail="autonomy_store_unavailable")
        body = await request.json()
        try:
            updated = autonomy_store.record_feedback(inbox_id, str(body.get("feedback") or ""))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        effect = {
            "useful": "Similar proactive help will be prioritized when relevant.",
            "not_useful": "Similar suggestions will be reduced unless they have stronger value.",
            "wrong_inference": "Similar claims will require stronger direct evidence.",
            "too_intrusive": "Similar proactive work will prefer silent reporting or wait for a request.",
        }.get(str(body.get("feedback") or ""), "")
        return {"ok": updated, "effect": effect}

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

    @app.get("/api/autonomy/delegations")
    def list_autonomy_delegations(
        limit: int = Query(default=100, ge=1, le=500),
    ) -> dict[str, Any]:
        rows = (
            autonomy_store.list_delegated_tasks(limit=limit)
            if autonomy_store is not None and hasattr(autonomy_store, "list_delegated_tasks")
            else []
        )
        return {"delegations": [_as_dict(row) for row in rows], "count": len(rows)}

    @app.get("/api/autonomy/delegations/{delegation_id}")
    def get_autonomy_delegation(delegation_id: str) -> dict[str, Any]:
        row = (
            autonomy_store.get_delegated_task(delegation_id)
            if autonomy_store is not None and hasattr(autonomy_store, "get_delegated_task")
            else None
        )
        if row is None:
            raise HTTPException(status_code=404, detail="delegation_not_found")
        return {"delegation": _as_dict(row)}

    @app.post("/api/autonomy/approvals/{approval_id}/decision")
    async def decide_autonomy_approval(approval_id: str, request: Request) -> dict[str, Any]:
        if autonomy_store is None:
            raise HTTPException(status_code=503, detail="autonomy_store_unavailable")
        body = await request.json()
        approved = bool(body.get("approved"))
        existing = (
            autonomy_store.get_approval(approval_id)
            if hasattr(autonomy_store, "get_approval")
            else next(
                (item for item in autonomy_store.list_approvals(limit=500) if item.approval_id == approval_id),
                None,
            )
        )
        if existing is None:
            raise HTTPException(status_code=404, detail="approval_not_found")
        if existing.status != "pending":
            raise HTTPException(status_code=409, detail=f"approval_is_{existing.status}")

        def finalize_waiting_origin(delegated, content: str) -> None:
            origin = _safe_json(delegated.origin_json, {})
            origin_kind = str(delegated.origin_kind or "")
            if origin_kind == "direct_chat" and chat_store is not None:
                message_id = str(origin.get("chat_message_id") or "")
                if message_id and chat_store.get_message(message_id):
                    chat_store.complete_message(
                        message_id, content, message_kind="delegated_result"
                    )
                    if chat_event_broker is not None:
                        chat_event_broker.publish(
                            message_id,
                            {"type": "done", "message": chat_store.get_message(message_id)},
                        )
                    return
            if origin_kind == "scheduled_task" and chat_store is not None:
                task_id = origin.get("scheduled_task_id")
                if task_id is not None and chat_store.complete_scheduled_pending(
                    int(task_id), content
                ):
                    if task_store is not None:
                        task_store.mark_task_complete(int(task_id), status="cancelled")
                    return
            opportunity_id = str(origin.get("opportunity_id") or "")
            inbox = (
                autonomy_store.get_inbox_for_opportunity(opportunity_id)
                if opportunity_id and hasattr(autonomy_store, "get_inbox_for_opportunity")
                else None
            )
            if inbox is not None:
                autonomy_store.add_inbox_item(
                    replace(
                        inbox,
                        summary=content[:280],
                        detailed_report=inbox.detailed_report.rstrip() + "\n\n" + content,
                        status="completed_with_blocker",
                        updated_at=datetime.now().astimezone().isoformat(),
                    )
                )
                activity_run_id = str(origin.get("activity_run_id") or "")
                if activity_run_id and hasattr(autonomy_store, "complete_run"):
                    autonomy_store.complete_run(
                        activity_run_id,
                        summary=content[:280],
                        output_text=content,
                        status="completed_with_blocker",
                    )
        try:
            expired = datetime.fromisoformat(existing.expires_at) <= datetime.now().astimezone()
        except (TypeError, ValueError):
            expired = True
        if expired:
            if hasattr(autonomy_store, "expire_approval"):
                autonomy_store.expire_approval(approval_id)
            delegated = (
                autonomy_store.get_delegated_task_by_approval(approval_id)
                if hasattr(autonomy_store, "get_delegated_task_by_approval") else None
            )
            if delegated is not None:
                autonomy_store.update_delegated_task(delegated.delegation_id, status="expired")
                finalize_waiting_origin(
                    delegated,
                    f"The requested {delegated.capability} approval expired before execution.",
                )
            raise HTTPException(status_code=409, detail="approval_expired")
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
                delegated = (
                    autonomy_store.get_delegated_task_by_approval(approval_id)
                    if hasattr(autonomy_store, "get_delegated_task_by_approval") else None
                )
                if delegated is not None:
                    autonomy_store.update_delegated_task(delegated.delegation_id, status="approved")
                    details["delegation_id"] = delegated.delegation_id
                now = datetime.now(timezone.utc).isoformat()
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
        elif changed:
            delegated = (
                autonomy_store.get_delegated_task_by_approval(approval_id)
                if hasattr(autonomy_store, "get_delegated_task_by_approval") else None
            )
            if delegated is not None:
                autonomy_store.update_delegated_task(delegated.delegation_id, status="denied")
                finalize_waiting_origin(
                    delegated,
                    f"The requested {delegated.capability} task was denied and was not executed.",
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
        if runtime_control is not None and hasattr(runtime_control, "parallel_chat_status"):
            payload["parallel_chat"] = runtime_control.parallel_chat_status()
        if autonomy_store is not None and hasattr(autonomy_store, "event_counts"):
            payload["event_counts"] = autonomy_store.event_counts()
        return payload

    @app.get("/api/runtime/interrupt/status")
    def runtime_interrupt_status() -> dict[str, Any]:
        if runtime_control is None or not hasattr(runtime_control, "interrupt_status"):
            raise HTTPException(status_code=503, detail="runtime_control_unavailable")
        return {"ok": True, "status": runtime_control.interrupt_status()}

    @app.post("/api/runtime/interrupt")
    async def request_runtime_interrupt(request: Request) -> dict[str, Any]:
        if runtime_control is None or not hasattr(runtime_control, "request_interrupt"):
            raise HTTPException(status_code=503, detail="runtime_control_unavailable")
        try:
            body = await request.json()
        except Exception:
            body = {}
        reason = str(body.get("reason") or "Interrupted from runtime UI").strip()
        result = runtime_control.request_interrupt(reason=reason)
        if autonomy_store is not None and hasattr(autonomy_store, "audit"):
            try:
                autonomy_store.audit("runtime_ui", "runtime_interrupt_requested", None, {"reason": reason})
            except Exception:
                pass
        return result

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

    @app.get("/api/runtime/reflection/status")
    def manual_reflection_status() -> dict[str, Any]:
        if runtime_control is None or not hasattr(runtime_control, "manual_reflection_status"):
            raise HTTPException(status_code=503, detail="runtime_control_unavailable")
        return {"ok": True, "status": runtime_control.manual_reflection_status()}

    @app.post("/api/runtime/reflection/run")
    def manual_reflection_run() -> dict[str, Any]:
        if runtime_control is None or not hasattr(runtime_control, "request_reflection"):
            raise HTTPException(status_code=503, detail="runtime_control_unavailable")
        return runtime_control.request_reflection()

    @app.get("/api/runtime/biodata/status")
    def manual_biodata_status() -> dict[str, Any]:
        if runtime_control is None or not hasattr(runtime_control, "manual_biodata_status"):
            raise HTTPException(status_code=503, detail="runtime_control_unavailable")
        return {"ok": True, "status": runtime_control.manual_biodata_status()}

    @app.post("/api/runtime/biodata/run")
    def manual_biodata_run() -> dict[str, Any]:
        if runtime_control is None or not hasattr(runtime_control, "request_biodata_update"):
            raise HTTPException(status_code=503, detail="runtime_control_unavailable")
        return runtime_control.request_biodata_update()

    @app.get("/api/artifacts/maintenance/status")
    def artifact_maintenance_status() -> dict[str, Any]:
        if runtime_control is None or not hasattr(runtime_control, "artifact_maintenance_status"):
            raise HTTPException(status_code=503, detail="artifact_maintenance_unavailable")
        return {"ok": True, "status": runtime_control.artifact_maintenance_status()}

    @app.get("/api/artifacts/maintenance/history")
    def artifact_maintenance_history(limit: int = Query(default=50, ge=1, le=200)) -> dict[str, Any]:
        if runtime_control is None or not hasattr(runtime_control, "artifact_maintenance_history"):
            raise HTTPException(status_code=503, detail="artifact_maintenance_unavailable")
        return {"ok": True, **runtime_control.artifact_maintenance_history(limit=limit)}

    @app.post("/api/artifacts/maintenance/run")
    def artifact_maintenance_run() -> dict[str, Any]:
        if runtime_control is None or not hasattr(runtime_control, "request_artifact_maintenance"):
            raise HTTPException(status_code=503, detail="artifact_maintenance_unavailable")
        return runtime_control.request_artifact_maintenance()

    @app.get("/api/artifacts")
    def list_artifacts(
        status: str = Query(default="active", pattern="^(active|archived)$"),
        limit: int = Query(default=500, ge=1, le=1000),
    ) -> dict[str, Any]:
        if runtime_control is None or not hasattr(runtime_control, "list_artifacts"):
            raise HTTPException(status_code=503, detail="artifact_library_unavailable")
        items = runtime_control.list_artifacts(status=status, limit=limit)
        return {"ok": True, "status": status, "count": len(items), "items": items}

    @app.get("/api/artifacts/{artifact_id}")
    def get_artifact(artifact_id: str) -> dict[str, Any]:
        if runtime_control is None or not hasattr(runtime_control, "get_artifact"):
            raise HTTPException(status_code=503, detail="artifact_library_unavailable")
        item = runtime_control.get_artifact(artifact_id)
        if item is None:
            raise HTTPException(status_code=404, detail="artifact_not_found")
        return {"ok": True, "artifact": item}

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
        try:
            capture_control.set_exclusions(apps=apps, domains=domains)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except OSError as exc:
            raise HTTPException(status_code=500, detail="capture_exclusions_persistence_failed") from exc
        if autonomy_store is not None:
            autonomy_store.audit(
                "local_user",
                "capture.exclusions_updated",
                "global",
                capture_control.status(),
            )
        return {"ok": True, **capture_control.status()}

    @app.post("/api/privacy/capture/exclusions/check")
    def privacy_capture_exclusions_check() -> dict[str, Any]:
        if runtime_control is None or not hasattr(runtime_control, "check_capture_exclusions"):
            raise HTTPException(status_code=503, detail="foreground_check_unavailable")
        result = runtime_control.check_capture_exclusions()
        if not result.get("ok"):
            raise HTTPException(status_code=503, detail=result.get("error", "foreground_check_failed"))
        return result

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

    @app.get("/api/processing-queue")
    def list_processing_queue(
        limit: int = Query(default=500, ge=1, le=2000),
    ) -> dict[str, Any]:
        if autonomy_store is None or not hasattr(autonomy_store, "list_pending_media_events"):
            raise HTTPException(status_code=503, detail="processing_queue_unavailable")
        events = autonomy_store.list_pending_media_events(limit=limit)
        items: list[dict[str, Any]] = []
        for event in events:
            payload = _safe_json(event.payload_json, {})
            capture_uri = str(
                payload.get("screenshot_ref")
                or payload.get("audio_ref")
                or event.source_ref
                or ""
            )
            metadata: dict[str, Any] = {}
            if capture_store is not None and capture_uri.startswith("capture://"):
                try:
                    metadata = capture_store.metadata(capture_uri)
                except (OSError, ValueError):
                    metadata = {}
            modality = "image" if event.event_type == "lightweight_visual_capture" else "audio"
            items.append(
                {
                    "event_id": event.event_id,
                    "modality": modality,
                    "event_type": event.event_type,
                    "status": event.status,
                    "deletable": event.status in {"pending", "resource_deferred"},
                    "capture_uri": capture_uri if capture_uri.startswith("capture://") else None,
                    "original_name": (
                        metadata.get("original_name")
                        or payload.get("original_name")
                        or Path(event.source_ref).name
                        or f"{modality}-{event.event_id[:8]}"
                    ),
                    "mime_type": metadata.get("mime_type") or (
                        "image/png" if modality == "image" else "audio/wav"
                    ),
                    "size_bytes": metadata.get("size"),
                    "duration_seconds": payload.get("duration_seconds"),
                    "occurred_at": event.occurred_at,
                    "available_at": event.available_at,
                    "attempt_count": event.attempt_count,
                    "error_text": event.error_text,
                    "preview_url": f"/api/processing-queue/{event.event_id}/media",
                }
            )
        return {
            "items": items,
            "count": len(items),
            "image_count": sum(item["modality"] == "image" for item in items),
            "audio_count": sum(item["modality"] == "audio" for item in items),
            "processing_count": sum(item["status"] == "leased" for item in items),
        }

    @app.get("/api/processing-queue/{event_id}/media")
    def get_processing_queue_media(event_id: str) -> Response:
        if autonomy_store is None or not hasattr(autonomy_store, "get_media_event"):
            raise HTTPException(status_code=503, detail="processing_queue_unavailable")
        event = autonomy_store.get_media_event(event_id)
        if event is None:
            raise HTTPException(status_code=404, detail="queue_item_not_found")
        payload = _safe_json(event.payload_json, {})
        capture_uri = str(
            payload.get("screenshot_ref") or payload.get("audio_ref") or event.source_ref or ""
        )
        if capture_store is None or not capture_uri.startswith("capture://"):
            raise HTTPException(status_code=404, detail="queue_media_not_available")
        try:
            data, metadata = capture_store.read_bytes(capture_uri)
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="queue_media_not_found") from exc
        return Response(
            content=data,
            media_type=metadata.get("mime_type") or "application/octet-stream",
            headers={"Content-Disposition": "inline"},
        )

    @app.delete("/api/processing-queue/{event_id}")
    def delete_processing_queue_item(event_id: str) -> dict[str, Any]:
        if autonomy_store is None or not hasattr(autonomy_store, "cancel_pending_media_event"):
            raise HTTPException(status_code=503, detail="processing_queue_unavailable")
        current = autonomy_store.get_media_event(event_id)
        if current is None:
            raise HTTPException(status_code=404, detail="queue_item_not_found")
        if current.status not in {"pending", "resource_deferred"}:
            raise HTTPException(status_code=409, detail=f"queue_item_is_{current.status}")
        event = autonomy_store.cancel_pending_media_event(event_id)
        if event is None:
            raise HTTPException(status_code=404, detail="queue_item_not_found")
        if event.status != "ignored":
            raise HTTPException(
                status_code=409,
                detail="queue_item_is_already_processing" if event.status == "leased" else f"queue_item_is_{event.status}",
            )
        payload = _safe_json(event.payload_json, {})
        capture_uri = str(
            payload.get("screenshot_ref") or payload.get("audio_ref") or event.source_ref or ""
        )
        file_deleted = False
        if capture_store is not None and capture_uri.startswith("capture://"):
            file_deleted = bool(capture_store.delete(capture_uri))
        if hasattr(autonomy_store, "audit"):
            autonomy_store.audit(
                "local_user",
                "processing_queue.removed",
                event_id,
                {"event_type": event.event_type, "capture_deleted": file_deleted},
            )
        return {"ok": True, "event_id": event_id, "file_deleted": file_deleted}

    @app.delete("/api/privacy/captures")
    def delete_capture_file(request: Request, uri: str = Query(..., min_length=10)) -> dict[str, Any]:
        if capture_store is None:
            raise HTTPException(status_code=503, detail="capture_store_unavailable")
        deleted = capture_store.delete(uri)
        if autonomy_store is not None:
            autonomy_store.audit("local_user", "capture.deleted", uri, {"deleted": deleted})
        return {"ok": deleted}

    @app.get("/home", response_class=HTMLResponse)
    @app.get("/reports", response_class=HTMLResponse)
    @app.get("/artifacts", response_class=HTMLResponse)
    @app.get("/inbox", response_class=HTMLResponse)
    @app.get("/approvals", response_class=HTMLResponse)
    @app.get("/recurring-tasks", response_class=HTMLResponse)
    @app.get("/chat", response_class=HTMLResponse)
    @app.get("/interactions", response_class=HTMLResponse)
    @app.get("/", response_class=HTMLResponse)
    @app.get("/logs", response_class=HTMLResponse)
    @app.get("/benchmarks", response_class=HTMLResponse)
    @app.get("/real-world-tests", response_class=HTMLResponse)
    @app.get("/processing-queue", response_class=HTMLResponse)
    @app.get("/training", response_class=HTMLResponse)
    def view_logs(request: Request) -> str:
        if request.url.path == "/real-world-tests" and real_world_lab is None:
            raise HTTPException(status_code=404, detail="real_world_lab_unavailable")
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
    runtime_control: Any = None,
    real_world_lab: Any = None,
    rocm_tuning_service: Any = None,
) -> RuntimeLogBuffer:
    global _SERVER_THREAD, _SERVER
    log_buffer = configure_runtime_log_streaming(max_entries=max_entries)
    with _SERVER_LOCK:
        if _SERVER_THREAD is not None and _SERVER_THREAD.is_alive():
            return log_buffer

        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise RuntimeError("The unauthenticated runtime API may only bind to loopback.")

        _SERVER_SHUTDOWN_EVENT.clear()
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
            runtime_control=runtime_control,
            real_world_lab=real_world_lab,
            rocm_tuning_service=rocm_tuning_service,
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
        _SERVER_SHUTDOWN_EVENT.set()
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
