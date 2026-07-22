import asyncio
import hmac
import json
import logging
import mimetypes
import queue
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from application.services.training_data_service import TrainingDataService
from infrastructure.adapter.SQLiteBenchmarkAdapter import SQLiteBenchmarkAdapter
from infrastructure.adapter.SQLiteChatAdapter import ChatEventBroker, SQLiteChatAdapter
from infrastructure.adapter.SQLiteInteractionLogAdapter import SQLiteInteractionLogAdapter
from infrastructure.adapter.SQLiteTaskQueueAdapter import SQLiteTaskQueueAdapter
from infrastructure.adapter.SQLiteTrainingDataAdapter import SQLiteTrainingDataAdapter


UI_ROOT = Path(__file__).resolve().parent / "runtime_ui"
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
    auth_token: str = "",
) -> FastAPI:
    app = FastAPI(title="Ambient Runtime Logs")
    normalized_media_roots = [Path(root) for root in (media_roots or [])]
    if UI_ROOT.exists():
        app.mount("/runtime-ui", StaticFiles(directory=str(UI_ROOT)), name="runtime_ui")

    @app.middleware("http")
    async def protect_api(request: Request, call_next):
        if request.url.path.startswith("/api/") and auth_token:
            authorization = request.headers.get("authorization", "")
            expected = f"Bearer {auth_token}"
            cookie_token = request.cookies.get("ambient_auth", "")
            if not (
                hmac.compare_digest(authorization, expected)
                or hmac.compare_digest(cookie_token, auth_token)
            ):
                return JSONResponse(status_code=401, content={"detail": "invalid_auth_token"})
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

    @app.get("/reports", response_class=HTMLResponse)
    @app.get("/chat", response_class=HTMLResponse)
    @app.get("/", response_class=HTMLResponse)
    @app.get("/logs", response_class=HTMLResponse)
    @app.get("/benchmarks", response_class=HTMLResponse)
    @app.get("/training", response_class=HTMLResponse)
    def view_logs() -> str:
        return _load_dashboard_html()

    return app


def start_runtime_log_server(
    *,
    host: str = "0.0.0.0",
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
    auth_token: str = "",
) -> RuntimeLogBuffer:
    global _SERVER_THREAD, _SERVER
    log_buffer = configure_runtime_log_streaming(max_entries=max_entries)
    with _SERVER_LOCK:
        if _SERVER_THREAD is not None and _SERVER_THREAD.is_alive():
            return log_buffer

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
            auth_token=auth_token,
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
