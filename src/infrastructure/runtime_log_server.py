import logging
import threading
from collections import deque
from datetime import datetime
from html import escape
from typing import Any

import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse


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
_SERVER_LOCK = threading.Lock()


def configure_runtime_log_streaming(max_entries: int = 2000) -> RuntimeLogBuffer:
    global _LOG_BUFFER, _LOG_HANDLER
    if _LOG_BUFFER is not None and _LOG_HANDLER is not None:
        return _LOG_BUFFER

    buffer = RuntimeLogBuffer(max_entries=max_entries)
    handler = RuntimeLogBufferHandler(buffer)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    _LOG_BUFFER = buffer
    _LOG_HANDLER = handler
    return buffer


def create_runtime_log_app(log_buffer: RuntimeLogBuffer) -> FastAPI:
    app = FastAPI(title="Ambient Runtime Logs")

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

    @app.get("/", response_class=HTMLResponse)
    @app.get("/logs", response_class=HTMLResponse)
    def view_logs() -> str:
        return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Ambient Runtime Logs</title>
  <style>
    body { font-family: monospace; background: #111; color: #eee; margin: 0; }
    header { position: sticky; top: 0; background: #1a1a1a; padding: 12px; border-bottom: 1px solid #333; }
    #logs { padding: 12px; white-space: pre-wrap; word-break: break-word; }
    .line { padding: 6px 0; border-bottom: 1px solid #222; }
    .INFO { color: #c8f7c5; }
    .WARNING { color: #ffe08a; }
    .ERROR, .CRITICAL { color: #ff8a8a; }
    .DEBUG { color: #9ecbff; }
  </style>
</head>
<body>
  <header>
    <strong>Ambient Runtime Logs</strong>
    <span id="status" style="margin-left: 12px; opacity: 0.7;">connecting...</span>
  </header>
  <div id="logs"></div>
  <script>
    let latestId = 0;
    const logs = document.getElementById("logs");
    const status = document.getElementById("status");

    function appendEntry(entry) {
      const div = document.createElement("div");
      div.className = "line " + entry.level;
      div.textContent = entry.message;
      logs.appendChild(div);
    }

    async function poll() {
      try {
        const res = await fetch(`/api/logs?after_id=${latestId}&limit=500`, { cache: "no-store" });
        const payload = await res.json();
        for (const entry of payload.entries) {
          appendEntry(entry);
          latestId = Math.max(latestId, entry.id);
        }
        if (payload.entries.length > 0) {
          window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
        }
        status.textContent = `latest log id ${payload.latest_id}`;
      } catch (error) {
        status.textContent = "disconnected";
      }
    }

    poll();
    setInterval(poll, 2000);
  </script>
</body>
</html>
        """.strip()

    return app


def start_runtime_log_server(
    *,
    host: str = "0.0.0.0",
    port: int = 8765,
    max_entries: int = 2000,
) -> RuntimeLogBuffer:
    global _SERVER_THREAD
    log_buffer = configure_runtime_log_streaming(max_entries=max_entries)
    with _SERVER_LOCK:
        if _SERVER_THREAD is not None and _SERVER_THREAD.is_alive():
            return log_buffer

        app = create_runtime_log_app(log_buffer)

        def _serve() -> None:
            config = uvicorn.Config(
                app=app,
                host=host,
                port=port,
                log_level="warning",
                access_log=False,
            )
            server = uvicorn.Server(config)
            server.run()

        thread = threading.Thread(
            target=_serve,
            daemon=True,
            name="RuntimeLogServerThread",
        )
        thread.start()
        _SERVER_THREAD = thread
    return log_buffer
