import logging
import threading
from collections import deque
from datetime import datetime
from html import escape
import json
from typing import Any

import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse

from infrastructure.adapter.SQLiteInteractionLogAdapter import SQLiteInteractionLogAdapter


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


def create_runtime_log_app(
    log_buffer: RuntimeLogBuffer,
    report_store: SQLiteInteractionLogAdapter | None = None,
) -> FastAPI:
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

    @app.get("/api/reports")
    def get_reports(limit: int = Query(default=50, ge=1, le=200)) -> dict[str, Any]:
        if report_store is None:
            return {"reports": [], "count": 0}
        rows = report_store.list_recent_reports(limit=limit)
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
        return {"reports": reports, "count": len(reports)}

    @app.get("/reports", response_class=HTMLResponse)
    @app.get("/", response_class=HTMLResponse)
    @app.get("/logs", response_class=HTMLResponse)
    def view_logs() -> str:
        return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Ambient Agent Dashboard</title>
  <style>
    :root {
      --bg: #f6f0e8;
      --panel: rgba(255,255,255,0.76);
      --panel-strong: rgba(255,255,255,0.9);
      --text: #1e1b18;
      --muted: #6f655c;
      --line: rgba(64,50,36,0.12);
      --shadow: 0 18px 40px rgba(48, 34, 18, 0.12);
      --accent: #b85c38;
      --accent-soft: #f0d5c9;
      --ok: #2f7d4f;
      --warn: #a06a00;
      --err: #a63c3c;
      --chip: #f3ece4;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--text);
      font-family: "Segoe UI", "SF Pro Text", "Helvetica Neue", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(184,92,56,0.18), transparent 28%),
        radial-gradient(circle at top right, rgba(95,146,130,0.16), transparent 32%),
        linear-gradient(180deg, #fbf7f2 0%, var(--bg) 100%);
      min-height: 100vh;
    }
    .shell {
      max-width: 980px;
      margin: 0 auto;
      padding: 20px 14px 40px;
    }
    header {
      position: sticky;
      top: 0;
      z-index: 10;
      backdrop-filter: blur(16px);
      background: rgba(246,240,232,0.82);
      border-bottom: 1px solid var(--line);
      padding: 16px 14px;
    }
    .title {
      font-size: 1.35rem;
      font-weight: 700;
      letter-spacing: 0.01em;
    }
    .subtitle {
      margin-top: 4px;
      color: var(--muted);
      font-size: 0.92rem;
    }
    .tabs {
      display: flex;
      gap: 10px;
      margin-top: 14px;
      flex-wrap: wrap;
    }
    .tab {
      border: 0;
      border-radius: 999px;
      padding: 10px 14px;
      background: rgba(255,255,255,0.62);
      color: var(--muted);
      font-size: 0.94rem;
      font-weight: 600;
      box-shadow: inset 0 0 0 1px rgba(64,50,36,0.08);
    }
    .tab.active {
      background: var(--accent);
      color: #fff8f2;
      box-shadow: none;
    }
    .status {
      margin-top: 10px;
      color: var(--muted);
      font-size: 0.9rem;
    }
    .section {
      display: none;
      margin-top: 18px;
    }
    .section.active {
      display: block;
    }
    .empty {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 22px;
      color: var(--muted);
      box-shadow: var(--shadow);
    }
    .report-card, .log-card {
      background: var(--panel-strong);
      border: 1px solid rgba(64,50,36,0.08);
      border-radius: 24px;
      padding: 18px;
      box-shadow: var(--shadow);
      margin-bottom: 14px;
    }
    .report-top {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }
    .chips {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    .chip {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 6px 10px;
      background: var(--chip);
      color: var(--muted);
      font-size: 0.8rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .chip.category-search_result { background: #e7f2ff; color: #2d5f95; }
    .chip.category-task_created { background: #eaf7ee; color: #2f7d4f; }
    .chip.category-reminder_created { background: #fff5df; color: #8d6200; }
    .chip.category-ambient_finding { background: #f8ebff; color: #7e4ea8; }
    .chip.category-error { background: #fde8e8; color: var(--err); }
    .report-text {
      font-size: 1.02rem;
      line-height: 1.55;
      margin: 0 0 14px;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .meta {
      color: var(--muted);
      font-size: 0.88rem;
      line-height: 1.5;
    }
    .meta strong {
      color: var(--text);
      font-weight: 600;
    }
    .log-card {
      font-family: Consolas, "SFMono-Regular", monospace;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .log-line {
      padding-bottom: 10px;
      border-bottom: 1px solid var(--line);
      margin-bottom: 10px;
    }
    .log-line:last-child {
      border-bottom: 0;
      margin-bottom: 0;
      padding-bottom: 0;
    }
    .log-level-INFO { color: var(--ok); }
    .log-level-WARNING { color: var(--warn); }
    .log-level-ERROR, .log-level-CRITICAL { color: var(--err); }
    .log-level-DEBUG { color: #275d8f; }
    @media (max-width: 640px) {
      .shell { padding: 16px 10px 32px; }
      .report-card, .log-card, .empty { border-radius: 18px; padding: 16px; }
      .title { font-size: 1.18rem; }
      .report-text { font-size: 0.98rem; }
    }
  </style>
</head>
<body>
  <header>
    <div class="title">Ambient Agent Dashboard</div>
    <div class="subtitle">Meaningful agent work first. Debug logs second.</div>
    <div class="tabs">
      <button class="tab active" data-tab="reports">Reports</button>
      <button class="tab" data-tab="logs">Logs</button>
    </div>
    <div id="status" class="status">connecting...</div>
  </header>
  <main class="shell">
    <section id="reportsSection" class="section active">
      <div id="reportsEmpty" class="empty">No user-relevant agent reports yet.</div>
      <div id="reports"></div>
    </section>
    <section id="logsSection" class="section">
      <div id="logs"></div>
    </section>
  </main>
  <script>
    let latestId = 0;
    let latestReportKey = "";
    const logs = document.getElementById("logs");
    const reports = document.getElementById("reports");
    const reportsEmpty = document.getElementById("reportsEmpty");
    const status = document.getElementById("status");
    const tabs = Array.from(document.querySelectorAll(".tab"));
    const sections = {
      reports: document.getElementById("reportsSection"),
      logs: document.getElementById("logsSection"),
    };

    tabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        const key = tab.dataset.tab;
        tabs.forEach((item) => item.classList.toggle("active", item === tab));
        Object.entries(sections).forEach(([name, section]) => {
          section.classList.toggle("active", name === key);
        });
      });
    });

    function appendEntry(entry) {
      const wrapper = document.createElement("div");
      wrapper.className = "log-card";
      const line = document.createElement("div");
      line.className = "log-line";
      const levelClass = "log-level-" + entry.level;
      line.innerHTML = `<div class="${levelClass}">${entry.message}</div>`;
      wrapper.appendChild(line);
      logs.appendChild(wrapper);
    }

    function renderReports(items) {
      reports.innerHTML = "";
      reportsEmpty.style.display = items.length ? "none" : "block";
      for (const item of items) {
        const report = item.report || {};
        const card = document.createElement("article");
        card.className = "report-card";
        const chips = [];
        if (report.category) {
          chips.push(`<span class="chip category-${report.category}">${report.category.replaceAll("_", " ")}</span>`);
        }
        if (report.status) {
          chips.push(`<span class="chip">${report.status}</span>`);
        }
        const tools = Array.isArray(report.tools_used) && report.tools_used.length
          ? report.tools_used.join(", ")
          : "none";
        card.innerHTML = `
          <div class="report-top">
            <div class="chips">${chips.join("")}</div>
            <div class="meta">${new Date(item.created_at).toLocaleString()}</div>
          </div>
          <p class="report-text">${report.report_to_user || ""}</p>
          <div class="meta">
            <div><strong>Source:</strong> ${item.source || "unknown"}</div>
            <div><strong>Model:</strong> ${item.model || "unknown"}</div>
            <div><strong>Tools:</strong> ${tools}</div>
          </div>
        `;
        reports.appendChild(card);
      }
    }

    async function pollReports() {
      const res = await fetch(`/api/reports?limit=50`, { cache: "no-store" });
      const payload = await res.json();
      const serialized = JSON.stringify(payload.reports || []);
      if (serialized !== latestReportKey) {
        latestReportKey = serialized;
        renderReports(payload.reports || []);
      }
    }

    async function poll() {
      try {
        const [logRes] = await Promise.all([
          fetch(`/api/logs?after_id=${latestId}&limit=500`, { cache: "no-store" }),
          pollReports(),
        ]);
        const payload = await logRes.json();
        for (const entry of payload.entries || []) {
          appendEntry(entry);
          latestId = Math.max(latestId, entry.id);
        }
        if ((payload.entries || []).length > 0 && sections.logs.classList.contains("active")) {
          window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
        }
        status.textContent = `reports live • latest log id ${payload.latest_id}`;
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
    report_store: SQLiteInteractionLogAdapter | None = None,
) -> RuntimeLogBuffer:
    global _SERVER_THREAD
    log_buffer = configure_runtime_log_streaming(max_entries=max_entries)
    with _SERVER_LOCK:
        if _SERVER_THREAD is not None and _SERVER_THREAD.is_alive():
            return log_buffer

        app = create_runtime_log_app(log_buffer, report_store=report_store)

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
