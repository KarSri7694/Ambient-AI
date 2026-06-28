import logging
import threading
from collections import deque
from datetime import datetime
from html import escape
import json
from typing import Any

import uvicorn
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse

from infrastructure.adapter.SQLiteBenchmarkAdapter import SQLiteBenchmarkAdapter
from infrastructure.adapter.SQLiteInteractionLogAdapter import SQLiteInteractionLogAdapter
from infrastructure.adapter.SQLiteTaskQueueAdapter import SQLiteTaskQueueAdapter


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
    task_store: SQLiteTaskQueueAdapter | None = None,
    benchmark_store: SQLiteBenchmarkAdapter | None = None,
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
            for task in task_store.get_pending_tasks():
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
                    }
                )
        return {"reports": reports, "queued_tasks": queued_tasks, "count": len(reports)}

    @app.get("/api/benchmarks/runs")
    def get_benchmark_runs(
        limit: int = Query(default=50, ge=1, le=200),
        service_name: str | None = Query(default=None),
    ) -> dict[str, Any]:
        rows = benchmark_store.list_runs(limit=limit, service_name=service_name) if benchmark_store is not None else []
        return {
            "runs": [row.__dict__ for row in rows],
            "count": len(rows),
        }

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
            item = dict(row.__dict__)
            review = benchmark_store.get_manual_review(row.result_id) if benchmark_store is not None else None
            item["manual_review"] = review.__dict__ if review is not None else None
            payload.append(item)
        return {"results": payload, "count": len(payload)}

    @app.get("/api/benchmarks/results/{result_id}")
    def get_benchmark_result(result_id: str) -> dict[str, Any]:
        if benchmark_store is None:
            return {"result": None}
        row = benchmark_store.get_result(result_id)
        if row is None:
            return {"result": None}
        payload = dict(row.__dict__)
        review = benchmark_store.get_manual_review(result_id)
        payload["manual_review"] = review.__dict__ if review is not None else None
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
        return {"ok": True, "review": review.__dict__}

    @app.get("/reports", response_class=HTMLResponse)
    @app.get("/", response_class=HTMLResponse)
    @app.get("/logs", response_class=HTMLResponse)
    @app.get("/benchmarks", response_class=HTMLResponse)
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
      max-width: 1180px;
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
    .reports-grid {
      display: grid;
      grid-template-columns: minmax(0, 1.7fr) minmax(280px, 0.9fr);
      gap: 16px;
      align-items: start;
    }
    .report-card, .log-card, .task-card, .task-panel, .benchmark-card, .review-panel {
      background: var(--panel-strong);
      border: 1px solid rgba(64,50,36,0.08);
      border-radius: 24px;
      padding: 18px;
      box-shadow: var(--shadow);
      margin-bottom: 14px;
    }
    .task-panel {
      position: sticky;
      top: 126px;
    }
    .panel-title {
      font-size: 1rem;
      font-weight: 700;
      margin: 0 0 12px;
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
    .report-title {
      font-size: 1.08rem;
      font-weight: 700;
      line-height: 1.35;
      margin: 0 0 10px;
      word-break: break-word;
    }
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
    .task-card {
      margin-bottom: 12px;
      background: rgba(255,255,255,0.82);
    }
    .task-text {
      font-size: 0.96rem;
      line-height: 1.45;
      margin: 0 0 10px;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .log-card {
      font-family: Consolas, "SFMono-Regular", monospace;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 5px 9px;
      background: var(--accent-soft);
      color: #7b452f;
      font-size: 0.76rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.03em;
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
    .bench-grid {
      display: grid;
      grid-template-columns: minmax(0, 1.6fr) minmax(300px, 1fr);
      gap: 16px;
      align-items: start;
    }
    .review-panel {
      position: sticky;
      top: 126px;
    }
    .benchmark-title {
      font-size: 1rem;
      font-weight: 700;
      margin: 0 0 10px;
    }
    .benchmark-meta {
      color: var(--muted);
      font-size: 0.88rem;
      line-height: 1.55;
      margin-bottom: 10px;
    }
    .benchmark-output {
      background: rgba(248, 243, 235, 0.85);
      border-radius: 14px;
      padding: 12px;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: Consolas, "SFMono-Regular", monospace;
      font-size: 0.86rem;
      max-height: 280px;
      overflow: auto;
    }
    .benchmark-json {
      margin-top: 12px;
      border-top: 1px solid var(--line);
      padding-top: 12px;
    }
    .benchmark-json details {
      background: rgba(255,255,255,0.65);
      border: 1px solid rgba(64,50,36,0.08);
      border-radius: 12px;
      padding: 10px 12px;
      margin-top: 8px;
    }
    .benchmark-json summary {
      cursor: pointer;
      font-weight: 700;
      color: var(--text);
    }
    .benchmark-json pre {
      margin: 10px 0 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: Consolas, "SFMono-Regular", monospace;
      font-size: 0.83rem;
      max-height: 420px;
      overflow: auto;
    }
    .review-panel input, .review-panel textarea {
      width: 100%;
      border-radius: 12px;
      border: 1px solid var(--line);
      padding: 10px 12px;
      font: inherit;
      background: rgba(255,255,255,0.92);
      margin-bottom: 10px;
    }
    .review-panel button {
      border: 0;
      border-radius: 999px;
      padding: 10px 14px;
      background: var(--accent);
      color: #fff8f2;
      font-weight: 700;
      cursor: pointer;
    }
    @media (max-width: 640px) {
      .shell { padding: 16px 10px 32px; }
      .report-card, .log-card, .empty, .task-card, .task-panel, .benchmark-card, .review-panel { border-radius: 18px; padding: 16px; }
      .title { font-size: 1.18rem; }
      .report-text { font-size: 0.98rem; }
      .reports-grid { grid-template-columns: 1fr; }
      .bench-grid { grid-template-columns: 1fr; }
      .task-panel { position: static; }
      .review-panel { position: static; }
    }
  </style>
</head>
<body>
  <header>
    <div class="title">Ambient Agent Dashboard</div>
    <div class="subtitle">Meaningful agent work first. Debug logs second.</div>
    <div class="tabs">
      <button class="tab active" data-tab="reports">Reports</button>
      <button class="tab" data-tab="benchmarks">Benchmarks</button>
      <button class="tab" data-tab="logs">Logs</button>
    </div>
    <div id="status" class="status">connecting...</div>
  </header>
  <main class="shell">
    <section id="reportsSection" class="section active">
      <div class="reports-grid">
        <div>
          <div id="reportsEmpty" class="empty">No user-relevant agent reports yet.</div>
          <div id="reports"></div>
        </div>
        <aside class="task-panel">
          <h2 class="panel-title">Queued Tasks</h2>
          <div id="tasksEmpty" class="empty">No queued tasks right now.</div>
          <div id="queuedTasks"></div>
        </aside>
      </div>
    </section>
    <section id="benchmarksSection" class="section">
      <div class="bench-grid">
        <div>
          <div id="benchmarksEmpty" class="empty">No benchmark results stored yet.</div>
          <div id="benchmarks"></div>
        </div>
        <aside class="review-panel">
          <h2 class="panel-title">Manual Review</h2>
          <div id="reviewHint" class="meta">Select a benchmark result to review.</div>
          <input id="reviewerInput" type="text" placeholder="Reviewer" value="local-user" />
          <input id="scoreInput" type="number" min="0" max="1" step="0.01" placeholder="Score 0.00 - 1.00" />
          <textarea id="notesInput" rows="8" placeholder="Manual review notes"></textarea>
          <button id="saveReviewButton" type="button">Save Review</button>
        </aside>
      </div>
    </section>
    <section id="logsSection" class="section">
      <div id="logs"></div>
    </section>
  </main>
  <script>
    let latestId = 0;
    let latestReportKey = "";
    let latestBenchmarkKey = "";
    let selectedBenchmarkId = "";
    const logs = document.getElementById("logs");
    const reports = document.getElementById("reports");
    const reportsEmpty = document.getElementById("reportsEmpty");
    const queuedTasks = document.getElementById("queuedTasks");
    const tasksEmpty = document.getElementById("tasksEmpty");
    const benchmarks = document.getElementById("benchmarks");
    const benchmarksEmpty = document.getElementById("benchmarksEmpty");
    const reviewerInput = document.getElementById("reviewerInput");
    const scoreInput = document.getElementById("scoreInput");
    const notesInput = document.getElementById("notesInput");
    const reviewHint = document.getElementById("reviewHint");
    const saveReviewButton = document.getElementById("saveReviewButton");
    const status = document.getElementById("status");
    const tabs = Array.from(document.querySelectorAll(".tab"));
    const sections = {
      reports: document.getElementById("reportsSection"),
      benchmarks: document.getElementById("benchmarksSection"),
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
          <h2 class="report-title">${report.title || ""}</h2>
          <p class="report-text">${report.summary || ""}</p>
          <div class="meta">
            <div><strong>Source:</strong> ${item.source || "unknown"}</div>
            <div><strong>Model:</strong> ${item.model || "unknown"}</div>
            <div><strong>Tools:</strong> ${tools}</div>
            <div><strong>Artifact:</strong> ${report.artifact_path || ""}</div>
          </div>
        `;
        reports.appendChild(card);
      }
    }

    function renderQueuedTasks(items) {
      queuedTasks.innerHTML = "";
      tasksEmpty.style.display = items.length ? "none" : "block";
      for (const item of items) {
        const card = document.createElement("article");
        card.className = "task-card";
        const createdAt = item.created_at ? new Date(item.created_at).toLocaleString() : "unknown";
        card.innerHTML = `
          <div class="report-top">
            <span class="badge">${item.priority || "medium"}</span>
            <div class="meta">${createdAt}</div>
          </div>
          <p class="task-text">${item.description || ""}</p>
          <div class="meta">
            <div><strong>Status:</strong> ${item.status || "pending"}</div>
            <div><strong>ID:</strong> ${item.id ?? ""}</div>
          </div>
        `;
        queuedTasks.appendChild(card);
      }
    }

    function renderBenchmarks(items) {
      benchmarks.innerHTML = "";
      benchmarksEmpty.style.display = items.length ? "none" : "block";
      for (const item of items) {
        const card = document.createElement("article");
        card.className = "benchmark-card";
        card.dataset.resultId = item.result_id;
        const review = item.manual_review || {};
        const output = item.response_text || item.error_text || item.structured_output_json || "";
        const structuredJson = item.structured_output_json || "";
        const metadataJson = item.metadata_json || "";
        const scoreJson = item.auto_score_details_json || "";
        card.innerHTML = `
          <div class="report-top">
            <div class="chips">
              <span class="chip">${item.status || "completed"}</span>
              <span class="chip">${item.service_name || ""}</span>
            </div>
            <div class="meta">${new Date(item.created_at).toLocaleString()}</div>
          </div>
          <h2 class="benchmark-title">${item.case_title || item.case_id}</h2>
          <div class="benchmark-meta">
            <div><strong>Model:</strong> ${item.model_name || "unknown"}</div>
            <div><strong>Auto score:</strong> ${item.auto_score ?? "n/a"}</div>
            <div><strong>Prefill speed:</strong> ${item.prefill_tokens_per_second ?? 0} tok/s</div>
            <div><strong>Total tokens:</strong> ${item.total_tokens ?? 0}</div>
            <div><strong>Generation speed:</strong> ${item.generation_tokens_per_second ?? 0} tok/s</div>
            <div><strong>Manual score:</strong> ${review.score ?? "n/a"}</div>
          </div>
          <div class="benchmark-output">${output}</div>
          <div class="benchmark-json">
            ${structuredJson ? `<details><summary>Structured Output JSON</summary><pre>${structuredJson}</pre></details>` : ""}
            ${scoreJson ? `<details><summary>Auto Score Details</summary><pre>${scoreJson}</pre></details>` : ""}
            ${metadataJson ? `<details><summary>Benchmark Metadata</summary><pre>${metadataJson}</pre></details>` : ""}
          </div>
        `;
        card.addEventListener("click", () => {
          selectedBenchmarkId = item.result_id;
          reviewerInput.value = review.reviewer || "local-user";
          scoreInput.value = review.score ?? "";
          notesInput.value = review.notes || "";
          reviewHint.textContent = `Reviewing ${item.case_id} on ${item.model_name}`;
        });
        benchmarks.appendChild(card);
      }
    }

    async function pollReports() {
      const res = await fetch(`/api/reports?limit=50`, { cache: "no-store" });
      const payload = await res.json();
      const serialized = JSON.stringify({
        reports: payload.reports || [],
        queued_tasks: payload.queued_tasks || [],
      });
      if (serialized !== latestReportKey) {
        latestReportKey = serialized;
        renderReports(payload.reports || []);
        renderQueuedTasks(payload.queued_tasks || []);
      }
    }

    async function pollBenchmarks() {
      const res = await fetch(`/api/benchmarks/results?limit=100`, { cache: "no-store" });
      const payload = await res.json();
      const serialized = JSON.stringify(payload.results || []);
      if (serialized !== latestBenchmarkKey) {
        latestBenchmarkKey = serialized;
        renderBenchmarks(payload.results || []);
      }
    }

    saveReviewButton.addEventListener("click", async () => {
      if (!selectedBenchmarkId) {
        reviewHint.textContent = "Select a benchmark result before saving a review.";
        return;
      }
      const response = await fetch(`/api/benchmarks/results/${selectedBenchmarkId}/review`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          reviewer: reviewerInput.value,
          score: scoreInput.value,
          notes: notesInput.value,
        }),
      });
      const payload = await response.json();
      if (payload.ok) {
        reviewHint.textContent = "Manual review saved.";
        await pollBenchmarks();
      } else {
        reviewHint.textContent = "Failed to save manual review.";
      }
    });

    async function poll() {
      try {
        const [logRes] = await Promise.all([
          fetch(`/api/logs?after_id=${latestId}&limit=500`, { cache: "no-store" }),
          pollReports(),
          pollBenchmarks(),
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
    task_store: SQLiteTaskQueueAdapter | None = None,
    benchmark_store: SQLiteBenchmarkAdapter | None = None,
) -> RuntimeLogBuffer:
    global _SERVER_THREAD
    log_buffer = configure_runtime_log_streaming(max_entries=max_entries)
    with _SERVER_LOCK:
        if _SERVER_THREAD is not None and _SERVER_THREAD.is_alive():
            return log_buffer

        app = create_runtime_log_app(
            log_buffer,
            report_store=report_store,
            task_store=task_store,
            benchmark_store=benchmark_store,
        )

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
