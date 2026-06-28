const state = {
  latestId: 0,
  latestReportKey: "",
  latestBenchmarkKey: "",
  latestTrainingKey: "",
  selectedBenchmarkId: "",
  trainingMode: "llm",
  selectedTrainingId: "",
  selectedTrainingRecord: null,
  trainingFilters: {
    reviewStatus: "",
    source: "",
    model: "",
  },
  logCount: 0,
};

const els = {
  status: document.getElementById("status"),
  tabs: Array.from(document.querySelectorAll(".tab")),
  sections: {
    reports: document.getElementById("reportsSection"),
    benchmarks: document.getElementById("benchmarksSection"),
    training: document.getElementById("trainingSection"),
    logs: document.getElementById("logsSection"),
  },
  reports: document.getElementById("reports"),
  reportsEmpty: document.getElementById("reportsEmpty"),
  reportsCount: document.getElementById("reportsCount"),
  queuedTasks: document.getElementById("queuedTasks"),
  tasksEmpty: document.getElementById("tasksEmpty"),
  tasksCount: document.getElementById("tasksCount"),
  benchmarks: document.getElementById("benchmarks"),
  benchmarksEmpty: document.getElementById("benchmarksEmpty"),
  benchmarksCount: document.getElementById("benchmarksCount"),
  reviewHint: document.getElementById("reviewHint"),
  reviewerInput: document.getElementById("reviewerInput"),
  scoreInput: document.getElementById("scoreInput"),
  notesInput: document.getElementById("notesInput"),
  saveReviewButton: document.getElementById("saveReviewButton"),
  trainingRecords: document.getElementById("trainingRecords"),
  trainingEmpty: document.getElementById("trainingEmpty"),
  trainingFilterStatus: document.getElementById("trainingFilterStatus"),
  trainingFilterSource: document.getElementById("trainingFilterSource"),
  trainingFilterModel: document.getElementById("trainingFilterModel"),
  trainingFilterSourceWrap: document.getElementById("trainingFilterSourceWrap"),
  trainingFilterModelWrap: document.getElementById("trainingFilterModelWrap"),
  applyTrainingFiltersButton: document.getElementById("applyTrainingFiltersButton"),
  clearTrainingFiltersButton: document.getElementById("clearTrainingFiltersButton"),
  trainingViewer: document.getElementById("trainingViewer"),
  trainingHint: document.getElementById("trainingHint"),
  trainingPrimaryText: document.getElementById("trainingPrimaryText"),
  trainingSecondaryText: document.getElementById("trainingSecondaryText"),
  trainingPrimaryHelp: document.getElementById("trainingPrimaryHelp"),
  trainingSecondaryHelp: document.getElementById("trainingSecondaryHelp"),
  trainingNotesInput: document.getElementById("trainingNotesInput"),
  trainingReviewerInput: document.getElementById("trainingReviewerInput"),
  trainingStatusInput: document.getElementById("trainingStatusInput"),
  trainingExports: document.getElementById("trainingExports"),
  saveTrainingReviewButton: document.getElementById("saveTrainingReviewButton"),
  syncTrainingButton: document.getElementById("syncTrainingButton"),
  exportTrainingButton: document.getElementById("exportTrainingButton"),
  modeLlm: document.getElementById("modeLlm"),
  modeAsr: document.getElementById("modeAsr"),
  logs: document.getElementById("logs"),
  logsCount: document.getElementById("logsCount"),
  imageLightbox: document.getElementById("imageLightbox"),
  imageLightboxImage: document.getElementById("imageLightboxImage"),
  imageLightboxClose: document.getElementById("imageLightboxClose"),
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatDate(value) {
  if (!value) return "unknown";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

function prettyJson(value) {
  if (value === undefined || value === null || value === "") return "";
  if (typeof value === "string") {
    try {
      return JSON.stringify(JSON.parse(value), null, 2);
    } catch {
      return value;
    }
  }
  return JSON.stringify(value, null, 2);
}

function chip(value) {
  const normalized = String(value || "unknown").toLowerCase();
  return `<span class="chip ${escapeHtml(normalized)}">${escapeHtml(value || "unknown")}</span>`;
}

function mediaUrl(path) {
  return `/api/training/media?path=${encodeURIComponent(path)}`;
}

function buildQuery(params) {
  const query = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && String(value).trim() !== "") {
      query.set(key, String(value).trim());
    }
  }
  return query.toString();
}

function clearSelectedTrainingRecord() {
  state.selectedTrainingId = "";
  state.selectedTrainingRecord = null;
  els.trainingViewer.innerHTML = "";
  els.trainingHint.textContent = `Select a ${state.trainingMode.toUpperCase()} record to inspect and correct.`;
  els.trainingPrimaryText.value = "";
  els.trainingSecondaryText.value = "";
  els.trainingNotesInput.value = "";
  els.trainingStatusInput.value = "pending";
}

function syncTrainingFilterStateFromInputs() {
  state.trainingFilters = {
    reviewStatus: els.trainingFilterStatus.value.trim(),
    source: els.trainingFilterSource.value.trim(),
    model: els.trainingFilterModel.value.trim(),
  };
}

function openImageLightbox(src) {
  if (!src) return;
  els.imageLightboxImage.src = src;
  els.imageLightbox.classList.add("open");
  els.imageLightbox.setAttribute("aria-hidden", "false");
  els.imageLightboxClose.focus();
}

function closeImageLightbox() {
  els.imageLightbox.classList.remove("open");
  els.imageLightbox.setAttribute("aria-hidden", "true");
  els.imageLightboxImage.removeAttribute("src");
}

async function getJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  return response.json();
}

function renderReports(items) {
  els.reports.innerHTML = "";
  els.reportsEmpty.style.display = items.length ? "none" : "block";
  els.reportsCount.textContent = String(items.length);
  for (const item of items) {
    const report = item.report || {};
    const tools = Array.isArray(report.tools_used) && report.tools_used.length ? report.tools_used.join(", ") : "none";
    const card = document.createElement("article");
    card.className = "record-card";
    card.innerHTML = `
      <div class="record-top">
        <div class="chips">${report.status ? chip(report.status) : ""}${item.source ? chip(item.source) : ""}</div>
        <div class="timestamp">${escapeHtml(formatDate(item.created_at))}</div>
      </div>
      <h3 class="record-title">${escapeHtml(report.title || "Untitled report")}</h3>
      <p class="record-text">${escapeHtml(report.summary || "No summary provided.")}</p>
      <div class="meta-grid">
        <div><strong>Model</strong>${escapeHtml(item.model || "unknown")}</div>
        <div><strong>Tools</strong>${escapeHtml(tools)}</div>
        <div><strong>Artifact</strong>${escapeHtml(report.artifact_path || "none")}</div>
      </div>
    `;
    els.reports.appendChild(card);
  }
}

function renderQueuedTasks(items) {
  els.queuedTasks.innerHTML = "";
  els.tasksEmpty.style.display = items.length ? "none" : "block";
  els.tasksCount.textContent = String(items.length);
  for (const item of items) {
    const card = document.createElement("article");
    card.className = "record-card";
    card.innerHTML = `
      <div class="record-top">
        <div class="chips">${chip(item.priority || "medium")}${chip(item.status || "pending")}</div>
        <div class="timestamp">${escapeHtml(formatDate(item.created_at))}</div>
      </div>
      <h3 class="record-title">Task ${escapeHtml(item.id ?? "")}</h3>
      <p class="record-text">${escapeHtml(item.description || "")}</p>
    `;
    els.queuedTasks.appendChild(card);
  }
}

function renderBenchmarks(items) {
  els.benchmarks.innerHTML = "";
  els.benchmarksEmpty.style.display = items.length ? "none" : "block";
  els.benchmarksCount.textContent = String(items.length);
  for (const item of items) {
    const review = item.manual_review || {};
    const output = item.response_text || item.error_text || item.structured_output_json || "";
    const card = document.createElement("article");
    card.className = `record-card selectable ${item.result_id === state.selectedBenchmarkId ? "selected" : ""}`;
    card.innerHTML = `
      <div class="record-top">
        <div class="chips">${chip(item.status || "completed")}${chip(item.service_name || "service")}</div>
        <div class="timestamp">${escapeHtml(formatDate(item.created_at))}</div>
      </div>
      <h3 class="record-title">${escapeHtml(item.case_title || item.case_id || "Benchmark case")}</h3>
      <div class="meta-grid">
        <div><strong>Model</strong>${escapeHtml(item.model_name || "unknown")}</div>
        <div><strong>Total tokens</strong>${escapeHtml(item.total_tokens ?? 0)}</div>
        <div><strong>Prefill speed</strong>${escapeHtml(item.prefill_tokens_per_second ?? 0)} tok/s</div>
        <div><strong>Generation speed</strong>${escapeHtml(item.generation_tokens_per_second ?? 0)} tok/s</div>
        <div><strong>Auto score</strong>${escapeHtml(item.auto_score ?? "n/a")}</div>
        <div><strong>Manual score</strong>${escapeHtml(review.score ?? "n/a")}</div>
      </div>
      <div class="output">${escapeHtml(output)}</div>
      ${item.structured_output_json ? `<details><summary>Structured Output JSON</summary><pre>${escapeHtml(prettyJson(item.structured_output_json))}</pre></details>` : ""}
      ${item.auto_score_details_json ? `<details><summary>Auto Score Details</summary><pre>${escapeHtml(prettyJson(item.auto_score_details_json))}</pre></details>` : ""}
      ${item.metadata_json ? `<details><summary>Benchmark Metadata</summary><pre>${escapeHtml(prettyJson(item.metadata_json))}</pre></details>` : ""}
    `;
    card.addEventListener("click", () => {
      state.selectedBenchmarkId = item.result_id;
      els.reviewerInput.value = review.reviewer || "local-user";
      els.scoreInput.value = review.score ?? "";
      els.notesInput.value = review.notes || "";
      els.reviewHint.textContent = `Reviewing ${item.case_id || "benchmark"} on ${item.model_name || "unknown model"}`;
      renderBenchmarks(items);
    });
    els.benchmarks.appendChild(card);
  }
}

function renderTrainingList(items) {
  els.trainingRecords.innerHTML = "";
  els.trainingEmpty.style.display = items.length ? "none" : "block";
  for (const item of items) {
    const title = state.trainingMode === "llm"
      ? `${item.model || "unknown model"} - ${item.source || "unknown source"}`
      : String(item.transcript_path || "").split(/[\\/]/).slice(-1)[0];
    const preview = state.trainingMode === "llm"
      ? (item.response_text || item.error_text || "").slice(0, 360)
      : (item.transcript_text || "").slice(0, 360);
    const card = document.createElement("article");
    card.className = `record-card selectable ${item.record_id === state.selectedTrainingId ? "selected" : ""}`;
    card.innerHTML = `
      <div class="record-top">
        <div class="chips">${chip(item.review_status || "pending")}${state.trainingMode === "llm" ? chip(item.model || "model") : chip("transcript")}</div>
        <div class="timestamp">${escapeHtml(formatDate(item.created_at))}</div>
      </div>
      <h3 class="record-title">${escapeHtml(title || "Training record")}</h3>
      <p class="record-text">${escapeHtml(preview || "No preview available.")}</p>
    `;
    card.addEventListener("click", async () => {
      state.selectedTrainingId = item.record_id;
      await loadTrainingRecord(item.record_id);
      renderTrainingList(items);
    });
    els.trainingRecords.appendChild(card);
  }
}

function renderTrainingViewer(record) {
  const review = record.review || {};
  els.trainingReviewerInput.value = review.reviewer || "local-user";
  els.trainingStatusInput.value = review.status || record.review_status || "pending";
  els.trainingNotesInput.value = review.notes || "";

  if (state.trainingMode === "llm") {
    els.trainingPrimaryText.value = review.corrected_response_text ?? record.response_text ?? "";
    els.trainingSecondaryText.value = review.corrected_reasoning_text ?? "";
    els.trainingSecondaryText.disabled = false;
    els.trainingSecondaryText.placeholder = "Corrected reasoning text, or corrected messages JSON if needed";
    els.trainingPrimaryHelp.textContent = "For LLM records: write the corrected final assistant response you want in the training dataset.";
    els.trainingSecondaryHelp.textContent = "Optional: write corrected reasoning only if this record should teach an internal reasoning trace. Usually leave blank.";
    els.trainingViewer.innerHTML = `
      ${record.image_path ? `<img class="preview-image" src="${mediaUrl(record.image_path)}" alt="Screenshot preview" title="Click to enlarge screenshot" />` : ""}
      <div class="meta-grid">
        <div><strong>Model</strong>${escapeHtml(record.model || "unknown")}</div>
        <div><strong>Source</strong>${escapeHtml(record.source || "unknown")}</div>
        <div><strong>Interaction</strong>${escapeHtml(record.interaction_id || "")}</div>
      </div>
      <details open><summary>Raw Response</summary><pre>${escapeHtml(record.response_text || "")}</pre></details>
      <details><summary>Messages JSON</summary><pre>${escapeHtml(prettyJson(record.messages || []))}</pre></details>
      <details><summary>Tools JSON</summary><pre>${escapeHtml(prettyJson(record.tools || null))}</pre></details>
      <details><summary>Tool Calls JSON</summary><pre>${escapeHtml(prettyJson(record.tool_calls || null))}</pre></details>
      <details><summary>Metadata JSON</summary><pre>${escapeHtml(prettyJson(record.metadata || {}))}</pre></details>
      <details><summary>Report JSON</summary><pre>${escapeHtml(prettyJson(record.report || null))}</pre></details>
    `;
    els.trainingHint.textContent = `Reviewing LLM record ${record.record_id}`;
  } else {
    const audioPath = record.cleaned_audio_path || record.upload_audio_path || "";
    els.trainingPrimaryText.value = review.corrected_transcript_text ?? record.transcript_text ?? "";
    els.trainingSecondaryText.value = "";
    els.trainingSecondaryText.disabled = true;
    els.trainingSecondaryText.placeholder = "Unused for ASR review";
    els.trainingPrimaryHelp.textContent = "For ASR records: write the corrected transcript exactly as it should appear in the ASR training manifest.";
    els.trainingSecondaryHelp.textContent = "ASR review does not use secondary correction.";
    els.trainingViewer.innerHTML = `
      ${audioPath ? `<audio controls src="${mediaUrl(audioPath)}"></audio>` : `<div class="hint">No audio file matched for this transcript yet.</div>`}
      <div class="meta-grid">
        <div><strong>Transcript</strong>${escapeHtml(record.transcript_path || "")}</div>
        <div><strong>Upload audio</strong>${escapeHtml(record.upload_audio_path || "none")}</div>
        <div><strong>Cleaned audio</strong>${escapeHtml(record.cleaned_audio_path || "none")}</div>
      </div>
      <details open><summary>Raw Transcript</summary><pre>${escapeHtml(record.transcript_text || "")}</pre></details>
      <details><summary>Metadata JSON</summary><pre>${escapeHtml(prettyJson(record.metadata || {}))}</pre></details>
    `;
    els.trainingHint.textContent = `Reviewing ASR record ${record.record_id}`;
  }
}

async function loadTrainingRecord(recordId) {
  const url = state.trainingMode === "llm" ? `/api/training/llm/${recordId}` : `/api/training/asr/${recordId}`;
  const payload = await getJson(url);
  state.selectedTrainingRecord = payload.record;
  if (!state.selectedTrainingRecord) {
    els.trainingHint.textContent = "Record no longer exists.";
    return;
  }
  renderTrainingViewer(state.selectedTrainingRecord);
}

els.trainingViewer.addEventListener("click", (event) => {
  const image = event.target.closest(".preview-image");
  if (!image) {
    return;
  }
  openImageLightbox(image.currentSrc || image.src);
});

async function pollReports() {
  const payload = await getJson("/api/reports?limit=50");
  const serialized = JSON.stringify({
    reports: payload.reports || [],
    queued_tasks: payload.queued_tasks || [],
  });
  if (serialized !== state.latestReportKey) {
    state.latestReportKey = serialized;
    renderReports(payload.reports || []);
    renderQueuedTasks(payload.queued_tasks || []);
  }
}

async function pollBenchmarks() {
  const payload = await getJson("/api/benchmarks/results?limit=100");
  const serialized = JSON.stringify(payload.results || []);
  if (serialized !== state.latestBenchmarkKey) {
    state.latestBenchmarkKey = serialized;
    renderBenchmarks(payload.results || []);
  }
}

async function pollTraining(force = false) {
  const params = {
    limit: 100,
    review_status: state.trainingFilters.reviewStatus,
  };
  if (state.trainingMode === "llm") {
    params.source = state.trainingFilters.source;
    params.model = state.trainingFilters.model;
  }
  const query = buildQuery(params);
  const endpoint = state.trainingMode === "llm" ? `/api/training/llm?${query}` : `/api/training/asr?${query}`;
  const [recordsPayload, exportsPayload] = await Promise.all([
    getJson(endpoint),
    getJson("/api/training/exports?limit=10"),
  ]);
  const serialized = JSON.stringify(recordsPayload.records || []);
  if (force || serialized !== state.latestTrainingKey) {
    state.latestTrainingKey = serialized;
    renderTrainingList(recordsPayload.records || []);
  }
  const exportText = (exportsPayload.exports || [])
    .map((item) => `${String(item.dataset_kind || "").toUpperCase()} - ${item.record_count} records - ${item.output_path}`)
    .join("\n");
  els.trainingExports.textContent = exportText || "No dataset exports yet.";
}

function appendLog(entry) {
  const line = document.createElement("div");
  line.className = "log-line";
  line.innerHTML = `
    <span class="level ${escapeHtml(entry.level || "")}">${escapeHtml(entry.level || "log")}</span>
    <span class="log-message">${escapeHtml(entry.message || "")}</span>
  `;
  els.logs.appendChild(line);
  state.logCount += 1;
  els.logsCount.textContent = String(state.logCount);
  const maxLogNodes = 600;
  while (els.logs.children.length > maxLogNodes) {
    els.logs.removeChild(els.logs.firstElementChild);
  }
}

function setTrainingMode(mode) {
  state.trainingMode = mode;
  clearSelectedTrainingRecord();
  state.latestTrainingKey = "";
  els.trainingFilterSourceWrap.style.display = mode === "llm" ? "" : "none";
  els.trainingFilterModelWrap.style.display = mode === "llm" ? "" : "none";
  els.trainingSecondaryText.disabled = mode !== "llm";
  els.modeLlm.classList.toggle("active", mode === "llm");
  els.modeAsr.classList.toggle("active", mode === "asr");
  pollTraining(true);
}

function bindEvents() {
  els.tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const key = tab.dataset.tab;
      els.tabs.forEach((item) => item.classList.toggle("active", item === tab));
      Object.entries(els.sections).forEach(([name, section]) => {
        section.classList.toggle("active", name === key);
      });
    });
  });

  els.modeLlm.addEventListener("click", () => setTrainingMode("llm"));
  els.modeAsr.addEventListener("click", () => setTrainingMode("asr"));

  els.applyTrainingFiltersButton.addEventListener("click", async () => {
    syncTrainingFilterStateFromInputs();
    clearSelectedTrainingRecord();
    state.latestTrainingKey = "";
    await pollTraining(true);
  });

  els.clearTrainingFiltersButton.addEventListener("click", async () => {
    els.trainingFilterStatus.value = "";
    els.trainingFilterSource.value = "";
    els.trainingFilterModel.value = "";
    syncTrainingFilterStateFromInputs();
    clearSelectedTrainingRecord();
    state.latestTrainingKey = "";
    await pollTraining(true);
  });

  els.imageLightboxClose.addEventListener("click", closeImageLightbox);
  els.imageLightbox.addEventListener("click", (event) => {
    if (event.target.matches("[data-lightbox-close]")) {
      closeImageLightbox();
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && els.imageLightbox.classList.contains("open")) {
      closeImageLightbox();
    }
  });

  [els.trainingFilterStatus, els.trainingFilterSource, els.trainingFilterModel].forEach((control) => {
    control.addEventListener("keydown", async (event) => {
      if (event.key !== "Enter") {
        return;
      }
      event.preventDefault();
      syncTrainingFilterStateFromInputs();
      clearSelectedTrainingRecord();
      state.latestTrainingKey = "";
      await pollTraining(true);
    });
  });

  els.saveReviewButton.addEventListener("click", async () => {
    if (!state.selectedBenchmarkId) {
      els.reviewHint.textContent = "Select a benchmark result before saving a review.";
      return;
    }
    const payload = await fetch(`/api/benchmarks/results/${state.selectedBenchmarkId}/review`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        reviewer: els.reviewerInput.value,
        score: els.scoreInput.value,
        notes: els.notesInput.value,
      }),
    }).then((response) => response.json());
    els.reviewHint.textContent = payload.ok ? "Manual review saved." : "Failed to save manual review.";
    await pollBenchmarks();
  });

  els.saveTrainingReviewButton.addEventListener("click", async () => {
    if (!state.selectedTrainingId || !state.selectedTrainingRecord) {
      els.trainingHint.textContent = "Select a training record before saving.";
      return;
    }
    const body = {
      reviewer: els.trainingReviewerInput.value,
      status: els.trainingStatusInput.value,
      notes: els.trainingNotesInput.value,
    };
    const endpoint = state.trainingMode === "llm"
      ? `/api/training/llm/${state.selectedTrainingId}/review`
      : `/api/training/asr/${state.selectedTrainingId}/review`;
    if (state.trainingMode === "llm") {
      body.corrected_response_text = els.trainingPrimaryText.value;
      body.corrected_reasoning_text = els.trainingSecondaryText.value;
    } else {
      body.corrected_transcript_text = els.trainingPrimaryText.value;
    }
    const payload = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((response) => response.json());
    if (payload.ok) {
      els.trainingHint.textContent = "Training review saved.";
      await pollTraining(true);
      await loadTrainingRecord(state.selectedTrainingId);
    } else {
      els.trainingHint.textContent = "Failed to save training review.";
    }
  });

  els.syncTrainingButton.addEventListener("click", async () => {
    const endpoint = state.trainingMode === "llm" ? "/api/training/sync/llm" : "/api/training/sync/asr";
    const payload = await fetch(endpoint, { method: "POST" }).then((response) => response.json());
    els.trainingHint.textContent = payload.ok
      ? `Synced ${payload.synced || 0} ${state.trainingMode.toUpperCase()} records.`
      : "Sync failed.";
    await pollTraining(true);
  });

  els.exportTrainingButton.addEventListener("click", async () => {
    const endpoint = state.trainingMode === "llm" ? "/api/training/export/llm" : "/api/training/export/asr";
    const payload = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ statuses: ["approved"] }),
    }).then((response) => response.json());
    els.trainingHint.textContent = payload.ok
      ? `Exported ${payload.record_count || 0} approved ${state.trainingMode.toUpperCase()} records.`
      : "Export failed.";
    await pollTraining(true);
  });
}

async function poll() {
  try {
    const [logPayload] = await Promise.all([
      getJson(`/api/logs?after_id=${state.latestId}&limit=500`),
      pollReports(),
      pollBenchmarks(),
      pollTraining(),
    ]);
    for (const entry of logPayload.entries || []) {
      appendLog(entry);
      state.latestId = Math.max(state.latestId, entry.id);
    }
    els.status.textContent = `dashboard live - latest log id ${logPayload.latest_id}`;
  } catch (error) {
    els.status.textContent = "disconnected";
  }
}

bindEvents();
setTrainingMode("llm");
poll();
setInterval(poll, 2000);
