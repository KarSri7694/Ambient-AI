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
  chatToken: sessionStorage.getItem("ambientChatToken") || "",
  chatSessions: [],
  selectedChatId: "",
  chatMessageKey: "",
  activeChatStreams: new Set(),
};

const els = {
  status: document.getElementById("status"),
  tabs: Array.from(document.querySelectorAll(".tab")),
  sections: {
    chat: document.getElementById("chatSection"),
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
  chatAuthPanel: document.getElementById("chatAuthPanel"),
  chatTokenInput: document.getElementById("chatTokenInput"),
  saveChatTokenButton: document.getElementById("saveChatTokenButton"),
  newChatButton: document.getElementById("newChatButton"),
  renameChatButton: document.getElementById("renameChatButton"),
  chatSessions: document.getElementById("chatSessions"),
  chatSessionsEmpty: document.getElementById("chatSessionsEmpty"),
  chatTitle: document.getElementById("chatTitle"),
  chatMessages: document.getElementById("chatMessages"),
  chatWelcome: document.getElementById("chatWelcome"),
  chatActivity: document.getElementById("chatActivity"),
  chatComposer: document.getElementById("chatComposer"),
  chatInput: document.getElementById("chatInput"),
  sendChatButton: document.getElementById("sendChatButton"),
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
  const response = await apiFetch(url, { cache: "no-store" });
  return response.json();
}

function authHeaders(extra = {}) {
  const headers = { ...extra };
  if (state.chatToken) headers.Authorization = `Bearer ${state.chatToken}`;
  return headers;
}

async function apiFetch(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    headers: authHeaders(options.headers || {}),
  });
  if (response.status === 401) {
    els.chatAuthPanel.classList.add("needs-token");
    els.status.textContent = "access token required";
    throw new Error("Invalid or missing access token");
  }
  return response;
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
      ${item.run_at_utc ? `<div class="meta-grid"><div><strong>Runs</strong>${escapeHtml(formatDate(item.run_at_utc))}</div></div>
      <button class="button" type="button" data-cancel-task="${escapeHtml(item.id)}">Cancel scheduled task</button>` : ""}
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

function chatWelcomeMarkup() {
  return `
    <div class="chat-welcome">
      <div class="welcome-mark">A</div>
      <h3>Ask, act, or schedule.</h3>
      <p>Ask about local information, give Ambient AI something to do now, or describe when it should run later.</p>
      <div class="chat-examples">
        <button type="button" data-chat-example="Summarize the latest local runtime reports.">Summarize local reports</button>
        <button type="button" data-chat-example="Open the project README and tell me the most important setup steps.">Inspect a local file</button>
        <button type="button" data-chat-example="Tomorrow at 9:00 AM, check the queued tasks and report what is still pending.">Schedule a task</button>
      </div>
    </div>`;
}

function messageContent(message) {
  if (message.status === "failed") return message.error_text || message.content || "Response failed.";
  if (message.content) return message.content;
  if (message.status === "queued") return "Queued…";
  if (message.status === "running") return "Thinking…";
  return message.error_text || "No response returned.";
}

function chatMessageElement(message) {
  const article = document.createElement("article");
  article.className = `chat-message ${message.role} ${message.status || ""}`;
  article.dataset.messageId = message.id;
  article.innerHTML = `
    <div class="chat-bubble">${escapeHtml(messageContent(message))}</div>
    <div class="chat-message-meta">
      <span>${escapeHtml(message.role === "user" ? "You" : "Ambient AI")}</span>
      <span>${escapeHtml(message.status || "completed")}</span>
      ${message.message_kind === "scheduled_result" ? "<span>scheduled result</span>" : ""}
      <span>${escapeHtml(formatDate(message.created_at))}</span>
    </div>`;
  return article;
}

function updateChatMessage(message) {
  if (!message || !message.id) return;
  const existing = els.chatMessages.querySelector(`[data-message-id="${CSS.escape(message.id)}"]`);
  const replacement = chatMessageElement(message);
  if (existing) existing.replaceWith(replacement);
  else els.chatMessages.appendChild(replacement);
  els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
}

function renderChatMessages(messages) {
  els.chatMessages.innerHTML = "";
  if (!messages.length) {
    els.chatMessages.innerHTML = chatWelcomeMarkup();
    return;
  }
  for (const message of messages) {
    els.chatMessages.appendChild(chatMessageElement(message));
    if (message.role === "assistant" && ["queued", "running"].includes(message.status)) {
      streamChatMessage(message.id);
    }
  }
  els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
}

function renderChatSessions() {
  els.chatSessions.innerHTML = "";
  els.chatSessionsEmpty.style.display = state.chatSessions.length ? "none" : "block";
  for (const session of state.chatSessions) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `chat-session ${session.id === state.selectedChatId ? "active" : ""}`;
    button.innerHTML = `
      <strong>${escapeHtml(session.title)}</strong>
      <small>${escapeHtml(session.preview || "No messages yet")}</small>`;
    button.addEventListener("click", () => selectChatSession(session.id));
    els.chatSessions.appendChild(button);
  }
}

async function loadChatSessions() {
  const payload = await getJson("/api/chat/sessions?limit=100");
  state.chatSessions = payload.sessions || [];
  renderChatSessions();
  if (!state.selectedChatId && state.chatSessions.length) {
    await selectChatSession(state.chatSessions[0].id);
  }
}

async function createChatSession() {
  const response = await apiFetch("/api/chat/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: "New conversation" }),
  });
  const payload = await response.json();
  await loadChatSessions();
  await selectChatSession(payload.session.id);
  els.chatInput.focus();
  return payload.session;
}

async function selectChatSession(sessionId, force = false) {
  if (!force && state.selectedChatId === sessionId && state.chatMessageKey) return;
  const payload = await getJson(`/api/chat/sessions/${sessionId}/messages?limit=500`);
  state.selectedChatId = sessionId;
  state.chatMessageKey = JSON.stringify(payload.messages || []);
  els.chatTitle.textContent = payload.session.title;
  els.renameChatButton.disabled = false;
  renderChatSessions();
  renderChatMessages(payload.messages || []);
  updateComposerState(payload.messages || []);
}

function updateComposerState(messages = []) {
  const hasActive = messages.some(
    (message) => message.role === "assistant" && ["queued", "running"].includes(message.status),
  );
  els.sendChatButton.disabled = hasActive;
  els.chatInput.disabled = hasActive;
  if (hasActive) els.chatActivity.textContent = "Ambient AI is working on this conversation…";
  else if (!state.activeChatStreams.size) els.chatActivity.textContent = "";
}

function parseSseBlock(block) {
  let eventType = "message";
  const dataLines = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) eventType = line.slice(6).trim();
    if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  if (!dataLines.length) return null;
  try {
    return { type: eventType, data: JSON.parse(dataLines.join("\n")) };
  } catch {
    return null;
  }
}

function handleChatStreamEvent(messageId, eventType, data) {
  if (eventType === "snapshot") {
    const message = data.message || data;
    updateChatMessage(message);
  } else if (eventType === "delta") {
    const existing = els.chatMessages.querySelector(`[data-message-id="${CSS.escape(messageId)}"]`);
    if (existing) {
      const bubble = existing.querySelector(".chat-bubble");
      if (["Thinking…", "Queued…"].includes(bubble.textContent)) bubble.textContent = "";
      bubble.textContent += data.content || "";
      els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
    }
  } else if (eventType === "tool_started") {
    els.chatActivity.textContent = `Using ${data.tool_name || "a tool"}…`;
  } else if (eventType === "tool_finished") {
    els.chatActivity.textContent = `${data.tool_name || "Tool"} ${data.ok ? "finished" : "failed"}.`;
  } else if (eventType === "status") {
    els.chatActivity.textContent = data.status === "running" ? "Ambient AI is thinking…" : data.status;
  } else if (eventType === "done") {
    updateChatMessage(data.message || data);
    els.chatActivity.textContent = "";
  } else if (eventType === "error") {
    const existing = els.chatMessages.querySelector(`[data-message-id="${CSS.escape(messageId)}"]`);
    if (existing) {
      existing.classList.add("failed");
      existing.querySelector(".chat-bubble").textContent = data.error_text || data.error || "Response failed.";
    }
    els.chatActivity.textContent = "The response failed. You can send the request again.";
  }
}

async function streamChatMessage(messageId) {
  if (state.activeChatStreams.has(messageId)) return;
  state.activeChatStreams.add(messageId);
  try {
    const response = await apiFetch(`/api/chat/messages/${messageId}/events`, {
      headers: { Accept: "text/event-stream" },
      cache: "no-store",
    });
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replaceAll("\r\n", "\n");
      let separator = buffer.indexOf("\n\n");
      while (separator >= 0) {
        const parsed = parseSseBlock(buffer.slice(0, separator));
        buffer = buffer.slice(separator + 2);
        if (parsed) handleChatStreamEvent(messageId, parsed.type, parsed.data);
        separator = buffer.indexOf("\n\n");
      }
    }
  } catch (error) {
    els.chatActivity.textContent = "Live connection interrupted; saved messages will continue processing.";
  } finally {
    state.activeChatStreams.delete(messageId);
    if (state.selectedChatId) await selectChatSession(state.selectedChatId, true);
    await loadChatSessions();
  }
}

async function submitChatMessage() {
  const content = els.chatInput.value.trim();
  if (!content) return;
  if (!state.selectedChatId) await createChatSession();
  els.sendChatButton.disabled = true;
  els.chatInput.disabled = true;
  try {
    const response = await apiFetch(`/api/chat/sessions/${state.selectedChatId}/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content }),
    });
    const payload = await response.json();
    els.chatInput.value = "";
    updateChatMessage(payload.user_message);
    updateChatMessage(payload.assistant_message);
    els.chatActivity.textContent = "Queued for Ambient AI…";
    await loadChatSessions();
    streamChatMessage(payload.assistant_message.id);
  } catch (error) {
    els.chatActivity.textContent = error.message || "Could not send the message.";
    els.sendChatButton.disabled = false;
    els.chatInput.disabled = false;
  }
}

async function pollChat() {
  await loadChatSessions();
  if (!state.selectedChatId || state.activeChatStreams.size) return;
  const payload = await getJson(`/api/chat/sessions/${state.selectedChatId}/messages?limit=500`);
  const serialized = JSON.stringify(payload.messages || []);
  if (serialized !== state.chatMessageKey) {
    state.chatMessageKey = serialized;
    renderChatMessages(payload.messages || []);
    updateComposerState(payload.messages || []);
  }
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

  els.saveChatTokenButton.addEventListener("click", async () => {
    state.chatToken = els.chatTokenInput.value.trim();
    sessionStorage.setItem("ambientChatToken", state.chatToken);
    document.cookie = `ambient_auth=${encodeURIComponent(state.chatToken)}; Path=/; SameSite=Strict`;
    els.chatAuthPanel.classList.remove("needs-token");
    try {
      await pollChat();
      els.status.textContent = "access granted";
    } catch (error) {
      els.status.textContent = "invalid access token";
    }
  });

  els.newChatButton.addEventListener("click", createChatSession);
  els.renameChatButton.addEventListener("click", async () => {
    const selected = state.chatSessions.find((session) => session.id === state.selectedChatId);
    if (!selected) return;
    const title = window.prompt("Conversation name", selected.title);
    if (!title || !title.trim()) return;
    await apiFetch(`/api/chat/sessions/${selected.id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: title.trim() }),
    });
    await loadChatSessions();
    els.chatTitle.textContent = title.trim();
  });

  els.chatComposer.addEventListener("submit", async (event) => {
    event.preventDefault();
    await submitChatMessage();
  });
  els.chatInput.addEventListener("keydown", async (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      await submitChatMessage();
    }
  });
  els.chatMessages.addEventListener("click", (event) => {
    const example = event.target.closest("[data-chat-example]");
    if (!example) return;
    els.chatInput.value = example.dataset.chatExample;
    els.chatInput.focus();
  });
  els.queuedTasks.addEventListener("click", async (event) => {
    const button = event.target.closest("[data-cancel-task]");
    if (!button) return;
    await apiFetch(`/api/chat/scheduled/${button.dataset.cancelTask}/cancel`, { method: "POST" });
    await pollReports();
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
    const payload = await apiFetch(`/api/benchmarks/results/${state.selectedBenchmarkId}/review`, {
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
    const payload = await apiFetch(endpoint, {
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
    const payload = await apiFetch(endpoint, { method: "POST" }).then((response) => response.json());
    els.trainingHint.textContent = payload.ok
      ? `Synced ${payload.synced || 0} ${state.trainingMode.toUpperCase()} records.`
      : "Sync failed.";
    await pollTraining(true);
  });

  els.exportTrainingButton.addEventListener("click", async () => {
    const endpoint = state.trainingMode === "llm" ? "/api/training/export/llm" : "/api/training/export/asr";
    const payload = await apiFetch(endpoint, {
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
      pollChat(),
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
els.chatTokenInput.value = state.chatToken;
if (state.chatToken) {
  document.cookie = `ambient_auth=${encodeURIComponent(state.chatToken)}; Path=/; SameSite=Strict`;
}
setTrainingMode("llm");
poll();
setInterval(poll, 2000);
