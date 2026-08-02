import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Activity, AlertTriangle, ArrowLeft, ArrowRight, AudioLines, Check, ChevronRight, Clock3,
  Cpu, Download, FileAudio, FileImage, Gauge, GripVertical, ImageIcon, Layers3, Play,
  Save, Settings2, ShieldAlert, Sparkles, Square, UploadCloud, Wrench, X,
} from "lucide-react";
import { getJson, sendJson, uploadBinary } from "../api";
import {
  Badge, Button, EmptyState, ErrorState, formatDate, JsonDetails, LoadingState,
  Markdown, PageHeader,
} from "../components/ui";

type Uploaded = { media_id: string; kind: "image" | "audio"; original_name: string };
type SourceMode = "suite" | "upload" | "agent";
type TraceFilter = "all" | "model" | "tools" | "pipeline";
type AgentTaskDraft = {
  id: string;
  agent_kind: "browser" | "computer";
  title: string;
  instruction: string;
  start_url: string;
  read_only: boolean;
  max_steps: string;
  timeout_seconds: string;
  success_criteria: string;
};

const SPEEDS = [1, 2, 5, 10];
const ACCEPTED_MEDIA = {
  image: ".png,.jpg,.jpeg,.webp,image/png,image/jpeg,image/webp",
  audio: ".wav,.mp3,.m4a,.opus,.flac,audio/wav,audio/mpeg,audio/mp4,audio/ogg,audio/flac",
} as const;
const SCORE_FIELDS = [
  ["perception_score", "Perception"],
  ["decision_score", "Decision"],
  ["tool_choice_score", "Tool choice"],
  ["tool_execution_score", "Execution"],
  ["final_response_score", "Final response"],
  ["overall_score", "Overall"],
] as const;

export function RealWorldTestsPage() {
  const client = useQueryClient();
  const suites = useQuery({ queryKey: ["rw-suites"], queryFn: () => getJson<any>("/api/real-world/suites") });
  const models = useQuery({ queryKey: ["rw-models"], queryFn: () => getJson<any>("/api/real-world/models"), retry: false });
  const runs = useQuery({ queryKey: ["rw-runs"], queryFn: () => getJson<any>("/api/real-world/runs"), refetchInterval: 1500 });
  const rocm = useQuery({ queryKey: ["rocm-tuning-status"], queryFn: () => getJson<any>("/api/rocm-tuning/status"), refetchInterval: 5000 });
  const [sourceMode, setSourceMode] = useState<SourceMode>("suite");
  const [suiteId, setSuiteId] = useState("");
  const [scenarioIds, setScenarioIds] = useState<string[]>([]);
  const [speed, setSpeed] = useState("1");
  const [confirmation, setConfirmation] = useState("");
  const [overrides, setOverrides] = useState<Record<string, string>>({});
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [uploads, setUploads] = useState<Uploaded[]>([]);
  const [uploadKind, setUploadKind] = useState<"image" | "audio">("image");
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [dropzoneActive, setDropzoneActive] = useState(false);
  const [draggedUploadId, setDraggedUploadId] = useState<string | null>(null);
  const [traceFilter, setTraceFilter] = useState<TraceFilter>("all");
  const [agentTasks, setAgentTasks] = useState<AgentTaskDraft[]>([
    defaultAgentTask("browser", 1),
    defaultAgentTask("computer", 2),
  ]);
  const selectedSuite = useMemo(
    () => suites.data?.suites?.find((item: any) => item.suite_id === suiteId),
    [suites.data, suiteId],
  );

  useEffect(() => {
    if (!suiteId && suites.data?.suites?.length) setSuiteId(suites.data.suites[0].suite_id);
  }, [suites.data, suiteId]);
  useEffect(() => {
    if (selectedSuite) setScenarioIds(selectedSuite.scenarios.map((item: any) => item.scenario_id));
  }, [selectedSuite?.suite_id]);
  useEffect(() => {
    if (models.data?.roles && !Object.keys(overrides).length) setOverrides(models.data.roles);
  }, [models.data]);

  const detail = useQuery({
    queryKey: ["rw-run", selectedRun], enabled: Boolean(selectedRun),
    queryFn: () => getJson<any>(`/api/real-world/runs/${selectedRun}`), refetchInterval: 1000,
  });
  const trace = useQuery({
    queryKey: ["rw-trace", selectedRun], enabled: Boolean(selectedRun),
    queryFn: () => getJson<any>(`/api/real-world/runs/${selectedRun}/trace?limit=5000`), refetchInterval: 1000,
  });
  const start = useMutation({
    mutationFn: () => sendJson<any>("/api/real-world/runs", "POST", sourceMode === "agent" ? {
      playback_speed: Number(speed), model_overrides: overrides, live_tools_confirmation: confirmation,
      inline_scenario: {
        title: "Live browser and computer agent tasks",
        modality: "agent_task_sequence",
        tasks: agentTasks.filter((item) => item.instruction.trim()).map((item, index) => ({
          task_id: item.id || `task_${index + 1}`,
          agent_kind: item.agent_kind,
          title: item.title || `${item.agent_kind === "browser" ? "Browser" : "Computer"} task ${index + 1}`,
          instruction: item.instruction,
          start_url: item.start_url,
          read_only: item.read_only,
          max_steps: item.max_steps ? Number(item.max_steps) : undefined,
          timeout_seconds: item.timeout_seconds ? Number(item.timeout_seconds) : undefined,
          success_criteria: item.success_criteria,
        })),
      },
    } : sourceMode === "upload" ? {
      playback_speed: Number(speed), model_overrides: overrides, live_tools_confirmation: confirmation,
      inline_scenario: {
        title: "Uploaded media scenario",
        modality: uploadKind === "image" ? "image_sequence" : "audio_sequence",
        events: uploads.map((item, index) => ({ media_id: item.media_id, offset_seconds: index * 10 })),
      },
    } : {
      suite_id: suiteId, scenario_ids: scenarioIds, playback_speed: Number(speed),
      model_overrides: overrides, live_tools_confirmation: confirmation,
    }),
    onSuccess: (payload) => {
      setSelectedRun(payload.run.run_id);
      setConfirmation("");
      client.invalidateQueries({ queryKey: ["rw-runs"] });
    },
  });
  const cancel = useMutation({ mutationFn: (id: string) => sendJson(`/api/real-world/runs/${id}/cancel`, "POST") });

  const upload = async (files: FileList | null) => {
    if (!files?.length) return;
    if (suites.data?.available !== true) {
      setUploadError("The real-world test lab is not running. Stop the normal Ambient AI runtime, start tests/real_world_tests/run_lab.py, and open http://127.0.0.1:8766/real-world-tests.");
      return;
    }
    setUploadError("");
    setUploading(true);
    try {
      const results = await Promise.allSettled(Array.from(files).map(async (file) => {
        const payload = await uploadBinary<{ media: Uploaded }>(
          `/api/real-world/uploads?filename=${encodeURIComponent(file.name)}&kind=${uploadKind}`,
          file,
        );
        return payload.media;
      }));
      const added = results
        .filter((result): result is PromiseFulfilledResult<Uploaded> => result.status === "fulfilled")
        .map((result) => result.value);
      setUploads((current) => [...current, ...added]);
      const failed = results.filter((result): result is PromiseRejectedResult => result.status === "rejected");
      if (failed.length) {
        const firstError = failed[0].reason;
        const message = firstError instanceof Error ? firstError.message : "Upload failed.";
        setUploadError(`${failed.length} of ${results.length} file${failed.length === 1 ? "" : "s"} failed to upload: ${message}`);
      }
    } finally {
      setUploading(false);
    }
  };

  const moveUpload = (mediaId: string, direction: -1 | 1) => {
    setUploads((current) => {
      const fromIndex = current.findIndex((item) => item.media_id === mediaId);
      const toIndex = fromIndex + direction;
      if (fromIndex < 0 || toIndex < 0 || toIndex >= current.length) return current;
      const next = [...current];
      const [moved] = next.splice(fromIndex, 1);
      next.splice(toIndex, 0, moved);
      return next;
    });
  };

  const dragUploadTo = (targetId: string) => {
    if (!draggedUploadId || draggedUploadId === targetId) return;
    setUploads((current) => {
      const fromIndex = current.findIndex((item) => item.media_id === draggedUploadId);
      const toIndex = current.findIndex((item) => item.media_id === targetId);
      if (fromIndex < 0 || toIndex < 0) return current;
      const next = [...current];
      const [moved] = next.splice(fromIndex, 1);
      next.splice(toIndex, 0, moved);
      return next;
    });
  };

  const updateAgentTask = (id: string, patch: Partial<AgentTaskDraft>) => {
    setAgentTasks((current) => current.map((item) => item.id === id ? { ...item, ...patch } : item));
  };

  const addAgentTask = (kind: "browser" | "computer") => {
    setAgentTasks((current) => [...current, defaultAgentTask(kind, current.length + 1)]);
  };

  const run = detail.data?.run;
  const events = trace.data?.events || [];
  const active = run && ["queued", "running"].includes(run.status);
  const runItems = runs.data?.runs || [];
  const completedCount = runItems.filter((item: any) => item.status === "completed").length;
  const hardware = run?.config?.accelerator || events.find((event: any) => event.event_type === "accelerator_detected")?.payload;
  const tunedProfile = rocm.data?.latest_profile;
  const modelEvents = events.filter((event: any) => event.stage === "model");
  const toolEvents = events.filter((event: any) => event.event_type.includes("tool"));
  const failedEvents = events.filter((event: any) => event.status === "failed");
  const avgModelMs = modelEvents.length ? Math.round(modelEvents.reduce((total: number, event: any) => total + (Number(event.duration_ms) || 0), 0) / modelEvents.length) : null;
  const filteredEvents = events.filter((event: any) => {
    if (traceFilter === "model") return event.stage === "model";
    if (traceFilter === "tools") return event.event_type.includes("tool");
    if (traceFilter === "pipeline") return !["model", "agent"].includes(event.stage);
    return true;
  });
  const validSource = sourceMode === "agent"
    ? agentTasks.some((item) => item.instruction.trim())
    : sourceMode === "upload" ? uploads.length > 0 : scenarioIds.length > 0;
  const armed = confirmation === "RUN LIVE TOOLS";
  const labAvailable = suites.data?.available === true;

  return <div className="rw-page">
    <PageHeader
      eyebrow="Evaluation studio"
      title="Real-world tests"
      description="Replay authentic media through the complete ambient pipeline, then inspect every perception, model, and tool decision in one trace."
      actions={<><Badge tone={active ? "warn" : labAvailable ? "good" : suites.isLoading ? "neutral" : "danger"}>{active ? "Run in progress" : labAvailable ? "Lab ready" : suites.isLoading ? "Checking lab" : "Lab offline"}</Badge><Badge>{tunedProfile ? "ROCm tuned" : "ROCm untuned"}</Badge><Button variant="secondary" onClick={() => downloadRocm("json")}><Download size={14} />Tuning JSON</Button><Button variant="secondary" onClick={() => downloadRocm("csv")}><Download size={14} />CSV</Button></>}
    />

    <section className="rw-hero">
      <div className="rw-hero-copy">
        <div className="rw-hero-icon"><Sparkles size={22} /></div>
        <div><p className="eyebrow">Production-faithful replay</p><h2>Build confidence before the model meets live context.</h2><p>Use recorded screenshots or conversations to evaluate perception, reasoning, and tool behavior with complete auditability.</p></div>
      </div>
      <div className="rw-stat-grid">
        <Metric icon={<Layers3 size={17} />} value={suites.data?.suites?.length || 0} label="Suites" />
        <Metric icon={<Activity size={17} />} value={runItems.length} label="Runs" />
        <Metric icon={<Check size={17} />} value={completedCount} label="Completed" />
      </div>
    </section>

    {!labAvailable && !suites.isLoading && <div className="panel mb-5 p-4"><EmptyState title="Start the evaluation lab" description="The normal runtime cannot execute isolated real-world tests. Stop python src/app.py, run python tests/real_world_tests/run_lab.py, then open http://127.0.0.1:8766/real-world-tests." /></div>}

    <section className="rw-launch-card">
      <div className="rw-section-head">
        <div><p className="rw-step">01 · Configure replay</p><h2>Choose what the agent should experience</h2><p>Select a reusable suite, assemble media, or run live browser/computer agent tasks.</p></div>
        <div className="segmented" aria-label="Input source">
          <button type="button" className={sourceMode === "suite" ? "active" : ""} onClick={() => setSourceMode("suite")}><Layers3 size={15} />Suite</button>
          <button type="button" className={sourceMode === "upload" ? "active" : ""} onClick={() => setSourceMode("upload")}><UploadCloud size={15} />Upload</button>
          <button type="button" className={sourceMode === "agent" ? "active" : ""} onClick={() => setSourceMode("agent")}><Wrench size={15} />Agent tasks</button>
        </div>
      </div>

      <div className="rw-config-grid">
        <div className="rw-config-main">
          {sourceMode === "suite" ? <>
            <label className="rw-field">Evaluation suite<select value={suiteId} onChange={(event) => setSuiteId(event.target.value)}>{suites.data?.suites?.map((item: any) => <option key={item.suite_id} value={item.suite_id}>{item.title}</option>)}</select></label>
            {selectedSuite && <div className="rw-scenario-list">
              {selectedSuite.scenarios.map((item: any) => {
                const selected = scenarioIds.includes(item.scenario_id);
                return <button type="button" key={item.scenario_id} className={`rw-scenario ${selected ? "selected" : ""}`} onClick={() => setScenarioIds((current) => selected ? current.filter((id) => id !== item.scenario_id) : [...current, item.scenario_id])}>
                  <span className="rw-check">{selected && <Check size={14} />}</span>
                  <span className="rw-scenario-icon">{item.modality === "agent_task_sequence" ? <Wrench size={19} /> : item.modality === "audio_sequence" ? <AudioLines size={19} /> : <ImageIcon size={19} />}</span>
                  <span className="min-w-0 flex-1 text-left"><strong>{item.title}</strong><small>{scenarioCountLabel(item)}</small></span>
                  <ChevronRight size={17} className="text-muted" />
                </button>;
              })}
            </div>}
          </> : sourceMode === "agent" ? <AgentTaskEditor
            tasks={agentTasks}
            updateTask={updateAgentTask}
            addTask={addAgentTask}
            removeTask={(id) => setAgentTasks((current) => current.filter((item) => item.id !== id))}
          /> : <>
            <div className="segmented mb-3"><button type="button" className={uploadKind === "image" ? "active" : ""} onClick={() => { setUploadKind("image"); setUploads([]); }}><FileImage size={15} />Images</button><button type="button" className={uploadKind === "audio" ? "active" : ""} onClick={() => { setUploadKind("audio"); setUploads([]); }}><FileAudio size={15} />Audio</button></div>
            <label
              className={`rw-dropzone ${dropzoneActive ? "active" : ""} ${!labAvailable ? "unavailable" : ""}`}
              onDragEnter={(event) => { event.preventDefault(); setDropzoneActive(true); }}
              onDragOver={(event) => { event.preventDefault(); event.dataTransfer.dropEffect = "copy"; }}
              onDragLeave={(event) => { if (!event.currentTarget.contains(event.relatedTarget as Node)) setDropzoneActive(false); }}
              onDrop={(event) => { event.preventDefault(); setDropzoneActive(false); void upload(event.dataTransfer.files); }}
            >
              <input className="sr-only" type="file" multiple disabled={!labAvailable || uploading} accept={ACCEPTED_MEDIA[uploadKind]} onChange={(event) => { void upload(event.target.files); event.target.value = ""; }} />
              <span className="rw-drop-icon"><UploadCloud size={24} /></span>
              <strong>{uploading ? "Uploading media…" : suites.isLoading ? "Checking lab availability…" : !labAvailable ? "Real-world test lab is offline" : `Drop or choose ${uploadKind} files`}</strong>
              <span>{suites.isLoading ? "Uploads enable when the isolated backend responds" : labAvailable ? "Files stay in the local evaluation workspace" : "Start the standalone lab on port 8766 to enable uploads"}</span>
            </label>
            {uploadError && <div className="rw-inline-error mt-3" role="alert">{uploadError}</div>}
            {uploads.length > 0 && <div className="rw-sequence-editor">
              <div className="rw-sequence-head">
                <div><strong>Ambient AI input order</strong><span>Drag to edit the sequence. Items are sent from left to right.</span></div>
                <Badge>{uploads.length} {uploadKind === "image" ? "frames" : "clips"}</Badge>
              </div>
              <div className="rw-upload-list" role="list" aria-label="Ambient AI input order">
                {uploads.map((item, index) => <div
                  className={`rw-upload-item ${draggedUploadId === item.media_id ? "dragging" : ""}`}
                  key={item.media_id}
                  role="listitem"
                  draggable={uploads.length > 1}
                  onDragStart={(event) => {
                    setDraggedUploadId(item.media_id);
                    event.dataTransfer.effectAllowed = "move";
                    event.dataTransfer.setData("text/plain", item.media_id);
                  }}
                  onDragEnter={(event) => { event.preventDefault(); dragUploadTo(item.media_id); }}
                  onDragOver={(event) => { event.preventDefault(); event.dataTransfer.dropEffect = "move"; }}
                  onDrop={(event) => { event.preventDefault(); setDraggedUploadId(null); }}
                  onDragEnd={() => setDraggedUploadId(null)}
                >
                  <div className="rw-upload-preview">
                    {item.kind === "image"
                      ? <img src={`/api/real-world/media/${item.media_id}`} alt={`Input ${index + 1}: ${item.original_name}`} />
                      : <FileAudio size={28} />}
                    <span className="rw-drag-handle" aria-hidden="true"><GripVertical size={17} /></span>
                    <span className="rw-sequence-number">{index + 1}</span>
                  </div>
                  <div className="rw-upload-copy"><strong title={item.original_name}>{item.original_name}</strong><span>{index === 0 ? "First input" : `After ${index * 10} seconds`}</span></div>
                  <div className="rw-upload-actions">
                    <button type="button" disabled={index === 0} aria-label={`Move ${item.original_name} earlier`} title="Move earlier" onClick={() => moveUpload(item.media_id, -1)}><ArrowLeft size={15} /></button>
                    <button type="button" disabled={index === uploads.length - 1} aria-label={`Move ${item.original_name} later`} title="Move later" onClick={() => moveUpload(item.media_id, 1)}><ArrowRight size={15} /></button>
                    <button className="remove" type="button" aria-label={`Remove ${item.original_name}`} title="Remove" onClick={() => setUploads((current) => current.filter((media) => media.media_id !== item.media_id))}><X size={15} /></button>
                  </div>
                </div>)}
              </div>
            </div>}
          </>}
        </div>

        <aside className="rw-playback-card">
          <div className="rw-card-title"><Gauge size={18} /><div><strong>Playback speed</strong><span>Preserve order, shorten waits</span></div></div>
          <div className="rw-speed-grid">{SPEEDS.map((item) => <button type="button" key={item} className={speed === String(item) ? "active" : ""} onClick={() => setSpeed(String(item))}>{item}×</button>)}</div>
          <div className="rw-playback-note"><Clock3 size={15} /><span>A 10 second interval replays in {(10 / Number(speed)).toFixed(Number(speed) === 1 ? 0 : 1)} seconds.</span></div>
        </aside>
      </div>

      {models.data?.roles && <details className="rw-model-settings"><summary><span><Settings2 size={17} /><strong>Task-specific models</strong><small>{Object.keys(overrides).length} roles configured</small></span><ChevronRight size={17} /></summary><div className="rw-model-grid">{Object.entries(models.data.roles).map(([role]) => <label className="rw-field" key={role}><span>{friendlyRole(role)}</span><select value={overrides[role] || ""} onChange={(event) => setOverrides((current) => ({ ...current, [role]: event.target.value }))}>{Array.from(new Set([overrides[role], ...(models.data.presets || [])])).filter(Boolean).map((model: any) => <option key={model} value={model}>{model}</option>)}</select></label>)}</div></details>}

      <div className="rw-arm-panel">
        <div className="rw-arm-copy"><span className="rw-danger-icon"><ShieldAlert size={20} /></span><div><strong>Arm live tool execution</strong><p>The production MCP surface can change external systems. Actions are audited and cannot be rolled back.</p></div></div>
        <div className="rw-arm-action"><label><span>Type RUN LIVE TOOLS</span><input value={confirmation} onChange={(event) => setConfirmation(event.target.value)} placeholder="RUN LIVE TOOLS" /></label><Button variant="primary" disabled={start.isPending || !labAvailable || !armed || !validSource} onClick={() => start.mutate()}><Play size={16} />{start.isPending ? "Starting…" : "Start evaluation"}</Button></div>
      </div>
      {start.isError && <div className="mt-4"><ErrorState error={start.error} /></div>}
    </section>

    <section className="rw-workspace">
      <div className="rw-history-panel">
        <div className="rw-section-head compact"><div><p className="rw-step">02 · Results</p><h2>Run history</h2></div><Badge>{runItems.length}</Badge></div>
        {runs.isLoading && <LoadingState />}
        {!runs.isLoading && !runItems.length && <EmptyState title="No evaluations yet" description="Configure your first replay above to begin building a result history." />}
        <div className="rw-run-list">{runItems.map((item: any) => <button type="button" key={item.run_id} className={`rw-run-row ${selectedRun === item.run_id ? "selected" : ""}`} onClick={() => setSelectedRun(item.run_id)}><span className={`rw-run-status ${item.status}`} /><span className="min-w-0 flex-1"><strong>{item.suite_id}</strong><small>{item.scenario_ids.join(", ") || "Uploaded scenario"}</small></span><span className="rw-run-meta"><Badge tone={statusTone(item.status)}>{humanStatus(item.status)}</Badge><time>{formatDate(item.created_at)}</time></span><ChevronRight size={17} /></button>)}</div>
      </div>

      <div className="rw-result-panel">
        {!run ? <EmptyState title="Select a run" description="Choose an evaluation to inspect its media, outputs, and manual review rubric." /> : <>
          <div className="rw-selected-head"><div><p className="rw-step">Selected evaluation</p><h2>{run.suite_id}</h2><p>{run.scenario_ids?.join(", ") || "Uploaded media scenario"}</p></div><div className="flex items-center gap-2"><Badge tone={statusTone(run.status)}>{humanStatus(run.status)}</Badge><Button variant="secondary" onClick={() => downloadRun(run.run_id, "json")}><Download size={14} />JSON</Button><Button variant="secondary" onClick={() => downloadRun(run.run_id, "csv")}><Download size={14} />CSV</Button>{active && <Button variant="danger" onClick={() => cancel.mutate(run.run_id)}><Square size={14} />Cancel</Button>}</div></div>
          <div className="rw-amd-summary">
            <div className="rw-amd-chip"><Cpu size={18} /><div><strong>{hardware?.gpu_name || "GPU not reported"}</strong><span>{hardware?.backend || "unknown"}{hardware?.runtime_version ? ` · ${hardware.runtime_version}` : ""}</span></div></div>
            <Metric icon={<Sparkles size={17} />} value={modelEvents.length} label="Model events" />
            <Metric icon={<Wrench size={17} />} value={toolEvents.length} label="Tool events" />
            <Metric icon={<AlertTriangle size={17} />} value={failedEvents.length} label="Failures" />
            <div className="rw-stat"><span><Clock3 size={17} /></span><strong>{avgModelMs ?? "n/a"}</strong><small>Avg model ms</small></div>
          </div>
          {tunedProfile && <div className="rw-tuning-strip">
            <div><strong>{tunedProfile.model_name}</strong><span>{tunedFlags(tunedProfile.candidate)}</span></div>
            <div><strong>{formatMetric(tunedProfile.summary?.ttft_seconds_median, "s")}</strong><span>Median TTFT</span></div>
            <div><strong>{formatMetric(tunedProfile.summary?.chars_per_second_median, " cps")}</strong><span>Generation</span></div>
            <div><strong>{formatMetric(tunedProfile.summary?.vram_delta_mb_max, " MB")}</strong><span>VRAM delta</span></div>
          </div>}
          {run.error_text && <div className="rw-inline-error">{run.error_text}</div>}
          <div className="space-y-4">{run.results?.map((result: any) => <ResultReview key={result.result_id} result={result} runId={run.run_id} onSaved={() => client.invalidateQueries({ queryKey: ["rw-run", selectedRun] })} />)}</div>
        </>}
      </div>
    </section>

    {run && <section className="rw-timeline-panel">
      <div className="rw-section-head"><div><p className="rw-step">03 · Audit trail</p><h2>Pipeline timeline</h2><p>Inspect each transition from raw input to final tool result.</p></div><div className="segmented">{(["all", "pipeline", "model", "tools"] as TraceFilter[]).map((filter) => <button type="button" key={filter} className={traceFilter === filter ? "active" : ""} onClick={() => setTraceFilter(filter)}>{filter}</button>)}</div></div>
      {!events.length && <EmptyState title="Waiting for events" description="Structured trace events will appear here as the pipeline advances." />}
      {!!events.length && !filteredEvents.length && <EmptyState title="No matching events" description="Choose another timeline filter to inspect this run." />}
      <div className="rw-timeline">{filteredEvents.map((event: any) => <TraceEventCard key={event.event_id} event={event} />)}</div>
    </section>}
  </div>;
}

function AgentTaskEditor({
  tasks,
  updateTask,
  addTask,
  removeTask,
}: {
  tasks: AgentTaskDraft[];
  updateTask: (id: string, patch: Partial<AgentTaskDraft>) => void;
  addTask: (kind: "browser" | "computer") => void;
  removeTask: (id: string) => void;
}) {
  return <div className="space-y-3">
    <div className="rw-sequence-head">
      <div><strong>Live delegated agent tasks</strong><span>Add 2–3 browser or computer-use tasks. These run on live sites/apps after confirmation.</span></div>
      <div className="flex gap-2"><Button variant="secondary" onClick={() => addTask("browser")}>Browser task</Button><Button variant="secondary" onClick={() => addTask("computer")}>Computer task</Button></div>
    </div>
    {tasks.map((task, index) => <div className="rw-agent-task" key={task.id}>
      <div className="rw-agent-task-head">
        <Badge>{index + 1}</Badge>
        <select value={task.agent_kind} onChange={(event) => {
          const kind = event.target.value as "browser" | "computer";
          updateTask(task.id, { agent_kind: kind, read_only: kind === "computer" });
        }}>
          <option value="browser">Browser</option>
          <option value="computer">Computer</option>
        </select>
        <button type="button" onClick={() => removeTask(task.id)} aria-label={`Remove task ${index + 1}`}><X size={15} /></button>
      </div>
      <label className="rw-field">Title<input value={task.title} onChange={(event) => updateTask(task.id, { title: event.target.value })} placeholder="Compare running shoes" /></label>
      <label className="rw-field">Instruction<textarea rows={4} value={task.instruction} onChange={(event) => updateTask(task.id, { instruction: event.target.value })} placeholder="Find 3 similar products and return links. Do not log in, add to cart, or download files." /></label>
      {task.agent_kind === "browser" && <label className="rw-field">Start URL<input value={task.start_url} onChange={(event) => updateTask(task.id, { start_url: event.target.value })} placeholder="https://duckduckgo.com/" /></label>}
      <div className="grid gap-3 md:grid-cols-3">
        <label className="rw-field">Max steps<input type="number" min="1" max="200" value={task.max_steps} onChange={(event) => updateTask(task.id, { max_steps: event.target.value })} /></label>
        <label className="rw-field">Timeout seconds<input type="number" min="10" max="1800" value={task.timeout_seconds} onChange={(event) => updateTask(task.id, { timeout_seconds: event.target.value })} /></label>
        <label className="rw-field">Mode<select value={task.read_only ? "read" : "write"} onChange={(event) => updateTask(task.id, { read_only: event.target.value === "read" })}><option value="read">Read-only</option><option value="write">Allow typing/clicking</option></select></label>
      </div>
      <label className="rw-field">Success criteria<input value={task.success_criteria} onChange={(event) => updateTask(task.id, { success_criteria: event.target.value })} placeholder="Returns links, actions taken, and blockers clearly." /></label>
    </div>)}
  </div>;
}

function defaultAgentTask(kind: "browser" | "computer", index: number): AgentTaskDraft {
  const browser = kind === "browser";
  return {
    id: `${kind}_${Date.now()}_${index}`,
    agent_kind: kind,
    title: browser ? "Browser research task" : "Computer-use desktop task",
    instruction: browser
      ? "Find 3 public web results for a product or topic and return the links with short reasons. Do not log in, add to cart, or download files."
      : "Inspect the current desktop or a safe local app and complete a simple read-only task. Do not type credentials, send messages, delete files, or download files.",
    start_url: browser ? "https://duckduckgo.com/" : "",
    read_only: true,
    max_steps: browser ? "40" : "25",
    timeout_seconds: browser ? "900" : "300",
    success_criteria: browser ? "Returns 3 useful links with reasons." : "Reports visible result, actions taken, and blockers.",
  };
}

function Metric({ icon, value, label }: { icon: React.ReactNode; value: number; label: string }) {
  return <div className="rw-stat"><span>{icon}</span><strong>{value}</strong><small>{label}</small></div>;
}

function downloadRun(runId: string, format: "json" | "csv") {
  window.location.href = `/api/real-world/runs/${runId}/export.${format}`;
}

function downloadRocm(format: "json" | "csv") {
  window.location.href = `/api/rocm-tuning/export.${format}`;
}

function formatMetric(value: any, suffix: string) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  return `${Number(value).toFixed(suffix === "s" ? 2 : 0)}${suffix}`;
}

function tunedFlags(candidate: any) {
  if (!candidate) return "No candidate details";
  return `ctx ${candidate.context_size} | ngl ${candidate.gpu_layers} | fa ${candidate.flash_attention ? "on" : "off"} | ${candidate.cache_type_k}/${candidate.cache_type_v}`;
}

function scenarioCountLabel(item: any) {
  if (item.modality === "agent_task_sequence") {
    const count = item.tasks?.length || 0;
    return `${count} agent task${count === 1 ? "" : "s"}`;
  }
  const count = item.events?.length || 0;
  return `${count} input${count === 1 ? "" : "s"} · ${String(item.modality || "").replace("_sequence", "")}`;
}

function TraceEventCard({ event }: { event: any }) {
  const toolEvent = event.event_type.includes("tool");
  const Icon = toolEvent ? Wrench : event.stage === "model" ? Sparkles : event.stage === "input" ? UploadCloud : Activity;
  return <article className={`rw-trace-event ${event.status === "failed" ? "failed" : ""}`}>
    <div className="rw-trace-rail"><span><Icon size={16} /></span></div>
    <div className="rw-trace-card">
      <div className="rw-trace-head"><div><span className="rw-event-index">{String(event.sequence).padStart(2, "0")}</span><Badge>{event.stage}</Badge><h3>{humanEvent(event.event_type)}</h3></div><time>{formatDate(event.created_at)}</time></div>
      <div className="rw-trace-meta">{event.duration_ms != null && <span><Clock3 size={13} />{event.duration_ms} ms</span>}{event.model && <span><Sparkles size={13} />{event.model}</span>}<Badge tone={statusTone(event.status)}>{humanStatus(event.status)}</Badge></div>
      <JsonDetails label="Inspect exact event payload" value={event.payload} open={["tool_started", "tool_finished"].includes(event.event_type)} />
    </div>
  </article>;
}

function ResultReview({ result, runId, onSaved }: { result: any; runId: string; onSaved: () => void }) {
  const [form, setForm] = useState<Record<string, string>>({
    reviewer: "local-user", perception_score: "", decision_score: "", tool_choice_score: "",
    tool_execution_score: "", final_response_score: "", overall_score: "", notes: "",
  });
  const save = useMutation({ mutationFn: () => sendJson(`/api/real-world/results/${result.result_id}/review`, "POST", form), onSuccess: onSaved });
  return <article className="rw-result-card">
    <header><span className="rw-result-icon">{result.modality === "agent_task_sequence" ? <Wrench size={19} /> : result.modality === "audio_sequence" ? <AudioLines size={19} /> : <ImageIcon size={19} />}</span><div className="min-w-0 flex-1"><h3>{result.title}</h3><p>{result.modality.replace("_sequence", "")} evaluation</p></div><Badge tone={statusTone(result.status)}>{humanStatus(result.status)}</Badge></header>
    <div className="rw-media-strip">{result.media?.map((_: any, index: number) => result.modality === "audio_sequence" ? <div className="rw-audio" key={index}><FileAudio size={18} /><audio controls src={`/api/real-world/runs/${runId}/results/${result.result_id}/media/${index}`} /></div> : <figure key={index}><img src={`/api/real-world/runs/${runId}/results/${result.result_id}/media/${index}`} alt={`Scenario input ${index + 1}`} /><figcaption>Frame {index + 1}</figcaption></figure>)}</div>
    {result.transcript_text && <div className="rw-output-block"><span>Transcript</span><p>{result.transcript_text}</p></div>}
    {result.final_response && <div className="rw-output-block"><span>Final response</span><Markdown>{result.final_response}</Markdown></div>}
    <div className="rw-review"><div className="rw-review-head"><div><strong>Manual quality review</strong><p>Rate each stage from 1 (poor) to 5 (excellent).</p></div><Badge>{form.overall_score ? `${form.overall_score}/5 overall` : "Not rated"}</Badge></div>
      <div className="rw-score-grid">{SCORE_FIELDS.map(([field, label]) => <div className="rw-score" key={field}><span>{label}</span><div>{[1, 2, 3, 4, 5].map((score) => <button type="button" key={score} className={form[field] === String(score) ? "active" : ""} onClick={() => setForm({ ...form, [field]: String(score) })}>{score}</button>)}</div></div>)}</div>
      <label className="rw-field mt-4">Review notes<textarea rows={3} value={form.notes} onChange={(event) => setForm({ ...form, notes: event.target.value })} placeholder="What worked? What should improve?" /></label>
      <div className="mt-3 flex items-center gap-3"><Button variant="secondary" onClick={() => save.mutate()} disabled={save.isPending}><Save size={15} />{save.isPending ? "Saving…" : "Save review"}</Button>{save.isSuccess && <span className="text-sm font-semibold text-good">Review saved</span>}</div>
    </div>
  </article>;
}

function friendlyRole(role: string) {
  return role.replaceAll("_", " ").replace(/\b\w/g, (character) => character.toUpperCase());
}

function humanEvent(value: string) {
  return value.replaceAll("_", " ").replace(/\b\w/g, (character) => character.toUpperCase());
}

function humanStatus(value: string) {
  return value?.replaceAll("_", " ") || "unknown";
}

function statusTone(status: string): "neutral" | "good" | "warn" | "danger" {
  if (["completed"].includes(status)) return "good";
  if (["queued", "running", "completed_with_errors"].includes(status)) return "warn";
  if (["failed", "cancelled", "interrupted"].includes(status)) return "danger";
  return "neutral";
}
