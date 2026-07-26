import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Activity, AlertTriangle, AudioLines, Check, ChevronRight, Clock3, FileAudio,
  FileImage, Gauge, ImageIcon, Layers3, Play, Save, Settings2, ShieldAlert,
  Sparkles, Square, UploadCloud, Wrench, X,
} from "lucide-react";
import { getJson, sendJson, uploadBinary } from "../api";
import {
  Badge, Button, EmptyState, ErrorState, formatDate, JsonDetails, LoadingState,
  Markdown, PageHeader,
} from "../components/ui";

type Uploaded = { media_id: string; kind: "image" | "audio"; original_name: string };
type SourceMode = "suite" | "upload";
type TraceFilter = "all" | "model" | "tools" | "pipeline";

const SPEEDS = [1, 2, 5, 10];
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
  const [traceFilter, setTraceFilter] = useState<TraceFilter>("all");
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
    mutationFn: () => sendJson<any>("/api/real-world/runs", "POST", sourceMode === "upload" ? {
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
    if (!files) return;
    setUploading(true);
    try {
      const added: Uploaded[] = [];
      for (const file of Array.from(files)) {
        const payload = await uploadBinary<any>(
          `/api/real-world/uploads?filename=${encodeURIComponent(file.name)}&kind=${uploadKind}`,
          file,
        );
        added.push(payload.media);
      }
      setUploads((current) => [...current, ...added]);
    } finally {
      setUploading(false);
    }
  };

  const run = detail.data?.run;
  const events = trace.data?.events || [];
  const active = run && ["queued", "running"].includes(run.status);
  const runItems = runs.data?.runs || [];
  const completedCount = runItems.filter((item: any) => item.status === "completed").length;
  const filteredEvents = events.filter((event: any) => {
    if (traceFilter === "model") return event.stage === "model";
    if (traceFilter === "tools") return event.event_type.includes("tool");
    if (traceFilter === "pipeline") return !["model", "agent"].includes(event.stage);
    return true;
  });
  const validSource = sourceMode === "upload" ? uploads.length > 0 : scenarioIds.length > 0;
  const armed = confirmation === "RUN LIVE TOOLS";

  return <div className="rw-page">
    <PageHeader
      eyebrow="Evaluation studio"
      title="Real-world tests"
      description="Replay authentic media through the complete ambient pipeline, then inspect every perception, model, and tool decision in one trace."
      actions={<><Badge tone={active ? "warn" : "good"}>{active ? "Run in progress" : "Lab ready"}</Badge><Badge>Local only</Badge></>}
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

    {!suites.data?.available && !suites.isLoading && <div className="panel mb-5 p-4"><EmptyState title="Start the evaluation lab" description="Launch tests/real_world_tests/run_lab.py to configure and run evaluations." /></div>}

    <section className="rw-launch-card">
      <div className="rw-section-head">
        <div><p className="rw-step">01 · Configure replay</p><h2>Choose what the agent should experience</h2><p>Select a reusable suite or assemble a private ad-hoc media sequence.</p></div>
        <div className="segmented" aria-label="Input source">
          <button type="button" className={sourceMode === "suite" ? "active" : ""} onClick={() => setSourceMode("suite")}><Layers3 size={15} />Suite</button>
          <button type="button" className={sourceMode === "upload" ? "active" : ""} onClick={() => setSourceMode("upload")}><UploadCloud size={15} />Upload</button>
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
                  <span className="rw-scenario-icon">{item.modality === "audio_sequence" ? <AudioLines size={19} /> : <ImageIcon size={19} />}</span>
                  <span className="min-w-0 flex-1 text-left"><strong>{item.title}</strong><small>{item.events.length} input{item.events.length === 1 ? "" : "s"} · {item.modality.replace("_sequence", "")}</small></span>
                  <ChevronRight size={17} className="text-muted" />
                </button>;
              })}
            </div>}
          </> : <>
            <div className="segmented mb-3"><button type="button" className={uploadKind === "image" ? "active" : ""} onClick={() => { setUploadKind("image"); setUploads([]); }}><FileImage size={15} />Images</button><button type="button" className={uploadKind === "audio" ? "active" : ""} onClick={() => { setUploadKind("audio"); setUploads([]); }}><FileAudio size={15} />Audio</button></div>
            <label className="rw-dropzone">
              <input className="sr-only" type="file" multiple accept={uploadKind === "image" ? "image/*" : "audio/*"} onChange={(event) => upload(event.target.files)} />
              <span className="rw-drop-icon"><UploadCloud size={24} /></span>
              <strong>{uploading ? "Uploading media…" : `Choose ${uploadKind} files`}</strong>
              <span>Files stay in the local evaluation workspace</span>
            </label>
            {uploads.length > 0 && <div className="rw-upload-list">{uploads.map((item, index) => <div className="rw-upload-item" key={item.media_id}><span className="rw-sequence-number">{index + 1}</span>{item.kind === "image" ? <FileImage size={17} /> : <FileAudio size={17} />}<span className="truncate">{item.original_name}</span><small>+{index * 10}s</small><button type="button" aria-label={`Remove ${item.original_name}`} onClick={() => setUploads((current) => current.filter((media) => media.media_id !== item.media_id))}><X size={15} /></button></div>)}</div>}
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
        <div className="rw-arm-action"><label><span>Type RUN LIVE TOOLS</span><input value={confirmation} onChange={(event) => setConfirmation(event.target.value)} placeholder="RUN LIVE TOOLS" /></label><Button variant="primary" disabled={start.isPending || !armed || !validSource} onClick={() => start.mutate()}><Play size={16} />{start.isPending ? "Starting…" : "Start evaluation"}</Button></div>
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
          <div className="rw-selected-head"><div><p className="rw-step">Selected evaluation</p><h2>{run.suite_id}</h2><p>{run.scenario_ids?.join(", ") || "Uploaded media scenario"}</p></div><div className="flex items-center gap-2"><Badge tone={statusTone(run.status)}>{humanStatus(run.status)}</Badge>{active && <Button variant="danger" onClick={() => cancel.mutate(run.run_id)}><Square size={14} />Cancel</Button>}</div></div>
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

function Metric({ icon, value, label }: { icon: React.ReactNode; value: number; label: string }) {
  return <div className="rw-stat"><span>{icon}</span><strong>{value}</strong><small>{label}</small></div>;
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
    <header><span className="rw-result-icon">{result.modality === "audio_sequence" ? <AudioLines size={19} /> : <ImageIcon size={19} />}</span><div className="min-w-0 flex-1"><h3>{result.title}</h3><p>{result.modality.replace("_sequence", "")} evaluation</p></div><Badge tone={statusTone(result.status)}>{humanStatus(result.status)}</Badge></header>
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
