import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Download, RefreshCw, Save, SlidersHorizontal } from "lucide-react";
import { getJson, queryString, sendJson } from "../api";
import { Badge, Button, EmptyState, ErrorState, formatDate, ImageModal, JsonDetails, LoadingState, PageHeader } from "../components/ui";

type Mode = "llm" | "asr";

export function TrainingPage() {
  const client = useQueryClient();
  const [mode, setMode] = useState<Mode>("llm");
  const [selectedId, setSelectedId] = useState("");
  const [status, setStatus] = useState("");
  const [source, setSource] = useState("");
  const [model, setModel] = useState("");
  const [draftFilters, setDraftFilters] = useState({ status: "", source: "", model: "" });
  const [reviewer, setReviewer] = useState("local-user");
  const [reviewStatus, setReviewStatus] = useState("pending");
  const [primary, setPrimary] = useState("");
  const [secondary, setSecondary] = useState("");
  const [notes, setNotes] = useState("");
  const [notice, setNotice] = useState("");
  const [image, setImage] = useState<string | null>(null);
  const filters = queryString({ limit: 100, review_status: status, source: mode === "llm" ? source : "", model: mode === "llm" ? model : "" });
  const records = useQuery({ queryKey: ["training-records", mode, status, source, model], queryFn: () => getJson<any>(`/api/training/${mode}?${filters}`), refetchInterval: 5000 });
  const exportsQuery = useQuery({ queryKey: ["training-exports"], queryFn: () => getJson<any>("/api/training/exports?limit=10"), refetchInterval: 10000 });
  const detail = useQuery({ queryKey: ["training-detail", mode, selectedId], queryFn: () => getJson<any>(`/api/training/${mode}/${selectedId}`), enabled: Boolean(selectedId) });
  const record = detail.data?.record;
  useEffect(() => {
    if (!record) return;
    const review = record.review || {};
    setReviewer(review.reviewer || "local-user"); setReviewStatus(review.status || record.review_status || "pending"); setNotes(review.notes || "");
    if (mode === "llm") { setPrimary(review.corrected_response_text ?? record.response_text ?? ""); setSecondary(review.corrected_reasoning_text ?? ""); }
    else { setPrimary(review.corrected_transcript_text ?? record.transcript_text ?? ""); setSecondary(""); }
  }, [record, mode]);
  const switchMode = (next: Mode) => { setMode(next); setSelectedId(""); setNotice(""); };
  const applyFilters = () => { setStatus(draftFilters.status); setSource(draftFilters.source); setModel(draftFilters.model); setSelectedId(""); };
  const clearFilters = () => { const empty = { status: "", source: "", model: "" }; setDraftFilters(empty); setStatus(""); setSource(""); setModel(""); setSelectedId(""); };
  const sync = useMutation({ mutationFn: () => sendJson<any>(`/api/training/sync/${mode}`, "POST"), onSuccess: (data) => { setNotice(`Synced ${data.synced || 0} ${mode.toUpperCase()} records.`); client.invalidateQueries({ queryKey: ["training-records"] }); } });
  const exportData = useMutation({ mutationFn: () => sendJson<any>(`/api/training/export/${mode}`, "POST", { statuses: ["approved"] }), onSuccess: (data) => { setNotice(`Exported ${data.record_count || 0} approved ${mode.toUpperCase()} records.`); client.invalidateQueries({ queryKey: ["training-exports"] }); } });
  const save = useMutation({ mutationFn: () => {
    const body: any = { reviewer, status: reviewStatus, notes };
    if (mode === "llm") { body.corrected_response_text = primary; body.corrected_reasoning_text = secondary; }
    else body.corrected_transcript_text = primary;
    return sendJson(`/api/training/${mode}/${selectedId}/review`, "POST", body);
  }, onSuccess: async () => { setNotice("Training review saved."); await Promise.all([client.invalidateQueries({ queryKey: ["training-records"] }), client.invalidateQueries({ queryKey: ["training-detail", mode, selectedId] })]); } });
  const items = records.data?.records || [];
  const mediaUrl = (path: string) => `/api/training/media?path=${encodeURIComponent(path)}`;
  const audioPath = record?.cleaned_audio_path || record?.upload_audio_path || "";

  return <><PageHeader eyebrow="Fine-tuning data" title="Training Review" description="Inspect, correct, approve, and export LLM and speech records without leaving the runtime console." actions={<><div className="segmented"><button className={mode === "llm" ? "active" : ""} onClick={() => switchMode("llm")}>LLM</button><button className={mode === "asr" ? "active" : ""} onClick={() => switchMode("asr")}>ASR</button></div><Button variant="primary" onClick={() => sync.mutate()} disabled={sync.isPending}><RefreshCw size={16} />Sync</Button><Button variant="secondary" onClick={() => exportData.mutate()} disabled={exportData.isPending}><Download size={16} />Export approved</Button></>} />
    {notice && <div className="notice">{notice}</div>}
    <div className="filter-panel mb-4"><div className="section-label mb-0"><SlidersHorizontal size={16} />Filters</div><label><span>Status</span><select value={draftFilters.status} onChange={(event) => setDraftFilters({ ...draftFilters, status: event.target.value })}><option value="">All statuses</option><option value="pending">pending</option><option value="approved">approved</option><option value="rejected">rejected</option></select></label>{mode === "llm" && <><label><span>Source</span><input type="search" value={draftFilters.source} onChange={(event) => setDraftFilters({ ...draftFilters, source: event.target.value })} placeholder="Any source" /></label><label><span>Model</span><input type="search" value={draftFilters.model} onChange={(event) => setDraftFilters({ ...draftFilters, model: event.target.value })} placeholder="Any model" /></label></>}<div className="flex self-end gap-2"><Button variant="primary" onClick={applyFilters}>Apply</Button><Button variant="ghost" onClick={clearFilters}>Clear</Button></div></div>
    <div className="two-column wide-left">
      <section className="space-y-3">{records.isLoading && <LoadingState />}{records.isError && <ErrorState error={records.error} />}{!items.length && !records.isLoading && <EmptyState title="No training records" description="Sync model interactions or speech transcripts to begin review." />}{items.map((item: any) => { const title = mode === "llm" ? `${item.model || "unknown model"} · ${item.source || "unknown source"}` : String(item.transcript_path || "").split(/[\\/]/).pop(); const preview = mode === "llm" ? item.response_text || item.error_text : item.transcript_text; return <button type="button" className={`record-card selectable w-full text-left ${selectedId === item.record_id ? "selected" : ""}`} key={item.record_id} onClick={() => setSelectedId(item.record_id)}><div className="record-top"><div className="flex gap-2"><Badge tone={item.review_status === "approved" ? "good" : item.review_status === "rejected" ? "danger" : "neutral"}>{item.review_status || "pending"}</Badge><Badge>{mode === "llm" ? item.model || "model" : "transcript"}</Badge></div><time>{formatDate(item.created_at)}</time></div><h2>{title || "Training record"}</h2><p>{String(preview || "No preview available.").slice(0, 360)}</p></button>; })}</section>
      <aside className="panel sticky-card p-4"><div className="section-label"><Save size={17} />Record editor</div>{detail.isLoading && <LoadingState />}{detail.isError && <ErrorState error={detail.error} />}{!record && !detail.isLoading ? <EmptyState title="Select a record" description="Choose a training record to inspect and correct." /> : record && <div className="space-y-4">{mode === "llm" ? <>{record.image_path && <button type="button" className="preview-button" onClick={() => setImage(mediaUrl(record.image_path))}><img src={mediaUrl(record.image_path)} alt="Training screenshot" /><span>Click to enlarge</span></button>}<div className="meta-grid"><div><strong>Model</strong><span>{record.model || "unknown"}</span></div><div><strong>Source</strong><span>{record.source || "unknown"}</span></div><div><strong>Interaction</strong><span>{record.interaction_id}</span></div></div><JsonDetails label="Raw response" value={record.response_text} open /><JsonDetails label="Messages" value={record.messages} /><JsonDetails label="Tools" value={record.tools} /><JsonDetails label="Tool calls" value={record.tool_calls} /><JsonDetails label="Metadata" value={record.metadata} /><JsonDetails label="Report" value={record.report} /></> : <>{audioPath ? <audio controls className="w-full" src={mediaUrl(audioPath)} /> : <p className="text-sm text-muted">No audio file matched this transcript.</p>}<div className="meta-grid"><div><strong>Transcript</strong><span>{record.transcript_path}</span></div><div><strong>Upload audio</strong><span>{record.upload_audio_path || "none"}</span></div><div><strong>Cleaned audio</strong><span>{record.cleaned_audio_path || "none"}</span></div></div><JsonDetails label="Raw transcript" value={record.transcript_text} open /><JsonDetails label="Metadata" value={record.metadata} /></>}
        <div className="form-stack"><div className="grid grid-cols-2 gap-3"><label>Reviewer<input value={reviewer} onChange={(event) => setReviewer(event.target.value)} /></label><label>Status<select value={reviewStatus} onChange={(event) => setReviewStatus(event.target.value)}><option value="pending">pending</option><option value="approved">approved</option><option value="rejected">rejected</option></select></label></div><label>{mode === "llm" ? "Corrected response" : "Corrected transcript"}<textarea rows={10} value={primary} onChange={(event) => setPrimary(event.target.value)} /></label>{mode === "llm" && <label>Corrected reasoning (optional)<textarea rows={5} value={secondary} onChange={(event) => setSecondary(event.target.value)} /></label>}<label>Review notes<textarea rows={4} value={notes} onChange={(event) => setNotes(event.target.value)} /></label><Button variant="primary" onClick={() => save.mutate()} disabled={save.isPending}>Save training review</Button></div>
      </div>}
      <div className="mt-5 border-t border-line pt-4"><p className="eyebrow mb-2">Recent exports</p><div className="whitespace-pre-wrap text-xs leading-5 text-muted">{(exportsQuery.data?.exports || []).map((item: any) => `${String(item.dataset_kind || "").toUpperCase()} · ${item.record_count} records · ${item.output_path}`).join("\n") || "No dataset exports yet."}</div></div></aside>
    </div><ImageModal src={image} alt="Expanded training screenshot" onClose={() => setImage(null)} /></>;
}
