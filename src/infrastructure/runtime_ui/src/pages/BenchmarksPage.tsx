import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Save, Scale } from "lucide-react";
import { getJson, sendJson } from "../api";
import { Badge, Button, EmptyState, ErrorState, formatDate, JsonDetails, LoadingState, PageHeader } from "../components/ui";

export function BenchmarksPage() {
  const queryClient = useQueryClient();
  const [selected, setSelected] = useState<any>(null);
  const [reviewer, setReviewer] = useState("local-user");
  const [score, setScore] = useState("");
  const [notes, setNotes] = useState("");
  const results = useQuery({ queryKey: ["benchmarks"], queryFn: () => getJson<any>("/api/benchmarks/results?limit=100"), refetchInterval: 5000 });
  useEffect(() => {
    if (!selected) return;
    const updated = results.data?.results?.find((item: any) => item.result_id === selected.result_id);
    if (updated) setSelected(updated);
  }, [results.data, selected?.result_id]);
  const choose = (item: any) => { setSelected(item); setReviewer(item.manual_review?.reviewer || "local-user"); setScore(item.manual_review?.score ?? ""); setNotes(item.manual_review?.notes || ""); };
  const save = useMutation({
    mutationFn: () => sendJson(`/api/benchmarks/results/${selected.result_id}/review`, "POST", { reviewer, score, notes }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["benchmarks"] }),
  });
  const items = results.data?.results || [];
  return <><PageHeader eyebrow="Model evaluation" title="Benchmarks" description="Compare model quality and performance, then attach a human review to any stored result." actions={<Badge>{items.length} results</Badge>} />
    {results.isLoading && <LoadingState />}{results.isError && <ErrorState error={results.error} />}
    <div className="two-column wide-left">
      <section className="space-y-3">
        {!items.length && !results.isLoading && <EmptyState title="No benchmark results" description="Run a benchmark to populate this workspace." />}
        {items.map((item: any) => (
          <article key={item.result_id} className={`record-card selectable ${selected?.result_id === item.result_id ? "selected" : ""}`}>
            <div className="record-top"><div className="flex gap-2"><Badge>{item.status || "completed"}</Badge><Badge>{item.service_name || "service"}</Badge></div><time>{formatDate(item.created_at)}</time></div>
            <h2>{item.case_title || item.case_id || "Benchmark case"}</h2>
            <div className="meta-grid"><div><strong>Model</strong><span>{item.model_name || "unknown"}</span></div><div><strong>Total tokens</strong><span>{item.total_tokens ?? 0}</span></div><div><strong>Prefill</strong><span>{item.prefill_tokens_per_second ?? 0} tok/s</span></div><div><strong>Generation</strong><span>{item.generation_tokens_per_second ?? 0} tok/s</span></div><div><strong>Auto score</strong><span>{item.auto_score ?? "n/a"}</span></div><div><strong>Manual score</strong><span>{item.manual_review?.score ?? "n/a"}</span></div></div>
            <p className="output-preview">{item.response_text || item.error_text || item.structured_output_json || "No output"}</p>
            <div className="my-3"><Button variant="secondary" onClick={() => choose(item)}>Review this result</Button></div>
            <JsonDetails label="Structured output" value={item.structured_output_json} /><JsonDetails label="Auto score details" value={item.auto_score_details_json} /><JsonDetails label="Metadata" value={item.metadata_json} />
          </article>
        ))}
      </section>
      <aside className="panel sticky-card"><div className="section-label"><Scale size={17} />Manual review</div>{!selected ? <EmptyState title="Select a result" description="Choose a benchmark result to add or update its review." /> : <div className="form-stack"><p className="text-sm text-muted">Reviewing <strong className="text-strong">{selected.case_id}</strong> on {selected.model_name}</p><label>Reviewer<input value={reviewer} onChange={(event) => setReviewer(event.target.value)} /></label><label>Score<input type="number" min="0" max="1" step="0.01" value={score} onChange={(event) => setScore(event.target.value)} placeholder="0.00 – 1.00" /></label><label>Notes<textarea rows={8} value={notes} onChange={(event) => setNotes(event.target.value)} placeholder="Manual review notes" /></label><Button variant="primary" onClick={() => save.mutate()} disabled={save.isPending}><Save size={16} />{save.isPending ? "Saving…" : "Save review"}</Button>{save.isSuccess && <p className="text-sm text-good">Review saved.</p>}</div>}</aside>
    </div>
  </>;
}
