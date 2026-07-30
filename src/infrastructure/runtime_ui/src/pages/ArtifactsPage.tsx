import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Archive, FileStack, History, RefreshCw } from "lucide-react";
import { getJson, sendJson } from "../api";
import {
  Badge, Button, EmptyState, ErrorState, formatDate, LoadingState, Markdown, PageHeader,
} from "../components/ui";

type ArtifactStatus = "active" | "archived";

export function ArtifactsPage() {
  const client = useQueryClient();
  const [status, setStatus] = useState<ArtifactStatus>("active");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const maintenance = useQuery({
    queryKey: ["artifact-maintenance-status"],
    queryFn: () => getJson<any>("/api/artifacts/maintenance/status"),
    refetchInterval: 2500,
    retry: false,
  });
  const artifacts = useQuery({
    queryKey: ["artifacts", status],
    queryFn: () => getJson<any>(`/api/artifacts?status=${status}&limit=500`),
    refetchInterval: 5000,
  });
  const history = useQuery({
    queryKey: ["artifact-maintenance-history"],
    queryFn: () => getJson<any>("/api/artifacts/maintenance/history?limit=20"),
    refetchInterval: 5000,
    retry: false,
  });
  const detail = useQuery({
    queryKey: ["artifact", selectedId],
    queryFn: () => getJson<any>(`/api/artifacts/${selectedId}`),
    enabled: Boolean(selectedId),
  });
  const run = useMutation({
    mutationFn: () => sendJson<any>("/api/artifacts/maintenance/run", "POST"),
    onSuccess: () => {
      client.invalidateQueries({ queryKey: ["artifact-maintenance-status"] });
      client.invalidateQueries({ queryKey: ["artifact-maintenance-history"] });
    },
  });

  const items = artifacts.data?.items || [];
  const state = maintenance.data?.status || {};
  const runtime = state.runtime || {};
  const busy = Boolean(runtime.requested || runtime.running);
  const runs = history.data?.runs || [];
  const merges = history.data?.merges || [];

  useEffect(() => {
    if (selectedId && !items.some((item: any) => item.artifact_id === selectedId)) setSelectedId(null);
  }, [items, selectedId]);

  return (
    <>
      <PageHeader
        eyebrow="Knowledge library"
        title="Artifacts"
        description="Ambient AI continuously finds notes from the same ongoing thread, consolidates their distinct information, and moves superseded copies into the archive."
        actions={
          <>
            <Badge tone="good">{state.active_count ?? 0} active</Badge>
            <Badge>{state.archived_count ?? 0} archived</Badge>
            {state.due && <Badge tone="warn">Maintenance due</Badge>}
            <Button variant="primary" onClick={() => run.mutate()} disabled={busy || run.isPending || state.available === false}>
              <RefreshCw size={16} className={runtime.running ? "animate-spin" : ""} />
              {runtime.running ? "Consolidating" : runtime.requested ? "Queued" : "Consolidate now"}
            </Button>
          </>
        }
      />

      {maintenance.isError && <ErrorState error={maintenance.error} />}
      {runtime.last_error && <div className="mb-4 rounded-2xl border border-danger/30 bg-danger/10 p-4 text-sm text-danger">Last maintenance error: {runtime.last_error}</div>}

      <div className="mb-5 flex flex-wrap items-center gap-2">
        <Button variant={status === "active" ? "primary" : "secondary"} onClick={() => setStatus("active")}><FileStack size={16} />Active library</Button>
        <Button variant={status === "archived" ? "primary" : "secondary"} onClick={() => setStatus("archived")}><Archive size={16} />Archive</Button>
        <span className="ml-auto text-xs text-muted">
          {state.changed_count ?? 0} changed since last run
          {state.last_success?.completed_at ? ` · last run ${formatDate(state.last_success.completed_at)}` : " · no completed run yet"}
        </span>
      </div>

      <div className="two-column wide-left">
        <section className="space-y-3">
          <div className="section-label"><FileStack size={17} />{status === "active" ? "Current artifacts" : "Archived duplicates"}</div>
          {artifacts.isLoading && <LoadingState label="Loading artifact library" />}
          {artifacts.isError && <ErrorState error={artifacts.error} />}
          {!items.length && !artifacts.isLoading && <EmptyState title={`No ${status} artifacts`} description={status === "active" ? "New agent-created notes will appear here." : "Superseded copies moved by consolidation will appear here."} />}
          {items.map((item: any) => (
            <button
              type="button"
              className={`record-card selectable w-full text-left ${selectedId === item.artifact_id ? "selected" : ""}`}
              key={item.artifact_id}
              onClick={() => setSelectedId(item.artifact_id)}
            >
              <div className="record-top">
                <div className="flex flex-wrap gap-2"><Badge tone={item.status === "active" ? "good" : "neutral"}>{item.status}</Badge><Badge>{item.artifact_kind || "note"}</Badge><Badge>{item.source_count} sources</Badge></div>
                <time>{formatDate(item.last_ai_edited_at)}</time>
              </div>
              <h2>{item.title}</h2>
              <p>{item.short_summary || "No summary available."}</p>
              {item.canonical_artifact_id && <p className="mt-2 text-xs text-muted">Consolidated into {item.canonical_artifact_id}</p>}
            </button>
          ))}
        </section>

        <aside className="space-y-4">
          <section className="panel p-5">
            <div className="section-label"><FileStack size={17} />Artifact detail</div>
            {!selectedId && <EmptyState title="Select an artifact" description="Open a note to inspect its full organized content." />}
            {detail.isLoading && <LoadingState />}
            {detail.isError && <ErrorState error={detail.error} />}
            {detail.data?.artifact && <><h2 className="mb-3 text-lg font-bold">{detail.data.artifact.title}</h2><Markdown>{detail.data.artifact.content}</Markdown></>}
          </section>

          <section className="panel p-5">
            <div className="section-label"><History size={17} />Maintenance history</div>
            {!runs.length && <p className="text-sm text-muted">No maintenance runs recorded yet.</p>}
            <div className="space-y-3">
              {runs.slice(0, 6).map((item: any) => <div className="rounded-xl border border-line p-3 text-sm" key={item.run_id}><div className="flex items-center justify-between gap-2"><Badge tone={item.status === "completed" ? "good" : item.status === "failed" ? "danger" : "warn"}>{item.status}</Badge><time className="text-xs text-muted">{formatDate(item.started_at)}</time></div><p className="mt-2 text-muted">{item.merged_cluster_count || 0} groups merged · {item.archived_count || 0} archived · {item.candidate_pair_count || 0} pairs reviewed</p></div>)}
            </div>
            {merges.length > 0 && <p className="mt-4 text-xs text-muted">{merges.length} recent merge decision{merges.length === 1 ? "" : "s"} retained for audit.</p>}
          </section>
        </aside>
      </div>
    </>
  );
}
