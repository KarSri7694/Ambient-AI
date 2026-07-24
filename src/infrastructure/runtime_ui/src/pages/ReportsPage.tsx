import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Clock3, FileCheck2, ListTodo, XCircle } from "lucide-react";
import { getJson, sendJson } from "../api";
import { Badge, Button, EmptyState, ErrorState, formatDate, LoadingState, PageHeader } from "../components/ui";

export function ReportsPage() {
  const client = useQueryClient();
  const reports = useQuery({ queryKey: ["reports"], queryFn: () => getJson<any>("/api/reports?limit=50"), refetchInterval: 3000 });
  const cancel = useMutation({
    mutationFn: (id: number) => sendJson(`/api/chat/scheduled/${id}/cancel`, "POST"),
    onSuccess: () => client.invalidateQueries({ queryKey: ["reports"] }),
  });
  const items = reports.data?.reports || [];
  const tasks = reports.data?.queued_tasks || [];
  return (
    <>
      <PageHeader eyebrow="Agent output" title="Reports" description="Finished work and tasks Ambient AI has queued for a future time." actions={<><Badge tone="good">{items.length} reports</Badge><Badge>{tasks.length} queued</Badge></>} />
      {reports.isLoading && <LoadingState />}{reports.isError && <ErrorState error={reports.error} />}
      <div className="two-column">
        <section className="space-y-3">
          <div className="section-label"><FileCheck2 size={17} />Completed reports</div>
          {!items.length && !reports.isLoading && <EmptyState title="No reports yet" description="User-relevant agent results will appear here." />}
          {items.map((item: any) => {
            const report = item.report || {};
            return <article className="record-card" key={item.interaction_id}><div className="record-top"><div className="flex gap-2"><Badge tone={report.status === "completed" ? "good" : "neutral"}>{report.status || "report"}</Badge><Badge>{item.source}</Badge></div><time>{formatDate(item.created_at)}</time></div><h2>{report.title || "Untitled report"}</h2><p>{report.summary || "No summary provided."}</p><div className="meta-grid"><div><strong>Model</strong><span>{item.model || "unknown"}</span></div><div><strong>Tools</strong><span>{Array.isArray(report.tools_used) && report.tools_used.length ? report.tools_used.join(", ") : "none"}</span></div><div><strong>Artifact</strong><span>{report.artifact_path || "none"}</span></div></div></article>;
          })}
        </section>
        <aside className="space-y-3">
          <div className="section-label"><ListTodo size={17} />Queued tasks</div>
          {!tasks.length && !reports.isLoading && <EmptyState title="Queue is clear" description="There are no scheduled tasks waiting to run." />}
          {tasks.map((task: any) => <article className="record-card compact" key={task.id}><div className="record-top"><div className="flex gap-2"><Badge>{task.priority || "medium"}</Badge><Badge>{task.status || "pending"}</Badge></div><time>{formatDate(task.created_at)}</time></div><h2>Task {task.id}</h2><p>{task.description}</p>{task.run_at_utc && <><div className="mt-3 flex items-center gap-2 text-sm text-muted"><Clock3 size={15} />Runs {formatDate(task.run_at_utc)}</div><Button className="mt-4" variant="danger" onClick={() => cancel.mutate(task.id)} disabled={cancel.isPending}><XCircle size={16} />Cancel task</Button></>}</article>)}
        </aside>
      </div>
    </>
  );
}
