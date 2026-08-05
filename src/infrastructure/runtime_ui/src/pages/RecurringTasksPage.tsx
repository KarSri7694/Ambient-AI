import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { BellRing, CirclePause, CirclePlay, Repeat2, XCircle } from "lucide-react";
import { getJson, sendJson } from "../api";
import { Badge, Button, EmptyState, ErrorState, LoadingState, PageHeader, formatDate } from "../components/ui";

export function RecurringTasksPage() {
  const client = useQueryClient();
  const tasks = useQuery({ queryKey: ["recurring-tasks"], queryFn: () => getJson<any>("/api/recurring-tasks"), refetchInterval: 5000 });
  const action = useMutation({
    mutationFn: ({ id, action }: { id: string; action: string }) => sendJson(`/api/recurring-tasks/${id}/${action}`, "POST"),
    onSuccess: () => client.invalidateQueries({ queryKey: ["recurring-tasks"] }),
  });
  const items = tasks.data?.tasks || [];
  return <>
    <PageHeader eyebrow="Durable background work" title="Monitors & recurring tasks" description="Tasks created in chat or Todoist with @ambient survive restarts, retain their evidence, and only perform their approved scope." actions={<Badge tone={items.filter((task: any) => task.status === "active").length ? "good" : "neutral"}>{items.filter((task: any) => task.status === "active").length} active</Badge>} />
    {tasks.isLoading && <LoadingState />}{tasks.isError && <ErrorState error={tasks.error} />}
    {!tasks.isLoading && !items.length && <EmptyState title="No monitors yet" description="Ask Ambient AI to monitor a condition or run a task at an interval. Todoist directives need the @ambient label." />}
    <section className="space-y-3">{items.map((task: any) => <article className="record-card" key={task.task_id}>
      <div className="record-top"><div className="flex gap-2"><Badge tone={task.status === "active" ? "good" : task.status === "failed" ? "danger" : "neutral"}>{task.status}</Badge><Badge>{task.task_kind}</Badge><Badge>{task.source_kind}</Badge></div><time>Next {formatDate(task.next_run_at)}</time></div>
      <h2>{task.title}</h2><p>{task.instruction}</p>
      {task.monitor_condition && <div className="meta-grid"><div><strong>Watching for</strong><span>{task.monitor_condition}</span></div><div><strong>Last state</strong><span>{task.monitor_state?.state || "waiting"}{task.monitor_state?.user_seen ? " · seen by you" : ""}</span></div><div><strong>Delivery</strong><span>{task.monitor_state?.notification_state || "inbox"}</span></div></div>}
      <div className="meta-grid mt-3"><div><strong>Interval</strong><span>{task.interval_seconds}s</span></div><div><strong>Safe actions</strong><span>{task.safe_actions?.join(", ") || "none"}</span></div><div><strong>Last run</strong><span>{task.last_run_at ? formatDate(task.last_run_at) : "Not run yet"}</span></div></div>
      {task.monitor_state?.evidence && <p className="mt-3 text-sm text-muted"><BellRing size={14} className="mr-1 inline" />{task.monitor_state.evidence}</p>}
      <div className="mt-4 flex flex-wrap gap-2">{task.status === "active" ? <Button variant="secondary" onClick={() => action.mutate({ id: task.task_id, action: "pause" })} disabled={action.isPending}><CirclePause size={15} />Pause</Button> : task.status === "paused" ? <Button variant="primary" onClick={() => action.mutate({ id: task.task_id, action: "resume" })} disabled={action.isPending}><CirclePlay size={15} />Resume</Button> : null}{["active", "paused", "awaiting_approval", "failed", "blocked"].includes(task.status) && <Button variant="ghost" onClick={() => action.mutate({ id: task.task_id, action: "run" })} disabled={action.isPending}><Repeat2 size={15} />Run now</Button>}{task.status === "cancelled" && <Button variant="primary" onClick={() => action.mutate({ id: task.task_id, action: "restart" })} disabled={action.isPending}><CirclePlay size={15} />Restart</Button>}<Button variant="danger" onClick={() => action.mutate({ id: task.task_id, action: "cancel" })} disabled={action.isPending || ["cancelled", "completed"].includes(task.status)}><XCircle size={15} />Cancel</Button></div>
    </article>)}</section>
  </>;
}
