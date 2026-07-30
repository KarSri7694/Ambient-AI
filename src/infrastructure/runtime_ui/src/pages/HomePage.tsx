import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Archive, ArrowLeft, ArrowRight, Bot, Check, ChevronRight, Clock3, FileText,
  Inbox, ListTodo, RefreshCw, ShieldAlert, Sparkles, X,
} from "lucide-react";
import { getJson, sendJson } from "../api";
import { Badge, Button, EmptyState, ErrorState, formatDate, LoadingState, Markdown } from "../components/ui";

function localDate(value = new Date()): string {
  const offset = value.getTimezoneOffset() * 60_000;
  return new Date(value.getTime() - offset).toISOString().slice(0, 10);
}

function moveDate(value: string, days: number): string {
  const date = new Date(`${value}T12:00:00`);
  date.setDate(date.getDate() + days);
  return localDate(date);
}

function visitKey(value: string): string {
  return `ambient-home-last-visited:${value}`;
}

function tone(status: string): "neutral" | "good" | "warn" | "danger" {
  if (["completed", "active", "processed", "approved"].includes(status)) return "good";
  if (["failed", "blocked", "dead_letter", "completed_with_blocker"].includes(status)) return "danger";
  if (["pending", "queued", "resource_deferred", "running"].includes(status)) return "warn";
  return "neutral";
}

function dateHeading(value: string, today: string): string {
  if (value === today) return "Today";
  if (value === moveDate(today, -1)) return "Yesterday";
  return new Date(`${value}T12:00:00`).toLocaleDateString(undefined, { weekday: "long", month: "long", day: "numeric" });
}

export function HomePage({ onNavigate }: { onNavigate: (path: string) => void }) {
  const client = useQueryClient();
  const today = localDate();
  const [selectedDate, setSelectedDate] = useState(today);
  const [since, setSince] = useState<string | null>(() => localStorage.getItem(visitKey(today)));
  const recordedVisits = useRef(new Set<string>());
  useEffect(() => setSince(localStorage.getItem(visitKey(selectedDate))), [selectedDate]);

  const query = useQuery({
    queryKey: ["home", selectedDate, since],
    queryFn: () => getJson<any>(`/api/home?date=${selectedDate}${since ? `&since=${encodeURIComponent(since)}` : ""}`),
    refetchInterval: 15_000,
  });
  useEffect(() => {
    if (!query.data?.server_time || recordedVisits.current.has(selectedDate)) return;
    recordedVisits.current.add(selectedDate);
    localStorage.setItem(visitKey(selectedDate), query.data.server_time);
  }, [query.data?.server_time, selectedDate]);

  const approval = useMutation({
    mutationFn: ({ id, approved }: { id: string; approved: boolean }) =>
      sendJson(`/api/autonomy/approvals/${id}/decision`, "POST", { approved }),
    onSuccess: () => {
      client.invalidateQueries({ queryKey: ["home"] });
      client.invalidateQueries({ queryKey: ["proactive-inbox"] });
    },
  });
  const data = query.data;
  const counts = data?.counts || {};
  const timeline = data?.timeline || [];
  const attention = data?.attention || [];
  const briefing = data?.briefing;
  const briefingRefresh = data?.briefing_refresh || {};
  const stats = useMemo(() => [
    { label: "Work completed", value: Number(counts.reports || 0) + Number(counts.activity_runs || 0), icon: Bot },
    { label: "Proactive updates", value: counts.proactive_updates || 0, icon: Inbox },
    { label: "Artifacts changed", value: counts.artifact_changes || 0, icon: Archive },
    { label: "Needs attention", value: counts.attention || 0, icon: ShieldAlert },
  ], [counts]);

  return <div className="home-page">
    <section className="home-hero">
      <div className="home-hero-copy">
        <div className="home-spark"><Sparkles size={22} /></div>
        <div>
          <p className="eyebrow">Your ambient day</p>
          <h1>{dateHeading(selectedDate, today)}</h1>
          <p>What Ambient AI finished, learned, organized, and left for your decision.</p>
        </div>
      </div>
      <div className="home-date-nav">
        <Button variant="secondary" onClick={() => setSelectedDate(moveDate(selectedDate, -1))} aria-label="Previous day"><ArrowLeft size={16} /></Button>
        <span>{new Date(`${selectedDate}T12:00:00`).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" })}</span>
        <Button variant="secondary" disabled={selectedDate >= today} onClick={() => setSelectedDate(moveDate(selectedDate, 1))} aria-label="Next day"><ArrowRight size={16} /></Button>
        {selectedDate !== today && <Button variant="ghost" onClick={() => setSelectedDate(today)}>Today</Button>}
        <Button variant="ghost" onClick={() => query.refetch()} disabled={query.isFetching} aria-label="Refresh Home"><RefreshCw className={query.isFetching ? "animate-spin" : ""} size={16} /></Button>
      </div>
    </section>

    {query.isLoading && <LoadingState label="Gathering your daily activity" />}
    {query.isError && <ErrorState error={query.error} />}
    {data && <>
      <section className="home-briefing">
        <div className="home-section-head">
          <div><p className="eyebrow">Latest update</p><h2>{briefing && !data.briefing_stale ? briefing.headline : data.latest_headline}</h2></div>
          <div className="flex flex-wrap gap-2">
            {data.new_count > 0 && <Badge tone="good">{data.new_count} new since last visit</Badge>}
            {data.briefing_pending && briefingRefresh.running && <Badge tone="warn">Personalized digest refreshing</Badge>}
            {data.briefing_pending && !briefingRefresh.running && briefingRefresh.last_error && <Badge tone="danger">Digest retry scheduled</Badge>}
            {data.briefing_pending && !briefingRefresh.running && !briefingRefresh.last_error && <Badge tone="warn">Personalized digest pending</Badge>}
          </div>
        </div>
        <div className="home-overview">
          <Markdown>{briefing && !data.briefing_stale ? briefing.overview : data.latest_narrative}</Markdown>
        </div>
        {briefing && !data.briefing_stale && <>
          <div className="home-brief-columns">
            <BriefList title="What I accomplished" items={briefing.accomplishments} />
            <BriefList title="What I learned for you" items={briefing.updates} />
            <BriefList title="What did not work" items={briefing.failures} />
            <BriefList title="What needs you" items={briefing.attention} />
          </div>
          <p className="home-generated">Personalized from your local profile and prepared {formatDate(briefing.generated_at)}</p>
        </>}
        {(!briefing || data.briefing_stale) && <p className="home-generated">This latest summary is built directly from current runtime outcomes. {briefingRefresh.last_error ? `The deeper personalized digest will retry after ${formatDate(briefingRefresh.retry_after)}.` : "Ambient AI will replace it with a deeper profile-aware narrative in the next idle resource window."}</p>}
      </section>

      <section className="home-stat-grid">
        {stats.map(({ label, value, icon: Icon }) => <article className="home-stat" key={label}><span><Icon size={18} /></span><strong>{value}</strong><small>{label}</small></article>)}
      </section>

      <div className="home-layout">
        <div className="space-y-5">
          <section className="panel overflow-hidden">
            <div className="home-section-head border-b border-line p-5"><div><p className="eyebrow">Activity timeline</p><h2>What the agent did</h2></div><Badge>{timeline.length} updates</Badge></div>
            {!timeline.length && <div className="p-5"><EmptyState title="A quiet day" description="Completed agent work, proactive reports, delegated actions, and artifact edits will appear here." /></div>}
            <div className="home-timeline">
              {timeline.map((item: any) => <article className="home-timeline-item" key={item.id}>
                <span className={`home-timeline-dot ${tone(item.status)}`} />
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2"><Badge tone={tone(item.status)}>{item.status.replaceAll("_", " ")}</Badge><Badge>{item.kind.replaceAll("_", " ")}</Badge>{item.is_new && <Badge tone="good">New</Badge>}<time>{formatDate(item.occurred_at)}</time></div>
                  <h3>{item.title}</h3><p>{item.summary || "No additional summary."}</p>
                </div>
                <Button variant="ghost" onClick={() => onNavigate(item.destination)} title="Inspect in its full view">Inspect <ChevronRight size={15} /></Button>
              </article>)}
            </div>
          </section>
        </div>

        <aside className="space-y-5">
          <section className="panel p-5">
            <div className="home-section-head"><div><p className="eyebrow">Action center</p><h2>Needs attention</h2></div>{attention.length > 0 && <Badge tone="warn">{attention.length}</Badge>}</div>
            {!attention.length && <p className="text-sm leading-6 text-muted">Nothing needs your decision right now.</p>}
            <div className="space-y-3">
              {attention.map((item: any) => <article className="home-attention" key={item.id}>
                <div className="flex flex-wrap items-center gap-2"><Badge tone={tone(item.status)}>{item.status.replaceAll("_", " ")}</Badge>{item.is_new && <Badge tone="good">New</Badge>}</div>
                <h3>{item.title}</h3><p>{item.summary}</p>
                {item.kind === "approval" ? <div className="mt-3 flex gap-2"><Button variant="primary" onClick={() => approval.mutate({ id: item.approval_id, approved: true })} disabled={approval.isPending}><Check size={15} />Approve</Button><Button variant="danger" onClick={() => approval.mutate({ id: item.approval_id, approved: false })} disabled={approval.isPending}><X size={15} />Deny</Button></div> : <Button className="mt-3" variant="ghost" onClick={() => onNavigate(item.destination)}>Inspect <ChevronRight size={15} /></Button>}
              </article>)}
            </div>
          </section>

          <section className="panel p-5">
            <div className="section-label"><Clock3 size={17} />Background pulse</div>
            <div className="home-pulse-grid">
              <Pulse label="Events processed" value={data.background?.events?.processed || 0} />
              <Pulse label="Events pending" value={data.background?.events?.pending || 0} />
              <Pulse label="Maintenance runs" value={data.background?.artifact_maintenance_runs || 0} />
              <Pulse label="Queued tasks" value={counts.queued_tasks || 0} />
            </div>
          </section>

          <section className="panel p-5">
            <div className="section-label"><ListTodo size={17} />Open the details</div>
            <div className="grid gap-2">
              <QuickLink icon={<Inbox size={16} />} label="Proactive inbox" onClick={() => onNavigate("/inbox")} />
              <QuickLink icon={<FileText size={16} />} label="Reports" onClick={() => onNavigate("/reports")} />
              <QuickLink icon={<Archive size={16} />} label="Artifact library" onClick={() => onNavigate("/artifacts")} />
            </div>
          </section>
        </aside>
      </div>
    </>}
  </div>;
}

function BriefList({ title, items }: { title: string; items?: string[] }) {
  return <div><h3>{title}</h3>{items?.length ? <ul>{items.map((item) => <li key={item}>{item}</li>)}</ul> : <p>Nothing to add.</p>}</div>;
}

function Pulse({ label, value }: { label: string; value: number }) {
  return <div><strong>{value}</strong><span>{label}</span></div>;
}

function QuickLink({ icon, label, onClick }: { icon: React.ReactNode; label: string; onClick: () => void }) {
  return <button className="home-quick-link" type="button" onClick={onClick}><span>{icon}</span><strong>{label}</strong><ChevronRight size={15} /></button>;
}
