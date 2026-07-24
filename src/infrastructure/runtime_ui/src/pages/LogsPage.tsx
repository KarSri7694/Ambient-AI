import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Pause, Play, Trash2 } from "lucide-react";
import { getJson } from "../api";
import { Badge, Button, ErrorState, PageHeader } from "../components/ui";
import type { RuntimeLog } from "../types";

export function LogsPage() {
  const [entries, setEntries] = useState<RuntimeLog[]>([]);
  const [latestId, setLatestId] = useState(0);
  const [paused, setPaused] = useState(false);
  const end = useRef<HTMLDivElement>(null);
  const logs = useQuery({ queryKey: ["runtime-logs", latestId], queryFn: () => getJson<any>(`/api/logs?after_id=${latestId}&limit=500`), refetchInterval: paused ? false : 2000 });
  useEffect(() => {
    const next = logs.data?.entries || [];
    if (!next.length) return;
    setEntries((current) => [...current, ...next].slice(-600));
    setLatestId(Math.max(latestId, ...next.map((entry: RuntimeLog) => entry.id)));
  }, [logs.data]);
  useEffect(() => { if (!paused) end.current?.scrollIntoView({ block: "end" }); }, [entries, paused]);
  return <><PageHeader eyebrow="Live runtime" title="Runtime Logs" description="Follow the most recent application events without leaving the control center." actions={<><Badge tone={paused ? "warn" : "good"}>{paused ? "Paused" : "Streaming"}</Badge><Button variant="secondary" onClick={() => setPaused(!paused)}>{paused ? <Play size={16} /> : <Pause size={16} />}{paused ? "Resume" : "Pause"}</Button><Button variant="ghost" onClick={() => setEntries([])}><Trash2 size={16} />Clear view</Button></>} />{logs.isError && <ErrorState error={logs.error} />}<div className="log-console" role="log" aria-live="polite">{entries.map((entry) => <div className="log-row" key={entry.id}><span className={`log-level ${entry.level.toLowerCase()}`}>{entry.level}</span><time>{new Date(entry.timestamp).toLocaleTimeString()}</time><span>{entry.message}</span></div>)}{!entries.length && <div className="p-8 text-center text-sm text-muted">Waiting for runtime events…</div>}<div ref={end} /></div></>;
}
