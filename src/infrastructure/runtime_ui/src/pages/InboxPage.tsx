import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, Gauge, Lock, Save, ScanSearch, ShieldCheck, ThumbsDown, ThumbsUp, X } from "lucide-react";
import { getJson, sendJson } from "../api";
import { Badge, Button, EmptyState, ErrorState, formatDate, JsonDetails, LoadingState, PageHeader } from "../components/ui";

function approvalDetails(value: any) {
  if (typeof value !== "string") return value || {};
  try { return JSON.parse(value || "{}"); } catch { return {}; }
}

function timestamp(value: any): number {
  const parsed = Date.parse(String(value || ""));
  return Number.isFinite(parsed) ? parsed : 0;
}

export function InboxPage({ privacy, resources }: { privacy: any; resources: any }) {
  const client = useQueryClient();
  const [apps, setApps] = useState("");
  const [domains, setDomains] = useState("");
  const [editingExclusions, setEditingExclusions] = useState(false);
  const [feedbackNotice, setFeedbackNotice] = useState("");
  useEffect(() => {
    if (editingExclusions) return;
    setApps((privacy?.capture?.excluded_apps || []).join(", "));
    setDomains((privacy?.capture?.excluded_domains || []).join(", "));
  }, [privacy?.capture?.excluded_apps, privacy?.capture?.excluded_domains, editingExclusions]);
  const inbox = useQuery({ queryKey: ["proactive-inbox"], queryFn: async () => {
    const [items, approvals, policies] = await Promise.all([getJson<any>("/api/autonomy/inbox?limit=100"), getJson<any>("/api/autonomy/approvals?status=pending&limit=100"), getJson<any>("/api/autonomy/policies")]);
    return { items: items.items || [], approvals: approvals.approvals || [], policies: policies.policies || [] };
  }, refetchInterval: 5000 });
  const refresh = () => client.invalidateQueries({ queryKey: ["proactive-inbox"] });
  const feedback = useMutation({ mutationFn: ({ id, value }: { id: string; value: string }) => sendJson<any>(`/api/autonomy/inbox/${id}/feedback`, "POST", { feedback: value }), onSuccess: (result) => { setFeedbackNotice(result.effect || "Preference saved locally."); refresh(); } });
  const approval = useMutation({ mutationFn: ({ id, approved }: { id: string; approved: boolean }) => sendJson(`/api/autonomy/approvals/${id}/decision`, "POST", { approved }), onSuccess: refresh });
  const policy = useMutation({ mutationFn: ({ capability, decision }: { capability: string; decision: string }) => sendJson(`/api/autonomy/policies/${encodeURIComponent(capability)}`, "PUT", { decision }), onSuccess: refresh });
  const exclusions = useMutation({ mutationFn: () => sendJson("/api/privacy/capture/exclusions", "PUT", { apps: apps.split(",").map((v) => v.trim()).filter(Boolean), domains: domains.split(",").map((v) => v.trim()).filter(Boolean) }), onSuccess: () => { setEditingExclusions(false); client.invalidateQueries({ queryKey: ["privacy-status"] }); } });
  const foregroundCheck = useMutation({ mutationFn: () => sendJson<any>("/api/privacy/capture/exclusions/check", "POST"), onSuccess: () => client.invalidateQueries({ queryKey: ["privacy-status"] }) });
  const preset = useMutation({ mutationFn: (value: string) => sendJson("/api/runtime/resource-policy", "PUT", { preset: value }), onSuccess: () => client.invalidateQueries({ queryKey: ["resource-status"] }) });
  const items = [...(inbox.data?.items || [])].sort(
    (left: any, right: any) =>
      timestamp(right.created_at || right.updated_at) - timestamp(left.created_at || left.updated_at)
      || timestamp(right.updated_at) - timestamp(left.updated_at)
  );
  const snapshot = resources?.snapshot || {};
  const freeRam = Number(snapshot.available_ram_mb || 0) / 1024;
  const freeVram = snapshot.free_vram_mb == null ? null : Number(snapshot.free_vram_mb) / 1024;
  const capture = privacy?.capture || {};
  const lastDecision = foregroundCheck.data?.decision || capture.last_decision;
  const lastContext = foregroundCheck.data?.context || capture.last_context || lastDecision?.detected;
  return <><PageHeader eyebrow="Judgment-driven assistance" title="Proactive Inbox" description="Review timely opportunities, approve bounded actions, and tune the runtime’s privacy and resource guardrails." actions={<Badge tone={items.length ? "warn" : "neutral"}>{items.length} opportunities</Badge>} />
    {inbox.isLoading && <LoadingState />}{inbox.isError && <ErrorState error={inbox.error} />}
    <div className="two-column wide-left">
      <section className="space-y-3">{feedbackNotice && <div className="sub-card text-sm text-good">Preference saved locally · {feedbackNotice}</div>}{!items.length && !inbox.isLoading && <EmptyState title="Inbox is clear" description="New proactive opportunities will appear when Ambient AI finds something timely and useful." />}{items.map((item: any) => <article className="record-card" key={item.inbox_id}><div className="record-top"><div className="flex gap-2"><Badge>{item.status}</Badge><Badge>Confidence {Number(item.confidence || 0).toFixed(2)}</Badge></div><time title={`Updated ${formatDate(item.updated_at)}`}>Found {formatDate(item.created_at || item.updated_at)}</time></div><h2>{item.title}</h2><p>{item.summary}</p><div className="meta-grid"><div><strong>Why now</strong><span>{item.why_now}</span></div></div><JsonDetails label="Evidence and sources" value={item.sources_json} /><JsonDetails label="Personalization used" value={item.personalization_json} /><JsonDetails label="Actions and verification" value={item.actions_json} /><JsonDetails label="Detailed result" value={item.detailed_report} /><div className="mt-4 flex flex-wrap gap-2"><Button variant="secondary" onClick={() => feedback.mutate({ id: item.inbox_id, value: "useful" })}><ThumbsUp size={15} />Useful</Button><Button variant="secondary" onClick={() => feedback.mutate({ id: item.inbox_id, value: "not_useful" })}><ThumbsDown size={15} />Not useful</Button><Button variant="ghost" onClick={() => feedback.mutate({ id: item.inbox_id, value: "wrong_inference" })}>Wrong inference</Button><Button variant="ghost" onClick={() => feedback.mutate({ id: item.inbox_id, value: "too_intrusive" })}>Too intrusive</Button>{item.feedback && <Badge tone="good">{item.feedback}</Badge>}</div>{item.feedback_history?.length > 0 && <p className="mt-3 text-xs text-muted">Preference audit · {item.feedback_history[0].feedback.replaceAll("_", " ")} saved {formatDate(item.feedback_history[0].created_at)}</p>}</article>)}</section>
      <aside className="space-y-4">
        <section className="panel p-4"><div className="section-label"><Lock size={17} />Pending approvals</div>{!inbox.data?.approvals.length && <p className="text-sm text-muted">No actions need approval.</p>}{inbox.data?.approvals.map((item: any) => { const raw = approvalDetails(item.constraints_json); const computer = item.capability === "computer.use"; const browser = item.capability === "browser.use"; const localControl = computer || browser; const destination = raw.origin_kind === "direct_chat" ? "Original chat" : raw.origin_kind === "autonomy" ? "Current proactive report" : "Proactive Inbox"; return <div className="sub-card" key={item.approval_id}><Badge>{item.capability}</Badge><p className="my-3 text-sm">{computer ? "Allowing this will deploy the computer-use agent for the approved task. Shift+Esc stops only that session." : browser ? "Allowing this launches the visible Fara research browser for the shown task. It works from screenshots, stays read-only, and Shift+Esc stops the session." : "This bounded action needs approval before it can run."}</p>{localControl && <div className="meta-grid mb-3"><div><strong>Task</strong><span>{raw.arguments?.task || "No task provided"}</span></div><div><strong>Reason</strong><span>{raw.arguments?.reason || "No reason provided"}</span></div><div><strong>Expected result</strong><span>{raw.arguments?.expected_result || "A verified task outcome"}</span></div><div><strong>After completion</strong><span>{raw.arguments?.continuation_instruction || "Finish and report the original goal"}</span></div><div><strong>Returns to</strong><span>{destination}</span></div></div>}<pre>{JSON.stringify(raw, null, 2)}</pre><div className="mt-3 flex gap-2"><Button variant="primary" onClick={() => approval.mutate({ id: item.approval_id, approved: true })}><Check size={15} />{computer ? "Allow computer use" : browser ? "Allow visual research" : "Approve once"}</Button><Button variant="danger" onClick={() => approval.mutate({ id: item.approval_id, approved: false })}><X size={15} />Deny</Button></div></div>; })}</section>
        <section className="panel p-4"><div className="section-label"><ShieldCheck size={17} />Capability policies</div><div className="space-y-3">{inbox.data?.policies.map((item: any) => <label className="flex items-center justify-between gap-3 text-sm" key={item.capability}><span className="font-medium">{item.capability}</span><select value={item.decision} onChange={(event) => policy.mutate({ capability: item.capability, decision: event.target.value })}><option value="deny">deny</option><option value="ask">ask</option><option value="auto_reversible">auto reversible</option><option value="trusted_bounded">trusted bounded</option></select></label>)}</div></section>
        <section className="panel p-4">
          <div className="section-label"><Save size={17} />Screen capture exclusions</div>
          <div className="form-stack">
            <p className="text-xs leading-5 text-muted">Matching applications or sites are blocked before a screenshot is taken or queued. Audio capture is unaffected.</p>
            <label>Applications or executables<input value={apps} onChange={(event) => { setApps(event.target.value); setEditingExclusions(true); }} placeholder="1password.exe, Signal, Visual Studio Code" /></label>
            <label>Domains<input value={domains} onChange={(event) => { setDomains(event.target.value); setEditingExclusions(true); }} placeholder="bank.example, localhost, 10.0.0.4" /></label>
            <div className="flex flex-wrap gap-2">
              <Button variant="secondary" onClick={() => exclusions.mutate()} disabled={exclusions.isPending}>{exclusions.isPending ? "Saving..." : "Save exclusions"}</Button>
              <Button variant="ghost" onClick={() => foregroundCheck.mutate()} disabled={foregroundCheck.isPending}><ScanSearch size={15} />{foregroundCheck.isPending ? "Checking..." : "Check foreground"}</Button>
            </div>
            {exclusions.isSuccess && <p className="text-xs text-good">Saved locally. The new policy applies only to future screen captures.</p>}
            {exclusions.isError && <p className="text-xs text-danger">{exclusions.error instanceof Error ? exclusions.error.message : "Could not save exclusions."}</p>}
            {foregroundCheck.isError && <p className="text-xs text-danger">{foregroundCheck.error instanceof Error ? foregroundCheck.error.message : "Could not inspect the foreground window."}</p>}
            {lastDecision && <div className="sub-card">
              <div className="mb-2 flex flex-wrap items-center gap-2"><Badge tone={lastDecision.excluded ? "warn" : "good"}>{lastDecision.excluded ? "Capture blocked" : "Capture allowed"}</Badge><span className="text-xs text-muted">Policy r{lastDecision.policy_revision} · {formatDate(lastDecision.checked_at)}</span></div>
              <div className="meta-grid"><div><strong>Process / app</strong><span>{lastContext?.process_name || lastContext?.app_name || "Not detected"}</span></div><div><strong>Domain</strong><span>{lastContext?.domain || "Not detected"}</span></div>{lastDecision.matched_rule && <div><strong>Matched rule</strong><span>{lastDecision.match_type}: {lastDecision.matched_rule}</span></div>}</div>
            </div>}
            <p className="text-xs text-muted">{capture.persistence?.enabled ? `Persisted locally · revision ${capture.policy_revision || 1}` : "Persistence unavailable in this runtime"}</p>
            <p className="text-xs text-muted">{privacy?.storage_pressure ? "Storage pressure detected. Plain captures will not be auto-deleted." : `Plain captures: ${Math.round(Number(privacy?.capture_size_bytes || 0) / 1048576)} MB · ${privacy?.capture_root || "capture folder"}`}</p>
          </div>
        </section>
        <section className="panel p-4"><div className="section-label"><Gauge size={17} />Resource policy</div><div className="form-stack"><label>Inference preset<select value={resources?.preset || "balanced"} onChange={(event) => preset.mutate(event.target.value)}><option value="capture_only">Capture only</option><option value="balanced">Balanced</option><option value="aggressive">Aggressive</option></select></label><p className="text-xs leading-5 text-muted">{freeRam.toFixed(1)} GB RAM free · {freeVram == null ? "GPU telemetry unavailable" : `${freeVram.toFixed(1)} GB VRAM free`} · model: {resources?.residency?.loaded_model || "on demand"}</p><p className="text-xs text-muted">{Number(resources?.event_counts?.resource_deferred || 0)} resource-deferred · {Number(resources?.event_counts?.pending || 0)} pending events</p></div></section>
      </aside>
    </div>
  </>;
}
