import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, Lock, X } from "lucide-react";
import { getJson, sendJson } from "../api";
import { Badge, Button, EmptyState, ErrorState, formatDate, LoadingState, PageHeader } from "../components/ui";

function approvalDetails(value: any) {
  if (typeof value !== "string") return value || {};
  try { return JSON.parse(value || "{}"); } catch { return {}; }
}

export function ApprovalsPage() {
  const client = useQueryClient();
  const approvals = useQuery({
    queryKey: ["approvals"],
    queryFn: async () => {
      const result = await getJson<any>("/api/autonomy/approvals?status=pending&limit=100");
      return result.approvals || [];
    },
    refetchInterval: 5000,
  });
  const decision = useMutation({
    mutationFn: ({ id, approved }: { id: string; approved: boolean }) =>
      sendJson(`/api/autonomy/approvals/${id}/decision`, "POST", { approved }),
    onSuccess: () => {
      client.invalidateQueries({ queryKey: ["approvals"] });
      client.invalidateQueries({ queryKey: ["proactive-inbox"] });
      client.invalidateQueries({ queryKey: ["home"] });
    },
  });
  const items = approvals.data || [];

  return <>
    <PageHeader
      eyebrow="Local control"
      title="Approvals"
      description="Approve or deny bounded actions before Ambient AI uses browser, computer, or other guarded capabilities."
      actions={<Badge tone={items.length ? "warn" : "neutral"}>{items.length} pending</Badge>}
    />
    {approvals.isLoading && <LoadingState />}
    {approvals.isError && <ErrorState error={approvals.error} />}
    {!approvals.isLoading && !items.length && <EmptyState title="No pending approvals" description="New approval requests will appear here when a task needs your decision." />}
    <section className="space-y-3">
      {items.map((item: any) => {
        const raw = approvalDetails(item.constraints_json);
        const computer = item.capability === "computer.use";
        const browser = item.capability === "browser.use";
        const localControl = computer || browser;
        const destination = raw.origin_kind === "direct_chat"
          ? "Original chat"
          : raw.origin_kind === "autonomy"
            ? "Current proactive report"
            : "Proactive Inbox";
        return <article className="record-card" key={item.approval_id}>
          <div className="record-top">
            <div className="flex flex-wrap gap-2"><Badge>{item.capability}</Badge><Badge>{item.status}</Badge></div>
            <time title={`Expires ${formatDate(item.expires_at)}`}>Requested {formatDate(item.created_at)}</time>
          </div>
          <div className="section-label"><Lock size={17} />Pending approval</div>
          <p className="my-3 text-base leading-7">
            {computer
              ? "Allowing this will deploy the computer-use agent for the approved task. Shift+Esc stops only that session."
              : browser
                ? "Allowing this launches the visible Fara research browser for the shown task. It works from screenshots, stays read-only, and Shift+Esc stops the session."
                : "This bounded action needs approval before it can run."}
          </p>
          {localControl && <div className="meta-grid mb-3 text-base">
            <div><strong>Task</strong><span>{raw.arguments?.task || "No task provided"}</span></div>
            <div><strong>Reason</strong><span>{raw.arguments?.reason || "No reason provided"}</span></div>
            <div><strong>Expected result</strong><span>{raw.arguments?.expected_result || "A verified task outcome"}</span></div>
            <div><strong>After completion</strong><span>{raw.arguments?.continuation_instruction || "Finish and report the original goal"}</span></div>
            <div><strong>Returns to</strong><span>{destination}</span></div>
          </div>}
          <div className="mt-3 flex gap-2">
            <Button variant="primary" onClick={() => decision.mutate({ id: item.approval_id, approved: true })} disabled={decision.isPending}>
              <Check size={15} />{computer ? "Allow computer use" : browser ? "Allow visual research" : "Approve once"}
            </Button>
            <Button variant="danger" onClick={() => decision.mutate({ id: item.approval_id, approved: false })} disabled={decision.isPending}>
              <X size={15} />Deny
            </Button>
          </div>
        </article>;
      })}
    </section>
  </>;
}
