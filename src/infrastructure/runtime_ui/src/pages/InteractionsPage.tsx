import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { CalendarDays, ChevronLeft, ChevronRight, Eye, ImageIcon, LockKeyhole, RefreshCw } from "lucide-react";
import { getJson, queryString } from "../api";
import { Badge, Button, EmptyState, ErrorState, formatDate, ImageModal, JsonDetails, LoadingState, Markdown, PageHeader } from "../components/ui";
import type { InteractionInput, InteractionPage } from "../types";

const PAGE_SIZE = 50;

export function InteractionsPage() {
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [sort, setSort] = useState<"newest" | "oldest">("newest");
  const [offset, setOffset] = useState(0);
  const [revealed, setRevealed] = useState<Record<string, InteractionInput>>({});
  const [revealErrors, setRevealErrors] = useState<Record<string, string>>({});
  const [revealing, setRevealing] = useState<string | null>(null);
  const [expandedImage, setExpandedImage] = useState<string | null>(null);
  const query = queryString({ date_from: dateFrom, date_to: dateTo, sort, limit: PAGE_SIZE, offset });
  const interactions = useQuery({
    queryKey: ["interactions", dateFrom, dateTo, sort, offset],
    queryFn: () => getJson<InteractionPage>(`/api/interactions?${query}`),
    refetchInterval: 5000,
  });
  const updateFilter = (callback: () => void) => { callback(); setOffset(0); };
  const reveal = async (id: string) => {
    setRevealing(id);
    setRevealErrors((current) => ({ ...current, [id]: "" }));
    try {
      const payload = await getJson<{ input: InteractionInput }>(`/api/interactions/${id}/input`);
      setRevealed((current) => ({ ...current, [id]: payload.input }));
    } catch (error) {
      setRevealErrors((current) => ({ ...current, [id]: error instanceof Error ? error.message : "Could not reveal this input." }));
    } finally {
      setRevealing(null);
    }
  };
  const total = interactions.data?.pagination.total || 0;
  const pageNumber = Math.floor(offset / PAGE_SIZE) + 1;
  const pageCount = Math.max(1, Math.ceil(total / PAGE_SIZE));

  return (
    <>
      <PageHeader eyebrow="Model audit trail" title="Interaction Logs" description="Inspect the exact request and response pair for every model interaction, including attached visual context and diagnostics." actions={<Button variant="secondary" onClick={() => interactions.refetch()}><RefreshCw size={16} />Refresh</Button>} />
      <section className="filter-panel" aria-label="Interaction filters">
        <label><span>From</span><div className="input-icon"><CalendarDays size={16} /><input type="date" value={dateFrom} max={dateTo || undefined} onChange={(event) => updateFilter(() => setDateFrom(event.target.value))} /></div></label>
        <label><span>To</span><div className="input-icon"><CalendarDays size={16} /><input type="date" value={dateTo} min={dateFrom || undefined} onChange={(event) => updateFilter(() => setDateTo(event.target.value))} /></div></label>
        <label><span>Order</span><select value={sort} onChange={(event) => updateFilter(() => setSort(event.target.value as "newest" | "oldest"))}><option value="newest">Newest first</option><option value="oldest">Oldest first</option></select></label>
        <div className="ml-auto self-end"><Button variant="ghost" onClick={() => { setDateFrom(""); setDateTo(""); setSort("newest"); setOffset(0); }}>Clear filters</Button></div>
      </section>

      <div className="mb-3 flex items-center justify-between text-sm text-muted"><span>{total.toLocaleString()} interactions</span><span>Page {pageNumber} of {pageCount}</span></div>
      {interactions.isLoading && <LoadingState label="Loading interactions" />}
      {interactions.isError && <ErrorState error={interactions.error} />}
      {!interactions.isLoading && !interactions.data?.items.length && <EmptyState title="No interactions found" description="Try a wider date range or wait for the runtime to record its next model call." />}
      <div className="space-y-5">
        {interactions.data?.items.map((interaction) => {
          const input = revealed[interaction.interaction_id] || interaction.input;
          const status = interaction.error_text ? "Failed" : interaction.response_text ? "Completed" : "Incomplete";
          return (
            <article key={interaction.interaction_id} className="interaction-card">
              <header className="interaction-head">
                <div className="flex flex-wrap items-center gap-2"><Badge tone={interaction.error_text ? "danger" : "good"}>{status}</Badge><Badge>{interaction.source || "unknown source"}</Badge><Badge>{interaction.model || "unknown model"}</Badge></div>
                <div className="text-right text-xs text-muted"><p>{formatDate(interaction.created_at)}</p><p>{interaction.duration_ms == null ? "Duration unavailable" : `${(interaction.duration_ms / 1000).toFixed(2)}s`}</p></div>
              </header>

              <div className="interaction-thread">
                <div className="request-wrap">
                  <p className="bubble-label">Request</p>
                  <div className="request-bubble">
                    {input.protected && !input.request ? (
                      <div className="flex flex-col items-start gap-3">
                        <div className="flex items-center gap-2 font-medium"><LockKeyhole size={17} />Protected ambient input</div>
                        <p className="text-sm opacity-80">This prompt is stored in the protected capture store and is only loaded when you reveal it.</p>
                        <Button variant="secondary" onClick={() => reveal(interaction.interaction_id)} disabled={revealing === interaction.interaction_id}><Eye size={16} />{revealing === interaction.interaction_id ? "Revealing…" : "Reveal input"}</Button>
                        {revealErrors[interaction.interaction_id] && <p className="text-sm text-danger">{revealErrors[interaction.interaction_id]}</p>}
                      </div>
                    ) : input.malformed ? <p className="italic opacity-75">The saved request payload is malformed and could not be rendered.</p>
                      : input.request ? <p className="whitespace-pre-wrap">{input.request.content || "The request contained no text."}</p>
                      : <p className="italic opacity-75">No request message was recorded.</p>}
                    {interaction.has_image && interaction.image_url && (
                      <button className="interaction-image" type="button" onClick={() => setExpandedImage(interaction.image_url!)} aria-label="Enlarge interaction image">
                        <img src={interaction.image_url} alt="Visual input attached to this interaction" loading="lazy" /><span><ImageIcon size={15} />Click to enlarge</span>
                      </button>
                    )}
                  </div>
                </div>

                <div className="response-wrap">
                  <p className="bubble-label">Ambient AI</p>
                  <div className={`response-bubble ${interaction.error_text ? "failed" : ""}`}>
                    {interaction.response_text ? <Markdown>{interaction.response_text}</Markdown> : <p className="italic text-muted">{interaction.error_text || "No response was recorded."}</p>}
                  </div>
                </div>
              </div>

              <div className="mt-4 space-y-2">
                {input.context_messages.length > 0 && <JsonDetails label={`Request context (${input.context_messages.length} messages)`} value={input.context_messages} />}
                <JsonDetails label="Reasoning" value={interaction.reasoning_text} />
                <JsonDetails label="Tools offered" value={interaction.tools} />
                <JsonDetails label="Tool calls" value={interaction.tool_calls} />
                <JsonDetails label="Metadata" value={interaction.metadata} />
                <JsonDetails label="Report" value={interaction.report} />
                {interaction.error_text && <JsonDetails label="Error details" value={interaction.error_text} open />}
              </div>
            </article>
          );
        })}
      </div>
      {total > 0 && <div className="mt-6 flex items-center justify-between"><Button variant="secondary" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}><ChevronLeft size={16} />Previous</Button><span className="text-sm text-muted">Showing {offset + 1}–{Math.min(offset + PAGE_SIZE, total)} of {total}</span><Button variant="secondary" disabled={!interactions.data?.pagination.has_more} onClick={() => setOffset(offset + PAGE_SIZE)}>Next<ChevronRight size={16} /></Button></div>}
      <ImageModal src={expandedImage} alt="Expanded interaction image" onClose={() => setExpandedImage(null)} />
    </>
  );
}
