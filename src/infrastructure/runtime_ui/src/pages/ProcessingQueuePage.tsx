import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AudioLines, Clock3, HardDrive, ImageIcon, ListFilter, RefreshCw, Trash2 } from "lucide-react";
import { getJson, sendJson } from "../api";
import { Badge, Button, EmptyState, ErrorState, LoadingState, PageHeader, formatDate } from "../components/ui";

type QueueModality = "image" | "audio";

interface ProcessingQueueItem {
  event_id: string;
  modality: QueueModality;
  event_type: string;
  status: string;
  deletable: boolean;
  original_name: string;
  mime_type: string;
  size_bytes?: number | null;
  duration_seconds?: number | null;
  occurred_at: string;
  available_at?: string | null;
  attempt_count: number;
  error_text?: string | null;
  preview_url: string;
}

interface ProcessingQueuePayload {
  items: ProcessingQueueItem[];
  count: number;
  image_count: number;
  audio_count: number;
  processing_count: number;
}

function formatBytes(value?: number | null): string {
  if (value == null || !Number.isFinite(value)) return "Size unavailable";
  if (value < 1024) return `${value} B`;
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1024 ** 2).toFixed(1)} MB`;
}

function statusTone(status: string): "neutral" | "good" | "warn" | "danger" {
  if (status === "leased") return "good";
  if (status === "resource_deferred") return "warn";
  return status === "pending" ? "neutral" : "danger";
}

export function ProcessingQueuePage() {
  const queryClient = useQueryClient();
  const [filter, setFilter] = useState<"all" | QueueModality>("all");
  const queue = useQuery({
    queryKey: ["processing-queue"],
    queryFn: () => getJson<ProcessingQueuePayload>("/api/processing-queue?limit=1000"),
    refetchInterval: 2500,
  });
  const remove = useMutation({
    mutationFn: (eventId: string) => sendJson(`/api/processing-queue/${eventId}`, "DELETE"),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["processing-queue"] }),
  });
  const items = useMemo(
    () => [...(queue.data?.items || [])]
      .filter((item) => filter === "all" || item.modality === filter)
      .sort((left, right) => Date.parse(right.occurred_at) - Date.parse(left.occurred_at)),
    [queue.data?.items, filter],
  );

  const removeItem = (item: ProcessingQueueItem) => {
    if (!item.deletable) return;
    if (window.confirm(`Remove ${item.original_name} from the processing queue and delete its stored media file?`)) {
      remove.mutate(item.event_id);
    }
  };

  return (
    <div>
      <PageHeader
        eyebrow="Perception pipeline"
        title="Processing Queue"
        description="Review image and audio inputs that Ambient AI has not finished processing. Removing an item also deletes its stored capture."
        actions={
          <Button variant="secondary" onClick={() => queue.refetch()} disabled={queue.isFetching}>
            <RefreshCw className={queue.isFetching ? "animate-spin" : ""} size={16} />Refresh
          </Button>
        }
      />

      <section className="mb-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <QueueStat icon={<ListFilter size={18} />} label="Remaining" value={queue.data?.count || 0} />
        <QueueStat icon={<ImageIcon size={18} />} label="Images" value={queue.data?.image_count || 0} />
        <QueueStat icon={<AudioLines size={18} />} label="Audio files" value={queue.data?.audio_count || 0} />
        <QueueStat icon={<Clock3 size={18} />} label="Processing now" value={queue.data?.processing_count || 0} />
      </section>

      <section className="panel overflow-hidden">
        <header className="flex flex-col gap-3 border-b border-line p-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className="font-bold">Pending media</h2>
            <p className="mt-1 text-xs text-muted">Active items cannot be removed until their current processing lease ends.</p>
          </div>
          <div className="segmented" aria-label="Filter processing queue">
            {(["all", "image", "audio"] as const).map((value) => (
              <button key={value} type="button" className={filter === value ? "active" : ""} onClick={() => setFilter(value)}>
                {value === "all" ? "All" : value === "image" ? "Images" : "Audio"}
              </button>
            ))}
          </div>
        </header>

        {queue.isLoading && <LoadingState label="Loading processing queue" />}
        {queue.isError && <div className="p-4"><ErrorState error={queue.error} /></div>}
        {remove.isError && <div className="p-4 pb-0"><ErrorState error={remove.error} /></div>}
        {!queue.isLoading && !queue.isError && !items.length && (
          <EmptyState
            title={filter === "all" ? "Processing queue is clear" : `No ${filter} inputs remain`}
            description="New passive captures will appear here while they wait for vision or audio processing."
          />
        )}

        {!!items.length && (
          <div className="grid gap-4 p-4 md:grid-cols-2 2xl:grid-cols-3">
            {items.map((item) => (
              <article key={item.event_id} className="overflow-hidden rounded-2xl border border-line bg-canvas">
                <div className="flex aspect-video items-center justify-center overflow-hidden bg-black/90">
                  {item.modality === "image" ? (
                    <img className="h-full w-full object-contain" src={item.preview_url} alt={item.original_name} loading="lazy" />
                  ) : (
                    <div className="w-full px-5 text-white">
                      <div className="mb-4 flex items-center justify-center"><AudioLines size={34} /></div>
                      <audio className="w-full" controls preload="metadata" src={item.preview_url} />
                    </div>
                  )}
                </div>
                <div className="space-y-4 p-4">
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <div className="mb-1 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-muted">
                        {item.modality === "image" ? <ImageIcon size={14} /> : <AudioLines size={14} />}
                        {item.modality}
                      </div>
                      <h3 className="truncate font-bold" title={item.original_name}>{item.original_name}</h3>
                    </div>
                    <Badge tone={statusTone(item.status)}>{item.status === "leased" ? "processing" : item.status.replaceAll("_", " ")}</Badge>
                  </div>

                  <div className="grid grid-cols-2 gap-2 text-xs text-muted">
                    <span className="flex items-center gap-1.5"><HardDrive size={13} />{formatBytes(item.size_bytes)}</span>
                    <span className="flex items-center justify-end gap-1.5"><Clock3 size={13} />{formatDate(item.occurred_at)}</span>
                    {item.duration_seconds != null && <span>{Number(item.duration_seconds).toFixed(1)} seconds</span>}
                    {item.attempt_count > 0 && <span className="text-right">{item.attempt_count} attempt{item.attempt_count === 1 ? "" : "s"}</span>}
                  </div>

                  {item.error_text && <p className="rounded-xl bg-soft p-3 text-xs leading-5 text-muted">{item.error_text}</p>}
                  <Button
                    className="w-full"
                    variant="danger"
                    disabled={!item.deletable || (remove.isPending && remove.variables === item.event_id)}
                    onClick={() => removeItem(item)}
                    title={item.deletable ? "Remove from queue and delete stored media" : "This item is currently processing"}
                  >
                    <Trash2 size={15} />
                    {!item.deletable ? "Currently processing" : "Remove from queue"}
                  </Button>
                </div>
              </article>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

function QueueStat({ icon, label, value }: { icon: React.ReactNode; label: string; value: number }) {
  return (
    <div className="panel flex items-center gap-3 p-4">
      <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-soft text-accent">{icon}</span>
      <div><strong className="block text-xl">{value}</strong><span className="text-xs text-muted">{label}</span></div>
    </div>
  );
}
