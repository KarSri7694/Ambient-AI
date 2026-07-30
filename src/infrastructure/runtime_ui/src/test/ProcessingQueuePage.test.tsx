import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ProcessingQueuePage } from "../pages/ProcessingQueuePage";

const payload = {
  items: [
    {
      event_id: "image-1",
      modality: "image",
      event_type: "lightweight_visual_capture",
      status: "pending",
      deletable: true,
      original_name: "lecture.png",
      mime_type: "image/png",
      size_bytes: 2048,
      occurred_at: "2026-07-30T10:00:00+00:00",
      attempt_count: 0,
      preview_url: "/api/processing-queue/image-1/media",
    },
    {
      event_id: "audio-1",
      modality: "audio",
      event_type: "audio_capture_pending",
      status: "leased",
      deletable: false,
      original_name: "meeting.wav",
      mime_type: "audio/wav",
      size_bytes: 4096,
      duration_seconds: 12.5,
      occurred_at: "2026-07-30T10:01:00+00:00",
      attempt_count: 1,
      preview_url: "/api/processing-queue/audio-1/media",
    },
  ],
  count: 2,
  image_count: 1,
  audio_count: 1,
  processing_count: 1,
};

describe("ProcessingQueuePage", () => {
  beforeEach(() => {
    vi.spyOn(window, "confirm").mockReturnValue(true);
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      if (init?.method === "DELETE") {
        return Promise.resolve(new Response(JSON.stringify({ ok: true }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }));
      }
      return Promise.resolve(new Response(JSON.stringify(payload), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }));
    }));
  });

  it("renders queued media and removes only deletable inputs", async () => {
    render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <ProcessingQueuePage />
      </QueryClientProvider>,
    );

    expect(await screen.findByText("lecture.png")).toBeInTheDocument();
    expect(screen.getByText("meeting.wav")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Currently processing" })).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: "Remove from queue" }));
    await waitFor(() => expect(fetch).toHaveBeenCalledWith(
      "/api/processing-queue/image-1",
      expect.objectContaining({ method: "DELETE" }),
    ));
  });
});
