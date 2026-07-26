import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { RealWorldTestsPage } from "../pages/RealWorldTestsPage";

function json(data: unknown) {
  return Promise.resolve(new Response(JSON.stringify(data), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  }));
}

function renderPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  return render(<QueryClientProvider client={client}><RealWorldTestsPage /></QueryClientProvider>);
}

describe("real-world evaluation studio", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => {
      const path = String(input);
      if (path.includes("/api/real-world/suites")) return json({ available: true, suites: [{
        suite_id: "desk_suite", title: "Desktop suite", scenarios: [{
          scenario_id: "desk_flow", title: "Desktop progression", modality: "image_sequence",
          events: [{ media_path: "one.png" }, { media_path: "two.png" }],
        }],
      }] });
      if (path.includes("/api/real-world/models")) return json({ roles: { followup_execution_model: "vision-a" }, presets: ["vision-a", "vision-b"] });
      if (path.includes("/api/real-world/runs/run-1/trace")) return json({ events: [{
        event_id: "event-1", sequence: 1, stage: "model", event_type: "model_request",
        status: "completed", created_at: "2026-07-26T12:00:00", payload: { messages: [] },
      }] });
      if (path.endsWith("/api/real-world/runs/run-1")) return json({ run: {
        run_id: "run-1", suite_id: "desk_suite", scenario_ids: ["desk_flow"], status: "completed", results: [],
      } });
      if (path.includes("/api/real-world/runs")) return json({ runs: [{
        run_id: "run-1", suite_id: "desk_suite", scenario_ids: ["desk_flow"],
        playback_speed: 1, status: "completed", created_at: "2026-07-26T12:00:00",
      }] });
      return json({});
    }));
  });

  it("presents a guided launch flow and structured run timeline", async () => {
    renderPage();
    expect(screen.getByRole("heading", { name: "Real-world tests" })).toBeInTheDocument();
    expect(await screen.findByText("Desktop progression")).toBeInTheDocument();

    const start = screen.getByRole("button", { name: "Start evaluation" });
    expect(start).toBeDisabled();
    fireEvent.change(screen.getByPlaceholderText("RUN LIVE TOOLS"), { target: { value: "RUN LIVE TOOLS" } });
    await waitFor(() => expect(start).toBeEnabled());

    fireEvent.click(screen.getByRole("button", { name: /desk_suite/i }));
    expect(await screen.findByRole("heading", { name: "Pipeline timeline" })).toBeInTheDocument();
    expect(await screen.findByRole("heading", { name: "Model Request" })).toBeInTheDocument();
  });
});
