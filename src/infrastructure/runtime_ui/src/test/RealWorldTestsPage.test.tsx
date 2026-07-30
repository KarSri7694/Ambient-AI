import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { RealWorldTestsPage } from "../pages/RealWorldTestsPage";

function json(data: unknown, status = 200) {
  return Promise.resolve(new Response(JSON.stringify(data), {
    status,
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

  it("lets uploaded images be dragged into the exact order sent to Ambient AI", async () => {
    let startBody: any;
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input);
      if (path.includes("/api/real-world/uploads")) {
        const originalName = new URL(path, "http://localhost").searchParams.get("filename") || "image.png";
        return json({ media: {
          media_id: originalName === "first.png" ? "image-1" : "image-2",
          kind: "image",
          original_name: originalName,
        } });
      }
      if (path === "/api/real-world/runs" && init?.method === "POST") {
        startBody = JSON.parse(String(init.body));
        return json({ run: { run_id: "uploaded-run" } });
      }
      if (path.includes("/api/real-world/suites")) return json({ available: true, suites: [] });
      if (path.includes("/api/real-world/models")) return json({ roles: {}, presets: [] });
      if (path.includes("/api/real-world/runs")) return json({ runs: [] });
      return json({});
    }));

    const { container } = renderPage();
    fireEvent.click(screen.getByRole("button", { name: "Upload" }));
    const input = container.querySelector<HTMLInputElement>('input[type="file"]');
    expect(input).not.toBeNull();
    await waitFor(() => expect(input).toBeEnabled());
    fireEvent.change(input!, { target: { files: [
      new File(["first"], "first.png", { type: "image/png" }),
      new File(["second"], "second.png", { type: "image/png" }),
    ] } });

    const firstImage = await screen.findByRole("img", { name: "Input 1: first.png" });
    const secondImage = await screen.findByRole("img", { name: "Input 2: second.png" });
    const dataTransfer = { effectAllowed: "none", dropEffect: "none", setData: vi.fn() };
    fireEvent.dragStart(firstImage.closest('[role="listitem"]')!, { dataTransfer });
    fireEvent.dragEnter(secondImage.closest('[role="listitem"]')!, { dataTransfer });
    fireEvent.drop(secondImage.closest('[role="listitem"]')!, { dataTransfer });

    expect(await screen.findByRole("img", { name: "Input 1: second.png" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Move second.png earlier" })).toBeDisabled();

    fireEvent.change(screen.getByPlaceholderText("RUN LIVE TOOLS"), { target: { value: "RUN LIVE TOOLS" } });
    fireEvent.click(screen.getByRole("button", { name: "Start evaluation" }));
    await waitFor(() => expect(startBody).toBeDefined());
    expect(startBody.inline_scenario.events).toEqual([
      { media_id: "image-2", offset_seconds: 0 },
      { media_id: "image-1", offset_seconds: 10 },
    ]);
  });

  it("disables uploads when opened from the normal runtime without a lab backend", async () => {
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => {
      const path = String(input);
      if (path.includes("/api/real-world/suites")) return json({ available: false, suites: [] });
      if (path.includes("/api/real-world/runs")) return json({ available: false, runs: [] });
      return json({ detail: "real_world_lab_unavailable" }, 503);
    }));

    const { container } = renderPage();
    fireEvent.click(screen.getByRole("button", { name: "Upload" }));
    expect(await screen.findByText("Real-world test lab is offline")).toBeInTheDocument();
    expect(container.querySelector<HTMLInputElement>('input[type="file"]')).toBeDisabled();
    expect((await screen.findByText("Start the evaluation lab")).parentElement).toHaveTextContent("tests/real_world_tests/run_lab.py");
  });

  it("shows the backend error when an image upload is rejected", async () => {
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => {
      const path = String(input);
      if (path.includes("/api/real-world/uploads")) return json({ detail: "image upload must be between 1 byte and 25 MB" }, 400);
      if (path.includes("/api/real-world/suites")) return json({ available: true, suites: [] });
      if (path.includes("/api/real-world/models")) return json({ roles: {}, presets: [] });
      if (path.includes("/api/real-world/runs")) return json({ runs: [] });
      return json({});
    }));

    const { container } = renderPage();
    fireEvent.click(screen.getByRole("button", { name: "Upload" }));
    const input = container.querySelector<HTMLInputElement>('input[type="file"]')!;
    await waitFor(() => expect(input).toBeEnabled());
    fireEvent.change(input, { target: { files: [new File(["bad"], "large.png", { type: "image/png" })] } });
    expect(await screen.findByRole("alert")).toHaveTextContent("image upload must be between 1 byte and 25 MB");
  });
});
