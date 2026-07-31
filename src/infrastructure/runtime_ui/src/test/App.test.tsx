import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "../App";

function json(data: unknown) {
  return Promise.resolve(new Response(JSON.stringify(data), { status: 200, headers: { "Content-Type": "application/json" } }));
}

function renderApp() {
  return render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })}><App /></QueryClientProvider>);
}

describe("runtime shell", () => {
  beforeEach(() => {
    window.history.replaceState({}, "", "/chat");
    localStorage.clear();
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => {
      const path = String(input);
      if (path.includes("/api/privacy/status")) return json({ capture: { paused: false }, capture_size_bytes: 0 });
      if (path.includes("/api/runtime/resources")) return json({ preset: "balanced", residency: {}, snapshot: {}, event_counts: {} });
      if (path.includes("/api/runtime/reflection/status")) return json({ ok: true, status: { requested: false, running: false } });
      if (path.includes("/api/runtime/reflection/run")) return json({ ok: true, accepted: true, status: { requested: true, running: false } });
      if (path.includes("/api/runtime/biodata/status")) return json({ ok: true, status: { requested: false, running: false } });
      if (path.includes("/api/runtime/biodata/run")) return json({ ok: true, accepted: true, status: { requested: true, running: false } });
      if (path.includes("/healthz")) return json({ status: "ok", latest_id: 12 });
      if (path.includes("/api/chat/sessions")) return json({ sessions: [], count: 0 });
      if (path.includes("/api/home")) return json({ date: "2026-07-30", today: "2026-07-30", server_time: new Date().toISOString(), counts: {}, background: { events: {} }, timeline: [], attention: [], new_count: 0, briefing: null, briefing_pending: false, briefing_refresh: { running: false, last_error: null }, latest_headline: "Nothing important changed yet", latest_narrative: "I have not completed any meaningful work yet." });
      return json({});
    }));
  });

  it("navigates between route-backed tabs", async () => {
    renderApp();
    fireEvent.click(screen.getAllByRole("button", { name: "Interactions" })[0]);
    expect(window.location.pathname).toBe("/interactions");
    expect(await screen.findByRole("heading", { name: "Interaction Logs" })).toBeInTheDocument();
  });

  it("uses Home as the default route", async () => {
    window.history.replaceState({}, "", "/");
    renderApp();
    expect(await screen.findByText("Your ambient day")).toBeInTheDocument();
    expect(await screen.findByText("Nothing important changed yet")).toBeInTheDocument();
    expect(await screen.findByText("I have not completed any meaningful work yet.")).toBeInTheDocument();
    expect(fetch).toHaveBeenCalledWith("/api/home", expect.objectContaining({ cache: "no-store" }));
    expect(screen.getAllByRole("button", { name: "Home" })[0]).toHaveAttribute("aria-current", "page");
  });

  it("opens the artifact library tab", async () => {
    renderApp();
    fireEvent.click(screen.getAllByRole("button", { name: "Artifacts" })[0]);
    expect(window.location.pathname).toBe("/artifacts");
    expect(await screen.findByRole("heading", { name: "Artifacts" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Consolidate now/i })).toBeInTheDocument();
  });

  it("persists the explicit dark theme", async () => {
    renderApp();
    const toggle = screen.getAllByRole("button", { name: "Switch to dark theme" })[0];
    fireEvent.click(toggle);
    await waitFor(() => expect(document.documentElement).toHaveClass("dark"));
    expect(localStorage.getItem("ambient-theme")).toBe("dark");
  });

  it("can request manual reflection from the topbar", async () => {
    renderApp();
    const button = await screen.findByRole("button", { name: /Run reflection/i });
    fireEvent.click(button);
    await waitFor(() => expect(fetch).toHaveBeenCalledWith("/api/runtime/reflection/run", expect.objectContaining({ method: "POST" })));
  });

  it("can request a manual biodata update from the topbar", async () => {
    renderApp();
    const button = await screen.findByRole("button", { name: /Update biodata/i });
    fireEvent.click(button);
    await waitFor(() => expect(fetch).toHaveBeenCalledWith("/api/runtime/biodata/run", expect.objectContaining({ method: "POST" })));
  });
});
