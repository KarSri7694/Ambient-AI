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
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => {
      const path = String(input);
      if (path.includes("/api/privacy/status")) return json({ capture: { paused: false }, capture_size_bytes: 0 });
      if (path.includes("/api/runtime/resources")) return json({ preset: "balanced", residency: {}, snapshot: {}, event_counts: {} });
      if (path.includes("/healthz")) return json({ status: "ok", latest_id: 12 });
      if (path.includes("/api/chat/sessions")) return json({ sessions: [], count: 0 });
      return json({});
    }));
  });

  it("navigates between route-backed tabs", async () => {
    renderApp();
    fireEvent.click(screen.getAllByRole("button", { name: "Interactions" })[0]);
    expect(window.location.pathname).toBe("/interactions");
    expect(await screen.findByRole("heading", { name: "Interaction Logs" })).toBeInTheDocument();
  });

  it("persists the explicit dark theme", async () => {
    renderApp();
    const toggle = screen.getAllByRole("button", { name: "Switch to dark theme" })[0];
    fireEvent.click(toggle);
    await waitFor(() => expect(document.documentElement).toHaveClass("dark"));
    expect(localStorage.getItem("ambient-theme")).toBe("dark");
  });
});
