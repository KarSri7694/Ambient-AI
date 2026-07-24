import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { InteractionsPage } from "../pages/InteractionsPage";

const protectedInput = { protected: true, request: null, context_messages: [], malformed: false };

function json(data: unknown) {
  return Promise.resolve(new Response(JSON.stringify(data), { status: 200, headers: { "Content-Type": "application/json" } }));
}

function renderPage() {
  return render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}><InteractionsPage /></QueryClientProvider>);
}

describe("interaction viewer", () => {
  it("renders pairs, reveals protected input, filters, and enlarges images", async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const path = String(input);
      if (path.endsWith("/input")) return json({ input: { protected: true, request: { role: "user", content: "private prompt" }, context_messages: [], malformed: false } });
      return json({ items: [{ interaction_id: "one", created_at: "2026-07-24T10:00:00", source: "passive_observer", model: "model-a", duration_ms: 100, input: protectedInput, response_text: "**answer**", error_text: null, reasoning_text: null, tools: null, tool_calls: null, metadata: {}, report: null, has_image: true, image_url: "/api/interactions/one/image" }], pagination: { limit: 50, offset: 0, total: 1, has_more: false }, sort: "newest" });
    });
    vi.stubGlobal("fetch", fetchMock);
    renderPage();
    expect(await screen.findByText("answer")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Reveal input" }));
    expect(await screen.findByText("private prompt")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Order"), { target: { value: "oldest" } });
    await waitFor(() => expect(fetchMock.mock.calls.some(([url]) => String(url).includes("sort=oldest"))).toBe(true));
    fireEvent.click(await screen.findByRole("button", { name: "Enlarge interaction image" }));
    expect(screen.getByRole("dialog", { name: "Expanded image preview" })).toBeInTheDocument();
    fireEvent.keyDown(document, { key: "Escape" });
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  });
});
