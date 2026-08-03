import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ChatPage } from "../pages/ChatPage";

function json(data: unknown) {
  return Promise.resolve(new Response(JSON.stringify(data), { status: 200, headers: { "Content-Type": "application/json" } }));
}

function renderChat() {
  return render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })}>
      <ChatPage />
    </QueryClientProvider>
  );
}

describe("ChatPage", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("creates a session and posts the first message from the composer", async () => {
    const calls: Array<{ path: string; init?: RequestInit }> = [];
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input);
      calls.push({ path, init });
      if (path === "/api/chat/sessions?limit=100") return json({ sessions: [], count: 0 });
      if (path === "/api/chat/sessions" && init?.method === "POST") {
        return json({ session: { id: "session-1", title: "New conversation", created_at: "", updated_at: "", preview: "" } });
      }
      if (path === "/api/chat/sessions/session-1/messages" && init?.method === "POST") {
        return json({
          user_message: { id: "user-1", session_id: "session-1", role: "user", content: "hello", status: "completed", message_kind: "chat", created_at: "", updated_at: "" },
          assistant_message: { id: "assistant-1", session_id: "session-1", role: "assistant", content: "queued", status: "completed", message_kind: "chat", created_at: "", updated_at: "" },
        });
      }
      if (path === "/api/chat/messages/assistant-1/events") return json({});
      return json({ session: { id: "session-1", title: "New conversation" }, messages: [], count: 0 });
    }));

    renderChat();
    const input = await screen.findByLabelText("Message Ambient AI");
    fireEvent.change(input, { target: { value: "hello" } });
    fireEvent.click(screen.getByRole("button", { name: /Send/i }));

    await waitFor(() => {
      expect(calls.some((call) => call.path === "/api/chat/sessions" && call.init?.method === "POST")).toBe(true);
      expect(calls.some((call) => call.path === "/api/chat/sessions/session-1/messages" && call.init?.method === "POST")).toBe(true);
    });
  });

  it("explains why the composer is locked by a queued assistant message", async () => {
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => {
      const path = String(input);
      if (path === "/api/chat/sessions?limit=100") {
        return json({ sessions: [{ id: "session-1", title: "Blocked", created_at: "", updated_at: "", preview: "" }], count: 1 });
      }
      if (path === "/api/chat/sessions/session-1/messages?limit=500") {
        return json({
          session: { id: "session-1", title: "Blocked", created_at: "", updated_at: "", preview: "" },
          messages: [
            { id: "assistant-1", session_id: "session-1", role: "assistant", content: "", status: "queued", message_kind: "chat", created_at: "", updated_at: "" },
          ],
          count: 1,
        });
      }
      return json({});
    }));

    renderChat();

    expect(await screen.findByText(/A response is queued/i)).toBeInTheDocument();
    expect(screen.getByLabelText("Message Ambient AI")).toBeDisabled();
  });
});
