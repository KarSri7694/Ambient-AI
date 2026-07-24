import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Bot, Edit3, MessageSquarePlus, Send, Sparkles } from "lucide-react";
import { apiFetch, getJson, sendJson } from "../api";
import { Badge, Button, EmptyState, ErrorState, Markdown } from "../components/ui";
import type { ChatMessage, ChatSession } from "../types";

interface SessionsPayload { sessions: ChatSession[]; count: number }
interface MessagesPayload { session: ChatSession; messages: ChatMessage[] }

function parseSseBlock(block: string): { type: string; data: any } | null {
  let type = "message";
  const lines: string[] = [];
  block.split("\n").forEach((line) => {
    if (line.startsWith("event:")) type = line.slice(6).trim();
    if (line.startsWith("data:")) lines.push(line.slice(5).trim());
  });
  if (!lines.length) return null;
  try { return { type, data: JSON.parse(lines.join("\n")) }; } catch { return null; }
}

function messageText(message: ChatMessage): string {
  if (message.status === "failed") return message.error_text || message.content || "Response failed.";
  if (message.content) return message.content;
  if (message.status === "queued") return "Queued…";
  if (message.status === "running") return "Thinking…";
  return message.error_text || "No response returned.";
}

export function ChatPage() {
  const queryClient = useQueryClient();
  const [selectedId, setSelectedId] = useState("");
  const [input, setInput] = useState("");
  const [activity, setActivity] = useState("");
  const [streamingIds, setStreamingIds] = useState<Set<string>>(new Set());
  const activeStreams = useRef<Set<string>>(new Set());
  const messageEnd = useRef<HTMLDivElement>(null);
  const sessions = useQuery({
    queryKey: ["chat-sessions"],
    queryFn: () => getJson<SessionsPayload>("/api/chat/sessions?limit=100"),
    refetchInterval: streamingIds.size ? false : 3000,
  });
  useEffect(() => {
    if (!selectedId && sessions.data?.sessions.length) setSelectedId(sessions.data.sessions[0].id);
  }, [selectedId, sessions.data]);
  const messages = useQuery({
    queryKey: ["chat-messages", selectedId],
    queryFn: () => getJson<MessagesPayload>(`/api/chat/sessions/${selectedId}/messages?limit=500`),
    enabled: Boolean(selectedId),
    refetchInterval: streamingIds.size ? false : 2500,
  });
  useEffect(() => { messageEnd.current?.scrollIntoView({ block: "end" }); }, [messages.data?.messages]);

  const createSession = useMutation({
    mutationFn: () => sendJson<{ session: ChatSession }>("/api/chat/sessions", "POST", { title: "New conversation" }),
    onSuccess: async ({ session }) => {
      await queryClient.invalidateQueries({ queryKey: ["chat-sessions"] });
      setSelectedId(session.id);
    },
  });
  const rename = async () => {
    const current = sessions.data?.sessions.find((session) => session.id === selectedId);
    if (!current) return;
    const title = window.prompt("Conversation name", current.title)?.trim();
    if (!title) return;
    await sendJson(`/api/chat/sessions/${selectedId}`, "PATCH", { title });
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: ["chat-sessions"] }),
      queryClient.invalidateQueries({ queryKey: ["chat-messages", selectedId] }),
    ]);
  };

  const streamMessage = async (messageId: string, sessionId: string) => {
    if (activeStreams.current.has(messageId)) return;
    activeStreams.current.add(messageId);
    setStreamingIds((current) => new Set(current).add(messageId));
    try {
      const response = await apiFetch(`/api/chat/messages/${messageId}/events`, { headers: { Accept: "text/event-stream" } });
      const reader = response.body?.getReader();
      if (!reader) throw new Error("Streaming response is unavailable.");
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true }).replaceAll("\r\n", "\n");
        let separator = buffer.indexOf("\n\n");
        while (separator >= 0) {
          const event = parseSseBlock(buffer.slice(0, separator));
          buffer = buffer.slice(separator + 2);
          if (event?.type === "tool_started") setActivity(`Using ${event.data.tool_name || "a tool"}…`);
          if (event?.type === "tool_finished") setActivity(`${event.data.tool_name || "Tool"} ${event.data.ok ? "finished" : "failed"}.`);
          if (event?.type === "status") setActivity(event.data.status === "running" ? "Ambient AI is thinking…" : event.data.status);
          if (["snapshot", "delta", "done", "error"].includes(event?.type || "")) {
            await queryClient.invalidateQueries({ queryKey: ["chat-messages", sessionId] });
          }
          if (event?.type === "done") setActivity("");
          if (event?.type === "error") setActivity("The response failed. You can send the request again.");
          separator = buffer.indexOf("\n\n");
        }
      }
    } catch {
      setActivity("Live connection interrupted; the saved response will continue processing.");
    } finally {
      activeStreams.current.delete(messageId);
      setStreamingIds((current) => { const next = new Set(current); next.delete(messageId); return next; });
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["chat-messages", sessionId] }),
        queryClient.invalidateQueries({ queryKey: ["chat-sessions"] }),
      ]);
    }
  };

  useEffect(() => {
    messages.data?.messages.forEach((message) => {
      if (message.role === "assistant" && ["queued", "running"].includes(message.status) && !activeStreams.current.has(message.id)) {
        void streamMessage(message.id, selectedId);
      }
    });
    // streamingIds intentionally excluded; the guard prevents duplicate connections.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [messages.data?.messages, selectedId]);

  const submit = async (event?: FormEvent) => {
    event?.preventDefault();
    const content = input.trim();
    if (!content) return;
    try {
      let sessionId = selectedId;
      if (!sessionId) {
        const created = await sendJson<{ session: ChatSession }>("/api/chat/sessions", "POST", { title: "New conversation" });
        sessionId = created.session.id;
        setSelectedId(sessionId);
      }
      setActivity("Queued for Ambient AI…");
      const payload = await sendJson<{ user_message: ChatMessage; assistant_message: ChatMessage }>(`/api/chat/sessions/${sessionId}/messages`, "POST", { content });
      setInput("");
      queryClient.setQueryData<MessagesPayload>(["chat-messages", sessionId], (current) => current ? { ...current, messages: [...current.messages, payload.user_message, payload.assistant_message] } : current);
      await queryClient.invalidateQueries({ queryKey: ["chat-sessions"] });
      void streamMessage(payload.assistant_message.id, sessionId);
    } catch (error) {
      setActivity(error instanceof Error ? error.message : "Could not send the message.");
    }
  };
  const active = messages.data?.messages.some((message) => message.role === "assistant" && ["queued", "running"].includes(message.status)) || streamingIds.size > 0;
  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void submit(); }
  };

  return (
    <div className="chat-grid">
      <aside className="panel min-h-0 overflow-hidden">
        <div className="flex items-center justify-between border-b border-line p-4">
          <div><p className="eyebrow">Conversations</p><h1 className="text-lg font-bold">Ambient AI</h1></div>
          <Button variant="primary" onClick={() => createSession.mutate()} disabled={createSession.isPending}><MessageSquarePlus size={16} />New</Button>
        </div>
        <div className="chat-session-list">
          {sessions.isError && <ErrorState error={sessions.error} />}
          {!sessions.isLoading && !sessions.data?.sessions.length && <EmptyState title="No conversations" description="Start a new conversation with Ambient AI." />}
          {sessions.data?.sessions.map((session) => (
            <button type="button" key={session.id} className={`chat-session ${selectedId === session.id ? "active" : ""}`} onClick={() => setSelectedId(session.id)}>
              <strong>{session.title}</strong><span>{session.preview || "No messages yet"}</span>
            </button>
          ))}
        </div>
      </aside>

      <section className="panel chat-panel">
        <header className="flex items-center justify-between border-b border-line px-5 py-4">
          <div><p className="eyebrow">Direct conversation</p><h2 className="font-bold">{messages.data?.session.title || "New conversation"}</h2></div>
          <Button variant="ghost" onClick={rename} disabled={!selectedId}><Edit3 size={16} />Rename</Button>
        </header>
        <div className="chat-messages" aria-live="polite">
          {messages.isError && <ErrorState error={messages.error} />}
          {!messages.isLoading && !messages.data?.messages.length && (
            <div className="chat-welcome">
              <div className="brand-mark mx-auto"><Sparkles size={22} /></div>
              <h2>Ask, act, or schedule.</h2>
              <p>Ask about local information, give Ambient AI something to do now, or schedule work for later.</p>
              <div className="mt-5 flex flex-wrap justify-center gap-2">
                {["Summarize the latest local runtime reports.", "Open the project README and tell me the most important setup steps.", "Tomorrow at 9:00 AM, check the queued tasks and report what is still pending."].map((example) => <Button key={example} variant="secondary" onClick={() => setInput(example)}>{example}</Button>)}
              </div>
            </div>
          )}
          {messages.data?.messages.map((message) => (
            <article key={message.id} className={`message-row ${message.role}`}>
              <div className={`message-bubble ${message.status === "failed" ? "failed" : ""}`}>
                {message.role === "assistant" ? <Markdown>{messageText(message)}</Markdown> : <p className="whitespace-pre-wrap">{messageText(message)}</p>}
              </div>
              <div className="message-meta"><span>{message.role === "user" ? "You" : "Ambient AI"}</span><Badge tone={message.status === "failed" ? "danger" : "neutral"}>{message.status}</Badge>{message.message_kind === "scheduled_result" && <Badge>Scheduled result</Badge>}</div>
            </article>
          ))}
          <div ref={messageEnd} />
        </div>
        <div className="min-h-7 px-5 text-xs text-accent" aria-live="polite">{activity}</div>
        <form className="chat-composer" onSubmit={submit}>
          <textarea value={input} onChange={(event) => setInput(event.target.value)} onKeyDown={onKeyDown} rows={2} maxLength={12000} placeholder="Message Ambient AI…" aria-label="Message Ambient AI" disabled={active} />
          <Button variant="primary" type="submit" disabled={active || !input.trim()}><Send size={17} />Send</Button>
        </form>
        <p className="pb-4 text-center text-xs text-muted">Enter to send · Shift+Enter for a new line · explicit action requests can use tools</p>
      </section>
    </div>
  );
}
