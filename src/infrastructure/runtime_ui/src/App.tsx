import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Activity, Archive, BarChart3, Bot, BrainCircuit, Database, FileText, Home, Inbox,
  ListRestart, MessageSquare, Moon, OctagonX, Pause, Play, RotateCcw, ScrollText, Sun, TestTube2,
  UserRound, Repeat2, ShieldCheck,
} from "lucide-react";
import { getJson, sendJson } from "./api";
import { Badge, Button } from "./components/ui";
import { ChatPage } from "./pages/ChatPage";
import { InboxPage } from "./pages/InboxPage";
import { ApprovalsPage } from "./pages/ApprovalsPage";
import { ReportsPage } from "./pages/ReportsPage";
import { BenchmarksPage } from "./pages/BenchmarksPage";
import { TrainingPage } from "./pages/TrainingPage";
import { LogsPage } from "./pages/LogsPage";
import { InteractionsPage } from "./pages/InteractionsPage";
import { RealWorldTestsPage } from "./pages/RealWorldTestsPage";
import { ProcessingQueuePage } from "./pages/ProcessingQueuePage";
import { ArtifactsPage } from "./pages/ArtifactsPage";
import { HomePage } from "./pages/HomePage";
import { RecurringTasksPage } from "./pages/RecurringTasksPage";

const routes = [
  { path: "/home", label: "Home", icon: Home },
  { path: "/chat", label: "Chat", icon: MessageSquare },
  { path: "/interactions", label: "Interactions", icon: Database },
  { path: "/inbox", label: "Proactive Inbox", icon: Inbox },
  { path: "/approvals", label: "Approvals", icon: ShieldCheck },
  { path: "/recurring-tasks", label: "Monitors", icon: Repeat2 },
  { path: "/processing-queue", label: "Processing Queue", icon: ListRestart },
  { path: "/reports", label: "Reports", icon: FileText },
  { path: "/artifacts", label: "Artifacts", icon: Archive },
  { path: "/benchmarks", label: "Benchmarks", icon: BarChart3 },
  { path: "/real-world-tests", label: "Real-world Tests", icon: TestTube2, labOnly: true },
  { path: "/training", label: "Training", icon: BrainCircuit },
  { path: "/logs", label: "Runtime Logs", icon: ScrollText },
] as const;

function normalizedPath(): string {
  const path = window.location.pathname.replace(/\/$/, "") || "/";
  return path === "/" ? "/home" : routes.some((route) => route.path === path) ? path : "/home";
}

function useRoute() {
  const [path, setPath] = useState(normalizedPath);
  useEffect(() => {
    const onPop = () => setPath(normalizedPath());
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);
  const navigate = (next: string) => {
    if (next !== window.location.pathname) window.history.pushState({}, "", next);
    setPath(next);
  };
  return { path, navigate };
}

function useTheme() {
  const [dark, setDark] = useState(() => document.documentElement.classList.contains("dark"));
  const toggle = () => {
    const next = !dark;
    setDark(next);
    document.documentElement.classList.toggle("dark", next);
    localStorage.setItem("ambient-theme", next ? "dark" : "light");
  };
  return { dark, toggle };
}

export function App() {
  const { path, navigate } = useRoute();
  const { dark, toggle } = useTheme();
  const queryClient = useQueryClient();
  const privacy = useQuery({
    queryKey: ["privacy-status"],
    queryFn: () => getJson<any>("/api/privacy/status"),
    refetchInterval: 5000,
  });
  const resources = useQuery({
    queryKey: ["resource-status"],
    queryFn: () => getJson<any>("/api/runtime/resources"),
    refetchInterval: 5000,
  });
  const health = useQuery({
    queryKey: ["health"],
    queryFn: () => getJson<any>("/healthz"),
    refetchInterval: 5000,
  });
  const reflection = useQuery({
    queryKey: ["manual-reflection-status"],
    queryFn: () => getJson<any>("/api/runtime/reflection/status"),
    refetchInterval: 2000,
    retry: false,
  });
  const biodata = useQuery({
    queryKey: ["manual-biodata-status"],
    queryFn: () => getJson<any>("/api/runtime/biodata/status"),
    refetchInterval: 2000,
    retry: false,
  });
  const interrupt = useQuery({
    queryKey: ["runtime-interrupt-status"],
    queryFn: () => getJson<any>("/api/runtime/interrupt/status"),
    refetchInterval: 2000,
    retry: false,
  });
  const capturePaused = Boolean(privacy.data?.capture?.paused);
  const reflectionStatus = reflection.data?.status || {};
  const reflectionBusy = Boolean(reflectionStatus.running || reflectionStatus.requested);
  const biodataStatus = biodata.data?.status || {};
  const biodataBusy = Boolean(biodataStatus.running || biodataStatus.requested);
  const interruptStatus = interrupt.data?.status || {};
  const interruptActive = Boolean(interruptStatus.active_work || interruptStatus.requested);
  const realWorldLabAvailable = Boolean(health.data?.real_world_lab);
  const visibleRoutes = useMemo(
    () => routes.filter((route) => !("labOnly" in route) || !route.labOnly || realWorldLabAvailable),
    [realWorldLabAvailable]
  );
  useEffect(() => {
    if (path === "/real-world-tests" && health.isSuccess && !realWorldLabAvailable) navigate("/home");
  }, [path, health.isSuccess, realWorldLabAvailable, navigate]);
  const captureMutation = useMutation({
    mutationFn: () => sendJson(`/api/privacy/capture/${capturePaused ? "resume" : "pause"}`, "POST"),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["privacy-status"] }),
  });
  const reflectionMutation = useMutation({
    mutationFn: () => sendJson("/api/runtime/reflection/run", "POST"),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["manual-reflection-status"] }),
  });
  const biodataMutation = useMutation({
    mutationFn: () => sendJson("/api/runtime/biodata/run", "POST"),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["manual-biodata-status"] }),
  });
  const interruptMutation = useMutation({
    mutationFn: () => sendJson("/api/runtime/interrupt", "POST", { reason: "Interrupted by local user" }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["runtime-interrupt-status"] });
      queryClient.invalidateQueries({ queryKey: ["resource-status"] });
      queryClient.invalidateQueries({ queryKey: ["processing-queue"] });
      queryClient.invalidateQueries({ queryKey: ["runtime-logs"] });
      queryClient.invalidateQueries({ queryKey: ["chat-sessions"] });
    },
  });
  const page = useMemo(() => {
    switch (path) {
      case "/home": return <HomePage onNavigate={navigate} />;
      case "/interactions": return <InteractionsPage />;
      case "/inbox": return <InboxPage privacy={privacy.data} resources={resources.data} />;
      case "/approvals": return <ApprovalsPage />;
      case "/recurring-tasks": return <RecurringTasksPage />;
      case "/processing-queue": return <ProcessingQueuePage />;
      case "/reports": return <ReportsPage />;
      case "/artifacts": return <ArtifactsPage />;
      case "/benchmarks": return <BenchmarksPage />;
      case "/real-world-tests": return realWorldLabAvailable ? <RealWorldTestsPage /> : <HomePage onNavigate={navigate} />;
      case "/training": return <TrainingPage />;
      case "/logs": return <LogsPage />;
      default: return <ChatPage />;
    }
  }, [path, privacy.data, resources.data, realWorldLabAvailable, navigate]);
  const loadedModel = resources.data?.residency?.loaded_model || "On demand";

  return (
    <div className="min-h-screen bg-canvas text-strong">
      <aside className="sidebar">
        <div className="flex items-center gap-3 px-3 pb-6 pt-2">
          <div className="brand-mark"><Bot size={21} /></div>
          <div className="min-w-0">
            <p className="truncate text-sm font-bold tracking-tight">Ambient Agent</p>
            <p className="text-xs text-muted">Runtime console</p>
          </div>
        </div>
        <nav className="flex flex-1 flex-col gap-1" aria-label="Dashboard sections">
          {visibleRoutes.map((route) => {
            const Icon = route.icon;
            return (
              <button key={route.path} type="button" className={`nav-item ${path === route.path ? "active" : ""}`} onClick={() => navigate(route.path)} aria-current={path === route.path ? "page" : undefined}>
                <Icon size={18} /><span>{route.label}</span>
              </button>
            );
          })}
        </nav>
        <div className="space-y-3 border-t border-line px-2 pt-4">
          <div className="rounded-xl bg-soft p-3 text-xs text-muted">
            <div className="mb-2 flex items-center gap-2 text-strong"><Activity size={14} className={health.isSuccess ? "text-good" : "text-danger"} /><span className="font-semibold">{health.isSuccess ? "Runtime connected" : "Runtime offline"}</span></div>
            <p className="truncate">Model · {loadedModel}</p>
          </div>
          <Button className="w-full justify-start" variant="ghost" onClick={toggle} aria-pressed={dark} aria-label={`Switch to ${dark ? "light" : "dark"} theme`}>
            {dark ? <Sun size={17} /> : <Moon size={17} />}{dark ? "Light theme" : "Dark theme"}
          </Button>
        </div>
      </aside>

      <div className="app-main">
        <header className="topbar">
          <div className="flex min-w-0 items-center gap-2">
            <span className={`status-dot ${health.isSuccess ? "online" : "offline"}`} />
            <span className="truncate text-sm text-muted">{health.isSuccess ? `Live · log ${health.data.latest_id}` : "Disconnected"}</span>
          </div>
          <div className="flex items-center gap-2">
            <Badge tone={capturePaused ? "warn" : "good"}>{capturePaused ? "Capture paused" : "Capture active"}</Badge>
            {interruptActive && <Badge tone="danger">{interruptStatus.requested ? "Interrupt requested" : `Working: ${interruptStatus.active_work?.kind || "runtime"}`}</Badge>}
            {reflectionBusy && <Badge tone={reflectionStatus.running ? "warn" : "neutral"}>{reflectionStatus.running ? "Reflection running" : "Reflection queued"}</Badge>}
            {biodataBusy && <Badge tone={biodataStatus.running ? "warn" : "neutral"}>{biodataStatus.running ? "Biodata running" : "Biodata queued"}</Badge>}
            <Button
              variant="danger"
              onClick={() => {
                if (window.confirm("Interrupt the current work item and skip it?")) interruptMutation.mutate();
              }}
              disabled={interruptMutation.isPending || !interruptActive}
              title="Stop the current LLM/tool loop and mark the current item as interrupted"
            >
              <OctagonX size={16} />Interrupt
            </Button>
            <Button variant="secondary" onClick={() => biodataMutation.mutate()} disabled={biodataMutation.isPending || biodataBusy} title="Extract pending observations into USER_INFO.md and MEMORY.md now">
              <UserRound size={16} />{biodataBusy ? "Biodata queued" : "Update biodata"}
            </Button>
            <Button variant="secondary" onClick={() => reflectionMutation.mutate()} disabled={reflectionMutation.isPending || reflectionBusy} title="Run reflection now, bypassing cadence and idle checks">
              <RotateCcw size={16} />{reflectionBusy ? "Reflection queued" : "Run reflection"}
            </Button>
            <Button variant="secondary" onClick={() => captureMutation.mutate()} disabled={captureMutation.isPending}>
              {capturePaused ? <Play size={16} /> : <Pause size={16} />}{capturePaused ? "Resume" : "Pause"}
            </Button>
            <Button className="theme-mobile" variant="ghost" onClick={toggle} aria-label={`Switch to ${dark ? "light" : "dark"} theme`}>
              {dark ? <Sun size={17} /> : <Moon size={17} />}
            </Button>
          </div>
        </header>
        <main className="content">{page}</main>
      </div>

      <nav className="mobile-nav" aria-label="Dashboard sections">
        {visibleRoutes.map((route) => {
          const Icon = route.icon;
          return <button key={route.path} type="button" className={path === route.path ? "active" : ""} onClick={() => navigate(route.path)} aria-label={route.label}><Icon size={19} /><span>{route.label.split(" ")[0]}</span></button>;
        })}
      </nav>
    </div>
  );
}
