import { useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { AlertCircle, ImageIcon, LoaderCircle, X } from "lucide-react";

export function formatDate(value?: string | null): string {
  if (!value) return "Unknown date";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

export function prettyJson(value: unknown): string {
  if (value === undefined || value === null || value === "") return "";
  if (typeof value === "string") {
    try {
      return JSON.stringify(JSON.parse(value), null, 2);
    } catch {
      return value;
    }
  }
  return JSON.stringify(value, null, 2);
}

export function Badge({ children, tone = "neutral" }: { children: React.ReactNode; tone?: "neutral" | "good" | "warn" | "danger" }) {
  return <span className={`badge badge-${tone}`}>{children}</span>;
}

export function Button({ className = "", variant = "secondary", ...props }: React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "secondary" | "ghost" | "danger" }) {
  return <button className={`button button-${variant} ${className}`} {...props} />;
}

export function PageHeader({ eyebrow, title, description, actions }: { eyebrow: string; title: string; description: string; actions?: React.ReactNode }) {
  return (
    <header className="mb-6 flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
      <div>
        <p className="eyebrow">{eyebrow}</p>
        <h1 className="page-title">{title}</h1>
        <p className="mt-2 max-w-2xl text-sm leading-6 text-muted">{description}</p>
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </header>
  );
}

export function EmptyState({ title, description }: { title: string; description: string }) {
  return (
    <div className="empty-state">
      <div className="empty-icon"><AlertCircle size={20} /></div>
      <h3 className="mt-3 font-semibold text-strong">{title}</h3>
      <p className="mt-1 max-w-md text-sm text-muted">{description}</p>
    </div>
  );
}

export function LoadingState({ label = "Loading" }: { label?: string }) {
  return <div className="flex min-h-40 items-center justify-center gap-2 text-sm text-muted"><LoaderCircle className="animate-spin" size={18} />{label}</div>;
}

export function ErrorState({ error }: { error: unknown }) {
  return <div className="rounded-2xl border border-danger/30 bg-danger/10 p-4 text-sm text-danger">{error instanceof Error ? error.message : "Something went wrong."}</div>;
}

export function Markdown({ children }: { children?: string | null }) {
  return <div className="markdown"><ReactMarkdown remarkPlugins={[remarkGfm]}>{children || ""}</ReactMarkdown></div>;
}

export function JsonDetails({ label, value, open = false }: { label: string; value: unknown; open?: boolean }) {
  if (value === undefined || value === null || value === "") return null;
  return <details className="details" open={open}><summary>{label}</summary><pre>{prettyJson(value)}</pre></details>;
}

export function ImageModal({ src, alt, onClose }: { src: string | null; alt: string; onClose: () => void }) {
  const closeRef = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (!src) return;
    const previous = document.activeElement as HTMLElement | null;
    closeRef.current?.focus();
    const onKey = (event: KeyboardEvent) => event.key === "Escape" && onClose();
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("keydown", onKey);
      previous?.focus();
    };
  }, [src, onClose]);
  if (!src) return null;
  return (
    <div className="modal" role="dialog" aria-modal="true" aria-label="Expanded image preview" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <div className="modal-panel">
        <button ref={closeRef} className="modal-close" type="button" onClick={onClose} aria-label="Close image preview"><X size={20} /></button>
        <img src={src} alt={alt} />
      </div>
    </div>
  );
}

export function ImagePlaceholder() {
  return <div className="flex aspect-video items-center justify-center rounded-xl bg-soft text-muted"><ImageIcon size={24} /></div>;
}
