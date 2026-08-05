import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit


class CaptureControlService:
    """Thread-safe screen-capture policy, status, and persistence service."""

    _APP_BOUNDARY = re.compile(r"[^a-z0-9]+")

    def __init__(
        self,
        *,
        excluded_apps=None,
        excluded_domains=None,
        persistence_path: str | Path | None = None,
    ):
        self._paused = threading.Event()
        self._lock = threading.RLock()
        self._persistence_path = Path(persistence_path) if persistence_path else None
        self._updated_at = self._now()
        self._revision = 0
        self._persistence_source = "memory"
        self._last_context: dict | None = None
        self._last_decision: dict | None = None

        loaded = self._load_persisted_policy()
        if loaded is None:
            self._excluded_apps = self._normalize_apps(excluded_apps or [])
            self._excluded_domains = self._normalize_domains(excluded_domains or [])
            self._revision = 1
            self._persistence_source = "config_seed" if self._persistence_path else "memory"
            if self._persistence_path is not None:
                self._persist_policy(
                    apps=self._excluded_apps,
                    domains=self._excluded_domains,
                    revision=self._revision,
                    updated_at=self._updated_at,
                )
        else:
            self._excluded_apps = loaded["apps"]
            self._excluded_domains = loaded["domains"]
            self._revision = loaded["revision"]
            self._updated_at = loaded["updated_at"]
            self._persistence_source = "persisted"

    def pause(self) -> None:
        with self._lock:
            self._paused.set()
            self._updated_at = self._now()

    def resume(self) -> None:
        with self._lock:
            self._paused.clear()
            self._updated_at = self._now()

    def is_paused(self) -> bool:
        return self._paused.is_set()

    def set_exclusions(self, *, apps=None, domains=None) -> None:
        """Replace exclusions and persist them before making them active."""
        with self._lock:
            next_apps = self._normalize_apps(apps) if apps is not None else set(self._excluded_apps)
            next_domains = (
                self._normalize_domains(domains) if domains is not None else set(self._excluded_domains)
            )
            if next_apps == self._excluded_apps and next_domains == self._excluded_domains:
                return
            updated_at = self._now()
            revision = self._revision + 1
            self._persist_policy(
                apps=next_apps,
                domains=next_domains,
                revision=revision,
                updated_at=updated_at,
            )
            self._excluded_apps = next_apps
            self._excluded_domains = next_domains
            self._revision = revision
            self._updated_at = updated_at
            if self._persistence_path is not None:
                self._persistence_source = "persisted"

    def evaluate_context(self, context: dict | None = None, *, record: bool = True) -> dict:
        """Evaluate one foreground context and return an auditable decision."""
        context = dict(context or {})
        process_name = self._process_basename(context.get("process_name"))
        app_name = str(context.get("app_name") or context.get("app_hint") or "").strip()
        window_title = str(context.get("window_title") or "").strip()
        window_class = str(context.get("window_class") or "").strip()
        url = str(context.get("url") or context.get("foreground_url") or "").strip()
        domain = ""
        for domain_candidate in (
            context.get("domain"),
            context.get("domain_hint"),
            url,
        ):
            domain = self.normalize_domain(domain_candidate)
            if domain:
                break

        with self._lock:
            matched_rule = None
            match_type = None
            for rule in sorted(self._excluded_apps):
                if self._app_rule_matches(
                    rule,
                    process_name=process_name,
                    app_name=app_name,
                    window_class=window_class,
                    window_title=window_title,
                ):
                    matched_rule = rule
                    match_type = "application"
                    break
            if matched_rule is None and domain:
                for rule in sorted(self._excluded_domains):
                    if domain == rule or domain.endswith(f".{rule}"):
                        matched_rule = rule
                        match_type = "domain"
                        break

            checked_at = self._now()
            detected = {
                "process_name": process_name or None,
                "app_name": app_name or None,
                "window_title": window_title or None,
                "window_class": window_class or None,
                "url": url or None,
                "domain": domain or None,
            }
            decision = {
                "excluded": matched_rule is not None,
                "match_type": match_type,
                "matched_rule": matched_rule,
                "policy_revision": self._revision,
                "checked_at": checked_at,
                "detected": detected,
            }
            if record:
                self._last_context = detected
                self._last_decision = dict(decision)
            return decision

    def is_excluded(
        self,
        *,
        app_name: str = "",
        domain: str = "",
        process_name: str = "",
        window_title: str = "",
        window_class: str = "",
        url: str = "",
    ) -> bool:
        return bool(
            self.evaluate_context(
                {
                    "app_name": app_name,
                    "domain": domain,
                    "process_name": process_name,
                    "window_title": window_title,
                    "window_class": window_class,
                    "url": url,
                },
                record=False,
            )["excluded"]
        )

    def status(self) -> dict:
        with self._lock:
            return {
                "paused": self.is_paused(),
                "excluded_apps": sorted(self._excluded_apps),
                "excluded_domains": sorted(self._excluded_domains),
                "updated_at": self._updated_at,
                "policy_revision": self._revision,
                "last_context": dict(self._last_context) if self._last_context else None,
                "last_decision": dict(self._last_decision) if self._last_decision else None,
                "persistence": {
                    "enabled": self._persistence_path is not None,
                    "source": self._persistence_source,
                    "path": str(self._persistence_path) if self._persistence_path else None,
                },
            }

    @classmethod
    def normalize_domain(cls, value) -> str:
        raw = str(value or "").strip().lower().rstrip(".")
        if not raw:
            return ""
        if raw.startswith("*."):
            raw = raw[2:]
        candidate = raw if "://" in raw else f"//{raw}"
        try:
            hostname = (urlsplit(candidate).hostname or "").strip().lower().rstrip(".")
        except ValueError:
            return ""
        if not hostname or any(char.isspace() for char in hostname):
            return ""
        # Treat www.example.com and example.com as the same site for privacy
        # rules. Subdomains are matched separately by evaluate_context.
        if hostname.startswith("www."):
            hostname = hostname[4:]
        try:
            return hostname.encode("idna").decode("ascii")
        except UnicodeError:
            return ""

    @classmethod
    def _normalize_apps(cls, values) -> set[str]:
        normalized = set()
        for value in values or []:
            item = str(value or "").strip().lower()
            if not item:
                continue
            if len(item) > 200 or any(char in item for char in "\r\n\0"):
                raise ValueError(f"invalid application exclusion: {value!r}")
            # Users commonly paste an executable path from Task Manager or
            # Explorer. Store its basename so it matches UI Automation's
            # process_name regardless of installation directory.
            normalized.add(cls._process_basename(item) if "\\" in item or "/" in item else item)
        return normalized

    @classmethod
    def _normalize_domains(cls, values) -> set[str]:
        normalized = set()
        for value in values or []:
            raw = str(value or "").strip()
            if raw.startswith("*."):
                raw = raw[2:]
            item = cls.normalize_domain(raw)
            if not item:
                raise ValueError(f"invalid domain exclusion: {value!r}")
            normalized.add(item)
        return normalized

    @classmethod
    def _app_rule_matches(
        cls,
        rule: str,
        *,
        process_name: str,
        app_name: str,
        window_class: str,
        window_title: str,
    ) -> bool:
        normalized_rule = rule.strip().lower()
        process = cls._process_basename(process_name)
        process_stem = Path(process).stem.lower() if process else ""
        app = app_name.strip().lower()
        app_base = cls._process_basename(app)
        app_stem = Path(app_base).stem.lower() if app_base else app
        window_cls = window_class.strip().lower()

        rule_base = cls._process_basename(normalized_rule)
        rule_stem = Path(rule_base).stem.lower() if rule_base else normalized_rule
        rule_tokens = cls._normalize_app_text(normalized_rule)
        rule_stem_tokens = cls._normalize_app_text(rule_stem)
        candidates = (process, process_stem, app, app_base, app_stem, window_cls, window_title)

        # Match against every foreground identifier, not only the window
        # title. UI Automation frequently reports "Google Chrome" as the app
        # while the process is chrome.exe, and reports the browser title when
        # the process name is unavailable.
        for candidate in candidates:
            candidate_tokens = cls._normalize_app_text(candidate)
            if not candidate_tokens:
                continue
            for target in (rule_tokens, rule_stem_tokens):
                if target and re.search(rf"(?:^|\s){re.escape(target)}(?:$|\s)", candidate_tokens):
                    return True
        return False

    @classmethod
    def _normalize_app_text(cls, value) -> str:
        raw = str(value or "").strip().lower().replace("\\", " ").replace("/", " ")
        return cls._APP_BOUNDARY.sub(" ", raw).strip()

    @staticmethod
    def _process_basename(value) -> str:
        raw = str(value or "").strip().replace("/", "\\")
        return raw.rsplit("\\", 1)[-1].lower() if raw else ""

    def _load_persisted_policy(self) -> dict | None:
        path = self._persistence_path
        if path is None or not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return {
                "apps": self._normalize_apps(payload.get("apps", [])),
                "domains": self._normalize_domains(payload.get("domains", [])),
                "revision": max(1, int(payload.get("revision", 1))),
                "updated_at": str(payload.get("updated_at") or self._now()),
            }
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Unable to load capture exclusions from {path}: {exc}") from exc

    def _persist_policy(self, *, apps: set[str], domains: set[str], revision: int, updated_at: str) -> None:
        path = self._persistence_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "revision": revision,
            "apps": sorted(apps),
            "domains": sorted(domains),
            "updated_at": updated_at,
        }
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()
