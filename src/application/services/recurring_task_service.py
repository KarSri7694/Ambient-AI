"""Durable, privacy-bound recurring work and condition monitoring.

The service deliberately keeps scheduling/state transitions deterministic.  An
LLM may create a task contract or execute a due task, but it never decides
whether a duplicate completion toast should be sent or whether a monitor may
capture an unrelated window.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from core.models import RecurringTask


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None = None) -> str:
    return (value or _utcnow()).isoformat()


def _normalize_schedule_time(value: str) -> str | None:
    match = re.fullmatch(r"\s*(\d{1,2})(?::(\d{2}))?\s*([ap]m)?\s*", str(value or ""), re.IGNORECASE)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    meridiem = (match.group(3) or "").lower()
    if minute > 59 or (not meridiem and hour > 23) or (meridiem and not 1 <= hour <= 12):
        return None
    if meridiem:
        hour = (hour % 12) + (12 if meridiem == "pm" else 0)
    return f"{hour:02d}:{minute:02d}"


def _time_from_instruction(instruction: str) -> str | None:
    match = re.search(r"\b(?:at|around|by)\s+(\d{1,2})(?::(\d{2}))?\s*([ap]m)\b", instruction, re.IGNORECASE)
    if not match:
        return None
    hour, minute, meridiem = match.groups()
    return _normalize_schedule_time(f"{hour}:{minute or '00'} {meridiem}")


def _next_local_time(schedule_time: str, *, now: datetime | None = None) -> datetime:
    local_now = (now or _utcnow()).astimezone()
    hour, minute = (int(part) for part in schedule_time.split(":"))
    candidate = local_now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= local_now:
        candidate += timedelta(days=1)
    return candidate.astimezone(timezone.utc)


class RecurringTaskService:
    """Coordinates interval tasks and monitor state using the autonomy store."""

    VALID_KINDS = {"interval", "monitor"}
    VALID_STATUSES = {"active", "paused", "completed", "cancelled", "awaiting_approval", "blocked", "failed"}

    def __init__(
        self,
        *,
        autonomy_store: Any,
        default_interval_minutes: int = 30,
        minimum_interval_seconds: int = 10,
        max_active_tasks: int = 20,
        absence_threshold_minutes: int = 5,
        toast_notifier: Callable[[str, str], None] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.store = autonomy_store
        self.default_interval_seconds = max(10, int(default_interval_minutes) * 60)
        self.minimum_interval_seconds = max(10, int(minimum_interval_seconds))
        self.max_active_tasks = max(1, int(max_active_tasks))
        self.absence_threshold_seconds = max(60, int(absence_threshold_minutes) * 60)
        self.toast_notifier = toast_notifier
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def create(
        self,
        *,
        title: str,
        instruction: str,
        task_kind: str,
        source_kind: str = "screen",
        interval_seconds: int | None = None,
        schedule_time_local: str | None = None,
        monitor_condition: str = "",
        stop_condition: str = "",
        source_scope: dict[str, Any] | None = None,
        safe_actions: list[str] | None = None,
        origin_kind: str = "chat",
        origin_ref: str = "",
    ) -> RecurringTask:
        kind = str(task_kind or "").strip().lower()
        if kind not in self.VALID_KINDS:
            raise ValueError("task_kind must be interval or monitor")
        normalized_instruction = str(instruction or "").strip()
        if not normalized_instruction:
            raise ValueError("instruction is required")
        if kind == "monitor" and not str(monitor_condition or "").strip():
            raise ValueError("monitor_condition is required for monitor tasks")
        if len(self.store.list_recurring_tasks(status="active", limit=self.max_active_tasks + 1)) >= self.max_active_tasks:
            raise RuntimeError(f"maximum active recurring tasks reached ({self.max_active_tasks})")
        seconds = max(self.minimum_interval_seconds, int(interval_seconds or self.default_interval_seconds))
        normalized_time = _normalize_schedule_time(schedule_time_local) or (
            _time_from_instruction(normalized_instruction) if kind == "interval" else None
        )
        now_dt = _utcnow()
        now = _iso(now_dt)
        task = RecurringTask(
            task_id=uuid.uuid4().hex,
            title=str(title or normalized_instruction[:100]).strip()[:160],
            instruction=normalized_instruction,
            task_kind=kind,
            source_kind=str(source_kind or "screen").strip().lower(),
            status="active",
            interval_seconds=seconds,
            next_run_at=_iso(_next_local_time(normalized_time, now=now_dt) if normalized_time else now_dt),
            schedule_time_local=normalized_time,
            monitor_condition=str(monitor_condition or "").strip(),
            stop_condition=str(stop_condition or "").strip(),
            source_scope_json=json.dumps(source_scope or {}, ensure_ascii=False),
            safe_actions_json=json.dumps(sorted(set(safe_actions or [])), ensure_ascii=False),
            origin_kind=str(origin_kind or "chat"),
            origin_ref=str(origin_ref or ""),
            created_at=now,
            updated_at=now,
        )
        self.store.upsert_recurring_task(task)
        self.store.audit("ambient_agent", "recurring_task.created", task.task_id, {"kind": kind, "source": task.source_kind})
        return task

    def list(self, *, status: str | None = None, limit: int = 100) -> list[RecurringTask]:
        return self.store.list_recurring_tasks(status=status, limit=limit)

    def set_status(self, task_id: str, status: str) -> RecurringTask | None:
        normalized = str(status).strip().lower()
        if normalized not in self.VALID_STATUSES:
            raise ValueError(f"unsupported recurring task status: {status}")
        task = self.store.update_recurring_task_status(task_id, normalized)
        if task is not None:
            self.store.audit("local_user", f"recurring_task.{normalized}", task_id, {})
        return task

    def due_tasks(self, *, now: datetime | None = None, limit: int = 5) -> list[RecurringTask]:
        return self.store.list_due_recurring_tasks(_iso(now), limit=max(1, int(limit)))

    def mark_run_finished(self, task: RecurringTask, *, result: dict[str, Any], status: str = "active") -> RecurringTask | None:
        now = _utcnow()
        next_run = (
            _next_local_time(task.schedule_time_local, now=now)
            if task.schedule_time_local
            else now + timedelta(seconds=max(self.minimum_interval_seconds, task.interval_seconds))
        )
        return self.store.record_recurring_task_run(
            task.task_id,
            result=result,
            status=status,
            next_run_at=_iso(next_run),
            last_run_at=_iso(now),
        )

    @staticmethod
    def _scope_matches(task: RecurringTask, observation: Any) -> bool:
        try:
            scope = json.loads(task.source_scope_json or "{}")
        except json.JSONDecodeError:
            scope = {}
        if not scope:
            return True
        expected = {
            "app_name": str(scope.get("app_name") or "").strip().lower(),
            "process_name": str(scope.get("process_name") or "").strip().lower(),
            "domain": str(scope.get("domain") or "").strip().lower(),
        }
        raw = {}
        try:
            raw = json.loads(getattr(observation, "raw_payload_json", "") or "{}")
        except json.JSONDecodeError:
            pass
        actual = {
            "app_name": str(getattr(observation, "app_name", "") or "").strip().lower(),
            "process_name": str(raw.get("_uiat_process_name") or raw.get("process_name") or "").strip().lower(),
            "domain": str(raw.get("_uiat_domain") or raw.get("domain") or "").strip().lower(),
        }
        return all(not expected[key] or actual[key] == expected[key] for key in expected)

    @staticmethod
    def _condition_state(task: RecurringTask, observation: Any) -> tuple[str, str, float]:
        """Conservative factual condition detection over existing VLM output."""
        text = " ".join(
            str(value or "") for value in (
                getattr(observation, "summary", ""),
                getattr(observation, "detailed_description", ""),
                getattr(observation, "inferred_user_activity", ""),
                getattr(observation, "completed_items", ""),
            )
        ).lower()
        condition = task.monitor_condition.lower()
        complete_words = r"\b(complet(?:e|ed|ion)|finished|done|successful|100%|downloaded|ready)\b"
        progress_words = r"\b(download(?:ing)?|progress|remaining|installing|processing|uploading|loading)\b"
        target_tokens = [token for token in re.findall(r"[a-z0-9]{4,}", condition) if token not in {"until", "monitor", "change", "finished", "finish"}]
        target_match = not target_tokens or any(token in text for token in target_tokens)
        if target_match and re.search(complete_words, text):
            return "condition_met", text[:500], 0.80
        if target_match and re.search(progress_words, text):
            return "in_progress", text[:500], 0.70
        return "unchanged", text[:500], 0.45

    def evaluate_visual_observation(self, observation: Any, *, user_idle: bool, idle_seconds: float = 0.0) -> list[dict[str, Any]]:
        """Evaluate matching active screen monitors after a lightweight VLM result exists."""
        outcomes: list[dict[str, Any]] = []
        for task in self.list(status="active", limit=self.max_active_tasks):
            if task.task_kind != "monitor" or task.source_kind not in {"screen", "visual"}:
                continue
            if not self._scope_matches(task, observation):
                continue
            if task.stop_condition:
                visible_text = " ".join(
                    str(value or "") for value in (
                        getattr(observation, "summary", ""),
                        getattr(observation, "detailed_description", ""),
                    )
                ).lower()
                stop_tokens = [token for token in re.findall(r"[a-z0-9]{4,}", task.stop_condition.lower())]
                if stop_tokens and all(token in visible_text for token in stop_tokens):
                    self.store.upsert_recurring_monitor_state(
                        task.task_id, state="stopped", evidence=visible_text[:500], user_seen=False,
                        notification_state="none",
                    )
                    self.set_status(task.task_id, "completed")
                    outcomes.append({"task_id": task.task_id, "state": "stopped", "confidence": 0.75})
                    continue
            state, evidence, confidence = self._condition_state(task, observation)
            previous = self.store.get_recurring_monitor_state(task.task_id) or {}
            if state == "condition_met":
                previously_complete = previous.get("state") == "condition_met"
                already_toasted = previous.get("notification_state") == "toast_and_inbox"
                delivery = str(previous.get("notification_state") or "inbox") if previously_complete else "inbox"
                should_toast = (
                    user_idle and float(idle_seconds) >= self.absence_threshold_seconds
                    and self.toast_notifier is not None and not already_toasted
                )
                if should_toast:
                    delivery = "toast_and_inbox"
                self.store.upsert_recurring_monitor_state(
                    task.task_id,
                    state="condition_met",
                    evidence=evidence,
                    completion_fingerprint=getattr(observation, "observation_id", ""),
                    user_seen=False,
                    notification_state=delivery,
                )
                if not previously_complete:
                    self._surface_completion(task, evidence, "inbox")
                    self.mark_run_finished(task, result={"state": state, "evidence": evidence, "confidence": confidence})
                if should_toast:
                    self._surface_completion(task, evidence, "toast_only")
                    self.set_status(task.task_id, "completed")
            else:
                self.store.upsert_recurring_monitor_state(
                    task.task_id, state=state, evidence=evidence, user_seen=False,
                    notification_state=str(previous.get("notification_state") or "none"),
                )
            outcomes.append({"task_id": task.task_id, "state": state, "confidence": confidence})
        return outcomes

    def mark_visual_completion_seen(self, observation: Any) -> int:
        """Suppress future toast delivery only when active-user visual evidence matches scope."""
        changed = 0
        for task in self.list(status="active", limit=self.max_active_tasks):
            if task.task_kind != "monitor" or task.source_kind not in {"screen", "visual"} or not self._scope_matches(task, observation):
                continue
            state = self.store.get_recurring_monitor_state(task.task_id) or {}
            if state.get("state") != "condition_met" or state.get("user_seen"):
                continue
            detected, _, confidence = self._condition_state(task, observation)
            if detected == "condition_met" and confidence >= 0.7:
                self.store.upsert_recurring_monitor_state(
                    task.task_id, state="condition_met", evidence=str(state.get("evidence") or ""),
                    completion_fingerprint=str(state.get("completion_fingerprint") or ""), user_seen=True,
                    notification_state="seen_by_user",
                )
                self.set_status(task.task_id, "completed")
                changed += 1
        return changed

    def _surface_completion(self, task: RecurringTask, evidence: str, delivery: str) -> None:
        title = f"Monitor complete: {task.title}"
        summary = (evidence or f"The monitored condition was met: {task.monitor_condition}")[:700]
        if delivery != "toast_only":
            self.store.add_recurring_task_inbox_item(task.task_id, title=title, summary=summary, delivery=delivery)
        if delivery in {"toast_and_inbox", "toast_only"} and self.toast_notifier is not None:
            try:
                self.toast_notifier(title, "A monitored condition finished. Open Ambient AI for details.")
            except Exception:
                self.logger.exception("Could not deliver recurring-task toast for %s", task.task_id)

    @staticmethod
    def _todoist_interval_seconds(task: dict[str, Any], fallback: int) -> int:
        text = " ".join(str(task.get(key) or "") for key in ("due_string", "content", "description")).lower()
        match = re.search(r"every\s+(\d+)\s*(second|minute|hour|day)s?", text)
        if not match:
            return fallback
        multiplier = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}[match.group(2)]
        return max(10, int(match.group(1)) * multiplier)

    def sync_todoist_tasks(self, tasks: list[dict[str, Any]], *, label: str = "ambient") -> dict[str, int]:
        """Synchronize only explicit @ambient Todoist directives.

        Todoist remains the source of truth: completing/removing a labelled task
        disables its linked monitor instead of treating one occurrence as done.
        """
        normalized_label = str(label or "ambient").strip().lower().lstrip("@")
        selected = [
            item for item in tasks
            if normalized_label in {str(value).strip().lower().lstrip("@") for value in (item.get("labels") or [])}
        ]
        selected_ids = {str(item.get("id")) for item in selected if item.get("id") is not None}
        existing = {task.origin_ref: task for task in self.list(limit=500) if task.origin_kind == "todoist"}
        created = updated = cancelled = 0
        for item in selected:
            source_id = str(item.get("id") or "")
            content = str(item.get("content") or "").strip()
            if not source_id or not content:
                continue
            lower = f"{content} {item.get('description') or ''}".lower()
            kind = "monitor" if any(term in lower for term in ("monitor", "until", "watch for", "notify when")) else "interval"
            condition = ""
            if kind == "monitor":
                condition = re.split(r"\b(?:until|when)\b", content, maxsplit=1, flags=re.IGNORECASE)[-1].strip() or content
            interval = self._todoist_interval_seconds(item, self.default_interval_seconds)
            source_kind = "gmail" if "gmail" in lower or "email" in lower else "calendar" if "calendar" in lower else "screen" if kind == "monitor" else "todoist"
            old = existing.get(source_id)
            if old is None:
                self.create(
                    title=content[:160], instruction=content, task_kind=kind,
                    source_kind=source_kind,
                    interval_seconds=interval, monitor_condition=condition,
                    origin_kind="todoist", origin_ref=source_id,
                )
                created += 1
            elif old.instruction != content or old.interval_seconds != interval or old.task_kind != kind:
                replacement = RecurringTask(
                    **{**old.__dict__, "title": content[:160], "instruction": content, "task_kind": kind,
                       "interval_seconds": interval, "monitor_condition": condition,
                       "source_kind": source_kind,
                       "status": "active", "updated_at": _iso()}
                )
                self.store.upsert_recurring_task(replacement)
                updated += 1
        for source_id, task in existing.items():
            if source_id and source_id not in selected_ids and task.status not in {"cancelled", "completed"}:
                self.set_status(task.task_id, "cancelled")
                cancelled += 1
        return {"created": created, "updated": updated, "cancelled": cancelled, "selected": len(selected)}
