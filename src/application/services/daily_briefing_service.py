import asyncio
import hashlib
import json
import logging
from datetime import date, datetime, time as datetime_time, timedelta, timezone
from typing import Any


logger = logging.getLogger(__name__)


class DailyBriefingService:
    """Build a privacy-bounded daily activity view and cache an AI-written digest."""

    SYSTEM_PROMPT = (
        "You are Ambient AI writing a concise, personal daily update directly to your user. "
        "Write the overview as natural prose in first person: explain what you handled for them, "
        "what useful information you found, what failed or was blocked, and what still needs them. "
        "Use the bounded user context only when it genuinely makes the update more relevant; never expose "
        "unrelated private profile details. Use only supplied sanitized activity summaries and never invent facts. "
        "Return JSON only with keys headline, overview, accomplishments, updates, failures, attention. "
        "Overview should be 2-4 short paragraphs. The last four values must be arrays of bullet-point strings. "
        "Every bullet in accomplishments, updates, failures, and attention must be 1-2 short lines, concrete, "
        "and based on an actual supplied item title or summary. Do not write aggregate count summaries like "
        "'completed or updated 62 meaningful items'."
    )

    def __init__(
        self,
        *,
        autonomy_store: Any,
        report_store: Any,
        task_store: Any,
        organizer: Any,
        llm_provider: Any,
        user_context_service: Any = None,
        model: str,
        enabled: bool = True,
        cooldown_minutes: int = 15,
        max_items_per_source: int = 30,
        generation_timeout_seconds: float = 300.0,
    ):
        self.autonomy_store = autonomy_store
        self.report_store = report_store
        self.task_store = task_store
        self.organizer = organizer
        self.llm_provider = llm_provider
        self.user_context_service = user_context_service
        self.model = model
        self.enabled = bool(enabled)
        self.cooldown_minutes = max(1, int(cooldown_minutes))
        self.max_items_per_source = max(5, min(int(max_items_per_source), 200))
        self.generation_timeout_seconds = max(10.0, float(generation_timeout_seconds))
        self._retry_after: datetime | None = None
        self._refresh_running = False
        self._last_attempt_at: datetime | None = None
        self._last_error: str | None = None

    def snapshot(self, *, date_value: str | None = None, since: str | None = None) -> dict[str, Any]:
        local_now = datetime.now().astimezone()
        selected = self._parse_date(date_value) if date_value else local_now.date()
        collected = self._collect(selected, local_now=local_now)
        cached = self.autonomy_store.get_daily_briefing(selected.isoformat())
        since_at = self._parse_timestamp(since)
        for item in collected["timeline"]:
            item["is_new"] = bool(since_at and self._instant(item.get("occurred_at")) > since_at)
        for item in collected["attention"]:
            item["is_new"] = bool(since_at and self._instant(item.get("occurred_at")) > since_at)
        for item in collected["upcoming"]:
            item["is_new"] = bool(since_at and self._instant(item.get("occurred_at")) > since_at)
        for item in collected["urgent"]:
            item["is_new"] = bool(since_at and self._instant(item.get("occurred_at")) > since_at)
        return {
            "date": selected.isoformat(),
            "today": local_now.date().isoformat(),
            "timezone": str(local_now.tzinfo),
            "server_time": local_now.isoformat(),
            "counts": collected["counts"],
            "background": collected["background"],
            "timeline": collected["timeline"],
            "attention": collected["attention"],
            "upcoming": collected["upcoming"],
            "urgent": collected["urgent"],
            "new_count": sum(
                1
                for item in collected["timeline"] + collected["upcoming"] + collected["urgent"]
                if item["is_new"]
            ),
            "briefing": cached,
            "latest_headline": self._latest_headline(selected, collected),
            "latest_narrative": self._latest_narrative(selected, collected),
            "briefing_stale": bool(cached and cached.get("source_watermark") != collected["watermark"]),
            "briefing_pending": bool(self.enabled and collected["meaningful"] and (
                cached is None or cached.get("source_watermark") != collected["watermark"]
            )),
            "briefing_refresh": {
                "running": self._refresh_running,
                "last_attempt_at": self._last_attempt_at.isoformat() if self._last_attempt_at else None,
                "retry_after": self._retry_after.isoformat() if self._retry_after else None,
                "last_error": self._last_error,
            },
        }

    async def refresh_if_due(self, *, force: bool = False) -> dict[str, Any]:
        if not self.enabled or not self.model:
            return {"ran": False, "reason": "disabled"}
        if not force and self._retry_after and datetime.now(timezone.utc) < self._retry_after:
            return {"ran": False, "reason": "retry_backoff"}
        local_now = datetime.now().astimezone()
        # The visible current-day briefing must never be blocked by a historical
        # finalization failure. Yesterday is finalized after today is current.
        candidates = [local_now.date(), local_now.date() - timedelta(days=1)]
        for selected in candidates:
            collected = self._collect(selected, local_now=local_now)
            if not collected["meaningful"]:
                continue
            cached = self.autonomy_store.get_daily_briefing(selected.isoformat())
            if not force and cached and cached.get("source_watermark") == collected["watermark"]:
                continue
            if not force and cached and selected == local_now.date():
                generated = self._parse_timestamp(cached.get("generated_at"))
                if generated and (datetime.now(timezone.utc) - generated).total_seconds() < self.cooldown_minutes * 60:
                    continue
            try:
                self._refresh_running = True
                self._last_attempt_at = datetime.now(timezone.utc)
                briefing = await asyncio.wait_for(
                    self._generate(selected, collected, local_now),
                    timeout=self.generation_timeout_seconds,
                )
            except Exception as exc:
                logger.exception("Daily briefing generation failed for %s.", selected)
                self._retry_after = datetime.now(timezone.utc) + timedelta(minutes=5)
                error_text = (
                    f"Daily briefing generation timed out after {self.generation_timeout_seconds:g} seconds."
                    if isinstance(exc, TimeoutError)
                    else (str(exc).strip() or exc.__class__.__name__)
                )
                self._last_error = error_text[:1000]
                return {"ran": True, "status": "failed", "date": selected.isoformat(), "error": error_text}
            finally:
                self._refresh_running = False
            self.autonomy_store.upsert_daily_briefing(briefing)
            self._retry_after = None
            self._last_error = None
            return {"ran": True, "status": "completed", "date": selected.isoformat()}
        return {"ran": False, "reason": "up_to_date"}

    def is_due(self) -> bool:
        if not self.enabled or not self.model:
            return False
        if self._retry_after and datetime.now(timezone.utc) < self._retry_after:
            return False
        local_now = datetime.now().astimezone()
        for selected in (local_now.date(), local_now.date() - timedelta(days=1)):
            collected = self._collect(selected, local_now=local_now)
            if not collected["meaningful"]:
                continue
            cached = self.autonomy_store.get_daily_briefing(selected.isoformat())
            if cached and cached.get("source_watermark") == collected["watermark"]:
                continue
            if cached and selected == local_now.date():
                generated = self._parse_timestamp(cached.get("generated_at"))
                if generated and (datetime.now(timezone.utc) - generated).total_seconds() < self.cooldown_minutes * 60:
                    continue
            return True
        return False

    async def _generate(
        self, selected: date, collected: dict[str, Any], local_now: datetime
    ) -> dict[str, Any]:
        safe_payload = {
            "date": selected.isoformat(),
            "counts": collected["counts"],
            "background": collected["background"],
            "activity": [
                {
                    "kind": item["kind"],
                    "title": item["title"][:180],
                    "summary": item["summary"][:700],
                    "status": item["status"],
                    "occurred_at": item["occurred_at"],
                }
                for item in collected["timeline"][: self.max_items_per_source * 3]
            ],
            "attention": [
                {
                    "kind": item["kind"],
                    "title": item["title"][:180],
                    "summary": item["summary"][:500],
                    "status": item["status"],
                }
                for item in collected["attention"][: self.max_items_per_source]
            ],
            "upcoming": [
                {
                    "kind": item["kind"],
                    "title": item["title"][:180],
                    "summary": item["summary"][:500],
                    "status": item["status"],
                    "scheduled_for": item.get("scheduled_for"),
                }
                for item in collected["upcoming"][: self.max_items_per_source]
            ],
            "urgent": [
                {
                    "kind": item["kind"],
                    "title": item["title"][:180],
                    "summary": item["summary"][:500],
                    "status": item["status"],
                }
                for item in collected["urgent"][: self.max_items_per_source]
            ],
            "user_context": self._personalization_context(),
        }
        completion = await self.llm_provider.chat_completion_stream(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(safe_payload, ensure_ascii=False)},
            ],
            tools=None,
            image="",
            temperature=0.2,
        )
        parts: list[str] = []
        async for chunk in completion:
            if not getattr(chunk, "choices", None):
                continue
            content = getattr(chunk.choices[0].delta, "content", None)
            if content:
                parts.append(content)
        parsed = self._parse_json("".join(parts))
        headline = str(parsed.get("headline") or "").strip()
        overview = str(parsed.get("overview") or "").strip()
        if not headline or not overview:
            raise ValueError("Daily briefing model returned incomplete JSON.")
        return {
            "briefing_date": selected.isoformat(),
            "timezone": str(local_now.tzinfo),
            "headline": headline[:240],
            "overview": overview[:3000],
            "accomplishments": self._string_list(parsed.get("accomplishments"), 12, max_chars=220),
            "updates": self._string_list(parsed.get("updates"), 12, max_chars=220),
            "failures": self._string_list(parsed.get("failures"), 12, max_chars=220),
            "attention": self._string_list(parsed.get("attention"), 12, max_chars=220),
            "source_counts": collected["counts"],
            "source_watermark": collected["watermark"],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _collect(self, selected: date, *, local_now: datetime) -> dict[str, Any]:
        local_tz = local_now.tzinfo or timezone.utc
        start_local = datetime.combine(selected, datetime_time.min, tzinfo=local_tz)
        end_local = start_local + timedelta(days=1)
        start_iso = start_local.astimezone(timezone.utc).isoformat()
        end_iso = end_local.astimezone(timezone.utc).isoformat()
        limit = self.max_items_per_source

        reports = self.report_store.list_reports_between(start_iso, end_iso, limit=limit)
        inbox = self.autonomy_store.list_inbox_items_between(start_iso, end_iso, limit=limit)
        delegations = self.autonomy_store.list_delegated_tasks_between(start_iso, end_iso, limit=limit)
        activities = self.autonomy_store.list_activity_runs_between(start_iso, end_iso, limit=limit)
        pending_approvals = self.autonomy_store.list_approvals(status="pending", limit=limit)
        tasks = self.task_store.get_all_pending_tasks()[:limit]
        if selected != local_now.date():
            pending_approvals = [
                item for item in pending_approvals
                if self._within_day(item.created_at, selected, local_tz)
            ]
            tasks = [
                item for item in tasks
                if self._within_day(item.created_at, selected, local_tz)
            ]
        artifacts = self._artifacts_for_day(selected, local_tz, limit)
        audits = self.autonomy_store.list_audit_between(start_iso, end_iso, limit=limit)
        event_counts = self.autonomy_store.daily_event_counts(start_iso, end_iso)

        timeline: list[dict[str, Any]] = []
        for row in reports:
            report = self._json(row.report_json, {})
            timeline.append(self._item(
                item_id=f"report:{row.interaction_id}", kind="report",
                title=str(report.get("title") or "Agent report"),
                summary=str(report.get("summary") or row.response_text or "")[:900],
                status=str(report.get("status") or ("failed" if row.error_text else "completed")),
                occurred_at=row.completed_at or row.created_at, destination="/reports",
                source_ref=row.interaction_id,
            ))
        for row in inbox:
            timeline.append(self._item(
                item_id=f"inbox:{row.inbox_id}", kind="proactive_update", title=row.title,
                summary=row.summary, status=row.status, occurred_at=row.updated_at,
                destination="/inbox", source_ref=row.inbox_id,
            ))
        for row in delegations:
            result = self._json(row.result_json, {})
            summary = str(result.get("summary") or row.final_response or row.error_text or row.reason)
            timeline.append(self._item(
                item_id=f"delegation:{row.delegation_id}", kind="delegated_task",
                title=f"{row.capability.replace('_', ' ').title()}: {row.task[:120]}",
                summary=summary[:900], status=row.status, occurred_at=row.completed_at or row.updated_at,
                destination="/inbox", source_ref=row.delegation_id,
            ))
        for row in activities:
            timeline.append(self._item(
                item_id=f"activity:{row.run_id}", kind="agent_activity", title=row.title,
                summary=(row.summary or row.error_text or "Agent work unit")[:900], status=row.status,
                occurred_at=row.completed_at or row.created_at, destination="/reports", source_ref=row.run_id,
            ))
        for artifact in artifacts:
            timeline.append(self._item(
                item_id=f"artifact:{artifact['artifact_id']}", kind="artifact",
                title=str(artifact.get("title") or "Artifact updated"),
                summary=str(artifact.get("short_summary") or "Knowledge artifact updated"),
                status=str(artifact.get("status") or "active"),
                occurred_at=str(artifact.get("last_ai_edited_at") or artifact.get("created_at")),
                destination="/artifacts", source_ref=str(artifact["artifact_id"]),
            ))
        timeline.sort(key=lambda item: self._instant(item["occurred_at"]), reverse=True)

        attention: list[dict[str, Any]] = []
        for approval in pending_approvals:
            constraints = self._json(approval.constraints_json, {})
            attention.append({
                **self._item(
                    item_id=f"approval:{approval.approval_id}", kind="approval",
                    title=f"Approve {approval.capability.replace('_', ' ')}",
                    summary=str(constraints.get("reason") or constraints.get("task") or "A bounded agent action needs your decision."),
                    status=approval.status, occurred_at=approval.created_at,
                    destination="/approvals", source_ref=approval.approval_id,
                ),
                "approval_id": approval.approval_id,
                "capability": approval.capability,
                "expires_at": approval.expires_at,
                "constraints": constraints,
            })
        for task in tasks:
            attention.append(self._item(
                item_id=f"task:{task.id}", kind="queued_task", title=task.description[:180],
                summary="Scheduled work waiting to run." if task.run_at_utc else "Background work waiting for an idle execution window.",
                status=task.status, occurred_at=task.created_at or local_now.isoformat(),
                destination="/reports", source_ref=str(task.id),
            ))
        for item in timeline:
            if item["status"] in {"failed", "blocked", "completed_with_blocker", "dead_letter"}:
                attention.append({**item, "id": f"attention:{item['id']}"})
        urgent = [
            {**item, "id": f"urgent:{item['id']}"}
            for item in attention
            if self._requires_urgent_attention(item)
        ]
        upcoming = self._upcoming_items(
            selected=selected,
            local_tz=local_tz,
            local_now=local_now,
            tasks=tasks,
            inbox=inbox,
        )

        background_actions: dict[str, int] = {}
        for audit in audits:
            action = str(audit.get("action") or "other")
            background_actions[action] = background_actions.get(action, 0) + 1
        maintenance_runs = []
        if self.organizer is not None:
            maintenance_runs = [
                row for row in self.organizer.list_maintenance_history(limit=limit).get("runs", [])
                if self._within_day(row.get("completed_at") or row.get("started_at"), selected, local_tz)
            ]
        background = {
            "events": event_counts,
            "audit_actions": background_actions,
            "artifact_maintenance_runs": len(maintenance_runs),
        }
        counts = {
            "reports": len(reports), "proactive_updates": len(inbox),
            "delegated_tasks": len(delegations), "activity_runs": len(activities),
            "artifact_changes": len(artifacts), "pending_approvals": len(pending_approvals),
            "queued_tasks": len(tasks), "attention": len(attention),
            "upcoming": len(upcoming), "urgent": len(urgent),
        }
        # Runtime event counts and resource/model audits are useful live
        # telemetry, but they must not invalidate the digest: restarts, leases,
        # retries, and digest generation itself can change those counters
        # without changing the human-meaningful daily story.
        stable_background = {
            "artifact_maintenance_runs": len(maintenance_runs),
        }
        canonical = {
            "date": selected.isoformat(), "counts": counts, "background": stable_background,
            "personalization_signature": hashlib.sha256(
                self._personalization_context().encode("utf-8")
            ).hexdigest(),
            "timeline": [{k: item[k] for k in ("id", "title", "summary", "status", "occurred_at")} for item in timeline],
            "attention": [{k: item.get(k) for k in ("id", "title", "status", "occurred_at")} for item in attention],
            "upcoming": [{k: item.get(k) for k in ("id", "title", "status", "occurred_at", "scheduled_for")} for item in upcoming],
            "urgent": [{k: item.get(k) for k in ("id", "title", "status", "occurred_at")} for item in urgent],
        }
        watermark = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
        ).hexdigest()
        return {
            "counts": counts, "background": background, "timeline": timeline,
            "attention": attention, "upcoming": upcoming, "urgent": urgent,
            "watermark": watermark,
            "meaningful": bool(timeline or attention or upcoming or urgent or any(event_counts.values()) or maintenance_runs),
        }

    def _personalization_context(self) -> str:
        if self.user_context_service is None:
            return ""
        try:
            context = self.user_context_service.build_context(include_semantic=False)
        except Exception:
            logger.exception("Could not load bounded personalization for the daily briefing.")
            return ""
        stable = str(context.get("stable_user_profile") or "").strip()[:1400]
        working = str(context.get("working_memory") or "").strip()[:900]
        parts = []
        if stable:
            parts.append(f"Stable profile:\n{stable}")
        if working:
            parts.append(f"Current priorities:\n{working}")
        return "\n\n".join(parts)[:2300]

    @staticmethod
    def _latest_headline(selected: date, collected: dict[str, Any]) -> str:
        counts = collected["counts"]
        failures = sum(
            1 for item in collected["timeline"]
            if item["status"] in {"failed", "blocked", "dead_letter", "completed_with_blocker"}
        )
        if failures:
            return f"I made progress, but {failures} item{'s' if failures != 1 else ''} need a closer look"
        if counts["reports"] or counts["activity_runs"] or counts["artifact_changes"]:
            return "Here’s what I handled for you"
        if counts["pending_approvals"] or counts["queued_tasks"]:
            return "I’m waiting on a few next steps"
        return "Nothing important changed yet"

    @staticmethod
    def _latest_narrative(selected: date, collected: dict[str, Any]) -> str:
        counts = collected["counts"]
        timeline = collected["timeline"]
        completed = [
            item for item in timeline
            if item["status"] in {"completed", "active", "processed", "ready"}
        ]
        failed = [
            item for item in timeline
            if item["status"] in {"failed", "blocked", "dead_letter", "completed_with_blocker"}
        ]
        parts: list[str] = []
        if completed:
            examples = "; ".join(
                DailyBriefingService._compact_item_text(item)
                for item in completed[:4]
            )
            parts.append(f"Recent completed work: {examples}.")
        if counts["proactive_updates"] or counts["artifact_changes"]:
            details = []
            if counts["proactive_updates"]:
                details.append(f"surfaced {counts['proactive_updates']} proactive update{'s' if counts['proactive_updates'] != 1 else ''}")
            if counts["artifact_changes"]:
                details.append(f"organized {counts['artifact_changes']} artifact{'s' if counts['artifact_changes'] != 1 else ''}")
            parts.append("I also " + " and ".join(details) + ".")
        if failed:
            examples = "; ".join(
                f"**{item['title']}** ({item['summary'][:180] or item['status']})" for item in failed[:3]
            )
            parts.append(
                f"I could not fully complete **{len(failed)}** item"
                f"{'s' if len(failed) != 1 else ''}: {examples}."
            )
        pending = counts["pending_approvals"] + counts["queued_tasks"]
        if pending:
            parts.append(
                f"There {'are' if pending != 1 else 'is'} **{pending}** pending item"
                f"{'s' if pending != 1 else ''}. I’ll continue automatically where permitted; approvals still need your decision."
            )
        if not parts:
            parts.append("I haven’t completed, failed, or queued any meaningful work for this day yet. I’ll update this space as soon as something changes.")
        return "\n\n".join(parts)

    def _upcoming_items(
        self,
        *,
        selected: date,
        local_tz: Any,
        local_now: datetime,
        tasks: list[Any],
        inbox: list[Any],
    ) -> list[dict[str, Any]]:
        upcoming: list[dict[str, Any]] = []
        for task in tasks:
            scheduled_at = self._parse_timestamp(getattr(task, "run_at_utc", None))
            scheduled_for = scheduled_at.astimezone(local_tz).isoformat() if scheduled_at else None
            title = str(getattr(task, "description", "") or "Queued task").strip()
            upcoming.append(self._item(
                item_id=f"upcoming-task:{getattr(task, 'id', title)}",
                kind="scheduled_task" if scheduled_for else "queued_task",
                title=title[:180],
                summary=(
                    f"Scheduled for {format(scheduled_at.astimezone(local_tz), '%b %d, %I:%M %p')}."
                    if scheduled_at
                    else "Waiting for an idle execution window."
                ),
                status=str(getattr(task, "status", "pending") or "pending"),
                occurred_at=str(getattr(task, "created_at", None) or scheduled_for or local_now.isoformat()),
                scheduled_for=scheduled_for,
                destination="/reports",
                source_ref=str(getattr(task, "id", "")),
            ))
        for item in inbox:
            if not self._looks_like_upcoming_event(item):
                continue
            updated_at = str(getattr(item, "updated_at", None) or local_now.isoformat())
            upcoming.append(self._item(
                item_id=f"upcoming-event:{getattr(item, 'inbox_id', updated_at)}",
                kind="upcoming_event",
                title=str(getattr(item, "title", "") or "Upcoming event")[:180],
                summary=str(getattr(item, "summary", "") or "Event-related proactive update.")[:900],
                status=str(getattr(item, "status", "pending") or "pending"),
                occurred_at=updated_at,
                scheduled_for=None,
                destination="/inbox",
                source_ref=str(getattr(item, "inbox_id", "")),
            ))
        upcoming.sort(
            key=lambda item: (
                self._instant(item.get("scheduled_for")) if item.get("scheduled_for") else datetime.max.replace(tzinfo=timezone.utc),
                self._instant(item.get("occurred_at")),
            )
        )
        if selected != local_now.date():
            return [
                item for item in upcoming
                if self._within_day(item.get("scheduled_for") or item.get("occurred_at"), selected, local_tz)
            ][: self.max_items_per_source]
        return upcoming[: self.max_items_per_source]

    @classmethod
    def _requires_urgent_attention(cls, item: dict[str, Any]) -> bool:
        if item.get("kind") == "approval":
            return True
        status = str(item.get("status") or "").strip().lower()
        if status in {"failed", "blocked", "dead_letter", "completed_with_blocker"}:
            return True
        text = f"{item.get('title', '')} {item.get('summary', '')}".lower()
        return any(marker in text for marker in ("urgent", "overdue", "deadline today", "requires your approval"))

    def _looks_like_upcoming_event(self, item: Any) -> bool:
        text = " ".join(
            str(value or "")
            for value in [
                getattr(item, "title", ""),
                getattr(item, "summary", ""),
                getattr(item, "sources_json", ""),
                getattr(item, "actions_json", ""),
            ]
        ).lower()
        if "calendar" in text:
            return True
        event_markers = ("meeting", "event", "appointment", "deadline", "interview", "exam", "test", "travel")
        date_markers = ("today", "tomorrow", "next ", " at ", " on ", "am", "pm")
        return any(marker in text for marker in event_markers) and any(marker in text for marker in date_markers)

    def _artifacts_for_day(self, selected: date, local_tz: Any, limit: int) -> list[dict[str, Any]]:
        if self.organizer is None:
            return []
        rows = self.organizer.list_artifacts(status="active", limit=max(limit * 4, 100))
        rows += self.organizer.list_artifacts(status="archived", limit=max(limit * 2, 50))
        return [row for row in rows if self._within_day(row.get("last_ai_edited_at"), selected, local_tz)][:limit]

    @staticmethod
    def _item(**kwargs: Any) -> dict[str, Any]:
        if "item_id" in kwargs:
            kwargs["id"] = kwargs.pop("item_id")
        return {"is_new": False, **kwargs}

    @staticmethod
    def _json(value: Any, fallback: Any) -> Any:
        if isinstance(value, (dict, list)):
            return value
        try:
            return json.loads(value or "")
        except (TypeError, ValueError):
            return fallback

    @classmethod
    def _parse_json(cls, raw: str) -> dict[str, Any]:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:])
        payload = cls._json(text, {})
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _string_list(value: Any, limit: int, *, max_chars: int = 500) -> list[str]:
        if not isinstance(value, list):
            return []
        output = []
        for item in value:
            text = " ".join(str(item or "").split())
            if not text:
                continue
            if len(text) > max_chars:
                text = text[: max(1, max_chars - 1)].rstrip() + "..."
            output.append(text)
            if len(output) >= limit:
                break
        return output

    @staticmethod
    def _compact_item_text(item: dict[str, Any], *, max_chars: int = 170) -> str:
        title = str(item.get("title") or "").strip()
        summary = str(item.get("summary") or "").strip()
        text = title if title else summary
        if title and summary and summary.lower() not in title.lower():
            text = f"{title} - {summary}"
        text = " ".join(text.split())
        if len(text) > max_chars:
            return text[: max(1, max_chars - 1)].rstrip() + "..."
        return text or str(item.get("kind") or "item")

    @staticmethod
    def _parse_date(value: str) -> date:
        return date.fromisoformat(str(value))

    @staticmethod
    def _parse_timestamp(value: Any) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=datetime.now().astimezone().tzinfo)
        return parsed.astimezone(timezone.utc)

    @classmethod
    def _instant(cls, value: Any) -> datetime:
        return cls._parse_timestamp(value) or datetime.min.replace(tzinfo=timezone.utc)

    @classmethod
    def _within_day(cls, value: Any, selected: date, local_tz: Any) -> bool:
        parsed = cls._parse_timestamp(value)
        return bool(parsed and parsed.astimezone(local_tz).date() == selected)
