import asyncio
import json
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from application.services.capability_policy_service import CapabilityRegistry
from application.services.interaction_trace import interaction_trace
from core.models import OpportunityCandidate, ProactiveInboxItem, VisualObservation


@dataclass(frozen=True)
class ProactiveFinding:
    source: str
    title: str
    summary: str
    importance: str
    evidence: list[str]
    suggested_next_step: str
    requires_user_action: bool
    confidence: float
    sensitive: bool = True


class ProactiveSweepService:
    """Idle-time read-only personal-source sweep for proactive Home/Inbox updates."""

    SYSTEM_PROMPT = """You are Ambient AI running an idle proactive source check.

Inspect only the approved source for important updates that may help the user.
You are an agent loop with tools available. First explore the approved source
with read-only tools as needed; you may call multiple tools across multiple
turns to understand the source before deciding. Do not finalize on the first
turn unless the user context and task explicitly prove no tool can help.

Never send, post, delete, edit, upload, download, purchase, submit forms, or
change account state. If a tool result reveals that the source requires login,
permission, or other missing access, stop exploration and report that blocker
as a finding.

Only after you are done exploring, return the final answer as JSON only:
{
  "findings": [
    {
      "source": "gmail|calendar|browser_research|whatsapp|filesystem|other",
      "title": "short finding title",
      "summary": "specific useful summary",
      "importance": "low|medium|high|urgent",
      "evidence": ["short evidence strings"],
      "suggested_next_step": "what the user or Ambient AI should do next",
      "requires_user_action": true,
      "confidence": 0.0,
      "sensitive": true
    }
  ]
}

Rules:
- Surface only useful findings: deadlines, commitments, important messages, conflicts,
  unanswered requests, urgent notifications, travel/work/calendar changes, or clear
  opportunities to help.
- If nothing meaningful is found, return {"findings":[]}.
- The final JSON must summarize what the tool exploration actually found; do not
  invent findings from user context alone.
- Prefer fewer high-signal findings over generic summaries.
- Do not include private raw message bodies unless needed as short evidence.
- Do not invent facts. If a source cannot be checked, return one medium-importance
  finding explaining the blocker.
"""

    SOURCE_TASKS = {
        "gmail": (
            "Check Gmail for important recent emails, unread messages, replies needed, "
            "deadlines, bills, meetings, travel, account/security notices, recruiter/work "
            "messages, or anything the user should not miss. Use read/search/list Gmail tools only. "
            "Required Gmail workflow: first call search_gmail_messages with a Gmail search query; "
            "then extract the returned Gmail message IDs; only then call get_gmail_messages_content_batch "
            "with those exact message IDs. Never call get_gmail_messages_content_batch with an empty "
            "message_ids list, and never use 'me' as a message_id."
        ),
        "calendar": (
            "Check Google Calendar for today and the next 7 days. Find upcoming events, "
            "conflicts, preparation needs, reminders, and schedule changes. Use read/list calendar tools only."
        ),
        "browser_research": (
            "Do one quick web research check related to the user's recent stable interests or working memory. "
            "Only research public information and return a concise useful update with sources."
        ),
        "whatsapp": (
            "Open or inspect the visible WhatsApp desktop/web session if available. Read only visible "
            "recent chats and identify important commitments, dates, requests, or follow-ups. Do not type, "
            "send, delete, download, or open private media. Stop at login, QR, locked, or ambiguous screens."
        ),
        "filesystem": (
            "Inspect only configured/granted local filesystem context for important recent documents or "
            "notes that need attention. Do not edit, move, delete, or run shell commands."
        ),
    }

    IMPORTANCE_ORDER = {"low": 0, "medium": 1, "high": 2, "urgent": 3}

    def __init__(
        self,
        *,
        autonomy_store: Any,
        llm_service: Any,
        capability_policy: Any,
        semantic_dedupe_service: Any = None,
        user_context_service: Any = None,
        memory: Any = None,
        model: str,
        enabled: bool = False,
        global_grant: bool = False,
        enabled_sources: Optional[list[str]] = None,
        cadence_minutes: int = 60,
        max_sweeps_per_day: int = 8,
        max_findings_per_sweep: int = 10,
        minimum_importance: str = "medium",
        max_source_seconds: float = 180.0,
        filesystem_paths: Optional[list[str]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.autonomy_store = autonomy_store
        self.llm_service = llm_service
        self.capability_policy = capability_policy
        self.semantic_dedupe = semantic_dedupe_service
        self.user_context_service = user_context_service
        self.memory = memory
        self.model = str(model or "").strip()
        self.enabled = bool(enabled)
        self.global_grant = bool(global_grant)
        self.enabled_sources = [
            item.strip().lower()
            for item in (enabled_sources or ["gmail", "calendar"])
            if item.strip()
        ]
        self.cadence_minutes = max(1, int(cadence_minutes))
        self.max_sweeps_per_day = max(1, int(max_sweeps_per_day))
        self.max_findings_per_sweep = max(1, int(max_findings_per_sweep))
        self.minimum_importance = str(minimum_importance or "medium").strip().lower()
        self.max_source_seconds = max(10.0, float(max_source_seconds))
        self.filesystem_paths = [str(path).strip() for path in (filesystem_paths or []) if str(path).strip()]
        self.registry = CapabilityRegistry()
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def is_due(self) -> bool:
        if not self.enabled or not self.global_grant or not self.model:
            return False
        runs = self._today_runs()
        completed = [
            run for run in runs
            if run.status in {"completed", "completed_with_blocker", "failed", "blocked"}
        ]
        if len(completed) >= self.max_sweeps_per_day:
            return False
        latest = max(
            (self._parse_time(run.completed_at or run.created_at) for run in runs),
            default=None,
        )
        if latest is None:
            return True
        return datetime.now(timezone.utc) - latest >= timedelta(minutes=self.cadence_minutes)

    async def run_if_due(self) -> dict[str, Any]:
        if not self.is_due():
            return {"ran": False, "reason": "not_due_or_disabled"}
        return await self.run()

    async def run(self) -> dict[str, Any]:
        run = self.autonomy_store.queue_run(
            title="Idle proactive personal-source sweep",
            source_kind="proactive_sweep",
            trigger_kind="idle",
            priority="medium",
            metadata={
                "sources": self.enabled_sources,
                "global_grant": self.global_grant,
            },
        )
        if hasattr(self.autonomy_store, "audit"):
            self.autonomy_store.audit(
                "ambient_agent",
                "proactive_sweep.started",
                run.run_id,
                {"sources": self.enabled_sources},
            )

        findings: list[ProactiveFinding] = []
        errors: dict[str, str] = {}
        try:
            self._apply_global_read_policy()
            for source in self.enabled_sources:
                if len(findings) >= self.max_findings_per_sweep:
                    break
                try:
                    source_findings = await self._scan_source_with_timeout(source)
                    findings.extend(source_findings)
                except Exception as exc:
                    self.logger.exception("Proactive source %s failed.", source)
                    errors[source] = str(exc)[:500]
                    findings.append(
                        ProactiveFinding(
                            source=source,
                            title=f"{source.replace('_', ' ').title()} check failed",
                            summary=str(exc)[:700] or "The source could not be checked.",
                            importance="medium",
                            evidence=[],
                            suggested_next_step="Check integration/session configuration.",
                            requires_user_action=True,
                            confidence=0.7,
                        )
                    )

            created = []
            for finding in self._filter_findings(findings)[: self.max_findings_per_sweep]:
                if await self._is_duplicate(finding):
                    continue
                item = self._create_inbox_item(finding)
                created.append(item)
                self._record_created(finding, item)
                self._record_biodata_candidate(finding, item)

            summary = f"Checked {len(self.enabled_sources)} source(s); surfaced {len(created)} finding(s)."
            if errors:
                summary += f" {len(errors)} source(s) had blockers."
            self.autonomy_store.complete_run(
                run.run_id,
                summary=summary,
                output_text=json.dumps(
                    {
                        "created_findings": [item.inbox_id for item in created],
                        "errors": errors,
                        "sources": self.enabled_sources,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                status="completed" if not errors else "completed_with_blocker",
                error_text=json.dumps(errors, ensure_ascii=False) if errors else None,
            )
            if hasattr(self.autonomy_store, "audit"):
                self.autonomy_store.audit(
                    "ambient_agent",
                    "proactive_sweep.completed",
                    run.run_id,
                    {"created_findings": len(created), "errors": errors},
                )
            return {
                "ran": True,
                "status": "completed",
                "run_id": run.run_id,
                "created_findings": len(created),
                "errors": errors,
            }
        except Exception as exc:
            self.autonomy_store.complete_run(
                run.run_id,
                summary="Idle proactive sweep failed.",
                output_text="",
                status="failed",
                error_text=str(exc)[:1000],
            )
            raise

    async def _scan_source_with_timeout(self, source: str) -> list[ProactiveFinding]:
        return await self._wait_for(self._scan_source(source), source=source)

    async def _scan_source(self, source: str) -> list[ProactiveFinding]:
        source = source.strip().lower()
        if source == "whatsapp":
            return await self._wait_for(self._scan_whatsapp(), source=source)
        if source == "filesystem":
            return await self._wait_for(self._scan_filesystem(), source=source)

        allowed = self._allowed_tools_for_source(source)
        if not allowed:
            raise RuntimeError(f"No read-only tools are available for proactive source '{source}'.")
        prompt = self.SOURCE_TASKS.get(source, self.SOURCE_TASKS["browser_research"])
        context = self._personalization_context(source)
        user_input = (
            f"Approved proactive source: {source}\n"
            f"Task: {prompt}\n\n"
            f"User context:\n{context or '(none)'}"
        )
        with interaction_trace(
            "autonomy_proactive_sweep",
            {
                "autonomy_confidence": 0.95,
                "proactive_source": source,
            },
        ):
            self.llm_service.reset_context()
            try:
                text = await self.llm_service.run_interaction(
                    user_input=user_input,
                    system_prompt=self.SYSTEM_PROMPT,
                    model=self.model,
                    allowed_tool_names=allowed,
                    report_policy="silent",
                )
            finally:
                self.llm_service.reset_context()
        return self._parse_findings(text, fallback_source=source)

    async def _wait_for(self, awaitable: Any, *, source: str) -> Any:
        try:
            return await asyncio.wait_for(awaitable, timeout=self.max_source_seconds)
        except TimeoutError as exc:
            raise RuntimeError(
                f"Proactive source '{source}' exceeded {self.max_source_seconds:g}s."
            ) from exc

    async def _scan_whatsapp(self) -> list[ProactiveFinding]:
        if not getattr(self.llm_service, "computer_enabled", False):
            raise RuntimeError("WhatsApp proactive check requires [computer] enabled=true.")
        result = await self.llm_service.deploy_computer_agent(
            task=self.SOURCE_TASKS["whatsapp"],
            approval_id="global_proactive_readonly_computer",
            read_only=True,
        )
        return [
            ProactiveFinding(
                source="whatsapp",
                title="WhatsApp proactive check",
                summary=result[:1200],
                importance="medium",
                evidence=[],
                suggested_next_step="Review the WhatsApp check result.",
                requires_user_action=True,
                confidence=0.65,
            )
        ]

    async def _scan_filesystem(self) -> list[ProactiveFinding]:
        if not self.filesystem_paths:
            raise RuntimeError("Filesystem proactive check has no configured granted paths.")
        allowed = {"use_filesystem"}
        user_input = (
            "Approved proactive source: filesystem\n"
            f"Task: {self.SOURCE_TASKS['filesystem']}\n"
            f"Granted paths: {json.dumps(self.filesystem_paths, ensure_ascii=False)}"
        )
        with interaction_trace(
            "autonomy_proactive_sweep",
            {"autonomy_confidence": 0.95, "proactive_source": "filesystem"},
        ):
            self.llm_service.reset_context()
            try:
                text = await self.llm_service.run_interaction(
                    user_input=user_input,
                    system_prompt=self.SYSTEM_PROMPT,
                    model=self.model,
                    allowed_tool_names=allowed,
                    report_policy="silent",
                )
            finally:
                self.llm_service.reset_context()
        return self._parse_findings(text, fallback_source="filesystem")

    def _allowed_tools_for_source(self, source: str) -> set[str]:
        names: set[str] = set()
        for tool in self.llm_service.available_tool_definitions():
            name = str(tool.get("function", {}).get("name") or "")
            lowered = name.lower()
            descriptor = self.registry.describe(name)
            if source == "gmail":
                if descriptor.capability == "communication.read" and any(
                    marker in lowered for marker in ("gmail", "email", "mail", "message")
                ):
                    names.add(name)
            elif source == "calendar":
                if descriptor.capability == "assistance.calendar.read":
                    names.add(name)
            elif source == "browser_research":
                if descriptor.capability == "research.web" and name != "use_browser":
                    names.add(name)
        return names

    def _apply_global_read_policy(self) -> None:
        if not self.global_grant or self.capability_policy is None:
            return
        store = getattr(self.capability_policy, "store", None)
        if store is None or not hasattr(store, "set_policy"):
            return
        for capability in ("communication.read", "assistance.calendar.read", "research.web", "filesystem.read"):
            store.set_policy(
                capability,
                "trusted_bounded",
                {"source": "proactive_global_grant", "read_only": True},
            )

    def _filter_findings(self, findings: list[ProactiveFinding]) -> list[ProactiveFinding]:
        threshold = self.IMPORTANCE_ORDER.get(self.minimum_importance, 1)
        filtered = [
            finding for finding in findings
            if self.IMPORTANCE_ORDER.get(finding.importance, 0) >= threshold
        ]
        return sorted(
            filtered,
            key=lambda item: (self.IMPORTANCE_ORDER.get(item.importance, 0), item.confidence),
            reverse=True,
        )

    async def _is_duplicate(self, finding: ProactiveFinding) -> bool:
        if self.semantic_dedupe is None:
            return False
        text = f"{finding.source}: {finding.title}\n{finding.summary}\n{finding.suggested_next_step}"
        result = await self.semantic_dedupe.evaluate_candidate(
            entity_kind="proactive_finding",
            source_kind="proactive_sweep",
            text=text,
            metadata={"source": finding.source, "importance": finding.importance},
            model=self.model,
            relevant_entity_kinds=["proactive_finding", "internal_task", "do_now_action"],
        )
        return result.get("decision") != "create_new"

    def _record_created(self, finding: ProactiveFinding, item: ProactiveInboxItem) -> None:
        if self.semantic_dedupe is None:
            return
        self.semantic_dedupe.record_created(
            entity_kind="proactive_finding",
            source_kind="proactive_sweep",
            text=f"{finding.source}: {finding.title}\n{finding.summary}",
            metadata={"source": finding.source, "importance": finding.importance},
            provider_ref=item.inbox_id,
        )

    def _record_biodata_candidate(self, finding: ProactiveFinding, item: ProactiveInboxItem) -> None:
        """Feed accepted proactive findings into the existing biodata pipeline.

        Proactive sweep should not directly decide what belongs in USER_INFO.md.
        It records a pending synthetic observation; UserBioDataService later
        decides whether the signal is durable user_info, short-lived memory, or
        not worth storing.
        """
        if self.memory is None or not hasattr(self.memory, "append_visual_observation"):
            return
        now = datetime.now(timezone.utc).isoformat()
        source_label = finding.source.replace("_", " ").title()
        evidence_text = "\n".join(f"- {entry}" for entry in finding.evidence[:8])
        details = "\n\n".join(
            part for part in [
                finding.summary,
                f"Suggested next step: {finding.suggested_next_step}" if finding.suggested_next_step else "",
                f"Evidence:\n{evidence_text}" if evidence_text else "",
            ]
            if part
        )
        hypotheses: list[dict[str, str]] = []
        if finding.importance in {"high", "urgent"} or finding.requires_user_action:
            hypotheses.append(
                {
                    "category": "commitment" if finding.requires_user_action else "concern",
                    "summary": finding.suggested_next_step or finding.summary[:240],
                    "confidence": f"{finding.confidence:.2f}",
                }
            )
        raw_payload = {
            "source": "proactive_sweep",
            "proactive_source": finding.source,
            "inbox_id": item.inbox_id,
            "opportunity_id": item.opportunity_id,
            "importance": finding.importance,
            "requires_user_action": finding.requires_user_action,
            "sensitive": finding.sensitive,
            "evidence": finding.evidence,
            "suggested_next_step": finding.suggested_next_step,
        }
        observation = VisualObservation(
            observation_id=f"proactive-{item.inbox_id}",
            screenshot_path=f"proactive://{finding.source}/{item.inbox_id}",
            created_at=now,
            observation_type="proactive_finding",
            app_name=f"Proactive {source_label}",
            window_title=finding.title,
            page_hint=finding.source,
            summary=finding.title,
            detailed_description=details,
            inferred_user_activity=(
                f"Ambient AI found a {finding.importance} proactive {finding.source} signal: "
                f"{finding.summary}"
            ),
            previous_activity_status="active",
            salient_entities=[finding.source, finding.importance],
            open_loops=[finding.suggested_next_step] if finding.suggested_next_step else [],
            possible_next_task=finding.suggested_next_step or None,
            user_fact_hypotheses=hypotheses,
            confidence=finding.confidence,
            followup_sent_at=now,
            biodata_sent_at=None,
            raw_payload_json=json.dumps(raw_payload, ensure_ascii=False),
            analysis_status="proactive_sweep",
            analysis_model=self.model,
        )
        try:
            self.memory.append_visual_observation(observation)
        except Exception:
            self.logger.exception("Could not record proactive finding as biodata candidate.")

    def _create_inbox_item(self, finding: ProactiveFinding) -> ProactiveInboxItem:
        now = datetime.now(timezone.utc).isoformat()
        opportunity_id = "proactive-" + uuid.uuid4().hex
        self.autonomy_store.upsert_opportunity(
            OpportunityCandidate(
                opportunity_id=opportunity_id,
                fingerprint=self._fingerprint(finding),
                title=finding.title[:180] or f"{finding.source.title()} update",
                goal=finding.suggested_next_step or finding.summary[:240],
                rationale=finding.summary[:900],
                source_event_ids=[],
                expected_value=self._importance_score(finding.importance),
                urgency=self._importance_score(finding.importance),
                confidence=max(0.0, min(1.0, float(finding.confidence))),
                cost_of_wrong=0.25 if finding.requires_user_action else 0.15,
                personalization_benefit=0.6,
                evidence_gaps=[],
                status="surfaced",
                created_at=now,
                updated_at=now,
                metadata_json=json.dumps(
                    {"source": finding.source, "proactive": True},
                    ensure_ascii=False,
                ),
            )
        )
        detailed = "\n\n".join(
            part for part in [
                finding.summary,
                "Evidence:\n" + "\n".join(f"- {item}" for item in finding.evidence) if finding.evidence else "",
                f"Suggested next step: {finding.suggested_next_step}" if finding.suggested_next_step else "",
            ]
            if part
        )
        item = ProactiveInboxItem(
            inbox_id=uuid.uuid4().hex,
            opportunity_id=opportunity_id,
            title=finding.title[:180] or f"{finding.source.title()} update",
            summary=finding.summary[:900],
            detailed_report=detailed[:5000],
            status="pending" if finding.requires_user_action else "ready",
            confidence=max(0.0, min(1.0, float(finding.confidence))),
            why_now=f"Idle proactive {finding.source} check found a {finding.importance} item.",
            sources_json=json.dumps(
                [{"source": finding.source, "evidence": finding.evidence}],
                ensure_ascii=False,
            ),
            personalization_json=json.dumps({"proactive": True, "sensitive": finding.sensitive}),
            actions_json=json.dumps(
                [{"label": "Review", "suggested_next_step": finding.suggested_next_step}],
                ensure_ascii=False,
            ),
            created_at=now,
            updated_at=now,
        )
        return self.autonomy_store.add_inbox_item(item)

    @staticmethod
    def _fingerprint(finding: ProactiveFinding) -> str:
        import hashlib

        material = json.dumps(
            {
                "source": finding.source,
                "title": finding.title,
                "summary": finding.summary,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    @classmethod
    def _importance_score(cls, importance: str) -> float:
        return min(1.0, 0.35 + cls.IMPORTANCE_ORDER.get(importance, 1) * 0.2)

    def _parse_findings(self, text: str, *, fallback_source: str) -> list[ProactiveFinding]:
        parsed = self._parse_json(text)
        raw_findings = parsed.get("findings") if isinstance(parsed, dict) else None
        if not isinstance(raw_findings, list):
            return []
        findings: list[ProactiveFinding] = []
        for raw in raw_findings:
            if not isinstance(raw, dict):
                continue
            title = str(raw.get("title") or "").strip()
            summary = str(raw.get("summary") or "").strip()
            if not title or not summary:
                continue
            evidence = raw.get("evidence")
            findings.append(
                ProactiveFinding(
                    source=str(raw.get("source") or fallback_source).strip().lower() or fallback_source,
                    title=title,
                    summary=summary,
                    importance=self._importance(raw.get("importance")),
                    evidence=[str(item).strip() for item in evidence if str(item).strip()]
                    if isinstance(evidence, list)
                    else [],
                    suggested_next_step=str(raw.get("suggested_next_step") or "").strip(),
                    requires_user_action=bool(raw.get("requires_user_action", True)),
                    confidence=self._confidence(raw.get("confidence")),
                    sensitive=bool(raw.get("sensitive", True)),
                )
            )
        return findings

    def _personalization_context(self, source: str) -> str:
        if self.user_context_service is None:
            return ""
        try:
            return self.user_context_service.build_prompt_context(
                query_text=f"proactive {source} sweep",
                include_semantic=True,
            )[:6000]
        except Exception:
            self.logger.exception("Could not build proactive personalization context.")
            return ""

    def _today_runs(self) -> list[Any]:
        now = datetime.now().astimezone()
        start = now.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc).isoformat()
        end = (now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)).astimezone(timezone.utc).isoformat()
        rows = self.autonomy_store.list_activity_runs_between(start, end, limit=500)
        return [row for row in rows if row.source_kind == "proactive_sweep"]

    @classmethod
    def _parse_json(cls, text: str) -> dict[str, Any]:
        candidate = str(text or "").strip()
        if not candidate:
            return {}
        try:
            payload = json.loads(candidate)
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{[\s\S]*\}", candidate)
        if not match:
            return {}
        try:
            payload = json.loads(match.group(0))
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            return {}

    @classmethod
    def _importance(cls, value: Any) -> str:
        normalized = str(value or "medium").strip().lower()
        return normalized if normalized in cls.IMPORTANCE_ORDER else "medium"

    @staticmethod
    def _confidence(value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.6

    @staticmethod
    def _parse_time(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
