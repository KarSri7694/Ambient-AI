import json
import logging
import re
import uuid
from dataclasses import replace
from datetime import datetime
from typing import Dict, Iterable, List

from application.ports.memory_port import MemoryPort
from core.models import VisualObservation, VisualUserFact


class VisualUserFactService:
    """Promote tentative visual user facts into durable user info."""

    MANAGED_START = "<!-- AMBIENT_VISUAL_FACTS_START -->"
    MANAGED_END = "<!-- AMBIENT_VISUAL_FACTS_END -->"
    STRENGTH_WEIGHTS = {
        "weak": 0.65,
        "medium": 1.0,
        "strong": 1.3,
    }
    DURABLE_CATEGORIES = {
        "device_interest": {"score": 2.6, "sessions": 2},
        "entertainment_preference": {"score": 2.6, "sessions": 2},
        "workflow_habit": {"score": 2.8, "sessions": 2},
        "current_interest": {"score": 3.0, "sessions": 3},
        "shopping_intent": {"score": 3.1, "sessions": 3},
        "research_interest": {"score": 3.1, "sessions": 3},
    }
    EMERGING_SCORE = 1.0
    TEMPORARY_CONTEXT_CATEGORIES = {"temporary_context"}
    SECTION_TITLES = (
        ("device_interest", "Current Shopping / Research Intent"),
        ("shopping_intent", "Current Shopping / Research Intent"),
        ("research_interest", "Current Shopping / Research Intent"),
        ("current_interest", "Current Interests"),
        ("entertainment_preference", "Durable Preferences"),
        ("workflow_habit", "Workflow Habits"),
        ("temporary_context", "Temporary Context"),
    )

    def __init__(self, memory: MemoryPort, logger: logging.Logger | None = None):
        self.memory = memory
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def update_from_observation(self, observation: VisualObservation) -> List[VisualUserFact]:
        raw_facts = observation.user_fact_hypotheses or self._extract_hypotheses_from_payload(observation.raw_payload_json)
        if not raw_facts:
            self.refresh_user_info()
            return []

        updated: List[VisualUserFact] = []
        for raw_fact in raw_facts:
            normalized = self._normalize_hypothesis(raw_fact)
            if normalized is None:
                continue
            fact_key = self._fact_key(normalized["category"], normalized["title"])
            existing = self.memory.get_visual_user_fact(fact_key)
            updated_fact = self._merge_fact(existing=existing, normalized=normalized, observation=observation)
            self.memory.upsert_visual_user_fact(updated_fact)
            updated.append(updated_fact)

        self.refresh_user_info()
        return updated

    def refresh_user_info(self) -> None:
        facts = self.memory.list_visual_user_facts(statuses=["durable", "emerging"], limit=200)
        grouped: Dict[str, List[VisualUserFact]] = {}
        for category, _ in self.SECTION_TITLES:
            grouped[category] = []
        for fact in facts:
            grouped.setdefault(fact.category, []).append(fact)

        lines = [
            self.MANAGED_START,
            "# USER_INFO",
            "",
            "## Durable Preferences",
        ]
        self._append_section_entries(lines, grouped, {"entertainment_preference"}, {"durable"})
        lines.extend(["", "## Current Interests"])
        self._append_section_entries(lines, grouped, {"current_interest"}, {"durable", "emerging"})
        lines.extend(["", "## Current Shopping / Research Intent"])
        self._append_section_entries(lines, grouped, {"device_interest", "shopping_intent", "research_interest"}, {"durable", "emerging"})
        lines.extend(["", "## Workflow Habits"])
        self._append_section_entries(lines, grouped, {"workflow_habit"}, {"durable", "emerging"})
        lines.extend(["", "## Temporary Context"])
        self._append_section_entries(lines, grouped, {"temporary_context"}, {"emerging"})
        lines.extend([self.MANAGED_END, ""])

        managed_block = "\n".join(lines)
        existing = self.memory.get_user_info()
        self.memory.save_user_info(self._merge_managed_block(existing, managed_block))

    def _append_section_entries(
        self,
        lines: List[str],
        grouped: Dict[str, List[VisualUserFact]],
        categories: set[str],
        allowed_statuses: set[str],
    ) -> None:
        entries: List[str] = []
        for category in categories:
            for fact in grouped.get(category, []):
                if fact.status not in allowed_statuses:
                    continue
                label = "durable" if fact.status == "durable" else "emerging"
                entries.append(f"- [{label}] {fact.summary}")
        lines.extend(entries or ["- None yet."])

    def _merge_fact(
        self,
        *,
        existing: VisualUserFact | None,
        normalized: dict,
        observation: VisualObservation,
    ) -> VisualUserFact:
        now = observation.created_at or datetime.now().isoformat()
        confidence = self._num(normalized.get("confidence"), observation.confidence or 0.5)
        strength_weight = self.STRENGTH_WEIGHTS.get(str(normalized.get("evidence_strength", "medium")).strip().lower(), 1.0)
        score_increment = confidence * strength_weight

        if existing is None:
            score = score_increment
            session_ids = [observation.session_id] if observation.session_id else []
            fact = VisualUserFact(
                fact_id=uuid.uuid4().hex,
                fact_key=self._fact_key(normalized["category"], normalized["title"]),
                category=normalized["category"],
                title=normalized["title"],
                summary=normalized["summary"],
                status="temporary",
                score=score,
                observation_count=1,
                session_count=1 if observation.session_id else 0,
                first_seen_at=now,
                last_seen_at=now,
                source_observation_ids=[observation.observation_id],
                source_session_ids=session_ids,
            )
            return replace(fact, status=self._derive_status(fact))

        new_session = bool(observation.session_id and observation.session_id not in existing.source_session_ids)
        weighted_increment = score_increment if new_session else score_increment * 0.35
        score = self._decay_score(existing, now) + weighted_increment
        merged = VisualUserFact(
            fact_id=existing.fact_id,
            fact_key=existing.fact_key,
            category=existing.category,
            title=existing.title,
            summary=self._prefer_summary(existing.summary, normalized["summary"]),
            status=existing.status,
            score=score,
            observation_count=existing.observation_count + 1,
            session_count=existing.session_count + (1 if new_session else 0),
            first_seen_at=existing.first_seen_at,
            last_seen_at=now,
            source_observation_ids=self._append_unique(existing.source_observation_ids, observation.observation_id),
            source_session_ids=self._append_unique(existing.source_session_ids, observation.session_id) if observation.session_id else list(existing.source_session_ids),
        )
        return replace(merged, status=self._derive_status(merged))

    def _derive_status(self, fact: VisualUserFact) -> str:
        if fact.category in self.TEMPORARY_CONTEXT_CATEGORIES:
            return "emerging" if fact.score >= self.EMERGING_SCORE or fact.observation_count >= 2 else "temporary"
        durable_rule = self.DURABLE_CATEGORIES.get(
            fact.category,
            {"score": 3.0, "sessions": 3},
        )
        if fact.score >= durable_rule["score"] and fact.session_count >= durable_rule["sessions"]:
            return "durable"
        if fact.score >= self.EMERGING_SCORE or fact.observation_count >= 2:
            return "emerging"
        return "temporary"

    def _extract_hypotheses_from_payload(self, raw_payload_json: str | None) -> List[dict]:
        if not raw_payload_json:
            return []
        try:
            parsed = json.loads(raw_payload_json)
        except json.JSONDecodeError:
            return []
        hypotheses = parsed.get("user_fact_hypotheses", [])
        return hypotheses if isinstance(hypotheses, list) else []

    def _normalize_hypothesis(self, raw_fact: dict) -> dict | None:
        if not isinstance(raw_fact, dict):
            return None
        category = str(raw_fact.get("category", "")).strip().lower()
        title = self._normalize_title(str(raw_fact.get("title", "")).strip())
        summary = str(raw_fact.get("summary", "")).strip()
        if not category or not title or not summary:
            return None
        return {
            "category": category,
            "title": title,
            "summary": summary,
            "confidence": raw_fact.get("confidence", 0.5),
            "scope": str(raw_fact.get("scope", "temporary")).strip().lower(),
            "evidence_strength": str(raw_fact.get("evidence_strength", "medium")).strip().lower(),
        }

    def _normalize_title(self, value: str) -> str:
        return re.sub(r"\s+", " ", value.strip().lower())

    def _fact_key(self, category: str, title: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", "-", f"{category}:{title}".lower()).strip("-")
        return normalized or f"{category}-fact"

    def _prefer_summary(self, existing: str, new: str) -> str:
        if len(new) > len(existing):
            return new
        return existing

    def _append_unique(self, values: Iterable[str], new_value: str | None) -> List[str]:
        result = list(values)
        if new_value and new_value not in result:
            result.append(new_value)
        return result

    def _decay_score(self, fact: VisualUserFact, now: str) -> float:
        try:
            last_seen = datetime.fromisoformat(fact.last_seen_at)
            current = datetime.fromisoformat(now)
        except ValueError:
            return fact.score
        days = max((current - last_seen).days, 0)
        if days <= 3:
            return fact.score
        if fact.status == "durable":
            return fact.score
        decay = 0.92 ** (days - 3)
        return max(fact.score * decay, 0.0)

    def _merge_managed_block(self, existing: str, managed_block: str) -> str:
        existing = existing or ""
        if self.MANAGED_START in existing and self.MANAGED_END in existing:
            pattern = re.compile(
                re.escape(self.MANAGED_START) + r"[\s\S]*?" + re.escape(self.MANAGED_END),
                re.MULTILINE,
            )
            return pattern.sub(managed_block.strip(), existing).strip() + "\n"
        if existing.strip():
            return existing.rstrip() + "\n\n" + managed_block + "\n"
        return managed_block + "\n"

    def _num(self, value, fallback: float) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return fallback
