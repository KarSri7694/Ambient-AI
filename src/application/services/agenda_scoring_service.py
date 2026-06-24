from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional

from application.ports.ambient_agenda_port import AmbientAgendaPort
from core.models import AmbientAgendaItem, MemoryFact, Notification, NightTask, ProactiveTopicCandidate


@dataclass(frozen=True)
class ReflectionCandidate:
    """A scored candidate passed into the bounded reflection step."""
    candidate_id: str
    candidate_type: str
    title: str
    score: float
    source_type: str
    source_ref: str
    proposed_kind: str
    summary: str
    due_at: Optional[str] = None
    backing_topic_id: Optional[str] = None
    backing_memory_ids: List[str] = field(default_factory=list)
    agenda_id: Optional[str] = None
    existing_status: Optional[str] = None


class AgendaScoringService:
    """Build and deterministically rank ambient follow-through candidates."""

    def __init__(self, agenda: AmbientAgendaPort):
        self.agenda = agenda

    def build_top_candidates(
        self,
        *,
        facts: List[MemoryFact],
        topics: List[ProactiveTopicCandidate],
        open_agenda_items: List[AmbientAgendaItem],
        pending_tasks: List[NightTask],
        unread_notifications: List[Notification],
        limit: int = 8,
    ) -> List[ReflectionCandidate]:
        candidates: List[ReflectionCandidate] = []
        candidates.extend(self._agenda_candidates(open_agenda_items))
        candidates.extend(self._topic_candidates(topics))
        candidates.extend(self._commitment_candidates(facts, pending_tasks))
        candidates.extend(self._notification_candidates(unread_notifications))
        ranked = sorted(candidates, key=lambda candidate: candidate.score, reverse=True)
        return ranked[:limit]

    def _agenda_candidates(self, items: List[AmbientAgendaItem]) -> List[ReflectionCandidate]:
        now = datetime.now()
        ranked: List[ReflectionCandidate] = []
        for item in items:
            score = item.priority_score
            if item.status == "open":
                score += 0.15
            if item.status == "surfaced":
                score -= 0.35
            if item.due_at:
                due_at = self._parse_dt(item.due_at)
                if due_at is not None:
                    if due_at <= now + timedelta(days=1):
                        score += 0.25
                    if due_at < now:
                        score += 0.2
            ranked.append(
                ReflectionCandidate(
                    candidate_id=f"agenda:{item.agenda_id}",
                    candidate_type="agenda_item",
                    title=item.title,
                    score=max(score, 0.0),
                    source_type=item.source_type,
                    source_ref=item.source_ref,
                    proposed_kind=item.kind,
                    summary=item.surface_message or item.title,
                    due_at=item.due_at,
                    backing_topic_id=item.backing_topic_id,
                    backing_memory_ids=item.backing_memory_ids,
                    agenda_id=item.agenda_id,
                    existing_status=item.status,
                )
            )
        return ranked

    def _topic_candidates(self, topics: List[ProactiveTopicCandidate]) -> List[ReflectionCandidate]:
        candidates: List[ReflectionCandidate] = []
        for topic in topics:
            existing = self.agenda.find_by_source(
                source_type="proactive_topic",
                source_ref=topic.topic_id,
                kind="research_review",
                statuses=["open", "surfaced", "completed"],
            )
            if topic.status == "completed":
                if existing is not None:
                    continue
                score = 0.7 + min(topic.salience_score, 1.0) * 0.25
                summary = f"Researched {topic.display_title} and saved a package in your research vault."
                candidates.append(
                    ReflectionCandidate(
                        candidate_id=f"topic:{topic.topic_id}",
                        candidate_type="new_research_package",
                        title=f"Review research package: {topic.display_title}",
                        score=score,
                        source_type="proactive_topic",
                        source_ref=topic.topic_id,
                        proposed_kind="research_review",
                        summary=summary,
                        backing_topic_id=topic.topic_id,
                    )
                )
            elif topic.status == "pending":
                score = 0.4 + min(topic.salience_score, 1.0) * 0.2
                candidates.append(
                    ReflectionCandidate(
                        candidate_id=f"topic-pending:{topic.topic_id}",
                        candidate_type="pending_topic",
                        title=f"Pending practical watch: {topic.display_title}",
                        score=score,
                        source_type="proactive_topic",
                        source_ref=topic.topic_id,
                        proposed_kind="practical_watch",
                        summary=f"Pending proactive topic still looks relevant: {topic.display_title}.",
                        backing_topic_id=topic.topic_id,
                    )
                )
        return candidates

    def _commitment_candidates(
        self,
        facts: List[MemoryFact],
        pending_tasks: List[NightTask],
    ) -> List[ReflectionCandidate]:
        pending_text = " ".join(task.description.lower() for task in pending_tasks)
        candidates: List[ReflectionCandidate] = []
        now = datetime.now()
        for fact in facts:
            normalized = fact.fact_text.lower()
            if fact.topic not in {"commitment", "schedule"}:
                continue
            existing = self.agenda.find_by_source(
                source_type="memory_fact",
                source_ref=fact.fact_id,
                kind="stale_commitment",
                statuses=["open", "surfaced", "completed"],
            )
            if existing is not None:
                continue
            score = 0.55
            age = now - self._parse_dt(fact.valid_from, default=now)
            if age >= timedelta(hours=12):
                score += 0.2
            if age >= timedelta(days=1):
                score += 0.1
            if "tomorrow" in normalized or "today" in normalized or "next week" in normalized:
                score += 0.15
            if any(token in normalized for token in ("send", "check", "watch", "read", "remind", "call")):
                score += 0.1
            if normalized and normalized in pending_text:
                score -= 0.25
            title = fact.fact_text[:90]
            candidates.append(
                ReflectionCandidate(
                    candidate_id=f"fact:{fact.fact_id}",
                    candidate_type="stale_commitment",
                    title=title,
                    score=max(score, 0.0),
                    source_type="memory_fact",
                    source_ref=fact.fact_id,
                    proposed_kind="stale_commitment",
                    summary=f"You committed to: {fact.fact_text}",
                    backing_memory_ids=[fact.fact_id],
                )
            )
        return candidates

    def _notification_candidates(self, unread_notifications: List[Notification]) -> List[ReflectionCandidate]:
        candidates: List[ReflectionCandidate] = []
        for note in unread_notifications:
            if note.source in {"proactive_research", "ambient_digest"}:
                continue
            score = 0.2
            candidates.append(
                ReflectionCandidate(
                    candidate_id=f"notification:{note.id}",
                    candidate_type="notification",
                    title=note.message[:90],
                    score=score,
                    source_type="notification",
                    source_ref=str(note.id),
                    proposed_kind="follow_up",
                    summary=note.message,
                )
            )
        return candidates

    def _parse_dt(self, value: Optional[str], default: Optional[datetime] = None) -> Optional[datetime]:
        if not value:
            return default
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return default
