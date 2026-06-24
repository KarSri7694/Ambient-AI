import json
import logging
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Dict, Iterable, List

from application.ports.LLMProvider import LLMProvider
from application.ports.memory_port import MemoryPort
from core.models import SpeakerRecord, TranscriptEvidence, UserProfileFacet


class UserProfileService:
    """Grow richer user state from repeated transcript evidence."""

    CATEGORY_MAP = {
        "relationship": "important_people",
        "commitment": "current_obligations",
        "follow_up_request": "current_obligations",
        "self_reminder": "current_obligations",
        "task_in_progress": "active_projects",
        "pending_decision": "active_projects",
        "curiosity_to_revisit": "recurring_friction",
        "preference": "preferences",
    }

    PROFILE_PROMPT = """You derive user-profile facets from ambient transcript evidence.

Return JSON only:
{
  "facets": [
    {
      "evidence_id": "evidence id",
      "category": "important_people|current_obligations|active_projects|routines|preferences|recurring_friction|none",
      "title": "short dedupe title",
      "summary": "one short user-profile statement",
      "confidence": 0.0
    }
  ]
}

Rules:
- Only create facets for the user speaker.
- Hinglish is expected.
- Use none when the evidence should not update the durable user profile.
- important_people should capture recurring named people relevant to the user.
- current_obligations should capture active commitments and follow-ups.
"""

    def __init__(self, memory: MemoryPort, llm_provider: LLMProvider | None = None, logger: logging.Logger | None = None):
        self.memory = memory
        self.llm = llm_provider
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    async def update_from_evidence(
        self,
        *,
        user_speaker: SpeakerRecord | None,
        evidence_items: List[TranscriptEvidence],
        model: str = "",
    ) -> List[UserProfileFacet]:
        if user_speaker is None:
            return []
        llm_facets = await self._llm_facets(user_speaker=user_speaker, evidence_items=evidence_items, model=model)
        llm_by_evidence = {facet["evidence_id"]: facet for facet in llm_facets if isinstance(facet, dict)}
        updated: List[UserProfileFacet] = []
        for item in evidence_items:
            if item.speaker_id != user_speaker.speaker_id:
                continue
            llm_choice = llm_by_evidence.get(item.evidence_id, {})
            category = str(llm_choice.get("category") or self.CATEGORY_MAP.get(item.signal_type) or "").strip()
            if category is None or item.trust_score < 0.45:
                continue
            if category == "none" or not category:
                continue
            title = str(llm_choice.get("title") or self._title(item)).strip()
            existing = self._find_existing(user_speaker.speaker_id, category, title)
            if existing is None:
                facet = UserProfileFacet(
                    facet_id=uuid.uuid4().hex,
                    speaker_id=user_speaker.speaker_id,
                    category=category,
                    title=title,
                    summary=str(llm_choice.get("summary") or item.content).strip(),
                    confidence=self._num(llm_choice.get("confidence"), item.trust_score),
                    strength=1,
                    status="durable" if self._num(llm_choice.get("confidence"), item.trust_score) >= 0.72 else "tentative",
                    source_event_ids=[item.evidence_id],
                    updated_at=item.created_at,
                )
            else:
                facet = UserProfileFacet(
                    facet_id=existing.facet_id,
                    speaker_id=existing.speaker_id,
                    category=existing.category,
                    title=existing.title,
                    summary=existing.summary if len(existing.summary) >= len(str(llm_choice.get("summary") or item.content)) else str(llm_choice.get("summary") or item.content),
                    confidence=max(existing.confidence, self._num(llm_choice.get("confidence"), item.trust_score)),
                    strength=existing.strength + 1,
                    status="durable" if existing.strength + 1 >= 2 or self._num(llm_choice.get("confidence"), item.trust_score) >= 0.72 else existing.status,
                    source_event_ids=self._append_unique(existing.source_event_ids, item.evidence_id),
                    updated_at=item.created_at,
                )
            self.memory.upsert_profile_facet(facet)
            updated.append(facet)

        self.refresh_user_profile(user_speaker.speaker_id)
        return updated

    def refresh_user_profile(self, speaker_id: str) -> None:
        speaker = self.memory.get_speaker(speaker_id)
        if speaker is None:
            return
        facets = self.memory.get_profile_facets(
            speaker_id,
            statuses=["durable", "tentative"],
            limit=100,
        )
        by_category: Dict[str, List[UserProfileFacet]] = defaultdict(list)
        for facet in facets:
            by_category[facet.category].append(facet)

        lines = [f"# {speaker.display_name}", "", "## Ambient User Profile"]
        for category, title in (
            ("important_people", "Important People"),
            ("current_obligations", "Current Obligations"),
            ("active_projects", "Active Projects"),
            ("routines", "Routines and Habits"),
            ("preferences", "Preferences"),
            ("recurring_friction", "Recurring Friction"),
        ):
            lines.extend(["", f"### {title}"])
            entries = by_category.get(category, [])
            if not entries:
                lines.append("- None yet.")
                continue
            for entry in entries[:6]:
                strength_label = "durable" if entry.status == "durable" else "tentative"
                lines.append(f"- [{strength_label}] {entry.summary}")

        self.memory.save_speaker_profile(speaker_id, "\n".join(lines) + "\n")

    def infer_user_speaker(self) -> SpeakerRecord | None:
        speakers = self.memory.list_speakers()
        for speaker in speakers:
            if speaker.is_user:
                return speaker
        for speaker in speakers:
            if speaker.display_name.strip().upper() == "USER" or speaker.source_label.strip().upper() == "USER":
                return speaker
        return speakers[0] if speakers else None

    def _find_existing(self, speaker_id: str, category: str, title: str) -> UserProfileFacet | None:
        facets = self.memory.get_profile_facets(
            speaker_id,
            categories=[category],
            statuses=["durable", "tentative"],
            limit=20,
        )
        normalized_title = self._normalize(title)
        for facet in facets:
            if self._normalize(facet.title) == normalized_title:
                return facet
        return None

    def _title(self, item: TranscriptEvidence) -> str:
        if item.normalized_entities:
            return item.normalized_entities[0]
        lowered = item.content.lower()
        for token in item.action_hints:
            if token in lowered:
                return token
        return " ".join(item.content.strip().split()[:6]).lower()

    def _append_unique(self, values: Iterable[str], new_value: str) -> List[str]:
        result = list(values)
        if new_value not in result:
            result.append(new_value)
        return result

    def _normalize(self, text: str) -> str:
        return " ".join(text.lower().split())

    async def _llm_facets(
        self,
        *,
        user_speaker: SpeakerRecord,
        evidence_items: List[TranscriptEvidence],
        model: str,
    ) -> List[dict]:
        if self.llm is None or not model:
            return []
        relevant = [item for item in evidence_items if item.speaker_id == user_speaker.speaker_id]
        if not relevant:
            return []
        try:
            payload = {
                "user_speaker": {
                    "speaker_id": user_speaker.speaker_id,
                    "display_name": user_speaker.display_name,
                },
                "evidence": [
                    {
                        "evidence_id": item.evidence_id,
                        "signal_type": item.signal_type,
                        "content": item.content,
                        "normalized_entities": item.normalized_entities,
                        "action_hints": item.action_hints,
                        "trust_score": item.trust_score,
                    }
                    for item in relevant
                ],
            }
            completion = await self.llm.chat_completion_stream(
                model=model,
                messages=[
                    {"role": "system", "content": self.PROFILE_PROMPT},
                    {"role": "user", "content": json.dumps(payload, indent=2)},
                ],
                tools=None,
                temperature=0.1,
                top_p=0.9,
            )
            text = await self._consume_stream_text(completion)
            parsed = self._parse_json_object(text)
            facets = parsed.get("facets", [])
            return facets if isinstance(facets, list) else []
        except Exception as exc:
            self.logger.warning("User-profile LLM decision failed, using fallback mapping: %s", exc)
            return []

    async def _consume_stream_text(self, completion) -> str:
        parts: List[str] = []
        async for chunk in completion:
            delta = chunk.choices[0].delta
            if delta.content:
                parts.append(delta.content)
        return "".join(parts)

    def _parse_json_object(self, response_text: str) -> dict:
        text = response_text.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}

    def _num(self, value, fallback: float) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return fallback
