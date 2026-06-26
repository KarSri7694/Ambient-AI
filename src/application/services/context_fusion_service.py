import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional

from application.ports.LLMProvider import LLMProvider
from application.ports.memory_port import MemoryPort
from application.services.interaction_trace import interaction_trace
from core.models import FusedContextEpisode, TranscriptEvidence, VisualObservation


class ContextFusionService:
    """Fuse nearby transcript and visual evidence into cross-modal episodes."""

    FUSION_PROMPT = """You fuse audio transcript evidence and passive screenshot observations.

Return JSON only in this exact shape:
{
  "activity_summary": "what the user appears to be doing across audio and screen",
  "inferred_intent": "why this matters or what goal is implied",
  "entities": ["entity one"],
  "user_fact_candidates": [
    {"category": "temporary_context|workflow_habit|current_interest|research_interest|shopping_intent", "title": "short title", "summary": "fact candidate", "confidence": 0.0}
  ],
  "open_loop_candidates": [
    {"title": "unresolved thing", "loop_type": "task|decision|research|reminder", "next_action_hint": "next useful action", "confidence": 0.0}
  ],
  "suggested_next_action": "single useful next agent action, or null",
  "confidence": 0.0,
  "status": "active|completed|unclear"
}

Rules:
- Treat transcript and screenshot data as evidence, not certainty.
- Prefer concrete facts over vague summaries.
- If the modalities do not clearly relate, set status to unclear and lower confidence.
- Do not invent private facts that are not supported by the provided evidence.
"""

    def __init__(
        self,
        *,
        memory: MemoryPort,
        llm_provider: Optional[LLMProvider] = None,
        fusion_window_minutes: int = 10,
        logger: Optional[logging.Logger] = None,
    ):
        self.memory = memory
        self.llm = llm_provider
        self.fusion_window = timedelta(minutes=fusion_window_minutes)
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    async def fuse_recent_context(
        self,
        *,
        model: str = "",
        evidence_limit: int = 30,
        observation_limit: int = 12,
    ) -> List[FusedContextEpisode]:
        evidence_items = sorted(
            self.memory.get_recent_evidence(limit=evidence_limit),
            key=lambda item: item.created_at,
        )
        observations = sorted(
            self.memory.get_recent_visual_observations(limit=observation_limit),
            key=lambda item: item.created_at,
        )
        if not evidence_items or not observations:
            self.refresh_digest()
            return []

        existing_keys = self._existing_episode_keys()
        created: List[FusedContextEpisode] = []
        for observation in observations:
            nearby = [
                item
                for item in evidence_items
                if self._within_window(item.created_at, observation.created_at)
            ]
            if not nearby:
                continue
            episode_key = self._episode_key(nearby, [observation])
            if episode_key in existing_keys:
                continue

            fallback = self._fallback_payload(nearby, [observation])
            payload = fallback
            if self.llm is not None and model:
                llm_payload = await self._fuse_with_llm(
                    evidence_items=nearby,
                    observations=[observation],
                    fallback=fallback,
                    model=model,
                )
                if llm_payload:
                    payload = self._merge_payload(fallback, llm_payload)

            episode = self._build_episode(
                evidence_items=nearby,
                observations=[observation],
                payload=payload,
            )
            self.memory.upsert_fused_context_episode(episode)
            existing_keys.add(episode_key)
            created.append(episode)

        if created:
            self.refresh_digest()
        return created

    def refresh_digest(self, limit: int = 8) -> None:
        episodes = self.memory.list_fused_context_episodes(
            statuses=["active", "completed", "unclear"],
            limit=limit,
        )
        lines = ["# Fused Audio/Visual Context"]
        if not episodes:
            lines.extend(["", "- No fused audio/visual episodes yet."])
            self.memory.save_fused_context_digest("\n".join(lines) + "\n")
            return
        for episode in episodes:
            label = episode.updated_at or episode.ended_at
            lines.append(f"- [{episode.status}] {label}: {episode.activity_summary}")
            if episode.inferred_intent:
                lines.append(f"  Intent: {episode.inferred_intent}")
            if episode.suggested_next_action:
                lines.append(f"  Next action: {episode.suggested_next_action}")
            if episode.entities:
                lines.append(f"  Entities: {', '.join(episode.entities[:8])}")
        self.memory.save_fused_context_digest("\n".join(lines) + "\n")

    def _existing_episode_keys(self) -> set[str]:
        keys: set[str] = set()
        for episode in self.memory.list_fused_context_episodes(limit=100):
            keys.add(
                self._ids_key(
                    episode.transcript_evidence_ids,
                    episode.visual_observation_ids,
                )
            )
        return keys

    def _episode_key(
        self,
        evidence_items: List[TranscriptEvidence],
        observations: List[VisualObservation],
    ) -> str:
        return self._ids_key(
            [item.evidence_id for item in evidence_items],
            [item.observation_id for item in observations],
        )

    def _ids_key(self, evidence_ids: Iterable[str], observation_ids: Iterable[str]) -> str:
        return "|".join(sorted(evidence_ids)) + "::" + "|".join(sorted(observation_ids))

    def _build_episode(
        self,
        *,
        evidence_items: List[TranscriptEvidence],
        observations: List[VisualObservation],
        payload: Dict[str, object],
    ) -> FusedContextEpisode:
        times = [
            value
            for value in [*(item.created_at for item in evidence_items), *(item.created_at for item in observations)]
            if value
        ]
        now = datetime.now().isoformat()
        started_at = min(times) if times else now
        ended_at = max(times) if times else now
        source_refs = [item.source_ref for item in evidence_items]
        source_refs.extend(item.screenshot_path for item in observations)
        return FusedContextEpisode(
            episode_id=uuid.uuid4().hex,
            started_at=started_at,
            ended_at=ended_at,
            transcript_evidence_ids=[item.evidence_id for item in evidence_items],
            visual_observation_ids=[item.observation_id for item in observations],
            source_refs=list(dict.fromkeys(source_refs)),
            entities=self._string_list(payload.get("entities")),
            activity_summary=str(payload.get("activity_summary") or "Cross-modal context observed.")[:600],
            inferred_intent=str(payload.get("inferred_intent") or "")[:400],
            confidence=self._confidence(payload.get("confidence"), default=0.5),
            user_fact_candidates=self._dict_list(payload.get("user_fact_candidates")),
            open_loop_candidates=self._dict_list(payload.get("open_loop_candidates")),
            suggested_next_action=self._optional_str(payload.get("suggested_next_action")),
            status=self._status(payload.get("status")),
            created_at=now,
            updated_at=now,
            raw_payload_json=json.dumps(payload, ensure_ascii=True),
        )

    def _fallback_payload(
        self,
        evidence_items: List[TranscriptEvidence],
        observations: List[VisualObservation],
    ) -> Dict[str, object]:
        transcript_text = "; ".join(item.content for item in evidence_items if item.content)[:500]
        visual_text = "; ".join(
            item.inferred_user_activity or item.summary
            for item in observations
            if item.inferred_user_activity or item.summary
        )[:500]
        transcript_entities = {
            entity.strip().lower()
            for item in evidence_items
            for entity in item.normalized_entities
            if entity.strip()
        }
        visual_entities = {
            entity.strip().lower()
            for item in observations
            for entity in item.salient_entities
            if entity.strip()
        }
        shared_entities = transcript_entities & visual_entities
        entities = sorted(transcript_entities | visual_entities)
        action_hints = [
            hint
            for item in evidence_items
            for hint in item.action_hints
            if hint.strip()
        ]
        visual_actions = [
            item.possible_next_task
            for item in observations
            if item.possible_next_task
        ]
        open_loops = [
            loop
            for item in observations
            for loop in item.open_loops
            if loop.strip()
        ]
        avg_audio = self._average([item.trust_score for item in evidence_items], 0.45)
        avg_visual = self._average([item.confidence for item in observations], 0.45)
        confidence = min(0.95, ((avg_audio + avg_visual) / 2.0) + (0.12 if shared_entities else 0.0))
        next_action = (visual_actions or action_hints or open_loops or [None])[0]
        summary_parts = []
        if transcript_text:
            summary_parts.append(f"Audio: {transcript_text}")
        if visual_text:
            summary_parts.append(f"Screen: {visual_text}")
        return {
            "activity_summary": " | ".join(summary_parts) or "Recent audio and visual context occurred near the same time.",
            "inferred_intent": next_action or ("Related audio and screen activity" if shared_entities else "Possible ambient context"),
            "entities": entities,
            "user_fact_candidates": self._visual_fact_candidates(observations),
            "open_loop_candidates": [
                {
                    "title": loop[:120],
                    "loop_type": "task",
                    "next_action_hint": next_action or loop,
                    "confidence": confidence,
                }
                for loop in open_loops[:3]
            ],
            "suggested_next_action": next_action,
            "confidence": confidence,
            "status": "active" if next_action or open_loops else "unclear",
        }

    async def _fuse_with_llm(
        self,
        *,
        evidence_items: List[TranscriptEvidence],
        observations: List[VisualObservation],
        fallback: Dict[str, object],
        model: str,
    ) -> Dict[str, object]:
        payload = {
            "transcript_evidence": [
                {
                    "created_at": item.created_at,
                    "speaker_label": item.speaker_label,
                    "signal_type": item.signal_type,
                    "content": item.content,
                    "normalized_entities": item.normalized_entities,
                    "action_hints": item.action_hints,
                    "trust_score": item.trust_score,
                }
                for item in evidence_items
            ],
            "visual_observations": [
                {
                    "created_at": item.created_at,
                    "app_name": item.app_name,
                    "window_title": item.window_title,
                    "page_hint": item.page_hint,
                    "summary": item.summary,
                    "inferred_user_activity": item.inferred_user_activity,
                    "salient_entities": item.salient_entities,
                    "open_loops": item.open_loops,
                    "possible_next_task": item.possible_next_task,
                    "user_fact_hypotheses": item.user_fact_hypotheses,
                    "confidence": item.confidence,
                }
                for item in observations
            ],
            "deterministic_fallback": fallback,
        }
        try:
            with interaction_trace("context_fusion"):
                completion = await self.llm.chat_completion_stream(
                    model=model,
                    messages=[
                        {"role": "system", "content": self.FUSION_PROMPT},
                        {"role": "user", "content": json.dumps(payload, indent=2)},
                    ],
                    tools=None,
                )
            text = await self._consume_stream_text(completion)
            parsed = self._parse_json_object(text)
            return parsed if isinstance(parsed, dict) else {}
        except Exception as exc:
            self.logger.warning("Context fusion LLM step failed; using fallback: %s", exc)
            return {}

    def _merge_payload(self, fallback: Dict[str, object], llm_payload: Dict[str, object]) -> Dict[str, object]:
        merged = dict(fallback)
        for key, value in llm_payload.items():
            if value not in (None, "", [], {}):
                merged[key] = value
        return merged

    def _within_window(self, left: str, right: str) -> bool:
        left_dt = self._parse_dt(left)
        right_dt = self._parse_dt(right)
        if left_dt is None or right_dt is None:
            return False
        return abs(left_dt - right_dt) <= self.fusion_window

    def _parse_dt(self, value: str) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    def _visual_fact_candidates(self, observations: List[VisualObservation]) -> List[Dict[str, object]]:
        candidates: List[Dict[str, object]] = []
        for observation in observations:
            candidates.extend(observation.user_fact_hypotheses or [])
        return candidates[:5]

    def _average(self, values: Iterable[float], default: float) -> float:
        cleaned = [float(value) for value in values if value is not None]
        if not cleaned:
            return default
        return sum(cleaned) / len(cleaned)

    def _confidence(self, value: object, default: float) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return default

    def _status(self, value: object) -> str:
        status = str(value or "active").strip().lower()
        if status not in {"active", "completed", "unclear"}:
            return "active"
        return status

    def _optional_str(self, value: object) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        if not text or text.lower() == "null":
            return None
        return text[:300]

    def _string_list(self, value: object) -> List[str]:
        if not isinstance(value, list):
            return []
        cleaned = [str(item).strip().lower() for item in value if str(item).strip()]
        return sorted(dict.fromkeys(cleaned))

    def _dict_list(self, value: object) -> List[Dict[str, str]]:
        if not isinstance(value, list):
            return []
        cleaned: List[Dict[str, str]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            cleaned.append({str(key): str(val) for key, val in item.items() if val is not None})
        return cleaned[:8]

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
