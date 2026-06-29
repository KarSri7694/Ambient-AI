import json
import logging
import re
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from application.ports.LLMProvider import LLMProvider
from application.ports.memory_port import MemoryPort


class SemanticDeduplicationService:
    """Use one LLM-backed decision flow to suppress duplicate created items."""

    ALLOWED_DECISIONS = {"create_new", "duplicate_skip", "duplicate_update_existing"}
    ACTIVE_REGISTRY_STATUSES = {"created", "completed"}
    ALL_ENTITY_KINDS = (
        "todoist_reminder",
        "internal_task",
        "do_now_action",
        "reflection_task",
        "calendar_event",
    )
    DEFAULT_PROMPT = """You decide whether a candidate ambient assistant item should be created or skipped as a duplicate.

Return JSON only:
{
  "decision": "create_new|duplicate_skip|duplicate_update_existing",
  "duplicate_of_item_id": "existing item id or null",
  "reason": "short explanation",
  "confidence": 0.0
}

Rules:
- The final duplicate decision must be based on semantic intent, not exact wording.
- Treat the prior items as the only candidates that can be matched.
- Use duplicate_skip when the candidate would create substantially the same reminder, task, do-now action, or future event as an existing item in scope.
- Use create_new when the candidate is related but still a genuinely distinct action, follow-up, recurrence instance, or event.
- A reminder, research task, todo, and calendar event about the same topic can still be duplicates if they serve the same concrete user intent and horizon.
- Do not merge genuinely different follow-up actions just because they mention the same topic.
- Recurring tasks are not duplicates when they represent a new occurrence or a materially different time window.
- If you are unsure, prefer create_new.
- Do not return any text outside the JSON object.
"""

    def __init__(
        self,
        *,
        memory: MemoryPort,
        llm_provider: LLMProvider,
        enabled: bool = True,
        model: str = "",
        candidate_limit: int = 8,
        default_ttl_seconds: int = 7 * 24 * 60 * 60,
        per_entity_ttl_seconds: Optional[Dict[str, int]] = None,
        debug_log_reasoning: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        self.memory = memory
        self.llm = llm_provider
        self.enabled = bool(enabled)
        self.model = str(model or "").strip()
        self.candidate_limit = max(1, int(candidate_limit))
        self.default_ttl_seconds = max(1, int(default_ttl_seconds))
        self.per_entity_ttl_seconds = {
            key: max(1, int(value))
            for key, value in (per_entity_ttl_seconds or {}).items()
        }
        self.debug_log_reasoning = bool(debug_log_reasoning)
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def ttl_seconds_for(self, entity_kind: str) -> int:
        return self.per_entity_ttl_seconds.get(entity_kind, self.default_ttl_seconds)

    async def evaluate_candidate(
        self,
        *,
        entity_kind: str,
        source_kind: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[int] = None,
        relevant_entity_kinds: Optional[List[str]] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        candidate_text = self._clean_text(text)
        ttl_value = max(1, int(ttl_seconds or self.ttl_seconds_for(entity_kind)))
        effective_model = self._clean_text(model or self.model)
        relevant_kinds = relevant_entity_kinds or list(self.ALL_ENTITY_KINDS)
        if not self.enabled or not effective_model or not candidate_text:
            return self._fallback_result(
                reason="semantic_dedupe_disabled_or_unconfigured",
                ttl_seconds=ttl_value,
            )

        recent = self.list_recent_candidates(
            entity_kinds=relevant_kinds,
            ttl_seconds=ttl_value,
            limit=max(self.candidate_limit * 4, self.candidate_limit),
        )
        shortlist = self._shortlist_candidates(candidate_text, recent)[: self.candidate_limit]
        if not shortlist:
            return self._fallback_result(reason="no_recent_candidates", ttl_seconds=ttl_value)

        payload = self._build_payload(
            entity_kind=entity_kind,
            source_kind=source_kind,
            text=candidate_text,
            metadata=metadata or {},
            ttl_seconds=ttl_value,
            prior_items=shortlist,
        )
        try:
            completion = await self.llm.chat_completion_stream(
                model=effective_model,
                messages=[
                    {"role": "system", "content": self._build_system_prompt(self.DEFAULT_PROMPT)},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
                ],
                tools=None,
            )
            parsed = self._parse_json_object(await self._consume_stream_text(completion))
        except Exception as exc:
            self.logger.warning("Semantic dedupe evaluation failed: %s", exc)
            return self._fallback_result(
                reason=f"semantic_dedupe_error:{exc.__class__.__name__}",
                ttl_seconds=ttl_value,
            )

        decision = self._clean_text(parsed.get("decision"))
        duplicate_of_item_id = self._clean_text(parsed.get("duplicate_of_item_id")) or None
        reason = self._clean_text(parsed.get("reason")) or "llm_decision"
        confidence = self._coerce_confidence(parsed.get("confidence"))
        if decision not in self.ALLOWED_DECISIONS:
            self.logger.warning("Semantic dedupe returned invalid decision payload.")
            return self._fallback_result(reason="semantic_dedupe_invalid_json", ttl_seconds=ttl_value)

        matched = None
        if duplicate_of_item_id:
            matched = next((item for item in shortlist if item.dedupe_item_id == duplicate_of_item_id), None)
        if decision != "create_new" and matched is None:
            self.logger.warning("Semantic dedupe referenced an unknown prior item; falling back to create_new.")
            return self._fallback_result(reason="semantic_dedupe_unknown_match", ttl_seconds=ttl_value)

        result = {
            "decision": "duplicate_skip" if decision == "duplicate_update_existing" else decision,
            "duplicate_of_item_id": duplicate_of_item_id,
            "reason": reason,
            "confidence": confidence,
            "ttl_seconds": ttl_value,
            "matched_item": matched,
            "candidate_count": len(shortlist),
        }
        if self.debug_log_reasoning:
            self.logger.info(
                "Semantic dedupe decision=%s entity_kind=%s match=%s reason=%s",
                result["decision"],
                entity_kind,
                duplicate_of_item_id,
                reason,
            )
        return result

    def record_created(
        self,
        *,
        entity_kind: str,
        source_kind: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[int] = None,
        provider_ref: Optional[str] = None,
        dedupe_item_id: Optional[str] = None,
    ):
        ttl_value = max(1, int(ttl_seconds or self.ttl_seconds_for(entity_kind)))
        return self.memory.add_semantic_dedupe_item(
            entity_kind=entity_kind,
            source_kind=source_kind,
            raw_text=self._clean_text(text),
            status="created",
            ttl_expires_at=(datetime.now() + timedelta(seconds=ttl_value)).isoformat(),
            provider_ref=provider_ref,
            metadata_json=self._serialize_metadata(metadata),
            dedupe_item_id=dedupe_item_id,
        )

    def record_skipped_duplicate(
        self,
        *,
        entity_kind: str,
        source_kind: str,
        text: str,
        duplicate_of_item_id: Optional[str],
        metadata: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[int] = None,
    ):
        ttl_value = max(1, int(ttl_seconds or self.ttl_seconds_for(entity_kind)))
        return self.memory.add_semantic_dedupe_item(
            entity_kind=entity_kind,
            source_kind=source_kind,
            raw_text=self._clean_text(text),
            status="duplicate_skipped",
            ttl_expires_at=(datetime.now() + timedelta(seconds=ttl_value)).isoformat(),
            duplicate_of_item_id=duplicate_of_item_id,
            metadata_json=self._serialize_metadata(metadata),
        )

    def mark_completed(self, dedupe_item_id: str):
        return self.memory.update_semantic_dedupe_item(dedupe_item_id, status="completed")

    def list_recent_candidates(
        self,
        *,
        entity_kinds: Optional[List[str]] = None,
        ttl_seconds: Optional[int] = None,
        limit: Optional[int] = None,
    ):
        ttl_value = max(1, int(ttl_seconds or self.default_ttl_seconds))
        created_after = (datetime.now() - timedelta(seconds=ttl_value)).isoformat()
        items = self.memory.list_semantic_dedupe_items(
            entity_kinds=entity_kinds,
            statuses=list(self.ACTIVE_REGISTRY_STATUSES),
            created_after=created_after,
            limit=limit or self.candidate_limit,
        )
        now_iso = datetime.now().isoformat()
        return [item for item in items if self._clean_text(item.ttl_expires_at) >= now_iso]

    def _build_payload(
        self,
        *,
        entity_kind: str,
        source_kind: str,
        text: str,
        metadata: Dict[str, Any],
        ttl_seconds: int,
        prior_items: List[Any],
    ) -> Dict[str, Any]:
        relevant_timestamps = {
            key: value
            for key, value in metadata.items()
            if "time" in key.lower() or "date" in key.lower() or key.lower().endswith("_at")
        }
        schedule_info = metadata.get("schedule_info")
        if not isinstance(schedule_info, dict):
            schedule_info = {
                key: value
                for key, value in metadata.items()
                if key in {"due_datetime", "due_date", "start_time", "end_time", "date", "time", "participants"}
            }
        return {
            "candidate": {
                "entity_kind": entity_kind,
                "source_kind": source_kind,
                "text": text,
                "relevant_timestamps": relevant_timestamps,
                "schedule_info": schedule_info,
                "metadata": metadata,
            },
            "ttl_seconds": ttl_seconds,
            "prior_items": [
                {
                    "dedupe_item_id": item.dedupe_item_id,
                    "entity_kind": item.entity_kind,
                    "source_kind": item.source_kind,
                    "text": item.raw_text,
                    "created_at": item.created_at,
                    "ttl_expires_at": item.ttl_expires_at,
                    "provider_ref": item.provider_ref,
                    "metadata": self._deserialize_metadata(item.metadata_json),
                }
                for item in prior_items
            ],
        }

    def _shortlist_candidates(self, candidate_text: str, recent_items: List[Any]) -> List[Any]:
        scored: List[tuple[float, Any]] = []
        for item in recent_items:
            score = self._retrieval_score(candidate_text, item.raw_text)
            scored.append((score, item))
        scored.sort(key=lambda pair: (pair[0], pair[1].created_at), reverse=True)
        return [item for score, item in scored if score > 0]

    def _retrieval_score(self, left: str, right: str) -> float:
        left_norm = self._normalize_text(left)
        right_norm = self._normalize_text(right)
        if not left_norm or not right_norm:
            return 0.0
        ratio = SequenceMatcher(None, left_norm, right_norm).ratio()
        left_tokens = set(left_norm.split())
        right_tokens = set(right_norm.split())
        overlap = len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))
        contains_bonus = 0.15 if left_norm in right_norm or right_norm in left_norm else 0.0
        return ratio + overlap + contains_bonus

    def _serialize_metadata(self, metadata: Optional[Dict[str, Any]]) -> str:
        try:
            return json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True)
        except TypeError:
            return json.dumps({"raw_metadata": str(metadata)}, ensure_ascii=False, sort_keys=True)

    def _deserialize_metadata(self, metadata_json: str) -> Dict[str, Any]:
        try:
            payload = json.loads(metadata_json or "{}")
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _fallback_result(self, *, reason: str, ttl_seconds: int) -> Dict[str, Any]:
        return {
            "decision": "create_new",
            "duplicate_of_item_id": None,
            "reason": reason,
            "confidence": 0.0,
            "ttl_seconds": ttl_seconds,
            "matched_item": None,
            "candidate_count": 0,
        }

    def _build_system_prompt(self, prompt: str) -> str:
        now = datetime.now()
        preamble = (
            f"Current day of week: {now.strftime('%A')}\n"
            f"Current date: {now.strftime('%Y-%m-%d')}\n"
            f"Current time: {now.strftime('%H:%M:%S')}\n\n"
        )
        return preamble + prompt

    async def _consume_stream_text(self, completion) -> str:
        parts: List[str] = []
        async for chunk in completion:
            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta
            if getattr(delta, "content", None):
                parts.append(delta.content)
        return "".join(parts)

    def _parse_json_object(self, response_text: str) -> Dict[str, Any]:
        text = response_text.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _coerce_confidence(self, value: Any) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, confidence))

    def _clean_text(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    def _normalize_text(self, value: Any) -> str:
        cleaned = self._clean_text(value).lower()
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned
