import json
import logging
import hashlib
from datetime import datetime
from typing import List

from application.ports.LLMProvider import LLMProvider
from application.ports.memory_port import MemoryPort
from application.services.semantic_memory_service import SemanticMemoryService
from application.services.interaction_trace import interaction_trace
from core.models import VisualObservation


class UserBioDataService:
    """Split recent passive observations into temporary memory and durable user info."""

    OBSERVATION_LIMIT = 20

    BIODATA_PROMPT = """You build append-only user biodata notes from recent passive observations.

Return JSON only:
{
  "entries": [
    {
      "note": "short durable biodata note about the user",
      "bucket": "memory|user_info",
      "category": "interest|habit|education|work|concern|preference|commitment|reminder",
      "confidence": 0.0
    }
  ]
}

Rules:
- memory means short-lived or currently relevant context such as upcoming events, pending reminders, active concerns, temporary plans, or recent ongoing situations.
- user_info means durable user profile information such as repeated habits, long-term interests, stable preferences, enduring commitments, recurring concerns, work context, or education context that is likely to remain useful over time.
- Extract only notes that are useful later.
- Ignore trivial UI actions, one-off mechanical interactions, and generic screen summaries.
- Do not restate the screenshot. Infer concise profile-style notes.
- Notes must be short, factual, and useful later.
- Use user_info only when the note looks stable enough to be part of a long-term profile.
- Use memory for one-off or short-horizon reminders and temporary context.
- If nothing is biodata-worthy, return an empty entries list.
- Do not add any keys other than entries, and each entry may only contain note, bucket, category, confidence.
"""

    def __init__(
        self,
        *,
        memory: MemoryPort,
        llm_provider: LLMProvider,
        semantic_memory: SemanticMemoryService | None = None,
        logger: logging.Logger | None = None,
    ):
        self.memory = memory
        self.llm = llm_provider
        self.semantic_memory = semantic_memory
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def _build_system_prompt(self, prompt: str) -> str:
        now = datetime.now()
        preamble = (
            f"Current day of week: {now.strftime('%A')}\n"
            f"Current date: {now.strftime('%Y-%m-%d')}\n"
            f"Current time: {now.strftime('%H:%M:%S')}\n\n"
        )
        return preamble + prompt

    async def update_biodata(self, *, model: str) -> dict:
        observations = self.memory.get_recent_biodata_pending_visual_observations(limit=self.OBSERVATION_LIMIT)
        candidates = self._candidate_rows(observations)
        if not candidates:
            return {"processed_observation_ids": [], "entries": [], "reason": "no biodata-pending observations"}

        payload = {
            "observations": [
                {
                    "observation_id": row["observation_id"],
                    "created_at": row["created_at"],
                    "app_name": row["app_name"],
                    "page_hint": row["page_hint"],
                    "summary": row["summary"],
                    "detailed_description": row["detailed_description"],
                    "inferred_user_activity": row["inferred_user_activity"],
                    "reminder_context": row["reminder_context"],
                }
                for row in candidates
            ],
            "existing_user_info": self.memory.get_user_info(),
            "existing_memory": self.memory.get_working_memory(),
            "semantic_context": self._semantic_context(candidates),
        }
        with interaction_trace("user_biodata"):
            completion = await self.llm.chat_completion_stream(
                model=model,
                messages=[
                    {"role": "system", "content": self._build_system_prompt(self.BIODATA_PROMPT)},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
                ],
                tools=None,
            )
        parsed = self._parse_json_object(await self._consume_stream_text(completion))
        entries = self._normalize_entries(parsed.get("entries"))
        appended_entries = self._append_entries(entries)
        sent_at = datetime.now().isoformat()
        self.memory.mark_visual_observations_biodata_sent(
            [row["observation_id"] for row in candidates],
            sent_at=sent_at,
        )
        return {
            "processed_observation_ids": [row["observation_id"] for row in candidates],
            "entries": appended_entries,
        }

    def _candidate_rows(self, observations: List[VisualObservation]) -> List[dict]:
        rows: List[dict] = []
        for observation in observations:
            reminder_context = self._extract_reminder_context(observation.raw_payload_json)
            content_present = any(
                [
                    self._clean_text(observation.inferred_user_activity),
                    self._clean_text(observation.summary),
                    self._clean_text(observation.detailed_description),
                    reminder_context,
                ]
            )
            if not content_present:
                continue
            rows.append(
                {
                    "observation_id": observation.observation_id,
                    "created_at": observation.created_at,
                    "app_name": self._clean_text(observation.app_name),
                    "page_hint": self._clean_text(observation.page_hint),
                    "summary": self._clean_text(observation.summary),
                    "detailed_description": self._clean_text(observation.detailed_description),
                    "inferred_user_activity": self._clean_text(observation.inferred_user_activity),
                    "reminder_context": reminder_context,
                }
            )
        return rows

    def _normalize_entries(self, value) -> List[dict]:
        if not isinstance(value, list):
            return []
        results: List[dict] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            note = self._clean_text(item.get("note"))
            if not note:
                continue
            bucket = self._bucket(item.get("bucket"), category=self._clean_text(item.get("category")), note=note)
            category = self._clean_text(item.get("category")) or "general"
            confidence = self._confidence(item.get("confidence"))
            results.append({"note": note, "bucket": bucket, "category": category, "confidence": confidence})
        return results

    def _append_entries(self, entries: List[dict]) -> List[dict]:
        if not entries:
            return []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        user_info_entries = [entry for entry in entries if entry["bucket"] == "user_info"]
        memory_entries = [entry for entry in entries if entry["bucket"] == "memory"]
        if user_info_entries:
            existing_user_info = self.memory.get_user_info().rstrip()
            user_lines: List[str] = []
            if existing_user_info:
                user_lines.append(existing_user_info)
                user_lines.append("")
            user_lines.append(f"## BioData Update - {timestamp}")
            for entry in user_info_entries:
                user_lines.append(f"- [{entry['category']}] {entry['note']}")
                self._index_entry(entry=entry, source_type="user_info_note")
            self.memory.save_user_info("\n".join(user_lines).strip() + "\n")
        if memory_entries:
            existing_memory = self.memory.get_working_memory().rstrip()
            memory_lines: List[str] = []
            if existing_memory:
                memory_lines.append(existing_memory)
                memory_lines.append("")
            memory_lines.append(f"## Memory Update - {timestamp}")
            for entry in memory_entries:
                memory_lines.append(f"- [{entry['category']}] {entry['note']}")
                self._index_entry(entry=entry, source_type="working_memory_note")
            self.memory.save_working_memory("\n".join(memory_lines).strip() + "\n")
        return entries

    def _semantic_context(self, rows: List[dict]) -> List[dict]:
        if self.semantic_memory is None:
            return []
        flattened_query = " ".join(
            " ".join(
                part
                for part in [
                    row.get("inferred_user_activity", ""),
                    row.get("summary", ""),
                    row.get("detailed_description", ""),
                    row.get("reminder_context", ""),
                ]
                if part
            )
            for row in rows
        ).strip()
        if not flattened_query:
            return []
        results = self.semantic_memory.retrieve(query=flattened_query, limit=8, rerank_limit=5)
        return self.semantic_memory.format_context(results)

    def _bucket(self, value, *, category: str, note: str) -> str:
        normalized = self._clean_text(value).lower()
        if normalized in {"memory", "user_info"}:
            return normalized
        category_normalized = category.lower()
        note_normalized = note.lower()
        if category_normalized in {"reminder"}:
            return "memory"
        if any(token in note_normalized for token in ["upcoming", "tomorrow", "today", "test", "exam", "deadline", "remind"]):
            return "memory"
        return "user_info"

    def _index_entry(self, *, entry: dict, source_type: str) -> None:
        note = self._clean_text(entry.get("note"))
        if not note:
            return
        content = f"[{entry.get('category', 'general')}] {note}"
        stable_id = hashlib.sha1(f"{source_type}:{content.lower()}".encode("utf-8")).hexdigest()
        self.memory.upsert_semantic_chunk(
            source_type=source_type,
            source_id=stable_id,
            source_ref=source_type,
            content=content,
            metadata_json=json.dumps(
                {
                    "category": entry.get("category"),
                    "bucket": entry.get("bucket"),
                    "confidence": entry.get("confidence"),
                    "stored_at": datetime.now().isoformat(),
                }
            ),
        )

    def _extract_reminder_context(self, raw_payload_json: str | None) -> str:
        if not raw_payload_json:
            return ""
        try:
            payload = json.loads(raw_payload_json)
        except json.JSONDecodeError:
            return ""
        return self._clean_text(payload.get("reminder_context"))

    def _clean_text(self, value) -> str:
        if value is None:
            return ""
        return str(value).strip()

    def _confidence(self, value) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    async def _consume_stream_text(self, completion) -> str:
        parts: List[str] = []
        async for chunk in completion:
            if not getattr(chunk, "choices", None):
                continue
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
            self.logger.warning("User biodata response was not valid JSON.")
            return {}
