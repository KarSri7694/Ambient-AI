import json
import logging
from datetime import datetime
from typing import List

from application.ports.LLMProvider import LLMProvider
from application.ports.memory_port import MemoryPort
from application.services.interaction_trace import interaction_trace
from core.models import VisualObservation


class UserBioDataService:
    """Append durable user biodata notes into USER_INFO.md during idle time."""

    OBSERVATION_LIMIT = 20

    BIODATA_PROMPT = """You build append-only user biodata notes from recent passive observations.

Return JSON only:
{
  "entries": [
    {
      "note": "short durable biodata note about the user",
      "category": "interest|habit|education|work|concern|preference|commitment|reminder",
      "confidence": 0.0
    }
  ]
}

Rules:
- Extract only durable or meaningfully recurring user biodata.
- Prefer user interests, habits, concerns, commitments, preferences, study/work context, and reminder-worthy ongoing themes.
- Ignore trivial UI actions, one-off mechanical interactions, and generic screen summaries.
- Do not restate the screenshot. Infer concise profile-style notes.
- Notes must be short, factual, and useful later.
- If nothing is biodata-worthy, return an empty entries list.
- Do not add any keys other than entries, and each entry may only contain note, category, confidence.
"""

    def __init__(
        self,
        *,
        memory: MemoryPort,
        llm_provider: LLMProvider,
        logger: logging.Logger | None = None,
    ):
        self.memory = memory
        self.llm = llm_provider
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
            "existing_user_biodata": self.memory.get_user_info(),
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
            category = self._clean_text(item.get("category")) or "general"
            confidence = self._confidence(item.get("confidence"))
            results.append({"note": note, "category": category, "confidence": confidence})
        return results

    def _append_entries(self, entries: List[dict]) -> List[dict]:
        if not entries:
            return []
        existing = self.memory.get_user_info().rstrip()
        lines: List[str] = []
        if existing:
            lines.append(existing)
            lines.append("")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines.append(f"## BioData Update - {timestamp}")
        for entry in entries:
            lines.append(f"- [{entry['category']}] {entry['note']}")
        self.memory.save_user_info("\n".join(lines).strip() + "\n")
        return entries

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
