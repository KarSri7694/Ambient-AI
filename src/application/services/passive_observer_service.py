import json
import logging
import uuid
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from application.ports.LLMProvider import LLMProvider
from application.ports.memory_port import MemoryPort
from application.ports.screen_capture_port import ScreenCapturePort
from application.services.interaction_trace import interaction_trace
from core.models import VisualObservation, VisualSession


class PassiveObserverService:
    """Capture periodic screenshots and persist passive visual context."""

    OBSERVER_PROMPT = """You are the passive visual observer for an ambient personal agent.

Look at the current screenshot, and return JSON only with exactly these fields:
{
  "app_page": "short app/site and page/screen description combined into one line",
  "summary": "1-2 sentence concrete summary of what is on screen",
  "detailed_description": "information-rich detail about what is visible",
  "inferred_user_activity": "what the user seems to be doing or trying to do",
  "maybe_require_a_reminder": "true or false, if the user might need a reminder about something on screen",
  "reminder_context": "if maybe_require_a_reminder is true, provide a short description of what the reminder should be about"
}

Rules:
- If you see anything important on screen that the user might need to remember, set maybe_require_a_reminder to true and provide a short reminder_context. 
- Return only those six fields. Do not add any other keys.
- When the screen contains a chat or messaging interface, extract all the visible information most importantly infer if the user made any commitments or decisions, or something of importance is told to the user.  
- Prefer visible facts over speculation.
- Use app_page as a compact combined label such as "Amazon.in / TV product listing page" or "VS Code / Python file editor".
- Make detailed_description high recall when the screen is relevant: mention visible products, filters, prices, ratings, titles, tabs, files, buttons, text, and comparison cues.
- Include every useful visible clue that would help a later agent understand what is on screen.
- Use the provided screenshot timestamp only as context for when the observation was captured.
- If the screen is idle, blank, locked, or not useful, keep the four fields minimal and factual.
"""

    def __init__(
        self,
        *,
        memory: MemoryPort,
        llm_provider: LLMProvider,
        screen_capture: ScreenCapturePort,
        screenshot_root: str,
        logger: logging.Logger | None = None,
    ):
        self.memory = memory
        self.llm = llm_provider
        self.screen_capture = screen_capture
        self.screenshot_root = Path(screenshot_root)
        self.screenshot_root.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def capture_screenshot(self) -> str:
        screenshot_path = self._capture_path()
        return str(Path(self.screen_capture.capture_screenshot(str(screenshot_path))))

    def _build_system_prompt(self, prompt: str) -> str:
        now = datetime.now()
        preamble = (
            f"Current day of week: {now.strftime('%A')}\n"
            f"Current date: {now.strftime('%Y-%m-%d')}\n"
            f"Current time: {now.strftime('%H:%M:%S')}\n\n"
        )
        return preamble + prompt

    async def process_screenshot(
        self,
        *,
        screenshot_path: str,
        model: str,
        recent_context: str,
        captured_at: str | None = None,
    ) -> Optional[VisualObservation]:
        parsed = await self._analyze(
            screenshot_path=screenshot_path,
            model=model,
            recent_context=recent_context,
            captured_at=captured_at,
        )
        if not parsed:
            return None

        app_name, page_hint = self._split_app_page(self._opt_text(parsed.get("app_page")))

        observation = VisualObservation(
            observation_id=uuid.uuid4().hex,
            screenshot_path=screenshot_path,
            created_at=captured_at or datetime.now().isoformat(),
            observation_type="screen",
            app_name=app_name,
            window_title=None,
            page_hint=page_hint,
            summary=self._opt_text(parsed.get("summary")) or "Passive observation captured.",
            detailed_description=self._opt_text(parsed.get("detailed_description")) or "",
            inferred_user_activity=self._opt_text(parsed.get("inferred_user_activity")) or "",
            previous_activity_status="unclear",
            salient_entities=[],
            completed_items=[],
            open_loops=[],
            possible_next_task=None,
            suggested_research_topics=[],
            user_fact_hypotheses=[],
            confidence=0.0,
            raw_payload_json=json.dumps(parsed, ensure_ascii=False, indent=2),
        )
        session = self._attach_to_session(observation)
        observation = replace(observation, session_id=session.session_id)
        self.memory.append_visual_observation(observation)
        self.memory.upsert_visual_session(
            replace(
                session,
                observation_ids=self._append_unique(session.observation_ids, observation.observation_id),
                status=self._session_status(observation),
                ended_at=observation.created_at,
                last_activity_at=observation.created_at,
            )
        )
        self.refresh_digest()
        return observation

    async def observe(self, *, model: str, recent_context: str) -> Optional[VisualObservation]:
        screenshot_path = self.capture_screenshot()
        return await self.process_screenshot(
            screenshot_path=screenshot_path,
            model=model,
            recent_context=recent_context,
        )

    def refresh_digest(self, session_limit: int = 4, observation_limit: int = 5) -> None:
        sessions = self.memory.list_visual_sessions(statuses=["open"], limit=session_limit)
        observations = self.memory.get_recent_visual_observations(limit=observation_limit)
        lines = ["# Passive Visual Context", ""]
        if sessions:
            lines.append("## Active Visual Sessions")
            for session in sessions:
                title = session.activity_summary or session.page_hint or session.app_name or "unknown activity"
                lines.append(f"- Session {session.session_id[:8]} | {title}")
                if session.app_name or session.page_hint:
                    lines.append(
                        f"  App/Page: {session.app_name or 'unknown'} / {session.page_hint or 'unknown'}"
                    )
        else:
            lines.append("## Active Visual Sessions")
            lines.append("- No active visual sessions.")

        lines.append("")
        lines.append("## Recent Observations")
        if observations:
            for observation in observations:
                lines.append(
                    f"- {observation.created_at}: {observation.summary}"
                )
                if observation.detailed_description:
                    lines.append(f"  Detail: {observation.detailed_description[:500]}")
                if observation.previous_activity_status and observation.previous_activity_status != "unclear":
                    lines.append(f"  Previous status: {observation.previous_activity_status}")
                if observation.completed_items:
                    lines.append(f"  Completed: {', '.join(observation.completed_items[:3])}")
                if observation.open_loops:
                    lines.append(f"  Open loops: {', '.join(observation.open_loops[:3])}")
                if observation.possible_next_task:
                    lines.append(f"  Possible next task: {observation.possible_next_task}")
                if observation.user_fact_hypotheses:
                    titles = [
                        str(item.get("title", "")).strip()
                        for item in observation.user_fact_hypotheses[:3]
                        if str(item.get("title", "")).strip()
                    ]
                    if titles:
                        lines.append(f"  User fact hypotheses: {', '.join(titles)}")
                if observation.suggested_research_topics:
                    lines.append(
                        f"  Research topics: {', '.join(observation.suggested_research_topics[:3])}"
                    )
        else:
            lines.append("- No recent visual observations.")
        self.memory.save_visual_digest("\n".join(lines) + "\n")

    async def _analyze(
        self,
        *,
        screenshot_path: str,
        model: str,
        recent_context: str,
        captured_at: str | None = None,
    ) -> dict:
        if not Path(screenshot_path).exists():
            self.logger.warning("Passive observer screenshot missing before analysis: %s", screenshot_path)
            return {}
        recent_observations = self.memory.get_recent_visual_observations(limit=3)
        previous_observation = recent_observations[0] if recent_observations else None
        payload = {
            "screenshot_captured_at": captured_at or datetime.now().isoformat(),
            "recent_context": recent_context,
            "visual_digest": self.memory.get_visual_digest(),
            "previous_observation": (
                {
                    "summary": previous_observation.summary,
                    "detailed_description": previous_observation.detailed_description,
                    "inferred_user_activity": previous_observation.inferred_user_activity,
                    "previous_activity_status": previous_observation.previous_activity_status,
                    "completed_items": previous_observation.completed_items,
                    "open_loops": previous_observation.open_loops,
                    "possible_next_task": previous_observation.possible_next_task,
                    "user_fact_hypotheses": previous_observation.user_fact_hypotheses,
                }
                if previous_observation is not None
                else None
            ),
            "recent_visual_observations": [
                {
                    "app_name": item.app_name,
                    "page_hint": item.page_hint,
                    "summary": item.summary,
                    "detailed_description": item.detailed_description,
                    "inferred_user_activity": item.inferred_user_activity,
                    "previous_activity_status": item.previous_activity_status,
                    "completed_items": item.completed_items,
                    "open_loops": item.open_loops,
                    "user_fact_hypotheses": item.user_fact_hypotheses,
                }
                for item in recent_observations
            ],
        }
        with interaction_trace("passive_observer", metadata={"image_path": screenshot_path}):
            completion = await self.llm.chat_completion_stream(
                model=model,
                messages=[
                    {"role": "system", "content": self._build_system_prompt(self.OBSERVER_PROMPT)},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
                ],
                tools=None,
                image=screenshot_path,
            )
        text = await self._consume_stream_text(completion)
        parsed = self._parse_json_object(text)
        return parsed if isinstance(parsed, dict) else {}

    def _attach_to_session(self, observation: VisualObservation) -> VisualSession:
        existing = self.memory.list_visual_sessions(statuses=["open"], limit=3)
        best: Optional[VisualSession] = None
        best_score = 0.0
        for session in existing:
            score = self._continuation_score(session, observation)
            if score > best_score:
                best_score = score
                best = session
        if best is None or best_score < 0.45:
            return VisualSession(
                session_id=uuid.uuid4().hex,
                started_at=observation.created_at,
                ended_at=observation.created_at,
                status="open",
                activity_summary=observation.inferred_user_activity or observation.summary,
                app_name=observation.app_name,
                window_title=observation.window_title,
                page_hint=observation.page_hint,
                last_activity_at=observation.created_at,
                continuation_score=0.0,
                observation_ids=[],
                related_loop_ids=[],
            )
        return replace(
            best,
            ended_at=observation.created_at,
            activity_summary=self._merge_text(best.activity_summary, observation.inferred_user_activity or observation.summary),
            app_name=best.app_name or observation.app_name,
            window_title=best.window_title or observation.window_title,
            page_hint=best.page_hint or observation.page_hint,
            last_activity_at=observation.created_at,
            continuation_score=best_score,
        )

    def _continuation_score(self, session: VisualSession, observation: VisualObservation) -> float:
        score = 0.0
        if session.app_name and observation.app_name and session.app_name.lower() == observation.app_name.lower():
            score += 0.35
        if session.page_hint and observation.page_hint and session.page_hint.lower() == observation.page_hint.lower():
            score += 0.25
        if session.window_title and observation.window_title and session.window_title.lower() == observation.window_title.lower():
            score += 0.2
        if session.activity_summary and observation.inferred_user_activity:
            left = set(session.activity_summary.lower().split())
            right = set(observation.inferred_user_activity.lower().split())
            if left and right:
                overlap = len(left & right) / max(len(left | right), 1)
                score += min(overlap, 1.0) * 0.2
        return min(score, 0.95)

    def _session_status(self, observation: VisualObservation) -> str:
        status = observation.previous_activity_status.lower()
        if status == "completed":
            return "completed"
        if status == "left_midway":
            return "open"
        return "open"

    def _capture_path(self) -> Path:
        return self.screenshot_root / f"observer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

    def _prune_screenshots(self, keep: int = 5) -> None:
        screenshots = sorted(self.screenshot_root.glob("observer_*.png"), key=lambda path: path.stat().st_mtime, reverse=True)
        for stale in screenshots[keep:]:
            self._trim_screenshot(stale)

    def _trim_screenshot(self, path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            self.logger.debug("Failed to remove screenshot %s", path)

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
            self.logger.warning("Passive observer response was not valid JSON.")
            return {}

    def _merge_text(self, existing: str, new: str) -> str:
        if not existing:
            return new
        if not new or new in existing:
            return existing
        return f"{existing} | {new}"[:400]

    def _append_unique(self, values: List[str], value: str) -> List[str]:
        if value in values:
            return list(values)
        return [*values, value]

    def _split_app_page(self, value: Optional[str]) -> tuple[Optional[str], Optional[str]]:
        if not value:
            return None, None
        for separator in (" / ", " | ", " - "):
            if separator in value:
                left, right = value.split(separator, 1)
                return self._opt_text(left), self._opt_text(right)
        return value, None

    def _list_text(self, value) -> List[str]:
        if not isinstance(value, list):
            return []
        results: List[str] = []
        for item in value:
            text = self._opt_text(item)
            if text:
                results.append(text)
        return results[:8]

    def _list_dicts(self, value) -> List[dict]:
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, dict)][:8]

    def _opt_text(self, value) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _num(self, value, default: float) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return default

    def _truthy(self, value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
