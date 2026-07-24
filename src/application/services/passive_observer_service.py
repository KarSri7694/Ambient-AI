import json
import logging
import re
import uuid
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from application.ports.LLMProvider import LLMProvider
from application.ports.memory_port import MemoryPort
from application.ports.screen_capture_port import ScreenCapturePort
from application.services.interaction_trace import interaction_trace
from core.models import VisualObservation, VisualSession


class PassiveObserverService:
    """Capture periodic screenshots and persist passive visual context."""

    FULL_OBSERVER_PROMPT = """You are the passive visual observer for an ambient personal agent.

Look at the current screenshot, and return JSON only with exactly these fields:
{
  "app_page": "short app/site and page/screen description combined into one line",
  "summary": "1-2 sentence concrete summary of what is on screen",
  "detailed_description": "information-rich detail about what is visible",
  "inferred_user_activity": "what the user seems to be doing or trying to do",
  "maybe_require_a_reminder": "true or false, if the user might need a reminder about something on screen",
  "reminder_context": {
    "message_to_user": "if maybe_require_a_reminder is true, provide the reminder text to send to the user",
    "due_date": "optional ISO datetime string when the due time can be inferred, otherwise empty string"
  }
}

Rules:
- If you see anything important on screen that the user might need to remember, set maybe_require_a_reminder to true and provide reminder_context.message_to_user.
- Fill reminder_context.due_date only when the screenshot provides enough evidence to infer a concrete due datetime.
- Return only those six fields. Do not add any other keys.
- When the screen contains a chat or messaging interface, extract all the visible information most importantly infer if the user made any commitments or decisions, or something of importance is told to the user.  
- Prefer visible facts over speculation.
- Use app_page as a compact combined label such as "Amazon.in / TV product listing page" or "VS Code / Python file editor".
- Make detailed_description high recall when the screen is relevant: mention visible products, filters, prices, ratings, titles, tabs, files, buttons, text, and comparison cues.
- Include every useful visible clue that would help a later agent understand what is on screen.
- Use the provided screenshot timestamp only as context for when the observation was captured.
- If the screen is idle, blank, locked, or not useful, keep the four fields minimal and factual.
"""
    FAST_ROUTER_PROMPT = """You are a fast screen routing model for an ambient personal agent.

Look at the screenshot and return JSON only with exactly these fields:
{
  "app_page": "short app/site and page/screen description combined into one line",
  "summary": "1 sentence summary of what is on screen",
  "detailed_description": "short but concrete description of the current screen state",
  "inferred_user_activity": "what the user appears to be doing",
  "maybe_require_a_reminder": false,
  "reminder_context": {
    "message_to_user": "",
    "due_date": ""
  }
}

Rules:
- Keep the response compact and concrete.
- Prefer continuity-sensitive observations over exhaustive description.
- Return valid JSON only.
"""

    def __init__(
        self,
        *,
        memory: MemoryPort,
        llm_provider: LLMProvider,
        screen_capture: ScreenCapturePort,
        screenshot_root: str,
        fast_model: Optional[str] = None,
        full_model: Optional[str] = None,
        full_vlm_ssim_threshold: float = 0.70,
        ignore_apps: Optional[List[str]] = None,
        ignore_domains: Optional[List[str]] = None,
        always_full_apps: Optional[List[str]] = None,
        always_full_domains: Optional[List[str]] = None,
        fast_model_retry_count: int = 2,
        uiat_adapter: Optional[Any] = None,
        persist_observations: bool = True,
        capture_store: Optional[Any] = None,
        persist_payloads: bool = False,
        capture_control: Optional[Any] = None,
        logger: logging.Logger | None = None,
    ):
        self.memory = memory
        self.llm = llm_provider
        self.screen_capture = screen_capture
        self.screenshot_root = Path(screenshot_root)
        self.screenshot_root.mkdir(parents=True, exist_ok=True)
        self.fast_model = (fast_model or "").strip()
        self.full_model = (full_model or self.fast_model).strip()
        self.full_vlm_ssim_threshold = max(0.0, min(1.0, float(full_vlm_ssim_threshold)))
        self.ignore_apps = {str(item).strip().lower() for item in (ignore_apps or []) if str(item).strip()}
        self.ignore_domains = {str(item).strip().lower() for item in (ignore_domains or []) if str(item).strip()}
        self.always_full_apps = {str(item).strip().lower() for item in (always_full_apps or []) if str(item).strip()}
        self.always_full_domains = {str(item).strip().lower() for item in (always_full_domains or []) if str(item).strip()}
        self.fast_model_retry_count = max(0, int(fast_model_retry_count))
        self.uiat_adapter = uiat_adapter
        self.persist_observations = bool(persist_observations)
        self.capture_store = capture_store
        self.persist_payloads = bool(persist_payloads)
        self.capture_control = capture_control
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def capture_screenshot(self) -> str:
        screenshot_path = self._capture_path()
        return str(Path(self.screen_capture.capture_screenshot(str(screenshot_path))))

    def capture_lightweight_context(self) -> Dict[str, Any]:
        """Collect foreground metadata without invoking an embedding or vision model."""
        payload = self._inspect_foreground_window()
        return {
            "window_title": payload.get("window_title"),
            "window_class": payload.get("window_class"),
            "app_name": payload.get("app_hint"),
            "url": payload.get("foreground_url"),
            "domain": payload.get("domain_hint"),
            "accessible_text": str(payload.get("visible_text_summary") or "")[:4000],
            "contains_dialog": bool(payload.get("contains_dialog")),
            "contains_notification": bool(payload.get("contains_notification")),
        }

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
        similarity_score: float | None = None,
        persisted_screenshot_path: str | None = None,
        archive_source: bool = True,
        uiat_context_override: Optional[Dict[str, Any]] = None,
    ) -> Optional[VisualObservation]:
        stored_screenshot_path = persisted_screenshot_path or screenshot_path
        try:
            parsed = await self._analyze(
                screenshot_path=screenshot_path,
                model=model,
                recent_context=recent_context,
                captured_at=captured_at,
                similarity_score=similarity_score,
                uiat_context_override=uiat_context_override,
            )
        finally:
            if archive_source and self.capture_store is not None and Path(screenshot_path).exists():
                stored_screenshot_path = self.capture_store.store_file(
                    screenshot_path, kind="screenshot", delete_source=True
                )
        if not parsed:
            return None

        app_name, page_hint = self._split_app_page(self._opt_text(parsed.get("app_page")))

        raw_payload_json = json.dumps(parsed, ensure_ascii=False, indent=2)
        if self.capture_store is not None and self.persist_payloads:
            raw_payload_json = self.capture_store.store_bytes(
                raw_payload_json.encode("utf-8"),
                original_name=f"visual_payload_{uuid.uuid4().hex}.json",
                kind="visual_model_payload",
                mime_type="application/json",
            )
        observation = VisualObservation(
            observation_id=uuid.uuid4().hex,
            screenshot_path=stored_screenshot_path,
            created_at=captured_at or datetime.now().isoformat(),
            observation_type="screen",
            app_name=app_name,
            window_title=self._opt_text(parsed.get("_uiat_window_title")),
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
            raw_payload_json=raw_payload_json,
        )
        if not self.persist_observations:
            self.logger.debug(
                "Passive observer persistence disabled; returning transient observation for %s",
                screenshot_path,
            )
            return observation
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
        similarity_score: float | None = None,
        uiat_context_override: Optional[Dict[str, Any]] = None,
    ) -> dict:
        if not Path(screenshot_path).exists():
            self.logger.warning("Passive observer screenshot missing before analysis: %s", screenshot_path)
            return {}
        recent_observations = self.memory.get_recent_visual_observations(limit=3)
        previous_observation = recent_observations[0] if recent_observations else None
        uiat_context = dict(uiat_context_override or self._inspect_foreground_window())
        route = self._route_screenshot(
            similarity_score=similarity_score,
            uiat_context=uiat_context,
            previous_observation=previous_observation,
        )
        if route == "skip":
            return {}

        if route == "fast_model":
            parsed = await self._run_fast_model(
                screenshot_path=screenshot_path,
                model=model,
                captured_at=captured_at,
                similarity_score=similarity_score,
                previous_observation=previous_observation,
                uiat_context=uiat_context,
            )
        else:
            parsed = await self._run_full_model(
                screenshot_path=screenshot_path,
                model=model,
                recent_context=recent_context,
                captured_at=captured_at,
                similarity_score=similarity_score,
                previous_observation=previous_observation,
                recent_observations=recent_observations,
                uiat_context=uiat_context,
            )
        if isinstance(parsed, dict) and parsed:
            parsed.setdefault("_analysis_mode", route)
            if similarity_score is not None:
                parsed.setdefault("_similarity_score", similarity_score)
            if uiat_context:
                parsed.setdefault("_uiat_window_title", uiat_context.get("window_title"))
                parsed.setdefault("_uiat_domain", uiat_context.get("domain_hint"))
                parsed.setdefault("_uiat_url", uiat_context.get("foreground_url"))
        return parsed if isinstance(parsed, dict) else {}

    def _inspect_foreground_window(self) -> Dict[str, Any]:
        if self.uiat_adapter is None:
            return {}
        payload = self.uiat_adapter.inspect_foreground_window() or {}
        if not isinstance(payload, dict):
            return {}
        payload["domain_hint"] = self._infer_domain_hint(payload)
        payload["app_hint"] = self._infer_app_hint(payload)
        self.logger.debug(
            "UIAT context ok=%s app_hint=%r domain_hint=%r dialog=%s notification=%s items=%s",
            payload.get("ok"),
            payload.get("app_hint"),
            payload.get("domain_hint"),
            payload.get("contains_dialog"),
            payload.get("contains_notification"),
            len(payload.get("visible_items") or []),
        )
        return payload

    def _infer_domain_hint(self, uiat_context: Dict[str, Any]) -> Optional[str]:
        candidates = [
            str(uiat_context.get("foreground_url") or ""),
            str(uiat_context.get("window_title") or ""),
            str(uiat_context.get("visible_text_summary") or ""),
        ]
        for candidate in candidates:
            match = re.search(r"\b([a-z0-9-]+\.(?:com|in|org|net|ai|io|co|app|dev))\b", candidate.lower())
            if match:
                return match.group(1)
        return None

    def _infer_app_hint(self, uiat_context: Dict[str, Any]) -> Optional[str]:
        title = str(uiat_context.get("window_title") or "").strip()
        window_class = str(uiat_context.get("window_class") or "").strip()
        if title:
            if " - " in title:
                return title.split(" - ")[-1].strip()
            if " | " in title:
                return title.split(" | ")[-1].strip()
            return title[:80]
        return window_class or None

    def _route_screenshot(
        self,
        *,
        similarity_score: float | None,
        uiat_context: Dict[str, Any],
        previous_observation: Optional[VisualObservation],
    ) -> str:
        app_name = str(uiat_context.get("app_hint") or "").strip().lower()
        domain_hint = str(uiat_context.get("domain_hint") or "").strip().lower()
        previous_app = str(previous_observation.app_name or "").strip().lower() if previous_observation else ""
        previous_page = str(previous_observation.page_hint or "").strip().lower() if previous_observation else ""
        window_title = str(uiat_context.get("window_title") or "").strip().lower()
        app_switched = bool(app_name and previous_app and app_name != previous_app)
        domain_switched = bool(domain_hint and previous_page and domain_hint not in previous_page)
        has_override = (
            bool(uiat_context.get("contains_dialog"))
            or bool(uiat_context.get("contains_notification"))
            or app_switched
            or domain_switched
            or app_name in self.always_full_apps
            or domain_hint in self.always_full_domains
            or "error" in window_title
        )
        self.logger.debug(
            "Routing screenshot similarity=%s app=%r domain=%r prev_app=%r prev_page=%r app_switched=%s domain_switched=%s override=%s",
            similarity_score,
            app_name,
            domain_hint,
            previous_app,
            previous_page,
            app_switched,
            domain_switched,
            has_override,
        )
        excluded = (
            self._matches_policy(app_name, self.ignore_apps)
            or self._matches_policy(domain_hint, self.ignore_domains)
            or (
                self.capture_control is not None
                and self.capture_control.is_excluded(app_name=app_name, domain=domain_hint)
            )
        )
        if excluded:
            self.logger.debug("Routing decision=skip reason=policy_ignore")
            return "skip"
        if has_override:
            self.logger.debug("Routing decision=full_vlm reason=override")
            return "full_vlm"
        if similarity_score is None:
            self.logger.debug("Routing decision=full_vlm reason=no_similarity_score")
            return "full_vlm"
        if similarity_score < self.full_vlm_ssim_threshold:
            self.logger.debug(
                "Routing decision=full_vlm reason=similarity_below_threshold full_threshold=%s",
                self.full_vlm_ssim_threshold,
            )
            return "full_vlm"
        self.logger.debug(
            "Routing decision=fast_model reason=similarity_band full_threshold=%s skip_threshold=queue",
            self.full_vlm_ssim_threshold,
        )
        return "fast_model"

    def _matches_policy(self, value: str, policy_values: set[str]) -> bool:
        normalized = str(value or "").strip().lower()
        if not normalized:
            return False
        for candidate in policy_values:
            if candidate and (normalized == candidate or normalized.startswith(candidate) or candidate in normalized):
                self.logger.debug("Policy match value=%r candidate=%r", normalized, candidate)
                return True
        return False

    async def _run_fast_model(
        self,
        *,
        screenshot_path: str,
        model: str,
        captured_at: str | None,
        similarity_score: float | None,
        previous_observation: Optional[VisualObservation],
        uiat_context: Dict[str, Any],
    ) -> dict:
        payload = {
            "screenshot_captured_at": captured_at or datetime.now().isoformat(),
            "similarity_score": similarity_score,
            "previous_observation": (
                {
                    "app_name": previous_observation.app_name,
                    "page_hint": previous_observation.page_hint,
                    "summary": previous_observation.summary,
                    "inferred_user_activity": previous_observation.inferred_user_activity,
                }
                if previous_observation is not None
                else None
            ),
            "uiat_context": {
                "window_title": uiat_context.get("window_title"),
                "window_class": uiat_context.get("window_class"),
                "domain_hint": uiat_context.get("domain_hint"),
                "app_hint": uiat_context.get("app_hint"),
                "visible_text_summary": str(uiat_context.get("visible_text_summary") or "")[:1500],
                "contains_dialog": bool(uiat_context.get("contains_dialog")),
                "contains_notification": bool(uiat_context.get("contains_notification")),
            },
        }
        attempts = self.fast_model_retry_count + 1
        for attempt in range(1, attempts + 1):
            self.logger.debug(
                "Fast model attempt=%s/%s model=%s similarity=%s screenshot=%s",
                attempt,
                attempts,
                self.fast_model or model,
                similarity_score,
                screenshot_path,
            )
            parsed = await self._invoke_model(
                prompt=self.FAST_ROUTER_PROMPT,
                payload=payload,
                screenshot_path=screenshot_path,
                model_name=self.fast_model or model,
            )
            if parsed:
                self.logger.debug("Fast model returned valid JSON on attempt=%s", attempt)
                return parsed
            self.logger.debug("Fast model returned invalid/empty JSON on attempt=%s", attempt)
        return {}

    async def _run_full_model(
        self,
        *,
        screenshot_path: str,
        model: str,
        recent_context: str,
        captured_at: str | None,
        similarity_score: float | None,
        previous_observation: Optional[VisualObservation],
        recent_observations: List[VisualObservation],
        uiat_context: Dict[str, Any],
    ) -> dict:
        payload = {
            "screenshot_captured_at": captured_at or datetime.now().isoformat(),
            "similarity_score": similarity_score,
            "recent_context": recent_context,
            "visual_digest": self.memory.get_visual_digest(),
            "uiat_context": {
                "window_title": uiat_context.get("window_title"),
                "window_class": uiat_context.get("window_class"),
                "domain_hint": uiat_context.get("domain_hint"),
                "app_hint": uiat_context.get("app_hint"),
                "visible_text_summary": str(uiat_context.get("visible_text_summary") or "")[:4000],
                "contains_dialog": bool(uiat_context.get("contains_dialog")),
                "contains_notification": bool(uiat_context.get("contains_notification")),
            },
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
        return await self._invoke_model(
            prompt=self.FULL_OBSERVER_PROMPT,
            payload=payload,
            screenshot_path=screenshot_path,
            model_name=self.full_model or model,
        )

    async def _invoke_model(
        self,
        *,
        prompt: str,
        payload: Dict[str, Any],
        screenshot_path: str,
        model_name: str,
    ) -> dict:
        self.logger.debug(
            "Invoking passive observer model=%s image=%s payload_keys=%s",
            model_name,
            screenshot_path,
            sorted(payload.keys()),
        )
        with interaction_trace("passive_observer", metadata={"image_path": screenshot_path, "model": model_name}):
            if hasattr(self.llm, "load_model"):
                await self.llm.load_model(model_name)
            completion = await self.llm.chat_completion_stream(
                model=model_name,
                messages=[
                    {"role": "system", "content": self._build_system_prompt(prompt)},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
                ],
                tools=None,
                image=screenshot_path,
            )
        text = await self._consume_stream_text(completion)
        parsed = self._parse_json_object(text)
        self.logger.debug(
            "Passive observer model=%s parsed_json=%s raw_text_chars=%s",
            model_name,
            bool(parsed),
            len(text or ""),
        )
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
