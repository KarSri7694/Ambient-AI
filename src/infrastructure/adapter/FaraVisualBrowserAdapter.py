from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import quote_plus

from application.ports.LLMProvider import LLMProvider
from application.ports.tool_bridge_port import (
    BrowserToolBridgePort,
    BrowserToolSessionPort,
)
from local_control.computer import EmergencyStopController
from local_control.safety import ComputerControlTerminated


class BrowserPolicyError(RuntimeError):
    """Raised when a visual browser action violates the host browser safety policy."""


class BrowserSafetyPolicy:
    """Deterministic host-side policy for an unsandboxed research browser.

    The browser agent is allowed to interact with normal sites, including POST
    requests and account-like pages. Downloads are disabled at the Playwright
    context and cancelled by the download event handler.
    """
    def __init__(
        self,
        *,
        blocked_domains: Optional[List[str]] = None,
        blocked_path_markers: Optional[List[str]] = None,
    ) -> None:
        # Deprecated compatibility knobs. Browser mutation/domain/path filtering is
        # no longer enforced here; downloads remain disabled at the browser context.
        self.blocked_domains = set()
        self.blocked_path_markers = set()

    def validate_navigation(self, url: str) -> str:
        value = str(url or "").strip()
        if not value:
            raise BrowserPolicyError("Navigation URL is empty.")
        return value

    def request_allowed(self, *, url: str, method: str) -> tuple[bool, str]:
        return True, "public browser request"


class FaraVisualBrowserSession(BrowserToolSessionPort):
    """Visible local browser controlled exclusively through screenshots and coordinates."""

    TOOL_NAME = "computer_use"
    ACTION_ALIASES = {
        "click": "left_click",
        "keypress": "key",
        "input_text": "type",
        "back": "history_back",
    }
    ALLOWED_ACTIONS = {
        "key",
        "type",
        "mouse_move",
        "left_click",
        "double_click",
        "right_click",
        "scroll",
        "hscroll",
        "visit_url",
        "web_search",
        "history_back",
        "pause_and_memorize_fact",
        "ask_user_question",
        "wait",
        "terminate",
    }
    FARA_TOOL = {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": (
                "Control the visible research browser from its latest screenshot. "
                "Choose exactly one action per turn."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": sorted(ALLOWED_ACTIONS)},
                    "coordinate": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "text": {"type": "string"},
                    "keys": {"type": "array", "items": {"type": "string"}},
                    "pixels": {"type": "number"},
                    "url": {"type": "string"},
                    "query": {"type": "string"},
                    "fact": {"type": "string"},
                    "question": {"type": "string"},
                    "time": {"type": "number"},
                    "answer": {"type": "string"},
                    "status": {"type": "string"},
                    "press_enter": {"type": "boolean"},
                    "delete_existing_text": {"type": "boolean"},
                },
                "required": ["action"],
                "additionalProperties": False,
            },
        },
    }
    SYSTEM_PROMPT = """You are Fara, a computer use agent (CUA) specialized for web browsers. You are developed by Microsoft AI Frontiers. You assist users with completing and automating tasks that require the use of a web browser.

The model was trained in the timeframe of January - April 2026. You can effectively perform tasks even beyond this range by accessing the web browser and using the latest information on the live web. But your knowledge cutoff is limited to early 2026, so you may not be aware of events or developments that occurred after that time, without explicitly browsing and searching for latest information on the web.

This edition of the model was trained using SFT on top of Qwen3.5-27B, using a synthetic data mixture generated and developed by Microsoft AI Frontiers.

A critical point is a situation where we must pause and request information or confirmation from the user before proceeding. There are three types:

Case 1: Missing User Information - The task requires personal information that the user has not provided (e.g., email, phone number, address, payment details). Never fabricate or assume personal information. Fill in only what the user has explicitly provided, then pause and ask for any missing required fields.

Case 2: Underspecified Task - The task description is ambiguous or missing details needed to make a decision at the current step. Pause and ask for clarification.

Case 3: Irreversible Action - We are about to perform an action that cannot be undone (e.g., submitting a form, completing a purchase, sending a message, deleting data). If the user explicitly authorized the action, proceed. Otherwise, stop and ask for confirmation.

Only stop at a critical point if (1) required information is missing, (2) the task is ambiguous, OR (3) an irreversible action lacks explicit user authorization.
"""
    XML_TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

    def __init__(
        self,
        *,
        llm_provider: LLMProvider,
        profile_dir: Path,
        screenshot_dir: Path,
        headless: bool,
        viewport_width: int,
        viewport_height: int,
        max_steps: int,
        settle_ms: int,
        search_url_template: str,
        browser_channel: str,
        browser_executable_path: str,
        policy: BrowserSafetyPolicy,
        screenshot_retention: bool,
        logger: logging.Logger,
        interrupt_checker: Optional[Callable[[], None]] = None,
    ) -> None:
        self.llm = llm_provider
        self.profile_dir = profile_dir
        self.screenshot_dir = screenshot_dir / uuid.uuid4().hex
        self.headless = headless
        self.viewport_width = max(640, int(viewport_width))
        self.viewport_height = max(480, int(viewport_height))
        self.max_steps = max(1, int(max_steps))
        self.settle_ms = max(0, int(settle_ms))
        self.search_url_template = search_url_template
        self.browser_channel = browser_channel.strip()
        self.browser_executable_path = browser_executable_path.strip()
        self.policy = policy
        self.screenshot_retention = screenshot_retention
        self.logger = logger
        self.interrupt_checker = interrupt_checker
        self._playwright: Any = None
        self._context: Any = None
        self._page: Any = None
        self._history: list[str] = []
        self._facts: list[dict[str, Any]] = []
        self._blocked_actions: list[dict[str, Any]] = []
        self._screenshot_paths: list[Path] = []
        self._last_screenshot_path: Optional[Path] = None
        self._last_action_signature = ""
        self._repeat_count = 0
        self._stop = EmergencyStopController()

    async def start(self) -> None:
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:
            raise RuntimeError(
                "The Fara visual browser requires Python Playwright. Install the Windows "
                "requirements and run 'python -m playwright install chromium'."
            ) from exc

        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self._playwright = await async_playwright().start()
        launch_kwargs = self._launch_kwargs()
        if self.browser_channel:
            launch_kwargs["channel"] = self.browser_channel
        if self.browser_executable_path:
            launch_kwargs["executable_path"] = self.browser_executable_path
            launch_kwargs.pop("channel", None)
        try:
            self._context = await self._playwright.chromium.launch_persistent_context(**launch_kwargs)
            await self._context.clear_permissions()
            await self._context.route("**/*", self._route_request)
            self._context.on("page", self._on_new_page)
            self._context.on("dialog", lambda dialog: asyncio.create_task(dialog.dismiss()))
            self._context.on("download", lambda download: asyncio.create_task(download.cancel()))
            self._page = self._context.pages[-1] if self._context.pages else await self._context.new_page()
            if self._page.url == "about:blank":
                await self._page.goto("https://duckduckgo.com/", wait_until="domcontentloaded")
            self._stop.start()
        except BaseException:
            await self.cleanup()
            raise

    def _launch_kwargs(self) -> dict[str, Any]:
        launch_kwargs: dict[str, Any] = {
            "user_data_dir": str(self.profile_dir.resolve()),
            "headless": self.headless,
            "viewport": {"width": self.viewport_width, "height": self.viewport_height},
            "accept_downloads": False,
            "service_workers": "block",
            "args": [
                "--disable-extensions",
                "--disable-features=AutofillServerCommunication,PasswordManagerOnboarding",
                "--disable-notifications",
                f"--window-size={self.viewport_width},{self.viewport_height}",
                "--start-maximized",
                "--no-first-run",
                "--no-default-browser-check",
            ],
        }
        return launch_kwargs

    async def _route_request(self, route: Any, request: Any) -> None:
        allowed, reason = self.policy.request_allowed(url=request.url, method=request.method)
        if allowed:
            await route.continue_()
            return
        self.logger.info("Blocked browser request %s %s: %s", request.method, request.url, reason)
        await route.abort("blockedbyclient")

    def _on_new_page(self, page: Any) -> None:
        """Follow a user-visible target=_blank navigation without inspecting its DOM."""
        self._page = page

    async def get_all_tools(self) -> List[Dict[str, Any]]:
        return []

    async def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        raise RuntimeError("Fara visual sessions execute their own screenshot/action loop.")

    async def run_task(
        self,
        *,
        task: str,
        model: str,
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> str:
        if self._page is None:
            await self.start()
        started_at = datetime.now(timezone.utc).isoformat()
        terminal_status = "step_limit"
        terminal_answer = "The visual browser reached its action limit before completing the task."

        for step in range(1, self.max_steps + 1):
            self._check_interrupted()
            self._stop.check()
            self._page = await self._active_page()
            screenshot_path = await self._capture(step)
            current_url = self._page.url
            action = await self._request_action(
                task=task,
                model=model,
                screenshot_path=screenshot_path,
                current_url=current_url,
                step=step,
            )
            signature = json.dumps(action, sort_keys=True, ensure_ascii=False)
            self._check_interrupted()
            if signature == self._last_action_signature:
                self._repeat_count += 1
            else:
                self._last_action_signature = signature
                self._repeat_count = 0
            if self._repeat_count >= 2:
                terminal_status = "blocked"
                terminal_answer = "Stopped after the model repeated the same browser action three times."
                break

            try:
                outcome = await self._execute_action(action)
            except BrowserPolicyError as exc:
                blocked = {
                    "step": step,
                    "url": current_url,
                    "action": action,
                    "reason": str(exc),
                }
                self._blocked_actions.append(blocked)
                outcome = f"Blocked by browser safety policy: {exc}"
            self._history.append(
                f"Step {step}: {json.dumps(action, ensure_ascii=False)} -> {outcome}"
            )
            self._history = self._history[-30:]
            self._emit(
                event_callback,
                {
                    "type": "browser_visual_step",
                    "step": step,
                    "url": self._page.url,
                    "action": action,
                    "outcome": outcome,
                    "screenshot_path": str(screenshot_path),
                },
            )
            normalized = self.ACTION_ALIASES.get(str(action.get("action") or ""), str(action.get("action") or ""))
            if normalized == "terminate":
                terminal_status = str(action.get("status") or "completed")
                terminal_answer = str(action.get("answer") or action.get("text") or "Browser research completed.")
                break
            if normalized == "ask_user_question":
                terminal_status = "needs_user_input"
                terminal_answer = str(action.get("question") or "The browser agent needs more information.")
                break

        result = {
            "status": terminal_status,
            "task_summary": terminal_answer,
            "candidates": self._facts,
            "comparison_summary": terminal_answer,
            "blocked_actions": self._blocked_actions,
            "final_url": self._page.url if self._page is not None else "",
            "steps": len(self._history),
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "security": {
                "mode": "host_read_only_visual",
                "dom_access": False,
                "uia_access": False,
                "mutating_http_requests_allowed": False,
                "dedicated_profile": str(self.profile_dir),
            },
        }
        self._emit(event_callback, {"type": "browser_visual_completed", "result": result})
        return json.dumps(result, ensure_ascii=False, indent=2)

    async def _request_action(
        self,
        *,
        task: str,
        model: str,
        screenshot_path: Path,
        current_url: str,
        step: int,
    ) -> dict[str, Any]:
        state = {
            "approved_task": task,
            "step": step,
            "current_url": current_url,
            "viewport": [self.viewport_width, self.viewport_height],
            "memorized_facts": self._facts[-12:],
            "recent_actions": self._history[-12:],
            "instruction": "Inspect the screenshot and choose exactly one next action.",
        }
        completion = await self.llm.chat_completion_stream(
            model=model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(state, ensure_ascii=False, indent=2)},
            ],
            tools=[self.FARA_TOOL],
            image=str(screenshot_path),
            temperature=0.0,
            chat_template_kwargs={"enable_thinking": False},
        )
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: dict[int, dict[str, str]] = {}
        async for chunk in completion:
            self._check_interrupted()
            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta
            if getattr(delta, "content", None):
                content_parts.append(delta.content)
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                reasoning_parts.append(reasoning)
            for tool_call in getattr(delta, "tool_calls", None) or []:
                entry = tool_calls.setdefault(tool_call.index, {"name": "", "arguments": ""})
                function = getattr(tool_call, "function", None)
                if function is not None:
                    entry["name"] += getattr(function, "name", None) or ""
                    entry["arguments"] += getattr(function, "arguments", None) or ""
        if tool_calls:
            first = tool_calls[min(tool_calls)]
            if first["name"] and first["name"] != self.TOOL_NAME:
                raise ValueError(f"Fara returned unsupported tool: {first['name']}")
            try:
                return self._validate_action(json.loads(first["arguments"] or "{}"))
            except json.JSONDecodeError as exc:
                raise ValueError("Fara returned malformed tool arguments.") from exc
        raw_text = "\n".join(part for part in ["".join(content_parts), "".join(reasoning_parts)] if part)
        return self._validate_action(self._parse_text_action(raw_text))

    def _parse_text_action(self, text: str) -> dict[str, Any]:
        candidate = str(text or "").strip()
        qwen_xml = self._parse_qwen_xml_action(candidate)
        if qwen_xml:
            return qwen_xml
        match = self.XML_TOOL_RE.search(candidate)
        if match:
            candidate = match.group(1)
        else:
            start, end = candidate.find("{"), candidate.rfind("}")
            if start >= 0 and end > start:
                candidate = candidate[start : end + 1]
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Fara did not return a parseable browser action: {text[:300]}") from exc
        if value.get("name") == self.TOOL_NAME and isinstance(value.get("arguments"), dict):
            value = value["arguments"]
        return value

    def _parse_qwen_xml_action(self, text: str) -> dict[str, Any]:
        for match in re.finditer(r"<function=([\w.-]+)>([\s\S]*?)</function>", text or ""):
            if match.group(1).strip() != self.TOOL_NAME:
                continue
            args: dict[str, Any] = {}
            for param_match in re.finditer(
                r"<parameter=([\w.-]+)>([\s\S]*?)</parameter>",
                match.group(2),
            ):
                key = param_match.group(1).strip()
                raw_value = param_match.group(2).strip()
                if key == "action":
                    args.update(self._recover_action_parameter(raw_value))
                    continue
                try:
                    args[key] = json.loads(raw_value)
                except json.JSONDecodeError:
                    args[key] = raw_value
            if not args:
                missing_close = re.search(
                    r"<parameter=([\w.-]+)>([\s\S]*)",
                    match.group(2),
                    re.DOTALL,
                )
                if missing_close:
                    key = missing_close.group(1).strip()
                    raw_value = re.sub(
                        r"</(?:function|tool_call)>\s*$",
                        "",
                        missing_close.group(2).strip(),
                    ).strip()
                    if key == "action":
                        args.update(self._recover_action_parameter(raw_value))
                    else:
                        try:
                            args[key] = json.loads(raw_value)
                        except json.JSONDecodeError:
                            args[key] = raw_value
            return args
        return {}

    def _recover_action_parameter(self, raw_value: str) -> dict[str, Any]:
        value = str(raw_value or "").strip()
        if not value:
            return {}
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, str):
                return {"action": parsed}
        except json.JSONDecodeError:
            pass

        wrapped = f'{{"action": "{value}'
        for _ in range(3):
            try:
                parsed = json.loads(wrapped)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                wrapped = wrapped.rstrip("}")

        recovered: dict[str, Any] = {}
        action_match = re.match(r"([A-Za-z_][\w-]*)", value)
        if action_match:
            recovered["action"] = action_match.group(1)
        for key, string_value in re.findall(r'"([\w.-]+)"\s*:\s*"([^"]*)"', value):
            recovered[key] = string_value
        for key, number_value in re.findall(r'"([\w.-]+)"\s*:\s*(-?\d+(?:\.\d+)?)', value):
            if key not in recovered:
                recovered[key] = float(number_value) if "." in number_value else int(number_value)
        return recovered

    def _validate_action(self, action: Any) -> dict[str, Any]:
        if not isinstance(action, dict):
            raise ValueError("Fara browser action must be an object.")
        normalized = dict(action)
        raw_action_value = str(normalized.get("action") or "").strip()
        if raw_action_value and raw_action_value not in self.ALLOWED_ACTIONS:
            recovered = self._recover_action_parameter(raw_action_value)
            if recovered:
                recovered.update({key: value for key, value in normalized.items() if key != "action"})
                normalized = recovered
        action_name = self.ACTION_ALIASES.get(
            str(normalized.get("action") or "").strip(),
            str(normalized.get("action") or "").strip(),
        )
        if action_name not in self.ALLOWED_ACTIONS:
            raise ValueError(f"Fara browser action is not allowed: {action_name or 'missing'}")
        normalized["action"] = action_name
        if action_name == "visit_url" and not str(normalized.get("url") or "").strip():
            text_value = str(normalized.get("text") or "").strip()
            if text_value.startswith(("http://", "https://")):
                normalized["url"] = text_value
        if action_name == "web_search" and not str(normalized.get("query") or "").strip():
            text_value = str(normalized.get("text") or "").strip()
            if text_value:
                normalized["query"] = text_value
        if action_name in {"mouse_move", "left_click", "double_click", "right_click"}:
            coordinate = normalized.get("coordinate")
            if not isinstance(coordinate, list) or len(coordinate) != 2:
                raise ValueError(f"{action_name} requires coordinate=[x, y].")
            x, y = float(coordinate[0]), float(coordinate[1])
            if not (0 <= x < self.viewport_width and 0 <= y < self.viewport_height):
                raise ValueError(f"Browser coordinate is outside the viewport: {coordinate}")
            normalized["coordinate"] = [x, y]
        if action_name == "type" and len(str(normalized.get("text") or "")) > 2000:
            raise ValueError("Browser text entry is capped at 2000 characters.")
        if action_name == "wait":
            normalized["time"] = min(10.0, max(0.0, float(normalized.get("time") or 1.0)))
        return normalized

    async def _execute_action(self, action: dict[str, Any]) -> str:
        self._check_interrupted()
        self._stop.check()
        action_name = str(action["action"])
        page = await self._active_page()
        if action_name == "mouse_move":
            await page.mouse.move(*action["coordinate"])
        elif action_name == "left_click":
            await page.mouse.click(*action["coordinate"])
        elif action_name == "double_click":
            await page.mouse.dblclick(*action["coordinate"])
        elif action_name == "right_click":
            await page.mouse.click(*action["coordinate"], button="right")
        elif action_name == "type":
            coordinate = action.get("coordinate")
            if isinstance(coordinate, list) and len(coordinate) == 2:
                await page.mouse.click(*coordinate)
            if bool(action.get("delete_existing_text")):
                await page.keyboard.press("Control+A")
                await page.keyboard.press("Backspace")
            await page.keyboard.type(str(action.get("text") or ""))
            if bool(action.get("press_enter")):
                await page.keyboard.press("Enter")
        elif action_name == "key":
            keys = action.get("keys") or action.get("text") or []
            if isinstance(keys, str):
                keys = [keys]
            if not isinstance(keys, list) or not keys:
                raise ValueError("key requires a non-empty keys list.")
            chord = "+".join(str(key) for key in keys)
            blocked = {"Control+L", "Control+O", "Control+Shift+I", "F12", "Alt+F4"}
            if chord in blocked:
                raise BrowserPolicyError(f"Keyboard chord is blocked: {chord}")
            await page.keyboard.press(chord)
        elif action_name in {"scroll", "hscroll"}:
            pixels = max(-1200.0, min(1200.0, float(action.get("pixels") or -600)))
            if action_name == "scroll":
                await page.mouse.wheel(0, -pixels)
            else:
                await page.mouse.wheel(-pixels, 0)
        elif action_name == "visit_url":
            url = self.policy.validate_navigation(str(action.get("url") or ""))
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        elif action_name == "web_search":
            query = str(action.get("query") or "").strip()
            if not query:
                raise ValueError("web_search requires a query.")
            url = self.policy.validate_navigation(self.search_url_template.format(query=quote_plus(query)))
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        elif action_name == "history_back":
            await page.go_back(wait_until="domcontentloaded", timeout=30_000)
        elif action_name == "pause_and_memorize_fact":
            fact_text = str(action.get("fact") or "").strip()
            if not fact_text:
                raise ValueError("pause_and_memorize_fact requires a fact.")
            fact: dict[str, Any] = {
                "note": fact_text,
                "url": page.url,
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "evidence_screenshot_reference": (
                    str(self._last_screenshot_path) if self._last_screenshot_path else ""
                ),
            }
            try:
                parsed = json.loads(fact_text)
                if isinstance(parsed, dict):
                    for key in ("title", "retailer", "observed_price", "currency", "similarity_or_savings_reason"):
                        if parsed.get(key) is not None:
                            fact[key] = parsed[key]
            except json.JSONDecodeError:
                pass
            if not any(item.get("url") == fact["url"] and item.get("note") == fact["note"] for item in self._facts):
                self._facts.append(fact)
            return f"Memorized research fact with exact URL: {page.url}"
        elif action_name in {"ask_user_question", "terminate"}:
            return f"Terminal action: {action_name}"
        elif action_name == "wait":
            await asyncio.sleep(float(action["time"]))
        else:
            raise ValueError(f"Unsupported browser action: {action_name}")

        if self.settle_ms:
            await page.wait_for_timeout(self.settle_ms)
        self._page = await self._active_page()
        try:
            self.policy.validate_navigation(self._page.url)
        except BrowserPolicyError:
            await self._page.go_back(wait_until="domcontentloaded", timeout=15_000)
            raise
        return f"Executed {action_name}; current URL is {self._page.url}"

    def _check_interrupted(self) -> None:
        if self.interrupt_checker is not None:
            self.interrupt_checker()

    async def _active_page(self) -> Any:
        if self._context is None:
            raise RuntimeError("Visual browser context is not running.")
        live_pages = [page for page in self._context.pages if not page.is_closed()]
        if not live_pages:
            self._page = await self._context.new_page()
        elif self._page not in live_pages:
            self._page = live_pages[-1]
        return self._page

    async def _capture(self, step: int) -> Path:
        page = await self._active_page()
        path = self.screenshot_dir / f"step-{step:04d}.png"
        await page.screenshot(path=str(path), full_page=False)
        self._screenshot_paths.append(path)
        self._last_screenshot_path = path
        return path

    @staticmethod
    def _emit(callback: Optional[Callable[[Dict[str, Any]], None]], event: Dict[str, Any]) -> None:
        if callback is None:
            return
        try:
            callback(event)
        except Exception:
            logging.getLogger("FaraVisualBrowserSession").exception("Browser event callback failed.")

    async def cleanup(self) -> None:
        self._stop.cleanup()
        context, playwright = self._context, self._playwright
        self._context = None
        self._playwright = None
        self._page = None
        if context is not None:
            try:
                await context.close()
            except Exception:
                self.logger.exception("Failed to close the Fara browser context.")
        if playwright is not None:
            try:
                await playwright.stop()
            except Exception:
                self.logger.exception("Failed to stop Playwright after Fara browser use.")
        if not self.screenshot_retention:
            for path in reversed(self._screenshot_paths):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    self.logger.warning("Could not remove browser screenshot %s", path)
            try:
                self.screenshot_dir.rmdir()
            except OSError:
                pass


class FaraVisualBrowserAdapter(BrowserToolBridgePort):
    """Factory for task-scoped, host-local Fara visual browser sessions."""

    def __init__(
        self,
        *,
        llm_provider: LLMProvider,
        profile_dir: str,
        screenshot_dir: str,
        viewport_width: int = 1440,
        viewport_height: int = 900,
        max_steps: int = 100,
        settle_ms: int = 700,
        search_url_template: str = "https://duckduckgo.com/?q={query}",
        browser_channel: str = "chromium",
        browser_executable_path: str = "",
        blocked_domains: Optional[List[str]] = None,
        blocked_path_markers: Optional[List[str]] = None,
        screenshot_retention: bool = True,
        interrupt_checker: Optional[Callable[[], None]] = None,
    ) -> None:
        self.llm = llm_provider
        self.profile_dir = Path(profile_dir)
        self.screenshot_dir = Path(screenshot_dir)
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.max_steps = max_steps
        self.settle_ms = settle_ms
        self.search_url_template = search_url_template
        self.browser_channel = browser_channel
        self.browser_executable_path = browser_executable_path
        self.policy = BrowserSafetyPolicy(
            blocked_domains=blocked_domains,
            blocked_path_markers=blocked_path_markers,
        )
        self.screenshot_retention = screenshot_retention
        self.interrupt_checker = interrupt_checker
        self.logger = logging.getLogger(self.__class__.__name__)

    async def open_session(self, *, headless: bool) -> FaraVisualBrowserSession:
        session = FaraVisualBrowserSession(
            llm_provider=self.llm,
            profile_dir=self.profile_dir,
            screenshot_dir=self.screenshot_dir,
            headless=headless,
            viewport_width=self.viewport_width,
            viewport_height=self.viewport_height,
            max_steps=self.max_steps,
            settle_ms=self.settle_ms,
            search_url_template=self.search_url_template,
            browser_channel=self.browser_channel,
            browser_executable_path=self.browser_executable_path,
            policy=self.policy,
            screenshot_retention=self.screenshot_retention,
            logger=self.logger,
            interrupt_checker=self.interrupt_checker,
        )
        await session.start()
        return session
