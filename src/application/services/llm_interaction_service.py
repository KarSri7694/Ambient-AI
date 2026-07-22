import asyncio
import copy
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from application.ports.LLMProvider import LLMProvider
from application.ports.tool_bridge_port import (
    BrowserToolBridgePort,
    BrowserToolSessionPort,
    ToolBridgePort,
)
from application.services.interaction_trace import (
    current_interaction_metadata,
    current_interaction_source,
    interaction_trace,
)
from application.services.scheduled_task_service import ScheduledTaskService
from utils.kv_state_handling import KVStateControl


@dataclass
class AgentFrame:
    messages: List[Dict[str, Any]] = field(default_factory=list)
    model: Optional[str] = None
    depth: int = 0
    tools: Optional[List[Dict[str, Any]]] = None
    tool_bridge: Optional[Any] = None
    browser_exit_requested: Optional[bool] = None


class LLMInteractionService:
    """
    Orchestrates the LLM streaming chat loop with tool execution.

    This service knows nothing about concrete LLM backends or tool systems —
    it only depends on the LLMProvider and ToolBridgePort abstractions.
    """

    AGENT_DEPTH = 0
    MAX_AGENT_DEPTH = 3
    MAX_ITERATIONS = 25
    TERMINAL_TOOL_NAMES = {"restore_previous_agent", "finish_browser_task"}
    FINISH_BROWSER_TASK_TOOL = {
        "type": "function",
        "function": {
            "name": "finish_browser_task",
            "description": (
                "Finish the delegated browser task and return control to the main model. "
                "Use exit_browser=false to leave the current browser running, including "
                "ongoing media playback; use true to close it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "exit_browser": {
                        "type": "boolean",
                        "description": "Whether to close the controlled browser before returning.",
                    },
                    "status": {
                        "type": "string",
                        "description": "Concise completion state such as completed, blocked, or failed.",
                    },
                    "summary": {
                        "type": "string",
                        "description": "Concrete result and material actions performed for the main model.",
                    },
                },
                "required": ["exit_browser", "status", "summary"],
                "additionalProperties": False,
            },
        },
    }
    PARENT_DIR = Path(__file__).parent.parent.parent.parent
    kv_state_dir = PARENT_DIR / "model_kv_states"
    kv_control = KVStateControl(kv_state_dir)

    AGENT_PROMPT = (
        "You are a deployed sub-agent working on a delegated task.\n"
        "\n"
        "Rules:\n"
        "- Complete only the delegated task.\n"
        "- Do not plan broadly, do not reframe the task, and do not restate tool inventories.\n"
        "- Do not call load_agent.\n"
        "- Do not call list_available_models.\n"
        "- Do not try to spawn another agent unless the delegated task explicitly requires it and the tool is available.\n"
        "- If the task can be completed directly from your own knowledge or the currently available context, do it directly.\n"
        "- Use other tools only if they are strictly necessary to complete the delegated task.\n"
        "- When the task is complete, immediately call restore_previous_agent exactly once.\n"
        "- In message_to_agent, return only the concrete result of the delegated task, with no extra planning.\n"
    )
    BROWSER_AGENT_PROMPT = (
        "You are a dedicated browser-control sub-agent working on one delegated task.\n"
        "\n"
        "Rules:\n"
        "- Perform only the browser task supplied by the parent agent.\n"
        "- Use the available browser tools to inspect pages before interacting with them.\n"
        "- Do not broaden the task, visit unrelated sites, or attempt to spawn another agent.\n"
        "- Do not bypass authentication, CAPTCHA, two-factor authentication, security warnings, "
        "or confirmation screens. Return a clear blocker instead.\n"
        "- When the task reaches a terminal state, call finish_browser_task exactly once.\n"
        "- Set exit_browser=false when the user needs the visible browser or media playback to "
        "remain open; otherwise set it to true.\n"
        "- In status and summary, provide the completion state, material actions performed, "
        "requested information, and any blocker.\n"
        "- Do not continue taking snapshots or other actions after the requested outcome has been verified.\n"
    )
    REPORTER_PROMPT = (
        "Create a user-facing report for an ambient agent execution.\n"
        "\n"
        "Return JSON only:\n"
        "{\n"
        '  "title": "short clear title",\n'
        '  "summary": "short summary for dashboard display",\n'
        '  "detailed_report": "highly detailed markdown-ready report that misses nothing important"\n'
        "}\n"
        "\n"
        "Rules:\n"
        "- Write for the user, not for developers.\n"
        "- summary must be concise and directly useful.\n"
        "- detailed_report must be highly detailed and miss nothing important from the task outcome.\n"
        "- Do not add any keys other than title, summary, and detailed_report.\n"
    )

    def __init__(
        self,
        llm_provider: LLMProvider,
        tool_bridge: ToolBridgePort,
        browser_tool_bridge: Optional[BrowserToolBridgePort] = None,
        browser_agent_model: Optional[str] = None,
        browser_task_timeout_seconds: float = 180.0,
        browser_headless: bool = False,
        scheduled_task_service: Optional[ScheduledTaskService] = None,
        reporter_model: Optional[str] = None,
        artifact_root: Optional[str] = None,
    ):
        self.llm = llm_provider
        self.tool_bridge = tool_bridge
        self.browser_tool_bridge = browser_tool_bridge
        self.browser_agent_model = browser_agent_model
        self.browser_task_timeout_seconds = browser_task_timeout_seconds
        self.browser_headless = browser_headless
        self.scheduled_task_service = scheduled_task_service
        self.logger = logging.getLogger(self.__class__.__name__)
        self._tools: Optional[List[Dict[str, Any]]] = None
        self._frame_stack: List[AgentFrame] = [AgentFrame(tool_bridge=tool_bridge)]
        self._browser_lock = asyncio.Lock()
        self._retained_browser_sessions: List[BrowserToolSessionPort] = []
        self.reporter_model = reporter_model
        self.artifact_root = Path(artifact_root) if artifact_root else (self.PARENT_DIR / "artifacts")
        self.artifact_root.mkdir(parents=True, exist_ok=True)

    @property
    def _frame(self) -> AgentFrame:
        return self._frame_stack[-1]

    def _push_frame(
        self,
        model: str,
        depth: int,
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_bridge: Optional[Any] = None,
    ) -> None:
        self._frame_stack.append(
            AgentFrame(
                model=model,
                depth=depth,
                tools=tools,
                tool_bridge=tool_bridge or self.tool_bridge,
            )
        )

    def _pop_frame(self) -> AgentFrame:
        if len(self._frame_stack) == 1:
            raise RuntimeError("Cannot pop root agent frame.")
        return self._frame_stack.pop()

    async def cleanup_browser_sessions(self) -> None:
        """Close browser sessions deliberately retained by finish_browser_task."""
        sessions = list(self._retained_browser_sessions)
        self._retained_browser_sessions.clear()
        for session in sessions:
            try:
                await session.cleanup()
            except Exception:
                self.logger.exception("Failed to close retained browser MCP session.")

    async def initialize_tools(self, force_refresh: bool = False) -> None:
        """Fetch available tools from the tool bridge and cache them."""
        if self._tools is not None and not force_refresh:
            return
        self._tools = await self.tool_bridge.get_all_tools()

    def reset_context(self) -> None:
        """Clear the message history for a new conversation."""
        self._frame.messages = []

    def get_context(self) -> List[Dict[str, Any]]:
        """Return a snapshot of the current message history."""
        return list(self._frame.messages)

    def restore_context(self, messages: List[Dict[str, Any]]) -> None:
        """Restore a previously saved message history."""
        self._frame.messages = list(messages)

    def restore_conversation(
        self,
        *,
        system_prompt: str,
        messages: List[Dict[str, Any]],
    ) -> None:
        """Restore a persisted conversation with a fresh dated system prompt."""
        self._frame.messages = [
            {"role": "system", "content": self._build_system_prompt(system_prompt)},
            *[dict(message) for message in messages],
        ]

    def _strip_think_tags(self, text: str) -> str:
        """Remove leaked Qwen/llama.cpp thinking tags from assistant text."""
        if not text:
            return ""
        text = re.sub(r"^</think>\s*", "", text)
        return re.sub(r"<think>[\s\S]*?</think>", "", text)

    def _parse_qwen_xml_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """Recover tool calls when the backend leaves Qwen XML in plain text."""
        recovered_calls: List[Dict[str, Any]] = []
        for match in re.finditer(r"<function=([\w.-]+)>([\s\S]*?)</function>", text or ""):
            tool_name = match.group(1).strip()
            raw_params = match.group(2)
            args: Dict[str, Any] = {}
            for param_match in re.finditer(
                r"<parameter=([\w.-]+)>([\s\S]*?)</parameter>",
                raw_params,
            ):
                key = param_match.group(1).strip()
                raw_value = param_match.group(2).strip()
                try:
                    value = json.loads(raw_value)
                except json.JSONDecodeError:
                    value = raw_value
                args[key] = value
            recovered_calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(args),
                    },
                }
            )
        return recovered_calls

    def _remove_qwen_xml_tool_calls(self, text: str) -> str:
        """Strip recovered XML tool-call blobs from assistant text before storing it."""
        if not text:
            return ""
        stripped = re.sub(r"<function=[\w.-]+>[\s\S]*?</function>", "", text)
        return stripped.strip()

    def _get_available_model_names(self) -> List[str]:
        """Read the local model registry used to constrain load_agent."""
        details_path = self.PARENT_DIR / "model_details.csv"
        if not details_path.exists():
            return []

        model_names: List[str] = []
        with open(details_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                name = line.split(":", 1)[0].strip()
                if name:
                    model_names.append(name)
        return model_names

    def _tools_for_agent_depth(
        self,
        agent_depth: int,
        allowed_tool_names: Optional[set[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Return the tool set allowed for the current agent depth."""
        available_tools = self._frame.tools if self._frame.tools is not None else self._tools
        if not available_tools:
            return []
        if agent_depth < self.MAX_AGENT_DEPTH:
            tools = list(available_tools)
        else:
            filtered_tools: List[Dict[str, Any]] = []
            for tool in available_tools:
                function_meta = tool.get("function", {})
                tool_name = function_meta.get("name")
                if tool_name in {"load_agent", "list_available_models"}:
                    continue
                filtered_tools.append(tool)
            tools = filtered_tools

        if agent_depth > 0:
            tools = [
                tool
                for tool in tools
                if tool.get("function", {}).get("name") != "use_browser"
            ]

        if allowed_tool_names is None:
            return tools

        filtered_tools: List[Dict[str, Any]] = []
        for tool in tools:
            function_meta = tool.get("function", {})
            tool_name = function_meta.get("name")
            if tool_name in allowed_tool_names:
                filtered_tools.append(tool)
        return filtered_tools

    async def _run_browser_agent(
        self,
        *,
        task: str,
        agent_depth: int,
    ) -> str:
        if agent_depth != 0:
            raise RuntimeError("use_browser can only be called by the root agent.")
        if not task.strip():
            raise ValueError("use_browser requires a non-empty task.")
        if self.browser_tool_bridge is None:
            raise RuntimeError("Browser MCP delegation is not configured.")
        if not self.browser_agent_model:
            raise RuntimeError("No browser model is configured.")

        async with self._browser_lock:
            parent_model_name = self.llm.get_current_model() or self._frame.model
            if not parent_model_name:
                raise RuntimeError("Cannot determine the parent model before browser delegation.")

            saved_parent_state = await self.llm.save_and_unload(self._frame.messages)
            if saved_parent_state is None:
                raise RuntimeError("Could not save the parent model state; browser task was not started.")

            browser_session: Optional[BrowserToolSessionPort] = None
            child_frame_pushed = False
            exit_browser = True
            browser_result = ""
            primary_error: Optional[BaseException] = None
            try:
                if self._retained_browser_sessions:
                    browser_session = self._retained_browser_sessions.pop()
                    self.logger.info("Reusing retained browser MCP session.")
                else:
                    browser_session = await self.browser_tool_bridge.open_session(
                        headless=self.browser_headless
                    )
                browser_tools = await browser_session.get_all_tools()
                if not browser_tools:
                    raise RuntimeError("The browser MCP server did not expose any safe tools.")
                browser_tools = [
                    *browser_tools,
                    copy.deepcopy(self.FINISH_BROWSER_TASK_TOOL),
                ]

                await self.llm.load_model(self.browser_agent_model)
                self._push_frame(
                    model=self.browser_agent_model,
                    depth=agent_depth + 1,
                    tools=browser_tools,
                    tool_bridge=browser_session,
                )
                child_frame_pushed = True
                allowed_tool_names = {
                    tool.get("function", {}).get("name")
                    for tool in browser_tools
                    if tool.get("function", {}).get("name")
                }
                browser_result = await asyncio.wait_for(
                    self.run_interaction(
                        user_input="You have been given this browser task:\n" + task.strip(),
                        system_prompt=self.BROWSER_AGENT_PROMPT,
                        model=self.browser_agent_model,
                        agent_depth=agent_depth + 1,
                        allowed_tool_names=allowed_tool_names,
                        report_policy="silent",
                    ),
                    timeout=self.browser_task_timeout_seconds,
                )
            except BaseException as exc:
                primary_error = exc
            finally:
                if child_frame_pushed:
                    exit_browser = self._frame.browser_exit_requested is not False
                    self._pop_frame()

                cleanup_error: Optional[BaseException] = None
                if browser_session is not None:
                    if primary_error is None and not exit_browser:
                        self._retained_browser_sessions.append(browser_session)
                        self.logger.info(
                            "Browser agent returned control while leaving the browser session open."
                        )
                    else:
                        try:
                            await browser_session.cleanup()
                        except BaseException as exc:
                            cleanup_error = exc
                            self.logger.exception("Failed to close browser MCP session.")

                restore_error: Optional[BaseException] = None
                try:
                    current_model_name = self.llm.get_current_model()
                    if current_model_name and current_model_name != parent_model_name:
                        await self.llm.unload_model()
                    await self.llm.load_and_restore()
                except BaseException as exc:
                    restore_error = exc
                    self.logger.exception("Failed to restore parent model after browser delegation.")

                if restore_error is not None:
                    if primary_error is not None:
                        raise RuntimeError(
                            f"Browser task failed ({primary_error}) and parent restoration also failed "
                            f"({restore_error})."
                        ) from restore_error
                    raise RuntimeError(
                        f"Browser task completed but parent restoration failed: {restore_error}"
                    ) from restore_error
                if primary_error is not None:
                    raise primary_error
                if cleanup_error is not None:
                    raise RuntimeError(
                        f"Browser task completed but browser cleanup failed: {cleanup_error}"
                    ) from cleanup_error

            return browser_result

    def _build_system_prompt(self, system_prompt: str) -> str:
        now = datetime.now()
        preamble = (
            f"Current day of week: {now.strftime('%A')}\n"
            f"Current date: {now.strftime('%Y-%m-%d')}\n"
            f"Current time: {now.strftime('%H:%M:%S')}\n\n"
        )
        return preamble + system_prompt

    def _emit_event(
        self,
        event_callback: Optional[Callable[[Dict[str, Any]], None]],
        event: Dict[str, Any],
    ) -> None:
        if event_callback is None:
            return
        try:
            event_callback(event)
        except Exception:
            self.logger.exception("Interaction event callback failed.")

    async def run_interaction(
        self,
        user_input: str,
        system_prompt: str,
        model: str,
        image_path: str = "",
        agent_depth: int = 0,
        allowed_tool_names: Optional[set[str]] = None,
        report_policy: str = "silent",
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> str:
        """
        Run a full LLM interaction: send user input, stream response,
        execute any tool calls, loop until the model is done.

        Returns the final assistant text response.
        """
        current_source = current_interaction_source()
        source_name = current_source if current_source != "unknown" else "ambient_execution"
        existing_metadata = current_interaction_metadata()
        interaction_run_id = existing_metadata.get("interaction_run_id") or uuid.uuid4().hex
        tools_used: List[str] = []

        with interaction_trace(source_name, {"interaction_run_id": interaction_run_id}):
            self._frame.model = model
            self._frame.depth = agent_depth
            if not self._frame.messages:
                self._frame.messages.append(
                    {"role": "system", "content": self._build_system_prompt(system_prompt)}
                )
            self._frame.messages.append({"role": "user", "content": user_input})

            iteration = 0
            assistant_text = ""

            while iteration < self.MAX_ITERATIONS:
                iteration += 1
                self.logger.info(
                    "--- Iteration %s (agent depth %s/%s) ---",
                    iteration,
                    agent_depth,
                    self.MAX_AGENT_DEPTH,
                )

                completion = await self.llm.chat_completion_stream(
                    model=model,
                    messages=self._frame.messages,
                    tools=self._tools_for_agent_depth(
                        agent_depth, allowed_tool_names=allowed_tool_names
                    ),
                    image=image_path if iteration == 1 else "",
                )

                assistant_text, tool_calls = await self._consume_stream(
                    completion,
                    event_callback=event_callback,
                )

                if not tool_calls:
                    self._frame.messages.append({"role": "assistant", "content": assistant_text})
                    self.logger.info("Model finished (no more tool calls)")
                    await self._attach_user_report(
                        report_policy=report_policy,
                        interaction_run_id=interaction_run_id,
                        model=model,
                        user_input=user_input,
                        final_response=assistant_text,
                        tools_used=tools_used,
                        source_name=source_name,
                    )
                    return assistant_text

                self._frame.messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_text if assistant_text else None,
                        "tool_calls": tool_calls,
                    }
                )

                tool_results = await self._execute_tool_calls(
                    tool_calls,
                    agent_depth=agent_depth,
                    allowed_tool_names=allowed_tool_names,
                    event_callback=event_callback,
                )
                tools_used.extend(name for name, _ in tool_results)
                if any(
                    name in self.TERMINAL_TOOL_NAMES and not result.startswith("Error:")
                    for name, result in tool_results
                ):
                    self.logger.info("Terminal tool executed; ending interaction loop.")
                    terminal_result = "\n".join(result for _, result in tool_results)
                    await self._attach_user_report(
                        report_policy=report_policy,
                        interaction_run_id=interaction_run_id,
                        model=model,
                        user_input=user_input,
                        final_response=terminal_result,
                        tools_used=tools_used,
                        source_name=source_name,
                    )
                    return terminal_result

            self.logger.warning(
                "Reached maximum iterations: (%s). Stopping.",
                self.MAX_ITERATIONS,
            )
            await self._attach_user_report(
                report_policy=report_policy,
                interaction_run_id=interaction_run_id,
                model=model,
                user_input=user_input,
                final_response=assistant_text,
                tools_used=tools_used,
                source_name=source_name,
            )
            return assistant_text

    async def _consume_stream(
        self,
        completion,
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> tuple[str, List[Dict]]:
        """
        Consume a streaming completion iterator.
        Returns (assistant_text, tool_calls_list).
        """
        assistant_text = ""
        raw_assistant_text = ""
        raw_reasoning_text = ""
        tool_calls: List[Dict] = []

        async for chunk in completion:
            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta

            if delta.content:
                raw_assistant_text += delta.content
                content = self._strip_think_tags(delta.content)
                if content:
                    assistant_text += content
                    self._emit_event(
                        event_callback,
                        {"type": "delta", "content": content},
                    )

            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                raw_reasoning_text += reasoning

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    index = tc_delta.index
                    while len(tool_calls) <= index:
                        tool_calls.append(
                            {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        )
                    if tc_delta.id:
                        tool_calls[index]["id"] = tc_delta.id
                    if tc_delta.function.name:
                        tool_calls[index]["function"]["name"] += tc_delta.function.name
                    if tc_delta.function.arguments:
                        tool_calls[index]["function"]["arguments"] += tc_delta.function.arguments

        assistant_text = self._strip_think_tags(assistant_text).strip()
        if not tool_calls:
            raw_tool_source = "\n".join(
                part for part in [raw_assistant_text, raw_reasoning_text] if part
            )
            tool_calls = self._parse_qwen_xml_tool_calls(raw_tool_source)
            if tool_calls:
                self.logger.info(
                    "Recovered %s XML tool call(s) from raw stream fallback.",
                    len(tool_calls),
                )
        assistant_text = self._remove_qwen_xml_tool_calls(assistant_text).strip()
        return assistant_text, tool_calls

    async def _execute_tool_calls(
        self,
        tool_calls: List[Dict],
        agent_depth: int = 0,
        allowed_tool_names: Optional[set[str]] = None,
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> List[tuple[str, str]]:
        """
        Execute tool calls, append results to history, and return the tool outputs.
        Returns:
            Tuple of (tool_name, tool_result) for each executed tool.
        """
        tool_results: List[tuple[str, str]] = []
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            tool_args_str = tool_call["function"]["arguments"]
            tool_id = tool_call["id"]

            self.logger.info("Calling tool: %s", tool_name)
            self.logger.info("Arguments: %s", tool_args_str)
            self._emit_event(
                event_callback,
                {"type": "tool_started", "tool_name": tool_name},
            )

            try:
                tool_args = json.loads(tool_args_str) if tool_args_str else {}
                if allowed_tool_names is not None and tool_name not in allowed_tool_names:
                    response_content = (
                        f"Error: tool '{tool_name}' is not allowed in this interaction."
                    )
                    self.logger.warning(response_content)
                    tool_results.append((tool_name, response_content))
                    self._frame.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "name": tool_name,
                            "content": response_content,
                        }
                    )
                    continue
                if tool_name == "use_browser":
                    task = tool_args.get("task", "")
                    response_content = await self._run_browser_agent(
                        task=str(task),
                        agent_depth=agent_depth,
                    )
                elif tool_name == "finish_browser_task":
                    if agent_depth == 0:
                        raise RuntimeError(
                            "finish_browser_task can only be called by the delegated browser agent."
                        )
                    exit_browser = tool_args.get("exit_browser")
                    status = tool_args.get("status")
                    summary = tool_args.get("summary")
                    if not isinstance(exit_browser, bool):
                        raise ValueError("finish_browser_task exit_browser must be a boolean.")
                    if not isinstance(status, str) or not status.strip():
                        raise ValueError("finish_browser_task status must be a non-empty string.")
                    if not isinstance(summary, str) or not summary.strip():
                        raise ValueError("finish_browser_task summary must be a non-empty string.")
                    self._frame.browser_exit_requested = exit_browser
                    response_content = json.dumps(
                        {
                            "status": status.strip(),
                            "summary": summary.strip(),
                            "browser_exited": exit_browser,
                        },
                        ensure_ascii=False,
                    )
                elif tool_name == "schedule_task_at":
                    if self.scheduled_task_service is None:
                        raise RuntimeError("Exact-time task scheduling is not configured.")
                    metadata = current_interaction_metadata()
                    scheduled = self.scheduled_task_service.schedule(
                        task=str(tool_args.get("task", "")),
                        run_at=str(tool_args.get("run_at", "")),
                        priority=str(tool_args.get("priority", "medium")),
                        metadata={
                            "source": current_interaction_source(),
                            "origin_chat_session_id": metadata.get("chat_session_id"),
                            "origin_chat_message_id": metadata.get("chat_message_id"),
                        },
                    )
                    response_content = json.dumps(scheduled, ensure_ascii=False)
                elif tool_name == "load_agent":
                    if agent_depth >= self.MAX_AGENT_DEPTH:
                        response_content = (
                            f"Error: maximum agent depth reached ({self.MAX_AGENT_DEPTH}). "
                            "Further agent spawning is not allowed."
                        )
                        self.logger.warning(response_content)
                        tool_results.append((tool_name, response_content))
                        self._frame.messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_id,
                                "name": tool_name,
                                "content": response_content,
                            }
                        )
                        continue
                    parent_model_name = self.llm.get_current_model() or self._frame.model
                    model_name = tool_args.get("model_name")
                    available_model_names = self._get_available_model_names()
                    if not model_name:
                        response_content = (
                            "Error: load_agent requires model_name. "
                            "Call list_available_models first and use an exact model name."
                        )
                        self.logger.warning(response_content)
                        tool_results.append((tool_name, response_content))
                        self._frame.messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_id,
                                "name": tool_name,
                                "content": response_content,
                            }
                        )
                        continue
                    if available_model_names and model_name not in available_model_names:
                        response_content = (
                            "Error: unknown model_name "
                            f"'{model_name}'. Call list_available_models first and use one "
                            f"of: {', '.join(available_model_names)}"
                        )
                        self.logger.warning(response_content)
                        tool_results.append((tool_name, response_content))
                        self._frame.messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_id,
                                "name": tool_name,
                                "content": response_content,
                            }
                        )
                        continue
                    saved_parent_kv_state = await self.llm.save_and_unload(self._frame.messages)
                    await self.llm.load_model(model_name)
                    self._push_frame(model=model_name, depth=agent_depth + 1)
                    try:
                        child_result = await self.run_interaction(
                            user_input="You have been given a task: " + tool_args.get("message", ""),
                            system_prompt=self.AGENT_PROMPT,
                            model=model_name,
                            agent_depth=agent_depth + 1,
                            allowed_tool_names=allowed_tool_names,
                            report_policy="silent",
                        )
                    finally:
                        self._pop_frame()

                    current_model_name = self.llm.get_current_model()
                    if parent_model_name and current_model_name != parent_model_name:
                        self.logger.info(
                            "Sub-agent returned without restoring parent state; recovering parent model %s.",
                            parent_model_name,
                        )
                        if current_model_name:
                            await self.llm.unload_model()
                        if saved_parent_kv_state is not None:
                            await self.llm.load_and_restore()
                        else:
                            await self.llm.load_model(parent_model_name)
                    response_content = child_result
                elif tool_name == "restore_previous_agent":
                    response_content = "Agent task completed. Restoring previous agent."
                    kv_state_file = await self.llm.load_and_restore()
                    messages_path = kv_state_file.with_suffix(".json")
                    if messages_path.exists():
                        with open(messages_path, "r", encoding="utf-8") as f:
                            messages = json.load(f)
                        target_frame = (
                            self._frame_stack[-2] if len(self._frame_stack) > 1 else self._frame
                        )
                        target_frame.messages = list(messages)
                        target_frame.messages.append(
                            {
                                "role": "assistant",
                                "content": "[Sub-Agent] Deployed sub-agent has worked on the given task and returned the following result: "
                                + tool_args.get("message_to_agent", ""),
                            }
                        )
                        self.logger.info("Conversation state restored from %s", messages_path.name)
                    else:
                        self.logger.warning(
                            "No conversation state file found at %s. Context not restored.",
                            messages_path,
                        )
                else:
                    active_tool_bridge = self._frame.tool_bridge or self.tool_bridge
                    response_content = await active_tool_bridge.execute_tool(tool_name, tool_args)
            except Exception as e:
                response_content = f"Error: {str(e)}"
            self._emit_event(
                event_callback,
                {
                    "type": "tool_finished",
                    "tool_name": tool_name,
                    "ok": not response_content.startswith("Error:"),
                },
            )
            tool_results.append((tool_name, response_content))

            self._frame.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "name": tool_name,
                    "content": response_content,
                }
            )
        return tool_results

    async def _attach_user_report(
        self,
        *,
        report_policy: str,
        interaction_run_id: str,
        model: str,
        user_input: str,
        final_response: str,
        tools_used: List[str],
        source_name: str,
    ) -> None:
        if report_policy != "auto_surface":
            return
        report = await self._build_user_report(
            model=model,
            user_input=user_input,
            final_response=final_response,
            tools_used=tools_used,
            source_name=source_name,
        )
        if not report:
            return
        if hasattr(self.llm, "attach_report"):
            self.llm.attach_report(interaction_run_id, report)

    async def _build_user_report(
        self,
        *,
        model: str,
        user_input: str,
        final_response: str,
        tools_used: List[str],
        source_name: str,
    ) -> Optional[Dict[str, Any]]:
        report_model = self.reporter_model or model
        deduped_tools = list(dict.fromkeys(tools_used))
        if report_model != model:
            report_text = await self._run_history_report_prompt(
                model=report_model,
                task_model=model,
                source_name=source_name,
                tool_names=deduped_tools,
            )
        else:
            payload = {
                "task_brief": user_input,
                "final_response": final_response,
                "tools_used": deduped_tools,
                "source": source_name,
            }
            report_text = await self._run_json_prompt(
                model=report_model,
                system_prompt=self.REPORTER_PROMPT,
                user_payload=payload,
            )
        parsed = self._safe_parse_json(report_text)
        if not isinstance(parsed, dict):
            return None
        title = str(parsed.get("title") or "").strip()
        summary = str(parsed.get("summary") or "").strip()
        detailed_report = str(parsed.get("detailed_report") or "").strip()
        if not title or not summary or not detailed_report:
            return None
        artifact_path = self._save_report_artifact(
            title=title,
            summary=summary,
            detailed_report=detailed_report,
        )
        report = {
            "title": title,
            "summary": summary,
            "artifact_path": str(artifact_path),
            "artifact_filename": artifact_path.name,
            "source": source_name,
            "tools_used": deduped_tools,
            "created_at": datetime.now().isoformat(),
            "status": "completed",
            "task_model": model,
            "report_model": report_model,
        }
        return report

    async def _run_json_prompt(
        self,
        *,
        model: str,
        system_prompt: str,
        user_payload: Dict[str, Any],
    ) -> str:
        provider = getattr(self.llm, "provider", self.llm)
        completion = await provider.chat_completion_stream(
            model=model,
            messages=[
                {"role": "system", "content": self._build_system_prompt(system_prompt)},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
            ],
            tools=None,
            image="",
        )
        text_parts: List[str] = []
        async for chunk in completion:
            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta
            if getattr(delta, "content", None):
                text_parts.append(delta.content)
        return "".join(text_parts).strip()

    async def _run_history_report_prompt(
        self,
        *,
        model: str,
        task_model: str,
        source_name: str,
        tool_names: List[str],
    ) -> str:
        provider = getattr(self.llm, "provider", self.llm)
        messages = [
            {"role": "system", "content": self._build_system_prompt(self.REPORTER_PROMPT)},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "report_request": {
                            "source": source_name,
                            "task_model": task_model,
                            "report_model": model,
                            "tools_used": tool_names,
                        },
                        "interaction_history": copy.deepcopy(self._frame.messages),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            },
        ]
        completion = await provider.chat_completion_stream(
            model=model,
            messages=messages,
            tools=None,
            image="",
        )
        text_parts: List[str] = []
        async for chunk in completion:
            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta
            if getattr(delta, "content", None):
                text_parts.append(delta.content)
        return "".join(text_parts).strip()

    def _save_report_artifact(self, *, title: str, summary: str, detailed_report: str) -> Path:
        safe_title = self._sanitize_artifact_name(title)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = self.artifact_root / f"{safe_title}_{timestamp}.md"
        suffix = 1
        while candidate.exists():
            candidate = self.artifact_root / f"{safe_title}_{timestamp}_{suffix}.md"
            suffix += 1
        content = "\n".join(
            [
                f"# {title}",
                "",
                "## Summary",
                summary,
                "",
                "## Detailed Report",
                detailed_report,
                "",
            ]
        )
        candidate.write_text(content, encoding="utf-8")
        return candidate

    def _sanitize_artifact_name(self, title: str) -> str:
        cleaned = re.sub(r"[^\w\s-]", "", title, flags=re.UNICODE)
        cleaned = re.sub(r"\s+", "_", cleaned.strip())
        return cleaned[:80] or "report"

    def _safe_parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        candidate = text.strip()
        if candidate.startswith("```"):
            candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
            candidate = re.sub(r"\s*```$", "", candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", candidate)
            if not match:
                return None
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
