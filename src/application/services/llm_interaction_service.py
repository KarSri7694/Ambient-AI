import asyncio
import copy
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
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
from application.services.recurring_task_service import RecurringTaskService
from application.services.capability_policy_service import (
    ApprovalRequiredError,
    CapabilityPolicyService,
    PolicyDeniedError,
)
from application.services.artifact_organizer_service import ArtifactOrganizer
from application.services.runtime_interrupt_service import WorkInterrupted
from core.models import ApprovalGrant, DelegatedTask
from local_control.computer import ComputerControlSession
from local_control.filesystem import FilesystemControlSession
from local_control.safety import ComputerControlTerminated
from utils.kv_state_handling import KVStateControl


@dataclass
class AgentFrame:
    messages: List[Dict[str, Any]] = field(default_factory=list)
    model: Optional[str] = None
    depth: int = 0
    tools: Optional[List[Dict[str, Any]]] = None
    tool_bridge: Optional[Any] = None
    browser_exit_requested: Optional[bool] = None
    computer_exit_requested: Optional[bool] = None
    delegated_approval_id: Optional[str] = None
    preauthorized_tool_names: set[str] = field(default_factory=set)


class InteractionSuspended(Exception):
    """Signals that an interaction is durably waiting for local approval."""

    def __init__(
        self,
        *,
        approval: ApprovalGrant,
        delegated_task: DelegatedTask,
        tool_call_id: str = "",
    ):
        super().__init__(f"Interaction is awaiting approval {approval.approval_id}.")
        self.approval = approval
        self.delegated_task = delegated_task
        self.tool_call_id = tool_call_id

    @property
    def approval_id(self) -> str:
        return self.approval.approval_id

    @property
    def delegation_id(self) -> str:
        return self.delegated_task.delegation_id


class LLMInteractionService:
    """
    Orchestrates the LLM streaming chat loop with tool execution.

    This service knows nothing about concrete LLM backends or tool systems —
    it only depends on the LLMProvider and ToolBridgePort abstractions.
    """

    AGENT_DEPTH = 0
    MAX_AGENT_DEPTH = 3
    MAX_ITERATIONS = 25
    TERMINAL_TOOL_NAMES = {
        "restore_previous_agent",
        "finish_browser_task",
        "finish_filesystem_task",
        "finish_computer_task",
    }
    LOCAL_CONTROL_REQUEST_TOOLS = {"use_browser", "request_computer_use"}
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
                    "details": {"type": "string"},
                    "actions_performed": {"type": "array", "items": {"type": "string"}},
                    "sources": {"type": "array", "items": {"type": "string"}},
                    "blockers": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["exit_browser", "status", "summary"],
                "additionalProperties": False,
            },
        },
    }
    FINISH_FILESYSTEM_TASK_TOOL = {
        "type": "function",
        "function": {
            "name": "finish_filesystem_task",
            "description": "Finish the delegated filesystem task and return control to the main model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "summary": {"type": "string"},
                    "details": {"type": "string"},
                    "actions_performed": {"type": "array", "items": {"type": "string"}},
                    "sources": {"type": "array", "items": {"type": "string"}},
                    "blockers": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["status", "summary"],
                "additionalProperties": False,
            },
        },
    }
    FINISH_COMPUTER_TASK_TOOL = {
        "type": "function",
        "function": {
            "name": "finish_computer_task",
            "description": "Finish the delegated computer-use task and return control to Ambient AI.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "summary": {"type": "string"},
                    "details": {"type": "string"},
                    "actions_performed": {"type": "array", "items": {"type": "string"}},
                    "sources": {"type": "array", "items": {"type": "string"}},
                    "blockers": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["status", "summary"],
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
        '  "substantive_outcome": true,\n'
        '  "title": "short clear title",\n'
        '  "summary": "short summary for dashboard display",\n'
        '  "detailed_report": "highly detailed markdown-ready report that misses nothing important"\n'
        "}\n"
        "\n"
        "Rules:\n"
        "- Write for the user, not for developers.\n"
        "- substantive_outcome is false for routine observation, duplicated work, or a result that adds no durable value.\n"
        "- Only mark it true for a verified recommendation, integration assessment, decision record, comparison, implementation plan, or useful draft.\n"
        "- summary must be concise and directly useful.\n"
        "- detailed_report must be highly detailed and miss nothing important from the task outcome.\n"
        "- Do not add any keys other than substantive_outcome, title, summary, and detailed_report.\n"
    )
    FILESYSTEM_AGENT_PROMPT = (
        "You are a dedicated read-only filesystem sub-agent working on one delegated task.\n"
        "\n"
        "Rules:\n"
        "- Operate only inside the user-granted paths available to your tools.\n"
        "- Filesystem tools require absolute paths. Start from the resolved granted roots in the task message; do not use '.', '/', '~', '/home', or guessed Linux paths.\n"
        "- Prefer fs_search_text on a resolved granted root when the task asks to find files or text recursively.\n"
        "- Do not request shell commands, deletion, overwrites, moves, renames, chmod, or hidden path expansion.\n"
        "- Prefer listing and stat before reading unfamiliar files.\n"
        "- Stop when the requested information has been found or the grant is insufficient.\n"
        "- When the task reaches a terminal state, call finish_filesystem_task exactly once.\n"
        "- In status and summary, report concrete files inspected and any blocker.\n"
    )
    COMPUTER_AGENT_PROMPT = (
        "You are a dedicated computer-use sub-agent working on one user-approved desktop-control task.\n"
        "\n"
        "Rules:\n"
        "- Perform only the approved task supplied by Ambient AI.\n"
        "- You receive a fresh full-screen screenshot before every model turn. Treat that screenshot as the current desktop state.\n"
        "- After each action, wait for the next turn's screenshot before deciding the next action.\n"
        "- Mouse coordinates must use Gemma-style normalized 0..1000 coordinates, where (0,0) is top-left and (1000,1000) is bottom-right.\n"
        "- computer_inspect only reports observation metadata; do not rely on UI Automation or accessibility trees.\n"
        "- Do not use shell commands, system shutdown/logout/lock, credential entry, payment, checkout, "
        "or destructive file-manager actions.\n"
        "- Do not bypass authentication, CAPTCHA, two-factor authentication, security warnings, or confirmation screens.\n"
        "- If Shift+Esc terminates your session, stop immediately.\n"
        "- When the task reaches a terminal state, call finish_computer_task exactly once.\n"
        "- In status and summary, provide the completion state, material actions performed, and any blocker.\n"
    )
    ARTIFACT_ORGANIZER_PROMPT = (
        "You organize Ambient AI artifacts for the user.\n\n"
        "Given one new report and candidate existing artifacts, decide whether to merge into an existing artifact "
        "or create a new artifact.\n\n"
        "Return JSON only with exactly these keys:\n"
        "{\n"
        '  "action": "merge_existing or create_new",\n'
        '  "target_artifact_id": "existing artifact id, or empty string for create_new",\n'
        '  "final_title": "best artifact title",\n'
        '  "updated_short_summary": "short summary of the full artifact",\n'
        '  "updated_detailed_summary": "detailed summary of the full artifact",\n'
        '  "merged_content": "complete markdown body for the artifact content section",\n'
        '  "dedupe_notes": ["specific duplicated facts skipped"],\n'
        '  "reason": "brief reason for the selected artifact or new artifact"\n'
        "}\n\n"
        "Rules:\n"
        "- Merge only when the existing artifact is clearly about the same topic, lecture, project, person, or workflow.\n"
        "- Remove duplicate information; preserve prior useful details.\n"
        "- If merging, merged_content must contain the full updated artifact body, not just the new addition.\n"
        "- If no candidate is a good match, use action=create_new and leave target_artifact_id empty.\n"
        "- Do not invent facts not present in the new report or existing artifact content.\n"
    )

    def __init__(
        self,
        llm_provider: LLMProvider,
        tool_bridge: ToolBridgePort,
        browser_tool_bridge: Optional[BrowserToolBridgePort] = None,
        browser_agent_model: Optional[str] = None,
        browser_task_timeout_seconds: float = 180.0,
        browser_headless: bool = False,
        filesystem_agent_model: Optional[str] = None,
        filesystem_task_timeout_seconds: float = 120.0,
        filesystem_max_read_bytes: int = 256_000,
        filesystem_max_list_entries: int = 200,
        computer_agent_model: Optional[str] = None,
        computer_agent_family: str = "gemma",
        computer_task_timeout_seconds: float = 180.0,
        computer_max_actions_per_task: int = 40,
        computer_screenshot_dir: str = ".ambient_data/computer/screenshots",
        computer_enabled: bool = False,
        local_control_approval_ttl_minutes: int = 30,
        scheduled_task_service: Optional[ScheduledTaskService] = None,
        recurring_task_service: Optional[RecurringTaskService] = None,
        reporter_model: Optional[str] = None,
        artifact_root: Optional[str] = None,
        capability_policy: Optional[CapabilityPolicyService] = None,
        artifact_organizer_enabled: bool = False,
        artifact_candidate_summary_words: int = 50,
        artifact_candidate_limit: int = 8,
        artifact_full_candidate_limit: int = 3,
        artifact_max_existing_chars: int = 50_000,
        semantic_memory: Optional[Any] = None,
        temporal_memory_service: Optional[Any] = None,
        interrupt_checker: Optional[Callable[[], None]] = None,
        shutdown_controller: Optional[Any] = None,
        max_interaction_iterations: int = MAX_ITERATIONS,
        final_turn_recovery_enabled: bool = True,
    ):
        self.llm = llm_provider
        self.tool_bridge = tool_bridge
        self.browser_tool_bridge = browser_tool_bridge
        self.browser_agent_model = browser_agent_model
        self.browser_task_timeout_seconds = browser_task_timeout_seconds
        self.browser_headless = browser_headless
        self.filesystem_agent_model = filesystem_agent_model
        self.filesystem_task_timeout_seconds = filesystem_task_timeout_seconds
        self.filesystem_max_read_bytes = filesystem_max_read_bytes
        self.filesystem_max_list_entries = filesystem_max_list_entries
        self.computer_agent_model = computer_agent_model
        self.computer_agent_family = (computer_agent_family or "gemma").strip().lower()
        self.computer_task_timeout_seconds = computer_task_timeout_seconds
        self.computer_max_actions_per_task = computer_max_actions_per_task
        self.computer_screenshot_dir = computer_screenshot_dir
        self.computer_enabled = computer_enabled
        self.local_control_approval_ttl_minutes = max(1, int(local_control_approval_ttl_minutes))
        self.scheduled_task_service = scheduled_task_service
        self.recurring_task_service = recurring_task_service
        self.logger = logging.getLogger(self.__class__.__name__)
        self._tools: Optional[List[Dict[str, Any]]] = None
        self._frame_stack: List[AgentFrame] = [AgentFrame(tool_bridge=tool_bridge)]
        self._browser_lock = asyncio.Lock()
        self._filesystem_lock = asyncio.Lock()
        self._computer_lock = asyncio.Lock()
        self._retained_browser_sessions: List[BrowserToolSessionPort] = []
        self.reporter_model = reporter_model
        self.capability_policy = capability_policy
        self.semantic_memory = semantic_memory
        self.temporal_memory_service = temporal_memory_service
        self.interrupt_checker = interrupt_checker
        self.shutdown_controller = shutdown_controller
        self.max_interaction_iterations = max(1, int(max_interaction_iterations or self.MAX_ITERATIONS))
        self.final_turn_recovery_enabled = bool(final_turn_recovery_enabled)
        self.artifact_root = Path(artifact_root) if artifact_root else (self.PARENT_DIR / "artifacts")
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.artifact_organizer = (
            ArtifactOrganizer(
                self.artifact_root,
                candidate_summary_words=artifact_candidate_summary_words,
                candidate_limit=artifact_candidate_limit,
                full_candidate_limit=artifact_full_candidate_limit,
                max_existing_artifact_chars=artifact_max_existing_chars,
                semantic_memory=semantic_memory,
            )
            if artifact_organizer_enabled
            else None
        )

    @property
    def _frame(self) -> AgentFrame:
        return self._frame_stack[-1]

    def _check_interrupted(self) -> None:
        if self.interrupt_checker is not None:
            self.interrupt_checker()

    def _push_frame(
        self,
        model: str,
        depth: int,
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_bridge: Optional[Any] = None,
        delegated_approval_id: Optional[str] = None,
        preauthorized_tool_names: Optional[set[str]] = None,
    ) -> None:
        self._frame_stack.append(
            AgentFrame(
                model=model,
                depth=depth,
                tools=tools,
                tool_bridge=tool_bridge or self.tool_bridge,
                delegated_approval_id=str(delegated_approval_id or "").strip() or None,
                preauthorized_tool_names=set(preauthorized_tool_names or set()),
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

    def fork_for_parallel_interaction(self) -> "LLMInteractionService":
        """Create an isolated message/frame state sharing the same providers/tools.

        A direct chat stream must never mutate the frame used by ambient
        background work. Provider, tool bridge, policy, memory, and artifact
        services remain shared; only per-interaction state and local-control
        locks are isolated.
        """
        child = copy.copy(self)
        child._frame_stack = [AgentFrame(tool_bridge=self.tool_bridge)]
        child._browser_lock = asyncio.Lock()
        child._filesystem_lock = asyncio.Lock()
        child._computer_lock = asyncio.Lock()
        child._retained_browser_sessions = []
        return child

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
                if tool.get("function", {}).get("name")
                not in {"use_browser", "use_filesystem", "request_computer_use"}
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

    def available_tool_definitions(self) -> List[Dict[str, Any]]:
        """Return a defensive copy of the currently initialized tool surface."""
        return copy.deepcopy(self._tools or [])

    def _default_final_turn_instruction(self, response_format: str) -> str:
        response_format = (response_format or "plain").strip().lower()
        if response_format == "json":
            return (
                "This is the final allowed model turn for this tool loop. Do not call tools. "
                "Return valid JSON only using the tool results and messages already in this conversation. "
                "If the available evidence is incomplete, return the required JSON shape with no findings "
                "or with a blocker/error item rather than inventing facts."
            )
        if response_format == "browser_result":
            return (
                "This is the final allowed model turn for this browser task. Do not call tools. "
                "Return a concise browser task result now: status, concrete summary, useful facts or links found, "
                "material actions taken, blockers, and whether the user needs to act next."
            )
        if response_format == "computer_result":
            return (
                "This is the final allowed model turn for this computer-use task. Do not call tools. "
                "Return a concise computer task result now: status, concrete summary, visible/current context, "
                "material actions taken, blockers, and the next action needed from the user if any."
            )
        if response_format == "filesystem_result":
            return (
                "This is the final allowed model turn for this filesystem task. Do not call tools. "
                "Return a concise result now: status, files or folders inspected, concrete findings, blockers, "
                "and next steps."
            )
        return (
            "This is the final allowed model turn for this tool loop. Do not call tools. "
            "Return the best final answer now using the context already collected. Include completed work, "
            "partial results, blockers, and concrete next steps where relevant."
        )

    def _with_loop_budget_messages(
        self,
        messages: List[Dict[str, Any]],
        *,
        iteration: int,
        max_iterations: int,
        turn_status_instructions: bool,
        is_final_turn: bool,
        final_turn_instruction: str,
    ) -> List[Dict[str, Any]]:
        if not turn_status_instructions and not is_final_turn:
            return messages
        request_messages = copy.deepcopy(messages)
        remaining = max(0, max_iterations - iteration)
        content = (
            f"Tool loop budget: this is model turn {iteration} of {max_iterations}; "
            f"{remaining} model turn(s) remain after this response."
        )
        if is_final_turn:
            content += "\n\n" + final_turn_instruction.strip()
        for message in request_messages:
            if message.get("role") == "system":
                message["content"] = f"{message.get('content') or ''}\n\n{content}".strip()
                break
        else:
            request_messages.insert(0, {"role": "system", "content": content})
        return request_messages

    async def _run_browser_agent(
        self,
        *,
        task: str,
        agent_depth: int,
        approval_id: str = "",
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> str:
        if agent_depth != 0:
            raise RuntimeError("use_browser can only be called by the root agent.")
        if not task.strip():
            raise ValueError("use_browser requires a non-empty task.")
        if self.browser_tool_bridge is None:
            raise RuntimeError("Browser delegation is not configured.")
        if not self.browser_agent_model:
            raise RuntimeError("No browser model is configured.")

        async with self._browser_lock:
            resident_model_name = self.llm.get_current_model()
            parent_model_name = resident_model_name or self._frame.model
            if not parent_model_name:
                raise RuntimeError("Cannot determine the parent model before browser delegation.")

            model_swapped = resident_model_name != self.browser_agent_model
            saved_parent_state = None
            if model_swapped:
                saved_parent_state = await self.llm.save_and_unload(self._frame.messages)
                if saved_parent_state is None:
                    raise RuntimeError("Could not save the parent model state; browser task was not started.")
            else:
                self.logger.info(
                    "Browser agent reusing resident model %s without a model transition.",
                    self.browser_agent_model,
                )

            browser_session: Optional[BrowserToolSessionPort] = None
            child_frame_pushed = False
            exit_browser = True
            browser_result = ""
            primary_error: Optional[BaseException] = None
            try:
                self._check_interrupted()
                if self._retained_browser_sessions:
                    browser_session = self._retained_browser_sessions.pop()
                    self.logger.info("Reusing retained browser session.")
                else:
                    browser_session = await self.browser_tool_bridge.open_session(
                        headless=self.browser_headless
                    )
                visual_runner = getattr(browser_session, "run_task", None)
                if model_swapped:
                    await self.llm.load_model(self.browser_agent_model)
                self._check_interrupted()
                if callable(visual_runner):
                    self.logger.info(
                        "Starting Fara visual browser task with model %s.",
                        self.browser_agent_model,
                    )
                    browser_result = await asyncio.wait_for(
                        visual_runner(
                            task=task.strip(),
                            model=self.browser_agent_model,
                            event_callback=event_callback,
                        ),
                        timeout=self.browser_task_timeout_seconds,
                    )
                else:
                    # Compatibility path for the legacy browser MCP backend.
                    browser_tools = await browser_session.get_all_tools()
                    if not browser_tools:
                        raise RuntimeError("The browser backend did not expose any safe tools.")
                    browser_tools = [
                        *browser_tools,
                        copy.deepcopy(self.FINISH_BROWSER_TASK_TOOL),
                    ]
                    delegated_tool_names = {
                        tool.get("function", {}).get("name")
                        for tool in browser_tools
                        if tool.get("function", {}).get("name")
                        and tool.get("function", {}).get("name") not in self.TERMINAL_TOOL_NAMES
                    }
                    self._push_frame(
                        model=self.browser_agent_model,
                        depth=agent_depth + 1,
                        tools=browser_tools,
                        tool_bridge=browser_session,
                        delegated_approval_id=approval_id,
                        preauthorized_tool_names=(delegated_tool_names if approval_id else set()),
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
                            event_callback=event_callback,
                            final_turn_response_format="browser_result",
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
                            self.logger.exception("Failed to close browser session.")

                restore_error: Optional[BaseException] = None
                if model_swapped:
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

    def _delegation_origin(self) -> tuple[str, dict[str, Any], Optional[str]]:
        metadata = current_interaction_metadata()
        source = current_interaction_source()
        if source == "direct_chat" and metadata.get("chat_session_id"):
            origin_kind = "direct_chat"
        elif metadata.get("opportunity_id"):
            origin_kind = "autonomy"
        elif source == "scheduled_chat_task" or metadata.get("scheduled_task_id"):
            origin_kind = "scheduled_task"
        else:
            origin_kind = "unknown"
        latest_goal = str(metadata.get("origin_goal") or "").strip()[:8000]
        if not latest_goal:
            for message in reversed(self._frame.messages):
                if message.get("role") == "user" and message.get("content"):
                    latest_goal = str(message["content"]).strip()[:8000]
                    break
        origin = {
            "source": source,
            "goal": latest_goal,
            "chat_session_id": metadata.get("chat_session_id"),
            "chat_message_id": metadata.get("chat_message_id"),
            "opportunity_id": metadata.get("opportunity_id"),
            "event_id": metadata.get("event_id"),
            "scheduled_task_id": metadata.get("scheduled_task_id"),
            "interaction_run_id": metadata.get("interaction_run_id"),
            "activity_run_id": metadata.get("activity_run_id"),
        }
        parent_delegation_id = str(metadata.get("delegation_id") or "").strip() or None
        return origin_kind, origin, parent_delegation_id

    def _create_local_control_approval(
        self,
        *,
        capability: str,
        tool_name: str,
        approval_kind: str,
        task: str,
        reason: str,
        expected_result: str,
        continuation_instruction: str,
    ) -> tuple[ApprovalGrant, DelegatedTask]:
        arguments = {
            "task": task.strip(),
            "reason": reason.strip(),
            "expected_result": expected_result.strip(),
            "continuation_instruction": continuation_instruction.strip()
            or "Use the delegated result to complete and report the original goal.",
        }
        fingerprint = self.capability_policy.action_fingerprint(tool_name, arguments)
        now = datetime.now(timezone.utc)
        delegation_id = uuid.uuid4().hex
        origin_kind, origin, parent_delegation_id = self._delegation_origin()
        if (
            parent_delegation_id
            and hasattr(self.capability_policy.store, "delegated_chain_depth")
            and self.capability_policy.store.delegated_chain_depth(parent_delegation_id) >= 5
        ):
            raise RuntimeError(
                "Delegated workflow reached the five-approval continuation limit; return control to the user."
            )
        approval = ApprovalGrant(
            approval_id=uuid.uuid4().hex,
            capability=capability,
            action_fingerprint=fingerprint,
            constraints_json=json.dumps(
                {
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "approval_kind": approval_kind,
                    "delegation_id": delegation_id,
                    "origin_kind": origin_kind,
                    "origin": origin,
                },
                ensure_ascii=False,
            ),
            status="pending",
            created_at=now.isoformat(),
            expires_at=(now + timedelta(minutes=self.local_control_approval_ttl_minutes)).isoformat(),
            approver="pending",
        )
        delegated_task = DelegatedTask(
            delegation_id=delegation_id,
            approval_id=approval.approval_id,
            capability=capability,
            task=arguments["task"],
            reason=arguments["reason"],
            expected_result=arguments["expected_result"],
            continuation_instruction=arguments["continuation_instruction"],
            origin_kind=origin_kind,
            origin_json=json.dumps(origin, ensure_ascii=False),
            parent_model=str(self.llm.get_current_model() or self._frame.model or ""),
            parent_delegation_id=parent_delegation_id,
            status="awaiting_approval",
            created_at=now.isoformat(),
            updated_at=now.isoformat(),
        )
        return approval, delegated_task

    def _request_browser_use(
        self,
        *,
        task: str,
        reason: str,
        expected_result: str = "",
        continuation_instruction: str = "",
        agent_depth: int,
    ) -> str:
        if agent_depth != 0:
            raise RuntimeError("use_browser can only be called by the root agent.")
        if not task.strip():
            raise ValueError("use_browser requires a non-empty task.")
        if self.browser_tool_bridge is None:
            raise RuntimeError("Browser delegation is not configured.")
        if not self.browser_agent_model:
            raise RuntimeError("No browser model is configured.")
        if self.capability_policy is None or not hasattr(self.capability_policy.store, "create_approval"):
            raise RuntimeError("Browser-use approvals require the autonomy approval store.")

        approval, delegated_task = self._create_local_control_approval(
            capability="browser.use",
            tool_name="use_browser",
            approval_kind="browser_use_deployment",
            task=task,
            reason=reason,
            expected_result=expected_result,
            continuation_instruction=continuation_instruction,
        )
        raise InteractionSuspended(
            approval=approval,
            delegated_task=delegated_task,
        )

    async def deploy_browser_agent(
        self,
        *,
        task: str,
        approval_id: str = "",
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> str:
        scoped_approval_id = ""
        if approval_id:
            if self.capability_policy is None:
                scoped_approval_id = approval_id
            else:
                store = self.capability_policy.store
                approval = (
                    store.get_approval(approval_id)
                    if hasattr(store, "get_approval")
                    else None
                )
                if (
                    approval is not None
                    and approval.capability == "browser.use"
                    and approval.status in {"approved", "used"}
                ):
                    scoped_approval_id = approval_id
                else:
                    self.logger.warning(
                        "Browser delegation %s has no approved browser.use grant; "
                        "individual tool policy remains active.",
                        approval_id,
                    )
        result = await self._run_browser_agent(
            task=task,
            agent_depth=0,
            approval_id=scoped_approval_id,
            event_callback=event_callback,
        )
        if self.capability_policy is not None and hasattr(self.capability_policy.store, "audit"):
            self.capability_policy.store.audit(
                "ambient_agent",
                "browser_use.completed",
                approval_id or "unknown",
                {"result": result[:2000]},
            )
        return result

    async def _run_filesystem_agent(
        self,
        *,
        task: str,
        granted_paths: list[str],
        agent_depth: int,
    ) -> str:
        if agent_depth != 0:
            raise RuntimeError("use_filesystem can only be called by the root agent.")
        if not task.strip():
            raise ValueError("use_filesystem requires a non-empty task.")
        if not self.filesystem_agent_model:
            raise RuntimeError("No filesystem model is configured.")
        resolved_grants = [
            str(Path(path).expanduser().resolve(strict=False))
            for path in granted_paths
        ]

        async with self._filesystem_lock:
            resident_model_name = self.llm.get_current_model()
            parent_model_name = resident_model_name or self._frame.model
            if not parent_model_name:
                raise RuntimeError("Cannot determine the parent model before filesystem delegation.")

            model_swapped = resident_model_name != self.filesystem_agent_model
            saved_parent_state = None
            if model_swapped:
                saved_parent_state = await self.llm.save_and_unload(self._frame.messages)
                if saved_parent_state is None:
                    raise RuntimeError("Could not save the parent model state; filesystem task was not started.")
            else:
                self.logger.info(
                    "Filesystem agent reusing resident model %s without a model transition.",
                    self.filesystem_agent_model,
                )

            session = FilesystemControlSession(
                granted_paths=resolved_grants,
                max_read_bytes=self.filesystem_max_read_bytes,
                max_list_entries=self.filesystem_max_list_entries,
            )
            child_frame_pushed = False
            fs_result = ""
            primary_error: Optional[BaseException] = None
            try:
                fs_tools = [
                    *await session.get_all_tools(),
                    copy.deepcopy(self.FINISH_FILESYSTEM_TASK_TOOL),
                ]
                if model_swapped:
                    await self.llm.load_model(self.filesystem_agent_model)
                self._push_frame(
                    model=self.filesystem_agent_model,
                    depth=agent_depth + 1,
                    tools=fs_tools,
                    tool_bridge=session,
                )
                child_frame_pushed = True
                allowed_tool_names = {
                    tool.get("function", {}).get("name")
                    for tool in fs_tools
                    if tool.get("function", {}).get("name")
                }
                fs_result = await asyncio.wait_for(
                    self.run_interaction(
                        user_input=(
                            "You have been given this read-only filesystem task:\n"
                            f"{task.strip()}\n\n"
                            "Resolved granted filesystem roots. Use only these absolute roots or child paths returned by filesystem tools:\n"
                            + json.dumps(resolved_grants, ensure_ascii=False, indent=2)
                        ),
                        system_prompt=self.FILESYSTEM_AGENT_PROMPT,
                        model=self.filesystem_agent_model,
                        agent_depth=agent_depth + 1,
                        allowed_tool_names=allowed_tool_names,
                        report_policy="silent",
                        final_turn_response_format="filesystem_result",
                    ),
                    timeout=self.filesystem_task_timeout_seconds,
                )
            except BaseException as exc:
                primary_error = exc
            finally:
                if child_frame_pushed:
                    self._pop_frame()
                await session.cleanup()
                restore_error: Optional[BaseException] = None
                if model_swapped:
                    try:
                        current_model_name = self.llm.get_current_model()
                        if current_model_name and current_model_name != parent_model_name:
                            await self.llm.unload_model()
                        await self.llm.load_and_restore()
                    except BaseException as exc:
                        restore_error = exc
                        self.logger.exception("Failed to restore parent model after filesystem delegation.")
                if restore_error is not None:
                    if primary_error is not None:
                        raise RuntimeError(
                            f"Filesystem task failed ({primary_error}) and parent restoration also failed ({restore_error})."
                        ) from restore_error
                    raise RuntimeError(
                        f"Filesystem task completed but parent restoration failed: {restore_error}"
                    ) from restore_error
                if primary_error is not None:
                    raise primary_error
            return fs_result

    def _request_computer_use(
        self,
        *,
        task: str,
        reason: str,
        expected_result: str = "",
        continuation_instruction: str = "",
        agent_depth: int,
    ) -> str:
        if agent_depth != 0:
            raise RuntimeError("request_computer_use can only be called by the root agent.")
        if not self.computer_enabled:
            raise RuntimeError("Computer use is disabled in configuration.")
        if not self.computer_agent_model:
            raise RuntimeError("No computer-use model is configured.")
        if not task.strip():
            raise ValueError("request_computer_use requires a non-empty task.")
        if self.capability_policy is None or not hasattr(self.capability_policy.store, "create_approval"):
            raise RuntimeError("Computer-use approvals require the autonomy approval store.")

        approval, delegated_task = self._create_local_control_approval(
            capability="computer.use",
            tool_name="request_computer_use",
            approval_kind="computer_use_deployment",
            task=task,
            reason=reason,
            expected_result=expected_result,
            continuation_instruction=continuation_instruction,
        )
        raise InteractionSuspended(
            approval=approval,
            delegated_task=delegated_task,
        )

    async def deploy_computer_agent(
        self,
        *,
        task: str,
        approval_id: str = "",
        read_only: bool = False,
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> str:
        if not self.computer_enabled:
            raise RuntimeError("Computer use is disabled in configuration.")
        if not self.computer_agent_model:
            raise RuntimeError("No computer-use model is configured.")
        if not task.strip():
            raise ValueError("Computer-use deployment requires a non-empty task.")
        async with self._computer_lock:
            resident_model_name = self.llm.get_current_model()
            parent_model_name = resident_model_name or self._frame.model
            if not parent_model_name:
                raise RuntimeError("Cannot determine the parent model before computer-use deployment.")

            model_swapped = resident_model_name != self.computer_agent_model
            saved_parent_state = None
            if model_swapped:
                saved_parent_state = await self.llm.save_and_unload(self._frame.messages)
                if saved_parent_state is None:
                    raise RuntimeError("Could not save the parent model state; computer-use task was not started.")
            else:
                self.logger.info(
                    "Computer agent reusing resident model %s without a model transition.",
                    self.computer_agent_model,
                )

            session = ComputerControlSession(
                max_actions=self.computer_max_actions_per_task,
                screenshot_dir=self.computer_screenshot_dir,
                read_only=read_only,
            )
            child_frame_pushed = False
            computer_result = ""
            primary_error: Optional[BaseException] = None
            try:
                computer_tools = [
                    *await session.get_all_tools(),
                    copy.deepcopy(self.FINISH_COMPUTER_TASK_TOOL),
                ]
                delegated_tool_names = {
                    tool.get("function", {}).get("name")
                    for tool in computer_tools
                    if tool.get("function", {}).get("name")
                    and tool.get("function", {}).get("name") not in self.TERMINAL_TOOL_NAMES
                }
                if model_swapped:
                    await self.llm.load_model(self.computer_agent_model)
                self._push_frame(
                    model=self.computer_agent_model,
                    depth=1,
                    tools=computer_tools,
                    tool_bridge=session,
                    delegated_approval_id=approval_id,
                    preauthorized_tool_names=(delegated_tool_names if approval_id else set()),
                )
                child_frame_pushed = True
                allowed_tool_names = {
                    tool.get("function", {}).get("name")
                    for tool in computer_tools
                    if tool.get("function", {}).get("name")
                }
                computer_result = await asyncio.wait_for(
                    self.run_interaction(
                        user_input="You have been approved for this computer-use task:\n" + task.strip(),
                        system_prompt=self._computer_agent_prompt(),
                        model=self.computer_agent_model,
                        agent_depth=1,
                        allowed_tool_names=allowed_tool_names,
                        report_policy="silent",
                        event_callback=event_callback,
                        iteration_image_provider=session.capture_screenshot_for_model,
                        final_turn_response_format="computer_result",
                    ),
                    timeout=self.computer_task_timeout_seconds,
                )
            except BaseException as exc:
                primary_error = exc
            finally:
                if child_frame_pushed:
                    self._pop_frame()
                await session.cleanup()
                restore_error: Optional[BaseException] = None
                if model_swapped:
                    try:
                        current_model_name = self.llm.get_current_model()
                        if current_model_name and current_model_name != parent_model_name:
                            await self.llm.unload_model()
                        await self.llm.load_and_restore()
                    except BaseException as exc:
                        restore_error = exc
                        self.logger.exception("Failed to restore parent model after computer-use deployment.")
                if restore_error is not None:
                    if primary_error is not None:
                        raise RuntimeError(
                            f"Computer-use task failed ({primary_error}) and parent restoration also failed ({restore_error})."
                        ) from restore_error
                    raise RuntimeError(
                        f"Computer-use task completed but parent restoration failed: {restore_error}"
                    ) from restore_error

            if isinstance(primary_error, ComputerControlTerminated):
                computer_result = json.dumps(
                    {
                        "status": "terminated",
                        "summary": "Computer-use agent was terminated by Shift+Esc.",
                        "approval_id": approval_id,
                    },
                    ensure_ascii=False,
                )
            elif primary_error is not None:
                raise primary_error
            if self.capability_policy is not None and hasattr(self.capability_policy.store, "audit"):
                self.capability_policy.store.audit(
                    "ambient_agent",
                    "computer_use.completed",
                    approval_id or "unknown",
                    {"result": computer_result[:2000]},
                )
            return computer_result

    def _computer_agent_prompt(self) -> str:
        if self.computer_agent_family == "ui_tars":
            return (
                "You are a dedicated UI-TARS-style computer-use executor working on one user-approved desktop-control task.\n\n"
                "Rules:\n"
                "- Perform only the approved task supplied by Ambient AI.\n"
                "- You receive a fresh full-screen screenshot before every model turn. Treat that screenshot as the current desktop state.\n"
                "- Use the provided computer_* tools only; do not answer with raw coordinates unless calling a tool.\n"
                "- Mouse coordinates must use normalized 0..1000 coordinates, where (0,0) is top-left and (1000,1000) is bottom-right.\n"
                "- Prefer short action sequences: inspect screenshot, act once, wait for the next screenshot, then continue.\n"
                "- Do not use shell commands, system shutdown/logout/lock, credential entry, payment, checkout, or destructive file-manager actions.\n"
                "- Do not bypass authentication, CAPTCHA, two-factor authentication, security warnings, or confirmation screens.\n"
                "- When the task reaches a terminal state, call finish_computer_task exactly once with status, summary, material actions, and blockers.\n"
            )
        if self.computer_agent_family == "qwen_generic":
            return (
                "You are a dedicated visual desktop-control agent working on one user-approved task.\n\n"
                "Rules:\n"
                "- Use the screenshot attached to each turn as the authoritative desktop state.\n"
                "- Call exactly one computer_* tool per step unless finishing.\n"
                "- Use normalized 0..1000 coordinates for all mouse actions.\n"
                "- Do not repeat an action if the previous screenshot did not visibly change; replan or inspect.\n"
                "- Do not use shell commands, system shutdown/logout/lock, credential entry, payment, checkout, or destructive file-manager actions.\n"
                "- When complete or blocked, call finish_computer_task exactly once.\n"
            )
        return self.COMPUTER_AGENT_PROMPT

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
        iteration_image_provider: Optional[Callable[[], str]] = None,
        max_iterations: Optional[int] = None,
        turn_status_instructions: bool = True,
        final_turn_instruction: Optional[str] = None,
        disable_tools_on_final_turn: bool = True,
        final_turn_response_format: str = "plain",
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

        trace_metadata = dict(existing_metadata)
        trace_metadata["interaction_run_id"] = interaction_run_id
        with interaction_trace(source_name, trace_metadata):
            self._frame.model = model
            self._frame.depth = agent_depth
            if not self._frame.messages:
                self._frame.messages.append(
                    {"role": "system", "content": self._build_system_prompt(system_prompt)}
                )
            self._frame.messages.append({"role": "user", "content": user_input})
            return await self._run_interaction_loop(
                model=model,
                user_input=user_input,
                image_path=image_path,
                agent_depth=agent_depth,
                allowed_tool_names=allowed_tool_names,
                report_policy=report_policy,
                event_callback=event_callback,
                source_name=source_name,
                trace_metadata=trace_metadata,
                interaction_run_id=interaction_run_id,
                tools_used=tools_used,
                iteration=0,
                iteration_image_provider=iteration_image_provider,
                max_iterations=max_iterations,
                turn_status_instructions=turn_status_instructions,
                final_turn_instruction=final_turn_instruction,
                disable_tools_on_final_turn=disable_tools_on_final_turn,
                final_turn_response_format=final_turn_response_format,
            )

    async def resume_interaction(
        self,
        *,
        checkpoint: dict[str, Any],
        tool_result: dict[str, Any] | str,
        delegation_id: str,
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> str:
        """Resume the original model loop by satisfying its suspended tool call."""
        messages = checkpoint.get("messages")
        tool_call_id = str(checkpoint.get("suspended_tool_call_id") or "")
        model = str(checkpoint.get("model") or "")
        if not isinstance(messages, list) or not messages or not tool_call_id or not model:
            raise ValueError("Delegated task checkpoint is incomplete and cannot be resumed.")
        assistant_index = -1
        suspended_call_index = -1
        for message_index, message in enumerate(messages):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            for call_index, call in enumerate(message.get("tool_calls") or []):
                if isinstance(call, dict) and str(call.get("id") or "") == tool_call_id:
                    assistant_index = message_index
                    suspended_call_index = call_index
                    break
        if assistant_index < 0:
            raise ValueError("Suspended tool call is not present in the delegated checkpoint.")

        self._frame.model = model
        self._frame.depth = int(checkpoint.get("agent_depth") or 0)
        self._frame.messages = copy.deepcopy(messages)
        serialized_result = (
            tool_result
            if isinstance(tool_result, str)
            else json.dumps(tool_result, ensure_ascii=False)
        )
        suspended_tool_message = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": str(checkpoint.get("suspended_tool_name") or "delegated_control"),
            "content": serialized_result,
        }
        prior_call_ids = {
            str(call.get("id") or "")
            for call in self._frame.messages[assistant_index].get("tool_calls", [])[:suspended_call_index]
        }
        insert_at = assistant_index + 1
        while (
            insert_at < len(self._frame.messages)
            and self._frame.messages[insert_at].get("role") == "tool"
            and str(self._frame.messages[insert_at].get("tool_call_id") or "") in prior_call_ids
        ):
            insert_at += 1
        self._frame.messages.insert(insert_at, suspended_tool_message)
        source_name = str(checkpoint.get("source_name") or "delegated_task_continuation")
        trace_metadata = checkpoint.get("trace_metadata")
        trace_metadata = dict(trace_metadata) if isinstance(trace_metadata, dict) else {}
        trace_metadata["delegation_id"] = delegation_id
        interaction_run_id = str(
            checkpoint.get("interaction_run_id")
            or trace_metadata.get("interaction_run_id")
            or uuid.uuid4().hex
        )
        trace_metadata["interaction_run_id"] = interaction_run_id
        allowed = checkpoint.get("allowed_tool_names")
        allowed_tool_names = set(allowed) if isinstance(allowed, list) else None
        tools_used = [str(item) for item in checkpoint.get("tools_used", [])]
        tools_used.append(str(checkpoint.get("suspended_tool_name") or "delegated_control"))
        with interaction_trace(source_name, trace_metadata):
            return await self._run_interaction_loop(
                model=model,
                user_input=str(checkpoint.get("user_input") or ""),
                image_path="",
                agent_depth=int(checkpoint.get("agent_depth") or 0),
                allowed_tool_names=allowed_tool_names,
                report_policy=str(checkpoint.get("report_policy") or "silent"),
                event_callback=event_callback,
                source_name=source_name,
                trace_metadata=trace_metadata,
                interaction_run_id=interaction_run_id,
                tools_used=tools_used,
                iteration=int(checkpoint.get("iteration") or 0),
                iteration_image_provider=None,
                max_iterations=checkpoint.get("max_iterations"),
                turn_status_instructions=bool(checkpoint.get("turn_status_instructions", True)),
                final_turn_instruction=checkpoint.get("final_turn_instruction"),
                disable_tools_on_final_turn=bool(checkpoint.get("disable_tools_on_final_turn", True)),
                final_turn_response_format=str(checkpoint.get("final_turn_response_format") or "plain"),
            )

    def _image_for_iteration(
        self,
        *,
        image_path: str,
        iteration: int,
        iteration_image_provider: Optional[Callable[[], str]],
    ) -> str:
        if iteration_image_provider is None:
            return image_path if iteration == 1 else ""
        try:
            return str(iteration_image_provider() or "")
        except Exception:
            self.logger.exception("Failed to capture per-iteration image for model turn.")
            return ""

    def _messages_for_iteration_image(
        self,
        *,
        request_image_path: str,
        iteration: int,
        iteration_image_provider: Optional[Callable[[], str]],
    ) -> List[Dict[str, Any]]:
        if not request_image_path or iteration_image_provider is None or iteration <= 1:
            return self._frame.messages
        messages = copy.deepcopy(self._frame.messages)
        messages.append(
            {
                "role": "user",
                "content": (
                    "Fresh screenshot after the latest computer action. "
                    "Use the attached image as the current desktop state before choosing the next tool call."
                ),
            }
        )
        return messages

    async def _run_interaction_loop(
        self,
        *,
        model: str,
        user_input: str,
        image_path: str,
        agent_depth: int,
        allowed_tool_names: Optional[set[str]],
        report_policy: str,
        event_callback: Optional[Callable[[Dict[str, Any]], None]],
        source_name: str,
        trace_metadata: dict[str, Any],
        interaction_run_id: str,
        tools_used: List[str],
        iteration: int,
        iteration_image_provider: Optional[Callable[[], str]],
        max_iterations: Optional[int],
        turn_status_instructions: bool,
        final_turn_instruction: Optional[str],
        disable_tools_on_final_turn: bool,
        final_turn_response_format: str,
    ) -> str:
        assistant_text = ""
        loop_max = max(1, int(max_iterations or self.max_interaction_iterations or self.MAX_ITERATIONS))
        recovery_enabled = self.final_turn_recovery_enabled
        final_instruction = (
            final_turn_instruction
            or self._default_final_turn_instruction(final_turn_response_format)
        )
        while iteration < loop_max:
            self._check_interrupted()
            iteration += 1
            is_final_turn = recovery_enabled and iteration >= loop_max
            self.logger.info(
                "--- Iteration %s (agent depth %s/%s) ---",
                iteration, agent_depth, self.MAX_AGENT_DEPTH,
            )
            request_image_path = self._image_for_iteration(
                image_path=image_path,
                iteration=iteration,
                iteration_image_provider=iteration_image_provider,
            )
            request_messages = self._messages_for_iteration_image(
                request_image_path=request_image_path,
                iteration=iteration,
                iteration_image_provider=iteration_image_provider,
            )
            request_messages = self._with_loop_budget_messages(
                request_messages,
                iteration=iteration,
                max_iterations=loop_max,
                turn_status_instructions=turn_status_instructions,
                is_final_turn=is_final_turn,
                final_turn_instruction=final_instruction,
            )
            request_tools = None
            if not (is_final_turn and disable_tools_on_final_turn):
                request_tools = self._tools_for_agent_depth(
                    agent_depth, allowed_tool_names=allowed_tool_names
                )
            completion = await self.llm.chat_completion_stream(
                model=model,
                messages=request_messages,
                tools=request_tools,
                image=request_image_path,
            )
            assistant_text, tool_calls = await self._consume_stream(
                completion,
                event_callback=event_callback,
                allow_text_tool_recovery=not is_final_turn,
            )
            if (
                self.shutdown_controller is not None
                and self.shutdown_controller.is_graceful_requested()
            ):
                # First Ctrl+C intentionally drains only the already-open
                # completion. Never turn its tool calls into more real-world
                # actions or another llama.cpp request.
                self._frame.messages.append({"role": "assistant", "content": assistant_text})
                self.logger.info(
                    "Graceful shutdown: current llama.cpp response finished; skipping tools and follow-up turns."
                )
                return assistant_text
            if tool_calls and hasattr(self.llm, "register_tool_calls"):
                self.llm.register_tool_calls(tool_calls)
            self._check_interrupted()
            if is_final_turn and disable_tools_on_final_turn and tool_calls:
                self.logger.warning(
                    "Ignoring %s tool call(s) emitted on final recovery turn.",
                    len(tool_calls),
                )
                tool_calls = []
            if not tool_calls:
                self._frame.messages.append({"role": "assistant", "content": assistant_text})
                self.logger.info("Model finished (no more tool calls)")
                await self._attach_user_report(
                    report_policy=report_policy, interaction_run_id=interaction_run_id,
                    model=model, user_input=user_input, final_response=assistant_text,
                    tools_used=tools_used, source_name=source_name,
                )
                self._record_temporal_interaction(
                    interaction_run_id=interaction_run_id,
                    source_name=source_name,
                    user_input=user_input,
                    final_response=assistant_text,
                    tools_used=tools_used,
                    semantic_index=self._is_final_json_response(assistant_text),
                )
                return assistant_text

            self._frame.messages.append(
                {"role": "assistant", "content": assistant_text or None, "tool_calls": tool_calls}
            )
            try:
                self._check_interrupted()
                tool_results = await self._execute_tool_calls(
                    tool_calls, agent_depth=agent_depth,
                    allowed_tool_names=allowed_tool_names, event_callback=event_callback,
                )
            except InteractionSuspended as suspended:
                self._append_deferred_sibling_tool_results(tool_calls, suspended.tool_call_id)
                checkpoint = {
                    "version": 1,
                    "messages": copy.deepcopy(self._frame.messages),
                    "model": model,
                    "source_name": source_name,
                    "trace_metadata": dict(trace_metadata),
                    "interaction_run_id": interaction_run_id,
                    "user_input": user_input,
                    "agent_depth": agent_depth,
                    "allowed_tool_names": sorted(allowed_tool_names) if allowed_tool_names is not None else None,
                    "report_policy": report_policy,
                    "iteration": iteration,
                    "tools_used": list(tools_used),
                    "max_iterations": loop_max,
                    "turn_status_instructions": turn_status_instructions,
                    "final_turn_instruction": final_turn_instruction,
                    "disable_tools_on_final_turn": disable_tools_on_final_turn,
                    "final_turn_response_format": final_turn_response_format,
                    "suspended_tool_call_id": suspended.tool_call_id,
                    "suspended_tool_name": next(
                        (str(call.get("function", {}).get("name") or "") for call in tool_calls
                         if str(call.get("id") or "") == suspended.tool_call_id),
                        "delegated_control",
                    ),
                }
                self._persist_suspended_interaction(suspended, checkpoint)
                raise
            tools_used.extend(name for name, _ in tool_results)
            if any(
                name in self.TERMINAL_TOOL_NAMES and not result.startswith("Error:")
                for name, result in tool_results
            ):
                terminal_result = "\n".join(result for _, result in tool_results)
                await self._attach_user_report(
                    report_policy=report_policy, interaction_run_id=interaction_run_id,
                    model=model, user_input=user_input, final_response=terminal_result,
                    tools_used=tools_used, source_name=source_name,
                )
                self._record_temporal_interaction(
                    interaction_run_id=interaction_run_id,
                    source_name=source_name,
                    user_input=user_input,
                    final_response=terminal_result,
                    tools_used=tools_used,
                    semantic_index=False,
                )
                return terminal_result

        self.logger.warning("Reached maximum iterations: (%s). Stopping.", loop_max)
        await self._attach_user_report(
            report_policy=report_policy, interaction_run_id=interaction_run_id,
            model=model, user_input=user_input, final_response=assistant_text,
            tools_used=tools_used, source_name=source_name,
        )
        self._record_temporal_interaction(
            interaction_run_id=interaction_run_id,
            source_name=source_name,
            user_input=user_input,
            final_response=assistant_text,
            tools_used=tools_used,
            semantic_index=False,
        )
        return assistant_text

    def _record_temporal_interaction(
        self,
        *,
        interaction_run_id: str,
        source_name: str,
        user_input: str,
        final_response: str,
        tools_used: List[str],
        semantic_index: bool = False,
    ) -> None:
        if self.temporal_memory_service is None:
            return
        event = SimpleNamespace(
            event_id=f"interaction:{interaction_run_id}",
            event_type="llm_interaction",
            source_kind=source_name,
            source_ref=f"interaction://{interaction_run_id}",
            occurred_at=datetime.now().isoformat(),
            confidence=0.75,
            payload_json=json.dumps(
                {
                    "task": user_input[:8000],
                    "summary": final_response[:10000],
                    "entities": list(dict.fromkeys(tools_used)),
                },
                ensure_ascii=False,
            ),
        )
        try:
            self.temporal_memory_service.record_ambient_event(
                event,
                outcome=final_response[:10000],
                semantic_index=semantic_index,
            )
        except Exception:
            self.logger.exception("Could not record interaction in temporal memory.")

    @staticmethod
    def _is_final_json_response(value: str) -> bool:
        """Return true only for a complete model-produced JSON object or array."""
        text = str(value or "").strip()
        if text.startswith("```") and text.endswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1]).strip() if len(lines) >= 2 else ""
        if not text:
            return False
        try:
            parsed = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            return False
        return isinstance(parsed, (dict, list))

    def _append_deferred_sibling_tool_results(
        self, tool_calls: List[Dict[str, Any]], suspended_tool_call_id: str
    ) -> None:
        existing = {
            str(message.get("tool_call_id") or "")
            for message in self._frame.messages
            if message.get("role") == "tool"
        }
        for call in tool_calls:
            call_id = str(call.get("id") or "")
            if not call_id or call_id == suspended_tool_call_id or call_id in existing:
                continue
            self._frame.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": str(call.get("function", {}).get("name") or "unknown"),
                    "content": "Error: not executed because this turn is awaiting local approval.",
                }
            )

    def _persist_suspended_interaction(
        self, suspended: InteractionSuspended, checkpoint: dict[str, Any]
    ) -> None:
        if self.capability_policy is None:
            raise RuntimeError("Cannot persist suspended interaction without an approval store.")
        store = self.capability_policy.store
        checkpoint_json = json.dumps(checkpoint, ensure_ascii=False)
        if not hasattr(store, "create_suspended_delegation"):
            raise RuntimeError("Approval store does not support durable interaction checkpoints.")
        store.create_suspended_delegation(
            approval=suspended.approval,
            task=suspended.delegated_task,
            checkpoint_json=checkpoint_json,
            tool_call_id=suspended.tool_call_id,
        )
        if hasattr(store, "audit"):
            store.audit(
                "ambient_agent",
                f"{suspended.delegated_task.capability}.requested",
                suspended.approval_id,
                {
                    "delegation_id": suspended.delegation_id,
                    "tool_call_id": suspended.tool_call_id,
                    "origin_kind": suspended.delegated_task.origin_kind,
                },
            )

    async def _consume_stream(
        self,
        completion,
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        allow_text_tool_recovery: bool = True,
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
            self._check_interrupted()
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
        if allow_text_tool_recovery and not tool_calls:
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

        def record_tool_result(
            *,
            tool_id: str,
            tool_name: str,
            arguments_json: str,
            output: str,
            ok: bool,
            status: str = "completed",
        ) -> None:
            if hasattr(self.llm, "attach_tool_result"):
                self.llm.attach_tool_result(
                    tool_id,
                    tool_name=tool_name,
                    arguments_json=arguments_json,
                    output=output,
                    ok=ok,
                    status=status,
                )

        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            tool_args_str = tool_call["function"]["arguments"]
            tool_id = tool_call["id"]

            self.logger.info("Calling tool: %s", tool_name)
            self.logger.info("Arguments: %s", tool_args_str)
            self._emit_event(
                event_callback,
                {
                    "type": "tool_started",
                    "tool_name": tool_name,
                    "tool_call_id": tool_id,
                    "arguments_json": tool_args_str,
                },
            )

            try:
                tool_args = json.loads(tool_args_str) if tool_args_str else {}
                idempotency_key = None
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
                    record_tool_result(
                        tool_id=tool_id, tool_name=tool_name, arguments_json=tool_args_str,
                        output=response_content, ok=False,
                    )
                    continue
                if (
                    self.capability_policy is not None
                    and tool_name not in self.TERMINAL_TOOL_NAMES
                    and tool_name not in self.LOCAL_CONTROL_REQUEST_TOOLS
                    and not (
                        self._frame.delegated_approval_id
                        and tool_name in self._frame.preauthorized_tool_names
                    )
                ):
                    metadata = current_interaction_metadata()
                    source = current_interaction_source()
                    observed_evidence = "\n".join(
                        str(message.get("content") or "")
                        for message in self._frame.messages
                        if message.get("role") == "tool"
                    )[-50000:]
                    self.capability_policy.authorize_or_raise(
                        tool_name=tool_name,
                        arguments=tool_args,
                        source=source,
                        confidence=float(metadata.get("autonomy_confidence") or 1.0),
                        explicit_user_request=bool(
                            metadata.get("explicit_user_request")
                            or source in {"direct_chat", "scheduled_chat_task"}
                        ),
                        evidence_context={"observed_text": observed_evidence},
                    )
                    idempotency_key, should_execute, prior_result = self.capability_policy.begin_execution(
                        tool_name=tool_name,
                        arguments=tool_args,
                        scope=str(
                            metadata.get("opportunity_id")
                            or metadata.get("scheduled_task_id")
                            or metadata.get("interaction_run_id")
                            or tool_id
                        ),
                    )
                    if not should_execute:
                        response_content = prior_result or "Action already attempted; duplicate execution suppressed."
                        tool_results.append((tool_name, response_content))
                        self._frame.messages.append(
                            {"role": "tool", "tool_call_id": tool_id, "name": tool_name, "content": response_content}
                        )
                        record_tool_result(
                            tool_id=tool_id, tool_name=tool_name, arguments_json=tool_args_str,
                            output=response_content, ok=True,
                        )
                        continue
                elif (
                    self.capability_policy is not None
                    and self._frame.delegated_approval_id
                    and tool_name in self._frame.preauthorized_tool_names
                    and hasattr(self.capability_policy.store, "audit")
                ):
                    self.capability_policy.store.audit(
                        "ambient_agent",
                        "browser_use.scoped_tool_authorized",
                        self._frame.delegated_approval_id,
                        {"tool_name": tool_name, "tool_call_id": tool_id},
                    )
                if tool_name == "use_browser":
                    task = tool_args.get("task", "")
                    response_content = self._request_browser_use(
                        task=str(task),
                        reason=str(tool_args.get("reason", "")),
                        expected_result=str(tool_args.get("expected_result", "")),
                        continuation_instruction=str(tool_args.get("continuation_instruction", "")),
                        agent_depth=agent_depth,
                    )
                elif tool_name == "use_filesystem":
                    granted_paths = tool_args.get("granted_paths")
                    if not isinstance(granted_paths, list):
                        raise ValueError("use_filesystem granted_paths must be a list of absolute paths.")
                    response_content = await self._run_filesystem_agent(
                        task=str(tool_args.get("task", "")),
                        granted_paths=[str(path) for path in granted_paths],
                        agent_depth=agent_depth,
                    )
                elif tool_name == "request_computer_use":
                    response_content = self._request_computer_use(
                        task=str(tool_args.get("task", "")),
                        reason=str(tool_args.get("reason", "")),
                        expected_result=str(tool_args.get("expected_result", "")),
                        continuation_instruction=str(tool_args.get("continuation_instruction", "")),
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
                            "status": status.strip(), "summary": summary.strip(),
                            "browser_exited": exit_browser,
                            "details": str(tool_args.get("details") or "").strip(),
                            "actions_performed": tool_args.get("actions_performed") or [],
                            "sources": tool_args.get("sources") or [],
                            "blockers": tool_args.get("blockers") or [],
                        },
                        ensure_ascii=False,
                    )
                elif tool_name == "finish_filesystem_task":
                    if agent_depth == 0:
                        raise RuntimeError(
                            "finish_filesystem_task can only be called by the delegated filesystem agent."
                        )
                    status = tool_args.get("status")
                    summary = tool_args.get("summary")
                    if not isinstance(status, str) or not status.strip():
                        raise ValueError("finish_filesystem_task status must be a non-empty string.")
                    if not isinstance(summary, str) or not summary.strip():
                        raise ValueError("finish_filesystem_task summary must be a non-empty string.")
                    response_content = json.dumps(
                        {
                            "status": status.strip(), "summary": summary.strip(),
                            "details": str(tool_args.get("details") or "").strip(),
                            "actions_performed": tool_args.get("actions_performed") or [],
                            "sources": tool_args.get("sources") or [],
                            "blockers": tool_args.get("blockers") or [],
                        },
                        ensure_ascii=False,
                    )
                elif tool_name == "finish_computer_task":
                    if agent_depth == 0:
                        raise RuntimeError(
                            "finish_computer_task can only be called by the delegated computer-use agent."
                        )
                    status = tool_args.get("status")
                    summary = tool_args.get("summary")
                    if not isinstance(status, str) or not status.strip():
                        raise ValueError("finish_computer_task status must be a non-empty string.")
                    if not isinstance(summary, str) or not summary.strip():
                        raise ValueError("finish_computer_task summary must be a non-empty string.")
                    response_content = json.dumps(
                        {
                            "status": status.strip(), "summary": summary.strip(),
                            "details": str(tool_args.get("details") or "").strip(),
                            "actions_performed": tool_args.get("actions_performed") or [],
                            "sources": tool_args.get("sources") or [],
                            "blockers": tool_args.get("blockers") or [],
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
                elif tool_name == "create_recurring_task":
                    if self.recurring_task_service is None:
                        raise RuntimeError("Recurring task scheduling is not configured.")
                    metadata = current_interaction_metadata()
                    raw_scope = tool_args.get("source_scope")
                    source_scope = raw_scope if isinstance(raw_scope, dict) else {}
                    task = self.recurring_task_service.create(
                        title=str(tool_args.get("title", "")),
                        instruction=str(tool_args.get("instruction", "")),
                        task_kind=str(tool_args.get("task_kind", "")),
                        source_kind=str(tool_args.get("source_kind", "screen")),
                        interval_seconds=int(tool_args.get("interval_seconds", 1800) or 1800),
                        monitor_condition=str(tool_args.get("monitor_condition", "")),
                        stop_condition=str(tool_args.get("stop_condition", "")),
                        source_scope=source_scope,
                        safe_actions=[str(item) for item in (tool_args.get("safe_actions") or [])],
                        origin_kind="chat",
                        origin_ref=str(metadata.get("chat_session_id") or ""),
                    )
                    response_content = json.dumps({"status": "created", "task": task.__dict__}, ensure_ascii=False)
                elif tool_name in {"list_recurring_tasks", "pause_recurring_task", "resume_recurring_task", "cancel_recurring_task"}:
                    if self.recurring_task_service is None:
                        raise RuntimeError("Recurring task scheduling is not configured.")
                    if tool_name == "list_recurring_tasks":
                        response_content = json.dumps(
                            {"tasks": [task.__dict__ for task in self.recurring_task_service.list(limit=100)]},
                            ensure_ascii=False,
                        )
                    else:
                        status = {
                            "pause_recurring_task": "paused",
                            "resume_recurring_task": "active",
                            "cancel_recurring_task": "cancelled",
                        }[tool_name]
                        task = self.recurring_task_service.set_status(str(tool_args.get("task_id", "")), status)
                        response_content = json.dumps(
                            {"status": status, "task": task.__dict__ if task else None}, ensure_ascii=False
                        )
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
                        record_tool_result(
                            tool_id=tool_id, tool_name=tool_name, arguments_json=tool_args_str,
                            output=response_content, ok=False,
                        )
                        continue
                    resident_model_name = self.llm.get_current_model()
                    parent_model_name = resident_model_name or self._frame.model
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
                        record_tool_result(
                            tool_id=tool_id, tool_name=tool_name, arguments_json=tool_args_str,
                            output=response_content, ok=False,
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
                    model_swapped = resident_model_name != model_name
                    saved_parent_kv_state = None
                    if model_swapped:
                        saved_parent_kv_state = await self.llm.save_and_unload(self._frame.messages)
                        if saved_parent_kv_state is None:
                            raise RuntimeError(
                                "Could not save the parent model state; sub-agent was not started."
                            )
                        await self.llm.load_model(model_name)
                    else:
                        self.logger.info(
                            "Sub-agent reusing resident model %s without a model transition.",
                            model_name,
                        )
                    self._push_frame(model=model_name, depth=agent_depth + 1)
                    try:
                        child_result = await self.run_interaction(
                            user_input="You have been given a task: " + tool_args.get("message", ""),
                            system_prompt=self.AGENT_PROMPT,
                            model=model_name,
                            agent_depth=agent_depth + 1,
                            allowed_tool_names=allowed_tool_names,
                            report_policy="silent",
                            final_turn_response_format="plain",
                        )
                    finally:
                        self._pop_frame()

                    current_model_name = self.llm.get_current_model()
                    if model_swapped and parent_model_name and current_model_name != parent_model_name:
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
                if self.capability_policy is not None:
                    self.capability_policy.finish_execution(
                        idempotency_key,
                        response_content,
                        succeeded=not response_content.startswith("Error:"),
                    )
            except InteractionSuspended as suspended:
                suspended.tool_call_id = tool_id
                record_tool_result(
                    tool_id=tool_id, tool_name=tool_name, arguments_json=tool_args_str,
                    output="Awaiting user approval before this tool can run.", ok=False,
                    status="awaiting_approval",
                )
                self._emit_event(
                    event_callback,
                    {
                        "type": "approval_required",
                        "tool_name": tool_name,
                        "tool_call_id": tool_id,
                        "approval_id": suspended.approval_id,
                        "delegation_id": suspended.delegation_id,
                    },
                )
                raise
            except Exception as e:
                response_content = f"Error: {str(e)}"
                if self.capability_policy is not None:
                    self.capability_policy.finish_execution(
                        locals().get("idempotency_key"), response_content, succeeded=False
                    )
            self._emit_event(
                event_callback,
                {
                    "type": "tool_finished",
                    "tool_name": tool_name,
                    "tool_call_id": tool_id,
                    "arguments_json": tool_args_str,
                    "result": response_content,
                    "ok": not response_content.startswith("Error:"),
                },
            )
            record_tool_result(
                tool_id=tool_id, tool_name=tool_name, arguments_json=tool_args_str,
                output=response_content, ok=not response_content.startswith("Error:"),
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
        substantive_outcome = bool(parsed.get("substantive_outcome", True))
        artifact = (
            await self._save_or_merge_report_artifact(
                model=report_model,
                title=title,
                summary=summary,
                detailed_report=detailed_report,
                source_name=source_name,
            )
            if substantive_outcome
            else {
                "artifact_path": "",
                "artifact_id": None,
                "artifact_action": "skipped",
                "artifact_reason": "Outcome was routine, duplicate, or not durable enough for an artifact.",
                "dedupe_notes": [],
            }
        )
        report = {
            "title": title,
            "summary": summary,
            "artifact_path": str(artifact["artifact_path"]),
            "artifact_filename": Path(str(artifact["artifact_path"])).name if artifact["artifact_path"] else "",
            "artifact_id": artifact.get("artifact_id"),
            "artifact_action": artifact.get("artifact_action"),
            "artifact_reason": artifact.get("artifact_reason"),
            "dedupe_notes": artifact.get("dedupe_notes", []),
            "source": source_name,
            "tools_used": deduped_tools,
            "created_at": datetime.now().isoformat(),
            "status": "completed",
            "substantive_outcome": substantive_outcome,
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

    async def _save_or_merge_report_artifact(
        self,
        *,
        model: str,
        title: str,
        summary: str,
        detailed_report: str,
        source_name: str,
    ) -> Dict[str, Any]:
        source_ref = f"{source_name}/{datetime.now().isoformat(timespec='seconds')}"
        if self.artifact_organizer is None:
            path = self._save_report_artifact(
                title=title,
                summary=summary,
                detailed_report=detailed_report,
            )
            return {
                "artifact_id": None,
                "artifact_path": str(path),
                "artifact_action": "created",
                "artifact_reason": "Artifact organizer disabled.",
                "dedupe_notes": [],
            }

        candidates = self.artifact_organizer.candidates_for(
            title=title,
            summary=summary,
            detailed_report=detailed_report,
        )
        if not candidates:
            return self.artifact_organizer.save_new(
                title=title,
                summary=summary,
                detailed_report=detailed_report,
                source_ref=source_ref,
            )

        payload = {
            "new_report": {
                "title": title,
                "summary": summary,
                "detailed_report": detailed_report,
                "source": source_name,
            },
            "candidate_artifacts": [
                candidate.prompt_summary(
                    summary_words=self.artifact_organizer.candidate_summary_words,
                )
                for candidate in candidates
            ],
            "existing_artifact_contents": self.artifact_organizer.build_existing_payload(candidates),
        }
        decision_text = await self._run_json_prompt(
            model=model,
            system_prompt=self.ARTIFACT_ORGANIZER_PROMPT,
            user_payload=payload,
        )
        decision = self._safe_parse_json(decision_text)
        if not isinstance(decision, dict):
            fallback = self.artifact_organizer.save_new(
                title=title,
                summary=summary,
                detailed_report=detailed_report,
                source_ref=source_ref,
            )
            fallback["artifact_reason"] = "Organizer model returned malformed JSON; created a new artifact safely."
            return fallback
        return self.artifact_organizer.apply_decision(
            decision=decision,
            fallback_title=title,
            fallback_summary=summary,
            fallback_detailed_report=detailed_report,
            source_ref=source_ref,
        )

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
