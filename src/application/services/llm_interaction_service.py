import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from application.ports.LLMProvider import LLMProvider
from application.ports.tool_bridge_port import ToolBridgePort
from utils.kv_state_handling import KVStateControl


@dataclass
class AgentFrame:
    messages: List[Dict[str, Any]] = field(default_factory=list)
    model: Optional[str] = None
    depth: int = 0

class LLMInteractionService:
    """
    Orchestrates the LLM streaming chat loop with tool execution.

    This service knows nothing about concrete LLM backends or tool systems —
    it only depends on the LLMProvider and ToolBridgePort abstractions.
    """

    AGENT_DEPTH = 0
    MAX_AGENT_DEPTH = 3
    MAX_ITERATIONS = 25
    TERMINAL_TOOL_NAMES = {"restore_previous_agent"}
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
    
    def __init__(self, llm_provider: LLMProvider, tool_bridge: ToolBridgePort):
        self.llm = llm_provider
        self.tool_bridge = tool_bridge
        self.logger = logging.getLogger(self.__class__.__name__)
        self._tools: Optional[List[Dict[str, Any]]] = None
        self._frame_stack: List[AgentFrame] = [AgentFrame()]

    @property
    def _frame(self) -> AgentFrame:
        return self._frame_stack[-1]

    def _push_frame(self, model: str, depth: int) -> None:
        self._frame_stack.append(AgentFrame(model=model, depth=depth))

    def _pop_frame(self) -> AgentFrame:
        if len(self._frame_stack) == 1:
            raise RuntimeError("Cannot pop root agent frame.")
        return self._frame_stack.pop()

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
            recovered_calls.append({
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(args),
                },
            })
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
        if not self._tools:
            return []
        if agent_depth < self.MAX_AGENT_DEPTH:
            tools = list(self._tools)
        else:
            filtered_tools: List[Dict[str, Any]] = []
            for tool in self._tools:
                function_meta = tool.get("function", {})
                tool_name = function_meta.get("name")
                if tool_name in {"load_agent", "list_available_models"}:
                    continue
                filtered_tools.append(tool)
            tools = filtered_tools

        if allowed_tool_names is None:
            return tools

        filtered_tools: List[Dict[str, Any]] = []
        for tool in tools:
            function_meta = tool.get("function", {})
            tool_name = function_meta.get("name")
            if tool_name in allowed_tool_names:
                filtered_tools.append(tool)
        return filtered_tools

    async def run_interaction(
        self,
        user_input: str,
        system_prompt: str,
        model: str,
        image_path: str = "",
        agent_depth: int = 0,
        allowed_tool_names: Optional[set[str]] = None,
    ) -> str:
        """
        Run a full LLM interaction: send user input, stream response,
        execute any tool calls, loop until the model is done.

        Returns the final assistant text response.
        """
        self._frame.model = model
        self._frame.depth = agent_depth
        if not self._frame.messages:
            self._frame.messages.append({"role": "system", "content": system_prompt})
        self._frame.messages.append({"role": "user", "content": user_input})

        iteration = 0

        while iteration < self.MAX_ITERATIONS:
            iteration += 1
            self.logger.info(
                f"--- Iteration {iteration} (agent depth {agent_depth}/{self.MAX_AGENT_DEPTH}) ---"
            )

            completion = await self.llm.chat_completion_stream(
                model=model,
                messages=self._frame.messages,
                tools=self._tools_for_agent_depth(agent_depth, allowed_tool_names=allowed_tool_names),
                image=image_path if iteration == 1 else "",
            )

            assistant_text, tool_calls = await self._consume_stream(completion)

            # If NO tool calls were made, the model is done
            if not tool_calls:
                self._frame.messages.append({"role": "assistant", "content": assistant_text})
                self.logger.info("✓ Model finished (no more tool calls)")
                return assistant_text

            # Record assistant message with tool calls
            self._frame.messages.append({
                "role": "assistant",
                "content": assistant_text if assistant_text else None,
                "tool_calls": tool_calls,
            })

            # Execute each tool call
            tool_results = await self._execute_tool_calls(
                tool_calls,
                agent_depth=agent_depth,
                allowed_tool_names=allowed_tool_names,
            )
            if any(name in self.TERMINAL_TOOL_NAMES for name, _ in tool_results):
                self.logger.info("Terminal tool executed; ending interaction loop.")
                # if tool_results[0][0] == "save_state":
                #     result = tool_results[0][1]
                #     filename = result.split("State save requested successfully: ")[-1].strip().split(".bin")[0]+".json"
                #     with open((self.kv_state_dir / filename).absolute(), "w") as f:
                #         json.dump(self._messages, f)
                #     self.logger.info(f"Conversation state saved to {filename}")
                # if tool_results[0][0] == "restore_state":
                #     result = tool_results[0][1]
                #     filename = result.split("State restoration requested successfully: ")[-1].strip().split(".bin")[0]+".json"
                #     with open((self.kv_state_dir / filename).absolute(), "r") as f:
                #         messages = json.load(f)
                #     self.reset_context()
                #     self.restore_context(messages)
                #     self.logger.info(f"Conversation state restored from {filename}")
                
                return "\n".join(result for _, result in tool_results)

        self.logger.warning(
            f"Reached maximum iterations: ({self.MAX_ITERATIONS}). Stopping."
        )
        return assistant_text

    async def _consume_stream(self, completion) -> tuple[str, List[Dict]]:
        """
        Consume a streaming completion iterator.
        Returns (assistant_text, tool_calls_list).
        """
        assistant_text = ""
        raw_assistant_text = ""
        raw_reasoning_text = ""
        tool_calls: List[Dict] = []

        async for chunk in completion:
            delta = chunk.choices[0].delta

            # Standard content (the final answer text)
            if delta.content:
                raw_assistant_text += delta.content
                content = self._strip_think_tags(delta.content)
                if content:
                    print(content, end="", flush=True)
                    assistant_text += content

            # Reasoning content (thinking / chain-of-thought)
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                raw_reasoning_text += reasoning
                cleaned_reasoning = self._strip_think_tags(reasoning)
                if cleaned_reasoning:
                    print(f"\033[93m{cleaned_reasoning}\033[0m", end="", flush=True)

            # Tool calls
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    index = tc_delta.index
                    while len(tool_calls) <= index:
                        tool_calls.append({
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        })
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
        print()  # newline after stream
        return assistant_text, tool_calls

    async def _execute_tool_calls(
        self,
        tool_calls: List[Dict],
        agent_depth: int = 0,
        allowed_tool_names: Optional[set[str]] = None,
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

            self.logger.info(f"Calling tool: {tool_name}")
            self.logger.info(f"Arguments: {tool_args_str}")

            try:
                tool_args = json.loads(tool_args_str) if tool_args_str else {}
                if allowed_tool_names is not None and tool_name not in allowed_tool_names:
                    response_content = f"Error: tool '{tool_name}' is not allowed in this interaction."
                    self.logger.warning(response_content)
                    tool_results.append((tool_name, response_content))
                    self._frame.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "name": tool_name,
                        "content": response_content,
                    })
                    continue
                if tool_name == "load_agent":
                    if agent_depth >= self.MAX_AGENT_DEPTH:
                        response_content = (
                            f"Error: maximum agent depth reached ({self.MAX_AGENT_DEPTH}). "
                            "Further agent spawning is not allowed."
                        )
                        self.logger.warning(response_content)
                        tool_results.append((tool_name, response_content))
                        self._frame.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "name": tool_name,
                            "content": response_content,
                        })
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
                        self._frame.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "name": tool_name,
                            "content": response_content,
                        })
                        continue
                    if available_model_names and model_name not in available_model_names:
                        response_content = (
                            "Error: unknown model_name "
                            f"'{model_name}'. Call list_available_models first and use one "
                            f"of: {', '.join(available_model_names)}"
                        )
                        self.logger.warning(response_content)
                        tool_results.append((tool_name, response_content))
                        self._frame.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "name": tool_name,
                            "content": response_content,
                        })
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
                    response_content = f"Agent task completed. Restoring previous agent."
                    kv_state_file = await self.llm.load_and_restore()
                    messages_path = kv_state_file.with_suffix(".json")
                    if messages_path.exists():
                        with open(messages_path, "r") as f:
                            messages = json.load(f)
                        target_frame = self._frame_stack[-2] if len(self._frame_stack) > 1 else self._frame
                        target_frame.messages = list(messages)
                        target_frame.messages.append({
                            "role": "assistant",
                            "content": f"[Sub-Agent] Deployed sub-agent has worked on the given task and returned the following result: {tool_args.get('message_to_agent', '')}."
                        })
                        self.logger.info(f"Conversation state restored from {messages_path.name}")
                    else:
                        self.logger.warning(f"No conversation state file found at {messages_path}. Context not restored.")
                else:
                    response_content = await self.tool_bridge.execute_tool(tool_name, tool_args)
                self.logger.info(f"   Result: {response_content}")
            except Exception as e:
                self.logger.error(f"   Error: {e}")
                response_content = f"Error: {str(e)}"
            tool_results.append((tool_name, response_content))

            self._frame.messages.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "name": tool_name,
                "content": response_content,
            })
        return tool_results
