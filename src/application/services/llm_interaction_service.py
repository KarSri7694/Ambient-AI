import json
import logging
from typing import List, Dict, Any, Optional

from application.ports.LLMProvider import LLMProvider
from application.ports.tool_bridge_port import ToolBridgePort


class LLMInteractionService:
    """
    Orchestrates the LLM streaming chat loop with tool execution.

    This service knows nothing about concrete LLM backends or tool systems —
    it only depends on the LLMProvider and ToolBridgePort abstractions.
    """

    MAX_ITERATIONS = 1

    def __init__(self, llm_provider: LLMProvider, tool_bridge: ToolBridgePort):
        self.llm = llm_provider
        self.tool_bridge = tool_bridge
        self.logger = logging.getLogger(self.__class__.__name__)
        self._tools: Optional[List[Dict[str, Any]]] = None
        self._messages: List[Dict[str, Any]] = []

    async def initialize_tools(self) -> None:
        """Fetch available tools from the tool bridge."""
        self._tools = await self.tool_bridge.get_all_tools()

    def reset_conversation(self) -> None:
        """Clear the message history for a new conversation."""
        self._messages = []

    def get_context(self) -> List[Dict[str, Any]]:
        """Return a snapshot of the current message history."""
        return list(self._messages)

    def restore_context(self, messages: List[Dict[str, Any]]) -> None:
        """Restore a previously saved message history."""
        self._messages = list(messages)

    async def run_interaction(
        self,
        user_input: str,
        system_prompt: str,
        model: str,
        image_path: str = "",
    ) -> str:
        """
        Run a full LLM interaction: send user input, stream response,
        execute any tool calls, loop until the model is done.

        Returns the final assistant text response.
        """
        self._messages = []
        self._messages.append({"role": "system", "content": system_prompt})
        self._messages.append({"role": "user", "content": user_input})

        iteration = 0

        while iteration < self.MAX_ITERATIONS:
            iteration += 1
            self.logger.info(f"--- Iteration {iteration} ---")

            completion = self.llm.chat_completion_stream(
                model=model,
                messages=self._messages,
                tools=self._tools,
                image=image_path if iteration == 1 else "",
            )

            assistant_text, tool_calls = self._consume_stream(completion)

            # If NO tool calls were made, the model is done
            if not tool_calls:
                self._messages.append({"role": "assistant", "content": assistant_text})
                self.logger.info("✓ Model finished (no more tool calls)")
                return assistant_text

            # Record assistant message with tool calls
            self._messages.append({
                "role": "assistant",
                "content": assistant_text if assistant_text else None,
                "tool_calls": tool_calls,
            })

            # Execute each tool call
            await self._execute_tool_calls(tool_calls)

        self.logger.warning(
            f"⚠️  Reached maximum iterations ({self.MAX_ITERATIONS}). Stopping."
        )
        return assistant_text

    def _consume_stream(self, completion) -> tuple[str, List[Dict]]:
        """
        Consume a streaming completion iterator.
        Returns (assistant_text, tool_calls_list).
        """
        assistant_text = ""
        tool_calls: List[Dict] = []

        for chunk in completion:
            delta = chunk.choices[0].delta

            # Standard content (the final answer text)
            if delta.content:
                print(delta.content, end="", flush=True)
                assistant_text += delta.content

            # Reasoning content (thinking / chain-of-thought)
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                print(f"\033[93m{reasoning}\033[0m", end="", flush=True)

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

        print()  # newline after stream
        return assistant_text, tool_calls

    async def _execute_tool_calls(self, tool_calls: List[Dict]) -> None:
        """Execute tool calls and append results to the message history."""
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            tool_args_str = tool_call["function"]["arguments"]
            tool_id = tool_call["id"]

            self.logger.info(f"🔧 Calling tool: {tool_name}")
            self.logger.info(f"   Arguments: {tool_args_str}")

            try:
                tool_args = json.loads(tool_args_str) if tool_args_str else {}
                response_content = await self.tool_bridge.execute_tool(tool_name, tool_args)
                self.logger.info(f"   Result: {response_content}")
            except Exception as e:
                self.logger.error(f"   Error: {e}")
                response_content = f"Error: {str(e)}"

            self._messages.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "name": tool_name,
                "content": response_content,
            })
