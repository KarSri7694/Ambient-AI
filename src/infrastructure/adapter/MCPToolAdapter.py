import json
import logging
from contextlib import AsyncExitStack
from typing import List, Dict, Any, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from application.ports.tool_bridge_port import ToolBridgePort


class MCPToolAdapter(ToolBridgePort):
    """
    Self-contained MCP adapter
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._sessions: Dict[str, ClientSession] = {}
        self._tool_server_map: Dict[str, str] = {}  # tool_name -> server_name
        self._exit_stack: Optional[AsyncExitStack] = None

    async def start_servers(self, config_path: str) -> None:
        self._exit_stack = AsyncExitStack()

        with open(config_path, "r") as f:
            config = json.load(f)

        for server_name, server_config in config.get("mcpServers", {}).items():
            self.logger.info(f"Connecting to MCP server: {server_name}")
            params = StdioServerParameters(
                command=server_config["command"],
                args=server_config.get("args", []),
                env=server_config.get("env"),
            )
            try:
                read, write = await self._exit_stack.enter_async_context(stdio_client(params))
                session = await self._exit_stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                self._sessions[server_name] = session
                self.logger.info(f"Connected to {server_name}")
            except Exception as e:
                self.logger.error(f"Failed to connect to {server_name}: {e}")

    async def get_all_tools(self) -> List[Dict[str, Any]]:
        all_tools: List[Dict[str, Any]] = []
        for server_name, session in self._sessions.items():
            try:
                mcp_tools = await session.list_tools()
                for tool in mcp_tools.tools:
                    self._tool_server_map[tool.name] = server_name
                    all_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    name: {k: v for k, v in vals.items() if k != "title"}
                                    for name, vals in tool.inputSchema.get("properties", {}).items()
                                },
                                "required": tool.inputSchema.get("required", []),
                            },
                        },
                    })
            except Exception as e:
                self.logger.error(f"Failed to fetch tools from {server_name}: {e}")
        return all_tools

    async def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        server_name = self._tool_server_map.get(tool_name)
        if not server_name:
            raise ValueError(f"Tool '{tool_name}' not found in any connected server.")
        try:
            response = await self._sessions[server_name].call_tool(tool_name, tool_args)
            if hasattr(response, "content"):
                return "\n".join(
                    item.text if hasattr(item, "text") else str(item)
                    for item in response.content
                )
            return str(response)
        except Exception as e:
            self.logger.error(f"Error executing tool {tool_name}: {e}")
            return f"Error: {str(e)}"

    async def cleanup(self) -> None:
        if self._exit_stack:
            try:
                await self._exit_stack.aclose()
                self.logger.info("MCP cleanup completed.")
            except Exception as e:
                self.logger.error(f"Error during MCP cleanup: {e}")
            finally:
                self._exit_stack = None
                self._sessions.clear()
                self._tool_server_map.clear()
