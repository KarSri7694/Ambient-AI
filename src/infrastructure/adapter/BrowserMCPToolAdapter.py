import json
import logging
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from application.ports.tool_bridge_port import (
    BrowserToolBridgePort,
    BrowserToolSessionPort,
)


class BrowserMCPToolSession(BrowserToolSessionPort):
    """One ephemeral Playwright MCP connection used by one browser agent."""

    UNSAFE_DESCRIPTION_MARKERS = (
        "arbitrary code execution",
        "rce-equivalent",
        "rce equivalent",
    )

    def __init__(
        self,
        *,
        server_name: str,
        server_config: Dict[str, Any],
        headless: bool,
        profile_dir: Path,
        denied_tool_names: Set[str],
    ):
        self.server_name = server_name
        self.server_config = dict(server_config)
        self.headless = headless
        self.profile_dir = profile_dir
        self.denied_tool_names = {name.lower() for name in denied_tool_names}
        self.logger = logging.getLogger(self.__class__.__name__)
        self._exit_stack: Optional[AsyncExitStack] = None
        self._session: Optional[ClientSession] = None
        self._allowed_tool_names: Set[str] = set()

    async def start(self) -> None:
        if self._session is not None:
            return

        args = self.launch_args()
        params = StdioServerParameters(
            command=self.server_config["command"],
            args=args,
            env=self.server_config.get("env"),
        )
        self._exit_stack = AsyncExitStack()
        try:
            read, write = await self._exit_stack.enter_async_context(stdio_client(params))
            self._session = await self._exit_stack.enter_async_context(ClientSession(read, write))
            await self._session.initialize()
        except Exception:
            await self.cleanup()
            raise

    def launch_args(self) -> List[str]:
        """Build Playwright MCP arguments for the selected browser mode."""
        args = list(self.server_config.get("args", []))
        if self.headless:
            if "--headless" not in args:
                args.append("--headless")
            if "--isolated" not in args:
                args.append("--isolated")
        else:
            self.profile_dir.mkdir(parents=True, exist_ok=True)
            if not any(arg == "--user-data-dir" or arg.startswith("--user-data-dir=") for arg in args):
                args.extend(["--user-data-dir", str(self.profile_dir.resolve())])
        return args

    def _is_allowed(self, *, name: str, description: str) -> bool:
        normalized_name = name.strip().lower()
        normalized_description = (description or "").lower()
        if normalized_name in self.denied_tool_names or "_unsafe" in normalized_name:
            return False
        return not any(
            marker in normalized_description
            for marker in self.UNSAFE_DESCRIPTION_MARKERS
        )

    async def get_all_tools(self) -> List[Dict[str, Any]]:
        if self._session is None:
            raise RuntimeError("Browser MCP session has not been started.")

        response = await self._session.list_tools()
        tools: List[Dict[str, Any]] = []
        self._allowed_tool_names.clear()
        for tool in response.tools:
            description = tool.description or ""
            if not self._is_allowed(name=tool.name, description=description):
                self.logger.info("Hiding unsafe browser MCP tool: %s", tool.name)
                continue
            self._allowed_tool_names.add(tool.name)
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": description,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                name: {k: v for k, v in values.items() if k != "title"}
                                for name, values in tool.inputSchema.get("properties", {}).items()
                            },
                            "required": tool.inputSchema.get("required", []),
                        },
                    },
                }
            )
        return tools

    async def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        if self._session is None:
            raise RuntimeError("Browser MCP session has not been started.")
        if tool_name not in self._allowed_tool_names:
            raise ValueError(f"Browser tool '{tool_name}' is not allowed in this session.")

        response = await self._session.call_tool(tool_name, tool_args)
        if hasattr(response, "content"):
            return "\n".join(
                item.text if hasattr(item, "text") else str(item)
                for item in response.content
            )
        return str(response)

    async def cleanup(self) -> None:
        stack = self._exit_stack
        self._exit_stack = None
        self._session = None
        self._allowed_tool_names.clear()
        if stack is not None:
            await stack.aclose()


class BrowserMCPToolAdapter(BrowserToolBridgePort):
    """Creates task-scoped sessions from a browser-only MCP server config."""

    def __init__(
        self,
        *,
        config_path: str,
        server_name: str,
        profile_dir: str,
        denied_tool_names: Optional[Set[str]] = None,
    ):
        self.config_path = config_path
        self.server_name = server_name
        self.profile_dir = Path(profile_dir)
        self.denied_tool_names = set(denied_tool_names or set())

    def _load_server_config(self) -> Dict[str, Any]:
        with open(self.config_path, "r", encoding="utf-8") as config_file:
            config = json.load(config_file)
        server_config = config.get("mcpServers", {}).get(self.server_name)
        if not isinstance(server_config, dict):
            raise ValueError(
                f"Browser MCP server '{self.server_name}' was not found in {self.config_path}."
            )
        if server_config.get("exposure") != "browser_agent":
            raise ValueError(
                f"Browser MCP server '{self.server_name}' must set exposure='browser_agent'."
            )
        if not server_config.get("command"):
            raise ValueError(f"Browser MCP server '{self.server_name}' has no command.")
        return server_config

    async def open_session(self, *, headless: bool) -> BrowserMCPToolSession:
        session = BrowserMCPToolSession(
            server_name=self.server_name,
            server_config=self._load_server_config(),
            headless=headless,
            profile_dir=self.profile_dir,
            denied_tool_names=self.denied_tool_names,
        )
        await session.start()
        return session
