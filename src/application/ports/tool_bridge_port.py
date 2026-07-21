from abc import ABC, abstractmethod
from typing import List, Dict, Any


class ToolBridgePort(ABC):
    """Port for interacting with external tool systems (e.g. MCP servers)."""

    @abstractmethod
    async def start_servers(self, config_path: str) -> None:
        """Initialize and connect to all tool servers."""
        pass

    @abstractmethod
    async def get_all_tools(self) -> List[Dict[str, Any]]:
        """Retrieve all available tool definitions in OpenAI function-calling format."""
        pass

    @abstractmethod
    async def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """Execute a tool by name with the given args. Returns the result as a string."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up all tool server connections."""
        pass


class BrowserToolSessionPort(ABC):
    """A task-scoped connection to a browser MCP server."""

    @abstractmethod
    async def get_all_tools(self) -> List[Dict[str, Any]]:
        """Return only the browser tools allowed for the delegated agent."""
        pass

    @abstractmethod
    async def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """Execute an allowed browser tool inside this session."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Close the browser MCP process and its transport."""
        pass


class BrowserToolBridgePort(ABC):
    """Factory for isolated, task-scoped browser MCP sessions."""

    @abstractmethod
    async def open_session(self, *, headless: bool) -> BrowserToolSessionPort:
        """Start and return a browser MCP session for one delegated task."""
        pass
