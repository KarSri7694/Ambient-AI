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
