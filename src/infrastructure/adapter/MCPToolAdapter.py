import logging
from typing import List, Dict, Any

import llama_MCP_bridge
from application.ports.tool_bridge_port import ToolBridgePort


class MCPToolAdapter(ToolBridgePort):
    """Adapter that wraps the llama_MCP_bridge module behind the ToolBridgePort interface."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    async def start_servers(self, config_path: str) -> None:
        try:
            await llama_MCP_bridge.start_servers(config_path)
        except Exception as e:
            self.logger.error(f"Error starting MCP servers: {e}")

    async def get_all_tools(self) -> List[Dict[str, Any]]:
        try:
            tools = await llama_MCP_bridge.get_all_mcp_tools()
            return tools
        except Exception as e:
            self.logger.error(f"Error retrieving MCP tools: {e}")
            return []

    async def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """Execute a tool and return the result as a plain string."""
        try:
            tool_response = await llama_MCP_bridge.execute_tool(tool_name, tool_args)

            # Convert MCP CallToolResult to string
            if hasattr(tool_response, 'content'):
                content_parts = []
                for item in tool_response.content:
                    if hasattr(item, 'text'):
                        content_parts.append(item.text)
                    else:
                        content_parts.append(str(item))
                return '\n'.join(content_parts)
            else:
                return str(tool_response)
        except Exception as e:
            self.logger.error(f"Error executing tool {tool_name}: {e}")
            return f"Error: {str(e)}"

    async def cleanup(self) -> None:
        try:
            await llama_MCP_bridge.cleanup()
            self.logger.info("MCP cleanup completed.")
        except Exception as e:
            self.logger.error(f"Error during MCP cleanup: {e}")
