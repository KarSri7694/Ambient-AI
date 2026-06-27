import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack

sessions = {}
tool_ids = {}
exit_stack = None


async def start_servers(config_path: str):
    """Initialize and connect to all MCP servers"""
    global exit_stack
    exit_stack = AsyncExitStack()
    await exit_stack.__aenter__()
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    servers = config.get("mcpServers", {})

    for server_name, server_config in servers.items():
        print(f"Connecting to {server_name}...")
        
        # Prepare the connection parameters
        server_params = StdioServerParameters(
            command=server_config["command"],
            args=server_config.get("args", []),
            env=server_config.get("env")
        )

        try:
            # Connect to the server
            read, write = await exit_stack.enter_async_context(stdio_client(server_params))
            session = await exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            sessions[server_name] = session
        except Exception as e:
            print(f"Failed to connect to {server_name}: {e}")


async def get_all_mcp_tools():
    """Get all tools from connected MCP servers"""
    all_openai_tools = []

    for server_name, session in sessions.items():
        try:
            # Get tools from this specific server
            mcp_tools = await session.list_tools()
            
            for tool in mcp_tools.tools:
                tool_ids[tool.name] = server_name
                # Convert to OpenAI Format
                openai_tool = {
                    "type": "function",
                    "function":{
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                name: {k: v for k, v in vals.items() if k != 'title'}
                                for name, vals in tool.inputSchema.get('properties', {}).items()
                            },
                            "required": tool.inputSchema.get('required', [])
                        }
                    }
                }
                all_openai_tools.append(openai_tool)
        except Exception as e:
            print(f"Failed to fetch tools from {server_name}: {e}")

    return all_openai_tools

async def execute_tool(tool_name: str, tool_args: dict):
    server_name = tool_ids.get(tool_name)
    if not server_name:
        raise ValueError(f"Tool {tool_name} not found in any server.")
    session = sessions[server_name]    
    response = await session.call_tool(tool_name, tool_args)
    return response

async def cleanup():
    """Clean up all MCP connections"""
    global exit_stack
    if exit_stack:
        await exit_stack.__aexit__(None, None, None)
    
async def main():
    config_file = "mcp.json" 
    try:
        await start_servers(config_file)
        final_tools = await get_all_mcp_tools()
        tool_response = await execute_tool("get_current_datetime", {})
        print(f"\nTool Response: {tool_response}\n")
    finally:
        await cleanup()

if __name__ == "__main__":
    asyncio.run(main())
    

