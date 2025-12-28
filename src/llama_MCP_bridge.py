import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def get_all_mcp_tools(config_path: str):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    all_openai_tools = []
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
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    # Get tools from this specific server
                    mcp_tools = await session.list_tools()
                    
                    for tool in mcp_tools.tools:
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

# if __name__ == "__main__":
#     # Path to your mcp.json
#     config_file = "mcp.json" 
#     final_tools = asyncio.run(get_all_mcp_tools(config_file))
    
#     print(json.dumps(final_tools, indent=2))
#     print(f"\nTotal Tools Found: {len(final_tools)}")    
    

