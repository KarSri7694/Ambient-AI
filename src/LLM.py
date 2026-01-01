import openai
import llama_MCP_bridge
import asyncio
import json

api_uri= "http://localhost:8080/v1"

client = openai.OpenAI(
    base_url=api_uri,
    api_key="testkey"
)
tools = None
async def start_mcp():
    try:
        await llama_MCP_bridge.start_servers("mcp.json")
    except Exception as e:
        print(f"Error starting MCP servers: {e}")

async def get_tools():
    try:
        global tools
        tools = await llama_MCP_bridge.get_all_mcp_tools()
        return tools
    except Exception as e:
        print(f"Error retrieving MCP tools: {e}")
        return []
    
async def execute_tool(tool_name, tool_args):
    try:
        result = await llama_MCP_bridge.execute_tool(tool_name, tool_args)
        return result
    except Exception as e:
        print(f"Error executing tool {tool_name}: {e}")
        return None

async def cleanup_mcp():
    try:
        await llama_MCP_bridge.cleanup()
    except Exception as e:
        print(f"Error during MCP cleanup: {e}")
    
async def start_input_loop():
    message = []
    print(f"\n\nEnter messages to send to the model (type 'exit' to quit):\n")
    while True:
        user_input = input()
        if user_input.lower() in ["exit"]:
            break
        user_message = {"role": "user", "content": user_input}
        message.append(user_message)
        
        completion = client.chat.completions.create(
            model="Qwen3-4B-Thinking-2507-Q4_K_M.gguf",
            messages=message,    
            tools=tools,
            stream=True
        )

        assistant_message = ""
        tool_calls = []
        current_tool_call = None
        
        print("Streaming response:")
        for chunk in completion:
            delta = chunk.choices[0].delta
            
            # 1. Check for standard content (The final answer)
            if delta.content:
                print(delta.content, end="", flush=True)
                assistant_message += delta.content
                
            # 2. Check for reasoning content (The "Thought")
            # We use getattr or dict access because the attribute might not exist on the object
            reasoning = getattr(delta, 'reasoning_content', None)
            if reasoning:
                # Print it differently so you know it's reasoning (e.g., in yellow)
                print(f"\033[93m{reasoning}\033[0m", end="", flush=True)    
            # print(delta)   
            
            if delta.tool_calls:
                for tool_call_delta in delta.tool_calls:
                    index = tool_call_delta.index
                    
                    # Initialize new tool call if needed
                    while len(tool_calls) <= index:
                        tool_calls.append({
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""}
                        })
                    
                    # Update tool call ID
                    if tool_call_delta.id:
                        tool_calls[index]["id"] = tool_call_delta.id
                    
                    # Update function name
                    if tool_call_delta.function.name:
                        tool_calls[index]["function"]["name"] += tool_call_delta.function.name
                    
                    # Update function arguments
                    if tool_call_delta.function.arguments:
                        tool_calls[index]["function"]["arguments"] += tool_call_delta.function.arguments
        
        print("\n")
        
        # If tool calls were made, execute them
        if tool_calls:
            # Add assistant message with tool calls
            assistant_message_obj = {
                "role": "assistant",
                "content": assistant_message if assistant_message else None,
                "tool_calls": tool_calls
            }
            message.append(assistant_message_obj)
            
            # Execute each tool call
            for tool_call in tool_calls:
                tool_name = tool_call["function"]["name"]
                tool_args_str = tool_call["function"]["arguments"]
                tool_id = tool_call["id"]
                
                print(f"\n🔧 Calling tool: {tool_name}")
                print(f"   Arguments: {tool_args_str}")
                
                try:
                    tool_args = json.loads(tool_args_str) if tool_args_str else {}
                    tool_response = await execute_tool(tool_name, tool_args)
                    
                    # Convert response to string
                    if hasattr(tool_response, 'content'):
                        response_content = str(tool_response.content)
                    else:
                        response_content = str(tool_response)
                    
                    print(f"   Result: {response_content}\n")
                    
                    # Add tool response to messages
                    tool_response_message = {
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "name": tool_name,
                        "content": response_content
                    }
                    message.append(tool_response_message)
                    
                except Exception as e:
                    print(f"   Error: {e}\n")
                    tool_response_message = {
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "name": tool_name,
                        "content": f"Error: {str(e)}"
                    }
                    message.append(tool_response_message)
            
            # Get the final response after tool execution
            print("Getting final response after tool execution...\n")
            completion = client.chat.completions.create(
                model="Qwen3-4B-Thinking-2507-Q4_K_M.gguf",
                messages=message,    
                tools=tools,
                stream=True
            )
            
            final_response = ""
            for chunk in completion:
                delta = chunk.choices[0].delta
                if delta.content:
                    print(delta.content, end="", flush=True)
                    final_response += delta.content
                
                reasoning = getattr(delta, 'reasoning_content', None)
                if reasoning:
                    print(f"\033[93m{reasoning}\033[0m", end="", flush=True)
            
            message.append({"role": "assistant", "content": final_response})
        else:
            # No tool calls, just add the assistant message
            assistant_message_obj = {"role": "assistant", "content": assistant_message}
            message.append(assistant_message_obj)
        
        print("\n")  
        print("User-->")

async def main():
    try:
        await start_mcp()
        tools = await get_tools()
        await start_input_loop()
    finally:
        await cleanup_mcp()

if __name__ == "__main__":
    asyncio.run(main())