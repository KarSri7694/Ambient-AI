import openai
import llama_MCP_bridge
import asyncio
import json
import requests

api_uri= "http://localhost:8080"
tools = None
s_prompt = """You are an advanced AI assistant integrated with various tools to help users with their requests.
When you need to perform a specific action or retrieve information, you can call the appropriate tool by name, providing the necessary arguments in JSON format.
Make sure to use the tools effectively to assist the user. """

client = openai.OpenAI(
    base_url=api_uri,
    api_key="testkey"
)

currently_loaded_model = None

def read_transcription(file_path):
    with open(file_path, 'r') as f:
        transcription = f.read()
    return transcription

async def load_model(model_name: str):
    global currently_loaded_model
    if currently_loaded_model == model_name:
        print(f"Model {model_name} is already loaded.")
        return
    
    # Unload previous model if any
    if currently_loaded_model is not None:
        unload_model(currently_loaded_model)
    
    model = {
        "model": model_name
    }
    
    response = requests.post(f"{api_uri}/models/load", json=model)
    if response.status_code == 200:
        print(f"Successfully loaded model: {model_name}")
        currently_loaded_model = model_name
    else:
        print(f"Failed to load model: {model_name}. Response: {response.text}")    

async def unload_model(model_name: str):
    global currently_loaded_model
    model = {
        "model": model_name
    }
    
    response = requests.post(f"{api_uri}/models/unload", json=model)
    if response.status_code == 200:
        print(f"Successfully unloaded model: {model_name}")
        currently_loaded_model = None
    else:
        print(f"Failed to unload model: {model_name}. Response: {response.text}")

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
    system_prompt = {"role": "system", "content": s_prompt}
    message.append(system_prompt)
    print(f"\n\nEnter messages to send to the model (type 'exit' to quit):\n")
    while True:
        user_input = input()
        if user_input.lower() in ["exit"]:
            break
        user_message = {"role": "user", "content": user_input}
        message.append(user_message)
        
        # Loop until the model stops calling tools
        max_iterations = 10  # Safety limit to prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")
            
            completion = client.chat.completions.create(
                model="Qwen3-4B-Thinking-2507-Q4_K_M.gguf",
                messages=message,    
                tools=tools,
                stream=True
            )

            assistant_message = ""
            tool_calls = []
            
            print("Streaming response:")
            for chunk in completion:
                delta = chunk.choices[0].delta
                
                # 1. Check for standard content (The final answer)
                if delta.content:
                    print(delta.content, end="", flush=True)
                    assistant_message += delta.content
                    
                # 2. Check for reasoning content (The "Thought")
                reasoning = getattr(delta, 'reasoning_content', None)
                if reasoning:
                    print(f"\033[93m{reasoning}\033[0m", end="", flush=True)    
                
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
            
            # If NO tool calls were made, the model is done
            if not tool_calls:
                # Add final assistant message and break the loop
                assistant_message_obj = {"role": "assistant", "content": assistant_message}
                message.append(assistant_message_obj)
                print("✓ Model finished (no more tool calls)\n")
                break
            
            # If tool calls were made, execute them and continue the loop
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
            
            # Loop continues - model will see tool results and decide next action
        
        if iteration >= max_iterations:
            print(f"⚠️  Reached maximum iterations ({max_iterations}). Stopping.\n")
        
        print("User-->")

async def main():
    try:
        await load_model("Qwen-4b-Thinking-2507-Q4_K_M")
        await start_mcp()
        tools = await get_tools()
        await start_input_loop()
    finally:
        await unload_model("Qwen-4b-Thinking-2507-Q4_K_M")
        await cleanup_mcp()

if __name__ == "__main__":
    asyncio.run(main())