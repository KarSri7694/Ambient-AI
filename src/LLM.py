import openai
import llama_MCP_bridge
import asyncio
import json
import requests
from pathlib import Path
import night_shift
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from utils.todoist_helper import TodoistHelper

class Handler(FileSystemEventHandler):
    def on_created(self, event):
        return event.src_path
            
observer = Observer()
observer.schedule(FileSystemEventHandler(), path='transcriptions/', recursive=False)
observer.start()

api_uri= "http://localhost:8080"
api_uri_v1 = f"{api_uri}/v1"
tools = None
transcription_files_content = {}

username = ""
s_prompt = f"""
You are assitant of {username}, you have to help him in his daily tasks.
You have access to various tools to help you accomplish tasks. The user will pass a conversation to you, you have to interpret it and decide which tools to use to best assist the user.
When you need to perform a task, use the appropriate tool from the available tools list 
if possible convert text to english before passing to tools
Any task that requires extensive time or resources should be queued for night-time execution using the `queue_night_task` tool. This reqires research, downloading large files, or any task that the user specifies to be done at night.
If a task is SLOW (Deep Research, Downloading huge files), use the `queue_night_task` tool. Dont use tavily for such tasks.
For all else, use standard tools.
You will also be provided with notifications from the system about important events that were started. Take these into account when assisting the user.
"""

night_shift_prompt = """
You are an autonomous agent working through a list of night-time tasks queued by Kartikeya Srivastava.
You have access to various tools to help you accomplish these tasks.
DO NOT use the `queue_night_task` tool here.
"""

client = openai.OpenAI(
    base_url=api_uri_v1,
    api_key="testkey"
)

message = []
currently_loaded_model = None

def read_transcription(transcriptions_dir):
    global transcription_files
    path = Path(transcriptions_dir)
    transcription_files = [f for f in path.iterdir() if f.is_file() and f.suffix == '.txt']
    for file in transcription_files:
        with open(file, 'r') as f:
            content = f.read()
            transcription_files_content[file.name] = content
         
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
    return    

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
        print("MCP cleanup completed.")
    except Exception as e:
        print(f"Error during MCP cleanup: {e}")

async def start_llm_interaction(mode:str = "user", user_input:str = "", system_prompt:str = s_prompt):
    global message
    system_prompt = {"role": "system", "content": system_prompt}
    message.append(system_prompt)
    print(f"\n\nEnter messages to send to the model (type 'exit' to quit):")
    user_message = {"role": "user", "content": user_input}
    message.append(user_message)    
    
    # Loop until the model stops calling tools
    max_iterations = 10  # Safety limit to prevent infinite loops
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")
        
        completion = client.chat.completions.create(
            model=currently_loaded_model,
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
                
                # Convert response to string - MCP returns CallToolResult with content list
                if hasattr(tool_response, 'content'):
                    # content is a list of TextContent/ImageContent/etc objects
                    content_parts = []
                    for item in tool_response.content:
                        if hasattr(item, 'text'):
                            content_parts.append(item.text)
                        else:
                            content_parts.append(str(item))
                    response_content = '\n'.join(content_parts)
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
    
async def start_input_loop():
    #Set modes 
    while True:
        print("1. Enter User interaction mode")
        print("2. Enter Transcription Automation mode")
        print("3. Enter late night execution mode")
        print("Type 'exit' to quit.\n\n")
        mode = input("Select mode (1, 2, or 3): ")
        notifications = night_shift.get_unread_notifications()
        if notifications == []:
            notifications_str = "No new notifications."
        else:
            notifications_str = "New notifications: \n"
            for note in notifications:
                notifications_str += f"- {note['message']} (Source: {note['source']})\n"
        if mode == '1':
            while True:
                user_input = input("User--> ")
                if user_input.lower() == 'exit':
                    break
                await start_llm_interaction(mode="user", user_input=user_input + "\n" + notifications_str, system_prompt="You are a helpful assistant. You can use tools to assist the user.")
        elif mode == '2':
            read_transcription("transcriptions/")
            for filename, content in transcription_files_content.items():
                print(f"\nProcessing transcription file: {filename}")
                await start_llm_interaction(mode="transcription", user_input=content)
        elif mode == '3':
            print("Late night execution mode selected.")
            
            while True:
                #First process night shift tasks from night_queue
                pending_tasks = night_shift.get_pending_tasks()
                if pending_tasks == []:
                    print("No pending tasks found.")
                else:
                    for task in pending_tasks:
                        task_description = task['description']
                        print(f"\nProcessing night task ID {task['id']}: {task_description}")
                        await start_llm_interaction(mode="night_task", user_input=task_description + "\n" + notifications_str, system_prompt=night_shift_prompt)
                        night_shift.mark_task_complete(task['id'], status="completed")
                    todoist_helper = TodoistHelper()
                    tasks = todoist_helper.get_tasks()
                    for task in tasks:
                        task_description = f"Task: {task['content']}, ID: {task['id']}"
                        print(f"\nProcessing Todoist task ID {task['id']}: {task['content']}")
                        await start_llm_interaction(mode="night_task", user_input=task_description + "\n" + notifications_str, system_prompt=night_shift_prompt)
                        todoist_helper.complete_task(task['id'])
                print("Sleeping for 30 seconds before checking for new night tasks...")
                await asyncio.sleep(30)  
                
    
        elif mode.lower() == 'exit':
            print("Exiting program.")
            break
        
async def main():
    try:
        await load_model("Qwen-4b-Thinking-2507-Q4_K_M")
        await start_mcp()
        await get_tools()
        await start_input_loop()
    except requests.exceptions.ConnectionError as e:
        print(f"LLAMA-SERVER is not running.")
        print(f"Error connecting to LLM API at {api_uri}: {e}")
    finally:
        await unload_model(currently_loaded_model)
        await cleanup_mcp()

if __name__ == "__main__":
    asyncio.run(main())