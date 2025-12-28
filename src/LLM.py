import openai
import llama_MCP_bridge
import asyncio

api_uri= "http://localhost:8080/v1"

client = openai.OpenAI(
    base_url=api_uri,
    api_key="testkey"
)

tools = asyncio.run(llama_MCP_bridge.get_all_mcp_tools("mcp.json"))
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
            for tool_call in delta.tool_calls:
                # print(f"\n\nTool Call Detected: {tool_call}\n")
                tool_name = tool_call.function.name
                tool_args = tool_call.function.arguments
                
                # Find the tool in the tools list
                # tool = next((t for t in tools if t['name'] == tool_name), None)
                # if tool:
                #     # Call the tool's function with the provided arguments
                #     tool_response = llama_MCP_bridge.call_mcp_tool(tool, tool_args)
                #     print(f"\nTool Response: {tool_response}\n")
                    
                #     # Append the tool response to the message history
                #     tool_response_message = {
                #         "role": "tool",
                #         "name": tool_name,
                #         "content": tool_response
                #     }
                #     message.append(tool_response_message)
                # else:
                #     print(f"Tool {tool_name} not found.")
                tool_response = asyncio.run(llama_MCP_bridge.execute_tool(tool_name, tool_args))
                print(f"tool name: {tool_name}, tool args: {tool_args}")
    assistant_message_obj = {"role": "assistant", "content": assistant_message}
    message.append(assistant_message_obj)
    print("\n\n\n")  
    

