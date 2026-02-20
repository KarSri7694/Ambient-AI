"""Local OpenVINO GenAI wrapper to accept OpenAI-style chat messages and return completions.

This module provides two primary helpers:
- `chat_completion(...)` — synchronous, non-streaming completion returning an OpenAI-compatible dict
- `stream_chat_completion(...)` — generator yielding tokens (plain strings) for streaming clients
- `stream_chat_completion_with_tools(...)` — generator yielding OpenAI-compatible chunks with tool call support
"""
from pathlib import Path
from typing import List, Dict, Optional, Generator, Any, Tuple
import time
import json
import os
import re
import threading
import queue

try:
    import openvino_genai as ov_genai
    from openvino_genai import TextStreamer
except Exception as e:
    ov_genai = None
    TextStreamer = None
Pipe = None
Loaded_model = None
Model_path = None


def _find_model_path(model_name: str) -> Optional[str]:
    """Try to locate a model path given a name or path.
    Returns the first existing path or None.
    """
    p = Path(model_name)
    if p.exists():
        return str(p)

    return None


def load_model(model_name_or_path: str, device: str = "GPU") -> None:
    """Load an OpenVINO GenAI model into the global pipeline.

    model_name_or_path can be a filesystem path or a short model name that will be
    searched for under `model_cache/`.
    """
    global Pipe, Loaded_model, Model_path
    if ov_genai is None:
        raise RuntimeError("openvino_genai is not installed. Install it with `pip install openvino-genai`")

    if Loaded_model == model_name_or_path:
        return

    path = _find_model_path(model_name_or_path)
    if not path:
        # If nothing found, assume model_name_or_path is a path and let the pipeline constructor fail with a clear error
        path = model_name_or_path

    # Create pipeline
    Pipe = ov_genai.LLMPipeline(path, device)
    Loaded_model = model_name_or_path
    Model_path = path


def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """Convert OpenAI-style messages to a single prompt string ."""
    prompt_parts: List[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            prompt_parts.append(f"System: {content}")
        elif role == "user":
            prompt_parts.append(f"User: {content}")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: {content}")
    prompt_parts.append("Assistant: ")
    return "\n\n".join(prompt_parts)


def chat_completion(
    messages: List[Dict[str, str]],
    model: str = "openvino-model",
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = -1,
    stop: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Synchronous, non-streaming chat completion compatible with OpenAI response format.

    Returns a dictionary with keys: id, object, created, model, choices, usage
    """
    global Pipe, Loaded_model
    if Pipe is None or Loaded_model != model:
        load_model(model)

    prompt = _messages_to_prompt(messages)

    config = ov_genai.GenerationConfig()
    # config.max_new_tokens = max_tokens
    config.temperature = temperature
    config.top_p = top_p
    if stop:
        config.stop_strings = stop

    # Generate text (synchronous)
    result = Pipe.generate(prompt, config)
    if isinstance(result, (list, tuple)):
        # Some pipeline versions might return tokenized chunks; join if needed
        result_text = "".join(result)
    else:
        result_text = str(result)

    now = int(time.time())
    response = {
        "id": f"chatcmpl-{now}",
        "object": "chat.completion",
        "created": now,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(result_text.split()),
            "total_tokens": len(prompt.split()) + len(result_text.split()),
        },
    }
    return response


def stream_chat_completion(
    messages: List[Dict[str, str]],
    model: str = "openvino-model",
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = -1,
    stop: Optional[List[str]] = None,
) -> Generator[str, None, None]:
    """Yield tokens (strings) from the model as they are produced.

    Yields plain text tokens (or chunks). Consumers can reassemble them into full text.
    """
    global Pipe, Loaded_model
    if Pipe is None or Loaded_model != model:
        load_model(model)

    prompt = _messages_to_prompt(messages)

    config = ov_genai.GenerationConfig()
    # config.max_new_tokens = max_tokens
    config.temperature = temperature
    config.top_p = top_p
    if stop:
        config.stop_strings = stop

    # Use queue for real-time streaming
    token_queue = queue.Queue()
    generation_done = threading.Event()
    
    def streamer_callback(token_str: str) -> bool:
        """Callback for each generated token. Return False to continue, True to stop."""
        token_queue.put(token_str)
        return False  # Continue generation
    
    # Run generation in background thread
    def generate_thread():
        try:
            Pipe.generate(prompt, config, streamer_callback)
        finally:
            generation_done.set()
    
    thread = threading.Thread(target=generate_thread, daemon=True)
    thread.start()
    
    # Yield tokens as they arrive
    while not generation_done.is_set() or not token_queue.empty():
        try:
            token = token_queue.get(timeout=0.1)
            yield token
        except queue.Empty:
            continue
    
    thread.join()


def _format_messages_with_tools(messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None) -> str:
    """Convert messages and tools to a prompt that instructs the model about tool usage.
    
    This formats the prompt to encourage the model to output tool calls in a parseable format.
    """
    prompt_parts: List[str] = []
    
    # Add system message with tool instructions if tools are available
    if tools:
        tool_descriptions = []
        for tool in tools:
            func = tool.get("function", {})
            tool_descriptions.append(
                f"- {func.get('name')}: {func.get('description', 'No description')}\n"
                f"  Parameters: {json.dumps(func.get('parameters', {}), indent=2)}"
            )
        
        tools_text = "\n".join(tool_descriptions)
        system_instruction = f"""You have access to the following tools:

{tools_text}

When you need to use a tool, respond ONLY with a JSON object in this exact format:
{{"tool_calls": [{{"name": "tool_name", "arguments": {{"param1": "value1"}}}}]}}

If you don't need to use a tool, respond normally."""
        
        prompt_parts.append(f"System: {system_instruction}")
    
    # Add conversation messages
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            prompt_parts.append(f"System: {content}")
        elif role == "user":
            prompt_parts.append(f"User: {content}")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: {content}")
        elif role == "tool":
            tool_name = m.get("name", "unknown")
            prompt_parts.append(f"Tool Result ({tool_name}): {content}")
    
    prompt_parts.append("Assistant: ")
    return "\n\n".join(prompt_parts)


def _parse_tool_calls_from_text(text: str) -> Tuple[Optional[List[Dict]], str]:
    """Parse tool calls from model output text.
    
    Returns: (tool_calls_list, remaining_text)
    tool_calls_list is None if no tool calls found, otherwise a list of dicts with 'name' and 'arguments'
    """
    # Try to find JSON tool call pattern
    json_pattern = r'\{["\']tool_calls["\']\s*:\s*\[.*?\]\s*\}'
    match = re.search(json_pattern, text, re.DOTALL)
    
    if match:
        try:
            parsed = json.loads(match.group(0))
            tool_calls = parsed.get("tool_calls", [])
            if tool_calls:
                # Remove the tool call JSON from the text
                remaining_text = text[:match.start()] + text[match.end():]
                return tool_calls, remaining_text.strip()
        except json.JSONDecodeError:
            pass
    
    return None, text


def stream_chat_completion_with_tools(
    messages: List[Dict[str, str]],
    tools: Optional[List[Dict]] = None,
    model: str = "openvino-model",
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 512,
    stop: Optional[List[str]] = None,
) -> Generator[Dict[str, Any], None, None]:
    """Yield OpenAI-compatible streaming chunks with tool call support.
    
    Yields dictionaries with keys: id, object, created, model, choices
    Each choice contains 'delta' with either 'content' or 'tool_calls'
    """
    global Pipe, Loaded_model
    if Pipe is None or Loaded_model != model:
        load_model(model)

    prompt = _format_messages_with_tools(messages, tools)

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = max_tokens
    config.temperature = temperature
    config.top_p = top_p
    if stop:
        config.stop_strings = stop

    completion_id = f"chatcmpl-{int(time.time())}"
    created = int(time.time())
    
    # First, send role
    yield {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": ""},
            "finish_reason": None
        }]
    }
    
    # Accumulate full text to check for tool calls
    accumulated_text = ""
    token_queue = queue.Queue()
    generation_done = threading.Event()
    
    # Use streamer callback for token-by-token generation
    def streamer_callback(token_str: str) -> bool:
        """Callback for each generated token. Return False to continue, True to stop."""
        nonlocal accumulated_text
        accumulated_text += token_str
        token_queue.put(token_str)
        return False  # Continue generation
    
    # Run generation in background thread
    def generate_thread():
        try:
            Pipe.generate(prompt, config, streamer_callback)
        finally:
            generation_done.set()
    
    thread = threading.Thread(target=generate_thread, daemon=True)
    thread.start()
    
    # Stream tokens as they arrive
    while not generation_done.is_set() or not token_queue.empty():
        try:
            token = token_queue.get(timeout=0.1)
            
            # Yield content chunk
            yield {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": token},
                    "finish_reason": None
                }]
            }
        except queue.Empty:
            continue
    
    thread.join()
    
    # After streaming, check if there are tool calls in the accumulated text
    tool_calls, remaining_text = _parse_tool_calls_from_text(accumulated_text)
    
    if tool_calls:
        # Send tool calls
        for idx, tc in enumerate(tool_calls):
            yield {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "index": idx,
                            "id": f"call_{completion_id}_{idx}",
                            "type": "function",
                            "function": {
                                "name": tc.get("name", ""),
                                "arguments": json.dumps(tc.get("arguments", {}))
                            }
                        }]
                    },
                    "finish_reason": None
                }]
            }
    
    # Send final chunk
    yield {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop" if not tool_calls else "tool_calls"
        }]
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Local OpenVINO GenAI chat completion demo")
    parser.add_argument("--model", default="Qwen3-4B-Instruct-2507-Q4_K_M", help="Model name or path to load")
    parser.add_argument("--stream", action="store_true", help="Stream output tokens instead of returning full response")
    parser.add_argument("--device", default="GPU", help="Device for OpenVINO (CPU,GPU,etc)")
    args = parser.parse_args()

    # Attempt to load model (will raise helpful error if openvino_genai missing)
    try:
        load_model(args.model, device=args.device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise

    print("Model loaded. Type a message (empty line to exit):")
    while True:
        user = input("You: ")
        if not user.strip():
            break
        messages = [{"role": "user", "content": user}]
        if args.stream:
            print("Assistant (stream):", end=" ", flush=True)
            for tok in stream_chat_completion(messages, model=args.model):
                print(tok, end="", flush=True)
            print("\n---\n")
        else:
            resp = chat_completion(messages, model=args.model)
            print("Assistant:", resp["choices"][0]["message"]["content"])
            print("---\n")
