"""Local OpenVINO GenAI wrapper — supports both text-only (LLMPipeline) and
vision-language (VLMPipeline) models with OpenAI-style chat messages.

This module provides two primary entry-points:
- `chat_completion(...)` — synchronous, non-streaming completion
- `stream_chat_completion_with_tools(...)` — streaming generator yielding
  OpenAI-compatible chunk dicts

Both accept an optional ``image`` argument (file path string) which activates
the VLM pipeline for multimodal inference.
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
except Exception:
    ov_genai = None
    TextStreamer = None

try:
    import numpy as np
    from PIL import Image as PILImage
except Exception:
    np = None
    PILImage = None

# ── Global pipeline state ─────────────────────────────────────
Pipe = None          # Active pipeline (LLMPipeline or VLMPipeline)
IsVLM = False        # True when a VLMPipeline is loaded
Loaded_model = None  # Currently loaded model path / name
Model_path = None    # Resolved filesystem path


# ── Model loading ─────────────────────────────────────────────

def _find_model_path(model_name: str) -> Optional[str]:
    """Return the first existing path for *model_name*, or None."""
    p = Path(model_name)
    if p.exists():
        return str(p)
    return None


def load_model(model_name_or_path: str, device: str = "GPU", vlm: bool = False) -> None:
    """Load an LLMPipeline (text) or VLMPipeline (vision+text) into the global state.

    Args:
        model_name_or_path: Filesystem path or model name.
        device: OpenVINO device string — ``"CPU"``, ``"GPU"``, or ``"NPU"``.
        vlm: Force VLMPipeline regardless of auto-detection. If *False*,
             the module auto-detects by looking for a ``config.json`` that
             declares a vision encoder.
    """
    global Pipe, IsVLM, Loaded_model, Model_path

    if ov_genai is None:
        raise RuntimeError(
            "openvino_genai is not installed. Run: pip install openvino-genai"
        )

    if Loaded_model == model_name_or_path:
        return

    path = _find_model_path(model_name_or_path) or model_name_or_path

    # Auto-detect VLM: check if config.json mentions a vision encoder
    use_vlm = vlm or _is_vlm_model(path)

    if use_vlm:
        Pipe = ov_genai.VLMPipeline(path, device)
        IsVLM = True
    else:
        Pipe = ov_genai.LLMPipeline(path, device)
        IsVLM = False

    Loaded_model = model_name_or_path
    Model_path = path


def _is_vlm_model(path: str) -> bool:
    """Heuristic: return True if the model directory looks like a VLM."""
    config_path = Path(path) / "config.json"
    if not config_path.exists():
        return False
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        # Common VLM indicators in HuggingFace / OpenVINO model configs
        return any(
            key in cfg
            for key in ("vision_config", "visual_encoder", "image_token_index",
                        "vision_encoder", "mm_vision_tower")
        )
    except Exception:
        return False


# ── Image helpers ─────────────────────────────────────────────

def _load_image_as_tensor(image_path: str):
    """Load an image file and return an ``ov.Tensor`` (H, W, 3) uint8 in RGB.

    VLMPipeline.generate() requires an ``openvino.Tensor``, not a raw ndarray.
    """
    if np is None or PILImage is None:
        raise RuntimeError(
            "Pillow and numpy are required for image input. "
            "Install with: pip install Pillow numpy"
        )
    try:
        import openvino as ov
    except ImportError:
        raise RuntimeError(
            "openvino core package is required for image tensors. "
            "Install with: pip install openvino"
        )
    img = PILImage.open(image_path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)   # shape: (H, W, 3)
    return ov.Tensor(arr)


# ── Prompt formatting ─────────────────────────────────────────

def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """Convert OpenAI-style messages to a plain-text prompt string."""
    parts: List[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
    parts.append("Assistant: ")
    return "\n\n".join(parts)


def _format_messages_with_tools(
    messages: List[Dict[str, str]],
    tools: Optional[List[Dict]] = None,
) -> str:
    """Convert messages + optional tools to a prompt string."""
    parts: List[str] = []

    if tools:
        tool_descriptions = []
        for tool in tools:
            func = tool.get("function", {})
            tool_descriptions.append(
                f"- {func.get('name')}: {func.get('description', 'No description')}\n"
                f"  Parameters: {json.dumps(func.get('parameters', {}), indent=2)}"
            )
        tools_text = "\n".join(tool_descriptions)
        system_instruction = (
            f"You have access to the following tools:\n\n{tools_text}\n\n"
            "When you need to use a tool, respond ONLY with a JSON object in this "
            'exact format:\n{"tool_calls": [{"name": "tool_name", "arguments": '
            '{"param1": "value1"}}]}\n\nIf you don\'t need to use a tool, respond normally.'
        )
        parts.append(f"System: {system_instruction}")

    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        elif role == "tool":
            tool_name = m.get("name", "unknown")
            parts.append(f"Tool Result ({tool_name}): {content}")

    parts.append("Assistant: ")
    return "\n\n".join(parts)


# ── Tool-call parsing ─────────────────────────────────────────

def _parse_tool_calls_from_text(text: str) -> Tuple[Optional[List[Dict]], str]:
    """Parse JSON tool calls embedded in model output.

    Returns: (tool_calls_list | None, remaining_text)
    """
    json_pattern = r'\{["\']tool_calls["\']\s*:\s*\[.*?\]\s*\}'
    match = re.search(json_pattern, text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            tool_calls = parsed.get("tool_calls", [])
            if tool_calls:
                remaining = text[: match.start()] + text[match.end() :]
                return tool_calls, remaining.strip()
        except json.JSONDecodeError:
            pass
    return None, text


# ── Generation config helper ──────────────────────────────────

def _make_config(temperature: float, top_p: float, max_tokens: int,
                 stop: Optional[List[str]]):
    """Build an ``ov_genai.GenerationConfig`` from common parameters."""
    cfg = ov_genai.GenerationConfig()
    cfg.temperature = temperature
    cfg.top_p = top_p
    if max_tokens > 0:
        cfg.max_new_tokens = max_tokens
    if stop:
        cfg.stop_strings = stop
    return cfg


# ── Public API ────────────────────────────────────────────────

def chat_completion(
    messages: List[Dict[str, str]],
    model: str = "openvino-model",
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = -1,
    stop: Optional[List[str]] = None,
    image: Optional[str] = None,
) -> Dict[str, Any]:
    """Synchronous chat completion — returns an OpenAI-compatible response dict.

    Args:
        messages: OpenAI-style message list.
        model: Model name (used to load if not already loaded).
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability.
        max_tokens: Maximum new tokens (-1 = unlimited).
        stop: Stop strings.
        image: Optional path to an image file (activates VLM inference).
    """
    global Pipe, Loaded_model
    if Pipe is None or Loaded_model != model:
        load_model(model, vlm=bool(image))

    prompt = _messages_to_prompt(messages)
    cfg = _make_config(temperature, top_p, max_tokens, stop)

    if image:
        if not IsVLM:
            raise RuntimeError(
                "An image was provided but the loaded pipeline is text-only. "
                "Reload with vlm=True or use a VLM model."
            )
        img_array = _load_image_as_array(image)
        result = Pipe.generate(prompt, image=img_array, generation_config=cfg)
    else:
        result = Pipe.generate(prompt, cfg)

    result_text = "".join(result) if isinstance(result, (list, tuple)) else str(result)

    now = int(time.time())
    return {
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


def stream_chat_completion(
    messages: List[Dict[str, str]],
    model: str = "openvino-model",
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = -1,
    stop: Optional[List[str]] = None,
    image: Optional[str] = None,
) -> Generator[str, None, None]:
    """Yield plain-text tokens from the model as they are produced.

    Args:
        image: Optional path to an image file (activates VLM inference).
    """
    global Pipe, Loaded_model
    if Pipe is None or Loaded_model != model:
        load_model(model, vlm=bool(image))

    prompt = _messages_to_prompt(messages)
    cfg = _make_config(temperature, top_p, max_tokens, stop)

    token_queue: queue.Queue = queue.Queue()
    generation_done = threading.Event()

    def streamer_callback(token_str: str) -> bool:
        token_queue.put(token_str)
        return False  # continue generation

    def generate_thread():
        try:
            if image:
                img_array = _load_image_as_tensor(image)
                Pipe.generate(prompt, image=img_array,
                              generation_config=cfg, streamer=streamer_callback)
            else:
                Pipe.generate(prompt, cfg, streamer_callback)
        finally:
            generation_done.set()

    threading.Thread(target=generate_thread, daemon=True).start()

    while not generation_done.is_set() or not token_queue.empty():
        try:
            yield token_queue.get(timeout=0.1)
        except queue.Empty:
            continue


def stream_chat_completion_with_tools(
    messages: List[Dict[str, str]],
    tools: Optional[List[Dict]] = None,
    model: str = "openvino-model",
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 512,
    stop: Optional[List[str]] = None,
    image: Optional[str] = None,
) -> Generator[Dict[str, Any], None, None]:
    """Yield OpenAI-compatible streaming chunk dicts with optional tool-call support.

    Args:
        image: Optional path to an image file (activates VLM inference).
    """
    global Pipe, Loaded_model
    if Pipe is None or Loaded_model != model:
        load_model(model, vlm=bool(image))

    prompt = _format_messages_with_tools(messages, tools)
    cfg = _make_config(temperature, top_p, max_tokens, stop)

    completion_id = f"chatcmpl-{int(time.time())}"
    created = int(time.time())

    # ── Role chunk ────────────────────────────────────────────
    yield {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0,
                     "delta": {"role": "assistant", "content": ""},
                     "finish_reason": None}],
    }

    accumulated_text = ""
    token_queue: queue.Queue = queue.Queue()
    generation_done = threading.Event()

    def streamer_callback(token_str: str) -> bool:
        nonlocal accumulated_text
        accumulated_text += token_str
        token_queue.put(token_str)
        return False

    def generate_thread():
        try:
            if image:
                img_array = _load_image_as_tensor(image)
                Pipe.generate(prompt, image=img_array,
                              generation_config=cfg, streamer=streamer_callback)
            else:
                Pipe.generate(prompt, cfg, streamer_callback)
        finally:
            generation_done.set()

    threading.Thread(target=generate_thread, daemon=True).start()

    # ── Stream tokens ─────────────────────────────────────────
    while not generation_done.is_set() or not token_queue.empty():
        try:
            token = token_queue.get(timeout=0.1)
            yield {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0,
                             "delta": {"content": token},
                             "finish_reason": None}],
            }
        except queue.Empty:
            continue

    # ── Post-stream: check for tool calls ─────────────────────
    tool_calls, _ = _parse_tool_calls_from_text(accumulated_text)

    if tool_calls:
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
                                "arguments": json.dumps(tc.get("arguments", {})),
                            },
                        }]
                    },
                    "finish_reason": None,
                }],
            }

    # ── Final chunk ───────────────────────────────────────────
    yield {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop" if not tool_calls else "tool_calls",
        }],
    }


# ── CLI demo ──────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OpenVINO GenAI chat demo")
    parser.add_argument("--model", default="Qwen3-4B-int4-ov")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--vlm", action="store_true",
                        help="Force VLMPipeline for vision-language models")
    parser.add_argument("--image", default=None,
                        help="Path to an image to send with each message")
    args = parser.parse_args()

    try:
        load_model(args.model, device=args.device, vlm=args.vlm)
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise

    pipeline_type = "VLM" if IsVLM else "LLM"
    print(f"Model loaded ({pipeline_type}). Type a message (empty line to exit):")

    while True:
        user = input("You: ")
        if not user.strip():
            break
        messages = [{"role": "user", "content": user}]
        if args.stream:
            print("Assistant (stream):", end=" ", flush=True)
            for tok in stream_chat_completion(messages, model=args.model,
                                              image=args.image):
                print(tok, end="", flush=True)
            print("\n---\n")
        else:
            resp = chat_completion(messages, model=args.model, image=args.image)
            print("Assistant:", resp["choices"][0]["message"]["content"])
            print("---\n")
