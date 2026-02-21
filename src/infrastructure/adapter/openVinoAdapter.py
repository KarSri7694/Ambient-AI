"""
OpenVINO GenAI adapter — implements both LLMProvider and ModelManager ports.

Supports both text-only (LLMPipeline) and vision-language (VLMPipeline) models.
Image paths are forwarded to openvino_backend which handles PIL loading and
numpy conversion before the OpenVINO pipeline.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Iterator

from application.ports.LLMProvider import LLMProvider
from application.ports.modelManager import ModelManager

import openvino_backend as _ov


# ── Shim objects ──────────────────────────────────────────────
# LLMInteractionService._consume_stream expects OpenAI-SDK-style objects
# with attribute access (chunk.choices[0].delta.content, etc.).
# The openvino_backend yields plain dicts, so we wrap them here.


@dataclass
class _FunctionShim:
    name: str = ""
    arguments: str = ""


@dataclass
class _ToolCallDeltaShim:
    index: int = 0
    id: Optional[str] = None
    type: str = "function"
    function: _FunctionShim = field(default_factory=_FunctionShim)


@dataclass
class _DeltaShim:
    role: Optional[str] = None
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[_ToolCallDeltaShim]] = None


@dataclass
class _ChoiceShim:
    index: int = 0
    delta: _DeltaShim = field(default_factory=_DeltaShim)
    finish_reason: Optional[str] = None


@dataclass
class _ChunkShim:
    id: str = ""
    object: str = "chat.completion.chunk"
    created: int = 0
    model: str = ""
    choices: List[_ChoiceShim] = field(default_factory=list)


def _dict_to_chunk(d: Dict[str, Any]) -> _ChunkShim:
    """Convert a raw dict from openvino_backend into an attribute-access shim."""
    choices: List[_ChoiceShim] = []
    for c in d.get("choices", []):
        delta_raw = c.get("delta", {})

        tc_shims: Optional[List[_ToolCallDeltaShim]] = None
        if delta_raw.get("tool_calls"):
            tc_shims = []
            for tc in delta_raw["tool_calls"]:
                fn_raw = tc.get("function", {})
                tc_shims.append(
                    _ToolCallDeltaShim(
                        index=tc.get("index", 0),
                        id=tc.get("id"),
                        type=tc.get("type", "function"),
                        function=_FunctionShim(
                            name=fn_raw.get("name", ""),
                            arguments=fn_raw.get("arguments", ""),
                        ),
                    )
                )

        choices.append(
            _ChoiceShim(
                index=c.get("index", 0),
                delta=_DeltaShim(
                    role=delta_raw.get("role"),
                    content=delta_raw.get("content"),
                    reasoning_content=delta_raw.get("reasoning_content"),
                    tool_calls=tc_shims,
                ),
                finish_reason=c.get("finish_reason"),
            )
        )

    return _ChunkShim(
        id=d.get("id", ""),
        object=d.get("object", "chat.completion.chunk"),
        created=d.get("created", 0),
        model=d.get("model", ""),
        choices=choices,
    )


# ── Adapter ───────────────────────────────────────────────────


class OpenVinoAdapter(LLMProvider, ModelManager):
    """Adapter for OpenVINO GenAI — implements both LLMProvider and ModelManager.

    Supports text-only (LLMPipeline) and vision-language (VLMPipeline) models.
    Pass a non-empty ``image`` path to ``generate_response`` or
    ``chat_completion_stream`` to activate multimodal inference.
    """

    def __init__(self, model_path: str, device: str = "GPU", vlm: bool = False):
        """
        Args:
            model_path: Path to the OpenVINO model directory.
            device: Target device — ``"CPU"``, ``"GPU"``, or ``"NPU"``.
            vlm: Force VLMPipeline. If *False*, the backend auto-detects from
                 the model's ``config.json``.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self._model_path = model_path
        self._device = device
        self._vlm = vlm
        self.currently_loaded_model: Optional[str] = None

    # ── ModelManager ──────────────────────────────────────────

    async def load_model(self, model_name: str) -> None:
        if self.currently_loaded_model == model_name:
            self.logger.info(f"Model {model_name} is already loaded.")
            return

        pipeline_type = "VLM" if self._vlm else "LLM (auto-detect)"
        self.logger.info(
            f"Loading OpenVINO model: {model_name} | device={self._device} "
            f"| pipeline={pipeline_type}"
        )
        _ov.load_model(model_name, device=self._device, vlm=self._vlm)
        self.currently_loaded_model = model_name
        self.logger.info(
            f"Loaded as {'VLMPipeline' if _ov.IsVLM else 'LLMPipeline'}"
        )

    async def unload_model(self) -> None:
        if self.currently_loaded_model is None:
            return
        self.logger.info(f"Unloading OpenVINO model: {self.currently_loaded_model}")
        _ov.Pipe = None
        _ov.IsVLM = False
        _ov.Loaded_model = None
        _ov.Model_path = None
        self.currently_loaded_model = None

    def get_current_model(self) -> Optional[str]:
        return self.currently_loaded_model

    # ── LLMProvider ───────────────────────────────────────────

    def generate_response(self, prompt: str, image: str = "") -> str:
        """Non-streaming generation. Pass ``image`` (file path) for VLM inference."""
        messages = [{"role": "user", "content": prompt}]
        model = self.currently_loaded_model or self._model_path
        resp = _ov.chat_completion(
            messages,
            model=model,
            image=image or None,
        )
        return resp["choices"][0]["message"]["content"]

    def chat_completion_stream(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        image: str = "",
    ) -> Iterator[_ChunkShim]:
        """Streaming chat completion.

        Args:
            image: Optional file path to an image for VLM inference.
        """
        raw_stream = _ov.stream_chat_completion_with_tools(
            messages=messages,
            tools=tools,
            model=model,
            temperature=0.7,
            top_p=0.95,
            max_tokens=32000,
            image=image or None,
        )
        for chunk_dict in raw_stream:
            yield _dict_to_chunk(chunk_dict)
