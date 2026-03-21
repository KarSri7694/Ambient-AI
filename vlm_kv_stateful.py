from __future__ import annotations

import argparse
import copy
import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import openvino as ov
import openvino.opset13 as _ov_ops
from PIL import Image
from transformers import AutoProcessor


# ---------------------------------------------------------------------------
# LM model graph patch — last-token logits only
# ---------------------------------------------------------------------------

def _patch_lm_last_token_only(model: ov.Model) -> ov.Model:
    """
    Splice a Gather node into the LM output graph so the model returns
    ONLY the last token's logits: shape [B, 1, vocab] instead of [B, S, vocab].

    Why this matters on Intel GPU
    --------------------------------
    Intel GPU has a hard per-single-allocation cap (~3.14 GB, OpenCL
    CL_DEVICE_MAX_MEM_ALLOC_SIZE).  For a 13 k-token prefill with vocab=151936
    the raw logits tensor is  1 × 13568 × 151936 × 2 bytes = 3.84 GB  which
    exceeds the cap and crashes with
      "[GPU] Exceeded max size of memory object allocation".

    openvino_genai.VLMPipeline applies exactly this optimisation internally;
    here we replicate it so our custom pipeline behaves the same way.

    The decode loop already uses only logits[0, -1, :] so behaviour is
    identical — only the GPU allocation shrinks from O(S × vocab) to
    O(vocab) per inference step.
    """
    result_node = model.output(0).get_node()   # ov.op.Result
    parent_out  = result_node.input_value(0)   # [B, S, vocab]

    # shape_of(logits)[1] - 1  → last sequence index,  shape [1]
    shape    = _ov_ops.shape_of(parent_out, output_type=ov.Type.i64)
    seq_dim  = _ov_ops.gather(
        shape,
        _ov_ops.constant(np.array(1, dtype=np.int64)),
        _ov_ops.constant(np.array(0, dtype=np.int64)),
    )
    last_idx = _ov_ops.subtract(
        seq_dim,
        _ov_ops.constant(np.array([1], dtype=np.int64)),
    )  # shape [1] → Gather will keep the size-1 axis

    # Gather on axis=1: [B, S, vocab] → [B, 1, vocab]
    last_logits = _ov_ops.gather(
        parent_out,
        last_idx,
        _ov_ops.constant(np.array(1, dtype=np.int32)),
    )

    result_node.input(0).replace_source_output(last_logits.output(0))
    return model


# ---------------------------------------------------------------------------
# Generic KV snapshot dataclass
# ---------------------------------------------------------------------------

@dataclass
class KVSnapshot:
    """
    Deep copies of all KV VariableState tensors + runtime counters.

    This is model-agnostic: it captures whatever VariableState tensors
    exist in the compiled LM — 56 for Qwen2.5-7B-VL, different counts
    for other models.

    Restoring a snapshot requires O(kv_size) memcpy and zero forward passes.
    """
    tensors:      dict           # {state_name: np.ndarray}
    past_seq_len: int            # LM tokens processed so far
    mrope_pos:    np.ndarray     # shape [3] — M-RoPE axis counters

class VLMStatefulBase(ABC):
    """
    Model-agnostic base for zero-recompute VLM pipelines.

    Concrete subclasses must implement:
        _load_submodels(core, model_path, device)
        _embed_tokens(token_ids)  -> np.ndarray [1, S, hidden]
        _encode_image(image)      -> (np.ndarray [N, hidden], tuple | None)
        _build_position_ids(input_ids, grid_thw, past_seq_len)  -> [3, 1, S]
        _get_image_token_id()     -> int

    Everything else (KV snapshot/restore, greedy decode, generate loop)
    lives here and never needs to be touched for a new model family.
    """

    def __init__(self, model_path: str, device: str = "CPU"):
        self._past_seq_len: int       = 0
        self._mrope_pos: np.ndarray   = np.zeros(3, dtype=np.int64)
        self._lm_req:    ov.InferRequest  # set by _load_submodels

        model_path = str(Path(model_path).resolve())
        print(f"[{self.__class__.__name__}] Loading from : {model_path}")
        print(f"[{self.__class__.__name__}] Device       : {device}")

        core = ov.Core()
        self._load_submodels(core, model_path, device)

        # EOS token IDs — set by subclass or here if processor is available
        eos_raw = self.tokenizer.eos_token_id
        self.eos_ids = set(eos_raw if isinstance(eos_raw, list) else [eos_raw])

        print(f"[{self.__class__.__name__}] Ready.")

    # ───────────────────────────────────────────────────────────────────────
    # Abstract interface — implement per model family
    # ───────────────────────────────────────────────────────────────────────

    @abstractmethod
    def _load_submodels(self, core: ov.Core, model_path: str, device: str) -> None:
        """
        Compile all sub-models and store their InferRequests as instance attrs.
        Must also set self.tokenizer and self.processor (or just self.tokenizer).
        The LM InferRequest MUST be stored as self._lm_req.
        """
        ...

    @abstractmethod
    def _embed_tokens(self, token_ids: np.ndarray) -> np.ndarray:
        """token_ids [1,S] → inputs_embeds [1, S, hidden]"""
        ...

    @abstractmethod
    def _encode_image(
        self, image: Image.Image
    ) -> Tuple[Optional[np.ndarray], Optional[tuple]]:
        """
        Encode image to visual token embeddings.
        Returns (vis_embeds [N, hidden], grid_thw (T, H, W)) or (None, None).
        """
        ...

    @abstractmethod
    def _build_position_ids(
        self,
        input_ids:   np.ndarray,   # [1, S]
        grid_thw:    Optional[tuple],
        past_seq_len: int,
    ) -> np.ndarray:               # [3, 1, S]  or  [1, 1, S] for 1D-RoPE models
        """Build position_ids for the LM prefill step."""
        ...

    @abstractmethod
    def _get_image_token_id(self) -> int:
        """Return the token ID used as an image-pad placeholder in input_ids."""
        ...

    def _before_prefill(self, input_ids: np.ndarray) -> None:
        """
        Optional hook called just before the prefill forward pass.

        Receives the FULL input_ids array (shape [1, S]) including all
        expanded image-pad tokens.  Subclasses can override this to compute
        tensors that depend on the complete sequence (e.g. visual_pos_masks
        for Qwen3-VL which needs to know which positions are image tokens).
        """
        pass


    def snapshot_kv(self) -> KVSnapshot:
        """
        Deep-copy all KV VariableState tensors.
        Cost: O(kv_size) memcpy.  Zero forward passes.
        """
        tensors = {
            s.name: s.state.data.copy()
            for s in self._lm_req.query_state()
        }
        return KVSnapshot(
            tensors      = tensors,
            past_seq_len = self._past_seq_len,
            mrope_pos    = self._mrope_pos.copy(),
        )

    def restore_kv(self, snapshot: KVSnapshot) -> None:
        """
        Write snapshot tensors back into the LM InferRequest KV state.
        Cost: O(kv_size) tensor write.  Zero forward passes.

        After this call the model's effective context is exactly the set of
        tokens present when snapshot_kv() was called — any intermediate turns
        (including their visual tokens) are completely absent.
        """
        for s in self._lm_req.query_state():
            s.state = ov.Tensor(snapshot.tensors[s.name])
        self._past_seq_len = snapshot.past_seq_len
        self._mrope_pos    = snapshot.mrope_pos.copy()

    def reset_kv(self) -> None:
        """Reset KV state to empty — start of a new conversation."""
        self._lm_req.reset_state()
        self._past_seq_len = 0
        self._mrope_pos    = np.zeros(3, dtype=np.int64)

    def _lm_forward(
        self,
        inputs_embeds:  np.ndarray,   # [1, S, hidden]
        attention_mask: np.ndarray,   # [1, total_S]
        position_ids:   np.ndarray,   # [3, 1, S] or [1, 1, S]
    ) -> np.ndarray:
        """Single LM forward pass. Returns logits [1, S, vocab]."""
        batch    = inputs_embeds.shape[0]
        beam_idx = np.zeros(batch, dtype=np.int32)

        self._lm_req.set_tensor("inputs_embeds",  ov.Tensor(inputs_embeds))
        self._lm_req.set_tensor("attention_mask", ov.Tensor(attention_mask))
        self._lm_req.set_tensor("position_ids",   ov.Tensor(position_ids))
        self._lm_req.set_tensor("beam_idx",        ov.Tensor(beam_idx))
        self._lm_req.infer()
        return self._lm_req.get_output_tensor().data  # [1, S, vocab]


    @staticmethod
    def _sample_token(
        logits_1d:   np.ndarray,   # shape [vocab]
        temperature: float,
        top_p:       float,
        top_k:       int = 0,      # 0 = disabled
    ) -> int:
        """
        Pick the next token from logits.

        temperature == 0.0  →  greedy (argmax), fully deterministic.
        temperature  > 0.0  →  softmax-temperature sampling.
                                top_k > 0  keeps only the k highest-logit tokens.
                                top_p < 1.0 applies nucleus filtering after top_k.
                                Both filters can be combined.
        """
        if temperature <= 0.0:
            return int(np.argmax(logits_1d))

        # Temperature scaling + numerically stable softmax
        scaled  = logits_1d.astype(np.float64) / temperature
        scaled -= scaled.max()
        probs   = np.exp(scaled)
        probs  /= probs.sum()

        # Top-k filtering
        if top_k > 0 and top_k < len(probs):
            top_k_idx = np.argpartition(probs, -top_k)[-top_k:]
            mask = np.zeros_like(probs)
            mask[top_k_idx] = 1.0
            probs = probs * mask
            probs /= probs.sum()

        # Nucleus (top-p) filtering
        if top_p < 1.0:
            sorted_idx  = np.argsort(probs)[::-1]
            cumulative   = np.cumsum(probs[sorted_idx])
            cutoff       = int(np.searchsorted(cumulative, top_p, side="right")) + 1
            cutoff       = max(cutoff, 1)
            mask         = np.zeros_like(probs)
            mask[sorted_idx[:cutoff]] = 1.0
            probs        = probs * mask
            probs       /= probs.sum()

        return int(np.random.choice(len(probs), p=probs))

    def _greedy_decode(
        self,
        max_new_tokens: int,
        first_logits:   np.ndarray,   # [1, S, vocab] from prefill
        temperature:    float = 0.0,  # 0.0 = greedy; > 0 enables sampling
        top_p:          float = 1.0,  # nucleus filter; only applied when temperature > 0
        top_k:          int   = 0,    # top-k filter; 0 = disabled
        streamer=None,                # Optional[Callable[[str], None]] — called per token
    ) -> str:
        """
        Autoregressive decode starting from a completed prefill.

        temperature == 0.0 (default)  →  greedy argmax, fully deterministic.
        temperature  > 0.0            →  softmax-temperature + nucleus (top-p) sampling.

        Advances self._past_seq_len in sync with the KV state.

        streamer: optional callable invoked with each decoded token string as it
                  is produced, enabling real-time streaming to a queue or UI.
        """
        generated_ids = []
        logits        = first_logits
        prev_text     = ""   # tracks decoded text so far for incremental delta

        for _ in range(max_new_tokens):
            next_id = self._sample_token(logits[0, -1, :], temperature, top_p, top_k)
            generated_ids.append(next_id)
            if next_id in self.eos_ids:
                break

            if streamer is not None:
                # Decode all tokens so far and emit only the new suffix.
                # This correctly handles multi-byte / split-character tokens.
                current_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                delta = current_text[len(prev_text):]
                if delta:
                    streamer(delta)
                prev_text = current_text

            token_ids   = np.array([[next_id]], dtype=np.int64)
            next_embeds = self._embed_tokens(token_ids)           # [1, 1, hidden]
            new_total   = self._past_seq_len + 1
            attn_mask   = np.ones((1, new_total), dtype=np.int64)
            new_pos     = np.full((3, 1, 1), self._past_seq_len, dtype=np.int64)

            logits             = self._lm_forward(next_embeds, attn_mask, new_pos)
            self._past_seq_len += 1

        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def generate(
        self,
        prompt_text:    str,
        image:          Optional[Image.Image] = None,
        max_new_tokens: int = 200,
        messages:       Optional[list] = None,
        temperature:    float = 0.0,   # 0.0 = greedy; > 0 enables sampling
        top_p:          float = 1.0,   # nucleus filter; only applied when temperature > 0
        tools:          Optional[list] = None,   # OpenAI-style tool definitions
        top_k:          int   = 0,     # top-k sampling filter; 0 = disabled
        streamer=None,                 # Optional[Callable[[str], None]] — real-time token streaming
    ) -> str:
        """
        Process one conversation turn.

        Appends to the live KV state (i.e. extends past_seq_len).
        Use snapshot_kv() before turns you may want to roll back.
        Use restore_kv(snap) to roll back — zero recompute.

        Args:
            prompt_text:    Raw user message text.  Used when *messages* is None.
            image:          PIL image to encode.  Optional.
            max_new_tokens: Maximum tokens to generate.
            messages:       OpenAI-style List[Dict] — when provided it is passed
                            directly to apply_chat_template so system prompts and
                            full multi-turn history are included.  If an image is
                            also given and the last user message has a plain-string
                            content, an {"type":"image"} entry is injected
                            automatically before calling the processor.
            temperature:    Sampling temperature.  0.0 (default) = greedy argmax.
                            Values > 0 enable softmax-temperature sampling.
            top_p:          Nucleus sampling threshold (0, 1].  Only applied when
                            temperature > 0.  Default 1.0 = no nucleus filtering.
            tools:          Optional list of OpenAI-style tool dicts passed directly
                            to apply_chat_template, enabling the model’s native
                            function-calling format via the Jinja chat template.
        """
        # Build the messages list for apply_chat_template ─────────────────
        if messages is not None:
            # Shallow-copy so we don't mutate the caller's list
            chat_messages = [dict(m) for m in messages]
            if image is not None:
                # Inject {"type":"image"} into the last user message if not present
                for i in reversed(range(len(chat_messages))):
                    if chat_messages[i].get("role") == "user":
                        content = chat_messages[i]["content"]
                        if isinstance(content, str):
                            chat_messages[i]["content"] = [
                                {"type": "image"},
                                {"type": "text", "text": content},
                            ]
                        elif isinstance(content, list) and not any(
                            isinstance(c, dict) and c.get("type") == "image"
                            for c in content
                        ):
                            chat_messages[i]["content"] = [{"type": "image"}] + list(content)
                        break
        else:
            message_content: list = []
            if image is not None:
                message_content.append({"type": "image"})
            message_content.append({"type": "text", "text": prompt_text})
            chat_messages = [{"role": "user", "content": message_content}]

        # Inject tool definitions into the system message ────────────────
        # The Qwen2.5-VL / Qwen3-VL chat templates have no native tool-call
        # support, so we manually build a system-level instruction that matches
        # the JSON format expected by _parse_tool_calls_from_text in
        # openvino_backend.py: {"tool_calls": [{"name": "...", "arguments": {...}}]}
        if tools:
            tool_lines = []
            for t in tools:
                func = t.get("function", t)   # handle both wrapped and bare dicts
                tool_lines.append(
                    f"- {func.get('name', '?')}: "
                    f"{func.get('description', 'No description')}\n"
                    f"  Parameters: {json.dumps(func.get('parameters', {}), indent=2)}"
                )
            tool_instruction = (
                "You have access to the following tools:\n\n"
                + "\n".join(tool_lines)
                + "\n\nWhen you need to use a tool, respond ONLY with a JSON "
                'object in this exact format:\n'
                '{"tool_calls": [{"name": "tool_name", "arguments": '
                '{"param1": "value1"}}]}\n\n'
                "If you don't need to use a tool, respond normally."
            )
            # Find existing system message and prepend; otherwise insert one
            sys_idx = next(
                (i for i, m in enumerate(chat_messages) if m.get("role") == "system"),
                None,
            )
            if sys_idx is not None:
                existing = chat_messages[sys_idx]["content"]
                if isinstance(existing, str):
                    chat_messages[sys_idx]["content"] = (
                        tool_instruction + "\n\n" + existing
                    )
                elif isinstance(existing, list):
                    chat_messages[sys_idx]["content"] = [
                        {"type": "text", "text": tool_instruction + "\n\n"}
                    ] + existing
            else:
                chat_messages.insert(0, {"role": "system", "content": tool_instruction})

        formatted = self.processor.apply_chat_template(
            chat_messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        if image is not None:
            proc_out  = self.processor(text=[formatted], images=[image], return_tensors="pt")
            input_ids = proc_out["input_ids"].numpy().astype(np.int64)   # [1, S_expanded]
        else:
            enc       = self.tokenizer(formatted, return_tensors="np")
            input_ids = enc["input_ids"].astype(np.int64)
        S         = input_ids.shape[1]

        # Text embeddings baseline
        combined = self._embed_tokens(input_ids)         # [1, S, hidden]

        # Splice visual embeddings if an image was given
        grid_thw = None
        if image is not None:
            vis_embeds, grid_thw = self._encode_image(image)  # [N_vis, hidden]
            if vis_embeds is not None:
                img_token_id = self._get_image_token_id()
                flat         = input_ids[0]
                img_mask     = flat == img_token_id
                n_img_tok    = int(np.sum(img_mask))
                assert n_img_tok == vis_embeds.shape[0], (
                    f"Visual token count mismatch: "
                    f"seq has {n_img_tok} image-pad tokens but "
                    f"vision encoder produced {vis_embeds.shape[0]} embeddings"
                )
                combined = combined.copy()
                combined[0, np.where(img_mask)[0], :] = vis_embeds

        # Attention mask covers past + current tokens
        total_len  = self._past_seq_len + S
        attn_mask  = np.ones((1, total_len), dtype=np.int64)

        # Position IDs (delegated to subclass)
        position_ids = self._build_position_ids(input_ids, grid_thw, self._past_seq_len)

        # Prefill
        self._before_prefill(input_ids)
        logits             = self._lm_forward(combined, attn_mask, position_ids)
        self._past_seq_len += S

        # Autoregressive decode
        return self._greedy_decode(max_new_tokens, logits, temperature=temperature, top_p=top_p, top_k=top_k, streamer=streamer)


# QWEN3-VL SPECIFIC HELPERS

def _qwen3vl_rotary_pos_emb(
    grid_t: int, grid_h: int, grid_w: int,
    num_heads: int = 16, hidden_size: int = 1024,
    theta: float = 10000.0,
    merge_size: int = 2,
) -> np.ndarray:
    """
    2D rotary position embeddings for Qwen3-VL vision patches.

    head_dim = hidden_size // num_heads = 1024 // 16 = 64
    rot_dim  = head_dim // 2            = 32   (matches merger input [N, 32])
    axis_dim = rot_dim  // 2            = 16   (16 freqs per spatial axis)

    Token ordering: **merge-group scan** (block_row × block_col × intra_row × intra_col)
    which matches `Qwen3VLVisionModel.rot_pos_emb()` in the transformers source.
    This is different from simple raster scan.

    Returns float32 [T*H*W, 32].
    """
    head_dim = hidden_size // num_heads
    rot_dim  = head_dim // 2
    axis_dim = rot_dim  // 2

    inv_freq = (1.0 / (theta ** (
        np.arange(0, axis_dim, dtype=np.float64) / axis_dim
    ))).astype(np.float32)

    merged_h = grid_h // merge_size
    merged_w = grid_w // merge_size

    # Merge-group scan: (merged_h, merged_w, merge_size, merge_size)
    # row[br, bc, ir, ic] = br * merge_size + ir
    # col[br, bc, ir, ic] = bc * merge_size + ic
    br = np.arange(merged_h, dtype=np.float32)
    bc = np.arange(merged_w, dtype=np.float32)
    ir = np.arange(merge_size, dtype=np.float32)
    ic = np.arange(merge_size, dtype=np.float32)

    row_idx = (br[:, None, None, None] * merge_size + ir[None, None, :, None])
    row_idx = np.broadcast_to(row_idx, (merged_h, merged_w, merge_size, merge_size)).reshape(-1)
    col_idx = (bc[None, :, None, None] * merge_size + ic[None, None, None, :])
    col_idx = np.broadcast_to(col_idx, (merged_h, merged_w, merge_size, merge_size)).reshape(-1)

    # Repeat for temporal frames
    row_idx = np.tile(row_idx, grid_t)
    col_idx = np.tile(col_idx, grid_t)

    emb_y = np.outer(row_idx, inv_freq)
    emb_x = np.outer(col_idx, inv_freq)
    return np.concatenate([emb_y, emb_x], axis=-1).astype(np.float32)  # [N, 32]


def _qwen3vl_mrope_position_ids(
    input_ids:       np.ndarray,   # [1, S]
    image_token_id:  int,
    grid_thw:        Optional[tuple],
    past_seq_len:    int,
    merge_size:      int = 2,
) -> np.ndarray:                   # [3, 1, S]
    """
    M-RoPE position IDs for Qwen3-VL LM.

    mrope_section = [24, 20, 20]  (vs [16, 24, 24] in Qwen2.5-VL)
    mrope_interleaved = True      (NEW — axes are interleaved within head_dim)

    The interleaved flag affects how the RoPE frequencies are applied at
    inference time inside the LM kernel; position_ids themselves are built
    the same way as Qwen2.5-VL (temporal / height / width axes).
    """
    S   = input_ids.shape[1]
    pos = np.zeros((3, 1, S), dtype=np.int64)

    if grid_thw is None or image_token_id not in input_ids[0]:
        for axis in range(3):
            pos[axis, 0, :] = np.arange(past_seq_len, past_seq_len + S)
        return pos

    flat      = input_ids[0]
    img_start = int(np.argmax(flat == image_token_id))
    T, H, W   = grid_thw
    H_m, W_m  = H // merge_size, W // merge_size
    N_m       = T * H_m * W_m   # number of visual tokens in LM after merge

    # ── Text before image ────────────────────────────────────────────────
    cursor = past_seq_len
    for axis in range(3):
        pos[axis, 0, :img_start] = np.arange(cursor, cursor + img_start)
    cursor += img_start

    # ── Visual tokens ────────────────────────────────────────────────────
    t_pos = np.repeat(np.arange(T), H_m * W_m)
    h_pos = np.tile(np.repeat(np.arange(H_m), W_m), T)
    w_pos = np.tile(np.arange(W_m), T * H_m)

    vis_end = img_start + N_m
    pos[0, 0, img_start:vis_end] = cursor + t_pos
    pos[1, 0, img_start:vis_end] = cursor + h_pos
    pos[2, 0, img_start:vis_end] = cursor + w_pos
    cursor += max(T, H_m, W_m)   # advance by the largest spatial extent

    # ── Text after image ─────────────────────────────────────────────────
    for axis in range(3):
        pos[axis, 0, vis_end:] = np.arange(cursor, cursor + (S - vis_end))

    return pos


# QWEN3-VL 4B STATEFUL PIPELINE

class Qwen3VLStatefulPipeline(VLMStatefulBase):
    """
    Zero-recompute VLM pipeline for Qwen3-VL (4B / 8B INT8 OpenVINO exports).

    Architecture: Qwen3VLForConditionalGeneration
    Model dir expected layout (same as Qwen2.5-VL but with an extra pos model):
        openvino_text_embeddings_model.xml/bin
        openvino_vision_embeddings_model.xml/bin
        openvino_vision_embeddings_pos_model.xml/bin   ← NEW in Qwen3-VL
        openvino_vision_embeddings_merger_model.xml/bin
        openvino_language_model.xml/bin

    Config diff vs Qwen2.5-7B-VL
    ─────────────────────────────
    | Property               | Qwen2.5-7B-VL  | Qwen3-VL-4B       |
    |------------------------|----------------|-------------------|
    | LM hidden_size         | 3584           | 2560              |
    | LM num_hidden_layers   | 28             | 36                |
    | LM num_kv_heads        | 8              | 8                 |
    | LM head_dim            | 128            | 128               |
    | LM rope_theta          | 1 000 000      | 5 000 000         |
    | LM mrope_section       | [16, 24, 24]   | [24, 20, 20]      |
    | LM mrope_interleaved   | False          | True              |
    | LM KV state tensors    | 56             | 72                |
    | Vis hidden_size        | 1280           | 1024              |
    | Vis num_heads          | 16             | 16                |
    | Vis depth              | 32             | 24                |
    | Vis patch_size         | 14             | 16                |
    | Vis rot_dim (merger)   | 40             | 32                |
    | Vis merger no-window   | No             | Yes               |
    | Vis pos_model          | —              | [4,N]→[4,N,1024]  |
    | Vis deepstack_indexes  | —              | [5, 11, 17]       |
    | LM extra inputs        | —              | visual_pos_masks  |
    |                        |                | deepstack_embeds  |

    ─── IMPLEMENTATION STATUS ────────────────────────────────────────────────
    All three sub-model inputs are now correctly computed:

    [1]  pos_model [4, H*W] int64 → [4, H*W, 1024]
         Implements fast_pos_embed_interpolate() logic:
         bilinear interpolation of a 48×48 fixed position grid onto the
         dynamic H×W patch grid, returning 4 corner lookup indices.
         Bilinear weights applied externally; result added to patch_embed
         output (enc_out) before passing to the merger.

    [2]  visual_pos_masks [1, S] int8
         Built in _before_prefill() from the FULL input_ids:
           (input_ids[0] == IMAGE_TOKEN_ID).astype(int8).reshape(1, -1)
         True at each of the N_m <|image_pad|> positions.

    [3]  deepstack_visual_embeds [3, N_m, 2560]
         Directly from the merger's second output (deepstack_feature_lists).
         Three intermediate vision features injected into LM layers 0, 1, 2.
    ─────────────────────────────────────────────────────────────────────────
    """

    # ── Architecture constants ────────────────────────────────────────────
    _LM_HIDDEN          = 2560
    _LM_NUM_LAYERS      = 36
    _LM_NUM_KV_HEADS    = 8
    _LM_HEAD_DIM        = 128
    _LM_ROPE_THETA      = 5_000_000
    _LM_MROPE_SECTION   = [24, 20, 20]   # sum = 64 = head_dim // 2
    _LM_MROPE_INTERLEAVED = True         # NEW vs Qwen2.5-VL

    _VIS_HIDDEN              = 1024
    _VIS_NUM_HEADS           = 16
    _VIS_DEPTH               = 24
    _VIS_PATCH_SIZE          = 16
    _VIS_MERGE_SIZE          = 2
    _VIS_TEMPORAL_PATCH_SIZE = 2
    _VIS_ROT_DIM             = 32            # = head_dim//2 = (1024//16)//2
    _VIS_DEEPSTACK_IDXS      = [5, 11, 17]  # intermediate layer indices
    _VIS_NUM_POS_EMBEDDINGS  = 2304         # pos_embed.weight shape[0]; num_grid_per_side=48

    _IMAGE_TOKEN_ID     = 151655
    _VIDEO_TOKEN_ID     = 151656
    _VISION_START_ID    = 151652
    _VISION_END_ID      = 151653
    _EOS_TOKEN_ID       = 151645
    _BOS_TOKEN_ID       = 151643

    # Expected number of KV VariableState tensors in the LM:
    #   36 layers × 2 (K + V) = 72
    _EXPECTED_KV_TENSORS = 72

    def _load_submodels(self, core: ov.Core, model_path: str, device: str) -> None:
        print(f"[{self.__class__.__name__}] Loading processor …")
        from transformers import AutoProcessor, AutoTokenizer
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer

        print(f"[{self.__class__.__name__}] Compiling sub-models …")
        # On GPU: quantise the KV-cache states to int8 to stay under
        # Intel GPU's per-allocation cap (~3.1 GB).
        lm_compile_cfg = {"KV_CACHE_PRECISION": ov.Type.u8} if device.upper() == "GPU" else {}

        def _compile(name: str, cfg: dict | None = None) -> ov.InferRequest:
            return core.compile_model(
                core.read_model(f"{model_path}/{name}.xml"), device, cfg or {}
            ).create_infer_request()

        self._te_req  = _compile("openvino_text_embeddings_model")
        self._ve_req  = _compile("openvino_vision_embeddings_model")
        self._vp_req  = _compile("openvino_vision_embeddings_pos_model")  # NEW
        self._vm_req  = _compile("openvino_vision_embeddings_merger_model")
        _lm_model     = core.read_model(f"{model_path}/openvino_language_model.xml")
        _lm_model     = _patch_lm_last_token_only(_lm_model)
        self._lm_req  = core.compile_model(_lm_model, device, lm_compile_cfg or {}).create_infer_request()

        self._image_token_id = self._IMAGE_TOKEN_ID
        self._merge_size     = self._VIS_MERGE_SIZE

        # Storage for Qwen3-VL specific LM extras (set during _encode_image)
        self._last_N_m:              int                       = 0
        self._last_deepstack_embeds: Optional[np.ndarray]     = None  # [3, N_m, 2560]
        self._last_visual_pos_mask:  Optional[np.ndarray]     = None  # [1, S]  int8

    def _embed_tokens(self, token_ids: np.ndarray) -> np.ndarray:
        """[1, S] int64 → [1, S, 2560] float32"""
        self._te_req.set_input_tensor(ov.Tensor(token_ids))
        self._te_req.infer()
        return self._te_req.get_output_tensor().data.copy()

    def _encode_image(
        self, image: Image.Image
    ) -> Tuple[Optional[np.ndarray], Optional[tuple]]:
        """
        Encode a PIL image into visual token embeddings for the Qwen3-VL LM.

        Pipeline:
          1. Processor  → pixel_values [N, 1536], image_grid_thw [T,H,W]
             (patch_dim = 3 * temporal_patch_size * patch_size^2 = 3*2*256 = 1536)
          2. VisionEncoder (openvino_vision_embeddings_model)
             pixel_values [N, 1536] → enc_out [N, 1024]  (patch_embed, no pos)
          3. PosModel (openvino_vision_embeddings_pos_model)
             Implements fast_pos_embed_interpolate logic:
               a. Build bilinear corner indices idx_tensor [4, H*W] and weights [4, H*W]
                  by interpolating a 48×48 fixed position grid (num_grid_per_side=48)
               b. idx_tensor [4, H*W] int64 → pos_lookup [4, H*W, 1024]
                  (embedding table lookups at 4 bilinear corners)
               c. Weighted sum → pos_embeds [H*W, 1024]
               d. Repeat for T frames → [T*H*W, 1024]
               e. Permute from raster to merge-group order
               f. enc_out += pos_embeds
          4. Merger (openvino_vision_embeddings_merger_model)
             hidden_states [N, 1024],
             attention_mask [1, N, N] (full self-attention),
             rotary_pos_emb [N, 32]  (merge-group order, from _qwen3vl_rotary_pos_emb)
             → last_hidden_state [N_merged, 2560]
             → deepstack_feature_lists [3, N_merged, 2560]
          5. visual_pos_masks [1, S] int8 is computed in _before_prefill() once the
             full input_ids (with expanded image-pad tokens) are available.

        Returns:
            vis_embeds  [N_merged, 2560]
            grid_thw    (T, H, W)
        """
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "x"}]}]
        text     = self.processor.apply_chat_template(messages, add_generation_prompt=False)
        inputs   = self.processor(text=[text], images=[image], return_tensors="pt")

        pixel_values = inputs["pixel_values"].numpy().astype(np.float32)  # [N, 1536]
        grid_raw     = inputs["image_grid_thw"].numpy().reshape(-1, 3)
        T, H, W      = int(grid_raw[0, 0]), int(grid_raw[0, 1]), int(grid_raw[0, 2])
        N            = T * H * W   # total patches (before spatial merge)
        N_m          = T * (H // self._merge_size) * (W // self._merge_size)  # merged

        # ── Step 2: Vision patch embed ─────────────────────────────────────
        # pixel_values [N, 1536] → enc_out [N, 1024]
        pv = pixel_values.reshape(N, -1).astype(np.float32)
        self._ve_req.set_input_tensor(ov.Tensor(pv))
        self._ve_req.infer()
        enc_out = self._ve_req.get_output_tensor().data.copy()  # [N, 1024]

        # ── Step 3: Positional embeddings via pos_model ────────────────────
        # Replicates Qwen3VLVisionModel.fast_pos_embed_interpolate() in NumPy.
        # num_grid_per_side = int(sqrt(num_position_embeddings)) = int(sqrt(2304)) = 48
        num_gps = int(self._VIS_NUM_POS_EMBEDDINGS ** 0.5)  # 48

        # Bilinear interpolation of a 48×48 grid onto an H×W patch grid
        h_f = np.linspace(0.0, num_gps - 1, H)   # H float positions in [0, 47]
        w_f = np.linspace(0.0, num_gps - 1, W)   # W float positions in [0, 47]

        h_lo = h_f.astype(np.int64)
        w_lo = w_f.astype(np.int64)
        h_hi = np.clip(h_lo + 1, 0, num_gps - 1)
        w_hi = np.clip(w_lo + 1, 0, num_gps - 1)

        dh = (h_f - h_lo).astype(np.float32)   # [H]   fractional offset
        dw = (w_f - w_lo).astype(np.float32)   # [W]

        base_h_lo = h_lo * num_gps   # row-start index in pos_embed weight table
        base_h_hi = h_hi * num_gps

        # Four bilinear corner indices into the 48×48 table [H*W each]:
        idx0 = (base_h_lo[:, None] + w_lo[None, :]).flatten().astype(np.int64)   # TL
        idx1 = (base_h_lo[:, None] + w_hi[None, :]).flatten().astype(np.int64)   # TR
        idx2 = (base_h_hi[:, None] + w_lo[None, :]).flatten().astype(np.int64)   # BL
        idx3 = (base_h_hi[:, None] + w_hi[None, :]).flatten().astype(np.int64)   # BR
        idx_tensor = np.stack([idx0, idx1, idx2, idx3], axis=0)  # [4, H*W]

        # Corresponding bilinear weights [4, H*W]:
        w0 = ((1.0 - dh)[:, None] * (1.0 - dw)[None, :]).flatten().astype(np.float32)
        w1 = ((1.0 - dh)[:, None] * dw[None, :]).flatten().astype(np.float32)
        w2 = (dh[:, None]         * (1.0 - dw)[None, :]).flatten().astype(np.float32)
        w3 = (dh[:, None]         * dw[None, :]).flatten().astype(np.float32)
        weight_tensor = np.stack([w0, w1, w2, w3], axis=0)  # [4, H*W]

        # Embedding lookup [4, H*W] int64 → [4, H*W, 1024]
        self._vp_req.set_input_tensor(ov.Tensor(idx_tensor))
        self._vp_req.infer()
        pos_lookup = self._vp_req.get_output_tensor().data.copy()  # [4, H*W, 1024]

        # Weighted sum over 4 corners → [H*W, 1024]
        pos_embeds = (pos_lookup * weight_tensor[:, :, None]).sum(axis=0)

        # Repeat for T temporal frames → [T*H*W, 1024]
        pos_embeds = np.tile(pos_embeds, (T, 1))

        # Permute from raster order (T, H, W) → merge-group order (T, H//2, W//2, 2, 2)
        # This matches what fast_pos_embed_interpolate's .permute(0,1,3,2,4,5) does.
        m = self._merge_size
        pos_embeds = pos_embeds.reshape(T, H // m, m, W // m, m, -1)  # (T,H//2,2,W//2,2,C)
        pos_embeds = pos_embeds.transpose(0, 1, 3, 2, 4, 5)            # (T,H//2,W//2,2,2,C)
        pos_embeds = pos_embeds.reshape(-1, pos_embeds.shape[-1])       # [N, 1024]

        # Add positional embeddings to patch embeddings
        enc_out = enc_out + pos_embeds   # [N, 1024]

        # ── Step 4: Merger  ────────────────────────────────────────────────
        # Tokens are in merge-group order; full self-attention (no windows)
        rope      = _qwen3vl_rotary_pos_emb(T, H, W, merge_size=m)  # [N, 32]
        full_mask = np.zeros((1, N, N), dtype=np.float32)

        self._vm_req.set_tensor("hidden_states",  ov.Tensor(enc_out))
        self._vm_req.set_tensor("attention_mask", ov.Tensor(full_mask))
        self._vm_req.set_tensor("rotary_pos_emb", ov.Tensor(rope))
        self._vm_req.infer()

        vis_embeds = self._vm_req.get_output_tensor(0).data.copy()  # [N_m, 2560]
        deepstack  = self._vm_req.get_output_tensor(1).data.copy()  # [3, N_m, 2560]

        # ── Step 5: visual_pos_masks ───────────────────────────────────────
        # Computed in _before_prefill() once the FULL input_ids (with N_m
        # image-pad tokens) are available.  Shape will be [1, S] int8.
        self._last_N_m              = N_m
        self._last_deepstack_embeds = deepstack   # [3, N_m, 2560]
        self._last_visual_pos_mask  = None         # filled by _before_prefill()

        return vis_embeds, (T, H, W)

    def _lm_forward(
        self,
        inputs_embeds:  np.ndarray,   # [1, S, 2560]
        attention_mask: np.ndarray,   # [1, total_S]
        position_ids:   np.ndarray,   # [3, 1, S]
    ) -> np.ndarray:
        """
        Override the base LM forward to inject Qwen3-VL-specific extras:
          - visual_pos_masks      [1, S] int8
            Boolean mask (as int8) over the FULL sequence; True at every
            <|image_pad|> position.  Used by the LM to locate the N_m rows
            in hidden_states where deepstack features are added.
          - deepstack_visual_embeds [3, N_m, 2560] float32
            Three intermediate vision features (from merger deepstack outputs
            at encoder layers 5, 11, 17) injected into LM layers 0, 1, 2.

        Prefill: consume stored tensors from _encode_image / _before_prefill.
        Decode : pass empty/zero versions so no deepstack injection occurs.
        """
        batch    = inputs_embeds.shape[0]
        beam_idx = np.zeros(batch, dtype=np.int32)

        if self._last_deepstack_embeds is not None:
            # ── Prefill: real deepstack and visual mask ─────────────────────
            deepstack_emb = self._last_deepstack_embeds   # [3, N_m, 2560]
            vis_pos_mask  = self._last_visual_pos_mask    # [1, S]  int8
            # Consume once (prefill only)
            self._last_deepstack_embeds = None
            self._last_visual_pos_mask  = None
        else:
            # ── Decode step: single new token, no visual positions ──────────
            # visual_pos_masks [1, 1] all-zero → no image pad positions
            # deepstack_visual_embeds [3, 0, 2560] → 3 features with 0 visual
            # tokens; the LM adds nothing to any hidden position
            vis_pos_mask  = np.zeros((1, 1), dtype=np.int8)
            deepstack_emb = np.zeros((3, 0, self._LM_HIDDEN), dtype=np.float32)

        self._lm_req.set_tensor("inputs_embeds",           ov.Tensor(inputs_embeds))
        self._lm_req.set_tensor("attention_mask",          ov.Tensor(attention_mask))
        self._lm_req.set_tensor("position_ids",            ov.Tensor(position_ids))
        self._lm_req.set_tensor("beam_idx",                ov.Tensor(beam_idx))
        self._lm_req.set_tensor("visual_pos_masks",        ov.Tensor(vis_pos_mask))
        self._lm_req.set_tensor("deepstack_visual_embeds", ov.Tensor(deepstack_emb))
        self._lm_req.infer()
        return self._lm_req.get_output_tensor().data  # [1, S, vocab]

    def _before_prefill(self, input_ids: np.ndarray) -> None:
        """
        Compute visual_pos_masks from the full input_ids.

        Now that we have the complete sequence (with N_m image-pad tokens
        expanded by the processor), we can build a [1, S] int8 mask where
        True = position holds an <|image_pad|> token (id 151655).
        This is passed to the LM as `visual_pos_masks`.
        """
        if self._last_deepstack_embeds is not None:
            mask = (input_ids[0] == self._IMAGE_TOKEN_ID).astype(np.int8)
            self._last_visual_pos_mask = mask.reshape(1, -1)  # [1, S]

    def _build_position_ids(
        self,
        input_ids:    np.ndarray,
        grid_thw:     Optional[tuple],
        past_seq_len: int,
    ) -> np.ndarray:
        return _qwen3vl_mrope_position_ids(
            input_ids, self._image_token_id, grid_thw, past_seq_len, self._merge_size
        )

    def _get_image_token_id(self) -> int:
        return self._IMAGE_TOKEN_ID


# ===========================================================================
# HOW TO ADD A NEW MODEL FAMILY
# ===========================================================================
#
# class LLaVAStatefulPipeline(VLMStatefulBase):
#     """LLaVA-style model: CLIP vision encoder, standard 1D RoPE, no merger."""
#
#     def _load_submodels(self, core, model_path, device):
#         # load openvino_vision_encoder.xml, openvino_language_model.xml
#         # set self._lm_req, self._ve_req, self.tokenizer, self.processor
#         ...
#
#     def _embed_tokens(self, token_ids):
#         # same pattern — text embeddings sub-model
#         ...
#
#     def _encode_image(self, image):
#         # CLIP encoder only — no merger, no window attention
#         ...
#
#     def _build_position_ids(self, input_ids, grid_thw, past_seq_len):
#         # Standard 1D sequential position IDs — shape [1, 1, S]
#         S = input_ids.shape[1]
#         pos = np.arange(past_seq_len, past_seq_len + S, dtype=np.int64)
#         return pos.reshape(1, 1, S)
#
#     def _get_image_token_id(self):
#         return 32000   # LLaVA image token id


# ===========================================================================
# DEMO
# ===========================================================================

def run_demo(model_path: str, image_path: str, device: str = "CPU"):
    print(f"\n{'='*60}")
    print(f"  Model  : {model_path}")
    print(f"  Device : {device}")
    print(f"  Image  : {image_path}")
    print(f"{'='*60}\n")

    pipe = Qwen3VLStatefulPipeline(model_path, device)
    pipe.reset_kv()

    # ------------------------------------------------------------------
    # Turn 1 — text only
    # ------------------------------------------------------------------
    print("── Turn 1 (Text Only) ───────────────────────────────────────")
    q1 = "Hello! Are you ready to help me analyse an image?"
    r1 = pipe.generate(q1, max_new_tokens=100)
    print(f"  User     : {q1}")
    print(f"  Assistant: {r1}")
    print(f"  [past_seq_len after T1: {pipe._past_seq_len}]\n")

    # Snapshot AFTER turn 1 — this is the rollback target
    snap1 = pipe.snapshot_kv()
    print(f"  ✓ KV snapshot taken  "
          f"({len(snap1.tensors)} state tensors, past_seq_len={snap1.past_seq_len})\n")

    # ------------------------------------------------------------------
    # Turn 2 — image + text
    # ------------------------------------------------------------------
    print("── Turn 2 (Image + Text) ────────────────────────────────────")
    try:
        img = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"  [ERROR] Image not found: {image_path}")
        return

    q2 = "What is the primary object in this image? Be concise."
    r2 = pipe.generate(q2, image=img, max_new_tokens=100)
    print(f"  User     : {q2}")
    print(f"  Assistant: {r2}")
    print(f"  [past_seq_len after T2: {pipe._past_seq_len}]")
    print(f"  [tokens added in T2: {pipe._past_seq_len - snap1.past_seq_len}]\n")

    # ------------------------------------------------------------------
    # Rollback — zero forward passes
    # ------------------------------------------------------------------
    print("── Rollback (zero recompute) ─────────────────────────────────")
    pipe.restore_kv(snap1)
    print(f"  KV restored to past_seq_len={pipe._past_seq_len}")
    print(f"  → image tokens are GONE from the KV cache\n")

    # ------------------------------------------------------------------
    # Turn 3 — text only, image NOT in context
    # ------------------------------------------------------------------
    print("── Turn 3 (Text Only — image NOT in context) ────────────────")
    q3 = (
        f'A previous analysis found the primary object was: "{r2}". '
        f"What colour is that typically? One sentence."
    )
    r3 = pipe.generate(q3, max_new_tokens=100)
    print(f"  User     : {q3}")
    print(f"  Assistant: {r3}")
    print(f"  [past_seq_len after T3: {pipe._past_seq_len}]\n")

    print("── Done ──────────────────────────────────────────────────────")
    print("  T1 KV reused for T3 with zero recompute.")
    print("  Visual tokens from T2 absent from T3 context.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="True zero-recompute VLM pipeline (OpenVINO KV state snapshots)"
    )
    p.add_argument("--model",  default="./Qwen2.5-7B-VL-int4-ov")
    p.add_argument("--image",  default="sample.jpg")
    p.add_argument("--device", default="CPU", choices=["CPU", "GPU", "NPU"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_demo(args.model, args.image, args.device)
