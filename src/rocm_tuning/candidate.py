from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class TuningCandidate:
    candidate_id: str
    model_name: str
    model_path: str
    mmproj_path: str | None
    context_size: int
    gpu_layers: int
    flash_attention: bool
    cache_type_k: str
    cache_type_v: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def preset_name(self) -> str:
        return f"{self.model_name}_{self.candidate_id}"


def generate_candidates(
    *,
    models: Iterable[dict[str, object]],
    contexts: Iterable[int],
    gpu_layers: Iterable[int],
    flash_attention: Iterable[bool],
    kv_cache: Iterable[tuple[str, str]],
) -> list[TuningCandidate]:
    candidates: list[TuningCandidate] = []
    index = 1
    for model in models:
        model_name = str(model.get("name") or "").strip()
        model_path = str(model.get("model") or model.get("model_path") or "").strip()
        if not model_name or not model_path:
            continue
        mmproj_path = str(model.get("mmproj") or model.get("mmproj_path") or "").strip() or None
        for context_size in contexts:
            for layers in gpu_layers:
                for use_fa in flash_attention:
                    for cache_k, cache_v in kv_cache:
                        candidates.append(
                            TuningCandidate(
                                candidate_id=f"c{index:04d}",
                                model_name=model_name,
                                model_path=model_path,
                                mmproj_path=mmproj_path,
                                context_size=int(context_size),
                                gpu_layers=int(layers),
                                flash_attention=bool(use_fa),
                                cache_type_k=str(cache_k),
                                cache_type_v=str(cache_v),
                            )
                        )
                        index += 1
    return candidates


def render_models_preset(candidate: TuningCandidate) -> str:
    lines = [
        f"[{candidate.preset_name}]",
        f"model={candidate.model_path}",
        f"c={candidate.context_size}",
        f"ngl={candidate.gpu_layers}",
        f"fa={'on' if candidate.flash_attention else 'off'}",
        f"ctk={candidate.cache_type_k}",
        f"ctv={candidate.cache_type_v}",
    ]
    if candidate.mmproj_path:
        lines.insert(2, f"mmproj={candidate.mmproj_path}")
    return "\n".join(lines) + "\n"


def write_candidate_preset(candidate: TuningCandidate, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_models_preset(candidate), encoding="utf-8")
    return destination


def render_winning_preset(candidate: TuningCandidate) -> str:
    text = render_models_preset(candidate)
    return text.replace(f"[{candidate.preset_name}]", f"[{candidate.model_name}]")
