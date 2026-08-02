#!/usr/bin/env python3
"""Compare faster-whisper ASR performance on CPU vs ROCm/CTranslate2 GPU.

This script:
1. Converts an input audio file to 16 kHz mono WAV using ffmpeg.
2. Runs faster-whisper on CPU.
3. Runs faster-whisper on the CTranslate2 GPU backend.
   Note: CTranslate2/faster-whisper still uses device="cuda" for ROCm builds.
4. Prints timing, realtime factor, and transcript previews.

Example:
    python scripts/compare_faster_whisper_cpu_rocm.py /path/to/audio.mp3

With explicit model:
    python scripts/compare_faster_whisper_cpu_rocm.py audio.wav --model large-v3

If you installed a ROCm CTranslate2 wheel, the GPU run should use:
    device="cuda"

Requirements:
    pip install faster-whisper
    ffmpeg must be available on PATH.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class AsrResult:
    name: str
    device: str
    compute_type: str
    model: str
    language: str | None
    duration_seconds: float
    wall_seconds: float
    realtime_factor: float
    audio_seconds_per_wall_second: float
    segment_count: int
    text_chars: int
    transcript_preview: str


def require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit(
            "ffmpeg was not found on PATH. Install ffmpeg first, then rerun this script."
        )
    return ffmpeg


def convert_to_16khz_mono(input_path: Path, output_path: Path) -> None:
    ffmpeg = require_ffmpeg()
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        "-f",
        "wav",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
    return frames / float(rate)


def join_segments(segments: Iterable[object]) -> tuple[str, int]:
    parts: list[str] = []
    count = 0
    for segment in segments:
        count += 1
        parts.append(getattr(segment, "text", ""))
    return "".join(parts).strip(), count


def run_faster_whisper(
    *,
    name: str,
    audio_path: Path,
    audio_duration: float,
    model_name: str,
    device: str,
    compute_type: str,
    language: str | None,
    beam_size: int,
    vad_filter: bool,
    condition_on_previous_text: bool,
) -> AsrResult:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise SystemExit(
            "faster-whisper is not installed. Install it in your venv first: "
            "pip install faster-whisper"
        ) from exc

    started = time.perf_counter()
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=beam_size,
        vad_filter=vad_filter,
        condition_on_previous_text=condition_on_previous_text,
    )
    text, segment_count = join_segments(segments)
    ended = time.perf_counter()

    wall = ended - started
    detected_language = language or getattr(info, "language", None)
    return AsrResult(
        name=name,
        device=device,
        compute_type=compute_type,
        model=model_name,
        language=detected_language,
        duration_seconds=audio_duration,
        wall_seconds=wall,
        realtime_factor=wall / audio_duration if audio_duration > 0 else float("inf"),
        audio_seconds_per_wall_second=audio_duration / wall if wall > 0 else 0.0,
        segment_count=segment_count,
        text_chars=len(text),
        transcript_preview=text[:1000],
    )


def print_result(result: AsrResult) -> None:
    print()
    print(f"=== {result.name} ===")
    print(f"model: {result.model}")
    print(f"device: {result.device}")
    print(f"compute_type: {result.compute_type}")
    print(f"language: {result.language}")
    print(f"audio duration: {result.duration_seconds:.2f}s")
    print(f"wall time: {result.wall_seconds:.2f}s")
    print(f"realtime factor: {result.realtime_factor:.3f}x")
    print(f"audio/sec per wall-sec: {result.audio_seconds_per_wall_second:.3f}x")
    print(f"segments: {result.segment_count}")
    print(f"text chars: {result.text_chars}")
    print("preview:")
    print(result.transcript_preview)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare faster-whisper ASR on CPU and ROCm/CTranslate2 GPU."
    )
    parser.add_argument("audio", type=Path, help="Input audio/video file.")
    parser.add_argument(
        "--model",
        default="large-v3",
        help="faster-whisper model name or local converted model path. Default: large-v3",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional language code, e.g. en, hi, ja. Default: auto-detect.",
    )
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument(
        "--cpu-compute-type",
        default="int8",
        help="CPU compute type. Default: int8",
    )
    parser.add_argument(
        "--rocm-compute-type",
        default="float16",
        help="ROCm/GPU compute type. Default: float16",
    )
    parser.add_argument(
        "--skip-cpu",
        action="store_true",
        help="Only run ROCm/GPU backend.",
    )
    parser.add_argument(
        "--skip-rocm",
        action="store_true",
        help="Only run CPU backend.",
    )
    parser.add_argument(
        "--vad-filter",
        action="store_true",
        help="Enable faster-whisper VAD filter.",
    )
    parser.add_argument(
        "--condition-on-previous-text",
        action="store_true",
        help="Enable condition_on_previous_text. Default is disabled for long-form stability.",
    )
    parser.add_argument(
        "--keep-normalized-audio",
        type=Path,
        default=None,
        help="Optional path to save the generated 16 kHz mono WAV.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write machine-readable benchmark results.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.audio.expanduser().resolve()
    if not input_path.exists():
        print(f"Input file does not exist: {input_path}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="ambient-asr-bench-") as tmp:
        if args.keep_normalized_audio:
            normalized = args.keep_normalized_audio.expanduser().resolve()
            normalized.parent.mkdir(parents=True, exist_ok=True)
        else:
            normalized = Path(tmp) / "normalized-16khz-mono.wav"

        print(f"Normalizing audio to 16 kHz mono WAV: {normalized}")
        convert_to_16khz_mono(input_path, normalized)
        duration = wav_duration_seconds(normalized)
        print(f"Normalized audio duration: {duration:.2f}s")

        results: list[AsrResult] = []

        if not args.skip_cpu:
            results.append(
                run_faster_whisper(
                    name="CPU",
                    audio_path=normalized,
                    audio_duration=duration,
                    model_name=args.model,
                    device="cpu",
                    compute_type=args.cpu_compute_type,
                    language=args.language,
                    beam_size=args.beam_size,
                    vad_filter=args.vad_filter,
                    condition_on_previous_text=args.condition_on_previous_text,
                )
            )
            print_result(results[-1])

        if not args.skip_rocm:
            results.append(
                run_faster_whisper(
                    name="ROCm/CTranslate2 GPU",
                    audio_path=normalized,
                    audio_duration=duration,
                    model_name=args.model,
                    device="cuda",
                    compute_type=args.rocm_compute_type,
                    language=args.language,
                    beam_size=args.beam_size,
                    vad_filter=args.vad_filter,
                    condition_on_previous_text=args.condition_on_previous_text,
                )
            )
            print_result(results[-1])

        if len(results) == 2:
            cpu, rocm = results
            if rocm.wall_seconds > 0:
                speedup = cpu.wall_seconds / rocm.wall_seconds
                print()
                print("=== Comparison ===")
                print(f"ROCm speedup over CPU: {speedup:.2f}x")
                print(
                    "ROCm wall-time improvement: "
                    f"{cpu.wall_seconds - rocm.wall_seconds:.2f}s"
                )

        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            payload = [asdict(result) for result in results]
            args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"\nWrote JSON results: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
