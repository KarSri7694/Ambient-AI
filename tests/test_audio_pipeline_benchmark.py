
"""Benchmark the live audio pipeline from save-to-disk through transcript write.

The benchmark measures:
1. WAV file write time after recording is finished.
2. Full AudioAgent processing time until the final transcript is written.
3. Total elapsed time from save start to transcript save signal.
"""

from __future__ import annotations

import argparse
import queue
import sys
import time
from pathlib import Path

__test__ = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from audio_agent import AudioAgent
from realtime_audio_input import RealTimeAudioInput


class TranscriptTimingQueue(queue.Queue):
    def __init__(self) -> None:
        super().__init__()
        self.put_at: float | None = None
        self.last_item: str | None = None

    def put(self, item, block=True, timeout=None):
        self.put_at = time.perf_counter()
        self.last_item = item
        return super().put(item, block=block, timeout=timeout)


def close_stream_safely(stream) -> None:
    if stream is None:
        return

    try:
        if stream.is_active():
            stream.stop_stream()
    except OSError:
        pass

    try:
        stream.close()
    except OSError:
        pass


def benchmark_audio_pipeline(
    duration_seconds: float,
    output_dir: Path,
    sample_rate: int = 16000,
    channels: int = 1,
    chunk_size: int = 1024,
    input_device_index: int | None = None,
):
    audio_input = RealTimeAudioInput(
        sample_rate=sample_rate,
        channels=channels,
        chunk=chunk_size,
        uploads_dir=str(output_dir),
        input_device_index=input_device_index,
    )
    timing_queue = TranscriptTimingQueue()
    agent = AudioAgent(transcription_queue=timing_queue)

    recording_path = Path(audio_input._recording_filename())
    stream = None
    try:
        stream = audio_input.open_stream()
        stream.start_stream()

        capture_started_at = time.perf_counter()
        while time.perf_counter() - capture_started_at < duration_seconds and stream.is_active():
            time.sleep(0.1)

        benchmark_start = time.perf_counter()
        audio_input.save_recording(str(recording_path))
    finally:
        if stream is not None:
            if stream.is_active():
                stream.stop_stream()
            stream.close()

    save_finished_at = time.perf_counter()

    agent.run(str(recording_path))

    if timing_queue.put_at is None or timing_queue.last_item is None:
        raise RuntimeError("Audio pipeline finished without writing a final transcript.")

    transcript_path = Path(timing_queue.last_item)
    if not transcript_path.exists():
        raise RuntimeError(f"Expected transcript file was not created: {transcript_path}")

    total_elapsed = timing_queue.put_at - benchmark_start
    save_elapsed = save_finished_at - benchmark_start
    processing_elapsed = timing_queue.put_at - save_finished_at

    return {
        "recording_path": recording_path,
        "transcript_path": transcript_path,
        "save_elapsed": save_elapsed,
        "processing_elapsed": processing_elapsed,
        "total_elapsed": total_elapsed,
    }


def format_seconds(value: float) -> str:
    return f"{value:.4f} s"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark the live audio pipeline from WAV save to transcript save."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="How long to record from the microphone before benchmarking the pipeline.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "uploads",
        help="Directory where the benchmark recording will be written.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Sample rate to use when capturing the benchmark recording.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        help="Number of recording channels.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Chunk size used while reading microphone input.",
    )
    parser.add_argument(
        "--input-device-index",
        type=int,
        default=None,
        help="Optional PyAudio input device index.",
    )
    args = parser.parse_args()

    results = benchmark_audio_pipeline(
        duration_seconds=args.duration,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        channels=args.channels,
        chunk_size=args.chunk_size,
        input_device_index=args.input_device_index,
    )

    print("Benchmark complete")
    print(f"Recording saved to: {results['recording_path']}")
    print(f"Transcript saved to: {results['transcript_path']}")
    print(f"WAV save time: {format_seconds(results['save_elapsed'])}")
    print(f"Pipeline processing time: {format_seconds(results['processing_elapsed'])}")
    print(f"Total timed interval: {format_seconds(results['total_elapsed'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())