import pyaudio
from datetime import datetime
import wave
import uuid
import time
import os
import silero_vad
from silero_vad import VADIterator
import numpy as np
import torch
from collections import deque
from typing import Any
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class RealTimeAudioInput:
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk: int = 512,
        pre_roll_ms: int = 400,
        uploads_dir: str = "uploads",
        threshold: float = 0.7,
        min_silence_duration_ms: int = 2000,
        input_device_index: int | None = None,
    ):
        """Initialize audio, VAD, and buffering state for real-time capture.

        Args:
            sample_rate: Target recording sample rate in Hz.
            channels: Number of audio input channels.
            chunk: Frames per callback chunk from the audio stream.
            pre_roll_ms: Amount of audio history (in milliseconds) to retain before speech.
            uploads_dir: Directory where recorded WAV files are saved.
            threshold: VAD confidence threshold used to detect speech.
            min_silence_duration_ms: Silence duration needed to mark speech end.
            input_device_index: Optional explicit PyAudio input device index.
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk = chunk
        self.format = pyaudio.paInt16
        self.uploads_dir = uploads_dir
        self.threshold = threshold
        self.min_silence_duration_ms = min_silence_duration_ms

        self.num_previous_samples = int(self.sample_rate * pre_roll_ms / 1000)
        pre_roll_chunks = (self.num_previous_samples + self.chunk - 1) // self.chunk

        self.frames = []
        self.confidence_list = []
        self.pre_roll_buffer = deque(maxlen=pre_roll_chunks)
        self.voice_started = False

        self.model = silero_vad.load_silero_vad()
        self.vad = self.initialize_vad()

        self.agent = pyaudio.PyAudio()
        self.device_info = self.agent.get_default_input_device_info()
        self.default_sample_rate = int(self.device_info.get("defaultSampleRate"))
        self.input_device_index = self._resolve_input_device_index(input_device_index)

        os.makedirs(self.uploads_dir, exist_ok=True)

    def initialize_vad(self):
        """Create and return a VAD iterator configured with current thresholds."""
        return VADIterator(
            model=self.model,
            threshold=self.threshold,
            min_silence_duration_ms=self.min_silence_duration_ms,
        )

    def _resolve_input_device_index(self, input_device_index: int | None):
        """Resolve the effective input device index, defaulting to system default.

        Args:
            input_device_index: Optional user-provided input device index.
        """
        if input_device_index is not None:
            return input_device_index
        default_idx = self.device_info.get("index")
        return int(default_idx) if default_idx is not None else None

    def int2float(self, sound: np.ndarray):
        """Convert int16 PCM numpy audio to normalized float32 samples.

        Args:
            sound: Numpy int16 audio samples.
        """
        max_abs = float(np.abs(sound).max())
        sound = sound.astype("float32")
        if max_abs > 0:
            sound *= 1 / 32768
        return sound.squeeze()

    def _recording_filename(self):
        """Build a unique output WAV filename in the uploads directory."""
        return os.path.join(
            self.uploads_dir,
            f"audio_record_{datetime.now().strftime('%d%m%Y_%H%M%S')}_{uuid.uuid4()}.wav",
        )

    def _reset_segment_state(self):
        """Clear current segment buffers and reset VAD state for next capture."""
        self.frames = []
        self.pre_roll_buffer.clear()
        self.voice_started = False
        self.vad.reset_states()

    def _is_input_device_usable(self, device_index: int):
        """Check whether an input device can be opened with current stream settings.

        Args:
            device_index: PyAudio device index to validate.
        """
        test_stream = None
        try:
            test_stream = self.agent.open(
                rate=self.sample_rate,
                channels=self.channels,
                format=self.format,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk,
            )
            return True
        except Exception as exc:
            logger.debug("Skipping unusable input device %s: %s", device_index, exc)
            return False
        finally:
            if test_stream is not None:
                test_stream.close()

    def _get_mic_info(self):
        """Log only microphones that are both input-capable and currently usable."""
        num_devices = self.agent.get_device_count()
        usable_count = 0

        for i in range(num_devices):
            info = self.agent.get_device_info_by_index(i)
            max_input_channels = int(info.get("maxInputChannels", 0))
            if max_input_channels <= 0:
                continue

            if not self._is_input_device_usable(i):
                continue

            usable_count += 1
            logger.info(
                "Usable mic: device_index=%s, name=%s, channels=%s, default_rate=%s",
                i,
                info.get("name", "Unknown"),
                max_input_channels,
                info.get("defaultSampleRate"),
            )

        logger.info("Total usable microphones: %s", usable_count)
                
        
    def callback(
        self,
        in_data: bytes,
        frame_count: int,
        time_info: dict[str, Any],
        status_flags: int,
    ):
        """Process incoming audio frames and stop stream on VAD speech end.

        Args:
            in_data: Raw PCM bytes for the current audio chunk.
            frame_count: Number of frames in this callback invocation.
            time_info: PyAudio timing metadata for the callback.
            status_flags: PyAudio status flags for stream health.
        """
        audio_int16 = np.frombuffer(in_data, np.int16)
        audio_float32 = self.int2float(sound=audio_int16)
        confidence = self.vad(torch.from_numpy(audio_float32))

        if confidence is not None:
            logger.debug("VAD confidence event: %s", confidence)
        self.confidence_list.append(confidence)

        if self.voice_started:
            self.frames.append(in_data)

        try:
            if confidence is not None:
                if confidence.get("start"):
                    self.voice_started = True

                    # Add pre-roll audio before speech start was detected by VAD.
                    starting_frame_in_chunk = confidence.get("start") % self.chunk
                    samples_needed_from_history = max(
                        0, self.num_previous_samples - starting_frame_in_chunk
                    )
                    bytes_needed_from_history = samples_needed_from_history * 2

                    if bytes_needed_from_history > 0 and self.pre_roll_buffer:
                        history_bytes = b"".join(self.pre_roll_buffer)
                        history_slice_size = min(
                            bytes_needed_from_history, len(history_bytes)
                        )
                        if history_slice_size > 0:
                            self.frames.append(history_bytes[-history_slice_size:])

                    if starting_frame_in_chunk > 0:
                        self.frames.append(in_data[: starting_frame_in_chunk * 2])

                    return (None, pyaudio.paContinue)

                if confidence.get("end"):
                    self.voice_started = False
                    return (None, pyaudio.paComplete)
        except KeyError:
            pass

        if not self.voice_started:
            self.pre_roll_buffer.append(in_data)

        return (None, pyaudio.paContinue)

    def open_stream(self):
        """Open and return a PyAudio input stream using the configured callback."""
        return self.agent.open(
            rate=self.sample_rate,
            channels=self.channels,
            format=self.format,
            input=True,
            frames_per_buffer=self.chunk,
            input_device_index=self.input_device_index,
            stream_callback=self.callback,
        )

    def save_recording(self, filename: str):
        """Write the collected in-memory frames to a WAV file.

        Args:
            filename: Full destination path for the output WAV file.
        """
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.agent.get_sample_size(self.format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b"".join(self.frames))

    def start_recording(self):
        """Continuously record voice segments and save each completed segment."""
        while True:
            stream = None
            try:
                filename = self._recording_filename()
                stream = self.open_stream()
                logger.info("Recording started")
                stream.start_stream()

                while stream.is_active():
                    time.sleep(1)

                if stream.is_active():
                    stream.stop_stream()
                stream.close()
                logger.info("Recording finished, cleaning up")

                self.save_recording(filename)
                logger.info("Saved to %s", filename)

                self._reset_segment_state()
            except KeyboardInterrupt:
                logger.info("Interrupt received, terminating")
                if stream is not None:
                    if stream.is_active():
                        stream.stop_stream()
                    stream.close()
                self.agent.terminate()
                break


def main():
    """Instantiate RealTimeAudioInput and start live audio recording."""
    audio_input = RealTimeAudioInput()
    audio_input._get_mic_info()


if __name__ == "__main__":
    main()