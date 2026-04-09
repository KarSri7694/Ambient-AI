from core.models import TranscriptionResult
from application.ports.asr_port import TranscriptionPort
from faster_whisper import WhisperModel, BatchedInferencePipeline
from typing import Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WhisperAdapter(TranscriptionPort):
    def __init__(self, model_size: str = "HIN2HINGLISH", device: str = "cpu"):
        self.model = WhisperModel(model_size, device=device)
        self.batched_model = BatchedInferencePipeline(self.model)

    def transcribe_audio(self, audio_input: Any, vad_filter: bool, word_timestamps: bool, batch_size: int = 8) -> list[TranscriptionResult]:
        """
        Generates transcriptions for the given audio input using a batched inference pipeline.
        Args:
            audio_input: Path to the audio file or in-memory waveform supported by faster-whisper.
            vad_filter (bool): Whether to apply VAD filtering.
            word_timestamps (bool): Whether to include word-level timestamps.
            batch_size (int): The number of audio files to process in a single batch.
        """
        logging.info("Transcribing audio file: %s", audio_input)
        segments, _ = self.batched_model.transcribe(audio_input, vad_filter=vad_filter, word_timestamps=word_timestamps, batch_size=batch_size)
        return [TranscriptionResult(
            start_time=segment.start,
            end_time=segment.end,
            speaker_label="UNKNOWN",
            transcription_text=segment.text,
            word_timestamps=[{"word": word.word, "start": word.start, "end": word.end, "probability": word.probability} for word in segment.words] if word_timestamps else None
        ) for segment in segments]
    
    def unload_model(self):
        """
        Unloads the Whisper model from memory.
        """
        self.model = None
        self.batched_model = None
        
