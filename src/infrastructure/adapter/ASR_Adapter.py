from core.models import TranscriptionResult
from application.ports.asr_port import TranscriptionPort
from faster_whisper import WhisperModel, BatchedInferencePipeline

class WhisperAdapter(TranscriptionPort):
    def __init__(self, model_size: str = "HIN2HINGLISH", device: str = "cpu"):
        self.model = WhisperModel(model_size, device=device)
        self.batched_model = BatchedInferencePipeline(self.model)

    def transcribe_audio(self, audio_file_path: str, vad_filter: bool, word_timestamps: bool) -> list[TranscriptionResult]:
        """
        Generates transcriptions for the given audio file using a batched inference pipeline.
        Args:
            audio_file_path (str): Path to the audio file to be transcribed.
            vad_filter (bool): Whether to apply VAD filtering.
            word_timestamps (bool): Whether to include word-level timestamps.
        """
        segments, _ = self.batched_model.transcribe(audio_file_path, vad_filter=vad_filter, word_timestamps=word_timestamps, batch_size=8)
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
        