import torch

from core.models import TranscriptionResult
from application.ports.asr_port import TranscriptionPort
from faster_whisper import WhisperModel, BatchedInferencePipeline
from pathlib import Path
import requests
import logging
import base64
from openai import OpenAI
from config import CONFIG

logging.basicConfig(level=logging.INFO)

API_BASE_URL = CONFIG.get_str("runtime", "api_base_url", "http://localhost:8080")
API_KEY = CONFIG.get_str("runtime", "api_key", "testkey")

class WhisperAdapter(TranscriptionPort):
    def __init__(self, model_size: str = "HIN2HINGLISH", device: str = "cpu"):
        self.model = WhisperModel(model_size, device=device)
        self.batched_model = BatchedInferencePipeline(self.model)

    def transcribe_audio(self, audio_file_path: str, vad_filter: bool, word_timestamps: bool, batch_size: int = 8) -> list[TranscriptionResult]:
        """
        Generates transcriptions for the given audio file using a batched inference pipeline.
        Args:
            audio_file_path (str): Path to the audio file to be transcribed.
            vad_filter (bool): Whether to apply VAD filtering.
            word_timestamps (bool): Whether to include word-level timestamps.
            batch_size (int): The number of audio files to process in a single batch.
        """
        segments, _ = self.batched_model.transcribe(audio_file_path, vad_filter=vad_filter, word_timestamps=word_timestamps, batch_size=batch_size)
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

 
class QwenASRAdapter(TranscriptionPort):
    def __init__(self, model_size: str = "QWEN_ASR", device: str = "cpu"):
        self.model_size = model_size
        self.model = self.load_model()
        self.client = OpenAI(base_url=f"{API_BASE_URL}/v1", api_key=API_KEY) 
    
    def load_model(self):
        """
        Loads the Qwen ASR model from the specified path.
        """
        BASE_URL = API_BASE_URL

        response = requests.post(
            f"{BASE_URL}/models/load",
            json={"model": self.model_size},
            timeout=30,
        )
        if response.status_code != 200:
            logging.error(f"Failed to load model {self.model_size}: {response.text}")
        elif response.status_code == 200:
            logging.info(f"Model loaded successfully: {self.model_size}")
    
    def transcribe_audio(self, audio_file_path: str, vad_filter: bool, word_timestamps: bool, batch_size: int = 1) -> list[TranscriptionResult]:
        """
        Transcribes the given audio file using the Qwen ASR model.
        Args:
            audio_file_path (str): Path to the audio file to be transcribed.
            vad_filter (bool): Whether to apply VAD filtering.
            word_timestamps (bool): Whether to include word-level timestamps.
        """
        if word_timestamps:
            logging.warning("Word timestamps are not supported in Qwen ASR. Ignoring the word_timestamps parameter.")
        audio_format = Path(audio_file_path).suffix.lower().lstrip(".")
        if audio_format not in {"wav", "mp3"}:
            raise ValueError(
                f"llama.cpp input_audio only supports wav or mp3, got '{audio_format}' from {audio_file_path}"
            )

        with open(audio_file_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        
        messages = []
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_base64,
                        "format": audio_format,
                    },
                },
                {"type": "text", "text": "Transcribe the audio."},
            ],
        })
        
        
        completions = self.client.chat.completions.create(
            model="Qwen_ASR",
            messages=messages,
            stream=True,
        )
        
        transcript = ""
        for chunk in completions:
            if hasattr(chunk.choices[0].delta, "content"):
                if chunk.choices[0].delta.content:
                    transcript += chunk.choices[0].delta.content 
            else:
                transcript += ""
        
        transcript = transcript.split("<asr_text>")[1]
        def get_timestamps():
            import torch
            from qwen_asr import Qwen3ForcedAligner

            aligner = Qwen3ForcedAligner.from_pretrained(
                "Qwen/Qwen3-ForcedAligner-0.6B",
                dtype=torch.bfloat16,
                device_map="cuda:0",
            )

            results = aligner.align(
                audio=audio_file_path,
                text=transcript,
                language=["Hindi"],
            )
            aligner = None
            return results[0]
        
        self.unload_model()
        aligned_transcript = get_timestamps()
        return [TranscriptionResult(
            start_time=segment.start_time,
            end_time=segment.end_time,
            speaker_label="UNKNOWN",
            transcription_text=segment.text,
            word_timestamps = [{"word": segment.text, "start": segment.start_time, "end": segment.end_time, "probability": 0.5}] if word_timestamps else None
        ) for segment in aligned_transcript]

    def unload_model(self):
        """
        Unloads the Qwen-ASR model from memory.
        """
        requests.post(
            f"http://localhost:8080/models/unload",
            json={"model": self.model_size},
            timeout=30,
        )
        
class MegaASRAdapter(TranscriptionPort):
    pass

class NemotronASRAdapter(TranscriptionPort):
    pass
