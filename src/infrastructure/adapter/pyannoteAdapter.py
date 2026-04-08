from core.models import DiarizationResult
from application.ports.asr_port import DiarizationPort
import logging
from pathlib import Path
import torchaudio
import torch
from pyannote.audio import Pipeline
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Don't change working directory - use absolute paths instead
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent  # Go up to ambient_ai root

DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
HF_TOKEN_FILE = "D:\\Projects\\ambient_ai\\HFToken.txt"

class PyannoteAdapter(DiarizationPort):
    def __init__(self, hf_token: str):
        self.diarization_model = Pipeline.from_pretrained(DIARIZATION_MODEL, token=hf_token)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.diarization_model.to(self.device)
        except RuntimeError as exc:
            # Recover from low-VRAM startup by running diarization on CPU.
            if self.device.type == "cuda" and "out of memory" in str(exc).lower():
                logging.warning("CUDA OOM while loading diarization model. Falling back to CPU.")
                torch.cuda.empty_cache()
                self.device = torch.device("cpu")
                self.diarization_model.to(self.device)
            else:
                raise

    def _diarize_audio_data(self, audio_data: dict, audio_label: str, sample_rate: int) -> List[DiarizationResult]:
        try:
            diarized_segments = self.diarization_model(audio_data)
        except RuntimeError as exc:
            error_text = str(exc).lower()
            recoverable_cuda_error = "out of memory" in error_text or "unable to find an engine" in error_text
            if self.device.type == "cuda" and recoverable_cuda_error:
                logging.warning("Diarization failed on CUDA (%s). Retrying on CPU.", exc)
                torch.cuda.empty_cache()
                self.device = torch.device("cpu")
                self.diarization_model.to(self.device)
                diarized_segments = self.diarization_model(audio_data)
            else:
                raise
        diarization_result: List[DiarizationResult] = []
        for segment, _, speaker in diarized_segments.speaker_diarization.itertracks(yield_label=True):
            diarization_result.append(
                DiarizationResult(
                    start_time=segment.start,
                    end_time=segment.end,
                    speaker_label=speaker,
                    audio_file=audio_label,
                    sample_rate=sample_rate,
                )
            )
        return diarization_result

    def diarize_audio(self, audio_file_path: str) -> List[DiarizationResult]:
        # Convert to absolute path to avoid working directory issues
        audio_path = Path(audio_file_path)
        if not audio_path.is_absolute():
            audio_path = project_root / audio_file_path
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        waveform, sample_rate = torchaudio.load(str(audio_path))
        audio_data = {
            "waveform": waveform,
            "sample_rate": sample_rate
        }
        return self._diarize_audio_data(audio_data, audio_file_path, sample_rate)
    
    def diarize_waveform(self, waveform: torch.Tensor, sample_rate: int, audio_label: str = "in_memory_audio") -> List[DiarizationResult]:
        audio_data = {
            "waveform": waveform,
            "sample_rate": sample_rate,
        }
        return self._diarize_audio_data(audio_data, audio_label, sample_rate)
    
    def unload_model(self):
        """
        Unloads the diarization model from memory.
        """
        self.diarization_model = None
