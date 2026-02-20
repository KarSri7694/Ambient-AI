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
    def __init__(self, hf_token_file: str = HF_TOKEN_FILE):
        with open(hf_token_file, 'r') as file:
            hf_token = file.read().strip()
        self.diarization_model = Pipeline.from_pretrained(DIARIZATION_MODEL, token=hf_token)

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
        
        self.diarization_model.to(torch.device("cuda"))
        diarized_segments = self.diarization_model(audio_data)
        diarization_result = []
        for segment, _, speaker in diarized_segments.speaker_diarization.itertracks(yield_label=True):
            diarization_result.append(
                DiarizationResult(
                    start_time=segment.start,
                    end_time=segment.end,
                    speaker_label=speaker,
                    audio_file=audio_file_path,
                    sample_rate=sample_rate,
                )
            )
            
        return diarization_result
    
    def unload_model(self):
        """
        Unloads the diarization model from memory.
        """
        self.diarization_model = None