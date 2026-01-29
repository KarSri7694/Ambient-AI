from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datetime import datetime

@dataclass(frozen=True)
class TranscriptionResult:
    """
    Represents a single transcription segment from an audio file.
    """
    start_time: float
    end_time: float
    speaker_label: str
    transcription_text: str
    word_timestamps: Optional[List[Dict[str, float]]] = None  

@dataclass(frozen=True)
class AudioMetadata:
    file_path: str
    source: str  # e.g., "web_upload", "local_mic"
    timestamp: datetime
    duration_seconds: float | None = None

@dataclass
class SpeakerSegment:
    """Represents a single speaker turn in the audio"""
    start_time: float
    end_time: float
    speaker_label: str
    audio_tensor: Optional[list[float]] = None
    embedding: Optional[list[float]] = None
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

@dataclass
class SpeakerInfo:
    """Metadata about an identified speaker"""
    speaker_label: str
    identified_name: str = "UNKNOWN"
    confidence_score: float = 0.0
    total_speaking_time: float = 0.0
    segment_count: int = 0
    combined_embedding: Optional[list[float]] = None

@dataclass
class DiarizationResult:
    """Complete diarization result with all speaker information"""
    start_time: float = 0.0
    end_time: float = 0.0
    speaker_label: str = "UNKNOWN"
    audio_file: Optional[str] = None
    sample_rate: int = 16000
    
    def get_segments_by_speaker(self, speaker_label: str) -> List[SpeakerSegment]:
        """Get all segments for a specific speaker"""
        return [seg for seg in self.segments if seg.speaker_label == speaker_label]
    
    def get_speaker_timeline(self) -> List[tuple]:
        """Get chronological timeline of speakers"""
        return [(seg.start_time, seg.end_time, seg.speaker_label) 
                for seg in sorted(self.segments, key=lambda x: x.start_time)]

@dataclass
class SpeakerEmbedding:
    """Holds speaker embedding data"""
    speaker_label: str
    embedding: list[float]

@dataclass
class SpeakerScore:
    """Holds speaker identification score data"""
    speaker_label: str
    score: float

@dataclass
class SpeakerMapping:
    """Mapping between diarization speaker labels and identified speaker names"""
    original_label: str
    identified_label: str
    score: float