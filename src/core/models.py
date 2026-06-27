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

@dataclass
class ChatMessage:
    """Represents a single message in an LLM conversation."""
    role: str  # "system", "user", "assistant", "tool"
    content: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    def to_dict(self) -> Dict:
        d = {"role": self.role}
        if self.content is not None:
            d["content"] = self.content
        if self.tool_calls is not None:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            d["name"] = self.name
        return d

@dataclass
class ToolCallInfo:
    """Represents a single tool invocation request from the LLM."""
    id: str
    name: str
    arguments: str  # JSON string

@dataclass
class NightTask:
    """A queued task for night-time autonomous execution."""
    id: int
    description: str
    priority: str = "medium"
    status: str = "pending"
    created_at: Optional[str] = None
    metadata_json: Optional[str] = None

@dataclass
class Notification:
    """A system notification."""
    id: int
    message: str
    source: str = "system"


@dataclass(frozen=True)
class SpeakerRecord:
    """A durable speaker identity used by the memory system."""
    speaker_id: str
    display_name: str
    source_label: str
    voice_embedding_uid: Optional[int] = None
    is_user: bool = False
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True)
class TranscriptTurn:
    """A parsed speaker turn from a merged transcript file."""
    speaker_label: str
    text: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None


@dataclass(frozen=True)
class TranscriptParticipant:
    """Resolved participant metadata for a transcript speaker label."""
    speaker_label: str
    speaker_id: str
    display_name: str
    confidence: float = 0.0
    durable: bool = True


@dataclass(frozen=True)
class MemoryEvent:
    """A candidate or consolidated memory event extracted from transcripts."""
    event_id: str
    speaker_id: str
    source_type: str
    source_ref: str
    event_kind: str
    content: str
    confidence: float
    status: str
    created_at: str
    consolidated_at: Optional[str] = None


@dataclass(frozen=True)
class MemoryFact:
    """A durable fact kept in long-term memory."""
    fact_id: str
    speaker_id: str
    fact_text: str
    topic: Optional[str] = None
    valid_from: str = ""
    valid_to: Optional[str] = None
    superseded_by: Optional[str] = None
    source_event_ids: List[str] = field(default_factory=list)
    updated_at: str = ""


@dataclass(frozen=True)
class MemoryReflection:
    """A stored summary generated during memory consolidation."""
    reflection_id: str
    speaker_id: Optional[str]
    summary: str
    created_at: str
    source_event_ids: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class TranscriptClassificationResult:
    """A single bounded classification result for one transcript chunk."""
    label: str
    speaker_label: str
    summary: str
    confidence: float
    reason: str
    suggested_action: Optional[str] = None
    memory_content: Optional[str] = None


@dataclass(frozen=True)
class ProactiveTopicCandidate:
    """A queued proactive research topic inferred from ambient context."""
    topic_id: str
    normalized_topic: str
    display_title: str
    source_ref: str
    speaker_label: str
    summary_hint: str
    salience_score: float
    status: str = "pending"
    first_seen_at: str = ""
    last_seen_at: str = ""
    artifact_path: Optional[str] = None
    last_researched_at: Optional[str] = None


@dataclass(frozen=True)
class ResearchPackageResult:
    """Result of producing or updating a proactive research package."""
    display_title: str
    artifact_path: str
    summary: str
    notes: str
    links: List[Dict[str, str]] = field(default_factory=list)
    was_update: bool = False
    meaningful_change: bool = True


@dataclass(frozen=True)
class AmbientAgendaItem:
    """A durable ambient priority considered between transcript turns."""
    agenda_id: str
    title: str
    kind: str
    source_type: str
    source_ref: str
    priority_score: float
    status: str
    created_at: str
    updated_at: str
    due_at: Optional[str] = None
    last_considered_at: Optional[str] = None
    backing_topic_id: Optional[str] = None
    backing_memory_ids: List[str] = field(default_factory=list)
    surface_message: Optional[str] = None


@dataclass(frozen=True)
class AmbientReflectionAction:
    """One bounded action chosen by the reflection layer."""
    action_type: str
    payload: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TranscriptEvidence:
    """Structured evidence extracted from one transcript turn."""
    evidence_id: str
    source_ref: str
    speaker_id: str
    speaker_label: str
    session_id: Optional[str]
    signal_type: str
    content: str
    normalized_entities: List[str] = field(default_factory=list)
    time_hints: List[str] = field(default_factory=list)
    action_hints: List[str] = field(default_factory=list)
    trust_score: float = 0.0
    created_at: str = ""


@dataclass(frozen=True)
class ConversationSession:
    """A transcript-backed conversation episode spanning multiple chunks."""
    session_id: str
    started_at: str
    ended_at: str
    participant_ids: List[str] = field(default_factory=list)
    status: str = "open"
    topic_summary: str = ""
    entity_summary: str = ""
    recent_turn_summary: str = ""
    last_activity_at: str = ""
    continuation_score: float = 0.0
    derived_loop_ids: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class OpenLoop:
    """A durable unresolved intent inferred from transcript evidence."""
    loop_id: str
    title: str
    loop_type: str
    status: str
    owner_speaker_id: str
    source_session_id: str
    supporting_event_ids: List[str] = field(default_factory=list)
    confidence: float = 0.0
    urgency: float = 0.0
    due_hint: Optional[str] = None
    next_action_hint: Optional[str] = None
    last_updated_at: str = ""
    resolution_summary: Optional[str] = None


@dataclass(frozen=True)
class UserProfileFacet:
    """One categorized profile entry derived from repeated transcript evidence."""
    facet_id: str
    speaker_id: str
    category: str
    title: str
    summary: str
    confidence: float = 0.0
    strength: int = 1
    status: str = "tentative"
    source_event_ids: List[str] = field(default_factory=list)
    updated_at: str = ""


@dataclass(frozen=True)
class VisualObservation:
    """One passive visual inference derived from a screenshot."""
    observation_id: str
    screenshot_path: str
    created_at: str
    observation_type: str = "screen"
    app_name: Optional[str] = None
    window_title: Optional[str] = None
    page_hint: Optional[str] = None
    summary: str = ""
    detailed_description: str = ""
    inferred_user_activity: str = ""
    previous_activity_status: str = "unclear"
    salient_entities: List[str] = field(default_factory=list)
    completed_items: List[str] = field(default_factory=list)
    open_loops: List[str] = field(default_factory=list)
    possible_next_task: Optional[str] = None
    suggested_research_topics: List[str] = field(default_factory=list)
    user_fact_hypotheses: List[Dict[str, str]] = field(default_factory=list)
    confidence: float = 0.0
    session_id: Optional[str] = None
    followup_sent_at: Optional[str] = None
    raw_payload_json: Optional[str] = None


@dataclass(frozen=True)
class VisualUserFact:
    """A visual user-fact candidate promoted from repeated passive observations."""
    fact_id: str
    fact_key: str
    category: str
    title: str
    summary: str
    status: str = "temporary"
    score: float = 0.0
    observation_count: int = 1
    session_count: int = 1
    first_seen_at: str = ""
    last_seen_at: str = ""
    source_observation_ids: List[str] = field(default_factory=list)
    source_session_ids: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class VisualSession:
    """A short-lived passive visual activity session spanning related screenshots."""
    session_id: str
    started_at: str
    ended_at: str
    status: str = "open"
    activity_summary: str = ""
    app_name: Optional[str] = None
    window_title: Optional[str] = None
    page_hint: Optional[str] = None
    last_activity_at: str = ""
    continuation_score: float = 0.0
    observation_ids: List[str] = field(default_factory=list)
    related_loop_ids: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class FusedContextEpisode:
    """A cross-modal episode derived from transcript evidence and visual observations."""
    episode_id: str
    started_at: str
    ended_at: str
    transcript_evidence_ids: List[str] = field(default_factory=list)
    visual_observation_ids: List[str] = field(default_factory=list)
    source_refs: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    activity_summary: str = ""
    inferred_intent: str = ""
    confidence: float = 0.0
    user_fact_candidates: List[Dict[str, str]] = field(default_factory=list)
    open_loop_candidates: List[Dict[str, str]] = field(default_factory=list)
    suggested_next_action: Optional[str] = None
    status: str = "active"
    created_at: str = ""
    updated_at: str = ""
    raw_payload_json: Optional[str] = None


@dataclass(frozen=True)
class SemanticMemoryChunk:
    """A searchable text chunk derived from durable ambient memory."""
    chunk_id: str
    source_type: str
    source_id: str
    source_ref: str
    speaker_id: Optional[str]
    content: str
    metadata_json: str = "{}"
    embedding: Optional[List[float]] = None
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True)
class SemanticMemoryResult:
    """A ranked semantic memory hit for prompt injection."""
    chunk: SemanticMemoryChunk
    vector_score: float = 0.0
    rerank_score: Optional[float] = None


@dataclass(frozen=True)
class InteractionLogEntry:
    """One persisted LLM interaction request/response pair."""
    interaction_id: str
    created_at: str
    completed_at: Optional[str]
    source: str
    model: str
    messages_json: str
    tools_json: Optional[str] = None
    image_path: Optional[str] = None
    response_text: Optional[str] = None
    reasoning_text: Optional[str] = None
    tool_calls_json: Optional[str] = None
    error_text: Optional[str] = None
    duration_ms: Optional[int] = None
    metadata_json: Optional[str] = None


@dataclass(frozen=True)
class ActivityRun:
    """One user-facing unit of meaningful agent work."""
    run_id: str
    created_at: str
    completed_at: Optional[str]
    status: str
    source_kind: str
    trigger_kind: str
    title: str
    summary: str = ""
    output_text: str = ""
    model: str = ""
    session_id: Optional[str] = None
    parent_run_id: Optional[str] = None
    priority: str = "medium"
    error_text: Optional[str] = None
    metadata_json: Optional[str] = None


@dataclass(frozen=True)
class ActivityStep:
    """One ordered sub-step within an activity run."""
    step_id: str
    run_id: str
    step_index: int
    step_kind: str
    title: str
    status: str
    started_at: str
    completed_at: Optional[str] = None
    input_ref: Optional[str] = None
    output_ref: Optional[str] = None
    error_text: Optional[str] = None
    metadata_json: Optional[str] = None


@dataclass(frozen=True)
class ActivityArtifact:
    """One linked file or small preview associated with a run."""
    artifact_id: str
    run_id: str
    step_id: Optional[str]
    artifact_kind: str
    title: str
    path: Optional[str]
    mime_type: Optional[str]
    text_preview: Optional[str]
    metadata_json: Optional[str]
    created_at: str


@dataclass(frozen=True)
class ActivityLink:
    """One cross-link from an activity run to another durable entity."""
    link_id: str
    run_id: str
    entity_type: str
    entity_id: str
    relation: str
    metadata_json: Optional[str]


@dataclass(frozen=True)
class ActivityTraceLink:
    """One linked raw interaction trace that belongs to a run."""
    interaction_id: str
    created_at: str
    source: str
    model: str
    response_text: Optional[str] = None
    reasoning_text: Optional[str] = None
    tool_calls_json: Optional[str] = None
    error_text: Optional[str] = None
    duration_ms: Optional[int] = None
    metadata_json: Optional[str] = None


@dataclass(frozen=True)
class ActivityRunDetail:
    """Expanded view of a run for dashboards and API consumers."""
    run: ActivityRun
    steps: List[ActivityStep] = field(default_factory=list)
    artifacts: List[ActivityArtifact] = field(default_factory=list)
    links: List[ActivityLink] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    traces: List[ActivityTraceLink] = field(default_factory=list)
