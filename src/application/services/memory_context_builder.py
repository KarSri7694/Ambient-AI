from pathlib import Path
from typing import Iterable

from application.ports.memory_port import MemoryPort
from core.models import TranscriptParticipant


class MemoryContextBuilder:
    """Assemble prompt context from static prompts and memory artifacts."""

    def __init__(self, memory: MemoryPort, prompts_root: str):
        self.memory = memory
        self.prompts_root = Path(prompts_root)

    def build_prompt(
        self,
        base_prompt_filename: str,
        skills_summary: str,
        participants: Iterable[TranscriptParticipant],
        include_skills: bool = True,
        semantic_memory_section: str = "",
    ) -> str:
        prompt_parts = [
            self._read_prompt_file(base_prompt_filename),
            self._read_prompt_file("USER.md"),
            self._build_user_info_section(),
            self._build_session_digest_section(),
            self._build_open_loop_digest_section(),
            self._build_visual_digest_section(),
            semantic_memory_section,
            self._build_recent_context_section(),
            self._build_participant_memory_section(participants),
        ]
        if include_skills:
            prompt_parts.append(skills_summary)
        return "\n\n".join(part for part in prompt_parts if part.strip())

    def _read_prompt_file(self, filename: str) -> str:
        path = self.prompts_root / filename
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8").strip()

    def _build_recent_context_section(self) -> str:
        recent_context = self.memory.get_recent_context().strip()
        if not recent_context:
            recent_context = "_No recent shared context yet._"
        return f"## Recent Shared Context\n{recent_context}"

    def _build_user_info_section(self) -> str:
        user_info = self.memory.get_user_info().strip()
        if not user_info:
            user_info = "_No durable visual user info yet._"
        return f"## USER_INFO\n{user_info}"

    def _build_session_digest_section(self) -> str:
        session_digest = self.memory.get_session_digest().strip()
        if not session_digest:
            session_digest = "_No active session digest yet._"
        return f"## Session Continuity\n{session_digest}"

    def _build_open_loop_digest_section(self) -> str:
        open_loop_digest = self.memory.get_open_loop_digest().strip()
        if not open_loop_digest:
            open_loop_digest = "_No open loops tracked yet._"
        return f"## Open Loops\n{open_loop_digest}"

    def _build_visual_digest_section(self) -> str:
        visual_digest = self.memory.get_visual_digest().strip()
        if not visual_digest:
            visual_digest = "_No passive visual context yet._"
        return f"## Passive Visual Context\n{visual_digest}"

    def _build_participant_memory_section(
        self,
        participants: Iterable[TranscriptParticipant],
    ) -> str:
        sections = ["## Participant Memory"]
        seen_speaker_ids: set[str] = set()

        for participant in participants:
            if participant.speaker_id in seen_speaker_ids:
                continue
            seen_speaker_ids.add(participant.speaker_id)
            profile = self.memory.get_speaker_profile(participant.speaker_id).strip()
            if not profile:
                profile = "_No curated memory yet._"

            recent_events = self.memory.get_recent_events(
                speaker_ids=[participant.speaker_id],
                limit=3,
            )
            event_lines = [f"- [{event.event_kind}] {event.content}" for event in recent_events]
            events_block = "\n".join(event_lines) if event_lines else "- No recent memory events."

            sections.append(
                "\n".join(
                    [
                        f"### {participant.display_name}",
                        f"Speaker label in transcript: {participant.speaker_label}",
                        profile,
                        "Recent memory events:",
                        events_block,
                    ]
                )
            )
        return "\n\n".join(sections)
