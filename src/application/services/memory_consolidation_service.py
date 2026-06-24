import uuid
from collections import defaultdict
from datetime import datetime
from typing import List

from application.ports.memory_port import MemoryPort
from core.models import MemoryFact, MemoryReflection


class MemoryConsolidationService:
    """Merge candidate memory events into durable facts and prompt files."""

    def __init__(self, memory: MemoryPort):
        self.memory = memory

    def consolidate(self, limit: int = 200) -> int:
        pending_events = self.memory.get_pending_consolidation(limit=limit)
        if not pending_events:
            return 0

        by_speaker = defaultdict(list)
        for event in pending_events:
            by_speaker[event.speaker_id].append(event)

        consolidated_at = datetime.now().isoformat()
        processed_event_ids: List[str] = []

        for speaker_id, events in by_speaker.items():
            existing_facts = self.memory.get_facts(speaker_id)
            existing_lookup = {self._normalize(fact.fact_text): fact for fact in existing_facts}

            accepted_facts: List[MemoryFact] = []
            for event in events:
                normalized = self._normalize(event.content)
                if normalized in existing_lookup:
                    processed_event_ids.append(event.event_id)
                    continue
                fact = MemoryFact(
                    fact_id=uuid.uuid4().hex,
                    speaker_id=speaker_id,
                    fact_text=event.content,
                    topic=event.event_kind,
                    valid_from=event.created_at,
                    source_event_ids=[event.event_id],
                    updated_at=consolidated_at,
                )
                self.memory.upsert_fact(fact)
                accepted_facts.append(fact)
                existing_lookup[normalized] = fact
                processed_event_ids.append(event.event_id)

            speaker = self.memory.get_speaker(speaker_id)
            if speaker is None:
                continue
            all_facts = self.memory.get_facts(speaker_id)
            profile_lines = [f"# {speaker.display_name}", "", "## Durable Facts"]
            if all_facts:
                for fact in all_facts:
                    topic = fact.topic or "fact"
                    profile_lines.append(f"- [{topic}] {fact.fact_text}")
            else:
                profile_lines.append("- No curated memory yet.")
            facets = self.memory.get_profile_facets(
                speaker_id,
                statuses=["durable", "tentative"],
                limit=30,
            )
            if facets:
                profile_lines.extend(["", "## Ambient Facets"])
                for facet in facets:
                    profile_lines.append(f"- [{facet.category}] {facet.summary}")
            self.memory.save_speaker_profile(speaker_id, "\n".join(profile_lines) + "\n")

            if accepted_facts:
                reflection = MemoryReflection(
                    reflection_id=uuid.uuid4().hex,
                    speaker_id=speaker_id,
                    summary=f"Consolidated {len(accepted_facts)} new durable memories for {speaker.display_name}.",
                    created_at=consolidated_at,
                    source_event_ids=[fact.source_event_ids[0] for fact in accepted_facts],
                )
                self.memory.add_reflection(reflection)

        recent_events = self.memory.get_recent_events(limit=10)
        recent_context_lines = ["# Recent Shared Context", ""]
        if recent_events:
            for event in reversed(recent_events):
                speaker = self.memory.get_speaker(event.speaker_id)
                name = speaker.display_name if speaker else event.speaker_id
                recent_context_lines.append(f"- {name}: {event.content}")
        else:
            recent_context_lines.append("- No recent shared context.")
        self.memory.save_recent_context("\n".join(recent_context_lines) + "\n")

        sessions = self.memory.list_sessions(statuses=["open"], limit=5)
        session_lines = ["# Session Digest", ""]
        if sessions:
            for session in sessions:
                session_lines.append(f"- {session.topic_summary or 'general conversation'}")
                if session.recent_turn_summary:
                    session_lines.append(f"  {session.recent_turn_summary}")
        else:
            session_lines.append("- No active sessions.")
        self.memory.save_session_digest("\n".join(session_lines) + "\n")

        loops = self.memory.list_open_loops(statuses=["open", "tentative"], limit=10)
        loop_lines = ["# Open Loops", ""]
        if loops:
            for loop in loops:
                suffix = f" | due: {loop.due_hint}" if loop.due_hint else ""
                loop_lines.append(f"- [{loop.status}] {loop.title}{suffix}")
        else:
            loop_lines.append("- No tracked open loops.")
        self.memory.save_open_loop_digest("\n".join(loop_lines) + "\n")

        self.memory.mark_events_consolidated(processed_event_ids, consolidated_at)
        return len(processed_event_ids)

    def _normalize(self, text: str) -> str:
        return " ".join(text.lower().split())
