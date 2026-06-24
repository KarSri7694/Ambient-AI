import re
import uuid
from datetime import datetime
from typing import Optional

from application.ports.proactive_topic_queue_port import ProactiveTopicQueuePort
from application.services.research_vault_service import ResearchVaultService
from core.models import ProactiveTopicCandidate, TranscriptClassificationResult


class ProactiveTopicDetectionService:
    """Nominate salient proactive research topics from transcript classifications."""

    ELIGIBLE_LABELS = {"FACT", "PREFERENCE", "TASK_COMPLEX"}
    MIN_SALIENCE_SCORE = 0.55
    STOPWORDS = {
        "the", "and", "about", "that", "this", "with", "from", "your", "have", "will",
        "please", "need", "notes", "resources", "learn", "game", "show", "topic", "somewhere",
    }

    def __init__(self, queue: ProactiveTopicQueuePort, vault: ResearchVaultService):
        self.queue = queue
        self.vault = vault

    def maybe_queue_topic(
        self,
        classification: TranscriptClassificationResult,
        transcript_text: str,
        source_ref: str,
    ) -> Optional[ProactiveTopicCandidate]:
        if classification.label not in self.ELIGIBLE_LABELS:
            return None

        display_title = self._extract_display_title(classification)
        normalized_topic = self.vault.slugify(display_title)
        salience = self._compute_salience(classification, normalized_topic)
        if salience < self.MIN_SALIENCE_SCORE:
            return None

        now = datetime.now().isoformat()
        candidate = ProactiveTopicCandidate(
            topic_id=uuid.uuid4().hex,
            normalized_topic=normalized_topic,
            display_title=display_title,
            source_ref=source_ref,
            speaker_label=classification.speaker_label,
            summary_hint=classification.summary or transcript_text[:160],
            salience_score=salience,
            status="pending",
            first_seen_at=now,
            last_seen_at=now,
            artifact_path=self.vault.get_existing_artifact_path(normalized_topic),
        )
        return self.queue.upsert_topic(candidate)

    def _extract_display_title(self, classification: TranscriptClassificationResult) -> str:
        text = classification.suggested_action or classification.memory_content or classification.summary
        text = re.sub(r"[^A-Za-z0-9\s-]", " ", text)
        words = [word for word in text.split() if word.lower() not in self.STOPWORDS]
        if not words:
            words = classification.summary.split()
        title = " ".join(words[:6]).strip()
        return title or "ambient-topic"

    def _compute_salience(self, classification: TranscriptClassificationResult, normalized_topic: str) -> float:
        score = 0.0
        score += min(max(classification.confidence, 0.0), 1.0) * 0.5
        if classification.label == "TASK_COMPLEX":
            score += 0.25
        elif classification.label == "PREFERENCE":
            score += 0.15
        else:
            score += 0.1

        novelty_bonus = 0.2 if self.queue.find_by_normalized_topic(normalized_topic) is None else 0.05
        artifact_bonus = 0.0 if self.vault.get_existing_artifact_path(normalized_topic) else 0.1
        token_bonus = 0.05 if len(normalized_topic.split("-")) >= 2 else 0.0
        score += novelty_bonus + artifact_bonus + token_bonus
        return min(score, 1.0)
