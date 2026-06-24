import json
import logging
from datetime import datetime
from typing import List

from application.ports.notification_port import NotificationPort
from application.ports.proactive_topic_queue_port import ProactiveTopicQueuePort
from application.services.llm_interaction_service import LLMInteractionService
from application.services.research_vault_service import ResearchVaultService
from core.models import ProactiveTopicCandidate, ResearchPackageResult


class ProactiveResearchService:
    """Drain the proactive topic queue into saved research packages."""

    ALLOWED_TOOL_NAMES = {
        "google_search",
        "powershell_terminal",
        "queue_night_task",
    }

    def __init__(
        self,
        llm_service: LLMInteractionService,
        topic_queue: ProactiveTopicQueuePort,
        vault: ResearchVaultService,
        notifications: NotificationPort,
        logger: logging.Logger | None = None,
    ):
        self.llm_service = llm_service
        self.topic_queue = topic_queue
        self.vault = vault
        self.notifications = notifications
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    async def process_next_topic(self, system_prompt: str, model: str) -> bool:
        pending = self.topic_queue.get_pending_topics(limit=1)
        if not pending:
            return False
        topic = pending[0]
        package = await self._research_topic(topic, system_prompt, model)
        now = datetime.now().isoformat()
        self.topic_queue.mark_topic_status(
            topic.topic_id,
            status="completed",
            artifact_path=package.artifact_path,
            last_researched_at=now,
        )
        if package.meaningful_change:
            action = "Updated" if package.was_update else "Created"
            self.notifications.add_notification(
                f"{action} research package: {package.display_title} at {package.artifact_path}",
                source="proactive_research",
            )
        return True

    async def process_topics(
        self,
        system_prompt: str,
        model: str,
        max_topics: int,
    ) -> int:
        processed = 0
        for _ in range(max_topics):
            worked = await self.process_next_topic(system_prompt=system_prompt, model=model)
            if not worked:
                break
            processed += 1
        return processed

    async def _research_topic(
        self,
        topic: ProactiveTopicCandidate,
        system_prompt: str,
        model: str,
    ) -> ResearchPackageResult:
        existing_notes = self.vault.read_existing_notes(topic.normalized_topic)
        user_input = (
            f"Topic title: {topic.display_title}\n"
            f"Normalized topic: {topic.normalized_topic}\n"
            f"Summary hint: {topic.summary_hint}\n"
            f"Source speaker: {topic.speaker_label}\n"
            f"Source reference: {topic.source_ref}\n"
            f"Existing notes:\n{existing_notes or 'None'}\n\n"
            "Research this topic in a bounded medium package. "
            "Use available tools to gather useful links and concise notes. "
            "Return JSON only with keys summary, notes, and links. "
            "links must be a JSON array of objects with title and url."
        )
        response = await self.llm_service.run_interaction(
            user_input=user_input,
            system_prompt=system_prompt,
            model=model,
            allowed_tool_names=self.ALLOWED_TOOL_NAMES,
        )
        parsed = self._parse_response(response, topic)
        return self.vault.save_package(
            normalized_topic=topic.normalized_topic,
            display_title=topic.display_title,
            summary=parsed["summary"],
            notes=parsed["notes"],
            links=parsed["links"],
        )

    def _parse_response(
        self,
        response: str,
        topic: ProactiveTopicCandidate,
    ) -> dict:
        response = response.strip()
        start = response.find("{")
        end = response.rfind("}")
        if start != -1 and end != -1:
            response = response[start : end + 1]
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            self.logger.warning("Research response was not valid JSON for topic %s", topic.display_title)
            parsed = {}

        summary = str(parsed.get("summary", "")).strip() or f"Research package for {topic.display_title}."
        notes = str(parsed.get("notes", "")).strip() or topic.summary_hint
        links = self._normalize_links(parsed.get("links", []))
        return {
            "summary": summary,
            "notes": notes,
            "links": links,
        }

    def _normalize_links(self, links) -> List[dict[str, str]]:
        normalized: List[dict[str, str]] = []
        if not isinstance(links, list):
            return normalized
        for item in links:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            url = str(item.get("url", "")).strip()
            if title and url:
                normalized.append({"title": title, "url": url})
        return normalized
