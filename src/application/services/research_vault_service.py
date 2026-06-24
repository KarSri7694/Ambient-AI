import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from core.models import ResearchPackageResult


class ResearchVaultService:
    """Manage the proactive research vault on disk."""

    def __init__(self, vault_root: str):
        self.vault_root = Path(vault_root)
        self.vault_root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.vault_root / "index.json"
        if not self.index_path.exists():
            self.index_path.write_text("[]", encoding="utf-8")

    def slugify(self, title: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        return slug or "untitled-topic"

    def topic_dir(self, normalized_topic: str) -> Path:
        return self.vault_root / normalized_topic

    def get_existing_artifact_path(self, normalized_topic: str) -> Optional[str]:
        topic_dir = self.topic_dir(normalized_topic)
        if topic_dir.exists():
            return str(topic_dir)
        return None

    def read_existing_notes(self, normalized_topic: str) -> str:
        notes_path = self.topic_dir(normalized_topic) / "notes.md"
        if not notes_path.exists():
            return ""
        return notes_path.read_text(encoding="utf-8")

    def save_package(
        self,
        normalized_topic: str,
        display_title: str,
        summary: str,
        notes: str,
        links: list[dict[str, str]],
    ) -> ResearchPackageResult:
        topic_dir = self.topic_dir(normalized_topic)
        was_update = topic_dir.exists()
        topic_dir.mkdir(parents=True, exist_ok=True)

        summary_path = topic_dir / "summary.md"
        notes_path = topic_dir / "notes.md"
        links_path = topic_dir / "links.json"
        previous_summary = summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""
        previous_links = links_path.read_text(encoding="utf-8") if links_path.exists() else ""

        summary_path.write_text(
            f"# {display_title}\n\n_Last updated: {datetime.now().isoformat()}_\n\n{summary.strip()}\n",
            encoding="utf-8",
        )
        notes_path.write_text(notes.strip() + "\n", encoding="utf-8")
        links_path.write_text(json.dumps(links, indent=2), encoding="utf-8")
        self._update_index(normalized_topic, display_title, str(topic_dir))

        meaningful_change = (
            not was_update
            or previous_summary.strip() != summary_path.read_text(encoding="utf-8").strip()
            or previous_links.strip() != links_path.read_text(encoding="utf-8").strip()
        )
        return ResearchPackageResult(
            display_title=display_title,
            artifact_path=str(topic_dir),
            summary=summary,
            notes=notes,
            links=links,
            was_update=was_update,
            meaningful_change=meaningful_change,
        )

    def _update_index(self, normalized_topic: str, display_title: str, artifact_path: str) -> None:
        try:
            index = json.loads(self.index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            index = []
        updated = False
        for item in index:
            if item.get("normalized_topic") == normalized_topic:
                item["display_title"] = display_title
                item["artifact_path"] = artifact_path
                updated = True
                break
        if not updated:
            index.append(
                {
                    "normalized_topic": normalized_topic,
                    "display_title": display_title,
                    "artifact_path": artifact_path,
                }
            )
        self.index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
