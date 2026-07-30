from __future__ import annotations

import json
import logging
from typing import Any

from application.services.artifact_organizer_service import ArtifactOrganizer


class ArtifactMaintenanceService:
    """Continuously consolidates active artifacts that belong to the same ongoing thread."""

    VALIDATION_PROMPT = """You maintain a private artifact library.

Decide whether each offered pair belongs to the same ongoing thread and should be one artifact.
Same-thread examples include continued notes for the same course lecture, or repeated status reports for the
same project workstream. Mere topical similarity is insufficient. Keep adjacent access/authentication reports,
different lectures, different project workstreams, and general runtime debugging separate.

Return JSON only:
{"pairs":[{"left_id":"...","right_id":"...","same_thread":true,"confidence":0.0,"reason":"..."}]}
Use confidence >= 0.85 only when an automatic merge is safe.
"""

    MERGE_PROMPT = """Consolidate the supplied artifacts into one durable Markdown note.

Remove repeated information but preserve every distinct fact, task, date, source, decision, caveat, and useful
technical detail. Do not add claims absent from the inputs. The merged_content field must contain the body of
the consolidated artifact; do not include the outer title or standard summary headings.

Return JSON only with exactly these keys:
{"final_title":"...","short_summary":"...","detailed_summary":"...","merged_content":"...","reason":"..."}
"""

    def __init__(
        self,
        *,
        organizer: ArtifactOrganizer,
        llm_provider: Any,
        model: str,
        min_changes: int = 3,
        interval_hours: float = 24.0,
        max_clusters_per_run: int = 3,
        candidate_neighbors: int = 8,
        confidence_threshold: float = 0.85,
        archive_dir: str = "archived",
        logger: logging.Logger | None = None,
    ) -> None:
        self.organizer = organizer
        self.llm_provider = llm_provider
        self.model = str(model or "").strip()
        self.min_changes = max(1, int(min_changes))
        self.interval_hours = max(1.0, float(interval_hours))
        self.max_clusters_per_run = max(1, int(max_clusters_per_run))
        self.candidate_neighbors = max(1, int(candidate_neighbors))
        self.confidence_threshold = max(0.0, min(1.0, float(confidence_threshold)))
        self.archive_dir = str(archive_dir or "archived").strip() or "archived"
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def status(self) -> dict[str, Any]:
        return self.organizer.maintenance_status(
            min_changes=self.min_changes,
            interval_hours=self.interval_hours,
        )

    def is_due(self) -> bool:
        return bool(self.status().get("due"))

    async def run(self, *, trigger_kind: str = "idle") -> dict[str, Any]:
        if not self.model:
            raise RuntimeError("Artifact maintenance model is not configured.")
        run_id = self.organizer.start_maintenance_run(trigger_kind)
        metrics = {
            "scanned_count": 0,
            "candidate_pair_count": 0,
            "cluster_count": 0,
            "merged_cluster_count": 0,
            "archived_count": 0,
            "continuation_required": False,
            "error_text": None,
        }
        try:
            metrics["scanned_count"] = self.organizer.reconcile_and_reindex()
            artifacts = self.organizer.list_artifacts(status="active", limit=1000)
            pairs = self._candidate_pairs(artifacts)
            metrics["candidate_pair_count"] = len(pairs)
            approved_edges = await self._validate_pairs(pairs)
            clusters = self._clusters_from_edges(approved_edges)
            metrics["cluster_count"] = len(clusters)
            selected = clusters[: self.max_clusters_per_run]
            metrics["continuation_required"] = len(clusters) > len(selected)
            errors: list[str] = []
            for cluster in selected:
                members = set(cluster)
                evidence = [
                    edge for edge in approved_edges
                    if edge["left_id"] in members and edge["right_id"] in members
                ]
                confidence = min(
                    (float(edge["confidence"]) for edge in evidence),
                    default=self.confidence_threshold,
                )
                rationale = "; ".join(
                    dict.fromkeys(str(edge.get("reason") or "") for edge in evidence if edge.get("reason"))
                )
                try:
                    result = await self._merge_cluster(
                        run_id=run_id,
                        artifact_ids=cluster,
                        confidence=confidence,
                        rationale=rationale,
                    )
                except Exception as exc:
                    self.logger.exception("Artifact maintenance cluster failed: %s", cluster)
                    errors.append(f"{','.join(cluster)}: {exc}")
                    continue
                metrics["merged_cluster_count"] += 1
                metrics["archived_count"] += int(result.get("archived_count") or 0)
            if errors:
                metrics["error_text"] = " | ".join(errors)[:4000]
            self.organizer.finish_maintenance_run(run_id, status="completed", **metrics)
        except Exception as exc:
            metrics["error_text"] = str(exc)[:4000]
            self.organizer.finish_maintenance_run(run_id, status="failed", **metrics)
            raise
        return {"run_id": run_id, "status": "completed", **metrics}

    def _candidate_pairs(self, artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_id = {str(item["artifact_id"]): item for item in artifacts}
        pairs: dict[tuple[str, str], dict[str, Any]] = {}
        for artifact in artifacts:
            candidates = self.organizer.candidates_for(
                title=str(artifact.get("title") or ""),
                summary=str(artifact.get("short_summary") or ""),
                detailed_report=str(artifact.get("detailed_summary") or ""),
            )
            for candidate in candidates[: self.candidate_neighbors]:
                left_id = str(artifact["artifact_id"])
                right_id = str(candidate.artifact_id)
                if left_id == right_id or right_id not in by_id:
                    continue
                key = tuple(sorted((left_id, right_id)))
                if key in pairs:
                    continue
                left, right = by_id[key[0]], by_id[key[1]]
                pairs[key] = {
                    "left_id": key[0],
                    "right_id": key[1],
                    "left": self._artifact_summary(left),
                    "right": self._artifact_summary(right),
                    "retrieval_score": round(float(candidate.score), 5),
                    "match_source": candidate.match_source,
                }
        return list(pairs.values())

    async def _validate_pairs(self, pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        approved: list[dict[str, Any]] = []
        offered = {
            tuple(sorted((str(item["left_id"]), str(item["right_id"]))))
            for item in pairs
        }
        for offset in range(0, len(pairs), 24):
            batch = pairs[offset : offset + 24]
            raw = await self._json_completion(self.VALIDATION_PROMPT, {"candidate_pairs": batch})
            payload = self._safe_parse_json(raw)
            decisions = payload.get("pairs") if isinstance(payload, dict) else None
            if not isinstance(decisions, list):
                self.logger.warning("Artifact pair validator returned malformed JSON; batch left unchanged.")
                continue
            for decision in decisions:
                if not isinstance(decision, dict) or not bool(decision.get("same_thread")):
                    continue
                key = tuple(sorted((str(decision.get("left_id") or ""), str(decision.get("right_id") or ""))))
                try:
                    confidence = float(decision.get("confidence") or 0.0)
                except (TypeError, ValueError):
                    continue
                if key not in offered or confidence < self.confidence_threshold:
                    continue
                approved.append(
                    {
                        "left_id": key[0], "right_id": key[1], "confidence": confidence,
                        "reason": str(decision.get("reason") or ""),
                    }
                )
        return approved

    @staticmethod
    def _clusters_from_edges(edges: list[dict[str, Any]]) -> list[list[str]]:
        parent: dict[str, str] = {}

        def find(item: str) -> str:
            parent.setdefault(item, item)
            if parent[item] != item:
                parent[item] = find(parent[item])
            return parent[item]

        def union(left: str, right: str) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for edge in edges:
            union(str(edge["left_id"]), str(edge["right_id"]))
        groups: dict[str, list[str]] = {}
        for item in parent:
            groups.setdefault(find(item), []).append(item)
        clusters = [sorted(group) for group in groups.values() if len(group) > 1]
        clusters.sort(key=lambda group: (-len(group), group))
        return clusters

    async def _merge_cluster(
        self,
        *,
        run_id: str,
        artifact_ids: list[str],
        confidence: float,
        rationale: str,
    ) -> dict[str, Any]:
        artifacts = [self.organizer.get_artifact(item, include_content=True) for item in artifact_ids]
        if any(item is None for item in artifacts):
            raise ValueError("Artifact disappeared before maintenance merge.")
        active = [item for item in artifacts if item is not None]
        canonical = min(active, key=self._canonical_sort_key)
        payload = {
            "canonical_artifact_id": canonical["artifact_id"],
            "artifacts": [
                {
                    "artifact_id": item["artifact_id"],
                    "title": item["title"],
                    "short_summary": item["short_summary"],
                    "detailed_summary": item["detailed_summary"],
                    "source_count": item["source_count"],
                    "content": str(item.get("content") or "")[:20000],
                }
                for item in active
            ],
        }
        raw = await self._json_completion(self.MERGE_PROMPT, payload)
        decision = self._safe_parse_json(raw)
        required = ("final_title", "short_summary", "detailed_summary", "merged_content")
        if not isinstance(decision, dict) or any(not str(decision.get(key) or "").strip() for key in required):
            raise ValueError("Artifact merge model returned malformed or incomplete JSON.")
        return self.organizer.consolidate_cluster(
            run_id=run_id,
            canonical_artifact_id=str(canonical["artifact_id"]),
            artifact_ids=artifact_ids,
            final_title=str(decision["final_title"]).strip(),
            short_summary=str(decision["short_summary"]).strip(),
            detailed_summary=str(decision["detailed_summary"]).strip(),
            merged_content=str(decision["merged_content"]).strip(),
            confidence=confidence,
            rationale="; ".join(
                item for item in (rationale, str(decision.get("reason") or "")) if item
            ) or "Same ongoing artifact thread.",
            archive_dir=self.archive_dir,
        )

    async def _json_completion(self, system_prompt: str, payload: dict[str, Any]) -> str:
        completion = await self.llm_provider.chat_completion_stream(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
            ],
            tools=None,
            image="",
        )
        parts: list[str] = []
        async for chunk in completion:
            if not getattr(chunk, "choices", None):
                continue
            content = getattr(chunk.choices[0].delta, "content", None)
            if content:
                parts.append(content)
        return "".join(parts).strip()

    @staticmethod
    def _artifact_summary(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "artifact_id": item["artifact_id"],
            "title": item["title"],
            "short_summary": item.get("short_summary") or "",
            "detailed_summary": str(item.get("detailed_summary") or "")[:1500],
            "source_count": item.get("source_count") or 0,
        }

    @staticmethod
    def _canonical_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
        content = str(item.get("content") or "")
        standard_sections = sum(
            marker in content for marker in ("## Short Summary", "## Detailed Summary", "## Content")
        )
        return (
            -int(item.get("source_count") or 0),
            -standard_sections,
            -len(content),
            str(item.get("created_at") or ""),
            str(item.get("artifact_id") or ""),
        )

    @staticmethod
    def _safe_parse_json(text: str) -> dict[str, Any]:
        cleaned = str(text or "").strip()
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
        try:
            value = json.loads(cleaned)
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}
