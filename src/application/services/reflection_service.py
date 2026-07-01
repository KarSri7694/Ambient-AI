import hashlib
import json
import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from application.ports.LLMProvider import LLMProvider
from application.ports.memory_port import MemoryPort
from application.ports.task_queue_port import TaskQueuePort
from application.services.semantic_deduplication_service import SemanticDeduplicationService
from application.services.semantic_memory_service import SemanticMemoryService
from application.services.interaction_trace import interaction_trace


class ReflectionService:
    """Maintains durable user info, trims temporary memory, and derives proactive tasks."""

    CLEANUP_PROMPT = """You maintain USER_INFO.md as a concise durable profile.

Return markdown only.

Rules:
- Reorganize the content into a clean readable structure.
- Remove exact duplicates and obvious near-duplicates.
- Remove clutter, weak restatements, and repeated updates that no longer add value.
- Preserve durable facts, interests, concerns, commitments, preferences, and reminders.
- Do not invent new facts.
- Keep it compact and useful for future agent reasoning.
- If the input is already clean, return the cleaned equivalent without commentary.
"""

    MEMORY_CLEANUP_PROMPT = """You maintain MEMORY.md as concise temporary working memory.

Return markdown only.

Rules:
- Keep short-lived reminders, ongoing concerns, current plans, recent active topics, and open temporary context.
- Remove duplicates, clutter, stale filler, and repeated restatements.
- Do not convert temporary memory into broad long-term profile claims.
- Preserve useful unresolved or recent items.
- If the input is already clean, return the cleaned equivalent without commentary.
"""

    TASK_GENERATION_PROMPT = """You propose proactive tasks that an ambient assistant could do for the user without being explicitly asked.

Return JSON only:
{
  "tasks": [
    {
      "description": "short actionable task",
      "priority": "low|medium|high",
      "reason": "why this could help the user"
    }
  ]
}

Rules:
- Generate between 5 and 10 tasks unless the profile is too sparse, then return fewer.
- Use the cleaned user profile as the main stable source of truth.
- Use cleaned working memory as temporary supporting context.
- Avoid duplicates of currently pending tasks.
- Avoid repeating tasks that were already generated in previous reflection runs.
- Prefer concrete actions the agent can realistically perform later.
- Do not generate vague filler like "learn more about the user".
- Do not add any keys other than description, priority, and reason.
"""

    def __init__(
        self,
        *,
        memory: MemoryPort,
        task_queue: TaskQueuePort,
        llm_provider: LLMProvider,
        semantic_memory: SemanticMemoryService | None = None,
        semantic_dedupe_service: SemanticDeduplicationService | None = None,
        history_path: str,
        cadence_mode: str = "daily",
        interval_hours: int = 24,
        max_generated_tasks: int = 8,
        max_task_generation_runs: int = 3,
        logger: Optional[logging.Logger] = None,
    ):
        self.memory = memory
        self.task_queue = task_queue
        self.llm = llm_provider
        self.semantic_memory = semantic_memory
        self.semantic_dedupe = semantic_dedupe_service
        self.history_path = Path(history_path)
        self.cadence_mode = (cadence_mode or "daily").strip().lower()
        self.interval_hours = max(1, int(interval_hours))
        self.max_generated_tasks = int(max_generated_tasks)
        self.max_task_generation_runs = max(0, int(max_task_generation_runs))
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)

    def _build_system_prompt(self, prompt: str) -> str:
        now = datetime.now()
        preamble = (
            f"Current day of week: {now.strftime('%A')}\n"
            f"Current date: {now.strftime('%Y-%m-%d')}\n"
            f"Current time: {now.strftime('%H:%M:%S')}\n\n"
        )
        return preamble + prompt

    async def run_if_due(self, *, model: str, now: Optional[datetime] = None) -> dict:
        now = now or datetime.now()
        history = self._load_history()
        if self._task_generation_limit_reached(history):
            return {
                "ran": False,
                "reason": "task_generation_limit_reached",
                "last_run_at": history.get("last_run_at"),
                "queued_tasks": [],
                "generated_tasks": [],
                "task_generation_runs": self._task_generation_run_count(history),
                "max_task_generation_runs": self.max_task_generation_runs,
            }
        if not self._is_due(history, now):
            return {
                "ran": False,
                "reason": "not_due",
                "last_run_at": history.get("last_run_at"),
                "queued_tasks": [],
                "generated_tasks": [],
            }
        return await self.run(model=model, now=now, history=history)

    async def run(self, *, model: str, now: Optional[datetime] = None, history: Optional[dict] = None) -> dict:
        now = now or datetime.now()
        history = history or self._load_history()
        original_user_info = self.memory.get_user_info().strip()
        original_working_memory = self.memory.get_working_memory().strip()
        if not original_user_info and not original_working_memory:
            self._record_run(
                history=history,
                run_payload={
                    "generated_at": now.isoformat(),
                    "mode": self.cadence_mode,
                    "user_info_digest": "",
                    "working_memory_digest": "",
                    "generated_tasks": [],
                    "queued_tasks": [],
                    "skipped_tasks": [],
                    "reason": "empty_memory",
                },
                last_run_at=now.isoformat(),
            )
            return {
                "ran": True,
                "reason": "empty_memory",
                "cleaned_user_info_changed": False,
                "cleaned_working_memory_changed": False,
                "generated_tasks": [],
                "queued_tasks": [],
                "skipped_tasks": [],
            }

        cleaned_user_info = await self._cleanup_user_info(model=model, user_info=original_user_info)
        cleaned_user_info = cleaned_user_info.strip() or original_user_info
        cleaned_changed = cleaned_user_info != original_user_info
        if cleaned_changed:
            self.memory.save_user_info(cleaned_user_info + "\n")

        cleaned_working_memory = await self._cleanup_working_memory(
            model=model,
            working_memory=original_working_memory,
        )
        cleaned_working_memory = cleaned_working_memory.strip() or original_working_memory
        cleaned_memory_changed = cleaned_working_memory != original_working_memory
        if cleaned_memory_changed:
            self.memory.save_working_memory(cleaned_working_memory + "\n")

        pending_tasks = self.task_queue.get_pending_tasks()
        pending_descriptions = [task.description for task in pending_tasks]
        generated_tasks = await self._generate_tasks(
            model=model,
            cleaned_user_info=cleaned_user_info,
            cleaned_working_memory=cleaned_working_memory,
            pending_tasks=pending_descriptions,
            history=history,
        )

        queued_tasks: List[dict] = []
        skipped_tasks: List[dict] = []
        pending_normalized = {
            self._normalize_text(description)
            for description in pending_descriptions
            if self._normalize_text(description)
        }
        historical_normalized = self._historical_task_descriptions(history)
        seen_this_run: set[str] = set()
        for task in generated_tasks[: self.max_generated_tasks]:
            description = task["description"]
            normalized = self._normalize_text(description)
            if not normalized:
                continue
            if normalized in pending_normalized:
                skipped_tasks.append(
                    {
                        **task,
                        "skip_reason": "pending_duplicate",
                    }
                )
                continue
            if normalized in historical_normalized:
                skipped_tasks.append(
                    {
                        **task,
                        "skip_reason": "historical_duplicate",
                    }
                )
                continue
            if normalized in seen_this_run:
                skipped_tasks.append(
                    {
                        **task,
                        "skip_reason": "run_duplicate",
                    }
                )
                continue
            dedupe = await self._evaluate_candidate(
                model=model,
                text=description,
                metadata={
                    "task_kind": "reflection_proactive_task",
                    "reason": task.get("reason", ""),
                    "priority": task.get("priority", "medium"),
                    "generated_at": now.isoformat(),
                },
            )
            if dedupe["decision"] != "create_new":
                skipped_tasks.append(
                    {
                        **task,
                        "skip_reason": "semantic_duplicate",
                        "duplicate_of_item_id": dedupe.get("duplicate_of_item_id"),
                    }
                )
                self._record_duplicate_skip(
                    text=description,
                    dedupe=dedupe,
                    metadata={
                        "task_kind": "reflection_proactive_task",
                        "reason": task.get("reason", ""),
                        "priority": task.get("priority", "medium"),
                        "generated_at": now.isoformat(),
                    },
                )
                continue

            metadata = {
                "task_kind": "reflection_proactive_task",
                "source": "reflection_service",
                "reason": task.get("reason", ""),
                "generated_at": now.isoformat(),
            }
            metadata["dedupe_item_id"] = uuid.uuid4().hex
            self.task_queue.add_task(
                description=description,
                priority=task.get("priority", "medium"),
                metadata=metadata,
            )
            self._record_created_item(
                text=description,
                metadata=metadata,
                dedupe_item_id=metadata["dedupe_item_id"],
            )
            seen_this_run.add(normalized)
            queued_tasks.append(task)

        run_payload = {
            "generated_at": now.isoformat(),
            "mode": self.cadence_mode,
            "user_info_digest": self._digest(cleaned_user_info),
            "working_memory_digest": self._digest(cleaned_working_memory),
            "generated_tasks": generated_tasks[: self.max_generated_tasks],
            "queued_tasks": queued_tasks,
            "skipped_tasks": skipped_tasks,
        }
        self._record_run(history=history, run_payload=run_payload, last_run_at=now.isoformat())
        return {
            "ran": True,
            "reason": "completed",
            "cleaned_user_info_changed": cleaned_changed,
            "cleaned_working_memory_changed": cleaned_memory_changed,
            "generated_tasks": generated_tasks[: self.max_generated_tasks],
            "queued_tasks": queued_tasks,
            "skipped_tasks": skipped_tasks,
        }

    async def _cleanup_user_info(self, *, model: str, user_info: str) -> str:
        with interaction_trace("reflection.cleanup_user_info"):
            completion = await self.llm.chat_completion_stream(
                model=model,
                messages=[
                    {"role": "system", "content": self._build_system_prompt(self.CLEANUP_PROMPT)},
                    {"role": "user", "content": user_info},
                ],
                tools=None,
            )
        return (await self._consume_stream_text(completion)).strip()

    async def _generate_tasks(
        self,
        *,
        model: str,
        cleaned_user_info: str,
        cleaned_working_memory: str,
        pending_tasks: List[str],
        history: dict,
    ) -> List[dict]:
        payload = {
            "cleaned_user_info": cleaned_user_info,
            "cleaned_working_memory": cleaned_working_memory,
            "pending_tasks": pending_tasks,
            "previously_generated_tasks": self._recent_generated_tasks(history),
            "max_tasks": self.max_generated_tasks,
            "semantic_context": self._semantic_context(cleaned_user_info, cleaned_working_memory),
        }
        with interaction_trace("reflection.generation"):
            completion = await self.llm.chat_completion_stream(
                model=model,
                messages=[
                    {"role": "system", "content": self._build_system_prompt(self.TASK_GENERATION_PROMPT)},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
                ],
                tools=None,
            )
        parsed = self._parse_json_object(await self._consume_stream_text(completion))
        return self._normalize_tasks(parsed.get("tasks"))

    async def _cleanup_working_memory(self, *, model: str, working_memory: str) -> str:
        if not working_memory.strip():
            return ""
        with interaction_trace("reflection.cleanup_memory"):
            completion = await self.llm.chat_completion_stream(
                model=model,
                messages=[
                    {"role": "system", "content": self._build_system_prompt(self.MEMORY_CLEANUP_PROMPT)},
                    {"role": "user", "content": working_memory},
                ],
                tools=None,
            )
        return (await self._consume_stream_text(completion)).strip()

    def _is_due(self, history: dict, now: datetime) -> bool:
        mode = self.cadence_mode
        if mode == "every_idle_cycle":
            return True

        last_run_at = history.get("last_run_at")
        if not last_run_at:
            return True
        try:
            last_dt = datetime.fromisoformat(last_run_at)
        except ValueError:
            return True

        if mode == "interval_hours":
            return now - last_dt >= timedelta(hours=self.interval_hours)

        return last_dt.date() < now.date()

    def _load_history(self) -> dict:
        if not self.history_path.exists():
            return {"last_run_at": None, "runs": []}
        try:
            payload = json.loads(self.history_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"last_run_at": None, "runs": []}
        if not isinstance(payload, dict):
            return {"last_run_at": None, "runs": []}
        runs = payload.get("runs")
        if not isinstance(runs, list):
            runs = []
        return {
            "last_run_at": payload.get("last_run_at"),
            "runs": runs,
        }

    def _record_run(self, *, history: dict, run_payload: dict, last_run_at: str) -> None:
        runs = list(history.get("runs", []))
        runs.append(run_payload)
        payload = {
            "last_run_at": last_run_at,
            "runs": runs[-50:],
        }
        self.history_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _recent_generated_tasks(self, history: dict) -> List[dict]:
        results: List[dict] = []
        for run in history.get("runs", [])[-10:]:
            generated_at = run.get("generated_at")
            for task in run.get("generated_tasks", []) or []:
                if not isinstance(task, dict):
                    continue
                description = self._clean_text(task.get("description"))
                if not description:
                    continue
                results.append(
                    {
                        "description": description,
                        "priority": self._clean_text(task.get("priority")) or "medium",
                        "reason": self._clean_text(task.get("reason")),
                        "generated_at": generated_at,
                    }
                )
        return results

    def _task_generation_run_count(self, history: dict) -> int:
        count = 0
        for run in history.get("runs", []):
            queued_tasks = run.get("queued_tasks") or []
            if isinstance(queued_tasks, list) and queued_tasks:
                count += 1
        return count

    def _task_generation_limit_reached(self, history: dict) -> bool:
        if self.cadence_mode != "every_idle_cycle":
            return False
        if self.max_task_generation_runs <= 0:
            return False
        return self._task_generation_run_count(history) >= self.max_task_generation_runs

    def _historical_task_descriptions(self, history: dict) -> set[str]:
        values: set[str] = set()
        for task in self._recent_generated_tasks(history):
            normalized = self._normalize_text(task.get("description"))
            if normalized:
                values.add(normalized)
        return values

    def _normalize_tasks(self, value: Any) -> List[dict]:
        if not isinstance(value, list):
            return []
        results: List[dict] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            description = self._clean_text(item.get("description"))
            if not description:
                continue
            priority = (self._clean_text(item.get("priority")) or "medium").lower()
            if priority not in {"low", "medium", "high"}:
                priority = "medium"
            reason = self._clean_text(item.get("reason"))
            results.append(
                {
                    "description": description,
                    "priority": priority,
                    "reason": reason,
                }
            )
        return results

    def _semantic_context(self, cleaned_user_info: str, cleaned_working_memory: str) -> List[dict]:
        if self.semantic_memory is None:
            return []
        query = " ".join(
            part
            for part in [
                cleaned_user_info,
                cleaned_working_memory,
                "proactive tasks based on user interests commitments reminders and current concerns",
            ]
            if part
        ).strip()
        if not query:
            return []
        results = self.semantic_memory.retrieve(query=query, limit=10, rerank_limit=6)
        return self.semantic_memory.format_context(results)

    def _parse_json_object(self, response_text: str) -> dict:
        text = response_text.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            self.logger.warning("ReflectionService task response was not valid JSON.")
            return {}

    async def _consume_stream_text(self, completion) -> str:
        parts: List[str] = []
        async for chunk in completion:
            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                parts.append(delta.content)
        return "".join(parts)

    def _digest(self, value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _clean_text(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    def _normalize_text(self, value: Any) -> str:
        text = self._clean_text(value).lower()
        return " ".join(text.split())

    async def _evaluate_candidate(self, *, model: str, text: str, metadata: Dict[str, Any]) -> dict:
        if self.semantic_dedupe is None:
            return {"decision": "create_new", "duplicate_of_item_id": None, "reason": "service_unavailable"}
        return await self.semantic_dedupe.evaluate_candidate(
            entity_kind="reflection_task",
            source_kind="reflection_service",
            text=text,
            metadata=metadata,
            model=model,
        )

    def _record_created_item(
        self,
        *,
        text: str,
        metadata: Dict[str, Any],
        dedupe_item_id: str | None = None,
    ) -> None:
        if self.semantic_dedupe is None:
            return
        self.semantic_dedupe.record_created(
            entity_kind="reflection_task",
            source_kind="reflection_service",
            text=text,
            metadata=metadata,
            dedupe_item_id=dedupe_item_id,
        )

    def _record_duplicate_skip(self, *, text: str, dedupe: dict, metadata: Dict[str, Any]) -> None:
        if self.semantic_dedupe is None:
            return
        payload = dict(metadata)
        payload.update(
            {
                "dedupe_reason": dedupe.get("reason"),
                "dedupe_confidence": dedupe.get("confidence"),
            }
        )
        self.semantic_dedupe.record_skipped_duplicate(
            entity_kind="reflection_task",
            source_kind="reflection_service",
            text=text,
            duplicate_of_item_id=dedupe.get("duplicate_of_item_id"),
            metadata=payload,
        )
