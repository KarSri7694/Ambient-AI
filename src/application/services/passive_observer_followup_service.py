import json
import logging
import re
import hashlib
import uuid
from datetime import datetime
from typing import Any, List

from application.ports.LLMProvider import LLMProvider
from application.ports.memory_port import MemoryPort
from application.ports.task_queue_port import TaskQueuePort
from application.services.semantic_deduplication_service import SemanticDeduplicationService
from application.services.semantic_memory_service import SemanticMemoryService
from application.services.interaction_trace import interaction_trace
from core.models import VisualObservation
from utils.todoist_helper import TodoistHelper


class PassiveObserverFollowupService:
    """Turn unsent inferred visual activities into immediate or queued follow-up work."""

    FOLLOWUP_TTL_SECONDS = 2 * 60 * 60
    UNSENT_OBSERVATION_LIMIT = 20

    UNIQUE_ACTIVITIES_PROMPT = """You receive a list of inferred user activities extracted from passive visual observations.

Return JSON only:
{
  "unique_activities": ["activity 1", "activity 2"]
}

Rules:
- Return only semantically unique activities.
- Merge duplicate or near-duplicate activities into one normalized activity string.
- Preserve the concrete user intent.
- Do not add commentary or any keys other than unique_activities.
"""

    USEFUL_ACTIVITIES_PROMPT = """You receive a list of unique inferred user activities.

Return JSON only:
{
  "useful_activities": ["activity 1", "activity 2"]
}

Rules:
- Keep only activities that would be useful for an ambient agent to act on later or now.
- Keep the activities that might need to be reminded to the user or that the user might forget to do maybe related to work, education or personal life.
- Remove trivial UI actions and low-value interaction such as scrolling, typing, clicking, opening tabs, navigating, dismissing popups, or similar mechanical actions.
- Keep only activities with meaningful intent.
- Do not add any keys other than useful_activities.
"""

    ACTIVITY_DECISION_PROMPT = """You receive an inferred user activity produced by a previous agent call together with full visual observation context. It is not a literal user command. You must infer what useful ambient action, if any, an AI agent should take from it.

Return JSON only:
{
  "action": "nothing|queue_task|do_now",
  "task": "concrete action for the AI agent to do or queue",
  "memory_updates": ["optional temporary memory or reminder notes"],
  "user_info_updates": ["optional user facts or notes worth saving"]
}

Rules:
- If the activity is coding or programming related, or related to git version control, just output "nothing" for action and do not queue or do_now any tasks.
- Prefer extracting the most directly useful action from the context, not merely restating or summarizing what the user was viewing.
- If the context suggests something the user may need to remember, track, revisit, prepare for, or not forget, strongly prefer a reminder/todo style action.
- Interpret the activity text as a description of what the user was doing, discussing, or thinking about.
- Use the full observation context such as app, page, summary, detailed description, and capture time to understand what the user likely needs.
- Infer the underlying useful action, reminder, search, or saved note. Do not just repeat the activity text when a more concrete action is obvious.
- Prefer the most actionable underlying need over generic summarization.
- Do not default to "summarize chat" or "create a note" 
- If the context contains uncertainty, speculation, comparison, investigation, or open questions about causes, prices, news, products, events, or decisions, consider whether a concrete research task is more useful than a summary.
- If multiple possible actions exist, choose the single one that is most immediately useful and actionable for the user.
- Use do_now when the agent should perform something immediately during idle time that may be of immediate use to the user.
- Good do_now examples: setting a reminder, creating a quick todo, doing a quick search, checking a score, checking a schedule, checking a price, answering a simple question, or briefly explaining something currently on screen.
- Use queue_task when the activity should be queued for an AI agent that will perform that task later on.
- Good queue_task examples: research, multi-step comparison, drafting, writing follow-up, document work, code work, or any longer task that should persist beyond this moment.
- If the activity is actionable now and the result would likely help the user immediately, prefer do_now.
- If the activity represents durable work that should be carried out later by an AI agent, prefer queue_task.
- Use nothing for irrelevant, unsafe, ambiguous, purely mechanical, or low-value activities.
- When action is do_now or queue_task, task must be a short concrete instruction for the AI agent.
- When action is nothing, task may be empty.
- Use memory_updates for short-term facts, active concerns, upcoming items, temporary reminders, or recent context that may matter for a while but should not become stable profile data yet.
- Use user_info_updates only for durable user facts, repeated interests, habits, stable preferences, or long-term concerns worth saving in the user's profile.
- Do not add commentary or any keys other than action, task, memory_updates, and user_info_updates.

Examples:
Input activity: "compare TV prices across Amazon and Flipkart"
Output:
{"action":"queue_task","task":"compare TV prices across Amazon and Flipkart and summarize the best options","memory_updates":[],"user_info_updates":[]}

Input activity: "set a reminder to call mom tonight"
Output:
{"action":"do_now","task":"set a reminder to call mom tonight","memory_updates":["User wants to remember calling mom tonight."],"user_info_updates":[]}

Input activity: "check today's India cricket score"
Output:
{"action":"do_now","task":"check today's India cricket score","memory_updates":[],"user_info_updates":[]}

Input activity: "write a follow-up email draft to the recruiter"
Output:
{"action":"queue_task","task":"draft a follow-up email to the recruiter","memory_updates":[],"user_info_updates":[]}

Input activity: "scrolling through Instagram feed"
Output:
{"action":"nothing","task":"","memory_updates":[],"user_info_updates":[]}

Input activity: "typing in a search box"
Output:
{"action":"nothing","task":"","memory_updates":[],"user_info_updates":[]}

Input activity: "Reviewing or replying to messages in a conversation that mentions an upcoming event the user may need to remember"
Output:
{"action":"do_now","task":"set a reminder about the upcoming event","memory_updates":["User may need to remember an upcoming event mentioned in conversation."],"user_info_updates":[]}

Input activity: "Reviewing a discussion that speculates about why a product price is increasing"
Output:
{"action":"queue_task","task":"research the possible reasons behind the product price increase","memory_updates":["User is currently concerned about a recent product price increase."],"user_info_updates":["User tracks pricing changes before making decisions."]}

"""

    def __init__(
        self,
        *,
        memory: MemoryPort,
        task_queue: TaskQueuePort,
        llm_provider: LLMProvider,
        semantic_memory: SemanticMemoryService | None = None,
        semantic_dedupe_service: SemanticDeduplicationService | None = None,
        activity_ledger: Any = None,
        reminder_helper: Any = None,
        logger: logging.Logger | None = None,
    ):
        self.memory = memory
        self.task_queue = task_queue
        self.llm = llm_provider
        self.semantic_memory = semantic_memory
        self.semantic_dedupe = semantic_dedupe_service
        self.activity_ledger = activity_ledger
        self.reminder_helper = reminder_helper
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def _build_system_prompt(self, prompt: str) -> str:
        now = datetime.now()
        preamble = (
            f"Current day of week: {now.strftime('%A')}\n"
            f"Current date: {now.strftime('%Y-%m-%d')}\n"
            f"Current time: {now.strftime('%H:%M:%S')}\n\n"
        )
        return preamble + prompt

    async def maybe_queue_followup(self, *, model: str) -> dict:
        observations = self.memory.get_recent_unsent_visual_observations(limit=self.UNSENT_OBSERVATION_LIMIT)
        return await self.process_observations(
            observations=observations,
            model=model,
            mark_sent=True,
            apply_memory_updates=True,
        )

    async def process_observations(
        self,
        *,
        observations: List[VisualObservation],
        model: str,
        mark_sent: bool = False,
        apply_memory_updates: bool = True,
    ) -> dict:
        activity_rows = self._extract_activity_rows(observations)
        if not activity_rows:
            return self._empty_result("no unsent inferred activities")

        pending_tasks = self.task_queue.get_pending_tasks()
        pending_descriptions = [str(task.description) for task in pending_tasks[:20]]
        direct_reminders = await self._create_direct_reminders(activity_rows, model=model)
        if mark_sent:
            sent_at = datetime.now().isoformat()
            self.memory.mark_visual_observations_followup_sent(
                [row["observation_id"] for row in activity_rows],
                sent_at=sent_at,
            )

        with interaction_trace("passive_observer_followup"):
            unique_activities = await self._request_unique_activities(
                model=model,
                activities=[row["activity"] for row in activity_rows],
            )
            useful_activities = await self._request_useful_activities(
                model=model,
                activities=unique_activities,
            )
            grouped_rows = self._group_rows_by_activity(activity_rows)
            useful_activities = self._ensure_reminder_candidates(
                useful_activities=useful_activities,
                grouped_rows=grouped_rows,
            )
            queued_activities: List[str] = []
            do_now_activities: List[str] = []
            ignored_activities: List[str] = []
            memory_updates: List[str] = []
            user_info_updates: List[str] = []

            for activity in useful_activities:
                matching_rows = grouped_rows.get(self._normalize_activity(activity), [])
                decision = await self._classify_activity(
                    model=model,
                    activity=activity,
                    rows=matching_rows,
                )
                action = decision["action"]
                task = decision["task"] or activity
                memory_updates.extend(decision["memory_updates"])
                user_info_updates.extend(decision["user_info_updates"])
                if action == "queue_task":
                    dedupe = await self._evaluate_candidate(
                        model=model,
                        entity_kind="internal_task",
                        source_kind="passive_observer_followup",
                        text=task,
                        metadata=self._build_dedupe_metadata(activity=activity, task=task, rows=matching_rows),
                    )
                    if dedupe["decision"] != "create_new":
                        self._record_duplicate_skip(
                            entity_kind="internal_task",
                            source_kind="passive_observer_followup",
                            text=task,
                            dedupe=dedupe,
                            metadata=self._build_dedupe_metadata(activity=activity, task=task, rows=matching_rows),
                        )
                        ignored_activities.append(task)
                        continue
                    task_metadata = self._build_task_metadata(activity=activity, task=task, rows=matching_rows)
                    task_metadata["dedupe_item_id"] = uuid.uuid4().hex
                    self.task_queue.add_task(
                        self._build_task_description(activity=task, rows=matching_rows),
                        priority="low",
                        metadata=task_metadata,
                    )
                    record = self._record_created_item(
                        entity_kind="internal_task",
                        source_kind="passive_observer_followup",
                        text=task,
                        metadata=task_metadata,
                        dedupe_item_id=task_metadata["dedupe_item_id"],
                    )
                    pending_descriptions.append(task)
                    queued_activities.append(task)
                    self._record_activity_ledger(activity=task, rows=matching_rows)
                elif action == "do_now":
                    dedupe = await self._evaluate_candidate(
                        model=model,
                        entity_kind="do_now_action",
                        source_kind="passive_observer_followup",
                        text=task,
                        metadata=self._build_dedupe_metadata(activity=activity, task=task, rows=matching_rows),
                    )
                    if dedupe["decision"] != "create_new":
                        self._record_duplicate_skip(
                            entity_kind="do_now_action",
                            source_kind="passive_observer_followup",
                            text=task,
                            dedupe=dedupe,
                            metadata=self._build_dedupe_metadata(activity=activity, task=task, rows=matching_rows),
                        )
                        ignored_activities.append(task)
                        continue
                    self._record_created_item(
                        entity_kind="do_now_action",
                        source_kind="passive_observer_followup",
                        text=task,
                        metadata=self._build_dedupe_metadata(activity=activity, task=task, rows=matching_rows),
                    )
                    do_now_activities.append(task)
                else:
                    ignored_activities.append(activity)

            if apply_memory_updates:
                saved_memory_updates = self._apply_memory_updates(memory_updates)
                saved_user_info_updates = self._apply_user_info_updates(user_info_updates)
            else:
                saved_memory_updates = self._dedupe_preserve_order(memory_updates)
                saved_user_info_updates = self._dedupe_preserve_order(user_info_updates)

        useful_set = {self._normalize_activity(item) for item in useful_activities}
        for activity in unique_activities:
            if self._normalize_activity(activity) not in useful_set:
                ignored_activities.append(activity)

        return {
            "processed_observation_ids": [row["observation_id"] for row in activity_rows],
            "unique_activities": unique_activities,
            "useful_activities": useful_activities,
            "queued_activities": queued_activities,
            "do_now_activities": do_now_activities,
            "ignored_activities": ignored_activities,
            "direct_reminders": direct_reminders,
            "memory_updates": saved_memory_updates,
            "user_info_updates": saved_user_info_updates,
        }

    def _empty_result(self, reason: str) -> dict:
        return {
            "processed_observation_ids": [],
            "unique_activities": [],
            "useful_activities": [],
            "queued_activities": [],
            "do_now_activities": [],
            "ignored_activities": [],
            "direct_reminders": [],
            "memory_updates": [],
            "user_info_updates": [],
            "reason": reason,
        }

    def _extract_activity_rows(self, observations: List[VisualObservation]) -> List[dict]:
        rows: List[dict] = []
        for observation in observations:
            reminder_hint = self._extract_reminder_hint(observation.raw_payload_json)
            reminder_text = self._reminder_message(reminder_hint["reminder_context"])
            activity = self._clean_text(observation.inferred_user_activity) or reminder_text
            if not activity:
                continue
            rows.append(
                {
                    "observation_id": observation.observation_id,
                    "created_at": observation.created_at,
                    "activity": activity,
                    "app_name": self._clean_text(observation.app_name),
                    "page_hint": self._clean_text(observation.page_hint),
                    "summary": self._clean_text(observation.summary),
                    "detailed_description": self._clean_text(observation.detailed_description),
                    "maybe_require_a_reminder": reminder_hint["maybe_require_a_reminder"],
                    "reminder_context": reminder_hint["reminder_context"],
                    "observation": observation,
                }
            )
        return rows

    async def _create_direct_reminders(self, rows: List[dict], *, model: str) -> List[str]:
        helper = self.reminder_helper
        if helper is None:
            return []
        try:
            if not helper.is_enabled():
                return []
        except Exception:
            return []

        existing_tasks: List[str] = []
        try:
            existing_tasks = [str(item.get("content", "")) for item in helper.get_tasks()[:100]]
        except Exception as exc:
            self.logger.warning("Failed to fetch Todoist tasks before creating reminders: %s", exc)

        created: List[str] = []
        seen: set[str] = set()
        for row in rows:
            if not row.get("maybe_require_a_reminder"):
                continue
            reminder_context = self._normalize_reminder_context(row.get("reminder_context"))
            reminder_text = self._reminder_message(reminder_context)
            due_date_text = self._clean_text(reminder_context.get("due_date"))
            normalized = self._normalize_activity(reminder_text)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            dedupe_metadata = {
                "task_kind": "passive_observer_direct_reminder",
                "source_observation_ids": [row["observation_id"]],
                "created_at": row["created_at"],
                "app_name": row["app_name"],
                "page_hint": row["page_hint"],
                "reminder_due_date": due_date_text,
            }
            dedupe = await self._evaluate_candidate(
                model=model,
                entity_kind="todoist_reminder",
                source_kind="passive_observer_direct_reminder",
                text=reminder_text,
                metadata=dedupe_metadata,
            )
            if dedupe["decision"] != "create_new":
                self._record_duplicate_skip(
                    entity_kind="todoist_reminder",
                    source_kind="passive_observer_direct_reminder",
                    text=reminder_text,
                    dedupe=dedupe,
                    metadata=dedupe_metadata,
                )
                continue
            due_datetime = self._parse_due_datetime(due_date_text, reminder_text=reminder_text)
            task = helper.add_task(reminder_text, due_datetime=due_datetime)
            if task is not None:
                provider_ref = self._clean_text(getattr(task, "id", None) if not isinstance(task, dict) else task.get("id")) or None
                self._record_created_item(
                    entity_kind="todoist_reminder",
                    source_kind="passive_observer_direct_reminder",
                    text=reminder_text,
                    metadata=dedupe_metadata,
                    provider_ref=provider_ref,
                )
                created.append(reminder_text)
                existing_tasks.append(reminder_text)
                self.logger.info("Created direct Todoist reminder from passive observation: %s", reminder_text)
        return created

    def _group_rows_by_activity(self, rows: List[dict]) -> dict[str, List[dict]]:
        grouped: dict[str, List[dict]] = {}
        for row in rows:
            key = self._normalize_activity(row["activity"])
            grouped.setdefault(key, []).append(row)
        return grouped

    async def _request_unique_activities(self, *, model: str, activities: List[str]) -> List[str]:
        if not activities:
            return []
        payload = {"activities": activities}
        completion = await self.llm.chat_completion_stream(
            model=model,
            messages=[
                {"role": "system", "content": self._build_system_prompt(self.UNIQUE_ACTIVITIES_PROMPT)},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
            ],
            tools=None,
        )
        parsed = self._parse_json_object(await self._consume_stream_text(completion))
        return self._dedupe_preserve_order(self._list_text(parsed.get("unique_activities")))

    async def _request_useful_activities(self, *, model: str, activities: List[str]) -> List[str]:
        if not activities:
            return []
        payload = {"activities": activities}
        completion = await self.llm.chat_completion_stream(
            model=model,
            messages=[
                {"role": "system", "content": self._build_system_prompt(self.USEFUL_ACTIVITIES_PROMPT)},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
            ],
            tools=None,
        )
        parsed = self._parse_json_object(await self._consume_stream_text(completion))
        return self._dedupe_preserve_order(self._list_text(parsed.get("useful_activities")))

    async def _classify_activity(self, *, model: str, activity: str, rows: List[dict]) -> dict:
        payload = {
            "activity": activity,
            "observation_context": [
                {
                    "observation_id": row["observation_id"],
                    "created_at": row["created_at"],
                    "app_name": row["app_name"],
                    "page_hint": row["page_hint"],
                    "summary": row["summary"],
                    "detailed_description": row["detailed_description"],
                    "maybe_require_a_reminder": row["maybe_require_a_reminder"],
                    "reminder_context": row["reminder_context"],
                }
                for row in rows
            ],
            "semantic_context": self._semantic_context(activity=activity, rows=rows),
        }
        completion = await self.llm.chat_completion_stream(
            model=model,
            messages=[
                {"role": "system", "content": self._build_system_prompt(self.ACTIVITY_DECISION_PROMPT)},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
            ],
            tools=None,
        )
        parsed = self._parse_json_object(await self._consume_stream_text(completion))
        decision = self._clean_text(parsed.get("action")).lower()
        task = self._clean_text(parsed.get("task"))
        memory_updates = self._list_text(parsed.get("memory_updates"))
        user_info_updates = self._list_text(parsed.get("user_info_updates"))
        if decision not in {"nothing", "queue_task", "do_now"}:
            decision = "nothing"
        return {
            "action": decision,
            "task": task,
            "memory_updates": memory_updates,
            "user_info_updates": user_info_updates,
        }

    def _ensure_reminder_candidates(self, *, useful_activities: List[str], grouped_rows: dict[str, List[dict]]) -> List[str]:
        results = list(useful_activities)
        seen = {self._normalize_activity(item) for item in results}
        for rows in grouped_rows.values():
            if not rows:
                continue
            if not any(
                row["maybe_require_a_reminder"] and self._reminder_message(row["reminder_context"])
                for row in rows
            ):
                continue
            activity = rows[0]["activity"]
            key = self._normalize_activity(activity)
            if key and key not in seen:
                results.append(activity)
                seen.add(key)
        return results

    def _build_task_description(self, *, activity: str, rows: List[dict]) -> str:
        lines = [f"[Passive observer] {activity}", activity]
        if not rows:
            return "\n".join(lines)
        lines.extend(["", "Passive observation context:"])
        for row in rows:
            lines.append(f"- Observation ID: {row['observation_id']}")
            lines.append(f"- Captured at: {row['created_at']}")
        return "\n".join(lines)

    def _build_task_metadata(self, *, activity: str, task: str, rows: List[dict]) -> dict:
        return {
            "task_kind": "passive_observer_followup",
            "activity": activity,
            "task": task,
            "normalized_activity": self._normalize_activity(task),
            "source": "inferred_user_activity",
            "source_observation_ids": [row["observation_id"] for row in rows],
            "ttl_seconds": self.FOLLOWUP_TTL_SECONDS,
        }

    def _record_activity_ledger(self, *, activity: str, rows: List[dict]) -> None:
        if self.activity_ledger is None:
            return
        run = self.activity_ledger.queue_run(
            source_kind="passive_observer_followup",
            trigger_kind="ambient_inference",
            title=activity,
            summary=activity,
            metadata={"source_observation_ids": [row["observation_id"] for row in rows]},
            tags=["passive_observer", "followup"],
        )
        for row in rows:
            self.activity_ledger.link_entity(
                run_id=run.run_id,
                entity_type="visual_observation",
                entity_id=row["observation_id"],
                relation="derived_from",
            )

    def _build_dedupe_metadata(self, *, activity: str, task: str, rows: List[dict]) -> dict:
        return {
            "activity": activity,
            "task": task,
            "source_observation_ids": [row["observation_id"] for row in rows],
            "created_at": rows[0]["created_at"] if rows else "",
            "app_names": [row["app_name"] for row in rows if row["app_name"]],
            "page_hints": [row["page_hint"] for row in rows if row["page_hint"]],
            "reminder_contexts": [
                row["reminder_context"]
                for row in rows
                if self._reminder_message(row["reminder_context"]) or self._clean_text(row["reminder_context"].get("due_date"))
            ],
            "reminder_messages": [self._reminder_message(row["reminder_context"]) for row in rows if self._reminder_message(row["reminder_context"])],
            "reminder_due_dates": [
                self._clean_text(row["reminder_context"].get("due_date"))
                for row in rows
                if isinstance(row.get("reminder_context"), dict) and self._clean_text(row["reminder_context"].get("due_date"))
            ],
        }

    async def _evaluate_candidate(
        self,
        *,
        model: str,
        entity_kind: str,
        source_kind: str,
        text: str,
        metadata: dict,
    ) -> dict:
        if self.semantic_dedupe is None:
            return {"decision": "create_new", "duplicate_of_item_id": None, "reason": "service_unavailable"}
        return await self.semantic_dedupe.evaluate_candidate(
            entity_kind=entity_kind,
            source_kind=source_kind,
            text=text,
            metadata=metadata,
            model=model,
        )

    def _record_created_item(
        self,
        *,
        entity_kind: str,
        source_kind: str,
        text: str,
        metadata: dict,
        provider_ref: str | None = None,
        dedupe_item_id: str | None = None,
    ):
        if self.semantic_dedupe is None:
            return None
        return self.semantic_dedupe.record_created(
            entity_kind=entity_kind,
            source_kind=source_kind,
            text=text,
            metadata=metadata,
            provider_ref=provider_ref,
            dedupe_item_id=dedupe_item_id,
        )

    def _record_duplicate_skip(
        self,
        *,
        entity_kind: str,
        source_kind: str,
        text: str,
        dedupe: dict,
        metadata: dict,
    ) -> None:
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
            entity_kind=entity_kind,
            source_kind=source_kind,
            text=text,
            duplicate_of_item_id=dedupe.get("duplicate_of_item_id"),
            metadata=payload,
        )

    def _apply_user_info_updates(self, updates: List[str]) -> List[str]:
        deduped = self._dedupe_preserve_order(updates)
        if not deduped:
            return []
        existing = self.memory.get_user_info().strip()
        existing_lines = {
            self._normalize_activity(line)
            for line in existing.splitlines()
            if self._normalize_activity(line)
        }
        new_updates = [item for item in deduped if self._normalize_activity(item) not in existing_lines]
        if not new_updates:
            return []
        lines: List[str] = []
        if existing:
            lines.append(existing)
            if not existing.endswith("\n"):
                lines.append("")
        lines.extend(f"- {item}" for item in new_updates)
        content = "\n".join(lines).strip() + "\n"
        self.memory.save_user_info(content)
        for item in new_updates:
            self._index_note(note=item, source_type="user_info_note", bucket="user_info")
        return new_updates

    def _apply_memory_updates(self, updates: List[str]) -> List[str]:
        deduped = self._dedupe_preserve_order(updates)
        if not deduped:
            return []
        existing = self.memory.get_working_memory().strip()
        existing_lines = {
            self._normalize_activity(line)
            for line in existing.splitlines()
            if self._normalize_activity(line)
        }
        new_updates = [item for item in deduped if self._normalize_activity(item) not in existing_lines]
        if not new_updates:
            return []
        lines: List[str] = []
        if existing:
            lines.append(existing)
            if not existing.endswith("\n"):
                lines.append("")
        lines.extend(f"- {item}" for item in new_updates)
        self.memory.save_working_memory("\n".join(lines).strip() + "\n")
        for item in new_updates:
            self._index_note(note=item, source_type="working_memory_note", bucket="memory")
        return new_updates

    def _index_note(self, *, note: str, source_type: str, bucket: str) -> None:
        clean_note = self._clean_text(note)
        if not clean_note:
            return
        stable_id = hashlib.sha1(f"{source_type}:{clean_note.lower()}".encode("utf-8")).hexdigest()
        self.memory.upsert_semantic_chunk(
            source_type=source_type,
            source_id=stable_id,
            source_ref=source_type,
            content=clean_note,
            metadata_json=json.dumps(
                {
                    "bucket": bucket,
                    "stored_at": datetime.now().isoformat(),
                }
            ),
        )

    def _extract_reminder_hint(self, raw_payload_json: str | None) -> dict:
        if not raw_payload_json:
            return {"maybe_require_a_reminder": False, "reminder_context": self._empty_reminder_context()}
        try:
            payload = json.loads(raw_payload_json)
        except json.JSONDecodeError:
            return {"maybe_require_a_reminder": False, "reminder_context": self._empty_reminder_context()}
        return {
            "maybe_require_a_reminder": self._truthy(payload.get("maybe_require_a_reminder")),
            "reminder_context": self._normalize_reminder_context(payload.get("reminder_context")),
        }

    def _empty_reminder_context(self) -> dict[str, str]:
        return {"message_to_user": "", "due_date": ""}

    def _normalize_reminder_context(self, value) -> dict[str, str]:
        if isinstance(value, dict):
            return {
                "message_to_user": self._clean_text(value.get("message_to_user")),
                "due_date": self._clean_text(value.get("due_date")),
            }
        legacy_text = self._clean_text(value)
        if not legacy_text:
            return self._empty_reminder_context()
        return {"message_to_user": legacy_text, "due_date": ""}

    def _reminder_message(self, reminder_context) -> str:
        normalized = self._normalize_reminder_context(reminder_context)
        return self._clean_text(normalized.get("message_to_user"))

    def _format_reminder_context(self, reminder_context) -> str:
        normalized = self._normalize_reminder_context(reminder_context)
        message = self._clean_text(normalized.get("message_to_user"))
        due_date = self._clean_text(normalized.get("due_date"))
        if message and due_date:
            return f"{message} Due: {due_date}"
        return message or due_date

    def _parse_due_datetime(self, due_date_text: str, *, reminder_text: str) -> datetime | None:
        clean_due_date = self._clean_text(due_date_text)
        if not clean_due_date:
            return None
        try:
            return datetime.fromisoformat(clean_due_date)
        except ValueError:
            self.logger.warning(
                "Invalid reminder due_date %r for passive observer reminder %r; creating task without due date.",
                clean_due_date,
                reminder_text,
            )
            return None

    def _dedupe_preserve_order(self, values: List[str]) -> List[str]:
        seen: set[str] = set()
        result: List[str] = []
        for value in values:
            key = self._normalize_activity(value)
            if not key or key in seen:
                continue
            seen.add(key)
            result.append(value)
        return result

    def _list_text(self, value) -> List[str]:
        if not isinstance(value, list):
            return []
        results: List[str] = []
        for item in value:
            text = self._clean_text(item)
            if text:
                results.append(text)
        return results

    def _normalize_activity(self, text: str) -> str:
        cleaned = self._clean_text(text).lower()
        if not cleaned:
            return ""
        return re.sub(r"\s+", " ", cleaned)

    def _clean_text(self, value) -> str:
        if value is None:
            return ""
        return str(value).strip()

    def _truthy(self, value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    async def _consume_stream_text(self, completion) -> str:
        parts: List[str] = []
        async for chunk in completion:
            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                parts.append(delta.content)
        return "".join(parts)

    def _parse_json_object(self, response_text: str) -> dict:
        text = response_text.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            self.logger.warning("Passive observer follow-up response was not valid JSON.")
            return {}

    def _semantic_context(self, *, activity: str, rows: List[dict]) -> List[dict]:
        if self.semantic_memory is None:
            return []
        query = " ".join(
            [
                activity,
                *[
                    " ".join(
                        part
                        for part in [
                            row.get("summary", ""),
                            row.get("detailed_description", ""),
                            self._format_reminder_context(row.get("reminder_context")),
                        ]
                        if part
                    )
                    for row in rows
                ],
            ]
        ).strip()
        if not query:
            return []
        results = self.semantic_memory.retrieve(query=query, limit=8, rerank_limit=5)
        return self.semantic_memory.format_context(results)
