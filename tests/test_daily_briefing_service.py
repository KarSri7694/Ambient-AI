import json
import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from application.services.daily_briefing_service import DailyBriefingService


class FakeAutonomyStore:
    def __init__(self, now):
        self.now = now
        self.cache = {}
        self.audits = []

    def list_inbox_items_between(self, start, end, *, limit):
        return [SimpleNamespace(
            inbox_id="inbox-1", opportunity_id="opp-1", title="Dependency update found",
            summary="A relevant local dependency has a newer stable release.", status="ready",
            updated_at=self.now.isoformat(),
        )] if datetime.fromisoformat(start) <= self.now < datetime.fromisoformat(end) else []

    def list_delegated_tasks_between(self, start, end, *, limit):
        return []

    def list_activity_runs_between(self, start, end, *, limit):
        return []

    def list_approvals(self, *, status, limit):
        return [SimpleNamespace(
            approval_id="approval-1", capability="browser.use", constraints_json=json.dumps({
                "reason": "Verify the release notes", "raw_capture": "must not enter digest"
            }), status="pending", created_at=self.now.isoformat(), expires_at=(self.now + timedelta(hours=1)).isoformat(),
        )]

    def list_audit_between(self, start, end, *, limit):
        return list(self.audits)

    def daily_event_counts(self, start, end):
        return {"processed": 2} if datetime.fromisoformat(start) <= self.now < datetime.fromisoformat(end) else {}

    def get_daily_briefing(self, day):
        return self.cache.get(day)

    def upsert_daily_briefing(self, briefing):
        self.cache[briefing["briefing_date"]] = dict(briefing)
        return briefing


class FakeReportStore:
    def __init__(self, now):
        self.now = now

    def list_reports_between(self, start, end, *, limit):
        if not datetime.fromisoformat(start) <= self.now < datetime.fromisoformat(end):
            return []
        return [SimpleNamespace(
            interaction_id="report-1", created_at=self.now.isoformat(), completed_at=self.now.isoformat(),
            report_json=json.dumps({"title": "Organized lecture notes", "summary": "Merged new material into an existing note.", "status": "completed"}),
            response_text="private full model output", error_text=None,
        )]


class FakeTasks:
    def get_all_pending_tasks(self):
        return []


class FakeOrganizer:
    def list_artifacts(self, *, status, limit):
        return []

    def list_maintenance_history(self, *, limit):
        return {"runs": [], "merges": []}


class FakeLLM:
    def __init__(self):
        self.messages = None

    async def chat_completion_stream(self, **kwargs):
        self.messages = kwargs["messages"]
        text = json.dumps({
            "headline": "Notes organized and one update awaits review",
            "overview": "Ambient AI consolidated a note and found a relevant update.",
            "accomplishments": ["Organized lecture notes"],
            "updates": ["A dependency update was found"],
            "failures": [],
            "attention": ["Approve the bounded browser verification"],
        })

        async def chunks():
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])
        return chunks()


class FakeUserContext:
    def build_context(self, *, include_semantic):
        assert include_semantic is False
        return {
            "stable_user_profile": "The user prefers concise technical summaries.",
            "working_memory": "The user is currently improving Ambient AI.",
        }


class HangingLLM:
    async def chat_completion_stream(self, **kwargs):
        await asyncio.sleep(1)


def make_service(now):
    llm = FakeLLM()
    service = DailyBriefingService(
        autonomy_store=FakeAutonomyStore(now), report_store=FakeReportStore(now),
        task_store=FakeTasks(), organizer=FakeOrganizer(), llm_provider=llm,
        user_context_service=FakeUserContext(), model="reporter", cooldown_minutes=15,
    )
    return service, llm


def test_snapshot_marks_items_new_without_needing_ai():
    now = datetime.now(timezone.utc)
    service, _ = make_service(now)
    result = service.snapshot(
        date_value=now.astimezone().date().isoformat(),
        since=(now - timedelta(minutes=1)).isoformat(),
    )
    assert result["briefing"] is None
    assert result["briefing_pending"] is True
    assert result["new_count"] >= 2
    assert result["counts"]["reports"] == 1
    assert result["counts"]["pending_approvals"] == 1
    assert "I completed or updated" in result["latest_narrative"]


def test_idle_refresh_caches_sanitized_ai_digest():
    now = datetime.now(timezone.utc)
    service, llm = make_service(now)
    result = asyncio.run(service.refresh_if_due())
    assert result["status"] == "completed"
    snapshot = service.snapshot(date_value=now.astimezone().date().isoformat())
    assert snapshot["briefing"]["headline"].startswith("Notes organized")
    prompt = llm.messages[1]["content"]
    assert "private full model output" not in prompt
    assert "raw_capture" not in prompt
    assert "prefers concise technical summaries" in prompt
    assert snapshot["briefing"]["failures"] == []
    service.autonomy_store.audits.append({"action": "resource.model_loaded"})
    after_model_audit = service.snapshot(date_value=now.astimezone().date().isoformat())
    assert after_model_audit["briefing_stale"] is False
    assert after_model_audit["background"]["audit_actions"]["resource.model_loaded"] == 1
    service.autonomy_store.daily_event_counts = (
        lambda start, end: {"processed": 99, "pending": 7}
        if datetime.fromisoformat(start) <= now < datetime.fromisoformat(end)
        else {}
    )
    after_event_count_churn = service.snapshot(date_value=now.astimezone().date().isoformat())
    assert after_event_count_churn["briefing_stale"] is False
    assert asyncio.run(service.refresh_if_due())["ran"] is False


def test_refresh_timeout_clears_running_state_and_schedules_retry():
    now = datetime.now(timezone.utc)
    service, _ = make_service(now)
    service.llm_provider = HangingLLM()
    service.generation_timeout_seconds = 0.01

    result = asyncio.run(service.refresh_if_due())

    assert result["status"] == "failed"
    snapshot = service.snapshot(date_value=now.astimezone().date().isoformat())
    assert snapshot["briefing_refresh"]["running"] is False
    assert snapshot["briefing_refresh"]["retry_after"] is not None
    assert snapshot["briefing_refresh"]["last_error"]
