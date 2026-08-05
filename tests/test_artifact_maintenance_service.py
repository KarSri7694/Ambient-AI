import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from application.services.artifact_maintenance_service import ArtifactMaintenanceService
from application.services.artifact_organizer_service import ArtifactOrganizer


class _Delta:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.delta = _Delta(content)


class _Chunk:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class _LLM:
    def __init__(self, responses):
        self.responses = list(responses)

    async def chat_completion_stream(self, **_kwargs):
        response = self.responses.pop(0)

        async def generate():
            yield _Chunk(response)

        return generate()


def test_maintenance_merges_only_llm_confirmed_same_thread(tmp_path):
    organizer = ArtifactOrganizer(tmp_path)
    first = organizer.save_new(
        title="Compiler Course Lecture 4",
        summary="Parsing lecture notes.",
        detailed_report="The lecture introduced LR parsing.",
        source_ref="screen/one",
    )
    second = organizer.save_new(
        title="Compiler Course Lecture 4 Continued",
        summary="Continued parsing lecture notes.",
        detailed_report="The same lecture continued with shift-reduce conflicts.",
        source_ref="screen/two",
    )
    pair_response = json.dumps({
        "pairs": [{
            "left_id": first["artifact_id"],
            "right_id": second["artifact_id"],
            "same_thread": True,
            "confidence": 0.96,
            "reason": "The same course lecture continues across both notes.",
        }]
    })
    merge_response = json.dumps({
        "final_title": "Compiler Course Lecture 4",
        "short_summary": "LR parsing and shift-reduce conflict notes.",
        "detailed_summary": "Consolidated notes from the same compiler lecture.",
        "merged_content": "LR parsers process viable prefixes. Shift-reduce conflicts require disambiguation.",
        "reason": "Combined sequential observations of one lecture.",
    })
    service = ArtifactMaintenanceService(
        organizer=organizer,
        llm_provider=_LLM([pair_response, merge_response]),
        model="test-model",
    )
    service._candidate_pairs = lambda _artifacts: [{
        "left_id": first["artifact_id"],
        "right_id": second["artifact_id"],
        "left": {},
        "right": {},
        "retrieval_score": 1.0,
        "match_source": "test",
    }]

    result = asyncio.run(service.run(trigger_kind="test"))

    assert result["merged_cluster_count"] == 1
    assert result["archived_count"] == 1
    assert len(organizer.list_artifacts(status="active")) == 1
    assert len(organizer.list_artifacts(status="archived")) == 1
    history = organizer.list_maintenance_history()
    assert history["runs"][0]["status"] == "completed"
    assert history["merges"][0]["canonical_artifact_id"] in {first["artifact_id"], second["artifact_id"]}


def test_maintenance_due_is_not_retriggered_by_artifact_changes_before_daily_interval(tmp_path):
    organizer = ArtifactOrganizer(tmp_path)
    organizer.save_new(
        title="Existing note",
        summary="A note already in the library.",
        detailed_report="Existing details.",
        source_ref="test/one",
    )
    run_id = organizer.start_maintenance_run("idle")
    organizer.finish_maintenance_run(run_id, status="completed")

    # Simulate reports arriving after the completed run. The idle loop may call
    # maintenance_status many times before the next daily boundary.
    for index in range(3):
        organizer.save_new(
            title=f"New note {index}",
            summary="A newly created note.",
            detailed_report="New details.",
            source_ref=f"test/new-{index}",
        )

    status = organizer.maintenance_status(min_changes=3, interval_hours=24)

    assert status["changed_count"] >= 3
    assert status["due"] is False
    assert status["due_reasons"] == []
