import json
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from scripts.inject_visual_captures import inject_images
from infrastructure.adapter.SQLiteAutonomyAdapter import SQLiteAutonomyAdapter


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (8, 8), color).save(path)


def test_injected_visual_captures_are_claimed_in_cli_order(tmp_path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    third = tmp_path / "third.png"
    _write_image(first, (255, 0, 0))
    _write_image(second, (0, 255, 0))
    _write_image(third, (0, 0, 255))

    db_path = tmp_path / "database" / "autonomy.db"
    capture_root = tmp_path / "captures"
    results = inject_images(
        [first, second, third],
        autonomy_db=db_path,
        capture_root=capture_root,
        app_name="Replay",
        window_title="Manual replay",
    )

    store = SQLiteAutonomyAdapter(str(db_path))
    claimed = [
        store.claim_next_event(event_types=["lightweight_visual_capture"])
        for _ in range(3)
    ]
    assert [event.event_id for event in claimed if event] == [result.event_id for result in results]

    payloads = [json.loads(event.payload_json) for event in claimed if event]
    assert [payload["original_name"] for payload in payloads] == ["first.png", "second.png", "third.png"]
    assert [payload["manual_sequence_index"] for payload in payloads] == [0, 1, 2]
    assert all(payload["manual_injection"] is True for payload in payloads)
    assert all(payload["screenshot_ref"].startswith("capture://") for payload in payloads)


def test_injected_visual_capture_rejects_non_images(tmp_path):
    bad_file = tmp_path / "not-an-image.txt"
    bad_file.write_text("plain text", encoding="utf-8")

    try:
        inject_images(
            [bad_file],
            autonomy_db=tmp_path / "database" / "autonomy.db",
            capture_root=tmp_path / "captures",
        )
    except ValueError as exc:
        assert "Not a readable image" in str(exc)
    else:
        raise AssertionError("Expected non-image injection to fail")
