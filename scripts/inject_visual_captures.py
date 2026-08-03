"""Inject local image files into Ambient AI's active visual pipeline.

This is useful for replaying screenshots while live capture is paused. The
script stores each image in the configured capture store and enqueues a durable
``lightweight_visual_capture`` event in the same SQLite backlog consumed by
``app.py``.
"""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import mimetypes
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from core.models import AmbientEvent  # noqa: E402
from infrastructure.adapter.SQLiteAutonomyAdapter import SQLiteAutonomyAdapter  # noqa: E402
from infrastructure.plain_capture_store import PlainCaptureStore  # noqa: E402


@dataclass(frozen=True)
class InjectionResult:
    event_id: str
    image_path: str
    capture_ref: str
    occurred_at: str
    status: str


def _utciso(value: datetime | None = None) -> str:
    return (value or datetime.now(timezone.utc)).isoformat()


def _config_path(value: str | None) -> Path:
    return Path(value).expanduser().resolve() if value else REPO_ROOT / "config.ini"


def _resolve_repo_path(value: str, *, base: Path = REPO_ROOT) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def resolve_paths(config_path: Path) -> tuple[Path, Path]:
    parser = configparser.ConfigParser()
    parser.read(config_path, encoding="utf-8")
    user_data_dir = _resolve_repo_path(
        parser.get("runtime", "user_data_dir", fallback=".ambient_data")
    )
    autonomy_db = user_data_dir / "database" / "autonomy.db"
    capture_root = _resolve_repo_path(
        parser.get("privacy", "capture_root", fallback=str(user_data_dir / "captures"))
    )
    return autonomy_db, capture_root


def _validate_image(path: Path) -> tuple[int, int, str]:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")
    try:
        with Image.open(path) as image:
            image.verify()
            width, height = image.size
            image_format = image.format or ""
    except Exception as exc:
        raise ValueError(f"Not a readable image: {path}") from exc
    return width, height, image_format


def _event_fingerprint(capture_ref: str, image_path: Path, sequence_index: int, captured_at: str) -> str:
    basis = {
        "capture_ref": capture_ref,
        "image_path": str(image_path),
        "sequence_index": sequence_index,
        "captured_at": captured_at,
        "injection": "manual_visual_capture",
    }
    return hashlib.sha256(json.dumps(basis, sort_keys=True).encode("utf-8")).hexdigest()


def inject_images(
    image_paths: list[Path],
    *,
    autonomy_db: Path,
    capture_root: Path,
    app_name: str = "Manual Image Injection",
    window_title: str = "Injected visual capture",
    url: str = "",
    domain: str = "",
    accessible_text: str = "",
    start_at: datetime | None = None,
    spacing_seconds: float = 0.001,
    dry_run: bool = False,
) -> list[InjectionResult]:
    if not image_paths:
        raise ValueError("At least one image path is required.")

    base_time = start_at or datetime.now(timezone.utc)
    spacing = max(float(spacing_seconds), 0.001)
    store = None if dry_run else SQLiteAutonomyAdapter(str(autonomy_db))
    capture_store = PlainCaptureStore(str(capture_root))
    results: list[InjectionResult] = []

    for index, image_path in enumerate(image_paths):
        resolved = image_path.expanduser().resolve()
        width, height, image_format = _validate_image(resolved)
        captured_at = _utciso(base_time + timedelta(seconds=index * spacing))
        mime_type = mimetypes.guess_type(resolved.name)[0] or "image/png"
        if dry_run:
            capture_ref = f"capture://{'0' * 31}{index % 10}"
        else:
            capture_ref = capture_store.store_bytes(
                resolved.read_bytes(),
                original_name=resolved.name,
                kind="screenshot",
                mime_type=mime_type,
            )

        payload: dict[str, Any] = {
            "screenshot_ref": capture_ref,
            "capture_mode": "manual_injected",
            "manual_injection": True,
            "manual_sequence_index": index,
            "original_path": str(resolved),
            "original_name": resolved.name,
            "image_width": width,
            "image_height": height,
            "image_format": image_format,
            "app_name": app_name,
            "window_title": window_title,
            "process_name": "manual_injection",
            "capture_policy_applied": True,
            "capture_decision": {
                "excluded": False,
                "source": "manual_injection",
                "reason": "explicit_cli_injection",
            },
        }
        if url:
            payload["url"] = url
        if domain:
            payload["domain"] = domain
        if accessible_text:
            payload["accessible_text"] = accessible_text

        event = AmbientEvent(
            event_id=uuid.uuid4().hex,
            event_type="lightweight_visual_capture",
            source_kind="manual_image_injection",
            source_ref=capture_ref,
            occurred_at=captured_at,
            payload_json=json.dumps(payload, ensure_ascii=False),
            confidence=0.55,
            privacy_label="sensitive_visual",
            fingerprint=_event_fingerprint(capture_ref, resolved, index, captured_at),
            status="pending",
            priority=0.55,
            available_at=captured_at,
        )
        stored = event if dry_run or store is None else store.enqueue_event(event)
        results.append(
            InjectionResult(
                event_id=stored.event_id,
                image_path=str(resolved),
                capture_ref=capture_ref,
                occurred_at=stored.occurred_at,
                status=stored.status,
            )
        )
    return results


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Insert local images into Ambient AI's active visual processing backlog."
    )
    parser.add_argument("images", nargs="+", help="Image files to enqueue in the given order.")
    parser.add_argument("--config", help="Path to config.ini. Defaults to the repo config.ini.")
    parser.add_argument("--autonomy-db", help="Override autonomy.db path.")
    parser.add_argument("--capture-root", help="Override plain capture-store root.")
    parser.add_argument("--app-name", default="Manual Image Injection")
    parser.add_argument("--window-title", default="Injected visual capture")
    parser.add_argument("--url", default="")
    parser.add_argument("--domain", default="")
    parser.add_argument("--accessible-text", default="")
    parser.add_argument(
        "--spacing-seconds",
        type=float,
        default=0.001,
        help="UTC timestamp spacing between queued images. Processing keeps this order.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print without writing.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    config_path = _config_path(args.config)
    autonomy_db, capture_root = resolve_paths(config_path)
    if args.autonomy_db:
        autonomy_db = _resolve_repo_path(args.autonomy_db)
    if args.capture_root:
        capture_root = _resolve_repo_path(args.capture_root)

    results = inject_images(
        [Path(value) for value in args.images],
        autonomy_db=autonomy_db,
        capture_root=capture_root,
        app_name=args.app_name,
        window_title=args.window_title,
        url=args.url,
        domain=args.domain,
        accessible_text=args.accessible_text,
        spacing_seconds=args.spacing_seconds,
        dry_run=args.dry_run,
    )
    print(
        json.dumps(
            {
                "dry_run": bool(args.dry_run),
                "autonomy_db": str(autonomy_db),
                "capture_root": str(capture_root),
                "inserted": [result.__dict__ for result in results],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
