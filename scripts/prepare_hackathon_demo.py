"""Create non-sensitive, deterministic inputs for the AMD hackathon recording.

The script never touches normal user data. It creates a standalone demo folder
with a tiny local project, five observation descriptions, a harmless mailbox
fixture, and a recording checklist. Point document tools at DemoOutput only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Ambient AI hackathon demo fixtures.")
    parser.add_argument("--output", default=".ambient_data/demo-fixtures")
    args = parser.parse_args()
    root = Path(args.output).resolve()
    project = root / "DemoProject"
    output = root / "DemoOutput"
    project.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    (project / "README.md").write_text(
        "# Demo Project\n\nA local ambient assistant prototype needs a model-integration assessment.\n",
        encoding="utf-8",
    )
    (project / "integration_notes.md").write_text(
        "Current goal: evaluate a quantized Qwen model for local tool planning.\n"
        "Constraint: preserve local ROCm deployment and existing permission gates.\n",
        encoding="utf-8",
    )
    observations = [
        "User is reading AMD Radeon Hackathon Track 2 requirements.",
        "User is comparing Qwen3.6-35B quantized ROCm benchmark settings.",
        "User is viewing the DemoProject integration notes.",
        "User is checking a model download or benchmark progress indicator.",
        "User is revisiting Track 2 requirements and preparing a demo plan.",
    ]
    (root / "visual_observations.json").write_text(json.dumps(observations, indent=2), encoding="utf-8")
    mailbox = [
        {"subject": "Demo: Track 2 submission checklist", "importance": "high", "needs_reply": True},
        {"subject": "Demo: Team update", "importance": "medium", "needs_reply": False},
        {"subject": "Demo: newsletter", "importance": "low", "needs_reply": False},
    ]
    (root / "gmail_demo_messages.json").write_text(json.dumps(mailbox, indent=2), encoding="utf-8")
    (root / "RECORDING_CHECKLIST.md").write_text(
        "# Recording checklist\n\n"
        "- Verify ROCm preflight and llama.cpp model readiness.\n"
        "- Use only this fixture folder and a dedicated harmless Gmail label.\n"
        "- Preload Qwen3.6-35B before browser research.\n"
        "- Approve browser use, then leave input idle before it starts.\n"
        "- Confirm generated documents stay in DemoOutput.\n"
        "- Clear private captures and redact configuration before recording.\n",
        encoding="utf-8",
    )
    print(json.dumps({"fixture_root": str(root), "project": str(project), "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
