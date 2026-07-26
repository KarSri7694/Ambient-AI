from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rocm_tuning.service import RocmTuningService, config_from_ini


def _run_preflight(config_path: Path, output_dir: Path) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "rocm_preflight.py"), "--backend", "amd_rocm", "--require-supported-gpu"],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError:
        payload = {"ok": False, "error": completed.stderr or completed.stdout}
    (output_dir / "preflight.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _write_report(output_dir: Path, *, preflight: dict[str, object], tuning: dict[str, object]) -> Path:
    latest = {}
    results = tuning.get("results") if isinstance(tuning, dict) else None
    completed = [item for item in (results or []) if item.get("status") == "completed"]
    if completed:
        latest = completed[0]
    report = output_dir / "demo_report.md"
    report.write_text(
        "\n".join([
            "# ROCm Demo Pack",
            "",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "## Preflight",
            "",
            f"- OK: {preflight.get('ok')}",
            f"- GPU: {(preflight.get('accelerator') or {}).get('gpu_name')}",
            f"- Backend: {(preflight.get('accelerator') or {}).get('backend')}",
            f"- Runtime: {(preflight.get('accelerator') or {}).get('runtime_version')}",
            "",
            "## Tuning",
            "",
            f"- Run status: {tuning.get('status')}",
            f"- Run id: {tuning.get('run_id')}",
            f"- Completed candidates: {len(completed)}",
            f"- Best median TTFT: {(latest.get('summary') or {}).get('ttft_seconds_median')}",
            f"- Best chars/sec: {(latest.get('summary') or {}).get('chars_per_second_median')}",
            "",
            "Artifacts: preflight.json, rocm_tuning.json, rocm_tuning.csv, tuned_models_preset.ini",
        ]),
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a ROCm hackathon evidence pack.")
    parser.add_argument("--config", default=str(ROOT / "config.ini"))
    parser.add_argument("--skip-tuning", action="store_true")
    args = parser.parse_args()

    config = config_from_ini(args.config, project_root=ROOT)
    output_dir = Path(config.data_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight = _run_preflight(Path(args.config), output_dir)
    tuning = {"status": "skipped", "run_id": None, "results": []}
    if not args.skip_tuning:
        tuning = RocmTuningService(config=config).run()
    report = _write_report(output_dir, preflight=preflight, tuning=tuning)
    print(json.dumps({"report": str(report), "preflight_ok": preflight.get("ok"), "tuning_status": tuning.get("status")}, indent=2))
    return 0 if preflight.get("ok") and tuning.get("status") in {"completed", "skipped"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
