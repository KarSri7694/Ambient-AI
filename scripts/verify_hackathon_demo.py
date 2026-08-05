"""Read-only readiness checks for the AMD hackathon recording."""

from __future__ import annotations

import argparse
import configparser
import json
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent.parent


def check_url(url: str) -> dict[str, object]:
    try:
        with urlopen(url, timeout=5) as response:  # nosec B310 -- local configured endpoint only
            return {"ok": 200 <= response.status < 500, "status": response.status}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify local Ambient AI demo readiness.")
    parser.add_argument("--config", default=str(ROOT / "config.ini"))
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config, encoding="utf-8")
    api = config.get("runtime", "api_base_url", fallback="http://127.0.0.1:8080").rstrip("/")
    checks: dict[str, object] = {
        "router_models": check_url(f"{api}/models"),
        "runtime_health": check_url("http://127.0.0.1:8000/healthz"),
        "semantic_embedding": check_url(config.get("semantic_memory", "embedding_api_base_url", fallback="http://127.0.0.1:8081").rstrip("/") + "/v1/models"),
        "required_flags": {
            "rocm": config.get("accelerator", "backend", fallback="auto") in {"amd_rocm", "auto"},
            "temporal_semantic_memory": config.getboolean("semantic_memory", "enabled", fallback=False),
            "artifact_organization": config.getboolean("artifacts", "organizer_enabled", fallback=False),
            "home": config.getboolean("home", "enabled", fallback=False),
            "recurring_tasks": config.getboolean("recurring_tasks", "enabled", fallback=False),
        },
    }
    preflight = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "rocm_preflight.py"), "--backend", "amd_rocm"],
        capture_output=True, text=True, check=False,
    )
    try:
        checks["rocm_preflight"] = json.loads(preflight.stdout or "{}")
    except json.JSONDecodeError:
        checks["rocm_preflight"] = {"ok": False, "error": preflight.stderr or preflight.stdout}
    required_flags = checks["required_flags"]
    preflight_ok = bool((checks.get("rocm_preflight") or {}).get("ok"))
    ok = (
        all(item.get("ok", False) for item in [checks["router_models"], checks["runtime_health"], checks["semantic_embedding"]])
        and all(bool(value) for value in required_flags.values())
        and preflight_ok
    )
    print(json.dumps({"ok": ok, "checks": checks}, indent=2))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
