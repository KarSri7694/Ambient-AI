from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from infrastructure.accelerator import SUPPORTED_ROCM_WINDOWS_ARCHES, detect_accelerator


def _llama_version(path: str | None) -> dict[str, str | int | None]:
    if not path:
        return {"path": None, "returncode": None, "version": None}
    try:
        completed = subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        output = (completed.stdout or completed.stderr or "").strip()
        return {"path": path, "returncode": completed.returncode, "version": output.splitlines()[0] if output else ""}
    except Exception as exc:
        return {"path": path, "returncode": -1, "version": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the AMD ROCm Windows runtime for Ambient AI.")
    parser.add_argument("--backend", default="amd_rocm", choices=["auto", "amd_rocm", "nvidia_cuda", "cpu"])
    parser.add_argument("--require-supported-gpu", action="store_true")
    parser.add_argument("--llama-server", default=None, help="Optional path to AMD validated llama-server.exe")
    args = parser.parse_args()

    info = detect_accelerator(args.backend, require_supported_gpu=args.require_supported_gpu)
    payload = {
        "ok": info.available and (not args.require_supported_gpu or info.supported is True),
        "accelerator": info.to_dict(),
        "supported_rocm_windows_arches": sorted(SUPPORTED_ROCM_WINDOWS_ARCHES),
        "llama_cpp": _llama_version(args.llama_server),
        "python": sys.version,
    }
    print(json.dumps(payload, indent=2))
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
