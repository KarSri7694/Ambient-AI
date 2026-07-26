from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from infrastructure.runtime_log_server import RuntimeLogBuffer, create_runtime_log_app
from real_world_testing.lab import RealWorldLab
from real_world_testing.runtime_lock import RuntimeOwnershipLock
from rocm_tuning.service import RocmTuningService, config_from_ini


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Ambient AI real-world evaluation lab.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.ini"))
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / ".ambient_data" / "real_world_tests"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("The unauthenticated lab may only bind to loopback")
    lab = RealWorldLab(
        project_root=PROJECT_ROOT, config_path=args.config, data_root=args.data_root,
        suites_root=Path(__file__).parent / "suites",
    )
    rocm_tuning = RocmTuningService(config=config_from_ini(args.config, project_root=PROJECT_ROOT))
    app = create_runtime_log_app(RuntimeLogBuffer(), real_world_lab=lab, rocm_tuning_service=rocm_tuning)
    lock = RuntimeOwnershipLock(PROJECT_ROOT / ".ambient_data" / "runtime.lock", "real-world-lab")
    with lock:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
