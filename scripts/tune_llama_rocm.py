from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rocm_tuning.service import RocmTuningService, config_from_ini, config_from_json_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an isolated llama.cpp ROCm tuning sweep.")
    parser.add_argument("--config", default=str(ROOT / "config.ini"), help="Ambient config.ini with [rocm_tuning].")
    parser.add_argument("--json-config", default=None, help="Optional standalone ROCm tuning JSON config.")
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()

    if args.json_config:
        config = config_from_json_file(args.json_config, data_root=args.data_root)
    else:
        config = config_from_ini(args.config, project_root=ROOT)
        if args.data_root:
            config = type(config)(**{**config.__dict__, "data_root": Path(args.data_root)})

    service = RocmTuningService(config=config)
    run = service.run()
    print(json.dumps(run, indent=2))
    return 0 if run.get("status") == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
