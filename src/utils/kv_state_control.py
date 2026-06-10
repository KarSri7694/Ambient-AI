from pathlib import Path
import re
from typing import Optional, List, Dict, Any
import base64
import json
from datetime import datetime

class KVStateControl:
    def __init__(self, owner):
        self.owner = owner

    def kv_state_dir(self) -> Path:
        """Return the local directory used for llama.cpp KV state files."""
        parent_dir = Path(__file__).parent.parent.parent
        kv_state_dir = parent_dir / "model_kv_states"
        kv_state_dir.mkdir(exist_ok=True)
        return kv_state_dir

    def shared_state_file(self) -> Path:
        """Return the shared metadata file used across app and MCP tool processes."""
        return self.kv_state_dir() / "shared_state.json"

    def read_shared_state(self) -> Dict[str, Any]:
        """Read cross-process llama.cpp state metadata from disk."""
        shared_state_file = self.shared_state_file()
        if not shared_state_file.exists():
            return {}

        try:
            return json.loads(shared_state_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            self.owner.logger.warning("Failed to read shared llama.cpp state file: %s", exc)
            return {}

    def write_shared_state(self, state: Dict[str, Any]) -> None:
        """Persist cross-process llama.cpp state metadata to disk."""
        shared_state_file = self.shared_state_file()
        shared_state_file.write_text(
            json.dumps(state, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        
    def update_shared_state(self, **updates: Any) -> None:
        """Merge updated metadata fields into the shared llama.cpp state file."""
        state = self.read_shared_state()
        state.update(updates)
        self.write_shared_state(state)

    def kv_state_stack(self) -> List[str]:
        """Return the persisted KV-state stack from shared metadata."""
        stack = self.read_shared_state().get("kv_state_stack", [])
        if not isinstance(stack, list):
            self.owner.logger.warning("Ignoring invalid kv_state_stack in shared_state.json.")
            return []
        return stack

    def push_kv_state(self, kv_state_file: Path | str) -> None:
        """Push a KV-state path onto the persisted LIFO stack."""
        stack = self.kv_state_stack()
        stack.append(str(kv_state_file))
        self.update_shared_state(kv_state_stack=stack)

    def pop_kv_state(self) -> str:
        """Pop the most recently saved KV-state path from the persisted LIFO stack."""
        stack = self.kv_state_stack()
        if not stack:
            raise RuntimeError("No saved KV state is recorded in shared_state.json.")

        kv_state_file = stack.pop()
        self.update_shared_state(kv_state_stack=stack)
        return kv_state_file

    def safe_kv_state_filename(self) -> str:
        """Build a timestamped KV state filename containing a reversible model name."""
        model_name = self.owner.currently_loaded_model or "current_model"
        encoded_model_name = self.encode_model_name_for_filename(model_name)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{encoded_model_name}_{timestamp}_kv_state.bin"

    @staticmethod
    def encode_model_name_for_filename(model_name: str) -> str:
        """Encode a model name into a filesystem-safe filename component."""
        encoded = base64.urlsafe_b64encode(model_name.encode("utf-8")).decode("ascii")
        return f"model_b64_{encoded.rstrip('=')}"

    @staticmethod
    def decode_model_name_from_filename(value: str) -> Optional[str]:
        """Decode a filename component created by `encode_model_name_for_filename`."""
        prefix = "model_b64_"
        if not value.startswith(prefix):
            return None

        encoded = value[len(prefix):]
        encoded += "=" * (-len(encoded) % 4)
        return base64.urlsafe_b64decode(encoded.encode("ascii")).decode("utf-8")

    @classmethod
    def extract_model_name_from_kv_state_file(cls, kv_state_file: str) -> str:
        """Extract the saved model name from a timestamped KV state filename."""
        kv_state_name = Path(kv_state_file).name
        match = re.fullmatch(
            r"(?P<model>.+)_\d{8}_\d{6}_kv_state\.bin",
            kv_state_name,
        )
        if not match:
            raise ValueError(
                "KV state filename must match '<model>_YYYYMMDD_HHMMSS_kv_state.bin'."
            )

        encoded_or_legacy_name = match.group("model")
        return cls.decode_model_name_from_filename(encoded_or_legacy_name) or encoded_or_legacy_name
