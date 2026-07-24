import contextlib
import json
import logging
import mimetypes
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


class PlainCaptureStore:
    """Plain on-disk capture storage with stable references and sidecar metadata."""

    uri_prefix = "capture://"
    storage_mode = "plain"

    def __init__(self, root: str, *, logger: logging.Logger | None = None):
        self.root = Path(root)
        self.metadata_root = self.root / "metadata"
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.root.mkdir(parents=True, exist_ok=True)
        self.metadata_root.mkdir(parents=True, exist_ok=True)

    def store_file(self, path: str, *, kind: str, delete_source: bool = True) -> str:
        source = Path(path)
        uri = self.store_bytes(
            source.read_bytes(),
            original_name=source.name,
            kind=kind,
            mime_type=mimetypes.guess_type(source.name)[0] or "application/octet-stream",
        )
        if delete_source:
            source.unlink(missing_ok=True)
        return uri

    def store_bytes(self, data: bytes, *, original_name: str, kind: str, mime_type: str) -> str:
        capture_id = uuid.uuid4().hex
        safe_kind = self._safe_component(kind or "other")
        safe_name = self._safe_filename(original_name or "capture.bin")
        kind_root = self.root / safe_kind
        kind_root.mkdir(parents=True, exist_ok=True)
        destination = kind_root / f"{capture_id}_{safe_name}"
        destination.write_bytes(data)
        metadata = {
            "capture_id": capture_id,
            "original_name": safe_name,
            "kind": safe_kind,
            "mime_type": mime_type,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "size": len(data),
            "relative_path": destination.relative_to(self.root).as_posix(),
            "storage_mode": "plain",
        }
        self._metadata_path(capture_id).write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return f"{self.uri_prefix}{capture_id}"

    def read_bytes(self, uri: str) -> tuple[bytes, dict]:
        metadata = self._metadata(uri)
        return self._data_path(metadata).read_bytes(), metadata

    @contextlib.contextmanager
    def materialize(self, uri: str) -> Iterator[str]:
        yield str(self._data_path(self._metadata(uri)))

    def delete(self, uri: str) -> bool:
        capture_id = self._capture_id(uri)
        metadata_path = self._metadata_path(capture_id)
        if not metadata_path.exists():
            return False
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self._data_path(metadata).unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        return True

    def read_stored_file(self, uri: str) -> tuple[bytes, str]:
        data, metadata = self.read_bytes(uri)
        return data, str(metadata.get("original_name") or f"{self._capture_id(uri)}.bin")

    def size_bytes(self) -> int:
        return sum(
            path.stat().st_size
            for path in self.root.rglob("*")
            if path.is_file() and self.metadata_root not in path.parents
        )

    def storage_status(self) -> dict:
        usage = shutil.disk_usage(self.root)
        return {
            "capture_size_bytes": self.size_bytes(),
            "disk_free_bytes": usage.free,
            "disk_total_bytes": usage.total,
            "storage_pressure": usage.free < max(2 * 1024**3, int(usage.total * 0.05)),
            "storage_mode": "plain",
            "capture_root": str(self.root),
        }

    def _metadata(self, uri: str) -> dict:
        path = self._metadata_path(self._capture_id(uri))
        if not path.exists():
            raise FileNotFoundError(uri)
        return json.loads(path.read_text(encoding="utf-8"))

    def _data_path(self, metadata: dict) -> Path:
        candidate = (self.root / str(metadata["relative_path"])).resolve()
        root = self.root.resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError("capture metadata points outside the capture root")
        return candidate

    def _metadata_path(self, capture_id: str) -> Path:
        return self.metadata_root / f"{capture_id}.json"

    @classmethod
    def _capture_id(cls, uri: str) -> str:
        value = str(uri or "")
        if not value.startswith(cls.uri_prefix):
            raise ValueError("expected a capture:// URI")
        capture_id = value.removeprefix(cls.uri_prefix)
        if len(capture_id) != 32 or any(ch not in "0123456789abcdef" for ch in capture_id):
            raise ValueError("invalid capture URI")
        return capture_id

    @staticmethod
    def _safe_component(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "other"

    @classmethod
    def _safe_filename(cls, value: str) -> str:
        name = Path(value).name
        stem = cls._safe_component(Path(name).stem)[:120] or "capture"
        suffix = re.sub(r"[^A-Za-z0-9.]", "", Path(name).suffix)[:16]
        return f"{stem}{suffix}"
