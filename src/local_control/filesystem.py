from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from local_control.safety import PathGrantError


class FileGrantSet:
    def __init__(self, granted_paths: list[str]):
        roots: list[Path] = []
        for raw_path in granted_paths:
            value = str(raw_path or "").strip()
            if not value:
                continue
            roots.append(Path(value).expanduser().resolve(strict=False))
        if not roots:
            raise PathGrantError("At least one user-granted file or folder path is required.")
        self.roots = tuple(dict.fromkeys(roots))

    def resolve(self, path: str) -> Path:
        candidate = Path(str(path or "")).expanduser()
        if not candidate.is_absolute():
            raise PathGrantError("Filesystem paths must be absolute.")
        resolved = candidate.resolve(strict=False)
        for root in self.roots:
            try:
                resolved.relative_to(root)
                return resolved
            except ValueError:
                if resolved == root:
                    return resolved
        raise PathGrantError(f"Path is outside the granted filesystem scope: {path}")


class FilesystemControlSession:
    def __init__(
        self,
        *,
        granted_paths: list[str],
        max_read_bytes: int = 256_000,
        max_list_entries: int = 200,
    ):
        self.grants = FileGrantSet(granted_paths)
        self.max_read_bytes = max(1, int(max_read_bytes))
        self.max_list_entries = max(1, int(max_list_entries))
        self._allowed_tool_names = {
            "fs_list",
            "fs_stat",
            "fs_read_text",
            "fs_search_text",
        }

    async def get_all_tools(self) -> list[dict[str, Any]]:
        return [
            self._tool(
                "fs_list",
                "List files and folders inside a user-granted directory. The path must be an absolute path returned in root_grants or a child path from a previous filesystem tool result.",
                {"path": {"type": "string"}},
                ["path"],
            ),
            self._tool(
                "fs_stat",
                "Return metadata for a user-granted file or folder. The path must be an absolute path returned in root_grants or a child path from a previous filesystem tool result.",
                {"path": {"type": "string"}},
                ["path"],
            ),
            self._tool(
                "fs_read_text",
                "Read a UTF-8 text file inside the user-granted scope, capped by configuration. The path must be an absolute path returned by a previous filesystem tool result.",
                {"path": {"type": "string"}},
                ["path"],
            ),
            self._tool(
                "fs_search_text",
                "Search text files under a user-granted folder with capped results. The path must be an absolute path returned in root_grants or a child path from a previous filesystem tool result.",
                {
                    "path": {"type": "string"},
                    "query": {"type": "string"},
                },
                ["path", "query"],
            ),
        ]

    async def execute_tool(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        if tool_name not in self._allowed_tool_names:
            raise ValueError(f"Filesystem tool '{tool_name}' is not allowed.")
        if tool_name == "fs_list":
            return self._list(str(tool_args.get("path") or ""))
        if tool_name == "fs_stat":
            return self._stat(str(tool_args.get("path") or ""))
        if tool_name == "fs_read_text":
            return self._read_text(str(tool_args.get("path") or ""))
        if tool_name == "fs_search_text":
            return self._search_text(
                str(tool_args.get("path") or ""),
                str(tool_args.get("query") or ""),
            )
        raise ValueError(f"Unknown filesystem tool: {tool_name}")

    async def cleanup(self) -> None:
        return None

    @staticmethod
    def _tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }

    def _list(self, path: str) -> str:
        target = self.grants.resolve(path)
        if not target.exists():
            return f"Error: path does not exist: {target}"
        if not target.is_dir():
            return f"Error: path is not a directory: {target}"
        entries = []
        for index, child in enumerate(sorted(target.iterdir(), key=lambda item: item.name.lower())):
            if index >= self.max_list_entries:
                entries.append({"truncated": True, "max_entries": self.max_list_entries})
                break
            resolved_child = self.grants.resolve(str(child))
            entries.append(
                {
                    "name": resolved_child.name,
                    "path": str(resolved_child),
                    "type": "directory" if resolved_child.is_dir() else "file",
                    "size_bytes": resolved_child.stat().st_size if resolved_child.is_file() else None,
                }
            )
        return json.dumps({"entries": entries, "root_grants": [str(root) for root in self.grants.roots]}, ensure_ascii=False)

    def _stat(self, path: str) -> str:
        target = self.grants.resolve(path)
        if not target.exists():
            return f"Error: path does not exist: {target}"
        stat = target.stat()
        return json.dumps({
            "path": str(target),
            "type": "directory" if target.is_dir() else "file",
            "size_bytes": stat.st_size,
            "modified_at": stat.st_mtime,
        }, ensure_ascii=False)

    def _read_text(self, path: str) -> str:
        target = self.grants.resolve(path)
        if not target.exists() or not target.is_file():
            return f"Error: path is not a readable file: {target}"
        size = target.stat().st_size
        if size > self.max_read_bytes:
            return f"Error: file exceeds read cap ({size} bytes > {self.max_read_bytes} bytes)."
        return target.read_text(encoding="utf-8", errors="replace")

    def _search_text(self, path: str, query: str) -> str:
        target = self.grants.resolve(path)
        if not query.strip():
            return "Error: query must be non-empty."
        if not target.exists():
            return f"Error: path does not exist: {target}"
        candidates = [target] if target.is_file() else [
            item for item in target.rglob("*") if item.is_file()
        ]
        matches: list[dict[str, Any]] = []
        lowered = query.lower()
        for file_path in candidates:
            resolved = self.grants.resolve(str(file_path))
            if resolved.stat().st_size > self.max_read_bytes:
                continue
            try:
                lines = resolved.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for line_number, line in enumerate(lines, start=1):
                if lowered in line.lower():
                    matches.append({"path": str(resolved), "line": line_number, "text": line[:500]})
                    if len(matches) >= self.max_list_entries:
                        return json.dumps({"matches": matches, "truncated": True}, ensure_ascii=False)
        return json.dumps({"matches": matches, "truncated": False}, ensure_ascii=False)
