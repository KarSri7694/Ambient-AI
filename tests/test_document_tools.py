import importlib
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

pytest.importorskip("docx")
pytest.importorskip("pptx")
pytest.importorskip("reportlab")

doc_tools = importlib.import_module("doc_tools")


def _call(tool, *args, **kwargs):
    if callable(tool):
        return tool(*args, **kwargs)
    for attribute in ("fn", "func", "function", "__wrapped__"):
        candidate = getattr(tool, attribute, None)
        if callable(candidate):
            return candidate(*args, **kwargs)
    raise TypeError(f"Could not call {tool!r}")


@pytest.fixture
def allowed_root(tmp_path, monkeypatch):
    original = doc_tools.CONFIG.get_str

    def get_str(section, option, fallback):
        if section == "documents" and option == "allowed_paths_json":
            return json.dumps([str(tmp_path)])
        return original(section, option, fallback)

    monkeypatch.setattr(doc_tools.CONFIG, "get_str", get_str)
    return tmp_path


def test_document_create_read_and_inspect_docx(allowed_root):
    path = allowed_root / "notes.docx"
    created = _call(doc_tools.document_create, "docx", str(path), "Notes", "# Heading\n- One")

    assert created["ok"] is True
    assert _call(doc_tools.document_inspect, str(path))["paragraph_count"] >= 2
    content = _call(doc_tools.document_read, str(path))
    assert "Heading" in content["paragraphs"]


def test_in_place_docx_edit_creates_adjacent_backup(allowed_root):
    path = allowed_root / "draft.docx"
    assert _call(doc_tools.document_create, "docx", str(path), "Draft", "original text")["ok"]

    edited = _call(doc_tools.document_edit, str(path), [{"type": "replace_text", "old": "original", "new": "updated"}])

    assert edited["ok"] is True
    assert Path(edited["backup_path"]).exists()
    assert "updated text" in _call(doc_tools.document_read, str(path))["paragraphs"]


def test_document_server_rejects_paths_outside_configured_root(allowed_root, tmp_path):
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")

    result = _call(doc_tools.document_read, str(outside))

    assert result["ok"] is False
    assert "outside the granted" in result["error"]


def test_pdf_edits_require_a_separate_output(allowed_root):
    source = allowed_root / "source.pdf"
    assert _call(doc_tools.document_create, "pdf", str(source), "Source", "hello")["ok"]

    rejected = _call(doc_tools.document_edit, str(source), [{"type": "watermark_text", "text": "DRAFT"}])
    output = allowed_root / "watermarked.pdf"
    edited = _call(doc_tools.document_edit, str(source), [{"type": "watermark_text", "text": "DRAFT"}], str(output))

    assert rejected["ok"] is False
    assert edited["ok"] is True
    assert output.exists()
