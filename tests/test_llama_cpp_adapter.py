import asyncio
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from infrastructure.adapter.llamaCppAdapter import LlamaCppAdapter
from utils.kv_state_handling import KVStateControl


def test_kv_state_filename_round_trips_model_name_with_path_chars():
    model_name = "Qwen/Qwen2.5-VL-3B Instruct:latest"
    filename = (
        f"{KVStateControl.encode_model_name_for_filename(model_name)}"
        "_20260608_123456_kv_state.bin"
    )

    assert KVStateControl.extract_model_name_from_kv_state_file(filename) == model_name


def test_kv_state_filename_extracts_legacy_model_name_with_underscores():
    filename = "Qwen_Qwen2.5_VL_3B_Instruct_20260608_123456_kv_state.bin"

    assert (
        KVStateControl.extract_model_name_from_kv_state_file(filename)
        == "Qwen_Qwen2.5_VL_3B_Instruct"
    )


def test_kv_state_filename_rejects_unrecognized_names():
    with pytest.raises(ValueError):
        KVStateControl.extract_model_name_from_kv_state_file("state.bin")


def test_restore_and_load_loads_extracted_model_before_restore(monkeypatch):
    model_name = "models/Qwen 3 VL"
    filename = (
        f"{KVStateControl.encode_model_name_for_filename(model_name)}"
        "_20260608_123456_kv_state.bin"
    )
    adapter = LlamaCppAdapter.__new__(LlamaCppAdapter)
    adapter.logger = logging.getLogger("test")
    adapter.currently_loaded_model = None
    calls = []

    async def fake_load_model(name, unload_previous=True):
        calls.append(("load", name, unload_previous))
        adapter.currently_loaded_model = name

    async def fake_restore(name):
        calls.append(("restore", name))

    class FakeKVState:
        def pop_kv_state(self):
            return filename

        @staticmethod
        def extract_model_name_from_kv_state_file(name):
            return KVStateControl.extract_model_name_from_kv_state_file(name)

    adapter.kv_state = FakeKVState()
    monkeypatch.setattr(adapter, "load_model", fake_load_model)
    monkeypatch.setattr(adapter, "restore_kv_state", fake_restore)

    asyncio.run(adapter.load_and_restore())

    assert calls == [
        ("load", model_name, True),
        ("restore", filename),
    ]


def test_save_and_unload_awaits_unload_after_successful_save(monkeypatch):
    adapter = LlamaCppAdapter.__new__(LlamaCppAdapter)
    adapter.currently_loaded_model = "model"
    calls = []

    def fake_save():
        calls.append("save")
        return Path("state.bin")

    async def fake_unload():
        calls.append("unload")
        adapter.currently_loaded_model = None
    
    monkeypatch.setattr(adapter, "save_current_kv_state", fake_save)
    monkeypatch.setattr(adapter, "unload_model", fake_unload)

    assert asyncio.run(adapter.save_and_unload()) == Path("state.bin")
    assert calls == ["save", "unload"]


def test_get_current_model_reads_shared_state_when_instance_is_empty(monkeypatch):
    adapter = LlamaCppAdapter.__new__(LlamaCppAdapter)
    adapter.currently_loaded_model = None
    adapter.kv_state = type(
        "FakeKVState",
        (),
        {"read_shared_state": staticmethod(lambda: {"currently_loaded_model": "shared-model"})},
    )()

    assert adapter.get_current_model() == "shared-model"


def test_kv_state_stack_is_lifo(monkeypatch):
    adapter = LlamaCppAdapter.__new__(LlamaCppAdapter)
    adapter.logger = logging.getLogger("test")
    shared_state = {}
    kv_state = KVStateControl(adapter)

    monkeypatch.setattr(kv_state, "read_shared_state", lambda: dict(shared_state))
    monkeypatch.setattr(kv_state, "write_shared_state", lambda state: shared_state.update(state))

    kv_state.push_kv_state("first.bin")
    kv_state.push_kv_state("second.bin")

    assert kv_state.pop_kv_state() == "second.bin"
    assert kv_state.pop_kv_state() == "first.bin"
    assert shared_state["kv_state_stack"] == []
