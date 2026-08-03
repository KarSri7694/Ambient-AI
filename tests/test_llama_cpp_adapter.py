import asyncio
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from infrastructure.adapter.llamaCppAdapter import LlamaCppAdapter
from application.services.runtime_interrupt_service import RuntimeShutdownController, ShutdownInProgress
from utils.kv_state_handling import KVStateControl


def test_adapter_normalizes_trailing_slash_in_base_url():
    adapter = LlamaCppAdapter("https://example.test/")

    assert adapter.base_url == "https://example.test"
    assert adapter.api_uri_v1 == "https://example.test/v1"


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


def test_restore_and_load_loads_extracted_model_before_restore(monkeypatch, tmp_path):
    model_name = "models/Qwen 3 VL"
    filename = (
        f"{KVStateControl.encode_model_name_for_filename(model_name)}"
        "_20260608_123456_kv_state.bin"
    )
    kv_path = tmp_path / filename
    kv_path.write_bytes(b"placeholder")
    adapter = LlamaCppAdapter.__new__(LlamaCppAdapter)
    adapter.logger = logging.getLogger("test")
    adapter.currently_loaded_model = None
    calls = []

    async def fake_load_model(name, unload_previous=True):
        calls.append(("load", name, unload_previous))
        adapter.currently_loaded_model = name

    def fake_restore(name):
        calls.append(("restore", name))

    class FakeKVState:
        def peek_kv_state(self):
            return str(kv_path)

        def pop_kv_state(self):
            return str(kv_path)

        @staticmethod
        def extract_model_name_from_kv_state_file(name):
            return KVStateControl.extract_model_name_from_kv_state_file(name)

    adapter.kv_state = FakeKVState()
    monkeypatch.setattr(adapter, "load_model", fake_load_model)
    monkeypatch.setattr(adapter, "restore_kv_state", fake_restore)
    monkeypatch.setattr(adapter, "_wait_for_model_restore_ready", lambda _model_name: None)

    asyncio.run(adapter.load_and_restore())

    assert calls == [
        ("load", model_name, True),
        ("restore", str(kv_path)),
    ]


def test_restore_and_load_skips_kv_restore_when_state_file_is_missing(monkeypatch, tmp_path):
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

    class FakeKVState:
        def peek_kv_state(self):
            return str(tmp_path / filename)

        def pop_kv_state(self):
            return str(tmp_path / filename)

        @staticmethod
        def extract_model_name_from_kv_state_file(name):
            return KVStateControl.extract_model_name_from_kv_state_file(name)

    adapter.kv_state = FakeKVState()
    monkeypatch.setattr(adapter, "load_model", fake_load_model)
    monkeypatch.setattr(adapter, "restore_kv_state", lambda _: calls.append(("restore", filename)))

    restored = asyncio.run(adapter.load_and_restore())

    assert restored.name == filename
    assert calls == [("load", model_name, True)]


def test_save_and_unload_awaits_unload_after_successful_save(monkeypatch):
    adapter = LlamaCppAdapter.__new__(LlamaCppAdapter)
    adapter.currently_loaded_model = "model"
    calls = []

    def fake_save(messages):
        calls.append(("save", messages))
        return Path("state.bin")

    async def fake_unload():
        calls.append("unload")
        adapter.currently_loaded_model = None
    
    monkeypatch.setattr(adapter, "save_current_kv_state", fake_save)
    monkeypatch.setattr(adapter, "unload_model", fake_unload)

    assert asyncio.run(adapter.save_and_unload([{"role": "user", "content": "hi"}])) == Path("state.bin")
    assert calls == [("save", [{"role": "user", "content": "hi"}]), "unload"]


def test_save_current_kv_state_uses_python_only_snapshot_for_multimodal(monkeypatch, tmp_path):
    adapter = LlamaCppAdapter.__new__(LlamaCppAdapter)
    adapter.logger = logging.getLogger("test")
    adapter.currently_loaded_model = "vision-model"
    pushed = []
    updated = {}
    shared_dir = tmp_path

    class FakeKVState:
        def kv_state_dir(self):
            return shared_dir

        def safe_kv_state_filename(self):
            return (
                f"{KVStateControl.encode_model_name_for_filename('vision-model')}"
                "_20260608_123456_kv_state.bin"
            )

        def update_shared_state(self, **kwargs):
            updated.update(kwargs)

        def push_kv_state(self, value):
            pushed.append(str(value))

    adapter.kv_state = FakeKVState()
    monkeypatch.setattr(adapter, "_is_multimodal_model", lambda _model_name=None: True)

    saved_path = adapter.save_current_kv_state([{"role": "user", "content": "hello"}])

    assert saved_path is not None
    assert saved_path.name.endswith("_kv_state.bin")
    assert pushed == [str(saved_path)]
    manifest = json.loads(saved_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert manifest["kv_cache_saved"] is False
    assert manifest["messages"] == [{"role": "user", "content": "hello"}]


def test_get_current_model_reads_shared_state_when_instance_is_empty(monkeypatch):
    adapter = LlamaCppAdapter.__new__(LlamaCppAdapter)
    adapter.currently_loaded_model = None
    adapter.kv_state = type(
        "FakeKVState",
        (),
        {"read_shared_state": staticmethod(lambda: {"currently_loaded_model": "shared-model"})},
    )()

    assert adapter.get_current_model() == "shared-model"


def test_sync_loaded_model_state_does_not_publish_unconfirmed_server_model(monkeypatch):
    adapter = LlamaCppAdapter.__new__(LlamaCppAdapter)
    adapter.logger = logging.getLogger("test")
    adapter.currently_loaded_model = "Qwen-3.5-9B"
    shared_state = {"currently_loaded_model": "Qwen-3.5-9B"}

    class FakeKVState:
        def read_shared_state(self):
            return dict(shared_state)

        def update_shared_state(self, **kwargs):
            shared_state.update(kwargs)

    adapter.kv_state = FakeKVState()
    monkeypatch.setattr(
        adapter,
        "_fetch_models",
        lambda: [
            {"id": "Qwen-3.5-4B", "status": {"value": "loaded"}},
            {"id": "Qwen-3.5-9B", "status": {"value": "unloaded"}},
        ],
    )

    loaded = adapter._sync_loaded_model_state()

    assert loaded == "Qwen-3.5-4B"
    assert adapter.currently_loaded_model is None
    assert shared_state["currently_loaded_model"] is None


def test_unload_model_uses_server_loaded_model_when_cache_is_stale(monkeypatch):
    adapter = LlamaCppAdapter.__new__(LlamaCppAdapter)
    adapter.logger = logging.getLogger("test")
    adapter.base_url = "http://localhost:8080"
    adapter.api_uri_v1 = "http://localhost:8080/v1"
    adapter.currently_loaded_model = "Qwen-3.5-9B"
    shared_state = {"currently_loaded_model": "Qwen-3.5-9B"}

    class FakeKVState:
        def read_shared_state(self):
            return dict(shared_state)

        def update_shared_state(self, **kwargs):
            shared_state.update(kwargs)

    adapter.kv_state = FakeKVState()
    monkeypatch.setattr(
        adapter,
        "_fetch_models",
        lambda: [
            {"id": "Qwen-3.5-4B", "status": {"value": "loaded"}},
            {"id": "Qwen-3.5-9B", "status": {"value": "unloaded"}},
        ],
    )
    monkeypatch.setattr(adapter, "_wait_for_model_status", lambda *args, **kwargs: None)

    calls = []

    class FakeResponse:
        status_code = 200
        text = "ok"

    def fake_post(url, json=None, timeout=None):
        calls.append((url, json))
        return FakeResponse()

    monkeypatch.setattr("infrastructure.adapter.llamaCppAdapter.requests.post", fake_post)

    asyncio.run(adapter.unload_model())

    assert calls == [("http://localhost:8080/models/unload", {"model": "Qwen-3.5-4B"})]
    assert adapter.currently_loaded_model is None
    assert shared_state["currently_loaded_model"] is None


class _HttpResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


def test_load_waits_for_router_and_health_before_publishing_model(monkeypatch):
    adapter = LlamaCppAdapter(
        "http://localhost:8080",
        model_load_timeout_seconds=5,
    )
    shared_state = {"currently_loaded_model": None}
    adapter.kv_state.read_shared_state = lambda: dict(shared_state)
    adapter.kv_state.update_shared_state = lambda **values: shared_state.update(values)
    model_calls = 0
    health_calls = 0

    def fake_get(url, params=None, timeout=None):
        nonlocal model_calls, health_calls
        if url.endswith("/models"):
            model_calls += 1
            status = "unloaded" if model_calls == 1 else "loading" if model_calls == 2 else "loaded"
            return _HttpResponse(payload={"data": [{"id": "large-model", "status": {"value": status}}]})
        assert url.endswith("/health")
        assert params == {"model": "large-model"}
        health_calls += 1
        return _HttpResponse(status_code=503 if health_calls == 1 else 200)

    monkeypatch.setattr(
        "infrastructure.adapter.llamaCppAdapter.requests.post",
        lambda *args, **kwargs: _HttpResponse(payload={"success": True}),
    )
    monkeypatch.setattr("infrastructure.adapter.llamaCppAdapter.requests.get", fake_get)
    monkeypatch.setattr("infrastructure.adapter.llamaCppAdapter.time.sleep", lambda _seconds: None)

    adapter.load_model_sync("large-model")

    assert health_calls == 2
    assert adapter.currently_loaded_model == "large-model"
    assert adapter._ready_model == "large-model"
    assert shared_state["currently_loaded_model"] == "large-model"


def test_load_http_failure_raises_and_does_not_publish_model(monkeypatch):
    adapter = LlamaCppAdapter("http://localhost:8080", model_load_timeout_seconds=5)
    monkeypatch.setattr(adapter, "_sync_loaded_model_state", lambda: None)
    monkeypatch.setattr(
        "infrastructure.adapter.llamaCppAdapter.requests.post",
        lambda *args, **kwargs: _HttpResponse(status_code=500, text="load exploded"),
    )

    with pytest.raises(RuntimeError, match="HTTP 500.*load exploded"):
        adapter.load_model_sync("broken-model")

    assert adapter.currently_loaded_model is None
    assert adapter._ready_model is None


def test_terminal_router_load_failure_raises(monkeypatch):
    adapter = LlamaCppAdapter("http://localhost:8080", model_load_timeout_seconds=5)
    monkeypatch.setattr(adapter, "_sync_loaded_model_state", lambda: None)
    monkeypatch.setattr(
        "infrastructure.adapter.llamaCppAdapter.requests.post",
        lambda *args, **kwargs: _HttpResponse(payload={"success": True}),
    )
    monkeypatch.setattr(
        adapter,
        "_fetch_models",
        lambda timeout_seconds=10: [
            {
                "id": "broken-model",
                "status": {"value": "unloaded", "failed": True, "exit_code": 1},
            }
        ],
    )

    with pytest.raises(RuntimeError, match="failed while loading.*exit_code=1"):
        adapter.load_model_sync("broken-model")

    assert adapter._ready_model is None


def test_configured_model_load_timeout_is_enforced(monkeypatch):
    adapter = LlamaCppAdapter("http://localhost:8080", model_load_timeout_seconds=0.01)
    monkeypatch.setattr(adapter, "_sync_loaded_model_state", lambda: None)
    monkeypatch.setattr(
        "infrastructure.adapter.llamaCppAdapter.requests.post",
        lambda *args, **kwargs: _HttpResponse(payload={"success": True}),
    )
    monkeypatch.setattr(
        adapter,
        "_fetch_models",
        lambda timeout_seconds=10: [
            {"id": "slow-model", "status": {"value": "loading"}}
        ],
    )

    with pytest.raises(RuntimeError, match="Timed out.*slow-model.*last_status='loading'"):
        adapter.load_model_sync("slow-model")

    assert adapter.model_load_timeout_seconds == 1.0
    assert adapter._ready_model is None


def test_inference_refuses_model_without_confirmed_readiness():
    adapter = LlamaCppAdapter("http://localhost:8080")
    adapter.currently_loaded_model = "router-visible-model"
    adapter._get_model_metadata = lambda _model_name=None: {
        "id": "router-visible-model",
        "status": {"value": "unloaded"},
    }

    with pytest.raises(RuntimeError, match="Refusing inference"):
        asyncio.run(
            adapter.chat_completion_stream(
                model="router-visible-model",
                messages=[{"role": "user", "content": "hello"}],
            )
        )


def test_bounded_visual_request_options_are_forwarded_to_llama_cpp():
    adapter = LlamaCppAdapter("http://localhost:8080")
    captured = {}

    class _Completions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return "stream"

    adapter.client = SimpleNamespace(
        chat=SimpleNamespace(completions=_Completions())
    )
    adapter._require_model_ready = lambda _model: None
    schema = {"type": "json_schema", "json_schema": {"name": "observation", "schema": {"type": "object"}}}

    result = asyncio.run(
        adapter.chat_completion_stream(
            model="fast-vlm",
            messages=[{"role": "user", "content": "inspect"}],
            max_tokens=256,
            response_format=schema,
            chat_template_kwargs={"enable_thinking": False},
            request_timeout_seconds=16,
        )
    )

    assert result == "stream"
    assert captured["max_tokens"] == 256
    assert captured["response_format"] == schema
    assert captured["timeout"] == 16
    assert captured["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_default_max_tokens_is_applied_when_request_does_not_override():
    adapter = LlamaCppAdapter("http://localhost:8080", default_max_tokens=60000)
    captured = {}

    class _Completions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return "stream"

    adapter.client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    adapter._require_model_ready = lambda _model: None

    result = asyncio.run(
        adapter.chat_completion_stream(
            model="main",
            messages=[{"role": "user", "content": "hello"}],
        )
    )

    assert result == "stream"
    assert captured["max_tokens"] == 60000


def test_dropped_llama_stream_retries_the_full_prompt_without_partial_output():
    adapter = LlamaCppAdapter(
        "http://localhost:8080",
        stream_retry_attempts=3,
        stream_retry_delay_seconds=0,
    )
    attempts = {"count": 0}

    class _Stream:
        def __init__(self, items):
            self.items = list(items)

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self.items:
                raise StopAsyncIteration
            next_item = self.items.pop(0)
            if isinstance(next_item, Exception):
                raise next_item
            return next_item

    class _Completions:
        async def create(self, **_kwargs):
            attempts["count"] += 1
            if attempts["count"] == 1:
                return _Stream(["partial", httpx.ReadError("connection dropped")])
            return _Stream(["complete"])

    adapter.client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    adapter._require_model_ready = lambda _model: None

    async def exercise():
        stream = await adapter.chat_completion_stream(model="main", messages=[])
        return [item async for item in stream]

    assert asyncio.run(exercise()) == ["complete"]
    assert attempts["count"] == 2


def test_shutdown_controller_allows_existing_stream_to_finish_but_blocks_next_request():
    controller = RuntimeShutdownController()
    adapter = LlamaCppAdapter("http://localhost:8080", shutdown_controller=controller)

    class _Stream:
        def __init__(self):
            self.items = ["first", "last"]

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self.items:
                raise StopAsyncIteration
            return self.items.pop(0)

        async def close(self):
            return None

    class _Completions:
        async def create(self, **_kwargs):
            return _Stream()

    adapter.client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    adapter._require_model_ready = lambda _model: None

    async def exercise():
        stream = await adapter.chat_completion_stream(model="main", messages=[])
        assert controller.active_stream_count() == 1
        assert controller.request_interrupt() == "graceful"
        assert [item async for item in stream] == ["first", "last"]
        assert controller.active_stream_count() == 0
        with pytest.raises(ShutdownInProgress):
            await adapter.chat_completion_stream(model="main", messages=[])

    asyncio.run(exercise())


def test_force_shutdown_closes_active_llama_stream_and_http_client():
    controller = RuntimeShutdownController()
    adapter = LlamaCppAdapter("http://localhost:8080", shutdown_controller=controller)
    closed = {"stream": False, "client": False}

    class _Stream:
        async def close(self):
            closed["stream"] = True

    class _Completions:
        async def create(self, **_kwargs):
            return _Stream()

    class _Client:
        chat = SimpleNamespace(completions=_Completions())

        async def close(self):
            closed["client"] = True

    adapter.client = _Client()
    adapter._require_model_ready = lambda _model: None

    async def exercise():
        await adapter.chat_completion_stream(model="main", messages=[])
        controller.request_interrupt()
        controller.request_interrupt()
        await adapter.close_active_connections()

    asyncio.run(exercise())
    assert closed == {"stream": True, "client": True}


def test_shutdown_blocks_model_load_and_unload_before_executor(monkeypatch):
    controller = RuntimeShutdownController()
    adapter = LlamaCppAdapter("http://localhost:8080", shutdown_controller=controller)
    calls = {"load": 0, "unload": 0}

    def fake_load(*args, **kwargs):
        calls["load"] += 1

    def fake_unload(*args, **kwargs):
        calls["unload"] += 1

    monkeypatch.setattr(adapter, "load_model_sync", fake_load)
    monkeypatch.setattr(adapter, "unload_model_sync", fake_unload)
    controller.request_interrupt()

    async def exercise():
        with pytest.raises(ShutdownInProgress):
            await adapter.load_model("main")
        with pytest.raises(ShutdownInProgress):
            await adapter.unload_model()

    asyncio.run(exercise())
    assert calls == {"load": 0, "unload": 0}


def test_per_request_max_tokens_overrides_adapter_default():
    adapter = LlamaCppAdapter("http://localhost:8080", default_max_tokens=60000)
    captured = {}

    class _Completions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return "stream"

    adapter.client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    adapter._require_model_ready = lambda _model: None

    asyncio.run(
        adapter.chat_completion_stream(
            model="main",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=512,
        )
    )

    assert captured["max_tokens"] == 512


def test_async_load_does_not_block_event_loop(monkeypatch):
    adapter = LlamaCppAdapter("http://localhost:8080")

    def slow_load(_model_name, _unload_previous=True):
        import time as stdlib_time

        stdlib_time.sleep(0.05)

    monkeypatch.setattr(adapter, "load_model_sync", slow_load)

    async def exercise():
        load_task = asyncio.create_task(adapter.load_model("large-model"))
        await asyncio.sleep(0.01)
        event_loop_remained_responsive = not load_task.done()
        await load_task
        return event_loop_remained_responsive

    assert asyncio.run(exercise()) is True


def test_isolated_adapter_does_not_adopt_another_resident_model(monkeypatch):
    adapter = LlamaCppAdapter(
        "http://localhost:8080",
        isolated_model_tracking=True,
    )
    shared_state = {"currently_loaded_model": "main-model"}
    adapter.kv_state.read_shared_state = lambda: dict(shared_state)
    adapter.kv_state.update_shared_state = lambda **values: shared_state.update(values)
    monkeypatch.setattr(
        adapter,
        "_fetch_models",
        lambda: [
            {"id": "main-model", "status": {"value": "loaded"}},
            {"id": "passive-vlm", "status": {"value": "loaded"}},
        ],
    )

    assert adapter._sync_loaded_model_state() is None
    assert adapter.get_current_model() is None
    assert shared_state["currently_loaded_model"] == "main-model"


def test_isolated_adapters_unload_only_the_model_they_own(monkeypatch):
    main = LlamaCppAdapter("http://localhost:8080", isolated_model_tracking=True)
    vision = LlamaCppAdapter("http://localhost:8080", isolated_model_tracking=True)
    main.currently_loaded_model = "main-model"
    main._ready_model = "main-model"
    vision.currently_loaded_model = "passive-vlm"
    vision._ready_model = "passive-vlm"
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append((url, json))
        return _HttpResponse(payload={"success": True})

    monkeypatch.setattr("infrastructure.adapter.llamaCppAdapter.requests.post", fake_post)
    monkeypatch.setattr(main, "_wait_for_model_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(vision, "_wait_for_model_status", lambda *args, **kwargs: None)

    main.unload_model_sync()

    assert calls == [
        ("http://localhost:8080/models/unload", {"model": "main-model"})
    ]
    assert main.currently_loaded_model is None
    assert vision.currently_loaded_model == "passive-vlm"


def test_isolated_adapter_reuses_already_resident_owned_model(monkeypatch):
    adapter = LlamaCppAdapter(
        "http://localhost:8080",
        isolated_model_tracking=True,
    )
    monkeypatch.setattr(
        adapter,
        "_fetch_models",
        lambda timeout_seconds=10: [
            {"id": "main-model", "status": {"value": "loaded"}},
            {"id": "passive-vlm", "status": {"value": "loaded"}},
        ],
    )

    def fake_ready(model_name, *, deadline, started_at=None):
        adapter._set_loaded_model_state(model_name, ready=True)

    monkeypatch.setattr(adapter, "_wait_until_model_ready", fake_ready)
    monkeypatch.setattr(
        "infrastructure.adapter.llamaCppAdapter.requests.post",
        lambda *args, **kwargs: pytest.fail("resident model should not be loaded or unloaded again"),
    )

    adapter.load_model_sync("passive-vlm")

    assert adapter.currently_loaded_model == "passive-vlm"
    assert adapter._ready_model == "passive-vlm"


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
