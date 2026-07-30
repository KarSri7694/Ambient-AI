import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from application.services.ddgs_search_service import DdgsSearchService


class _FakeDDGS:
    instances = []

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.calls = []
        self.__class__.instances.append(self)

    def text(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return [
            {"title": "First", "href": "https://example.com/one", "body": "One body"},
            {"title": "Duplicate", "href": "https://example.com/one", "body": "Duplicate"},
            {
                "title": "Second",
                "url": "https://example.org/two",
                "description": "Two body",
                "provider": "Example",
            },
        ]


def test_ddgs_search_normalizes_results_and_forwards_controls():
    _FakeDDGS.instances.clear()
    service = DdgsSearchService(
        timeout_seconds=12,
        proxy="http://proxy.example:8080",
        ddgs_factory=_FakeDDGS,
    )
    result = service.search_text(
        query="ROCm optimization",
        max_results=5,
        region="in-en",
        safesearch="off",
        timelimit="m",
        page=2,
        backend="duckduckgo, brave",
    )

    assert result["ok"] is True
    assert result["count"] == 2
    assert result["results"][0] == {
        "title": "First",
        "url": "https://example.com/one",
        "snippet": "One body",
        "source": None,
    }
    assert result["results"][1]["source"] == "Example"
    client = _FakeDDGS.instances[-1]
    assert client.init_kwargs == {"proxy": "http://proxy.example:8080", "timeout": 12}
    assert client.calls[0][0] == "ROCm optimization"
    assert client.calls[0][1]["timelimit"] == "m"
    assert client.calls[0][1]["backend"] == "duckduckgo, brave"


def test_ddgs_search_rejects_invalid_arguments_without_network_call():
    _FakeDDGS.instances.clear()
    result = DdgsSearchService(ddgs_factory=_FakeDDGS).search_text(
        query="test",
        max_results=0,
        safesearch="disabled",
    )
    assert result["ok"] is False
    assert result["error"] == "invalid_request"
    assert _FakeDDGS.instances == []


def test_ddgs_search_returns_structured_failure_without_leaking_exception_text():
    class FailingDDGS:
        def __init__(self, **_kwargs):
            pass

        def text(self, *_args, **_kwargs):
            raise RuntimeError("secret proxy password")

    result = DdgsSearchService(ddgs_factory=FailingDDGS).search_text(query="test")
    assert result["ok"] is False
    assert result["error"] == "search_failed"
    assert "secret proxy password" not in result["detail"]
