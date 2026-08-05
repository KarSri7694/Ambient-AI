import json
import sys
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from application.services.capture_control_service import CaptureControlService
from application.services.passive_observer_service import PassiveObserverService
from infrastructure.runtime_log_server import RuntimeLogBuffer, create_runtime_log_app


def test_exclusions_are_seeded_once_then_loaded_from_persistent_policy(tmp_path):
    policy_path = tmp_path / "privacy" / "capture_exclusions.json"
    first = CaptureControlService(
        excluded_apps=["Signal"],
        excluded_domains=["private.example"],
        persistence_path=policy_path,
    )
    assert policy_path.exists()
    assert first.status()["persistence"]["source"] == "config_seed"

    first.set_exclusions(apps=["1Password.exe"], domains=["https://bank.example/login"])
    saved = json.loads(policy_path.read_text(encoding="utf-8"))
    assert saved["domains"] == ["bank.example"]

    restarted = CaptureControlService(
        excluded_apps=["Config Must Not Win"],
        excluded_domains=["config.invalid"],
        persistence_path=policy_path,
    )
    assert restarted.status()["persistence"]["source"] == "persisted"
    assert restarted.is_excluded(process_name=r"C:\Program Files\1Password\1Password.exe")
    assert restarted.is_excluded(domain="login.bank.example")
    assert not restarted.is_excluded(app_name="Config Must Not Win")


def test_context_matching_supports_processes_arbitrary_hosts_and_safe_title_boundaries():
    service = CaptureControlService(
        excluded_apps=["chrome", "mail"],
        excluded_domains=["example.technology", "localhost", "10.20.30.40"],
    )

    process = service.evaluate_context({"process_name": "chrome.exe", "window_title": "News"})
    assert process["excluded"] is True
    assert process["match_type"] == "application"
    assert process["matched_rule"] == "chrome"

    assert service.is_excluded(url="https://account.example.technology:8443/login")
    assert service.is_excluded(url="http://localhost:5173/dashboard")
    assert service.is_excluded(domain="10.20.30.40")
    assert not service.is_excluded(window_title="Thumbnail editor")


def test_app_exclusions_match_ui_automation_aliases_and_executable_paths():
    service = CaptureControlService(
        excluded_apps=[r"C:\Program Files\Google\Chrome\chrome.exe", "Visual Studio Code"],
    )

    assert service.is_excluded(process_name="chrome.exe", app_name="Google Chrome")
    assert service.is_excluded(
        process_name="code.exe",
        app_name="Code",
        window_title="settings.json - Visual Studio Code",
    )
    assert not service.is_excluded(app_name="Notepad", window_title="Mail thumbnail editor")


def test_domain_exclusions_canonicalize_www_and_fall_back_from_bad_hint():
    service = CaptureControlService(excluded_domains=["www.example.com"])

    assert service.is_excluded(domain="login.example.com")
    assert service.is_excluded(domain="not a hostname", url="https://www.example.com/account")


def test_queued_capture_policy_is_not_reapplied_after_the_policy_changes(tmp_path):
    control = CaptureControlService(excluded_apps=["Signal"])
    observer = PassiveObserverService(
        memory=SimpleNamespace(),
        llm_provider=SimpleNamespace(),
        screen_capture=SimpleNamespace(),
        screenshot_root=str(tmp_path / "screens"),
        capture_control=control,
    )

    live_route = observer._route_screenshot(
        similarity_score=None,
        uiat_context={"process_name": "signal.exe", "app_hint": "Signal"},
        previous_observation=None,
    )
    queued_route = observer._route_screenshot(
        similarity_score=None,
        uiat_context={
            "process_name": "signal.exe",
            "app_hint": "Signal",
            "capture_policy_applied": True,
        },
        previous_observation=None,
    )
    assert live_route == "skip"
    assert queued_route == "fast_model"


def test_capture_boundary_blocks_excluded_apps_and_domains_before_pixels(tmp_path):
    class ScreenCapture:
        def __init__(self):
            self.calls = 0

        def capture_screenshot(self, output_path):
            self.calls += 1
            Path(output_path).write_bytes(b"pixels")
            return output_path

    control = CaptureControlService(
        excluded_apps=["Visual Studio Code"],
        excluded_domains=["netflix.com"],
    )
    capture = ScreenCapture()
    observer = PassiveObserverService(
        memory=SimpleNamespace(),
        llm_provider=SimpleNamespace(),
        screen_capture=capture,
        screenshot_root=str(tmp_path / "screens"),
        capture_control=control,
    )

    assert observer.capture_screenshot(context={
        "process_name": "Code.exe",
        "app_name": "Visual Studio Code",
    }) is None
    assert observer.capture_screenshot(context={
        "process_name": "firefox.exe",
        "url": "https://www.netflix.com/browse",
    }) is None
    assert capture.calls == 0


def test_capture_boundary_rechecks_foreground_after_the_loop_context_was_collected(tmp_path):
    class ScreenCapture:
        def __init__(self):
            self.calls = 0

        def capture_screenshot(self, output_path):
            self.calls += 1
            Path(output_path).write_bytes(b"pixels")
            return output_path

    class UIAT:
        def inspect_foreground_window(self):
            return {
                "process_name": "firefox.exe",
                "window_title": "Netflix â€” Mozilla Firefox",
                "foreground_url": "https://www.netflix.com/browse",
            }

    capture = ScreenCapture()
    observer = PassiveObserverService(
        memory=SimpleNamespace(),
        llm_provider=SimpleNamespace(),
        screen_capture=capture,
        screenshot_root=str(tmp_path / "screens"),
        capture_control=CaptureControlService(excluded_domains=["netflix.com"]),
        uiat_adapter=UIAT(),
    )
    stale_context = {"process_name": "explorer.exe", "app_name": "File Explorer"}

    assert observer.capture_screenshot(context=stale_context) is None
    assert stale_context["domain"] == "www.netflix.com"
    assert capture.calls == 0


def test_firefox_accessibility_toolbar_url_reaches_domain_policy(tmp_path):
    control = CaptureControlService(excluded_domains=["127.0.0.1", "netflix.com"])
    observer = PassiveObserverService(
        memory=SimpleNamespace(),
        llm_provider=SimpleNamespace(),
        screen_capture=SimpleNamespace(),
        screenshot_root=str(tmp_path / "screens"),
        capture_control=control,
    )
    payload = {
        "process_name": "firefox.exe",
        "window_class": "MozillaWindowClass",
        "window_title": "Ambient Agent Runtime — Mozilla Firefox",
        "visible_text_summary": (
            "Navigation\nView site information\n"
            "Search with DuckDuckGo or enter address\n"
            "http://127.0.0.1:8765/inbox\nBookmark this page"
        ),
    }
    domain = observer._infer_domain_hint(payload)
    assert domain == "127.0.0.1"
    assert control.is_excluded(
        process_name=payload["process_name"],
        window_class=payload["window_class"],
        window_title=payload["window_title"],
        domain=domain,
    )

    # A filename in a non-browser window must not be mistaken for a website.
    assert observer._infer_domain_hint(
        {
            "process_name": "code.exe",
            "window_class": "Chrome_WidgetWin_1",
            "window_title": "config.example.ini - Visual Studio Code",
        }
    ) is None


def test_foreground_check_api_uses_runtime_detector_without_capturing():
    control = CaptureControlService(excluded_domains=["bank.example"])

    class Runtime:
        def check_capture_exclusions(self):
            context = {"process_name": "chrome.exe", "domain": "login.bank.example"}
            return {"ok": True, "context": context, "decision": control.evaluate_context(context)}

    client = TestClient(
        create_runtime_log_app(
            RuntimeLogBuffer(), capture_control=control, runtime_control=Runtime()
        )
    )
    response = client.post("/api/privacy/capture/exclusions/check")
    assert response.status_code == 200
    assert response.json()["decision"]["excluded"] is True
    status = client.get("/api/privacy/status").json()["capture"]
    assert status["last_context"]["domain"] == "login.bank.example"
