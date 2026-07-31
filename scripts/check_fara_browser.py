"""Preflight the local visual browser and optional Fara multimodal endpoint.

Run this while the Ambient runtime is stopped:

    python scripts/check_fara_browser.py --browser-only
    python scripts/check_fara_browser.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from config import CONFIG  # noqa: E402
from infrastructure.adapter.FaraVisualBrowserAdapter import (  # noqa: E402
    BrowserSafetyPolicy,
    FaraVisualBrowserSession,
)
from infrastructure.adapter.llamaCppAdapter import LlamaCppAdapter  # noqa: E402


def _session(*, llm_provider: object, temp_root: Path) -> FaraVisualBrowserSession:
    return FaraVisualBrowserSession(
        llm_provider=llm_provider,
        profile_dir=temp_root / "profile",
        screenshot_dir=temp_root / "screenshots",
        headless=True,
        viewport_width=CONFIG.get_int("browser", "viewport_width", 1440),
        viewport_height=CONFIG.get_int("browser", "viewport_height", 900),
        max_steps=1,
        settle_ms=CONFIG.get_int("browser", "settle_ms", 700),
        search_url_template=CONFIG.get_str(
            "browser", "search_url_template", "https://duckduckgo.com/?q={query}"
        ),
        browser_channel=CONFIG.get_str("browser", "channel", ""),
        browser_executable_path=CONFIG.get_str("browser", "executable_path", ""),
        policy=BrowserSafetyPolicy(),
        screenshot_retention=False,
        logger=logging.getLogger("FaraBrowserPreflight"),
    )


async def _run(*, browser_only: bool) -> int:
    temp_root = Path(tempfile.mkdtemp(prefix="ambient-fara-preflight-"))
    provider: object
    model = CONFIG.get_model("browser_agent_model", "")
    if browser_only:
        provider = object()
    else:
        if not model:
            raise RuntimeError("[models] browser_agent_model must name Fara1.5-27B.")
        provider = LlamaCppAdapter(
            base_url=CONFIG.get_str("runtime", "api_base_url", "http://127.0.0.1:8080"),
            api_key=CONFIG.get_str("runtime", "api_key", "test"),
            model_load_timeout_seconds=CONFIG.get_float(
                "runtime", "model_load_timeout_seconds", 600.0
            ),
        )

    session = _session(llm_provider=provider, temp_root=temp_root)
    loaded = False
    try:
        await session.start()
        screenshot = await session._capture(1)
        browser_result = {
            "browser_started": True,
            "url": session._page.url,
            "screenshot_bytes": screenshot.stat().st_size,
            "viewport": [session.viewport_width, session.viewport_height],
            "dom_or_uia_exposed": False,
        }
        print(json.dumps(browser_result, indent=2))
        if browser_only:
            return 0

        await provider.load_model(model)
        loaded = True
        action = await session._request_action(
            task=(
                "This is a preflight only. Inspect the screenshot, then return a wait action "
                "without navigating, typing, clicking, or changing anything."
            ),
            model=model,
            screenshot_path=screenshot,
            current_url=session._page.url,
            step=1,
        )
        print(
            json.dumps(
                {
                    "model": model,
                    "multimodal_request_succeeded": True,
                    "parsed_action": action,
                    "action_is_valid": action.get("action") in session.ALLOWED_ACTIONS,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    finally:
        await session.cleanup()
        if loaded:
            try:
                await provider.unload_model()
            except Exception:
                logging.getLogger("FaraBrowserPreflight").exception(
                    "Fara preflight succeeded but the model could not be unloaded."
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--browser-only",
        action="store_true",
        help="Launch the local browser and capture a screenshot without calling Fara.",
    )
    args = parser.parse_args()
    try:
        return asyncio.run(_run(browser_only=args.browser_only))
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
