"""Regression tests for the "Gemini stuck for a day" incident.

What happened: the engine's login was lost, every request then burned ~8
minutes and reset the shared browser several times, the queue filled up, and --
because each reset killed Chromium before it flushed cookies -- no manual
sign-in ever survived. These tests pin the fixes:

* circuit breaker (fail fast, leave the browser alone, recover),
* login detection that does not silently assume "logged in",
* non-blocking status endpoints,
* zombie reaping.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import app as app_module
from app import app
from core import proc_utils
from core.engine_manager import EngineDescriptor, EngineManager
from core.errors import EngineUnavailableError, LoginRequiredError
from core.json_engine import JsonEngine

linux_only = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs /proc")


def _descriptor(name: str) -> EngineDescriptor:
    return EngineDescriptor(
        name=name,
        aliases=[name],
        display_name=name,
        service_url=f"https://{name}.example",
        models={"default": 1000},
        default_model="default",
        source="builtin",
        source_path="<test>",
    )


def _manager(*names: str) -> EngineManager:
    """A throwaway manager (not the singleton) with synthetic descriptors."""
    mgr = EngineManager()
    mgr._descriptors = {n: _descriptor(n) for n in names}
    mgr._alias_map = {n: n for n in names}
    mgr.engines.clear()
    mgr._health.clear()
    return mgr


class ScriptedEngine:
    """Engine whose generate_response follows a script of outcomes."""

    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0
        self._last_login_state: bool | None = None
        self._last_login_checked_at: float | None = None
        self.peek_results: list[bool | None] = []

    def get_current_model(self) -> str:
        return "default"

    async def generate_response(self, prompt, media=None, timeout=None, agent_mode=False):
        self.calls += 1
        outcome = self.outcomes.pop(0) if self.outcomes else "ok"
        if isinstance(outcome, BaseException):
            raise outcome
        if callable(outcome):
            return await outcome()
        return outcome

    async def peek_login_state(self) -> bool | None:
        return self.peek_results.pop(0) if self.peek_results else None


@pytest.fixture(autouse=True)
def _breaker_env(monkeypatch):
    monkeypatch.setenv("ZEN_BREAKER_THRESHOLD", "3")
    monkeypatch.setenv("ZEN_BREAKER_COOLDOWN", "300")
    monkeypatch.setenv("ZEN_BREAKER_MAX_COOLDOWN", "3600")


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


def test_breaker_opens_after_consecutive_failures_and_fails_fast():
    async def _run():
        mgr = _manager("eng")
        engine = ScriptedEngine([RuntimeError("boom")] * 3)
        mgr.engines["eng"] = engine
        for _ in range(3):
            with pytest.raises(RuntimeError, match="boom"):
                await mgr.enqueue("eng", "p")
        # Fourth request is rejected up front and never reaches the engine.
        with pytest.raises(EngineUnavailableError) as info:
            await mgr.enqueue("eng", "p")
        return engine.calls, info.value

    calls, err = asyncio.run(_run())
    assert calls == 3
    assert err.reason == "failures"
    assert 1 <= err.retry_after <= 300


def test_success_resets_failure_streak():
    async def _run():
        mgr = _manager("eng")
        mgr.engines["eng"] = ScriptedEngine(
            [RuntimeError("x"), RuntimeError("x"), "ok", RuntimeError("x"), RuntimeError("x")]
        )
        for expected in (RuntimeError, RuntimeError, None, RuntimeError, RuntimeError):
            if expected is None:
                await mgr.enqueue("eng", "p")
            else:
                with pytest.raises(expected):
                    await mgr.enqueue("eng", "p")
        return mgr._health["eng"]

    health = asyncio.run(_run())
    assert health.consecutive_failures == 2
    assert not health.breaker_open


def test_client_errors_do_not_trip_breaker():
    async def _run():
        mgr = _manager("eng")
        mgr.engines["eng"] = ScriptedEngine([ValueError("bad media")] * 6)
        for _ in range(6):
            with pytest.raises(ValueError):
                await mgr.enqueue("eng", "p")
        return mgr._health_for("eng")

    health = asyncio.run(_run())
    assert not health.breaker_open
    assert health.consecutive_failures == 0


def test_login_error_trips_immediately_and_fails_queued_jobs():
    """One login failure must also flush everything waiting behind it."""

    async def _run():
        mgr = _manager("eng")
        gate = asyncio.Event()

        async def _blocked():
            await gate.wait()
            raise LoginRequiredError("signed out")

        engine = ScriptedEngine([_blocked])
        mgr.engines["eng"] = engine
        first = asyncio.ensure_future(mgr.enqueue("eng", "1"))
        await asyncio.sleep(0.05)  # worker picks up job 1 and blocks
        queued = [asyncio.ensure_future(mgr.enqueue("eng", str(i))) for i in (2, 3)]
        await asyncio.sleep(0.05)
        gate.set()
        results = await asyncio.gather(first, *queued, return_exceptions=True)
        return engine.calls, results, mgr

    calls, results, mgr = asyncio.run(_run())
    assert calls == 1, "queued jobs must not be run against a signed-out engine"
    assert isinstance(results[0], LoginRequiredError)
    assert all(isinstance(r, EngineUnavailableError) and r.reason == "login" for r in results[1:])
    assert mgr._health["eng"].breaker_reason == "login"
    assert mgr.login_blocked("eng")


def test_login_breaker_stays_open_until_reset_and_never_half_opens():
    async def _run():
        mgr = _manager("eng")
        mgr.engines["eng"] = ScriptedEngine([LoginRequiredError("signed out"), "back"])
        with pytest.raises(LoginRequiredError):
            await mgr.enqueue("eng", "p")
        # Even long after any cooldown a "login" breaker admits nothing.
        mgr._health["eng"].breaker_opened_at = time.time() - 10_000
        with pytest.raises(EngineUnavailableError) as info:
            await mgr.enqueue("eng", "p")
        assert info.value.reason == "login"
        assert mgr.reset_breaker("eng") is True
        return (await mgr.enqueue("eng", "p")).text

    assert asyncio.run(_run()) == "back"


def test_half_open_trial_success_closes_breaker():
    async def _run():
        mgr = _manager("eng")
        mgr.engines["eng"] = ScriptedEngine([RuntimeError("x")] * 3 + ["recovered"])
        for _ in range(3):
            with pytest.raises(RuntimeError):
                await mgr.enqueue("eng", "p")
        h = mgr._health["eng"]
        assert h.breaker_open
        h.retry_at = time.time() - 1  # cooldown elapsed
        result = await mgr.enqueue("eng", "p")
        return result.text, h

    text, h = asyncio.run(_run())
    assert text == "recovered"
    assert not h.breaker_open and h.consecutive_failures == 0


def test_half_open_trial_failure_doubles_cooldown_and_admits_only_one():
    async def _run():
        mgr = _manager("eng")
        gate = asyncio.Event()

        async def _slow_fail():
            await gate.wait()
            raise RuntimeError("still broken")

        mgr.engines["eng"] = ScriptedEngine([RuntimeError("x")] * 3 + [_slow_fail])
        for _ in range(3):
            with pytest.raises(RuntimeError):
                await mgr.enqueue("eng", "p")
        h = mgr._health["eng"]
        first_cooldown = h.cooldown
        h.retry_at = time.time() - 1
        trial = asyncio.ensure_future(mgr.enqueue("eng", "trial"))
        await asyncio.sleep(0.05)
        # While the trial is in flight nobody else gets in.
        with pytest.raises(EngineUnavailableError):
            await mgr.enqueue("eng", "second")
        gate.set()
        with pytest.raises(RuntimeError, match="still broken"):
            await trial
        return first_cooldown, h

    first, h = asyncio.run(_run())
    assert h.breaker_open
    assert h.cooldown == first * 2


def test_assert_available_has_no_side_effects():
    mgr = _manager("eng")
    mgr._trip("eng", "failures")
    mgr._health["eng"].retry_at = time.time() - 1  # trial due
    mgr.assert_available("eng")
    mgr.assert_available("eng")  # still fine: it does not consume the trial
    assert mgr._health["eng"].trial_in_flight is False


def test_login_watcher_closes_breaker_after_two_good_peeks():
    async def _run():
        mgr = _manager("eng")
        engine = ScriptedEngine([])
        engine.peek_results = [None, True, False, True, True]
        mgr.engines["eng"] = engine
        mgr._trip("eng", "login")
        states = []
        for _ in range(5):
            await mgr._login_watch_tick()
            states.append(mgr._health["eng"].breaker_open)
        return states

    # None / True / False reset or build the streak; only two Trues in a row close it.
    assert asyncio.run(_run()) == [True, True, True, True, False]


def test_login_watcher_never_touches_a_busy_browser():
    from core import zendriver_llm_base

    async def _run():
        mgr = _manager("eng")
        engine = ScriptedEngine([])
        engine.peek_results = [True, True, True]
        mgr.engines["eng"] = engine
        mgr._trip("eng", "login")
        async with zendriver_llm_base._BROWSER_ACCESS_LOCK:
            await mgr._login_watch_tick()
            await mgr._login_watch_tick()
        return mgr._health["eng"].breaker_open, len(engine.peek_results)

    still_open, unused = asyncio.run(_run())
    assert still_open and unused == 3


def test_health_snapshot_states():
    async def _run():
        mgr = _manager("a", "b", "c")
        mgr.engines["a"] = ScriptedEngine(["ok"])
        mgr.engines["b"] = ScriptedEngine([RuntimeError("x")])
        await mgr.enqueue("a", "p")
        with pytest.raises(RuntimeError):
            await mgr.enqueue("b", "p")
        mgr._trip("c", "login")
        return {e["name"]: e for e in mgr.health_snapshot()}

    snap = asyncio.run(_run())
    assert snap["a"]["state"] == "ok" and snap["a"]["last_success_at"]
    assert snap["b"]["state"] == "degraded" and snap["b"]["consecutive_failures"] == 1
    assert snap["c"]["state"] == "down" and snap["c"]["breaker"]["reason"] == "login"


# ---------------------------------------------------------------------------
# Retry loop: login lost must stop the reset/retry storm
# ---------------------------------------------------------------------------


def _bare_engine(tmp_path, *, allow_unlogged: bool) -> JsonEngine:
    return JsonEngine(
        {
            "name": "probe",
            "service_url": "https://probe.example",
            "allow_unlogged": allow_unlogged,
            "models": {"default": 1000, "unlogged": 1000},
        },
        profile_dir=str(tmp_path),
    )


def _wire_failing_attempts(engine: JsonEngine, logged_out: bool) -> MagicMock:
    engine._generate_response_once = AsyncMock(
        side_effect=RuntimeError("selenium_response_detection_timeout: nothing appeared")
    )
    engine._reset_driver = AsyncMock()
    engine._login_lost = AsyncMock(return_value=logged_out)
    return engine._reset_driver


def test_retry_loop_stops_immediately_when_login_required_engine_is_signed_out(tmp_path):
    engine = _bare_engine(tmp_path, allow_unlogged=False)
    reset = _wire_failing_attempts(engine, logged_out=True)
    with pytest.raises(LoginRequiredError):
        asyncio.run(engine._generate_response_retry_loop("hello"))
    assert engine._generate_response_once.await_count == 1
    reset.assert_not_awaited()  # the browser is left alone


def test_retry_loop_gives_guest_capable_engines_one_more_attempt(tmp_path):
    engine = _bare_engine(tmp_path, allow_unlogged=True)
    reset = _wire_failing_attempts(engine, logged_out=True)
    with pytest.raises(LoginRequiredError):
        asyncio.run(engine._generate_response_retry_loop("hello"))
    assert engine._generate_response_once.await_count == 2
    # The guest retry must not relaunch the browser: the operator may be signing in.
    reset.assert_not_awaited()


def test_retry_loop_unchanged_when_page_is_not_logged_out(tmp_path):
    engine = _bare_engine(tmp_path, allow_unlogged=False)
    _wire_failing_attempts(engine, logged_out=False)
    with pytest.raises(RuntimeError, match="detection_timeout") as info:
        asyncio.run(engine._generate_response_retry_loop("hello"))
    assert not isinstance(info.value, LoginRequiredError)
    assert engine._generate_response_once.await_count == 5


def test_check_login_state_answers_from_cache_when_browser_is_busy(tmp_path):
    from core import zendriver_llm_base

    engine = _bare_engine(tmp_path, allow_unlogged=False)
    engine._last_login_state = False
    engine._initialized = True
    engine.driver = MagicMock()

    async def _run():
        async with zendriver_llm_base._BROWSER_ACCESS_LOCK:
            return await asyncio.wait_for(engine.check_login_state(), timeout=1.0)

    state = asyncio.run(_run())
    assert state == {"logged_in": False, "login_state": "unlogged", "busy": True}


# ---------------------------------------------------------------------------
# Login detection (JsonEngine)
# ---------------------------------------------------------------------------

_GUEST_PAGE_CONFIG: dict[str, Any] = {
    "name": "guest",
    "service_url": "https://svc.example",
    "models": {"default": 1000},
    "login_detection": {
        "url_prefix": "https://svc.example",
        "url_deny_keywords": ["signin"],
        # The XPath matches nothing on purpose: this is the incident, where the
        # legacy XPath rule did not match the live "Sign in" link.
        "login_button_xpath": "//button[@id='does-not-exist']",
        "login_button_css": ["a[aria-label='Sign in']"],
        "authenticated_css_selectors": ["a[aria-label*='Account']"],
    },
}


def _login_tab(url: str, *, marker_visible: bool, auth_visible: bool = False, ready: str = "complete"):
    tab = MagicMock()
    tab.url = url
    tab.query_selector_all = AsyncMock(return_value=[])
    tab.xpath = AsyncMock(return_value=[])
    tab.get = AsyncMock()

    async def _evaluate(expression, *a, **kw):
        if expression == "window.location.href":
            return url
        if expression == "document.readyState":
            return ready
        if "Sign in" in expression:
            return marker_visible
        if "Account" in expression:
            return auth_visible
        return None

    tab.evaluate = AsyncMock(side_effect=_evaluate)
    return tab


def _guest_engine(tmp_path) -> JsonEngine:
    return JsonEngine(_GUEST_PAGE_CONFIG, profile_dir=str(tmp_path))


def test_ensure_logged_in_detects_logged_out_page_via_css_marker(tmp_path):
    engine = _guest_engine(tmp_path)
    tab = _login_tab("https://svc.example/app", marker_visible=True)
    assert asyncio.run(engine._ensure_logged_in(tab)) is False


def test_ensure_logged_in_fallback_is_no_longer_silent(tmp_path, caplog):
    engine = _guest_engine(tmp_path)
    tab = _login_tab("https://svc.example/app", marker_visible=False)
    with caplog.at_level("WARNING", logger="json_engine"):
        assert asyncio.run(engine._ensure_logged_in(tab)) is True
        assert asyncio.run(engine._ensure_logged_in(tab)) is True
    warnings = [r for r in caplog.records if "login state undetermined" in r.getMessage()]
    assert len(warnings) == 1, "warn once per streak, not on every check"


def test_login_check_waits_for_a_late_rendered_marker(tmp_path):
    """The SPA draws its sign-in button after load; a single early check missed it."""
    import copy

    cfg = copy.deepcopy(_GUEST_PAGE_CONFIG)
    cfg["login_detection"]["login_settle_seconds"] = 3
    engine = JsonEngine(cfg, profile_dir=str(tmp_path))
    tab = _login_tab("https://svc.example/app", marker_visible=False)
    seen = {"checks": 0}
    original = tab.evaluate.side_effect

    async def _late(expression, *a, **kw):
        if "Sign in" in expression:
            seen["checks"] += 1
            return seen["checks"] >= 3  # appears on the third look
        return await original(expression, *a, **kw)

    tab.evaluate = AsyncMock(side_effect=_late)
    assert asyncio.run(engine._ensure_logged_in(tab)) is False
    assert seen["checks"] >= 3


def test_login_check_without_settle_still_checks_once(tmp_path):
    engine = _guest_engine(tmp_path)  # no login_settle_seconds: no added latency
    tab = _login_tab("https://svc.example/app", marker_visible=False)
    started = time.monotonic()
    assert asyncio.run(engine._ensure_logged_in(tab)) is True
    assert time.monotonic() - started < 0.4


def test_peek_logged_out_requires_declared_marker_on_service_page(tmp_path):
    engine = _guest_engine(tmp_path)
    assert asyncio.run(
        engine._peek_logged_out(_login_tab("https://svc.example/app", marker_visible=True))
    )
    assert not asyncio.run(
        engine._peek_logged_out(_login_tab("https://svc.example/app", marker_visible=False))
    )
    # Off the service host there is no *positive* evidence, even with the marker.
    assert not asyncio.run(
        engine._peek_logged_out(_login_tab("https://accounts.example/signin", marker_visible=True))
    )


def test_peek_login_state_never_navigates(tmp_path):
    engine = _guest_engine(tmp_path)
    engine._initialized = True

    def peek(tab):
        engine.driver = tab
        return asyncio.run(engine.peek_login_state())

    out = _login_tab("https://svc.example/app", marker_visible=True)
    assert peek(out) is False
    auth = _login_tab("https://svc.example/app", marker_visible=False, auth_visible=True)
    assert peek(auth) is True
    loaded = _login_tab("https://svc.example/app", marker_visible=False)
    assert peek(loaded) is True
    loading = _login_tab("https://svc.example/app", marker_visible=False, ready="loading")
    assert peek(loading) is None
    mid_login = _login_tab("https://accounts.example/signin", marker_visible=False)
    assert peek(mid_login) is None
    for tab in (out, auth, loaded, loading, mid_login):
        tab.get.assert_not_awaited()


def test_gemini_engine_declares_css_login_markers():
    import json
    from pathlib import Path

    cfg = json.loads((Path(__file__).parent.parent / "engines" / "gemini.json").read_text())
    markers = cfg["login_detection"]["login_button_css"]
    # Verified against the live guest page: the aria-label anchor sits in a hidden
    # Google bar (0 px wide), so only the avatar sign-in button is actually visible.
    assert "gem-button.mavatar-sign-in-button" in markers


# ---------------------------------------------------------------------------
# Zombie reaping
# ---------------------------------------------------------------------------


# Popen.__del__ reaps an unreferenced child, which would hide the zombie from the
# test. Keep the objects alive -- exactly the situation in production, where the
# dropped-but-still-referenced Browser held the Popen of a SIGKILLed Chromium.
_KEEP_ALIVE: list[subprocess.Popen] = []


def _make_zombie() -> int:
    proc = subprocess.Popen(["sleep", "60"])
    _KEEP_ALIVE.append(proc)
    proc.kill()
    deadline = time.time() + 5
    while time.time() < deadline and proc_utils.process_state(proc.pid) != "Z":
        time.sleep(0.02)
    assert proc_utils.process_state(proc.pid) == "Z"
    return proc.pid


@linux_only
def test_reap_pid_collects_a_killed_child():
    pid = _make_zombie()
    assert proc_utils.reap_pid(pid, timeout=1.0)
    assert proc_utils.process_state(pid) is None


@linux_only
def test_reap_zombie_children_and_count():
    pids = [_make_zombie() for _ in range(3)]
    assert set(pids) <= set(proc_utils.list_zombies())
    assert proc_utils.reap_zombie_children() >= 3
    assert not set(pids) & set(proc_utils.list_zombies())


@linux_only
def test_orphan_sweep_skips_zombies_and_reaps_them(monkeypatch):
    """pgrep -f keeps matching zombies; the sweep must reap, not re-"kill", them."""
    pid = _make_zombie()
    mgr = _manager("eng")

    def fake_run(args, **kwargs):
        out = f"{pid}\n" if args and args[0] == "pgrep" and "-P" not in args else ""
        return subprocess.CompletedProcess(args, 0, stdout=out, stderr="")

    monkeypatch.setattr("core.engine_manager.subprocess.run", fake_run)
    monkeypatch.setattr(mgr, "_protected_pids", lambda: set())
    assert mgr._cleanup_orphans() == 0
    assert proc_utils.process_state(pid) is None


def test_rss_mb_reports_a_number_on_linux():
    if sys.platform.startswith("linux"):
        assert proc_utils.rss_mb() > 0
    else:
        assert proc_utils.rss_mb() is None


# ---------------------------------------------------------------------------
# HTTP surface
# ---------------------------------------------------------------------------


class _ApiEngine:
    def __init__(self) -> None:
        self._last_login_state = None
        self._last_login_checked_at = None
        self.peek = None

    def get_current_model(self) -> str:
        return "default"

    async def generate_response(self, prompt, media=None, timeout=None, agent_mode=False):
        return "hello"

    async def peek_login_state(self):
        return self.peek

    async def check_login_state(self):  # must NOT be used while a login breaker is open
        raise AssertionError("check_login_state may navigate; peek must be used instead")

    start_state = "logged"

    async def start_login_flow(self):
        return {"logged_in": self.start_state == "logged", "login_state": self.start_state}


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setattr("app.inc_requests", lambda: None)
    monkeypatch.setattr("app.inc_responses", lambda: None)
    monkeypatch.setattr("app.inc_errors", lambda: None)
    monkeypatch.setattr("app.log_prompt", lambda *a, **kw: None)
    monkeypatch.setattr("app.inc_media_sent", lambda *a, **kw: None)
    app_module.rate_limit_store.clear()

    mgr = EngineManager.get()
    saved = (mgr._descriptors, mgr._alias_map, dict(mgr.engines), mgr._health)
    mgr._descriptors = {"chatgpt": _descriptor("chatgpt")}
    mgr._alias_map = {"chatgpt": "chatgpt"}
    mgr._health = {}
    mgr._job_queues.clear()
    mgr._queue_workers.clear()
    engine = _ApiEngine()
    mgr.engines.clear()
    mgr.engines["chatgpt"] = engine
    try:
        yield TestClient(app), mgr, engine
    finally:
        mgr._descriptors, mgr._alias_map, saved_engines, mgr._health = saved
        mgr.engines.clear()
        mgr.engines.update(saved_engines)
        mgr._job_queues.clear()
        mgr._queue_workers.clear()


def _chat(client: TestClient) -> Any:
    return client.post(
        "/v1/chat/completions",
        json={"model": "chatgpt", "messages": [{"role": "user", "content": "hi"}]},
    )


def test_chat_returns_503_with_retry_after_while_breaker_open(api):
    client, mgr, _engine = api
    mgr._trip("chatgpt", "login")
    response = _chat(client)
    assert response.status_code == 503
    assert int(response.headers["Retry-After"]) >= 1
    body = response.json()
    assert body["error"]["type"] == "engine_unavailable"
    assert body["error"]["reason"] == "login"
    assert body["error"]["engine"] == "chatgpt"


def test_streaming_request_is_rejected_before_the_stream_starts(api):
    client, mgr, _engine = api
    mgr._trip("chatgpt", "failures")
    response = client.post(
        "/v1/chat/completions",
        json={"model": "chatgpt", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 503
    assert response.headers["content-type"].startswith("application/json")


def test_recover_endpoint_reopens_the_engine(api):
    client, mgr, _engine = api
    mgr._trip("chatgpt", "login")
    assert _chat(client).status_code == 503
    response = client.post("/api/engines/chatgpt/recover")
    assert response.json() == {"status": "ok", "engine": "chatgpt", "was_open": True}
    assert _chat(client).status_code == 200
    assert client.post("/api/engines/nope/recover").status_code == 404


def test_status_endpoint_reports_breaker_and_process_vitals(api):
    client, mgr, _engine = api
    ok = client.get("/api/status").json()
    assert ok["status"] == "ok"
    assert {"rss_mb", "zombie_processes", "rss_warning"} <= set(ok["process"])
    assert {"launches", "last_launch_at", "pid"} <= set(ok["browser"])
    assert client.get("/api/health").status_code == 200

    mgr._trip("chatgpt", "login")
    status = client.get("/api/status")
    assert status.status_code == 200  # always 200: it is a report, not a probe
    assert status.json()["status"] == "down"
    assert status.json()["engines"][0]["breaker"]["reason"] == "login"
    health = client.get("/api/health")
    assert health.status_code == 503
    assert health.json()["status"] == "down"


def test_status_does_not_wait_for_the_browser_lock(api):
    from core import zendriver_llm_base

    async def _hold_lock_and_ask():
        async with zendriver_llm_base._BROWSER_ACCESS_LOCK:
            return await asyncio.wait_for(app_module.api_status(), timeout=1.0)

    response = asyncio.run(_hold_lock_and_ask())
    assert response.status_code == 200


def test_login_state_uses_read_only_peek_while_login_breaker_open(api):
    client, mgr, engine = api
    mgr._trip("chatgpt", "login")
    engine.peek = False
    state = client.get("/login/chatgpt/state").json()
    assert state == {"logged_in": False, "login_state": "unlogged", "breaker": "open"}
    assert mgr.login_blocked("chatgpt")

    engine.peek = True
    state = client.get("/login/chatgpt/state").json()
    assert state["login_state"] == "logged"
    assert not mgr.login_blocked("chatgpt"), "an explicit logged-in report closes a login breaker"


def test_explicit_login_does_not_close_a_failures_breaker(api):
    _client, mgr, _engine = api
    mgr._trip("chatgpt", "failures")
    mgr.note_login_state("chatgpt", True)
    assert mgr._health["chatgpt"].breaker_open


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def test_log_lines_are_bounded(monkeypatch):
    """A single 1.6 MB record (a DOM dumped into a StopIteration) must not reach the logs."""
    import logging

    monkeypatch.setattr(app_module, "_LOG_MAX_LINE", 100)
    record = logging.LogRecord("t", logging.ERROR, "f", 1, "x" * 5000, None, None)
    out = app_module._BoundedFormatter("%(message)s").format(record)
    assert len(out) < 200
    assert "truncated 4900 chars" in out

    monkeypatch.setattr(app_module, "_LOG_MAX_LINE", 0)  # 0 disables the limit
    assert len(app_module._BoundedFormatter("%(message)s").format(record)) == 5000


# ---------------------------------------------------------------------------
# Browser launch flags and shutdown order (session persistence)
# ---------------------------------------------------------------------------

# Flags that make Chromium behave unlike a normal desktop browser. The owner
# asked for a stock-behaving browser, and these are what bot detection keys on.
_FORBIDDEN_FLAGS = (
    "--disable-web-security",
    "--allow-running-insecure-content",
    "--disable-extensions",
    "--disable-plugins",
    "--disable-features",
    "--disable-gpu",
    "--disable-background-networking",
    "--disable-component-update",
    "--disable-renderer-backgrounding",
    "--disable-background-timer-throttling",
    "--disable-backgrounding-occluded-windows",
    "--disable-hang-monitor",
    "--disable-ipc-flooding-protection",
    "--disable-back-forward-cache",
    "--no-pings",
    "--js-flags",
    "--webrtc-ip-handling-policy",
    "--force-webrtc-ip-handling-policy",
)


def test_chromium_launches_like_a_normal_browser(tmp_path):
    engine = _bare_engine(tmp_path, allow_unlogged=False)
    args = engine._chromium_args(9222)
    for flag in _FORBIDDEN_FLAGS:
        assert not any(a == flag or a.startswith(flag + "=") for a in args), flag
    # What the container and the session genuinely need.
    for needed in ("--no-sandbox", "--disable-dev-shm-usage", "--restore-last-session",
                   "--password-store=basic", "--remote-debugging-port=9222"):
        assert needed in args
    assert f"--user-data-dir={tmp_path}" in args
    assert args[-1] == "about:blank"


class _FakeConnection:
    closed = False

    def __init__(self) -> None:
        self.sent: list[Any] = []

    async def send(self, command):
        self.sent.append(command)


class _FakeProc:
    """Popen stand-in that 'exits by itself' after a number of polls."""

    pid = 999_999  # not a child of ours: reap_pid() treats that as already gone

    def __init__(self, exit_after_polls: int) -> None:
        self._left = exit_after_polls
        self.returncode: int | None = None

    def poll(self):
        if self._left <= 0:
            self.returncode = 0
            return 0
        self._left -= 1
        return None


def _fake_browser(proc) -> MagicMock:
    browser = MagicMock()
    browser._process = proc
    browser._process_pid = proc.pid
    browser.connection = _FakeConnection()
    return browser


def test_stop_asks_chromium_to_quit_and_waits_without_signalling(monkeypatch):
    """Regression: zendriver's stop() sent SIGTERM right after Browser.close, which
    cut Chromium's shutdown short before the cookie flush, so a login made seconds
    earlier was lost on every 'graceful' reset."""
    from core import zendriver_llm_base as base

    signals: list[Any] = []
    monkeypatch.setattr(base.os, "kill", lambda *a: signals.append(a))
    browser = _fake_browser(_FakeProc(exit_after_polls=4))

    graceful = asyncio.run(base._stop_browser_with_timeout(browser, timeout=5.0))

    assert graceful is True
    assert signals == [], "must not signal a browser that is quitting by itself"
    assert len(browser.connection.sent) == 1  # Browser.close, and only that


def test_stop_escalates_to_sigterm_then_sigkill_for_a_hung_browser():
    from core import zendriver_llm_base as base

    proc = subprocess.Popen(
        ["sh", "-c", "trap '' TERM; while :; do sleep 1; done"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(0.3)  # let the shell install its trap
    browser = _fake_browser(proc)
    browser._process_pid = proc.pid
    try:
        graceful = asyncio.run(base._stop_browser_with_timeout(browser, timeout=0.3, term_grace=0.3))
        assert graceful is False
        assert proc.poll() is not None, "a browser that ignores SIGTERM must still be killed"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_zendriver_stop_is_no_longer_used_for_shutdown():
    """Guard against reintroducing Browser.stop() (SIGTERM-after-close, see above)."""
    import ast
    import inspect

    from core import zendriver_llm_base as base

    tree = ast.parse(inspect.getsource(base))
    calls = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "stop"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in ("browser", "_shared_browser")
    ]
    assert calls == [], f"Browser.stop() called at lines {calls}"


def test_pause_endpoint_blocks_requests_until_recover(api):
    client, mgr, _engine = api
    assert _chat(client).status_code == 200
    paused = client.post("/api/engines/chatgpt/pause").json()
    assert paused["paused"] is True
    blocked = _chat(client)
    assert blocked.status_code == 503 and blocked.json()["error"]["reason"] == "login"
    assert client.post("/api/engines/chatgpt/recover").json()["was_open"] is True
    assert _chat(client).status_code == 200
    assert client.post("/api/engines/nope/pause").status_code == 404


def test_starting_a_login_pauses_the_engine_and_a_logged_in_result_resumes_it(api):
    client, mgr, engine = api
    engine.start_state = "unlogged"
    assert client.post("/login/chatgpt").json()["login_state"] == "unlogged"
    assert mgr.login_blocked("chatgpt"), "requests must not disturb a manual sign-in"
    assert _chat(client).status_code == 503

    engine.start_state = "logged"
    client.post("/login/chatgpt")
    assert not mgr.login_blocked("chatgpt")
    assert _chat(client).status_code == 200
