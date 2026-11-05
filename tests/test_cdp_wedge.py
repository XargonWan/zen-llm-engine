"""Regression tests for the unanswered-CDP-command wedge.

Live symptom (2026-10): ~19% of requests sat for the full 480 s
``total_timeout`` and ended in a forced browser reset, because zendriver awaits
a CDP reply with no deadline of its own.
"""

import asyncio
import tempfile
import time
from unittest.mock import AsyncMock

import pytest

from core.zendriver_llm_base import (
    CdpWedgedError,
    ZendriverLLMBase,
    _clear_cdp_wedge,
    _guard_send,
    raise_if_cdp_wedged,
)


@pytest.fixture(autouse=True)
def _fresh_wedge_state():
    _clear_cdp_wedge()
    yield
    _clear_cdp_wedge()


def test_guard_send_raises_when_command_never_answers(monkeypatch):
    monkeypatch.setenv("ZEN_CDP_COMMAND_TIMEOUT", "0.05")

    async def never_answers(self, cdp_obj, _is_update=False):
        await asyncio.Event().wait()

    guarded = _guard_send(never_answers)

    async def scenario():
        t0 = time.time()
        with pytest.raises(CdpWedgedError):
            await guarded(object(), iter(()))
        return time.time() - t0

    assert asyncio.run(scenario()) < 2


def test_guard_send_fails_fast_after_a_wedge_until_cleared(monkeypatch):
    monkeypatch.setenv("ZEN_CDP_COMMAND_TIMEOUT", "0.05")
    calls = 0

    async def never_answers(self, cdp_obj, _is_update=False):
        nonlocal calls
        calls += 1
        await asyncio.Event().wait()

    guarded = _guard_send(never_answers)

    async def scenario():
        with pytest.raises(CdpWedgedError):
            await guarded(object(), iter(()))
        with pytest.raises(CdpWedgedError):
            await guarded(object(), iter(()))
        return calls

    assert asyncio.run(scenario()) == 1  # the second call never reached the wire
    with pytest.raises(CdpWedgedError):
        raise_if_cdp_wedged()
    _clear_cdp_wedge()
    raise_if_cdp_wedged()


def test_guard_send_passes_results_and_errors_through(monkeypatch):
    monkeypatch.setenv("ZEN_CDP_COMMAND_TIMEOUT", "5")

    async def answers(self, cdp_obj, _is_update=False):
        return "reply"

    async def fails(self, cdp_obj, _is_update=False):
        raise ValueError("protocol error")

    assert asyncio.run(_guard_send(answers)(object(), iter(()))) == "reply"
    with pytest.raises(ValueError, match="protocol error"):
        asyncio.run(_guard_send(fails)(object(), iter(())))
    raise_if_cdp_wedged()  # an ordinary failure is not a wedge


def test_guard_send_disabled_with_zero_timeout(monkeypatch):
    monkeypatch.setenv("ZEN_CDP_COMMAND_TIMEOUT", "0")

    async def slow(self, cdp_obj, _is_update=False):
        await asyncio.sleep(0.1)
        return "late"

    assert asyncio.run(_guard_send(slow)(object(), iter(()))) == "late"


def test_guard_send_cancels_the_inner_command_when_the_caller_is_cancelled(monkeypatch):
    monkeypatch.setenv("ZEN_CDP_COMMAND_TIMEOUT", "30")
    inner_cancelled = False

    async def hangs(self, cdp_obj, _is_update=False):
        nonlocal inner_cancelled
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            inner_cancelled = True
            raise

    guarded = _guard_send(hangs)

    async def scenario():
        task = asyncio.ensure_future(guarded(object(), iter(())))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0)

    asyncio.run(scenario())
    assert inner_cancelled


def test_cdp_guard_is_installed_on_the_zendriver_connection():
    from zendriver.core.connection import Connection

    assert getattr(Connection.send, "_zen_guarded", False)


def test_wedge_error_keeps_the_detection_timeout_marker():
    err = CdpWedgedError("evaluate got no reply within 45s")
    assert isinstance(err, RuntimeError)
    assert "selenium_response_detection_timeout" in str(err)


def _engine() -> ZendriverLLMBase:
    return ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 100},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )


def test_retry_loop_resets_browser_on_wedge_without_growing_the_chunk_count():
    engine = _engine()
    engine._split_prompt_parts = 2
    calls = 0

    async def once(prompt, media=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise CdpWedgedError("evaluate got no reply within 45s")
        return "recovered"

    engine._generate_response_once = once
    engine._reset_driver = AsyncMock()
    engine._should_split_prompt = lambda prompt: True  # would otherwise chunk-retry

    assert asyncio.run(engine._generate_response_retry_loop("A" * 200)) == "recovered"
    assert engine._reset_driver.await_count == 1
    assert engine._split_prompt_parts == 2
    assert engine._navigate_on_next_attempt is True


def test_retry_loop_gives_up_after_the_last_wedged_attempt():
    engine = _engine()
    engine._generate_response_once = AsyncMock(side_effect=CdpWedgedError("stuck"))
    engine._reset_driver = AsyncMock()

    with pytest.raises(CdpWedgedError):
        asyncio.run(engine._generate_response_retry_loop("hello"))
    assert engine._generate_response_once.await_count == 5
    assert engine._reset_driver.await_count == 4

