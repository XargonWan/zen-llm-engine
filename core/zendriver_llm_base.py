"""ZendriverLLMBase — engine-agnostic browser automation core built on zendriver.

zendriver drives a real, unmodified Chrome/Chromium instance directly over the
Chrome DevTools Protocol (CDP) — there is no separate chromedriver binary to
patch/re-patch on every launch (unlike the previous undetected_chromedriver
implementation this module replaces). Everything here is native asyncio; no
worker thread pool is used anywhere in the request path.
"""

from __future__ import annotations

import asyncio
import glob
import json
import logging
import math
import mimetypes
import os
import re
import shutil
import signal
import socket
import subprocess
import tempfile
import time
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import zendriver as zd
from zendriver.core.keys import KeyEvents, KeyModifiers, SpecialKeys

from core.agent_protocol import AGENT_TAIL_MARKER
from core import debug_mode, proc_utils
from core.errors import LoginRequiredError

logger = logging.getLogger("zendriver_llm_base")

# Browser launch bookkeeping, surfaced by GET /api/status. A healthy deployment
# launches the shared Chromium once per many hours; hundreds of launches per day
# means something is resetting it in a loop (which also destroys any login the
# operator is completing by hand, since Chromium flushes cookies lazily).
_BROWSER_LAUNCHES = 0
_LAST_BROWSER_LAUNCH_AT: Optional[float] = None


def get_browser_stats() -> Dict[str, Any]:
    """Return launch counters for the shared browser (no browser access)."""
    return {
        "launches": _BROWSER_LAUNCHES,
        "last_launch_at": _LAST_BROWSER_LAUNCH_AT,
        "pid": get_shared_browser_pid(),
    }

# Matches a bare error line that carries a numeric failure code, e.g.
# "Something went wrong (1076)", "Error 1076" or a lone parenthesised
# "(1076)". Requires either an explicit error keyword next to the code or the
# code to be parenthesised on its own, so a normal short reply ending in a
# number (e.g. "The answer is 1200") is not misclassified.
_NUMERIC_ERROR_CODE_RE = re.compile(
    r"(?:\berror\b\s*#?\s*\d{3,5})|(?:^\(\d{3,5}\)$)",
    re.IGNORECASE,
)

# Serialises browser creation across concurrently-instantiated engine objects.
# asyncio.Lock (not threading.Lock) — nothing runs in a worker thread pool
# anymore, everything is genuinely async end-to-end.
_BROWSER_INIT_LOCK = asyncio.Lock()

# Shared Browser instance — all engine instances reuse the same browser
# process to avoid profile-dir lock conflicts and preserve login sessions
# across engines.
_shared_browser: Optional[zd.Browser] = None

# Serialises actual USE of the shared browser (navigating, typing, clicking,
# waiting for a response) across every engine, not just within one engine's
# own per-engine FIFO queue. EngineManager gives each engine its own queue
# and worker task, which only serialises requests to the SAME engine --
# nothing previously stopped two different engines' workers from driving the
# one shared Tab at the same time. Observed live: switching from a
# still-running Copilot request to ChatGPT mid-flight let both coroutines
# type/click/navigate the same tab concurrently, corrupting both requests and
# eventually forcing a nuclear browser reset. Acquired only by the small
# number of true external entry points (generate_response, start_login_flow,
# check_login_state) -- internal helpers they call must NOT also acquire it
# (asyncio.Lock is not reentrant).
_BROWSER_ACCESS_LOCK = asyncio.Lock()


def _profile_dir_default() -> str:
    return os.getenv("CHROMIUM_PROFILE_DIR", "/config/.config/chromium-synth")


def _browser_pid(browser: zd.Browser) -> Optional[int]:
    proc = getattr(browser, "_process", None)
    if proc is not None:
        pid = getattr(proc, "pid", None)
        if pid:
            return pid
    return getattr(browser, "_process_pid", None)


def _free_port() -> int:
    """An unused localhost TCP port for the CDP endpoint."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def get_shared_browser_pid() -> Optional[int]:
    """Return the PID of the currently active shared browser, if any.

    Used by :meth:`EngineManager._cleanup_orphans` so its periodic orphan
    sweep never kills the browser actually in use just because it has been
    alive longer than the orphan-age threshold — a long-lived shared browser
    is the intended, normal state now that SELENIUM_SESSION_HARD_TIMEOUT
    defaults to hours, not the 5-minute cycle it used to be.
    """
    return _browser_pid(_shared_browser) if _shared_browser is not None else None


def _kill_process_tree(pid: int) -> None:
    """SIGKILL *pid* and its direct children.

    Unlike the previous implementation, this is only ever called with the
    exact PID of the Chrome process zendriver itself spawned and is holding a
    handle to (``Browser._process``) — never a system-wide pattern match — so
    there is no risk of reaping an unrelated process that merely happens to
    have "chromium" in its command line.
    """
    try:
        subprocess.run(
            ["pkill", "-9", "-P", str(pid)],
            check=False, capture_output=True, timeout=2,
        )
    except Exception:
        pass
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except Exception:
        pass
    # The browser is our direct child (zendriver keeps it as a Popen). Nobody
    # calls wait() on a SIGKILLed one, so without this it stays a zombie
    # forever -- 167 of them were found after a few days of resets.
    proc_utils.reap_pid(pid, timeout=2.0)


async def _dispose_browser(browser: Optional[zd.Browser]) -> None:
    """Release what ``Browser.stop()`` would have, after a hard kill.

    The nuclear paths SIGKILL the process and drop the reference, but leave the
    CDP websocket/listener tasks behind; with ~1000 resets in a day that is
    memory that is never returned. Closing the connection lets the Browser,
    its Tabs and their cached DOM nodes be garbage-collected.
    """
    if browser is not None:
        connection = getattr(browser, "connection", None)
        try:
            if connection is not None and not connection.closed:
                await asyncio.wait_for(connection.aclose(), timeout=3.0)
        except Exception as exc:
            logger.debug("[zendriver] dispose: closing CDP connection failed: %s", exc)
    await asyncio.to_thread(proc_utils.trim_memory)


def _process_exited(proc: Optional[subprocess.Popen], pid: Optional[int]) -> bool:
    """True once the browser process is gone (or was never known)."""
    if proc is not None:
        return proc.poll() is not None
    if pid:
        return proc_utils.process_state(pid) in (None, "Z")
    return True


async def _wait_for_exit(
    proc: Optional[subprocess.Popen], pid: Optional[int], timeout: float
) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max(0.0, timeout)
    while not _process_exited(proc, pid):
        if loop.time() >= deadline:
            return False
        await asyncio.sleep(0.1)
    return True


async def _stop_browser_with_timeout(
    browser: zd.Browser, timeout: float = 10.0, term_grace: float = 5.0
) -> bool:
    """Quit Chromium the way a user closing the window does, and let it finish.

    Order matters, and it is what keeps a manual sign-in alive:

    1. ``Browser.close`` over CDP, then wait up to *timeout* seconds for the
       process to exit **by itself**. Measured: Chromium exits in ~0.3 s this way
       and has already flushed its cookie store (persistent and session cookies
       both survive an immediate relaunch).
    2. Only if it does not, SIGTERM and a short wait, then SIGKILL.

    ``zendriver``'s own ``Browser.stop()`` is deliberately not used: it sends
    SIGTERM right after ``Browser.close``, which interrupts Chromium's orderly
    shutdown before the cookie flush -- cookies set in the last ~30 s were lost
    on every "graceful" reset (measured: both cookies gone, and stop() took 3.3 s,
    i.e. its SIGTERM-then-kill timeout).

    Returns True if Chromium quit on its own, False if it had to be signalled.
    """
    proc: Optional[subprocess.Popen] = getattr(browser, "_process", None)
    pid = _browser_pid(browser)
    connection = getattr(browser, "connection", None)

    try:
        if connection is not None and not connection.closed:
            await asyncio.wait_for(connection.send(zd.cdp.browser.close()), timeout=3.0)
    except Exception as exc:
        # Expected when Chromium drops the socket while quitting.
        logger.debug("[zendriver] Browser.close: %s", exc)

    graceful = await _wait_for_exit(proc, pid, timeout)
    if not graceful and pid:
        logger.warning(
            "[zendriver] Chromium did not quit within %ss of Browser.close — sending SIGTERM", timeout,
        )
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except Exception as exc:
            logger.debug("[zendriver] SIGTERM failed: %s", exc)
        if not await _wait_for_exit(proc, pid, term_grace):
            logger.warning("[zendriver] Chromium ignored SIGTERM — force killing")
            await asyncio.to_thread(_kill_process_tree, pid)
    if pid:
        await asyncio.to_thread(proc_utils.reap_pid, pid, 1.0)
    await _dispose_browser(browser)
    return graceful


def _remove_profile_lock_files(profile_dir: str) -> None:
    for name in ("SingletonLock", "SingletonCookie", "SingletonSocket"):
        path = os.path.join(profile_dir, name)
        try:
            # SingletonLock is a symlink to "<hostname>-<pid>", a target that
            # never exists as a real path -- os.path.exists() follows
            # symlinks and returns False for a dangling one, so a
            # exists()-then-remove() guard here silently never removes it
            # (observed live: a lock left by a container's previous instance,
            # a different hostname, blocked every relaunch after a rebuild).
            # os.remove() itself doesn't care whether the target exists.
            os.remove(path)
        except FileNotFoundError:
            pass
        except Exception:
            pass


async def shutdown_shared_driver() -> None:
    """Gracefully stop the shared browser and clean up its profile locks.

    Called once by :meth:`EngineManager.stop_all` during application
    shutdown. Uses the graceful path (``browser.stop()``, falling back to a
    hard kill of the known PID only if that hangs) — never an unconditional
    SIGKILL — so Chrome gets a chance to flush its cookie/session stores.
    """
    global _shared_browser
    async with _BROWSER_INIT_LOCK:
        browser = _shared_browser
        _shared_browser = None
    if browser is not None:
        graceful = await _stop_browser_with_timeout(browser, timeout=10.0)
        if not graceful:
            _remove_profile_lock_files(_profile_dir_default())
    logger.info("[zendriver] Shared browser shut down")


async def force_kill_session() -> None:
    """Immediately SIGKILL the shared browser session without trying ``stop()``.

    This is the "nuclear" option for when the browser is completely frozen.
    It nullifies ``_shared_browser`` under the init lock, sends SIGKILL to the
    exact known Chrome process (and its direct children), and removes profile
    lock files so the next launch is clean.

    Engine instances remain in memory but their ``driver``/tab reference is
    stale — the next request will automatically re-initialise the browser.
    """
    global _shared_browser
    logger.warning("[zendriver] force_kill_session invoked — SIGKILL mode")
    async with _BROWSER_INIT_LOCK:
        browser = _shared_browser
        _shared_browser = None

    pid = _browser_pid(browser) if browser is not None else None
    if pid:
        await asyncio.to_thread(_kill_process_tree, pid)
        await _dispose_browser(browser)
    else:
        # No known PID (browser never started, or already gone) — nothing
        # targeted to kill. We deliberately do NOT fall back to a system-wide
        # pattern-matched pkill here: that was the old behaviour and it risked
        # killing unrelated processes sharing a substring like "chrome".
        logger.debug("[zendriver] force_kill_session: no known browser PID to kill")

    _remove_profile_lock_files(_profile_dir_default())
    logger.info("[zendriver] force_kill_session complete")


class ZendriverLLMBase:
    def __init__(
        self,
        service_url: str,
        model_limits_map: Dict[str, int],
        default_model: str,
        headless: Optional[bool] = None,
        profile_dir: Optional[str] = None,
        allow_unlogged: bool = False,
    ):
        self.service_url = service_url
        self.model_limits_map = model_limits_map
        self.default_model = default_model
        self.allow_unlogged = allow_unlogged
        # Holds the zendriver Tab currently used by this engine — the page-
        # level handle equivalent of the old Selenium `self.driver`. Kept
        # under the same attribute name so app.py's existing
        # `engine.driver` / `getattr(engine, "driver", None)` plumbing (login
        # state checks, the kill-session endpoint, the debug page-html
        # endpoint) keeps working with a minimal diff.
        self.driver: Optional[zd.Tab] = None
        self.media_config: dict[str, Any] = {}
        self.paid_account_selector: Optional[str] = None

        if headless is None:
            env_headless = os.getenv("CHROMIUM_HEADLESS", "0")
            try:
                self.headless = bool(int(env_headless))
            except Exception:
                self.headless = False
        else:
            self.headless = headless
        self._initialized = False

        self.profile_dir = profile_dir or _profile_dir_default()
        os.makedirs(self.profile_dir, exist_ok=True)
        logger.info(f"[zendriver] Chrome profile_dir={self.profile_dir}")

        self._last_login_state: Optional[bool] = None
        self._last_login_checked_at: Optional[float] = None

        # Selector lists used by the generate flow — override in subclasses.
        self.prompt_area_selectors: list[str] = [
            "textarea",
            "div[contenteditable='true']",
        ]
        self.send_button_selectors: list[str] = [
            "button[type='submit']",
            "button[aria-label*='Send']",
        ]
        self.response_area_selectors: list[str] = [
            ".assistant-message",
            "div.markdown",
        ]
        self.stop_selectors: list[str] = [
            "button[aria-label*='Stop']",
            "[data-testid='stop-button']",
        ]
        self.accept_button_selectors: list[str] = []
        self.limit_selectors: list[str] = []
        # CSS selectors that match a transient error banner/toast/snackbar the web
        # UI shows when a generation fails (e.g. "Something went wrong (1076)").
        # Such errors are typically rendered outside the response area, so the
        # normal response-text detection never sees them and the request would
        # otherwise stall until timeout. Populated from the JSON "error_indicators"
        # key; empty by default to keep the core engine-agnostic.
        self.error_indicator_selectors: list[str] = []
        # CSS selectors whose matching elements must never be clicked as send button
        self.send_button_blacklist: list[str] = []

        # Cloudflare CAPTCHA challenge detectors
        self.captcha_challenge_selectors: list[str] = [
            "iframe#cf-chl-widget-ezspn",
            "iframe[src*='challenges.cloudflare.com/cdn-cgi/challenge-platform']",
        ]

        # Selector cache: remember the last working selector to try it first
        self._cached_prompt_selector: Optional[str] = None
        self._cached_send_selector: Optional[str] = None
        # Set by _post_send_check when a genuine cross-host redirect is seen so
        # the retry loop can fail fast on deterministic auth redirects.
        self._last_send_offsite_redirect: bool = False

        # Prompt chunking: split prompts that exceed the model char limit.
        # This is a safety cap — the actual number of parts is computed
        # dynamically from prompt length / model limit and never exceeds this.
        self._split_prompt_parts: int = max(2, int(os.getenv("SELENIUM_SPLIT_PROMPT_PARTS", "10")))
        self._skip_split_for_next: bool = False

        # Per-engine response timeout override (seconds). Set by JsonEngine from the
        # JSON config key "response_max_wait". None means use the built-in default.
        self._response_max_wait: int | None = None

        # Set to True when _wait_for_response observes the stop-button during the
        # last generation attempt. Used by the retry loop to decide whether to
        # reset the browser on a detection-timeout: if the model was actively
        # generating (slow thinking model) we skip the reset; if the stop-button
        # was never seen (stuck/dead session) we reset as before.
        self._generation_was_active: bool = False

        # Set per-request by generate_response(): when True the request is part
        # of the agentic (tool-calling) lane and short JSON replies must not be
        # discarded by the page-state junk filter.
        self._agent_mode: bool = False

        # Per-engine total timeout override (seconds). Set by JsonEngine from
        # the JSON config key "total_timeout". None means use the computed default.
        self._total_timeout: int | None = None

        # Per-engine session hard timeout (seconds): maximum wall-clock time
        # a single session (browser instance) can live before being recycled.
        # Set by JsonEngine from the JSON config key "session_hard_timeout".
        # None means use the SELENIUM_SESSION_HARD_TIMEOUT env default.
        self._session_hard_timeout: int | None = None

        # Some engines (Gemini) work more reliably when response detection uses
        # stable text instead of comparing against prior baseline text.
        self._use_baseline_comparison: bool = True

        # Some web chats stop accepting messages after the first exchange in a
        # conversation: the send button is clicked but the prompt is never
        # submitted. Such engines must open a fresh chat for every request.
        # Set by JsonEngine from the JSON config key "fresh_chat_per_request".
        self._fresh_chat_per_request: bool = False

        # One-shot flag raised when an attempt fails: the page it ran on is
        # suspect, so the next attempt navigates to a fresh chat instead of
        # retrying in place on the same, possibly blocked, conversation.
        self._navigate_on_next_attempt: bool = False

        # Per-engine "silent freeze" threshold (seconds): how long the stop button
        # may stay visible with no response-text activity before the page is
        # considered stuck. None means use the SELENIUM_SILENT_FREEZE_THRESHOLD
        # env default.
        self._silent_freeze_threshold: float | None = None

        # Silent freeze recovery: track page refresh attempts before browser reset
        self._page_refresh_attempts: int = 0
        self._max_page_refresh_attempts: int = int(os.getenv("SELENIUM_MAX_PAGE_REFRESH", "2"))

        # Timestamp of the last cookie save — used to rate-limit periodic saves.
        self._last_cookie_save: float = 0.0
        self._cookie_save_interval: int = int(os.getenv("COOKIE_SAVE_INTERVAL", "300"))

        # Whether this instance has already restored its cookies into the
        # shared browser. Reset on stop/reset so a fresh restore happens
        # on the next use.
        self._cookies_restored: bool = False

        # Optional prefix prepended to the prompt when media items are present.
        self._vision_prompt_prefix: str = ""

        # Optional prefix prepended to the prompt when NO media items are present.
        self._inline_response_prefix: str = (
            "IMPORTANT: Always respond inline, do NOT use canvas, documents, or "
            "any separate UI mode. Write the full response directly in this chat "
            "as plain text. Do NOT activate canvas, artifacts, or separate "
            "documents. The response must be complete and self-contained within "
            "this chat window.\n\n"
        )

        # Wall-clock timestamp when the current shared browser was created —
        # drives the session-hard-timeout watchdog.
        self._driver_start_time: Optional[float] = None

    def get_supported_models(self) -> list[str]:
        return list(self.model_limits_map.keys())

    def get_current_model(self) -> str:
        # Return 'unlogged' when the engine is not logged in, supports it, and has this model.
        # NOTE: uses the cached login state (see is_user_logged_in below) rather
        # than forcing a live browser check — this method is called
        # synchronously from app.py/engine_manager.py request handlers that run
        # directly on the event loop, so it cannot itself await a CDP round trip.
        if (
            not self.is_user_logged_in()
            and self.allow_unlogged
            and "unlogged" in self.model_limits_map
        ):
            return "unlogged"
        return self.default_model

    def _get_model_limit(self, model_name: str) -> int:
        model_name = model_name.lower().strip()
        if model_name in self.model_limits_map:
            return self.model_limits_map[model_name]
        if "default" in self.model_limits_map:
            return self.model_limits_map["default"]
        return 10000

    def get_interface_limits(self) -> dict[str, Any]:
        return {
            "max_prompt_chars": self._get_model_limit(self.get_current_model()),
            "model_name": self.get_current_model(),
        }

    def is_user_logged_in(self) -> bool:
        """Return the last known login state without touching the browser.

        This is intentionally synchronous and cache-only (see
        ``refresh_login_state`` for the coroutine that actually talks to the
        page): it is called from synchronous request handlers in app.py and
        core/engine_manager.py that are not themselves coroutines, so it
        cannot await a live CDP check. Every place in the async request flow
        that used to call the old (synchronous, live-checking) Selenium
        version now calls ``await self.refresh_login_state()`` instead, which
        performs the real check and updates the cache read here.
        """
        return bool(self._last_login_state)

    def _locate_chromium_binary(self) -> Optional[str]:
        possible = [
            "/usr/bin/chromium",
            "/usr/bin/chromium-browser",
            "/usr/bin/google-chrome",
            "/usr/bin/google-chrome-stable",
            "/opt/google/chrome/chrome",
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
            "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
        ]
        for path in possible:
            if os.path.exists(path):
                logger.info(f"[zendriver] Found Chromium at: {path}")
                return path
        logger.warning("[zendriver] Chromium binary not found in common locations")
        return None

    def _get_chromium_major_version(self, chromium_binary: Optional[str] = None) -> Optional[int]:
        """Diagnostic only — zendriver talks CDP directly and has no
        chromedriver binary to version-pin, so this no longer gates any
        launch decision (unlike the old ``version_main`` uc.Chrome kwarg)."""
        binary = chromium_binary or self._locate_chromium_binary()
        if not binary:
            return None
        try:
            result = subprocess.run(
                [binary, "--version"], capture_output=True, text=True, timeout=5,
            )
            version_str = result.stdout.strip()
            for part in version_str.split():
                if "." in part:
                    try:
                        major = int(part.split(".")[0])
                        if major > 50:
                            logger.info(f"[zendriver] Chromium major version: {major}")
                            return major
                    except ValueError:
                        continue
        except Exception as e:
            logger.warning(f"[zendriver] Could not get Chromium version: {e}")
        return None

    def _chromium_args(self, port: int) -> list[str]:
        """Command-line flags for Chromium: a normal browser, plus what a container needs.

        Deliberately minimal. This used to launch through zendriver's own
        ``Config``, which adds ~20 automation-style flags on top of ours
        (``--disable-web-security``, ``--disable-extensions``, ``--no-pings``,
        ``--disable-component-update``, ``--disable-background-networking``,
        ``--disable-features=...site-per-process...``, ``--disable-gpu``,
        throttling/hang-monitor switches, a WebRTC policy, ...). They change how
        pages behave and are exactly what a site's bot detection looks for; the
        owner wants this browser to behave like their desktop one.

        What remains, and why:

        * ``--user-data-dir`` / ``--remote-debugging-port`` / ``--remote-allow-origins``:
          the profile and the CDP channel the engine drives it through.
        * ``--no-sandbox`` and ``--disable-dev-shm-usage``: required in the container
          (no user namespaces, 64 MB /dev/shm). Invisible to websites.
        * ``--password-store=basic``: keeps the cookie-encryption key identical on
          every launch (autodetecting a keyring could change it and make every
          cookie undecryptable = logged out). Invisible to websites.
        * ``--restore-last-session``: Chrome's "continue where you left off", which
          also keeps session cookies across restarts.
        * ``--no-first-run`` / ``--no-default-browser-check`` /
          ``--disable-session-crashed-bubble``: suppress UI prompts only.
        * ``--start-maximized``: a normal window size instead of a fixed 1280x900.
        """
        args = [
            f"--user-data-dir={self.profile_dir}",
            f"--remote-debugging-port={port}",
            "--remote-allow-origins=*",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--password-store=basic",
            "--restore-last-session",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-session-crashed-bubble",
            "--start-maximized",
        ]
        if self.headless:
            args.append("--headless=new")
        args.append("about:blank")
        return args

    def _spawn_chromium(self, chromium_binary: str, port: int) -> subprocess.Popen[bytes]:
        logger.info("[zendriver] Launching Chromium: %s %s", chromium_binary, " ".join(self._chromium_args(port)))
        return subprocess.Popen(
            [chromium_binary, *self._chromium_args(port)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    async def _attach_browser(
        self, proc: subprocess.Popen[bytes], port: int, timeout: float = 30.0
    ) -> zd.Browser:
        """Connect zendriver to the Chromium we just started and make it own *proc*.

        ``user_data_dir`` is passed so zendriver treats the profile as custom and
        never deletes it. ``_process`` is set by hand because zendriver does not
        know the process when attached: ``Browser.stopped`` and our PID helpers
        read it.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        last: Optional[Exception] = None
        while loop.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"Chromium exited during startup (exit code {proc.returncode})")
            try:
                browser = await zd.Browser.create(
                    config=zd.Config(user_data_dir=self.profile_dir, host="127.0.0.1", port=port)
                )
            except Exception as exc:
                last = exc
                await asyncio.sleep(0.3)
                continue
            browser._process = proc
            browser._process_pid = proc.pid
            return browser
        raise RuntimeError(f"Could not attach to Chromium on port {port}: {last!r}")

    def _ensure_clean_exit_preference(self) -> None:
        """Tell Chrome the previous session ended cleanly.

        Selenium's ``prefs`` experimental option (``profile.exit_type: Normal``,
        ``profile.exited_cleanly: True``) has no zendriver Config equivalent, so
        this writes the same two keys directly into
        ``<profile_dir>/Default/Preferences`` before the browser starts —
        merging into any existing file rather than clobbering it, since that
        file also holds real user profile state (extensions, autofill, etc.)
        once a login has happened.
        """
        try:
            default_dir = os.path.join(self.profile_dir, "Default")
            os.makedirs(default_dir, exist_ok=True)
            prefs_path = os.path.join(default_dir, "Preferences")
            data: dict[str, Any] = {}
            if os.path.exists(prefs_path):
                try:
                    with open(prefs_path, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                except Exception:
                    data = {}
            profile_section = data.get("profile")
            if not isinstance(profile_section, dict):
                profile_section = {}
            profile_section["exit_type"] = "Normal"
            profile_section["exited_cleanly"] = True
            data["profile"] = profile_section
            with open(prefs_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
        except Exception as exc:
            logger.warning("[zendriver] Could not write clean-exit preference: %s", exc)

    def _cleanup_chromium_remnants(self) -> None:
        """Best-effort pre-launch cleanup of stale processes/lock files.

        Only relevant when a previous run crashed without releasing the
        profile directory lock. Runs before a browser exists, so pattern
        matching on the profile dir (not an exact PID) is the only option
        here — this is a narrow, one-time safety net, not the routine
        shutdown/kill path (which targets the exact known PID; see
        shutdown_shared_driver/force_kill_session above).
        """
        try:
            pattern = self.profile_dir
            subprocess.run(
                ["pkill", "-15", "-f", pattern], check=False, capture_output=True, timeout=2,
            )
            time.sleep(0.3)
            subprocess.run(
                ["pkill", "-9", "-f", pattern], check=False, capture_output=True, timeout=2,
            )

            temp_dir = tempfile.gettempdir()
            lock_patterns = [
                os.path.join(temp_dir, ".org.chromium.Chromium.*"),
                os.path.join(temp_dir, "uc_*", "SingletonLock"),
            ]
            for lock_pat in lock_patterns:
                for lock_file in glob.glob(lock_pat):
                    try:
                        os.remove(lock_file)
                    except Exception:
                        pass

            if os.path.exists(self.profile_dir):
                _remove_profile_lock_files(self.profile_dir)
        except Exception as e:
            logger.warning(f"[zendriver] Error during pre-launch cleanup: {e}")

    # ------------------------------------------------------------------ cookie persistence
    #
    # Login state is NOT persisted through these methods — they are kept as
    # no-op hooks (and still called from the usual places) so future engines
    # can opt back in without re-plumbing call sites. Persistence instead
    # relies entirely on Chrome's native `--user-data-dir` profile
    # (`self.profile_dir`, normally a mounted volume — see CHROMIUM_PROFILE_DIR),
    # exactly like a normal desktop browser: its cookie/IndexedDB/localStorage
    # stores are written in a crash-resilient format (SQLite WAL) that
    # survives an unclean exit reasonably well on its own. A prior version of
    # this file added a second layer on top -- a periodic CDP-based snapshot
    # of every cookie, reapplied on the next driver creation -- but that was
    # removed: replaying an old snapshot of a short-lived, rotating
    # anti-replay cookie (e.g. Google's __Secure-1PSIDTS) can itself look like
    # session hijacking to the site's own security systems and trigger the
    # very logout it was meant to prevent (observed live against Gemini).
    # zendriver exposes `browser.cookies.get_all()/set_all()/save()/load()`,
    # which would make rebuilding that mechanism trivial — do NOT use them for
    # this purpose. For native persistence to work, Chrome must still be
    # allowed to exit normally (browser.stop()) so it flushes those stores to
    # disk — never SIGKILL it directly outside of the documented nuclear path.

    def _cookie_path(self) -> str:
        """Return the file path for persisted cookies of this engine."""
        engine_name = getattr(self, "ENGINE_NAME", "default")
        return os.path.join(self.profile_dir, f"cookies_{engine_name}.json")

    def _save_cookies(self) -> None:
        """No-op — see the module note above cookie persistence."""
        pass

    def _restore_cookies(self) -> None:
        """No-op — see the module note above cookie persistence."""
        self._cookies_restored = True

    def _maybe_save_cookies(self) -> None:
        """No-op — see the module note above cookie persistence."""
        pass

    # ------------------------------------------------------------------ browser lifecycle

    async def _is_browser_alive(self, browser: zd.Browser) -> bool:
        if browser is None:
            return False
        try:
            if browser.stopped:
                return False
        except Exception:
            return False
        main_tab = browser.main_tab
        if main_tab is None:
            return False
        try:
            await asyncio.wait_for(main_tab.evaluate("1"), timeout=3.0)
            return True
        except Exception:
            return False

    async def _init_browser(self) -> zd.Tab:
        """Initialize or reuse the shared Browser + this engine's Tab.

        All engine instances share a single Chrome process to prevent
        profile-dir lock conflicts and to preserve a single login session
        across engines.
        """
        global _shared_browser, _BROWSER_LAUNCHES, _LAST_BROWSER_LAUNCH_AT

        if self.driver is not None and _shared_browser is not None and self.driver.browser is _shared_browser:
            return self.driver

        async with _BROWSER_INIT_LOCK:
            if self.driver is not None and _shared_browser is not None and self.driver.browser is _shared_browser:
                return self.driver

            if _shared_browser is not None:
                if await self._is_browser_alive(_shared_browser):
                    self.driver = _shared_browser.main_tab
                    self._initialized = True
                    logger.info("[zendriver] Reusing shared browser")
                    return self.driver
                logger.warning("[zendriver] Shared browser is dead, creating a new one")
                await _stop_browser_with_timeout(_shared_browser, timeout=5.0)
                _shared_browser = None

            logger.info("[zendriver] Initializing browser...")
            await asyncio.to_thread(self._cleanup_chromium_remnants)
            await asyncio.to_thread(self._ensure_clean_exit_preference)

            chromium_binary = self._locate_chromium_binary() or "/usr/bin/chromium"
            await asyncio.to_thread(self._get_chromium_major_version, chromium_binary)  # diagnostics only

            max_retries = 3
            last_err: Optional[Exception] = None
            browser: Optional[zd.Browser] = None
            for attempt in range(max_retries):
                proc: Optional[subprocess.Popen[bytes]] = None
                try:
                    logger.info(f"[zendriver] Browser start attempt {attempt + 1}/{max_retries}")
                    port = _free_port()
                    proc = await asyncio.to_thread(self._spawn_chromium, chromium_binary, port)
                    browser = await self._attach_browser(proc, port)
                    break
                except Exception as err:
                    last_err = err
                    logger.warning(f"[zendriver] Attempt {attempt + 1}/{max_retries} failed: {err}")
                    if proc is not None and proc.poll() is None:
                        await asyncio.to_thread(_kill_process_tree, proc.pid)
                    await asyncio.to_thread(self._cleanup_chromium_remnants)
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)

            if browser is None:
                raise RuntimeError(f"Browser initialization failed: {last_err!r}") from last_err

            # Clean up extra tabs, mirroring the old multi-window cleanup.
            extra_tabs = browser.tabs[1:]
            for extra in extra_tabs:
                try:
                    await extra.close()
                except Exception:
                    pass

            self.driver = browser.main_tab or await browser.get("about:blank")
            self._initialized = True
            self._driver_start_time = time.time()
            _shared_browser = browser
            _BROWSER_LAUNCHES += 1
            _LAST_BROWSER_LAUNCH_AT = time.time()
            logger.info("[zendriver] Browser initialized successfully (shared)")
            return self.driver

    # ------------------------------------------------------------------ readiness

    async def _ensure_ready(self) -> None:
        global _shared_browser
        if self.driver is not None and (
            _shared_browser is None or self.driver.browser is not _shared_browser
        ):
            logger.warning("[zendriver] Invalidating stale tab reference because shared browser was reset")
            self.driver = None
            self._initialized = False
            self._cookies_restored = False

        if not self._initialized or self.driver is None:
            await self._init_browser()
        if not self._cookies_restored:
            self._restore_cookies()
            self._cookies_restored = True

    async def _ensure_logged_in(self, tab: zd.Tab) -> bool:
        # Implemented by subclasses.
        raise NotImplementedError()

    async def refresh_login_state(self) -> bool:
        """Perform a live login check and update the cached state.

        This is the coroutine equivalent of the old synchronous
        ``is_user_logged_in()``. It is the one that actually talks to the
        page; ``is_user_logged_in()`` stays synchronous and cache-only so the
        many synchronous call sites in app.py/engine_manager.py keep working
        (see the docstring on ``is_user_logged_in``).
        """
        if not self._initialized or self.driver is None:
            return self.is_user_logged_in()
        try:
            logged = await self._ensure_logged_in(self.driver)
            self._last_login_state = logged
            self._last_login_checked_at = time.time()
            if logged:
                self._maybe_save_cookies()
            return logged
        except Exception as e:
            logger.warning(f"Unable to determine login state: {e}")
            return False

    async def _peek_logged_out(self, tab: zd.Tab) -> bool:
        """True only on *positive* evidence that the current page is logged out.

        Read-only by contract: must never navigate, reload or click, because the
        operator may be mid sign-in on this very tab. Subclasses with a
        ``login_detection`` config override it; the default has no evidence.
        """
        return False

    async def peek_login_state(self) -> Optional[bool]:
        """Non-navigating login probe: True / False, or None when unknown.

        Used by the background login watcher while the engine's circuit breaker
        is open. Unlike :meth:`check_login_state` it neither takes the browser
        lock nor navigates, so it cannot disturb a manual sign-in.
        """
        return None

    async def _login_lost(self) -> bool:
        """Bounded, never-raising wrapper around :meth:`_peek_logged_out`."""
        tab = self.driver
        if tab is None:
            return False
        try:
            lost = await asyncio.wait_for(self._peek_logged_out(tab), timeout=5.0)
        except Exception as exc:
            logger.debug("[zendriver] login peek failed: %s", exc)
            return False
        if lost:
            self._last_login_state = False
            self._last_login_checked_at = time.time()
        return bool(lost)

    async def start_login_flow(self, timeout: int = 60) -> dict[str, Any]:
        """Open the service URL in the browser (non-blocking) and return login state."""
        try:
            async with _BROWSER_ACCESS_LOCK:
                await self._ensure_ready()
                assert self.driver is not None, "_ensure_ready() must have initialized the tab"
                await self.driver.get(self.service_url)
                await asyncio.sleep(2)
                logged = await self.refresh_login_state()
            state = "logged" if logged else "unlogged"
            return {"logged_in": logged, "login_state": state}
        except Exception as e:
            logger.error(f"start_login_flow error: {e}")
            return {"logged_in": False, "login_state": "unknown", "error": str(e)}

    async def check_login_state(self) -> dict[str, Any]:
        """Return current login state without navigating."""
        try:
            if not self._initialized or self.driver is None:
                logged = bool(self._last_login_state)
                state = "logged" if logged else "unlogged"
                return {"logged_in": logged, "login_state": state}

            # A live check needs the shared tab, which an in-flight request owns
            # (possibly for many minutes if it is wedged). Waiting made this
            # endpoint hang for as long as the queue was stuck -- exactly when
            # someone asks "is it logged in?". Answer from the cached state and
            # say so instead.
            if _BROWSER_ACCESS_LOCK.locked():
                last = self._last_login_state
                state = "unknown" if last is None else ("logged" if last else "unlogged")
                return {"logged_in": bool(last), "login_state": state, "busy": True}

            # Waits for any in-flight generate_response/start_login_flow on
            # another engine to finish before touching the shared tab -- a
            # live check can navigate (see _ensure_logged_in's url_prefix
            # step), which must never happen mid-generation on another
            # engine's request.
            async with _BROWSER_ACCESS_LOCK:
                logged = await self.refresh_login_state()
            state = "logged" if logged else "unlogged"
            return {"logged_in": logged, "login_state": state}
        except Exception as e:
            logger.error(f"check_login_state error: {e}")
            return {"logged_in": False, "login_state": "unknown", "error": str(e)}

    # ------------------------------------------------------------------ generate_response entrypoint

    async def generate_response(
        self, prompt: str, media: list[Any] | None = None, timeout: int | None = None,
        agent_mode: bool = False,
    ) -> str:
        """Send prompt and optional media to the LLM service and return the response text.

        Parameters
        ----------
        timeout:
            Maximum number of seconds to wait for the full generation cycle
            (navigation + input + response). ``None`` uses the default of
            300 s (5 min), overridable via the ``SELENIUM_TOTAL_TIMEOUT``
            environment variable.
        agent_mode:
            When True the request is part of the agentic (tool-calling) lane.
            Structured JSON replies must NOT be discarded as page-state junk,
            so ``_looks_like_response_text_junk`` is relaxed for this request.
        """
        self._agent_mode = bool(agent_mode)
        if timeout is not None:
            total_timeout = int(timeout)
        elif self._total_timeout is not None:
            total_timeout = self._total_timeout
        elif os.getenv("SELENIUM_TOTAL_TIMEOUT"):
            total_timeout = int(os.getenv("SELENIUM_TOTAL_TIMEOUT"))
        else:
            effective_max_wait = self._response_max_wait or 120
            total_timeout = effective_max_wait + 180

        # Acquired *before* the timeout-guarded work starts (not inside the
        # asyncio.wait_for below): waiting for another engine's turn on the
        # shared browser must not eat into this request's own total_timeout,
        # and must never let this request's timeout handler force-reset a
        # browser another engine is still legitimately using. See the lock's
        # module-level docstring for why this exists.
        async with _BROWSER_ACCESS_LOCK:
            try:
                result = await asyncio.wait_for(
                    self._generate_response_retry_loop(prompt, media), timeout=total_timeout,
                )
            except asyncio.TimeoutError:
                logger.error(
                    "[zendriver] generate_response timed out after %ds — force-resetting browser",
                    total_timeout,
                )
                # There is no separate OS thread to be "stuck" anymore: the only
                # thing that can happen here is that the coroutine driving the
                # browser did not return within the outer timeout (e.g. a CDP
                # call is wedged). Cancelling it already happened implicitly when
                # asyncio.wait_for raised; force-reset (nuclear kill of the known
                # PID, no graceful quit attempt — that would hang for the same
                # reason) unblocks the browser for the next request.
                try:
                    await asyncio.wait_for(self._force_reset_driver(), timeout=15)
                except Exception as exc:
                    logger.warning("[zendriver] _force_reset_driver error: %s", exc)
                raise RuntimeError(
                    f"selenium_response_detection_timeout: generate_response "
                    f"exceeded total timeout of {total_timeout}s"
                )

            try:
                self._maybe_save_cookies()
            except Exception:
                pass
            return result

    # ------------------------------------------------------------------ session health

    def _is_dead_session(self, exc: Exception) -> bool:
        """Return True if *exc* signals a crashed/dead browser session."""
        msg = str(exc).lower()
        return any(
            marker in msg
            for marker in (
                "connection refused",
                "errno 111",
                "errno 113",
                "failed to establish a new connection",
                "max retries exceeded",
                "no such session",
                "invalid session id",
                "no such window",
                "target window already closed",
                "web view not found",
                "connection closed",
                "connectionclosed",
                "not yet started",
                "no targets",
                "target closed",
                "browser not yet started",
                # zendriver's CDP transport raises this (via the `websockets`
                # library) when the underlying TCP connection to the browser
                # is gone without a proper WS close handshake -- i.e. the
                # browser process itself died. Observed live: every retry hit
                # the same dead connection and failed in ~1ms because this
                # message wasn't recognised as a dead session, so the browser
                # was never recreated.
                "no close frame received or sent",
                "close frame",
            )
        )
        # NOTE: "could not find node" / stale-node ProtocolExceptions are
        # deliberately NOT treated as a dead session — that is a per-element
        # staleness condition (see _is_stale_element_error), not a dead
        # browser, and tab.query_selector(_all) already retries it once
        # internally before ever propagating it here.

    def _is_redirect_stall(self, exc: Exception) -> bool:
        return "redirect-stall:" in str(exc).lower()

    def _is_offsite_redirect_stall(self, exc: Exception) -> bool:
        return "redirect-stall (off-site)" in str(exc).lower()

    def _is_response_detection_timeout(self, exc: Exception) -> bool:
        return "selenium_response_detection_timeout" in str(exc)

    def _is_page_refresh_required(self, exc: Exception) -> bool:
        return "page_refresh_required" in str(exc)

    def _is_stale_element_error(self, exc: Exception) -> bool:
        """zendriver's equivalent of Selenium's StaleElementReferenceException.

        A resolved backend_node_id that no longer exists in the DOM surfaces
        as a ProtocolException whose message mentions the node/object could
        not be found (query_selector/query_selector_all already retry once
        internally on this exact condition; this is for the operations that
        don't, e.g. click()/apply() on an Element handle we cached earlier).

        Matched by message rather than requiring ``isinstance(exc,
        ProtocolException)``: unlike Selenium, where StaleElementReference is
        always that one specific exception class, zendriver has no dedicated
        staleness exception type, so classification has to be structural
        (any exception, from any layer, whose message says the node/object is
        gone) rather than type-based.
        """
        msg = str(exc).lower()
        return "could not find node" in msg or "no node" in msg or "object not found" in msg

    def _is_engine_error_response(self, text: str) -> bool:
        """Return True if *text* looks like an LLM web UI error rather than a valid response."""
        if not text:
            return False
        stripped = text.strip()
        error_patterns = [
            "something went wrong",
            "qualcosa è andato storto",
            "an error occurred",
            "si è verificato un errore",
            "internal server error",
            "errore interno del server",
            "request failed",
            "richiesta fallita",
            "service unavailable",
            "servizio non disponibile",
        ]
        lower = stripped.lower()
        for pattern in error_patterns:
            if pattern in lower:
                logger.warning("[zendriver] Detected engine error response: %r", stripped[:200])
                return True
        if len(stripped) <= 80 and _NUMERIC_ERROR_CODE_RE.search(stripped):
            logger.warning("[zendriver] Detected numeric error-code response: %r", stripped[:200])
            return True
        return False

    async def _force_reset_driver(self) -> None:
        """Force-kill the browser without trying a graceful stop.

        Used after a total-timeout when the coroutine driving the browser is
        wedged. Skips the polite stop() call (which would also hang) and goes
        straight to SIGKILL of the known process.
        """
        global _shared_browser
        logger.warning("[zendriver] Force-resetting browser (skipping graceful stop)…")
        async with _BROWSER_INIT_LOCK:
            browser = _shared_browser
            self.driver = None
            _shared_browser = None
            self._initialized = False
            self._cookies_restored = False
            self._page_refresh_attempts = 0
            self._driver_start_time = None
        pid = _browser_pid(browser) if browser is not None else None
        if pid:
            await asyncio.to_thread(_kill_process_tree, pid)
            await _dispose_browser(browser)
        _remove_profile_lock_files(self.profile_dir)

    async def _reset_driver(self) -> None:
        """Kill the existing (dead) browser and reset state so _ensure_ready re-inits."""
        global _shared_browser
        logger.warning("[zendriver] Resetting dead browser session…")
        async with _BROWSER_INIT_LOCK:
            browser = _shared_browser
            self.driver = None
            _shared_browser = None
        if browser is not None:
            await _stop_browser_with_timeout(browser, timeout=5.0)
        self._initialized = False
        self._cookies_restored = False
        self._page_refresh_attempts = 0
        self._driver_start_time = None
        await asyncio.to_thread(self._cleanup_chromium_remnants)

    # ------------------------------------------------------------------ element helpers

    async def _query_all(self, tab: zd.Tab, selector: str) -> list[Any]:
        """CSS-selector "find elements" with Selenium find_elements semantics:
        an empty result is normal, never an exception.

        IMPORTANT: this deliberately uses ``tab.query_selector_all`` (the
        zero-wait DOM primitive), NOT ``tab.select_all``. Despite its
        docstring suggesting it "returns []" like Selenium's find_elements,
        the real installed zendriver 0.16.0 source shows ``select_all`` loops
        internally and *raises* ``asyncio.TimeoutError`` if nothing matches
        within its timeout — i.e. it behaves like Selenium's
        WebDriverWait+presence_of, not find_elements. Every "is this present"
        check in this file needs the true zero-wait primitive.
        """
        try:
            return await tab.query_selector_all(selector)
        except Exception:
            return []

    async def _xpath_all(self, tab: zd.Tab, xpath: str, timeout: float = 2.5) -> list[Any]:
        try:
            return await tab.xpath(xpath, timeout=timeout)
        except Exception:
            return []

    async def _element_is_displayed(self, element: Any) -> bool:
        try:
            pos = await element.get_position()
            return pos is not None
        except Exception:
            return False

    async def _element_is_enabled(self, element: Any) -> bool:
        try:
            attrs = element.attrs
            if attrs.get("disabled") is not None:
                return False
            if (attrs.get("aria-disabled") or "").lower() == "true":
                return False
            return True
        except Exception:
            return True

    async def _element_text(self, element: Any) -> str:
        """innerText equivalent (rendered text) — always re-resolved live via
        CDP, so unlike the cached `.text`/`.text_all` properties this cannot
        go stale between polling iterations."""
        try:
            result = await element.apply("(e) => e.innerText")
            return (result or "").strip() if isinstance(result, str) else ""
        except Exception:
            return ""

    async def _element_text_content(self, element: Any) -> str:
        try:
            result = await element.apply("(e) => e.textContent")
            return (result or "").strip() if isinstance(result, str) else ""
        except Exception:
            return ""

    async def _element_attr(self, element: Any, name: str) -> str:
        try:
            value = element.attrs.get(name)
            return value if isinstance(value, str) else ""
        except Exception:
            return ""

    async def _is_captcha_present(self, tab: zd.Tab) -> bool:
        for selector in self.captcha_challenge_selectors:
            els = await self._query_all(tab, selector)
            if els:
                logger.debug(f"[zendriver] Captcha selector matched: {selector}")
                return True
        return False

    async def _is_limit_present(self, tab: zd.Tab) -> bool:
        for selector in self.limit_selectors:
            els = await self._query_all(tab, selector)
            for el in els:
                if await self._element_is_displayed(el):
                    logger.debug(f"[zendriver] Limit selector matched: {selector}")
                    return True
        return False

    async def _get_error_indicator_text(self, tab: zd.Tab) -> Optional[str]:
        for selector in self.error_indicator_selectors:
            els = await self._query_all(tab, selector)
            for el in els:
                if not await self._element_is_displayed(el):
                    continue
                text = await self._element_text(el)
                logger.warning(
                    "[zendriver] Error indicator matched selector %r: %r", selector, text[:200],
                )
                return text or "engine error indicator detected"
        return None

    async def _current_url(self, tab: zd.Tab) -> str:
        """Best-effort current URL. Prefers a live JS read over `tab.url`
        (which reflects the last CDP TargetInfoChanged event and can lag
        briefly right after a navigation) for the redirect-stall checks that
        need precision; falls back to the cached property on failure."""
        try:
            result = await tab.evaluate("window.location.href")
            if isinstance(result, str) and result:
                return result
        except Exception:
            pass
        return tab.url or ""

    async def _wait_for_page_ready(self, tab: zd.Tab, timeout: float = 30.0) -> bool:
        deadline = time.time() + timeout
        try:
            await asyncio.wait_for(
                tab.wait_for_ready_state(until="complete", timeout=timeout), timeout=timeout + 1,
            )
            logger.debug("[zendriver] Page readyState complete")
        except Exception:
            logger.debug("[zendriver] Page did not reach readyState complete")
            return False

        remaining = max(5.0, deadline - time.time())
        poll_end = time.time() + remaining
        while time.time() < poll_end:
            for sel in self.prompt_area_selectors:
                els = await self._query_all(tab, sel)
                if els:
                    logger.debug(f"[zendriver] Prompt area detected in DOM after navigation: {sel}")
                    return True
            await asyncio.sleep(0.5)

        logger.warning("[zendriver] Prompt area selector never appeared after page load — proceeding anyway")
        return False

    # ------------------------------------------------------------------ prompt chunking

    def _should_split_prompt(self, prompt: str) -> bool:
        if self._split_prompt_parts <= 1:
            return False
        limit = self._get_model_limit(self.get_current_model())
        return len(prompt) > limit

    def _split_prompt_into_parts(self, prompt: str, n: int) -> list[str]:
        """Split *prompt* into *n* roughly equal text chunks.

        If the prompt carries a protected tail marker (an opaque boundary the
        caller inserts to keep a trailing instruction intact), everything from
        that marker onward is kept as a single final chunk so it reaches the
        model whole — never split across chunk boundaries. The marker itself is
        stripped from the text that is actually sent. This layer treats the
        marker as opaque and never inspects the tail's contents.
        """
        marker_pos = prompt.rfind(AGENT_TAIL_MARKER)
        if marker_pos != -1:
            head = prompt[:marker_pos]
            tail = prompt[marker_pos + len(AGENT_TAIL_MARKER):].lstrip("\n")
            head_parts_n = max(n - 1, 1)
            head_chunk_size = max(math.ceil(len(head) / head_parts_n), 1)
            head_parts = [
                head[i: i + head_chunk_size] for i in range(0, len(head), head_chunk_size)
            ]
            if not head_parts:
                head_parts = [""]
            return head_parts + [tail]
        chunk_size = math.ceil(len(prompt) / n)
        return [prompt[i: i + chunk_size] for i in range(0, len(prompt), chunk_size)]

    async def _execute_chunked_send(self, prompt: str, tab: zd.Tab) -> str:
        """Send an oversized *prompt* in sequential chunks, keeping the session open.

        Parts 1..N-1 are prefixed with an instruction telling the LLM not to
        respond yet. Only the final part triggers a real reply.

        After each intermediate chunk is accepted (_post_send_check sees the
        stop button), the *next* chunk is pre-filled into the input area
        immediately — while the model is still generating its acknowledgement
        — and sending is deferred until just the send button becomes
        available again (_wait_for_send_ready), skipping the full
        _wait_for_response round-trip for every intermediate chunk.
        """
        limit = self._get_model_limit(self.get_current_model())
        min_parts = math.ceil(len(prompt) / limit)
        n = min(self._split_prompt_parts, max(min_parts, 2))
        parts = self._split_prompt_into_parts(prompt, n)
        n = len(parts)
        chunk_t0 = time.time()
        logger.info(
            f"[zendriver] Prompt chunking: {len(prompt)} chars split into {n} parts "
            f"(limit={limit}, env_max={self._split_prompt_parts})"
        )
        _engine_name = getattr(self, "ENGINE_NAME", "default")

        def _intermediate_text(idx: int, part: str) -> str:
            header = f"[PART {idx}/{n}] Reply ONLY: OK\n\n"
            return header + part

        first_text = _intermediate_text(1, parts[0])
        logger.debug(f"[zendriver] Sending chunk 1/{n} ({len(first_text)} chars)")
        debug_mode.record_event(
            "chunk", _engine_name, index=1, total=n, size=len(first_text), text=first_text,
        )
        input_el = await self._find_interactable_element(
            tab, self.prompt_area_selectors, timeout=20.0, cache_attr="_cached_prompt_selector",
        )
        if input_el is None:
            raise RuntimeError(f"Could not find prompt input area for chunk 1/{n}")
        await self._fill_input(tab, input_el, first_text)
        await self._click_accept_buttons(tab, timeout=2.0)
        await self._click_send(tab, input_el)
        await self._click_accept_buttons(tab, timeout=2.0)
        if not await self._post_send_check(tab, timeout=8.0):
            self._cached_prompt_selector = None
            self._cached_send_selector = None
            raise RuntimeError(f"redirect-stall: chunk 1/{n} not accepted after redirect")
        logger.info(f"[timing] chunk 1/{n} sent+accepted: {time.time() - chunk_t0:.2f}s")

        for idx in range(2, n + 1):
            is_final = idx == n
            chunk_start = time.time()

            next_text = parts[idx - 1] if not is_final else parts[-1]
            if not is_final:
                next_text = _intermediate_text(idx, next_text)

            input_el = await self._find_interactable_element(
                tab, self.prompt_area_selectors, timeout=5.0, cache_attr="_cached_prompt_selector",
            )
            if input_el is None:
                raise RuntimeError(f"Could not find prompt input area for chunk {idx}/{n}")

            pre_send_text = await self._read_pre_send_text(tab)
            await self._fill_input(tab, input_el, next_text)
            logger.debug(f"[zendriver] Filled chunk {idx} ({len(next_text)} chars)")
            debug_mode.record_event(
                "chunk", _engine_name, index=idx, total=n, size=len(next_text), text=next_text,
            )

            chunk_timeout = 30.0 if is_final else 5.0
            send_ready_start = time.time()
            if not await self._wait_for_send_ready(tab, timeout=chunk_timeout):
                if is_final:
                    logger.warning(f"[zendriver] Send button not found after {chunk_timeout}s for final chunk, proceeding...")
                else:
                    logger.debug(f"[zendriver] Chunk {idx-1} not ready after {chunk_timeout}s, proceeding")

            logger.info(f"[timing] chunk {idx}/{n} wait={time.time() - send_ready_start:.2f}s")

            await self._click_accept_buttons(tab, timeout=2.0)
            try:
                await self._click_send(tab, input_el)
            except Exception:
                logger.warning("[zendriver] Click send failed, using Enter key fallback")
                try:
                    await input_el.send_keys(SpecialKeys.ENTER)
                except Exception:
                    pass
            await self._click_accept_buttons(tab, timeout=2.0)
            logger.debug(f"[zendriver] Sent chunk {idx}")

            if is_final:
                self._skip_split_for_next = True
                try:
                    if not await self._post_send_check(tab):
                        self._cached_prompt_selector = None
                        self._cached_send_selector = None
                        raise RuntimeError("redirect-stall: final chunk not accepted by UI after send")
                    response = await self._wait_for_response(tab, pre_send_text=pre_send_text)

                    try:
                        if await self._stop_button_present(tab):
                            logger.info(
                                "[zendriver] Stop button still visible after chunked "
                                "response detection — refreshing page for clean state"
                            )
                            await tab.reload(ignore_cache=True)
                            await self._wait_for_page_ready(tab, timeout=30.0)
                            self._cached_prompt_selector = None
                            self._cached_send_selector = None
                    except Exception as refresh_err:
                        logger.warning("[zendriver] Post-chunked-response cleanup refresh failed: %s", refresh_err)

                    logger.info(f"[timing] chunk {idx}/{n} (final) response: {time.time() - chunk_start:.2f}s")
                    logger.info(f"[timing] TOTAL chunked send: {time.time() - chunk_t0:.2f}s")
                    return response
                finally:
                    self._skip_split_for_next = False
            else:
                if not await self._post_send_check(tab, timeout=5.0):
                    self._cached_prompt_selector = None
                    self._cached_send_selector = None
                    raise RuntimeError(f"redirect-stall: chunk {idx}/{n} not accepted by UI after send")
                logger.debug(f"[zendriver] Chunk {idx}/{n} accepted")

        raise RuntimeError("_execute_chunked_send: unexpected exit after loop")

    # ------------------------------------------------------------------ core flow

    async def _generate_response_retry_loop(self, prompt: str, media: list[Any] | None = None) -> str:
        """Retry wrapper around a single generation attempt."""
        max_attempts = 5
        self._page_refresh_attempts = 0
        for attempt in range(max_attempts):
            try:
                return await self._generate_response_once(prompt, media)
            except (RuntimeError, asyncio.TimeoutError) as e:
                if self._is_offsite_redirect_stall(e):
                    logger.warning(
                        f"[zendriver] Off-site redirect-stall on attempt {attempt + 1}: "
                        "service requires authentication — failing fast without retry."
                    )
                    raise

                if isinstance(e, LoginRequiredError):
                    raise

                # A failed attempt on a page that visibly asks to sign in is a
                # login problem, not a flaky page: resetting the browser and
                # retrying cannot fix it, and every relaunch can kill a login the
                # operator is finishing by hand in this very browser (Chromium
                # only flushes cookies lazily -- observed: 848 relaunches in a
                # day, no manual sign-in ever survived). Engines that allow
                # guest use get one more attempt before we give up, since a
                # guest session can legitimately work -- but WITHOUT resetting
                # the browser: seconds after a manual sign-in was pasted in, that
                # reset wiped it (observed live right after the first deploy).
                if not self._is_dead_session(e) and await self._login_lost():
                    if attempt >= 1 or not self.allow_unlogged:
                        logger.error(
                            "[zendriver] Attempt %d failed and the page shows a logged-out "
                            "state — login required, not retrying (%s)", attempt + 1, e,
                        )
                        raise LoginRequiredError(
                            f"login required: the service shows a logged-out page ({e})"
                        ) from e
                    logger.warning(
                        "[zendriver] Attempt %d failed on a logged-out page; one more try "
                        "as a guest, leaving the browser untouched (%s)", attempt + 1, e,
                    )
                    self._navigate_on_next_attempt = True
                    continue

                is_chunking_freeze = any(msg in str(e) for msg in [
                    "Send button did not become ready",
                    "redirect-stall",
                    "not accepted by UI",
                    "Could not find prompt input area",
                    "selenium_response_detection_timeout",
                    "TimeoutError",
                    "disconnected",
                ])

                if is_chunking_freeze and self._should_split_prompt(prompt) and attempt < max_attempts - 1:
                    self._split_prompt_parts += 1
                    logger.warning(
                        f"[zendriver] Chunking failure on attempt {attempt + 1}: {e}. "
                        f"Dynamically resizing split_prompt_parts to {self._split_prompt_parts} and retrying."
                    )
                    await self._reset_driver()
                    continue

                if attempt >= max_attempts - 1:
                    raise

                logger.warning(
                    "[zendriver] Attempt %d/%d failed: %s — retrying on a fresh chat",
                    attempt + 1, max_attempts, e,
                )
                self._navigate_on_next_attempt = True

                if self._is_page_refresh_required(e):
                    if self._page_refresh_attempts < self._max_page_refresh_attempts:
                        logger.warning(
                            f"[zendriver] Page refresh was performed on attempt {attempt + 1}, "
                            "retrying without browser reset…"
                        )
                        continue
                    logger.warning(
                        f"[zendriver] Max page refresh attempts reached on attempt {attempt + 1}, "
                        "resetting browser and retrying…"
                    )
                    await self._reset_driver()
                    continue
                if self._is_dead_session(e):
                    logger.warning(f"[zendriver] Dead session on attempt {attempt + 1}, resetting and retrying…")
                    await self._reset_driver()
                    continue
                if self._is_redirect_stall(e):
                    logger.warning(f"[zendriver] Redirect-stall on attempt {attempt + 1}, retrying without browser reset…")
                    continue
                if self._is_response_detection_timeout(e):
                    if getattr(self, "_generation_was_active", False):
                        logger.warning(
                            f"[zendriver] Response detection timeout on attempt {attempt + 1} "
                            "(model was actively generating — slow/thinking model, skipping browser reset)"
                        )
                    else:
                        logger.warning(
                            f"[zendriver] Response detection timeout on attempt {attempt + 1}, "
                            "resetting browser and retrying…"
                        )
                        await self._reset_driver()
                    continue

        raise RuntimeError("_generate_response_retry_loop exhausted retries")

    async def _generate_response_once(self, prompt: str, media: list[Any] | None = None) -> str:
        """Single attempt of the core generate flow.

        Note: ``_page_refresh_attempts`` is intentionally *not* reset here. The
        budget is reset once per request in ``_generate_response_retry_loop``
        so that stuck-stop-button page refreshes accumulate across attempts
        and can eventually exhaust the budget (triggering a browser reset)
        instead of looping forever.
        """
        t0 = time.time()

        session_hard_timeout = self._session_hard_timeout
        if session_hard_timeout is None:
            session_hard_timeout = int(os.getenv("SELENIUM_SESSION_HARD_TIMEOUT", "21600"))
        if self._driver_start_time and (time.time() - self._driver_start_time) > session_hard_timeout:
            logger.warning(
                "[zendriver] Session hard timeout (%ds) exceeded — resetting browser",
                session_hard_timeout,
            )
            # Graceful path deliberately: this watchdog fires on a live,
            # responsive browser (unlike the outer generate_response timeout
            # path, which force-resets only because the driving coroutine is
            # itself wedged), so a normal stop() is expected to succeed and
            # flush Chrome's cookie/session stores normally. See CLAUDE.md.
            await self._reset_driver()

        await self._ensure_ready()
        assert self.driver is not None, "_ensure_ready() must have set self.driver"
        tab = self.driver

        logged_in = await self.refresh_login_state()
        unlogged = not logged_in
        if unlogged:
            logger.warning("[zendriver] User is unlogged; continuing with unlogged mode (restricted/unreliable).")

        needs_nav = True
        try:
            current_url = await self._current_url(tab)
            if current_url.startswith(self.service_url):
                fresh_chat = getattr(self, "_fresh_chat_per_request", False)
                retry_nav = getattr(self, "_navigate_on_next_attempt", False)
                if fresh_chat or retry_nav:
                    logger.debug(
                        "[zendriver] On service URL but a fresh chat is required "
                        "(fresh_chat_per_request=%s, retry=%s) — navigating",
                        fresh_chat, retry_nav,
                    )
                else:
                    needs_nav = False
                    logger.debug("[zendriver] Already on service URL, skipping navigation")
        except Exception:
            pass

        if needs_nav:
            try:
                await asyncio.wait_for(tab.get(self.service_url), timeout=120)
            except Exception as nav_err:
                if self._is_dead_session(nav_err):
                    await self._reset_driver()
                    raise RuntimeError(f"Driver session died during navigation: {nav_err}") from nav_err
                raise
            await self._wait_for_page_ready(tab, timeout=30.0)

        self._navigate_on_next_attempt = False

        t1 = time.time()
        logger.info(f"[timing] page_ready: {t1 - t0:.2f}s")

        await self._click_accept_buttons(tab, timeout=2.0)

        if await self._is_captcha_present(tab):
            logger.warning("[zendriver] Cloudflare captcha challenge detected on page")
            return "⚠️ Cloudflare CAPTCHA detected. Please complete the CAPTCHA on the page and try again."

        if await self._is_limit_present(tab):
            logger.warning("[zendriver] Limit warning detected on page")
            return "⚠️ The service appears to have hit a usage limit. Please upgrade or wait, usually until tomorrow, before retrying."

        if media:
            tier = await self._check_account_tier(tab)
            limit_error = self._check_media_limits(media, tier)
            if limit_error:
                return limit_error
            if not await self._upload_media(media, tab):
                return "⚠️ Media upload failed. Please verify the file and try again."
            if self._vision_prompt_prefix:
                prompt = self._vision_prompt_prefix + prompt
        else:
            if self._inline_response_prefix:
                prompt = self._inline_response_prefix + prompt

        if not self._skip_split_for_next and not media and self._should_split_prompt(prompt):
            return await self._execute_chunked_send(prompt, tab)

        if AGENT_TAIL_MARKER in prompt:
            prompt = prompt.replace(AGENT_TAIL_MARKER, "")

        stale_retries = 0
        while True:
            try:
                input_el = await self._find_interactable_element(
                    tab, self.prompt_area_selectors, timeout=20.0, cache_attr="_cached_prompt_selector",
                )
                if input_el is None:
                    raise RuntimeError("Could not find prompt input area")

                t2 = time.time()
                logger.info(f"[timing] find_element: {t2 - t1:.2f}s")

                pre_send_text = await self._read_pre_send_text(tab)
                await self._fill_input(tab, input_el, prompt)
                t3 = time.time()
                logger.info(f"[timing] fill_input: {t3 - t2:.2f}s ({len(prompt)} chars)")

                await self._click_accept_buttons(tab, timeout=2.0)
                if media:
                    if not await self._wait_for_send_button_after_media_upload(tab):
                        logger.warning("[zendriver] Send button not ready after media upload")
                        return "⚠️ Media upload failed. Please verify the file and try again."
                await self._click_send(tab, input_el)
                t4 = time.time()
                logger.info(f"[timing] click_send: {t4 - t3:.2f}s")

                await self._click_accept_buttons(tab, timeout=2.0)
                if not await self._post_send_check(tab):
                    self._cached_prompt_selector = None
                    self._cached_send_selector = None
                    if getattr(self, "_last_send_offsite_redirect", False):
                        raise RuntimeError("redirect-stall (off-site): send not accepted after redirect")
                    raise RuntimeError("redirect-stall: send not accepted after redirect")
                t5 = time.time()
                logger.info(f"[timing] post_send_check: {t5 - t4:.2f}s")

                response = await self._wait_for_response(tab, pre_send_text=pre_send_text)

                if self._is_engine_error_response(response):
                    logger.warning(
                        "[zendriver] Engine error detected in response: %r — raising for retry with browser reset",
                        response[:200],
                    )
                    await self._reset_driver()
                    raise RuntimeError(
                        "selenium_response_detection_timeout: engine error response detected — browser reset for retry"
                    )

                try:
                    if await self._stop_button_present(tab):
                        logger.info(
                            "[zendriver] Stop button still visible after response detection — refreshing page for clean state"
                        )
                        await tab.reload(ignore_cache=True)
                        await self._wait_for_page_ready(tab, timeout=30.0)
                        self._cached_prompt_selector = None
                        self._cached_send_selector = None
                except Exception as refresh_err:
                    logger.warning("[zendriver] Post-response cleanup refresh failed: %s", refresh_err)

                t6 = time.time()
                logger.info(f"[timing] wait_for_response: {t6 - t5:.2f}s")
                logger.info(f"[timing] TOTAL generate: {t6 - t0:.2f}s")
                return response

            except Exception as e:
                if self._is_stale_element_error(e):
                    stale_retries += 1
                    self._cached_prompt_selector = None
                    self._cached_send_selector = None
                    if stale_retries >= 2:
                        logger.error("[zendriver] _generate_response_once failed after stale element retry: %s", e)
                        raise
                    logger.warning(
                        "[zendriver] Stale element detected mid-prompt; retrying prompt flow (%s/2): %s",
                        stale_retries, e,
                    )
                    continue

                if self._is_dead_session(e):
                    await self._reset_driver()
                    raise RuntimeError(f"Driver session died mid-prompt: {e}") from e
                logger.error(f"[zendriver] _generate_response_once failed: {e}")
                if unlogged:
                    raise RuntimeError(f"Unlogged session: could not run full prompt flow. Error: {e}") from e
                raise

    # ------------------------------------------------------------------ media helpers

    async def _check_account_tier(self, tab: zd.Tab) -> str:
        if not await self.refresh_login_state():
            return "unlogged"
        if getattr(self, "base_account_selector", None):
            for el in await self._query_all(tab, self.base_account_selector):
                if await self._element_is_displayed(el):
                    return "base"
        if self.paid_account_selector:
            for el in await self._query_all(tab, self.paid_account_selector):
                if await self._element_is_displayed(el):
                    return "paid"
        return "base"

    def _check_media_limits(self, media: list[Any], tier: str) -> str | None:
        counts: dict[str, int] = {}
        for item in media:
            media_type = getattr(item, "media_type", None)
            if not media_type:
                continue
            counts[media_type] = counts.get(media_type, 0) + 1

        configured_types = [
            key for key in self.media_config.keys()
            if key not in {"paid_account_selector", "base_account_selector", "total_limits", "shared_limits"}
        ]
        if not configured_types:
            return None

        for media_type, count in counts.items():
            cfg = self.media_config.get(media_type)
            if cfg is None:
                logger.debug(f"[zendriver] No media_support entry for '{media_type}', will try upload anyway")
                continue
            limit_error = self._evaluate_media_limit(media_type, cfg, count, tier)
            if limit_error:
                return limit_error

        total_cfg = self.media_config.get("total_limits") or self.media_config.get("shared_limits")
        if total_cfg is not None:
            total_count = sum(counts.values())
            total_error = self._evaluate_media_limit("media", total_cfg, total_count, tier, total=True)
            if total_error:
                return total_error

        return None

    def _evaluate_media_limit(self, media_type: str, cfg: Any, count: int, tier: str, total: bool = False) -> str | None:
        if cfg is None:
            return None

        supported_models = cfg.get("supported_models")
        if supported_models is not None:
            current_model = self.get_current_model()
            if "all" not in supported_models:
                if "not-unlogged" in supported_models:
                    if current_model == "unlogged":
                        logger.debug(f"[zendriver] Media '{media_type}' explicitly not supported for unlogged sessions")
                elif current_model not in supported_models:
                    logger.debug(f"[zendriver] Model '{current_model}' not listed for '{media_type}' media; will still attempt upload")

        limits = cfg.get("limits", {})
        if not limits:
            return f"⚠️ Media type '{media_type}' is not supported by this engine."

        tier_limit = limits.get(tier, 0)
        if tier_limit == -1 or tier_limit == 0:
            logger.debug(f"[zendriver] Media '{media_type}' has zero/unlimited limit for tier '{tier}'; attempting upload anyway")
            return None
        if count > tier_limit:
            kind = "media" if total else media_type
            return f"⚠️ The use of '{kind}' is exhausted for today. Please try again tomorrow."
        return None

    async def _wait_for_media_upload_complete(self, item: Any, tab: zd.Tab, timeout: float = 20.0) -> bool:
        selectors = self.media_config.get(item.media_type, {}).get("upload_complete_selectors", [])
        if not selectors:
            return True

        start = time.time()
        timeout = float(os.getenv("SELENIUM_MEDIA_UPLOAD_COMPLETE_WAIT", str(timeout)))
        deadline = start + timeout

        while time.time() < deadline:
            all_satisfied = True
            for sel in selectors:
                absent = False
                raw_sel = sel
                if sel.startswith("!"):
                    absent = True
                    raw_sel = sel[1:]

                elements = await self._query_all(tab, raw_sel)
                if absent:
                    if elements:
                        all_satisfied = False
                        break
                else:
                    displayed = False
                    for el in elements:
                        if await self._element_is_displayed(el):
                            displayed = True
                            break
                    if not displayed:
                        all_satisfied = False
                        break

            if all_satisfied:
                logger.debug(f"[zendriver] Media upload completion selectors satisfied: {selectors}")
                return True
            await asyncio.sleep(0.25)

        logger.warning(f"[zendriver] Media upload completion wait timed out for selectors: {selectors}")
        return False

    async def _upload_media(self, media: list[Any], tab: zd.Tab) -> bool:
        for item in media:
            tmp_path = self._write_temp_media_file(item)
            try:
                upload_success = await self._upload_via_file_input(item, tmp_path, tab)
                if not upload_success:
                    upload_success = await self._upload_via_clipboard(item, tmp_path, tab)

                if not upload_success:
                    logger.warning(f"[zendriver] Media upload failed for type '{item.media_type}'")
                    return False

                await self._click_accept_buttons(tab, timeout=5.0)

                if not await self._wait_for_media_upload_complete(item, tab):
                    logger.warning(f"[zendriver] Media upload did not reach completion state for type '{item.media_type}'")
                    return False
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
        return True

    async def _wait_for_send_button_after_media_upload(self, tab: zd.Tab, timeout: float = 15.0) -> bool:
        start = time.time()
        timeout = float(os.getenv("SELENIUM_MEDIA_UPLOAD_WAIT", str(timeout)))
        deadline = start + timeout

        while time.time() < deadline:
            for sel in self.send_button_selectors:
                for button in await self._query_all(tab, sel):
                    if await self._element_is_displayed(button) and await self._element_is_enabled(button):
                        elapsed = time.time() - start
                        logger.debug(f"[zendriver] Send button ready after media upload for selector {sel} in {elapsed:.2f}s")
                        return True
            await asyncio.sleep(0.25)

        logger.warning(f"[zendriver] Send button did not become ready after media upload within {timeout:.1f}s")
        return False

    def _write_temp_media_file(self, item: Any) -> str:
        suffix = ""
        if item.mime_type and "/" in item.mime_type:
            ext = mimetypes.guess_extension(item.mime_type)
            if ext:
                suffix = ext
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(item.data)
        tmp.flush()
        tmp.close()
        return tmp.name

    async def _upload_via_file_input(self, item: Any, path: str, tab: zd.Tab) -> bool:
        selectors = self.media_config.get(item.media_type, {}).get("upload_selectors", ["input[type='file']"])
        for sel in selectors:
            inputs = await self._query_all(tab, sel)
            if not inputs:
                logger.debug(f"[zendriver] File input selector found no elements: {sel}")
                continue

            for inp in inputs:
                try:
                    if (inp.tag_name or "").lower() != "input":
                        continue
                    if (await self._element_attr(inp, "type")).lower() != "file":
                        continue
                    send_ok = False
                    try:
                        await inp.send_file(path)
                        send_ok = True
                    except Exception as exc:
                        logger.debug(f"[zendriver] send_file error for selector {sel}: {exc}")
                        try:
                            await inp.apply(
                                "(el) => { el.style.display='block'; el.style.visibility='visible';"
                                " el.style.opacity='1'; el.style.position='fixed';"
                                " el.style.width='1px'; el.style.height='1px'; }"
                            )
                        except Exception:
                            pass
                        try:
                            await inp.send_file(path)
                            send_ok = True
                        except Exception as exc2:
                            logger.warning(f"[zendriver] send_file failed after visibility fallback for selector {sel}: {exc2}")
                            continue

                    end_time = time.time() + 2.0
                    while time.time() < end_time:
                        try:
                            # Live JS reads, not element.attrs — see _get_input_text
                            # for why the static HTML attribute is the wrong source
                            # for a form control's live value/files state.
                            files_count = await inp.apply("(el) => el.files ? el.files.length : 0")
                            value = await inp.apply("(el) => el.value || ''")
                            if value or files_count:
                                logger.debug(f"[zendriver] Media file input received value/files for selector {sel}")
                                return True
                        except Exception:
                            pass
                        await asyncio.sleep(0.1)
                    if send_ok:
                        logger.debug(f"[zendriver] File input value/files empty after send_file (SPA reset?); trusting send for selector {sel}")
                        return True
                    logger.debug(f"[zendriver] Media file input did not report a value/files for selector {sel}")
                    continue
                except Exception:
                    continue
        return False

    async def _upload_via_clipboard(self, item: Any, path: str, tab: zd.Tab) -> bool:
        clipboard_cmd = None
        if shutil.which("xclip"):
            clipboard_cmd = ["xclip", "-selection", "clipboard", "-t", item.mime_type, "-i", path]
        elif shutil.which("xsel"):
            clipboard_cmd = ["xsel", "--clipboard", "--input"]
        elif shutil.which("wl-copy"):
            clipboard_cmd = ["wl-copy", "--type", item.mime_type]

        if clipboard_cmd is None:
            logger.warning("[zendriver] Clipboard upload skipped: no clipboard utility found")
            return False

        xclip_proc = None
        try:
            if clipboard_cmd[0] in ("xsel", "wl-copy"):
                with open(path, "rb") as fh:
                    data = fh.read()
                await asyncio.to_thread(
                    subprocess.run, clipboard_cmd, input=data, check=True, capture_output=True, timeout=15,
                )
            else:
                # xclip -i blocks until the clipboard is consumed by a reader;
                # run it in the background so that the paste below can succeed.
                xclip_proc = subprocess.Popen(clipboard_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                await asyncio.sleep(0.3)
            logger.debug(f"[zendriver] Copied media to clipboard using {clipboard_cmd[0]}")
        except Exception as exc:
            logger.warning(f"[zendriver] Clipboard transfer failed with {clipboard_cmd[0]}: {exc}")
            return False

        input_el = await self._find_interactable_element(
            tab, self.prompt_area_selectors, timeout=10.0, cache_attr="_cached_prompt_selector",
        )
        if input_el is None:
            logger.warning("[zendriver] Clipboard upload failed: prompt area not found")
            return False

        try:
            try:
                await input_el.apply("(e) => e.focus()")
            except Exception:
                pass
            try:
                await input_el.click()
            except Exception:
                pass
            await asyncio.sleep(0.1)
            try:
                await input_el.send_keys(KeyEvents.from_mixed_input([("v", KeyModifiers.Ctrl)]))
            except Exception:
                pass
            await asyncio.sleep(0.5)
            if xclip_proc is not None:
                try:
                    xclip_proc.terminate()
                except Exception:
                    pass
            logger.debug("[zendriver] Clipboard paste attempted")
            return True
        except Exception as exc:
            if xclip_proc is not None:
                try:
                    xclip_proc.terminate()
                except Exception:
                    pass
            logger.warning(f"[zendriver] Clipboard paste failed: {exc}")
            return False

    # ------------------------------------------------------------------ response text helpers

    def _looks_like_response_text_junk(self, text: str) -> bool:
        """Return True for text that appears to be page state JSON rather than chat output."""
        if not text:
            return True

        stripped = text.strip()
        if len(stripped) > 300:
            return False

        if not self._agent_mode and (stripped.startswith("{") or stripped.startswith("[")):
            try:
                json.loads(stripped)
                logger.debug("[zendriver] _looks_like_response_text_junk: rejected JSON-like text")
                return True
            except Exception:
                pass

        if stripped.startswith("Gemini said"):
            after = stripped[len("Gemini said"):].strip()
            if not after:
                logger.debug("[zendriver] _looks_like_response_text_junk: rejected bare Gemini said prefix")
                return True
            if after.startswith("{") or after.startswith("["):
                try:
                    json.loads(after)
                    logger.debug("[zendriver] _looks_like_response_text_junk: rejected Gemini state dump")
                    return True
                except Exception:
                    pass
            if "\"memory_search\"" in after or "\"recovery_actions\"" in after:
                logger.debug("[zendriver] _looks_like_response_text_junk: rejected Gemini state prefix")
                return True

        if "\"memory_search\"" in stripped or "\"recovery_actions\"" in stripped:
            logger.debug("[zendriver] _looks_like_response_text_junk: rejected memory/search state text")
            return True

        return False

    async def _get_response_text_js(self, tab: zd.Tab) -> str:
        """Return fallback response text using JavaScript when CSS selectors fail."""
        script = """
        (() => {
            const selectors = [
                '.assistant-message', '.assistant', '.gemini-response',
                'message-content .markdown-main-panel', '.markdown-main-panel',
                'model-response .markdown', 'model-response', '.model-response',
                '.response-container', '.presented-response-container',
                '.structured-content-container', '.markdown', 'message-content',
                '.message-content', 'div[role="article"]', 'article'
            ];
            for (const sel of selectors) {
                const els = Array.from(document.querySelectorAll(sel));
                for (let i = els.length - 1; i >= 0; i--) {
                    const el = els[i];
                    if (el && el.textContent && el.textContent.trim()) {
                        return el.textContent.trim();
                    }
                }
            }
            return '';
        })()
        """
        try:
            result = await tab.evaluate(script)
            if isinstance(result, str):
                result = result.strip()
                if not self._looks_like_response_text_junk(result):
                    return result
        except Exception:
            pass
        return ""

    async def _get_latest_response_text(self, tab: zd.Tab) -> str:
        """Return latest non-empty text from response selectors, or empty if none."""
        candidates: list[str] = []
        logger.debug("[zendriver] _get_latest_response_text: trying selectors=%s", self.response_area_selectors)
        for sel in self.response_area_selectors:
            els = await self._query_all(tab, sel)
            logger.debug("[zendriver] _get_latest_response_text: selector=%s found %d elements", sel, len(els))
            if not els:
                continue

            for elem in reversed(els):
                text = await self._element_text(elem)
                tc = await self._element_text_content(elem)

                if tc and self._looks_like_response_text_junk(tc):
                    logger.debug(
                        "[zendriver] _get_latest_response_text: selector=%s text=%r tc=%r rejected as junk",
                        sel, text[:120], tc[:120],
                    )
                    continue

                if text and not self._looks_like_response_text_junk(text):
                    logger.debug("[zendriver] _get_latest_response_text: selector=%s returned visible text=%r", sel, text[:120])
                    return text
                if tc and not self._looks_like_response_text_junk(tc):
                    logger.debug("[zendriver] _get_latest_response_text: selector=%s used textContent fallback=%r", sel, tc[:120])
                    return tc

                if text:
                    candidates.append(text)
                if tc and tc != text:
                    candidates.append(tc)

        for candidate in candidates:
            if not self._looks_like_response_text_junk(candidate):
                return candidate

        return await self._get_response_text_js(tab)

    async def _send_button_present(self, tab: zd.Tab) -> bool:
        for sel in self.send_button_selectors:
            for b in await self._query_all(tab, sel):
                if await self._element_is_displayed(b) and await self._element_is_enabled(b):
                    return True
        return False

    async def _response_area_present(self, tab: zd.Tab) -> bool:
        for sel in self.response_area_selectors:
            if await self._query_all(tab, sel):
                return True
        return False

    async def _stop_button_present(self, tab: zd.Tab) -> bool:
        for sel in self.stop_selectors:
            for b in await self._query_all(tab, sel):
                if await self._element_is_displayed(b):
                    return True
        return False

    async def _find_response_container_element(self, tab: zd.Tab) -> tuple[Any | None, str | None]:
        for sel in self.response_area_selectors:
            elements = await self._query_all(tab, sel)
            logger.debug("[zendriver] _find_response_container_element: selector=%s found %d elements", sel, len(elements))
            if not elements:
                continue
            visible = [el for el in elements if await self._element_is_displayed(el)]
            if visible:
                return visible[-1], sel
            return elements[-1], sel
        return None, None

    async def _get_response_container_stats(self, element: Any) -> tuple[int, int]:
        try:
            result = await element.apply(
                "(el) => [el?.innerText?.length || 0, el?.childElementCount || 0]"
            )
            if isinstance(result, list) and len(result) == 2:
                return int(result[0] or 0), int(result[1] or 0)
        except Exception as exc:
            logger.debug("[zendriver] _get_response_container_stats: JS failed: %s", exc)
        try:
            text = await self._element_text(element)
            return len(text), 0
        except Exception:
            return 0, 0

    async def _extract_response_text_from_element(self, element: Any) -> str:
        try:
            result = await element.apply(
                "(el) => el && (el.innerText || el.textContent) ? (el.innerText || el.textContent).trim() : ''"
            )
            if isinstance(result, str):
                return result.strip()
        except Exception as exc:
            logger.debug("[zendriver] _extract_response_text_from_element: JS failed: %s", exc)
        return await self._element_text(element)

    def _log_response_container_diagnostics(
        self, selector, current_text_length, current_child_count,
        previous_text_length, previous_child_count, stable_counter, iteration,
    ) -> None:
        logger.debug(
            "[zendriver] watcher iteration=%d selector=%s current_text_length=%d current_child_count=%d "
            "previous_text_length=%d previous_child_count=%d stable_counter=%d",
            iteration, selector, current_text_length, current_child_count,
            previous_text_length, previous_child_count, stable_counter,
        )

    async def _find_interactable_element(
        self, tab: zd.Tab, selectors: list[str], timeout: float = 20.0, cache_attr: Optional[str] = None,
    ) -> Optional[Any]:
        """Try CSS selectors in order; return first element that is clickable.

        Reimplemented as a bounded poll loop over ``tab.query_selector_all``
        (Selenium's WebDriverWait+element_to_be_clickable has no zendriver
        equivalent) rather than a long per-selector wait, matching the
        original's "cache the winner, short-circuit on first interactable
        match" behaviour.
        """
        ordered: list[str] = list(selectors)
        cached: Optional[str] = getattr(self, cache_attr, None) if cache_attr else None
        if cached and cached in ordered:
            ordered.remove(cached)
            ordered.insert(0, cached)

        deadline = time.time() + timeout
        while time.time() < deadline:
            for sel in ordered:
                try:
                    els = await tab.query_selector_all(sel)
                except Exception as e:
                    if self._is_dead_session(e):
                        raise
                    continue
                for el in els:
                    if await self._element_is_displayed(el) and await self._element_is_enabled(el):
                        logger.debug(f"[zendriver] Found clickable element: {sel}")
                        if cache_attr:
                            setattr(self, cache_attr, sel)
                        return el
            await asyncio.sleep(0.3)

        # Fallback: some rich editor areas are visible but not considered "enabled".
        for sel in ordered:
            els = await self._query_all(tab, sel)
            for el in els:
                if await self._element_is_displayed(el):
                    logger.debug(f"[zendriver] Found visible element fallback: {sel}")
                    if cache_attr:
                        setattr(self, cache_attr, sel)
                    return el

        logger.warning("[zendriver] No interactable element found")
        return None

    def _normalize_input_text(self, text: str) -> str:
        return " ".join(text.replace("\r\n", "\n").strip().split())

    # Very long text is slow enough through per-character CDP key events to be
    # worth avoiding, and non-BMP characters (emoji) cannot be sent as CDP
    # "char" key events at all. Both were observed in production with the old
    # Selenium/ChromeDriver keyboard fallback, so it is kept only for short,
    # BMP-safe text; anything else goes through the JS insertText path.
    SEND_KEYS_MAX_CHARS = 5000

    @staticmethod
    def _is_bmp_safe(text: str) -> bool:
        return all(ord(c) <= 0xFFFF for c in text)

    def _send_keys_is_safe(self, text: str) -> bool:
        limit = int(os.getenv("SELENIUM_SEND_KEYS_MAX_CHARS", str(self.SEND_KEYS_MAX_CHARS)))
        return len(text) <= limit and self._is_bmp_safe(text)

    def _log_insert_mismatch(self, current: str, text: str) -> None:
        cur_c = self._strip_all_whitespace(current)
        exp_c = self._strip_all_whitespace(text)
        diff_at = next(
            (i for i, (a, b) in enumerate(zip(cur_c, exp_c)) if a != b),
            min(len(cur_c), len(exp_c)),
        )
        logger.warning(
            "[zendriver] JS insert not verified: editor has %d chars, expected %d (first difference at %d)",
            len(cur_c), len(exp_c), diff_at,
        )
        logger.warning(
            "[zendriver] JS insert mismatch context: editor=%r expected=%r",
            cur_c[max(0, diff_at - 60): diff_at + 60], exp_c[max(0, diff_at - 60): diff_at + 60],
        )

    async def _get_input_text(self, element: Any) -> str:
        """Live DOM read via JS, for both form controls and contenteditable.

        NOTE: deliberately does NOT use ``element.attrs``/``_element_attr``
        for a textarea/input's value. zendriver's ``Element.attrs`` reflects
        the static HTML *attribute* list captured when the node was resolved
        (DOM.getDocument/DOM.resolveNode), not the live JS ``.value``
        *property* — the two diverge the moment a user (or our own
        send_keys) types into the field, since typing never touches the HTML
        attribute. Selenium's ``get_attribute("value")`` special-cases this
        and returns the live property; zendriver has no such bridge, so a
        live ``element.apply()`` read is the only way to see typed text.
        """
        try:
            actual = await element.apply(
                "(el) => { if (el.value !== undefined) return el.value || '';"
                " return el.innerText || el.textContent || ''; }"
            )
            return str(actual or "")
        except Exception:
            return ""

    @staticmethod
    def _strip_all_whitespace(text: str) -> str:
        return "".join(text.split())

    async def _verify_input_text(self, element: Any, expected: str) -> bool:
        expected_norm = self._normalize_input_text(expected)
        expected_compact = self._strip_all_whitespace(expected)
        end_time = time.time() + 1.0
        actual = ""
        while time.time() < end_time:
            actual = await self._get_input_text(element)
            if self._normalize_input_text(actual) == expected_norm:
                return True
            if self._strip_all_whitespace(actual) == expected_compact:
                logger.debug("[zendriver] fill_input matched after whitespace-insensitive comparison (editor re-flowed whitespace)")
                return True
            await asyncio.sleep(0.1)
        actual_norm = self._normalize_input_text(actual)
        logger.warning(
            "[zendriver] fill_input verification failed: expected %s chars, got %s chars",
            len(expected_norm), len(actual_norm),
        )
        logger.warning(
            "[zendriver] fill_input expected_tail=%r actual_tail=%r", expected_norm[-160:], actual_norm[-160:],
        )
        return False

    async def _fill_input(self, tab: zd.Tab, element: Any, text: str) -> None:
        """Type *text* into a textarea or contenteditable element."""
        attempts = 0
        while True:
            try:
                try:
                    await element.scroll_into_view()
                    await asyncio.sleep(0.3)
                except Exception:
                    pass

                try:
                    await element.click()
                except Exception:
                    try:
                        await element.apply("(el) => el.click()")
                    except Exception:
                        pass
                    await asyncio.sleep(0.1)

                tag = (element.tag_name or "").lower()
                text_json = json.dumps(text)
                if tag in ("textarea", "input"):
                    try:
                        await element.clear_input()
                    except Exception:
                        try:
                            await element.send_keys(KeyEvents.from_mixed_input([("a", KeyModifiers.Ctrl)]))
                            await element.send_keys(SpecialKeys.DELETE)
                        except Exception:
                            pass
                    await element.send_keys(text)
                else:
                    # contenteditable (ProseMirror, Lexical, Quill, …)
                    # First try JS execCommand: instant, no char-by-char latency. Some
                    # rich-text editors (notably ProseMirror-based ones) intercept and
                    # ignore execCommand, leaving stale content untouched, so we always
                    # verify the DOM afterwards and fall back to native key events, which
                    # produce real keyboard input the editor cannot ignore.
                    try:
                        await element.apply(
                            "(el) => { el.focus(); document.execCommand('selectAll', false, null);"
                            " document.execCommand('delete', false, null); }"
                        )
                        await asyncio.sleep(0.05)
                        await element.apply(
                            "(el) => { el.focus(); document.execCommand('selectAll', false, null);"
                            f" document.execCommand('insertText', false, {text_json}); }}"
                        )
                    except Exception:
                        pass

                    exec_ok = False
                    current = ""
                    expected_compact = self._strip_all_whitespace(text)
                    deadline = time.time() + 1.5
                    while True:
                        try:
                            current = await self._get_input_text(element)
                            exec_ok = self._strip_all_whitespace(current) == expected_compact
                        except Exception:
                            exec_ok = False
                        if exec_ok or time.time() >= deadline:
                            break
                        await asyncio.sleep(0.1)

                    if not exec_ok and not self._send_keys_is_safe(text):
                        self._log_insert_mismatch(current, text)
                        logger.warning(
                            "[zendriver] Not typing %d chars through the keyboard (bmp_safe=%s); retrying the JS insert instead",
                            len(text), self._is_bmp_safe(text),
                        )
                        try:
                            await element.apply(
                                "(el) => { el.focus(); document.execCommand('selectAll', false, null);"
                                f" document.execCommand('insertText', false, {text_json}); }}"
                            )
                        except Exception as exc:
                            logger.warning("[zendriver] JS insert retry failed: %s", exc)
                    elif not exec_ok:
                        self._log_insert_mismatch(current, text)
                        try:
                            await element.click()
                        except Exception:
                            try:
                                await element.apply("(el) => el.focus()")
                            except Exception:
                                pass
                        try:
                            await element.send_keys(KeyEvents.from_mixed_input([("a", KeyModifiers.Ctrl)]))
                            await element.send_keys(SpecialKeys.DELETE)
                            await asyncio.sleep(0.05)
                            await element.send_keys(text)
                        except Exception as e:
                            if self._is_stale_element_error(e):
                                raise
                            logger.error(f"[zendriver] fill_input send_keys failed: {e}")
                            raise
                    else:
                        # Some frameworks distinguish programmatic updates from user key
                        # events. Trigger a whitespace keypress + backspace to force UI
                        # internals to re-evaluate and enable the send button.
                        try:
                            await element.send_keys(SpecialKeys.SPACE)
                            await element.send_keys(SpecialKeys.BACKSPACE)
                        except Exception:
                            pass

                try:
                    await element.apply(
                        "(el) => { const evOpt = {bubbles:true, cancelable:true, composed:true};"
                        " el.dispatchEvent(new InputEvent('input', evOpt));"
                        " el.dispatchEvent(new KeyboardEvent('keyup', evOpt));"
                        " el.dispatchEvent(new Event('change', evOpt));"
                        " el.dispatchEvent(new Event('blur', evOpt)); }"
                    )
                except Exception:
                    pass
                logger.debug(f"[zendriver] Filled input ({len(text)} chars)")
                if not await self._verify_input_text(element, text):
                    raise RuntimeError("[zendriver] fill_input verification failed: prompt content did not match expected text")
                return
            except Exception as exc:
                if not self._is_stale_element_error(exc):
                    raise
                attempts += 1
                logger.warning("[zendriver] fill_input stale element reference; retrying (%s/2): %s", attempts, exc)
                if attempts >= 2:
                    raise
                element = await self._find_interactable_element(
                    tab, self.prompt_area_selectors, timeout=5.0, cache_attr="_cached_prompt_selector",
                )
                if element is None:
                    raise RuntimeError("Could not find prompt input area after stale element during fill_input")
                continue

    async def _click_send(self, tab: zd.Tab, input_el: Any) -> None:
        """Click the send button, or fall back to the Enter key."""
        if await self._stop_button_present(tab):
            logger.info(
                "[zendriver] _click_send: stop button visible (previous generation in progress), "
                "waiting for it to disappear before sending…"
            )
            wait_deadline = time.time() + 30.0
            settled = False
            while time.time() < wait_deadline:
                if not await self._stop_button_present(tab):
                    logger.info("[zendriver] _click_send: stop button disappeared, generation complete")
                    settled = True
                    break
                await asyncio.sleep(0.5)
            if not settled:
                logger.warning("[zendriver] _click_send: stop button did not disappear within 30s — refreshing page")
                if self._page_refresh_attempts < self._max_page_refresh_attempts:
                    try:
                        await tab.reload(ignore_cache=True)
                    except Exception as refresh_err:
                        logger.warning("[zendriver] _click_send tab.reload() failed: %s", refresh_err)
                    await self._wait_for_page_ready(tab, timeout=30.0)
                    self._cached_prompt_selector = None
                    self._cached_send_selector = None
                    self._page_refresh_attempts += 1
                    logger.info(
                        "[zendriver] _click_send: page refreshed (attempt %d/%d), retrying",
                        self._page_refresh_attempts, self._max_page_refresh_attempts,
                    )
                    raise RuntimeError("page_refresh_required: stop button stuck before send — page refreshed, retry without browser reset")
                logger.warning(
                    "[zendriver] _click_send: max page refresh attempts (%d) reached, proceeding anyway",
                    self._max_page_refresh_attempts,
                )

        max_attempts = int(os.getenv("SELENIUM_SEND_CLICK_RETRIES", "1"))

        async def _is_button_blacklisted(btn: Any) -> bool:
            for bl_sel in self.send_button_blacklist:
                matches = await self._query_all(tab, bl_sel)
                if any(m == btn for m in matches):
                    logger.debug(f"[zendriver] Skipping blacklisted button for selector: {bl_sel}")
                    return True
            return False

        async def _safe_click(btn: Any, selector: str) -> bool:
            if await _is_button_blacklisted(btn):
                return False
            try:
                await btn.click()
                logger.debug(f"[zendriver] Sent via button click: {selector}")
                return True
            except Exception as e:
                logger.warning(f"[zendriver] Button.click() failed for {selector}: {e}")
            try:
                await btn.apply("(el) => el.click()")
                logger.debug(f"[zendriver] Sent via JS click: {selector}")
                return True
            except Exception as e:
                logger.warning(f"[zendriver] JS click failed for {selector}: {e}")
            return False

        async def _click_climbing_to_button(element: Any, selector: str) -> bool:
            """If *element* is an icon/span inside a real button, climb to and
            click that ancestor in a single JS round trip instead of the
            element itself — mirrors the old Selenium version's
            ``_resolve_click_target``. Done as one atomic JS call (rather than
            resolving the ancestor back into a separate zendriver Element)
            because ``element.apply()`` only ever returns a plain JS value,
            not a re-usable Element handle, for a DOM node it climbed to.
            """
            tag = (element.tag_name or "").lower()
            if tag not in ("svg", "path", "span", "mat-icon", "i"):
                return await _safe_click(element, selector)
            try:
                if await _is_button_blacklisted(element):
                    return False
                clicked = await element.apply(
                    "(el) => { let e = el;"
                    " while (e && e.tagName !== 'BODY') {"
                    "   if (e.tagName === 'BUTTON' || e.getAttribute('role') === 'button' || e.onclick) { e.click(); return true; }"
                    "   e = e.parentElement;"
                    " }"
                    " el.click(); return true; }"
                )
                if clicked:
                    logger.debug(f"[zendriver] Sent via climb-to-button JS click: {selector}")
                    return True
            except Exception as e:
                logger.debug(f"[zendriver] climb-to-button click failed for {selector}: {e}")
            return await _safe_click(element, selector)

        async def _attempt_click() -> bool:
            ordered_send: list[str] = list(self.send_button_selectors)
            if self._cached_send_selector and self._cached_send_selector in ordered_send:
                ordered_send.remove(self._cached_send_selector)
                ordered_send.insert(0, self._cached_send_selector)

            for sel in ordered_send:
                candidates = await self._query_all(tab, sel)
                for btn in candidates:
                    try:
                        if await self._element_is_displayed(btn) and await self._element_is_enabled(btn):
                            if await _click_climbing_to_button(btn, sel):
                                self._cached_send_selector = sel
                                return True
                    except Exception as e:
                        if self._is_stale_element_error(e):
                            if self._cached_send_selector == sel:
                                self._cached_send_selector = None
                            continue
                        logger.debug(f"[zendriver] selector {sel} click attempt failed: {e}")
            return False

        for attempt in range(max_attempts):
            if attempt > 0:
                logger.info(f"[zendriver] _click_send retry {attempt + 1}/{max_attempts}")
                self._cached_prompt_selector = None
                self._cached_send_selector = None
                try:
                    await tab.evaluate(
                        "document.body.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}));"
                    )
                    await input_el.send_keys(SpecialKeys.ESCAPE)
                except Exception:
                    pass
                await asyncio.sleep(0.5)

            try:
                if await _attempt_click():
                    return
            except Exception as e:
                logger.warning(f"[zendriver] Error during send click attempt, retrying: {e}")
                continue

        try:
            await input_el.send_keys(SpecialKeys.ENTER)
            logger.debug("[zendriver] Sent via Enter key")
            return
        except Exception as e:
            logger.error(f"[zendriver] Could not send prompt: {e}")

    async def _wait_for_send_ready(self, tab: zd.Tab, timeout: float = 10.0) -> bool:
        timeout = float(os.getenv("SELENIUM_SEND_READY_TIMEOUT", str(timeout)))
        deadline = time.time() + timeout
        while time.time() < deadline:
            for sel in self.send_button_selectors:
                try:
                    btns = await self._query_all(tab, sel)
                except Exception as e:
                    if self._is_dead_session(e):
                        raise
                    btns = []
                for btn in btns:
                    if not (await self._element_is_displayed(btn) and await self._element_is_enabled(btn)):
                        continue
                    aria_label = (await self._element_attr(btn, "aria-label")).lower()
                    data_testid = (await self._element_attr(btn, "data-testid")).lower()
                    tag = btn.tag_name or ""
                    skip = any(p in aria_label or p in data_testid for p in ("stop", "cancel", "mic", "voice"))
                    if tag == "mat-icon" and "mic" in (await self._element_attr(btn, "fonticon")).lower():
                        skip = True
                    if not skip:
                        logger.debug(f"[zendriver] Send button ready: {sel}")
                        return True
            await asyncio.sleep(0.2)
        logger.warning(f"[zendriver] Send button did not become ready within {timeout:.1f}s")
        return False

    def _is_same_site(self, current_url: str) -> bool:
        """Return True if *current_url* is on the same host as service_url.

        Some engines create a new conversation on send and navigate to a
        different path on the same host. That in-site navigation is normal
        and must NOT be treated as a redirect-stall. A genuine stall sends
        the browser to a different host (e.g. an auth/login domain).
        """
        if not self.service_url or not current_url:
            return True
        try:
            base_host = urlparse(self.service_url).netloc
            cur_host = urlparse(current_url).netloc
        except Exception:
            return True
        if not base_host or not cur_host:
            return True
        return base_host == cur_host

    async def _post_send_check(self, tab: zd.Tab, timeout: float = 15.0) -> bool:
        """Return True if the LLM accepted the prompt (stop button or new text appeared)."""
        baseline = await self._get_latest_response_text(tab)
        self._last_send_offsite_redirect = False
        timeout = float(os.getenv("SELENIUM_POST_SEND_TIMEOUT", str(timeout)))
        deadline = time.time() + timeout
        while time.time() < deadline:
            if await self._stop_button_present(tab):
                logger.debug("[zendriver] post_send_check: stop button visible — send accepted")
                return True
            cur = await self._get_latest_response_text(tab)
            if cur and cur != baseline:
                logger.debug("[zendriver] post_send_check: new response text appeared — send accepted")
                return True
            fb_url = await self._current_url(tab)
            if self.service_url and fb_url and not self._is_same_site(fb_url):
                logger.warning(f"[zendriver] post_send_check: off-site redirect detected ('{fb_url}') — returning False")
                self._last_send_offsite_redirect = True
                return False
            if not await self._send_button_present(tab):
                if await self._response_area_present(tab):
                    logger.debug("[zendriver] post_send_check fallback: send button absent and response area detected — generation in progress")
                    return True
                logger.debug("[zendriver] post_send_check fallback: send button absent but no response area detected yet")
            await asyncio.sleep(0.2)

        cur_url = await self._current_url(tab)
        if self.service_url and cur_url and not self._is_same_site(cur_url):
            logger.warning(f"[zendriver] post_send_check: timeout with off-site URL '{cur_url}' — redirect-stall detected")
            self._last_send_offsite_redirect = True
            return False

        logger.debug("[zendriver] post_send_check: timeout but URL looks ok — assuming slow model")
        return True

    async def _read_pre_send_text(self, tab: zd.Tab) -> str:
        try:
            text = await self._get_latest_response_text(tab)
        except Exception as exc:
            logger.debug("[zendriver] Could not read pre-send text: %s", exc)
            return ""
        return text if isinstance(text, str) else ""

    async def _wait_for_response(self, tab: zd.Tab, max_wait: int = 120, pre_send_text: str | None = None) -> str:
        """Wait for the LLM response to fully stream, then return its text.

        Strategy:
        1. Wait for the response container to appear and begin emitting text.
        2. After a 3 second grace delay, poll the container every 1 second.
        3. Track innerText length and childElementCount.
        4. If either metric changes, generation is still in progress.
        5. If both remain unchanged for 2 consecutive checks after activity starts,
           assume generation has finished.
        """
        await self._click_accept_buttons(tab, timeout=2.0)

        baseline = ""
        if self._use_baseline_comparison:
            baseline = await self._get_latest_response_text(tab)
            logger.debug("[zendriver] Baseline text length: %d", len(baseline))
        else:
            logger.debug("[zendriver] Skipping baseline comparison for this engine")

        self._generation_was_active = False
        logger.debug("[zendriver] watcher initial delay: 3.0s before response monitoring")
        await asyncio.sleep(3.0)

        _engine_max_wait = getattr(self, "_response_max_wait", None)
        effective_max_wait = int(_engine_max_wait) if _engine_max_wait else max_wait
        max_wait = int(os.getenv("SELENIUM_RESPONSE_MAX_WAIT", str(effective_max_wait)))
        wait_start = time.time()
        deadline = wait_start + max_wait

        previous_text_length = -1
        previous_child_count = -1
        stable_counter = 0
        iteration = 0
        last_container = None
        last_activity_time = time.time()
        current_text_length = 0
        current_child_count = 0

        hard_cap = max_wait + max(30, int(max_wait * 0.5))

        while time.time() < deadline and (time.time() - wait_start) < hard_cap:
            iteration += 1
            container, selector = await self._find_response_container_element(tab)
            if container is None:
                logger.debug("[zendriver] watcher iteration=%d no response container element found", iteration)
                self._log_response_container_diagnostics(
                    None, 0, 0, previous_text_length, previous_child_count, stable_counter, iteration,
                )
            else:
                last_container = container
                current_text_length, current_child_count = await self._get_response_container_stats(container)
                self._log_response_container_diagnostics(
                    selector, current_text_length, current_child_count,
                    previous_text_length, previous_child_count, stable_counter, iteration,
                )

                if previous_text_length != -1 and previous_child_count != -1:
                    if current_text_length != previous_text_length or current_child_count != previous_child_count:
                        stable_counter = 0
                        self._generation_was_active = True
                        last_activity_time = time.time()
                    elif current_text_length > 0 or current_child_count > 0:
                        stable_counter += 1
                    else:
                        stable_counter = 0
                previous_text_length = current_text_length
                previous_child_count = current_child_count

                if stable_counter >= 2:
                    response = await self._extract_response_text_from_element(container)
                    if (
                        response and pre_send_text and not self._generation_was_active
                        and response.strip() == pre_send_text.strip()
                    ):
                        raise RuntimeError(
                            "Stale response: the text is unchanged from before the "
                            "prompt was sent and no generation activity was observed"
                        )
                    if response:
                        logger.debug("[zendriver] watcher detected stable response after %d iterations", iteration)
                        return response

            silent_freeze_threshold = self._silent_freeze_threshold
            if silent_freeze_threshold is None:
                silent_freeze_threshold = float(os.getenv("SELENIUM_SILENT_FREEZE_THRESHOLD", "10"))
            stop_visible = await self._stop_button_present(tab)
            is_silent_freeze = stop_visible and (time.time() - last_activity_time) > silent_freeze_threshold
            is_ui_stuck = stop_visible and stable_counter >= 2 and (current_text_length > 0 or current_child_count > 0)
            no_progress_grace = max(silent_freeze_threshold * 2, 30.0)
            is_no_progress_freeze = (
                not self._generation_was_active
                and current_text_length == 0
                and current_child_count == 0
                and (time.time() - wait_start) > no_progress_grace
            )

            if is_silent_freeze or is_ui_stuck or is_no_progress_freeze:
                freeze_reason = "silent freeze" if is_silent_freeze else ("UI stuck after completion" if is_ui_stuck else "no-progress freeze (no response detected)")
                logger.warning(
                    "[zendriver] %s detected: Stop button visible but no proper UI transition (inactive %.1fs, text_len=%d, stable=%d)",
                    freeze_reason, time.time() - last_activity_time, current_text_length, stable_counter,
                )
                if self._page_refresh_attempts < self._max_page_refresh_attempts:
                    refreshed = False
                    try:
                        logger.info(
                            "[zendriver] Attempting page refresh for %s recovery (attempt %d/%d)",
                            freeze_reason, self._page_refresh_attempts + 1, self._max_page_refresh_attempts,
                        )
                        await tab.reload(ignore_cache=True)
                        await self._wait_for_page_ready(tab, timeout=30.0)
                        self._cached_prompt_selector = None
                        self._cached_send_selector = None
                        self._page_refresh_attempts += 1
                        refreshed = True
                    except Exception as refresh_err:
                        logger.error("[zendriver] Page refresh failed: %s", refresh_err)
                    if refreshed:
                        raise RuntimeError(
                            "page_refresh_required: %s — page refreshed, retry without browser reset" % freeze_reason
                        )
                raise asyncio.TimeoutError("%s detected during generation" % freeze_reason)

            error_text = await self._get_error_indicator_text(tab)
            if error_text:
                try:
                    debug_mode.record_event(
                        "error", getattr(self, "ENGINE_NAME", "default"),
                        stage="response_wait", detail="error_indicator_detected", text=error_text[:500],
                    )
                except Exception:
                    pass
                if self._page_refresh_attempts < self._max_page_refresh_attempts:
                    recovered = False
                    try:
                        logger.info(
                            "[zendriver] Engine error indicator detected — navigating to service home to recover (attempt %d/%d): %s",
                            self._page_refresh_attempts + 1, self._max_page_refresh_attempts, error_text[:120],
                        )
                        await tab.get(self.service_url)
                        await self._wait_for_page_ready(tab, timeout=30.0)
                        self._cached_prompt_selector = None
                        self._cached_send_selector = None
                        self._page_refresh_attempts += 1
                        recovered = True
                    except Exception as nav_err:
                        logger.error("[zendriver] Navigation to service home after engine error failed: %s", nav_err)
                    if recovered:
                        self._skip_split_for_next = False
                        raise RuntimeError(
                            "page_refresh_required: engine error indicator detected — returned to "
                            "service home, resend full prompt: %s" % error_text[:200]
                        )
                raise RuntimeError(
                    "selenium_response_detection_timeout: engine error indicator detected during response wait: %s" % error_text[:200]
                )

            if await self._is_limit_present(tab):
                raise RuntimeError("selenium_response_detection_timeout: service limit detected during response wait (quota/rate-limit page shown)")
            if await self._is_captcha_present(tab):
                raise RuntimeError("selenium_response_detection_timeout: CAPTCHA detected during response wait")

            await asyncio.sleep(1.0)

        if last_container is not None:
            result = await self._extract_response_text_from_element(last_container)
        else:
            result = await self._get_latest_response_text(tab)

        if result:
            logger.warning("[zendriver] Response wait timed out, returning best-effort result")
            return result

        raise RuntimeError(
            "selenium_response_detection_timeout: no new response text appeared in expected selectors within the allotted time"
        )

    async def _click_accept_buttons(self, tab: zd.Tab, timeout: float = 2.0) -> None:
        """Click any configured accept buttons that appear before continuing."""
        if not self.accept_button_selectors:
            return

        deadline = time.time() + float(timeout)
        clicked_selectors: set[str] = set()
        while time.time() < deadline:
            clicked_any = False
            for sel in self.accept_button_selectors:
                if sel in clicked_selectors:
                    continue
                buttons = await self._query_all(tab, sel)
                for button in buttons:
                    try:
                        if await self._element_is_displayed(button) and await self._element_is_enabled(button):
                            await button.click()
                            clicked_any = True
                            clicked_selectors.add(sel)
                            logger.debug(f"[zendriver] Clicked accept button with selector: {sel}")
                    except Exception:
                        continue

            if clicked_any:
                await asyncio.sleep(0.25)
                continue
            if clicked_selectors:
                return
            await asyncio.sleep(0.25)

    # ------------------------------------------------------------------ /helpers

    async def stop(self) -> None:
        """Detach from the shared browser and persist cookies.

        The shared Chrome process is **not** stopped here — other engines may
        still need it. Call :func:`shutdown_shared_driver` (via
        ``EngineManager.stop_all``) to actually terminate Chrome.
        """
        try:
            if self.driver is not None:
                self._save_cookies()
        except Exception as e:
            logger.warning(f"[zendriver] stop() error: {e}")
        finally:
            self.driver = None
            self._initialized = False
            self._cookies_restored = False
