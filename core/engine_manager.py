"""EngineManager — discovers, registers, and manages Selenium LLM engines.

Engines are loaded from the ``engines/`` directory (relative to this file's
project root) and can be defined in two ways:

JSON definition (recommended for simple engines)
    Any ``*.json`` file inside the ``engines/`` directory is loaded as a
    :class:`~core.json_engine.JsonEngine`.  See ``DEVELOPERS.md`` for the
    full schema.

Python definition (required for complex / custom logic)
    Any ``*.py`` file (excluding files whose name starts with ``_``) inside the
    ``engines/`` directory is imported dynamically.  The module must expose
    **exactly one** class that:

    * inherits from :class:`~core.zendriver_llm_base.ZendriverLLMBase`
    * defines a class-level ``ENGINE_NAME: str`` attribute
    * optionally defines ``ENGINE_ALIASES: list[str]``

Both file types can coexist — a ``.py`` engine with the same name as a
``.json`` engine is ignored (JSON takes precedence).

If no ``engines/`` directory exists (or it is empty), the manager starts
with no engines registered and logs a warning.
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import logging
import math
import os
import subprocess
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, Optional

from core import proc_utils
from core.errors import EngineUnavailableError, LoginRequiredError

if TYPE_CHECKING:
    from core.zendriver_llm_base import ZendriverLLMBase

logger = logging.getLogger("engine_manager")

# Root of the project (parent of the ``core/`` package directory).
_PROJECT_ROOT = Path(__file__).parent.parent
_ENGINES_DIR = _PROJECT_ROOT / "engines"

# Key under which the default engine is persisted, so a default chosen through
# the API survives a restart instead of reverting to whichever engine sorts
# first in engines/.
_DEFAULT_ENGINE_SETTING = "default_engine"


# ---------------------------------------------------------------------------
# Descriptor (lightweight — no browser started)
# ---------------------------------------------------------------------------


@dataclass
class EngineDescriptor:
    """Metadata about an engine without instantiating it."""

    name: str
    aliases: list[str]
    display_name: str
    service_url: str
    models: dict[str, int]
    default_model: str
    source: str  # "json" | "python"
    source_path: str  # filesystem path
    allow_unlogged: bool = False
    notes: str | None = None
    # Maximum number of parallel workers for this engine.  1 = fully serial
    # (default).  Values > 1 are reserved for future parallel-session support;
    # the queue infrastructure already handles them correctly.
    max_workers: int = 1
    media_capabilities: list[str] = field(default_factory=list)
    media_support: dict[str, Any] = field(default_factory=dict)

    def limits_dict(self) -> dict:
        """Return interface-limits metadata without starting a browser."""
        model_name = self.default_model
        limit = self.models.get(model_name) or self.models.get("default") or 10000
        return {"max_prompt_chars": limit, "model_name": model_name}

    def supported_models_list(self) -> list[str]:
        """Return the list of model names known at descriptor level."""
        return list(self.models.keys())

    def to_dict(self) -> dict:
        data = {
            "name": self.name,
            "display_name": self.display_name,
            "aliases": self.aliases,
            "service_url": self.service_url,
            "models": self.models,
            "default_model": self.default_model,
            "allow_unlogged": self.allow_unlogged,
            "max_workers": self.max_workers,
            "source": self.source,
            "source_path": self.source_path,
        }
        if self.notes:
            data["notes"] = self.notes
        if self.media_support:
            data["media_support"] = self.media_support
        data["media_capabilities"] = list(self.media_capabilities)
        return data


# ---------------------------------------------------------------------------
# Engine scanning helpers
# ---------------------------------------------------------------------------


def _supports_media_capability(cfg_item: Any) -> bool:
    if not isinstance(cfg_item, dict):
        return False

    limits = cfg_item.get("limits")
    if isinstance(limits, dict):
        for value in limits.values():
            if value == -1:
                return True
            if isinstance(value, int) and value > 0:
                return True

    supported_models = cfg_item.get("supported_models")
    if isinstance(supported_models, list) and len(supported_models) > 0:
        return True

    return False


def _scan_json(path: Path) -> Optional[EngineDescriptor]:
    try:
        with path.open(encoding="utf-8") as fh:
            cfg = json.load(fh)
        name = cfg.get("name")
        media_support = cfg.get("media_support", {}) or {}
        if not name:
            logger.warning(f"[engine_manager] JSON engine without 'name': {path}")
            return None
        media_support = cfg.get("media_support", {}) or {}
        capabilities = [
            key
            for key in ("image", "audio", "document")
            if key in media_support and _supports_media_capability(media_support[key])
        ]
        return EngineDescriptor(
            name=name,
            aliases=list(cfg.get("aliases", [name])),
            display_name=cfg.get("display_name", name),
            service_url=cfg.get("service_url", ""),
            models=dict(cfg.get("models", {"default": 10000})),
            default_model=cfg.get("default_model", "default"),
            allow_unlogged=bool(cfg.get("allow_unlogged", False)),
            notes=cfg.get("notes"),
            max_workers=int(cfg.get("max_workers", 1)),
            source="json",
            source_path=str(path),
            media_capabilities=capabilities,
            media_support=media_support,
        )
    except Exception as exc:
        logger.warning(f"[engine_manager] Failed to scan JSON engine {path}: {exc}")
        return None


def _scan_python(path: Path) -> Optional[EngineDescriptor]:
    if path.name.startswith("_"):
        return None  # private / template files
    from core.zendriver_llm_base import ZendriverLLMBase  # lazy import to avoid heavy deps at startup
    try:
        spec = importlib.util.spec_from_file_location(f"engines._dyn.{path.stem}", path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[attr-defined]

        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                obj is not ZendriverLLMBase
                and issubclass(obj, ZendriverLLMBase)
                and hasattr(obj, "ENGINE_NAME")
            ):
                engine_name: str = obj.ENGINE_NAME
                return EngineDescriptor(
                    name=engine_name,
                    aliases=list(getattr(obj, "ENGINE_ALIASES", [engine_name])),
                    display_name=getattr(obj, "ENGINE_DISPLAY_NAME", engine_name),
                    service_url=getattr(obj, "ENGINE_SERVICE_URL", ""),
                    models=dict(getattr(obj, "ENGINE_MODELS", {"default": 10000})),
                    default_model=getattr(obj, "ENGINE_DEFAULT_MODEL", "default"),
                    allow_unlogged=bool(getattr(obj, "ENGINE_ALLOW_UNLOGGED", False)),
                    max_workers=int(getattr(obj, "ENGINE_MAX_WORKERS", 1)),
                    source="python",
                    source_path=str(path),
                )
        logger.warning(
            f"[engine_manager] No ZendriverLLMBase subclass with ENGINE_NAME in {path}"
        )
        return None
    except Exception as exc:
        logger.warning(f"[engine_manager] Failed to scan Python engine {path}: {exc}")
        return None


def scan_engines(engines_dir: Path) -> dict[str, EngineDescriptor]:
    """Scan *engines_dir* and return a name→descriptor mapping.

    JSON files take precedence over Python files with the same engine name.
    """
    descriptors: dict[str, EngineDescriptor] = {}

    if not engines_dir.is_dir():
        logger.warning(
            f"[engine_manager] engines/ directory not found at {engines_dir} — "
            "falling back to built-in engines"
        )
        return descriptors

    # --- pass 1: JSON ---
    for path in sorted(engines_dir.glob("*.json")):
        desc = _scan_json(path)
        if desc and desc.name not in descriptors:
            descriptors[desc.name] = desc
            for alias in desc.aliases:
                if alias != desc.name:
                    logger.debug(
                        f"[engine_manager] Registered JSON engine '{desc.name}' "
                        f"(alias: {alias})"
                    )
            logger.info(
                f"[engine_manager] Loaded JSON engine '{desc.name}' from {path.name}"
            )

    # --- pass 2: Python ---
    for path in sorted(engines_dir.glob("*.py")):
        desc = _scan_python(path)
        if desc and desc.name not in descriptors:
            descriptors[desc.name] = desc
            logger.info(
                f"[engine_manager] Loaded Python engine '{desc.name}' from {path.name}"
            )
        elif desc:
            logger.debug(
                f"[engine_manager] Python engine '{desc.name}' skipped "
                f"(already registered from JSON)"
            )

    return descriptors


# ---------------------------------------------------------------------------
# Engine instantiation
# ---------------------------------------------------------------------------


def _instantiate(descriptor: EngineDescriptor, **kwargs) -> "ZendriverLLMBase":
    """Create a live engine instance from its descriptor."""
    from core.zendriver_llm_base import ZendriverLLMBase  # lazy import to avoid heavy deps at startup
    if descriptor.source == "json":
        from core.json_engine import JsonEngine

        return JsonEngine(Path(descriptor.source_path), **kwargs)

    if descriptor.source == "python":
        spec = importlib.util.spec_from_file_location(
            f"engines._dyn.{descriptor.name}", descriptor.source_path
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                obj is not ZendriverLLMBase
                and issubclass(obj, ZendriverLLMBase)
                and getattr(obj, "ENGINE_NAME", None) == descriptor.name
            ):
                return obj(**kwargs)
        raise RuntimeError(
            f"Python engine class for '{descriptor.name}' not found in "
            f"{descriptor.source_path}"
        )

    raise ValueError(f"Unknown engine source type: {descriptor.source!r}")


# ---------------------------------------------------------------------------
# FIFO queue helpers
# ---------------------------------------------------------------------------


@dataclass
class _PromptResult:
    """Result of a queued prompt job."""

    text: str
    model_name: str


@dataclass
class _PromptJob:
    """A single prompt task placed on a per-engine FIFO queue."""

    prompt: str
    future: asyncio.Future  # type: ignore[type-arg]
    media: list[Any] = field(default_factory=list)
    timeout: int | None = None
    agent_mode: bool = False


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass
class EngineHealth:
    """Per-engine request outcomes and circuit-breaker state (no browser access).

    The breaker exists because a failing engine used to keep its queue full of
    requests that each burned ~8 minutes and reset the shared browser several
    times. Resetting the browser is destructive: Chromium flushes cookies
    lazily, so a login the operator completes by hand in that browser was lost
    before it reached disk (848 relaunches in a day, no sign-in survived).
    While the breaker is open nothing is queued and nothing touches the browser.
    """

    consecutive_failures: int = 0
    last_success_at: float | None = None
    last_error: str | None = None
    last_error_at: float | None = None
    breaker_open: bool = False
    breaker_reason: str | None = None  # "login" | "failures"
    breaker_opened_at: float | None = None
    retry_at: float | None = None  # "failures" only: when one trial is allowed
    cooldown: float = 0.0
    trial_in_flight: bool = False
    recovery_checks: int = 0  # consecutive "looks logged in" peeks ("login" only)


# Request errors caused by the caller's input, not by the engine's health.
_CLIENT_ERRORS: tuple[type[BaseException], ...] = (ValueError, TypeError, NotImplementedError)


# ---------------------------------------------------------------------------
# EngineManager singleton
# ---------------------------------------------------------------------------


class EngineManager:
    """Thread-safe singleton that manages engine lifecycle and discovery."""

    _instance: Optional["EngineManager"] = None
    _lock: Lock = Lock()

    def __init__(self) -> None:
        self.engines: dict[str, ZendriverLLMBase] = {}
        self.active_engine: Optional[ZendriverLLMBase] = None
        self.default_engine: str | None = None
        self._descriptors: dict[str, EngineDescriptor] = {}
        self._alias_map: dict[str, str] = {}  # alias → canonical name
        # Per-engine FIFO queues and their worker tasks.
        # Each engine gets exactly max_workers coroutine workers consuming
        # the queue; this serialises requests (max_workers=1) while
        # providing the scaffolding for future parallel-session support.
        self._job_queues: dict[str, asyncio.Queue[_PromptJob]] = {}
        self._queue_workers: dict[str, list[asyncio.Task]] = {}  # type: ignore[type-arg]
        # Periodic orphan cleanup task — kills chromedriver/chromium processes
        # that are older than the configured threshold and no longer owned.
        self._orphan_cleanup_task: asyncio.Task | None = None  # type: ignore[type-arg]
        # Circuit breaker / health bookkeeping and the passive login watcher.
        self._health: dict[str, EngineHealth] = {}
        self._login_watch_task: asyncio.Task | None = None  # type: ignore[type-arg]
        self._load_descriptors()
        self._restore_default_engine()

    @classmethod
    def get(cls) -> "EngineManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = EngineManager()
        return cls._instance

    # ---------------------------------------------------------------------- discovery

    def _load_descriptors(self) -> None:
        """Scan engines/ and build the descriptor + alias maps."""
        self._descriptors.clear()
        self._alias_map.clear()

        raw = scan_engines(_ENGINES_DIR)
        if not raw:
            logger.warning(
                "[engine_manager] No engines found in engines/ — no engines registered"
            )

        for desc in raw.values():
            if desc.name in self._descriptors:
                continue  # already registered
            self._descriptors[desc.name] = desc
            for alias in desc.aliases:
                self._alias_map[alias.strip().lower()] = desc.name

    def reload_engines(self) -> list[dict]:
        """Re-scan the engines/ directory and refresh the registry.

        Already-running engine instances are **not** stopped — they
        remain available until :meth:`stop_all` is called.
        """
        logger.info("[engine_manager] Reloading engine registry…")
        self._load_descriptors()
        return self.list_engines()

    def list_engines(self) -> list[dict]:
        """Return metadata for all registered engines (no browser started)."""
        seen: set[str] = set()
        result = []
        for desc in self._descriptors.values():
            if desc.name not in seen:
                entry = desc.to_dict()
                # Include queue depth for live engines
                queue = self._job_queues.get(desc.name)
                entry["queue_depth"] = queue.qsize() if queue else 0
                result.append(entry)
                seen.add(desc.name)
        return result

    # ---------------------------------------------------------------------- access

    def _resolve(self, name: str) -> str:
        """Resolve an alias or name to the canonical engine name."""
        key = name.strip().lower()
        if key in self._alias_map:
            return self._alias_map[key]
        # Also try direct name match
        if key in self._descriptors:
            return key
        raise ValueError(f"Unknown engine: '{name}'")

    def get_engine(self, name: str) -> ZendriverLLMBase:
        """Return (and lazy-instantiate) the engine identified by *name* or an alias."""
        canonical = self._resolve(name)
        if canonical not in self.engines:
            desc = self._descriptors[canonical]
            logger.info(f"[engine_manager] Instantiating engine '{canonical}'…")
            self.engines[canonical] = _instantiate(desc)
        return self.engines[canonical]

    def set_active_engine(self, name: str) -> ZendriverLLMBase:
        engine = self.get_engine(name)
        self.active_engine = engine
        return engine

    def _restore_default_engine(self) -> None:
        """Restore the default engine from persistent storage or the environment.

        Without this the attribute starts empty on every boot and
        :meth:`get_default_engine` silently falls back to whichever engine sorts
        first, discarding the operator's choice.
        """
        from db.db import get_setting

        candidate = get_setting(_DEFAULT_ENGINE_SETTING) or os.getenv(
            "SELENIUM_DEFAULT_ENGINE"
        )
        if not candidate:
            return
        try:
            self.default_engine = self._resolve(candidate)
        except ValueError:
            logger.warning(
                "[engine_manager] Stored default engine '%s' is not among the "
                "registered engines; ignoring it",
                candidate,
            )
        else:
            logger.info(
                "[engine_manager] Restored default engine '%s'", self.default_engine
            )

    def set_default_engine(self, name: str) -> str:
        canonical = self._resolve(name)
        if canonical not in self._descriptors:
            raise ValueError(f"Unknown engine: '{name}'")
        self.default_engine = canonical

        from db.db import set_setting

        if not set_setting(_DEFAULT_ENGINE_SETTING, canonical):
            logger.warning(
                "[engine_manager] Default engine '%s' applies to this process "
                "only; it could not be persisted and will not survive a restart",
                canonical,
            )
        return canonical

    def get_default_engine(self) -> str:
        if self.default_engine and self.default_engine in self._descriptors:
            return self.default_engine
        if self._descriptors:
            # No operator choice available: fall back to the first registered
            # engine, but say so — silently answering with an arbitrary engine
            # is how a misconfiguration goes unnoticed.
            fallback = next(iter(self._descriptors))
            logger.warning(
                "[engine_manager] No default engine configured; falling back to "
                "'%s' (first registered). Set one via POST /api/engines/default "
                "or the SELENIUM_DEFAULT_ENGINE environment variable.",
                fallback,
            )
            return fallback
        raise ValueError("No engines registered")

    def get_active_engine(self) -> ZendriverLLMBase:
        if not self.active_engine:
            raise RuntimeError("No active engine set")
        return self.active_engine

    def get_descriptor(self, name: str) -> Optional[EngineDescriptor]:
        """Return the descriptor for *name* (or an alias) without starting a browser."""
        try:
            canonical = self._resolve(name)
            return self._descriptors.get(canonical)
        except ValueError:
            return None

    # ---------------------------------------------------------------------- health / circuit breaker

    def _health_for(self, canonical: str) -> EngineHealth:
        return self._health.setdefault(canonical, EngineHealth())

    @staticmethod
    def _trial_due(h: EngineHealth, now: float) -> bool:
        return (
            h.breaker_reason == "failures"
            and not h.trial_in_flight
            and h.retry_at is not None
            and now >= h.retry_at
        )

    def _unavailable(self, canonical: str, h: EngineHealth, now: float) -> EngineUnavailableError:
        if h.breaker_reason == "login":
            retry_after = _env_int("ZEN_LOGIN_RETRY_AFTER", 60)
        elif h.trial_in_flight:
            retry_after = 30
        else:
            retry_after = max(1, math.ceil((h.retry_at or now) - now))
        return EngineUnavailableError(
            canonical, h.breaker_reason or "failures", retry_after, h.last_error
        )

    def assert_available(self, name: str) -> None:
        """Raise :class:`EngineUnavailableError` if the breaker would reject a request.

        Side-effect free, so HTTP handlers can reject *before* they start a
        streaming response. :meth:`enqueue` performs the real admission.
        """
        canonical = self._resolve(name)
        h = self._health.get(canonical)
        if h is None or not h.breaker_open:
            return
        now = _time.time()
        if not self._trial_due(h, now):
            raise self._unavailable(canonical, h, now)

    def _admit(self, canonical: str) -> None:
        h = self._health_for(canonical)
        if not h.breaker_open:
            return
        now = _time.time()
        if self._trial_due(h, now):
            h.trial_in_flight = True
            logger.info("[engine_manager] '%s': breaker half-open, admitting one trial request", canonical)
            return
        raise self._unavailable(canonical, h, now)

    def _record_success(self, canonical: str) -> None:
        h = self._health_for(canonical)
        if h.breaker_open:
            logger.info("[engine_manager] '%s': trial request succeeded, breaker closed", canonical)
        h.consecutive_failures = 0
        h.last_success_at = _time.time()
        self._close_breaker(h)

    @staticmethod
    def _close_breaker(h: EngineHealth) -> None:
        h.breaker_open = False
        h.breaker_reason = None
        h.breaker_opened_at = None
        h.retry_at = None
        h.cooldown = 0.0
        h.trial_in_flight = False
        h.recovery_checks = 0

    def _record_failure(self, canonical: str, exc: BaseException) -> None:
        if isinstance(exc, (EngineUnavailableError, *_CLIENT_ERRORS)):
            return
        h = self._health_for(canonical)
        h.last_error = str(exc)[:300]
        h.last_error_at = _time.time()
        if isinstance(exc, LoginRequiredError):
            self._trip(canonical, "login")
            return
        if h.breaker_open:
            # The trial request failed: back off further. (A failure surfacing
            # while the breaker is open for "login" changes nothing.)
            if h.breaker_reason == "failures":
                self._trip(canonical, "failures", escalate=True)
            return
        h.consecutive_failures += 1
        if h.consecutive_failures >= max(1, _env_int("ZEN_BREAKER_THRESHOLD", 3)):
            self._trip(canonical, "failures")

    def _trip(self, canonical: str, reason: str, escalate: bool = False) -> None:
        h = self._health_for(canonical)
        now = _time.time()
        if not h.breaker_open:
            h.breaker_opened_at = now
        h.breaker_open = True
        h.breaker_reason = reason
        h.trial_in_flight = False
        h.recovery_checks = 0
        if reason == "failures":
            base = _env_int("ZEN_BREAKER_COOLDOWN", 300)
            cap = _env_int("ZEN_BREAKER_MAX_COOLDOWN", 3600)
            h.cooldown = float(min(max(h.cooldown * 2, base) if escalate else base, cap))
            h.retry_at = now + h.cooldown
        else:
            h.retry_at = None
        logger.error(
            "[engine_manager] '%s': circuit breaker OPEN (%s) after %d consecutive failure(s); "
            "requests are rejected with 503 and the browser is left alone. Last error: %s",
            canonical, reason, h.consecutive_failures, h.last_error,
        )
        self._fail_queued(canonical, self._unavailable(canonical, h, now))

    def _fail_queued(self, canonical: str, error: EngineUnavailableError) -> None:
        """Fail every job still waiting so callers get an answer now, not in hours."""
        queue = self._job_queues.get(canonical)
        if queue is None:
            return
        failed = 0
        while True:
            try:
                job = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if not job.future.done():
                job.future.set_exception(error)
            queue.task_done()
            failed += 1
        if failed:
            logger.warning("[engine_manager] '%s': failed %d queued request(s) immediately", canonical, failed)

    def reset_breaker(self, name: str) -> bool:
        """Close the breaker and clear the failure streak. True if it was open."""
        canonical = self._resolve(name)
        h = self._health_for(canonical)
        was_open = h.breaker_open
        h.consecutive_failures = 0
        self._close_breaker(h)
        if was_open:
            logger.info("[engine_manager] '%s': circuit breaker closed manually/by login watcher", canonical)
        return was_open

    def pause_for_login(self, name: str) -> None:
        """Stop sending requests to the browser so a human can sign in undisturbed.

        Opens the "login" breaker: queued requests fail at once, new ones get a
        503 and nothing navigates or resets the shared tab. It closes again when
        the login watcher sees the page signed in, or via :meth:`reset_breaker`.
        """
        canonical = self._resolve(name)
        h = self._health_for(canonical)
        h.last_error = "paused for manual sign-in"
        h.last_error_at = _time.time()
        self._trip(canonical, "login")

    def login_blocked(self, name: str) -> bool:
        """True while the breaker is open because a manual sign-in is needed."""
        h = self._health.get(self._resolve(name))
        return bool(h and h.breaker_open and h.breaker_reason == "login")

    def note_login_state(self, name: str, logged_in: bool) -> None:
        """Close a "login" breaker once an explicit check reports a logged-in state.

        Deliberately ignores "failures" breakers: being logged in says nothing
        about whether the engine's selectors/page flow work again.
        """
        if logged_in and self.login_blocked(name):
            self.reset_breaker(name)

    def health_snapshot(self) -> list[dict[str, Any]]:
        """Engine health for the status endpoint. Never touches the browser or DB."""
        now = _time.time()
        result: list[dict[str, Any]] = []
        for desc in self._descriptors.values():
            canonical = desc.name
            h = self._health.get(canonical) or EngineHealth()
            engine = self.engines.get(canonical)
            queue = self._job_queues.get(canonical)
            busy = [
                now - started
                for t in self._queue_workers.get(canonical, [])
                if (started := getattr(t, "_job_started_at", None)) is not None
            ]
            login_state = "unknown"
            checked_at = None
            if engine is not None:
                last = getattr(engine, "_last_login_state", None)
                checked_at = getattr(engine, "_last_login_checked_at", None)
                if last is not None:
                    login_state = "logged" if last else "unlogged"
            if h.breaker_open:
                state = "down"
            elif h.consecutive_failures > 0:
                state = "degraded"
            elif h.last_success_at is not None:
                state = "ok"
            else:
                state = "idle"
            retry_in = None
            if h.breaker_open and h.breaker_reason == "failures" and h.retry_at is not None:
                retry_in = max(0, math.ceil(h.retry_at - now))
            result.append(
                {
                    "name": canonical,
                    "state": state,
                    "queue_depth": queue.qsize() if queue else 0,
                    "worker_busy_seconds": round(max(busy), 1) if busy else None,
                    "consecutive_failures": h.consecutive_failures,
                    "last_success_at": h.last_success_at,
                    "last_error": h.last_error,
                    "last_error_at": h.last_error_at,
                    "login_state": login_state,
                    "login_checked_at": checked_at,
                    "breaker": {
                        "open": h.breaker_open,
                        "reason": h.breaker_reason,
                        "opened_at": h.breaker_opened_at,
                        "retry_in_seconds": retry_in,
                    },
                }
            )
        return result

    # ---------------------------------------------------------------------- FIFO queue

    def _get_or_create_queue(self, canonical: str) -> asyncio.Queue[_PromptJob]:
        if canonical not in self._job_queues:
            self._job_queues[canonical] = asyncio.Queue()
        return self._job_queues[canonical]

    def _ensure_workers(self, canonical: str) -> None:
        """Start worker coroutines for *canonical* if they are not already running.

        ``max_workers`` from the descriptor controls the pool size.  With the
        default value of 1 the engine is fully serial: requests are processed
        one at a time in FIFO order.

        Additionally detects workers that have been stuck on a single job
        for longer than a safety threshold and cancels them so a fresh
        worker can take over.
        """
        desc = self._descriptors.get(canonical)
        target = max(1, desc.max_workers if desc else 1)

        existing = self._queue_workers.get(canonical, [])
        # Prune finished workers before counting live ones
        alive = [t for t in existing if not t.done()]

        # Detect stuck workers: if a worker has been processing a single job
        # for longer than the stuck threshold, cancel it.
        # Threshold = SELENIUM_TOTAL_TIMEOUT + 30s grace (default 330s).
        stuck_timeout = int(
            __import__("os").getenv("SELENIUM_WORKER_STUCK_TIMEOUT", "330")
        )
        still_alive: list[asyncio.Task] = []  # type: ignore[type-arg]
        for t in alive:
            started = getattr(t, "_job_started_at", None)
            if started is not None and (_time.time() - started) > stuck_timeout:
                logger.warning(
                    "[engine_manager] Worker for '%s' stuck for >%ds, cancelling",
                    canonical,
                    stuck_timeout,
                )
                t.cancel()
            else:
                still_alive.append(t)
        alive = still_alive

        self._queue_workers[canonical] = alive

        needed = target - len(alive)
        for _ in range(needed):
            task = asyncio.ensure_future(self._queue_worker_loop(canonical))
            self._queue_workers[canonical].append(task)

    async def _queue_worker_loop(self, engine_name: str) -> None:
        """Coroutine that processes prompt jobs for *engine_name* sequentially."""
        queue = self._get_or_create_queue(engine_name)
        current_task = asyncio.current_task()
        while True:
            job = await queue.get()
            queued_at = getattr(job, "_queued_at", None)
            worker_start = _time.time()
            # Tag the task with the job-start timestamp for stuck detection.
            if current_task is not None:
                current_task._job_started_at = worker_start  # type: ignore[attr-defined]
            if queued_at is not None:
                logger.info(
                    f"[timing] queue_wait ({engine_name}): "
                    f"{worker_start - queued_at:.2f}s"
                )
            try:
                engine = self.get_engine(engine_name)  # lazy-init browser here
                self.active_engine = engine
                try:
                    result_text = await engine.generate_response(
                        job.prompt, job.media, timeout=job.timeout,
                        agent_mode=job.agent_mode,
                    )
                except TypeError:
                    try:
                        result_text = await engine.generate_response(
                            job.prompt, job.media, timeout=job.timeout,
                        )
                    except TypeError:
                        result_text = await engine.generate_response(job.prompt)
                model_name = engine.get_current_model()
                elapsed = _time.time() - worker_start
                logger.info(
                    f"[timing] worker_total ({engine_name}): {elapsed:.2f}s"
                )
                self._record_success(engine_name)
                if not job.future.done():
                    job.future.set_result(_PromptResult(text=result_text, model_name=model_name))
            except asyncio.CancelledError:
                # A cancelled trial must not wedge the breaker in "trial in flight".
                self._health_for(engine_name).trial_in_flight = False
                if not job.future.done():
                    job.future.cancel()
                raise
            except Exception as exc:
                self._record_failure(engine_name, exc)
                if not job.future.done():
                    job.future.set_exception(exc)
            finally:
                queue.task_done()
                # Clear the job-start marker so stuck detection doesn't
                # trigger while the worker is idle waiting for the next job.
                if current_task is not None:
                    current_task._job_started_at = None  # type: ignore[attr-defined]

    async def enqueue(
        self,
        engine_name: str,
        prompt: str,
        media: list[Any] | None = None,
        timeout: int | None = None,
        agent_mode: bool = False,
    ) -> _PromptResult:
        """Submit *prompt* and optional media to the named engine's FIFO queue.

        The engine browser is started lazily by the worker, not by the HTTP
        handler.  Concurrent callers on the same engine are serialised
        automatically (max_workers=1 default).

        Parameters
        ----------
        timeout:
            Per-request total timeout in seconds.  Passed through to
            ``engine.generate_response()``.  ``None`` means use the
            engine/environment default (300 s).
        """
        canonical = self._resolve(engine_name)
        # Fail fast while the circuit breaker is open: queueing behind a dead
        # engine only delays the inevitable error by minutes per request.
        self._admit(canonical)
        queue = self._get_or_create_queue(canonical)
        loop = asyncio.get_event_loop()
        future: asyncio.Future[_PromptResult] = loop.create_future()
        job = _PromptJob(
            prompt=prompt, future=future, media=media or [], timeout=timeout,
            agent_mode=agent_mode,
        )
        job._queued_at = _time.time()  # type: ignore[attr-defined]
        self._ensure_workers(canonical)
        await queue.put(job)
        return await future

    async def drain_queues(self) -> None:
        """Cancel all worker tasks and drain pending jobs from every queue.

        Called during reset so in-flight and queued requests are terminated
        cleanly before the engine instances are destroyed.
        """
        # Cancel all workers
        all_workers = [
            t for tasks in self._queue_workers.values() for t in tasks
        ]
        for t in all_workers:
            t.cancel()
        if all_workers:
            await asyncio.gather(*all_workers, return_exceptions=True)
        self._queue_workers.clear()

        # Drain remaining queued jobs (cancel their futures)
        for queue in self._job_queues.values():
            while not queue.empty():
                try:
                    job = queue.get_nowait()
                    if not job.future.done():
                        job.future.cancel()
                    queue.task_done()
                except asyncio.QueueEmpty:
                    break
        self._job_queues.clear()

    # ---------------------------------------------------------------------- lifecycle

    def _protected_pids(self) -> set[int]:
        """Return the PIDs that must never be reaped as "orphans".

        The currently active shared browser (see
        core.zendriver_llm_base.get_shared_browser_pid) plus its whole
        process tree (renderer/GPU child processes have their own,
        independently-old start times, so protecting only the root PID would
        still let a long-open tab's renderer get killed out from under it).

        Regression: before this existed, `_cleanup_orphans` had no notion of
        "in use" at all -- it killed *any* chromium process older than
        SELENIUM_ORPHAN_AGE_THRESHOLD (default 600s), including the one
        actively serving requests. This was masked for a long time because
        the session-hard-timeout watchdog used to recycle the browser every
        ~300s by default, always well under the 600s orphan threshold; once
        that default was raised to hours (a long-lived shared browser is the
        intended normal state), the orphan sweep started SIGKILLing the live
        browser every 5 minutes, surfacing as an immediate CDP-transport
        "no close frame received or sent" error on whatever request was
        using it at the time.
        """
        from core.zendriver_llm_base import get_shared_browser_pid

        root = get_shared_browser_pid()
        if root is None:
            return set()
        protected = {root}
        frontier = [root]
        while frontier:
            pid = frontier.pop()
            try:
                result = subprocess.run(
                    ["pgrep", "-P", str(pid)],
                    check=False, capture_output=True, timeout=2, text=True,
                )
                children = [int(p) for p in result.stdout.split() if p.isdigit()]
            except Exception:
                children = []
            for child in children:
                if child not in protected:
                    protected.add(child)
                    frontier.append(child)
        return protected

    def _cleanup_orphans(self) -> int:
        """Find and kill orphaned chromedriver / chromium processes.

        Kills processes that match known patterns and are older than the
        configured ``SELENIUM_ORPHAN_AGE_THRESHOLD`` (default 600 s) --
        except the active shared browser's own process tree, see
        ``_protected_pids``.

        Returns the number of processes killed.
        """
        orphan_age = int(os.getenv("SELENIUM_ORPHAN_AGE_THRESHOLD", "600"))
        killed = 0
        seen: set[int] = set()
        protected = self._protected_pids()
        # "chromedriver"/"undetected_chromedriver" no longer exist in the
        # zendriver stack (no separate driver binary), kept here only so a
        # stray leftover from a not-yet-restarted older deployment still gets
        # reaped; "chromium"/"chrome" are what actually matches now.
        patterns = ["chromedriver", "undetected_chromedriver", "chromium", "chrome"]
        try:
            for pattern in patterns:
                try:
                    result = subprocess.run(
                        ["pgrep", "-f", pattern],
                        check=False, capture_output=True, timeout=5, text=True,
                    )
                    for pid_str in result.stdout.strip().split():
                        if not pid_str.isdigit():
                            continue
                        pid = int(pid_str)
                        if pid in protected or pid in seen:
                            continue
                        # "chromium" and "chrome" match the same process: count
                        # and kill it once. A zombie is already dead -- it only
                        # needs reaping (below), not another SIGKILL; pgrep -f
                        # keeps matching zombies, which is how the sweep used to
                        # "kill" the same 165 processes on every pass.
                        seen.add(pid)
                        if proc_utils.process_state(pid) == "Z":
                            continue
                        # Check process age via /proc/<pid>/stat
                        try:
                            with open(f"/proc/{pid}/stat", "r") as f:
                                stat = f.read()
                            # Field 22 is starttime (in clock ticks since boot)
                            parts = stat.split()
                            if len(parts) >= 22:
                                # Read uptime from /proc/uptime
                                with open("/proc/uptime", "r") as f:
                                    uptime_sec = float(f.read().split()[0])
                                # starttime is in clock ticks (typically 100/s)
                                clk_tck = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
                                starttime_ticks = int(parts[21])
                                starttime_sec = starttime_ticks / clk_tck
                                age_sec = uptime_sec - starttime_sec
                                if age_sec > orphan_age:
                                    logger.info(
                                        "[engine_manager] Killing orphan %s (PID %d, age %.0fs > %ds)",
                                        pattern, pid, age_sec, orphan_age,
                                    )
                                    # Kill process tree
                                    subprocess.run(
                                        ["pkill", "-9", "-P", str(pid)],
                                        check=False, capture_output=True, timeout=2,
                                    )
                                    subprocess.run(
                                        ["kill", "-9", str(pid)],
                                        check=False, capture_output=True, timeout=2,
                                    )
                                    proc_utils.reap_pid(pid, timeout=1.0)
                                    killed += 1
                        except (FileNotFoundError, PermissionError, ValueError, IndexError):
                            # Process already gone or inaccessible
                            pass
                except Exception:
                    pass
        except Exception:
            pass
        reaped = proc_utils.reap_zombie_children()
        if reaped:
            logger.info("[engine_manager] Reaped %d zombie child process(es)", reaped)
        if killed > 0:
            logger.warning(
                "[engine_manager] Orphan cleanup killed %d stale processes", killed
            )
        return killed

    async def _orphan_cleanup_loop(self) -> None:
        """Background coroutine that periodically cleans up orphaned processes."""
        interval = int(os.getenv("SELENIUM_CLEANUP_INTERVAL", "300"))
        logger.info(
            "[engine_manager] Orphan cleanup started (interval=%ds, age_threshold=%ds)",
            interval,
            int(os.getenv("SELENIUM_ORPHAN_AGE_THRESHOLD", "600")),
        )
        while True:
            try:
                await asyncio.sleep(interval)
                killed = await asyncio.to_thread(self._cleanup_orphans)
                if killed == 0:
                    logger.debug("[engine_manager] Orphan cleanup: no orphans found")
                # Hand freed heap back to the OS; RSS otherwise only ratchets up.
                await asyncio.to_thread(proc_utils.trim_memory)
            except asyncio.CancelledError:
                logger.info("[engine_manager] Orphan cleanup task cancelled")
                return
            except Exception as exc:
                logger.warning("[engine_manager] Orphan cleanup error: %s", exc)

    def start_orphan_cleanup(self) -> None:
        """Start the periodic orphan cleanup background task."""
        if self._orphan_cleanup_task is None or self._orphan_cleanup_task.done():
            self._orphan_cleanup_task = asyncio.ensure_future(self._orphan_cleanup_loop())
            logger.info("[engine_manager] Orphan cleanup task started")

    async def _login_watch_tick(self) -> None:
        """One pass of the passive login watcher.

        For each engine whose breaker is open because a sign-in is needed, peek
        (read-only, no navigation, no browser lock) at the page the operator is
        working on. Two consecutive "looks logged in" peeks close the breaker.
        """
        from core.zendriver_llm_base import _BROWSER_ACCESS_LOCK

        for name, h in list(self._health.items()):
            if not (h.breaker_open and h.breaker_reason == "login"):
                continue
            engine = self.engines.get(name)
            if engine is None or _BROWSER_ACCESS_LOCK.locked():
                continue
            try:
                state = await asyncio.wait_for(engine.peek_login_state(), timeout=8.0)
            except Exception:
                state = None
            if state is True:
                h.recovery_checks += 1
                if h.recovery_checks >= 2:
                    self.reset_breaker(name)
                    logger.info("[engine_manager] '%s': login detected, circuit breaker closed", name)
            else:
                h.recovery_checks = 0

    async def _login_watch_loop(self) -> None:
        interval = max(1, _env_int("ZEN_LOGIN_WATCH_INTERVAL", 15))
        while True:
            try:
                await asyncio.sleep(interval)
                await self._login_watch_tick()
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.warning("[engine_manager] Login watcher error: %s", exc)

    def start_login_watch(self) -> None:
        """Start the passive login watcher (see :meth:`_login_watch_tick`)."""
        if self._login_watch_task is None or self._login_watch_task.done():
            self._login_watch_task = asyncio.ensure_future(self._login_watch_loop())

    async def stop_login_watch(self) -> None:
        if self._login_watch_task and not self._login_watch_task.done():
            self._login_watch_task.cancel()
            try:
                await self._login_watch_task
            except asyncio.CancelledError:
                pass
        self._login_watch_task = None

    async def stop_orphan_cleanup(self) -> None:
        """Cancel the periodic orphan cleanup task."""
        if self._orphan_cleanup_task and not self._orphan_cleanup_task.done():
            self._orphan_cleanup_task.cancel()
            try:
                await self._orphan_cleanup_task
            except asyncio.CancelledError:
                pass
        self._orphan_cleanup_task = None
        logger.info("[engine_manager] Orphan cleanup task stopped")

    async def stop_all(self) -> None:
        """Save cookies for every engine, then quit the shared Chrome driver."""
        # First stop the background tasks
        await self.stop_orphan_cleanup()
        await self.stop_login_watch()
        for engine in self.engines.values():
            try:
                await engine.stop()
            except Exception:
                pass
        # Actually terminate the shared browser now that all engines detached.
        from core.zendriver_llm_base import shutdown_shared_driver
        await shutdown_shared_driver()
