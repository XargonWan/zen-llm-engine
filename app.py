import asyncio
import base64
import json
import logging
import mimetypes
import os
import signal
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Set

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)



from db.db import (
    clear_prompt_logs,
    clear_stats,
    get_logged_engines,
    get_media_sent_today,
    get_prompt_logs,
    get_response_time_stats,
    get_stats,
    init_database,
    log_prompt,
    inc_errors,
    inc_media_sent,
    inc_requests,
    inc_responses,
)
from core.agent_protocol import (
    build_agent_system_prompt,
    build_agent_turn_reminder,
    build_reformulation_prompt,
    detect_agent_context,
    needs_reformulation,
    parse_agent_response,
    to_openai_tool_calls,
)
from core.engine_manager import EngineManager
from core.errors import EngineUnavailableError
from core import debug_mode, proc_utils
from core.zendriver_llm_base import get_browser_stats
from core.models import (
    ChatCompletion,
    LegacyModelList,
    LegacyModelEntry,
    ModelEntry,
    ModelList,
    PingResponse,
)

# ---------------------------------------------------------------------------
# In-memory application log buffer (survives for the process lifetime)
# ---------------------------------------------------------------------------

_LOG_BUFFER: deque[Dict[str, Any]] = deque(maxlen=500)
_LOG_SEQ = 0
_LOG_BUFFER_LOCK = threading.Lock()


class _BufferHandler(logging.Handler):
    """Appends log records to the in-memory ring buffer."""

    def emit(self, record: logging.LogRecord) -> None:
        global _LOG_SEQ
        try:
            with _LOG_BUFFER_LOCK:
                _LOG_SEQ += 1
                _LOG_BUFFER.append(
                    {
                        "seq": _LOG_SEQ,
                        "time": time.strftime(
                            "%H:%M:%S", time.localtime(record.created)
                        ),
                        "level": record.levelname,
                        "name": record.name,
                        "msg": self.format(record),
                    }
                )
        except Exception:
            self.handleError(record)


@dataclass
class MediaItem:
    media_type: str
    data: bytes
    mime_type: str
    filename: str


def _parse_data_uri(data_uri: str) -> tuple[bytes, str]:
    if not isinstance(data_uri, str) or not data_uri.startswith("data:"):
        raise ValueError("Invalid data URI")
    header, _, payload = data_uri.partition(",")
    if not payload:
        raise ValueError("Malformed data URI")
    if ";base64" not in header:
        raise ValueError("Only base64-encoded data URIs are supported")
    mime_type = header[5:].split(";", 1)[0] or "application/octet-stream"
    return base64.b64decode(payload), mime_type


def _guess_filename(media_type: str, mime_type: str, index: int) -> str:
    extension = mimetypes.guess_extension(mime_type) or ""
    if not extension:
        extension = ".bin"
    return f"{media_type}_{index}{extension}"


def _map_media_capabilities_to_model_caps(media_capabilities: list[str]) -> dict[str, bool]:
    capabilities: dict[str, bool] = {}
    if "image" in media_capabilities:
        capabilities["vision"] = True
    if "audio" in media_capabilities:
        capabilities["audio"] = True
    return capabilities


def _parse_media_part(part: dict, index: int) -> MediaItem:
    raw_media_type = str(part.get("type", "")).strip()
    if not raw_media_type:
        mime_type_hint = str(part.get("mime_type") or part.get("content_type") or "").strip().lower()
        if mime_type_hint.startswith("image/"):
            raw_media_type = "image"
        elif mime_type_hint.startswith("audio/"):
            raw_media_type = "audio"
        elif mime_type_hint:
            raw_media_type = "input_file"

    if raw_media_type in ("image_url", "image"):
        media_type = "image"
    elif raw_media_type in ("input_audio", "audio"):
        media_type = "audio"
    elif raw_media_type in ("input_file", "file", "document", "input_document"):
        media_type = "document"
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported media type: {raw_media_type}")

    if media_type == "image":
        # Support both flat {"url": "..."} and OpenAI vision {"image_url": {"url": "..."}}
        image_url_nested = part.get("image_url")
        source = (
            part.get("url")
            or part.get("data")
            or (image_url_nested.get("url") if isinstance(image_url_nested, dict) else None)
        )
        if not source:
            raise HTTPException(status_code=400, detail="Missing image URL or data")
    else:
        source = part.get("data")
        if not source:
            raise HTTPException(status_code=400, detail="Missing file data")

    if isinstance(source, str) and source.startswith("data:"):
        try:
            data, mime_type = _parse_data_uri(source)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    elif isinstance(source, str):
        try:
            data = base64.b64decode(source)
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid base64 media data") from exc
        if media_type == "audio":
            mime_type = part.get("mime_type") or "audio/mpeg"
        else:
            mime_type = part.get("mime_type") or "application/octet-stream"
    else:
        raise HTTPException(status_code=400, detail="Unsupported media payload format")

    filename = str(part.get("filename") or _guess_filename(media_type, mime_type, index))
    return MediaItem(media_type=media_type, data=data, mime_type=mime_type, filename=filename)


def _try_parse_json_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    trimmed = value.strip()
    if not trimmed or trimmed[0] not in ("{", "["):
        return value
    try:
        parsed = json.loads(trimmed)
        logger.debug("[selenium] Parsed JSON string content into structured payload")
        return parsed
    except Exception:
        return value


def _is_media_part(part: Any) -> bool:
    if not isinstance(part, dict):
        return False
    raw_media_type = str(part.get("type", "")).strip().lower()
    if raw_media_type in (
        "image_url",
        "image",
        "input_audio",
        "audio",
        "input_file",
        "file",
        "document",
        "input_document",
    ):
        return True
    mime_type = str(part.get("mime_type") or part.get("content_type") or "").strip().lower()
    return bool(mime_type and (mime_type.startswith("image/") or mime_type.startswith("audio/") or mime_type.startswith("video/") or mime_type.startswith("application/")))


def _extract_text_and_media(content: Any, media_items: list[MediaItem]) -> str:
    content = _try_parse_json_string(content)
    if isinstance(content, dict):
        if _is_media_part(content):
            try:
                media_items.append(_parse_media_part(content, len(media_items)))
                logger.debug("[selenium] Extracted media part from structured payload")
            except HTTPException:
                pass
            return ""

        text_parts: list[str] = []
        attachments = content.get("attachments")
        if isinstance(attachments, list):
            for attachment in attachments:
                if _is_media_part(attachment):
                    try:
                        media_items.append(_parse_media_part(attachment, len(media_items)))
                        logger.debug("[selenium] Extracted media attachment from attachments array")
                    except HTTPException:
                        pass
                else:
                    text_parts.append(_extract_text_and_media(attachment, media_items))

        if "content" in content:
            text_parts.append(_extract_text_and_media(content["content"], media_items))
        if "text" in content and content["text"] is not content.get("content"):
            text_parts.append(_extract_text_and_media(content["text"], media_items))

        # Only recurse into 'parts' or 'messages' if they exist, but avoid
        # generic iteration over all keys to prevent metadata/internal field leakage.
        for key in ("parts", "messages"):
            if key in content and isinstance(content[key], list):
                text_parts.append(_extract_text_and_media(content[key], media_items))

        return "".join(text_parts)

    if isinstance(content, list):
        return "".join(_extract_text_and_media(item, media_items) for item in content)

    return str(content)


def _normalize_prompt_payload(payload: Any) -> tuple[str, list[MediaItem]]:
    prompt_text = ""
    media_items: list[MediaItem] = []

    if isinstance(payload, dict) and "role" in payload and "content" in payload:
        payload = [payload]

    if isinstance(payload, list):
        system_parts: list[str] = []
        user_parts: list[str] = []
        for message in payload:
            role = "user"
            content = message
            if isinstance(message, dict):
                role = str(message.get("role", "user"))
                content = message.get("content", "")

            message_text = _extract_text_and_media(content, media_items)

            if role == "system":
                if message_text:
                    system_parts.append(message_text)
            else:
                if message_text:
                    user_parts.append(message_text)

        parts: list[str] = []
        if system_parts:
            parts.append("[INSTRUCTIONS]:\n" + "\n\n".join(system_parts))
        if user_parts:
            parts.extend(user_parts)
        prompt_text = "\n\n".join(parts)
    else:
        prompt_text = _extract_text_and_media(payload, media_items)

    if not prompt_text and media_items:
        prompt_text = "[Media attachments included]"
    return prompt_text, media_items


# A single log record can be enormous: zendriver's CDP listener once logged a
# ``StopIteration`` carrying an entire DOM tree (1.6 MB in ONE line, four times in
# three days -- a quarter of the container's log). Bound every rendered line so
# neither the log files, docker's json log nor the in-memory ring buffer can be
# flooded by a single record. 0 disables the limit.
_LOG_MAX_LINE = int(os.getenv("ZEN_LOG_MAX_LINE", "20000") or 0)


class _BoundedFormatter(logging.Formatter):
    """Formatter that truncates the rendered record (message + traceback)."""

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        if _LOG_MAX_LINE > 0 and len(text) > _LOG_MAX_LINE:
            dropped = len(text) - _LOG_MAX_LINE
            text = f"{text[:_LOG_MAX_LINE]}... [truncated {dropped} chars]"
        return text


_buf_handler = _BufferHandler()
_buf_handler.setLevel(logging.DEBUG)
_buf_handler.setFormatter(_BoundedFormatter())

logging.basicConfig(level=logging.INFO)
for _root_handler in logging.getLogger().handlers:
    # Same layout as basicConfig's default, just bounded.
    _root_handler.setFormatter(_BoundedFormatter("%(levelname)s:%(name)s:%(message)s"))
# Attach after basicConfig so the root logger already exists
logging.getLogger().addHandler(_buf_handler)

# Also persist to disk. Until now the container's stdout was the only copy, so
# every diagnosis needed `docker logs` and nothing survived a recreate -- the
# per-step [timing] lines that explain a stuck prompt were simply unavailable
# to anyone reading a mounted log directory.
_LOG_DIR = Path(os.getenv("ZEN_LOG_DIR", "./logs"))
try:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    _file_handler = RotatingFileHandler(
        _LOG_DIR / "zen-llm-engine.log",
        maxBytes=20 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    _file_handler.setLevel(logging.INFO)
    _file_handler.setFormatter(
        _BoundedFormatter("%(asctime)s [%(levelname)s] [%(name)s] %(message)s")
    )
    logging.getLogger().addHandler(_file_handler)
except OSError as exc:  # pragma: no cover - read-only or absent mount
    logging.getLogger().warning(
        "[logging] Cannot write log file in %s: %s", _LOG_DIR, exc
    )

logger = logging.getLogger("zen-llm-api")


# Signal handling for graceful shutdown
_shutdown_event = asyncio.Event()


def _signal_handler(sig: int) -> None:
    logger.info(f"[signal] Received signal {sig}, initiating graceful shutdown...")
    _shutdown_event.set()


@asynccontextmanager
async def _lifespan(app: FastAPI):  # type: ignore[type-arg]
    # ---- startup ----
    init_database()
    manager = EngineManager.get()           # initialize manager
    manager.start_orphan_cleanup()          # start periodic orphan cleanup
    manager.start_login_watch()             # passive re-login detection (breaker recovery)
    _register_engine_routes(app)            # dynamic per-engine /name/prompt routes

    # Register signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _signal_handler, sig)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass

    yield

    # ---- shutdown ----
    # Gracefully quit all Chrome instances so they can flush cookies/profile to
    # disk before the container is killed.  This keeps login sessions alive
    # across docker stop / docker restart.
    try:
        await asyncio.wait_for(manager.stop_all(), timeout=15)
    except Exception as exc:
        logger.warning(f"[shutdown] stop_all error: {exc}")


app = FastAPI(title="Zen LLM Engine", version="0.1", lifespan=_lifespan)
_STARTED_AT = time.time()


@app.exception_handler(EngineUnavailableError)
async def _engine_unavailable_handler(
    request: Request, exc: EngineUnavailableError
) -> JSONResponse:
    """Map an open circuit breaker to a clear 503 that clients can back off on."""
    message = str(exc)
    return JSONResponse(
        status_code=503,
        headers={"Retry-After": str(exc.retry_after)},
        content={
            "detail": message,
            "error": {
                "type": "engine_unavailable",
                "engine": exc.engine,
                "reason": exc.reason,
                "retry_after": exc.retry_after,
                "message": message,
            },
        },
    )

# Rate limiting (per ip, sliding window)
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX = 20
rate_limit_store: Dict[str, list[float]] = defaultdict(list)


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _rate_limit_exceeded(request: Request) -> bool:
    key = _client_ip(request)
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW
    entries = rate_limit_store[key]
    rate_limit_store[key] = [t for t in entries if t >= window_start]
    if len(rate_limit_store[key]) >= RATE_LIMIT_MAX:
        return True
    rate_limit_store[key].append(now)
    return False


# Reset/state coordination helpers
RESET_IN_PROGRESS = False
IN_FLIGHT_TASKS: Set[asyncio.Task] = set()


def _register_task(task: asyncio.Task) -> None:
    IN_FLIGHT_TASKS.add(task)


def _unregister_task(task: asyncio.Task) -> None:
    IN_FLIGHT_TASKS.discard(task)


async def _cancel_inflight_tasks() -> None:
    tasks = list(IN_FLIGHT_TASKS)
    if not tasks:
        return
    for t in tasks:
        t.cancel()
    try:
        # Wait for cancel to propagate (including exceptions raised on cancellation)
        await asyncio.gather(*tasks, return_exceptions=True)
    except Exception:
        pass
    IN_FLIGHT_TASKS.clear()


async def _safe_parse_json(request: Request) -> Dict[str, Any]:
    try:
        return await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {e}")


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


def _openai_response(
    engine_name: str,
    model_name: str,
    prompt: str,
    response_text: str,
    elapsed_ms: int,
    agent_parsed: Any = None,
) -> Dict[str, Any]:
    prompt_tokens = _estimate_tokens(prompt)
    completion_tokens = _estimate_tokens(response_text)

    # Agent mode: emit OpenAI tool_calls when the model requested tool use.
    if agent_parsed is not None and getattr(agent_parsed, "tool_calls", None):
        message: Dict[str, Any] = {
            "role": "assistant",
            "content": None,
            "tool_calls": to_openai_tool_calls(agent_parsed.tool_calls),
        }
        finish_reason = "tool_calls"
    else:
        # Prefer parsed final content when available, else the raw scraped text.
        content = response_text
        if agent_parsed is not None and getattr(agent_parsed, "content", None):
            content = agent_parsed.content
        message = {"role": "assistant", "content": content}
        finish_reason = "stop"

    return {
        "id": f"llm_{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "engine": engine_name,
        "prompt": prompt,
        "elapsed_ms": elapsed_ms,
    }


def _openai_chunk(
    chunk_id: str,
    model_name: str,
    content: str,
    finish_reason: Any,
    tool_calls: Any = None,
) -> str:
    """Format a single SSE chunk in OpenAI chat.completion.chunk format.

    When *tool_calls* is provided (agent mode), it is emitted in the delta
    instead of textual content so streaming clients receive structured tool
    calls exactly like the non-streaming path.
    """
    if tool_calls:
        delta: Dict[str, Any] = {"role": "assistant", "tool_calls": tool_calls}
    elif content:
        delta = {"role": "assistant", "content": content}
    else:
        delta = {}
    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(payload)}\n\n"





@app.get("/")
async def root() -> RedirectResponse:
    # Redirect to the web UI for convenience
    return RedirectResponse(url="/ui")


@app.get("/api/ping", response_model=PingResponse)
async def ping() -> PingResponse:
    return PingResponse(status="ok", service="zen-llm-engine")


@app.get("/api/engines")
async def api_engines() -> Dict[str, Any]:
    """List all discovered engines with metadata (no browser started)."""
    mgr = EngineManager.get()
    return {"data": mgr.list_engines()}


@app.get("/api/status")
async def api_status() -> JSONResponse:
    """Real health: per-engine breaker/queue/login state plus process vitals.

    Unlike ``/api/ping`` (which only proves the HTTP server is up) this reports
    whether engines can actually answer. It never takes the browser lock, never
    touches the browser and never queries the database, so it stays responsive
    when a request is wedged. Returns 200 always; ``status`` is ``ok``,
    ``degraded`` (recent failures) or ``down`` (some engine's breaker is open).
    """
    engines = EngineManager.get().health_snapshot()
    states = {e["state"] for e in engines}
    overall = "down" if "down" in states else ("degraded" if "degraded" in states else "ok")
    rss = proc_utils.rss_mb()
    warn_mb = int(os.getenv("ZEN_RSS_WARN_MB", "2048") or 0)
    body = {
        "status": overall,
        "uptime_seconds": round(time.time() - _STARTED_AT),
        "process": {
            "rss_mb": rss,
            "rss_warning": bool(rss is not None and warn_mb and rss > warn_mb),
            "zombie_processes": await asyncio.to_thread(proc_utils.count_zombies),
        },
        "browser": get_browser_stats(),
        "engines": engines,
    }
    return JSONResponse(content=body)


@app.get("/api/health")
async def api_health() -> JSONResponse:
    """Same body as ``/api/status`` but 503 when any engine is ``down``."""
    response = await api_status()
    payload = json.loads(response.body)
    return JSONResponse(status_code=503 if payload["status"] == "down" else 200, content=payload)


@app.post("/api/engines/{engine_name}/pause")
async def api_engine_pause(engine_name: str) -> Dict[str, Any]:
    """Pause an engine so you can sign in by hand without requests disturbing the tab.

    Requests get a 503 while paused. It resumes by itself once the page looks signed
    in (read-only watcher), or immediately with ``POST /api/engines/{engine}/recover``.
    """
    mgr = EngineManager.get()
    try:
        canonical = mgr._resolve(engine_name)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Engine not found: {engine_name}")
    mgr.pause_for_login(canonical)
    return {
        "status": "ok",
        "engine": canonical,
        "paused": True,
        "resume": f"sign in, or POST /api/engines/{canonical}/recover",
    }


@app.post("/api/engines/{engine_name}/recover")
async def api_engine_recover(engine_name: str) -> Dict[str, Any]:
    """Close an engine's circuit breaker (e.g. right after signing in by hand)."""
    mgr = EngineManager.get()
    try:
        canonical = mgr._resolve(engine_name)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Engine not found: {engine_name}")
    return {"status": "ok", "engine": canonical, "was_open": mgr.reset_breaker(canonical)}


@app.get("/api/debug/page-html", response_class=HTMLResponse)
async def api_debug_page_html(engine_name: str | None = None) -> HTMLResponse:
    """Return the current page HTML from the requested engine's browser session."""
    mgr = EngineManager.get()
    try:
        if engine_name:
            engine = mgr.get_engine(engine_name)
        else:
            engine = mgr.get_active_engine()
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    # `engine.driver` holds the engine's current zendriver Tab (the page-level
    # handle equivalent of the old Selenium WebDriver) — see
    # core/zendriver_llm_base.py for why the attribute name was kept.
    driver = getattr(engine, "driver", None)
    if not driver:
        raise HTTPException(
            status_code=404,
            detail="No active browser session for the selected engine",
        )

    try:
        html = await driver.get_content()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not retrieve page source: {exc}",
        )

    return HTMLResponse(content=html)


@app.get("/api/engines/default")
async def api_engines_default() -> Dict[str, Any]:
    """Get the currently configured default engine."""
    mgr = EngineManager.get()
    try:
        default_engine = mgr.get_default_engine()
    except ValueError:
        raise HTTPException(status_code=404, detail="No engines available")
    return {"default_engine": default_engine}


@app.post("/api/engines/default")
async def api_set_default_engine(body: Dict[str, Any]) -> Dict[str, Any]:
    """Set the default engine by name or alias."""
    engine_name = body.get("engine")
    if not engine_name:
        raise HTTPException(status_code=400, detail="Missing engine")

    mgr = EngineManager.get()
    try:
        canonical = mgr.set_default_engine(engine_name)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Engine not found: {engine_name}")

    return {"status": "ok", "default_engine": canonical}


@app.post("/api/engines/reload")
async def api_engines_reload() -> Dict[str, Any]:
    """Re-scan the engines/ directory and refresh the registry."""
    mgr = EngineManager.get()
    updated = mgr.reload_engines()
    return {"status": "ok", "data": updated}


@app.get("/models", response_model=LegacyModelList)
async def models() -> LegacyModelList:
    """Legacy endpoint — returns OpenAI-compatible model list (same as /v1/models).
    Also includes legacy 'name'/'limits'/'supported_models' fields for backward
    compatibility with older clients."""
    mgr = EngineManager.get()
    created = int(time.time())
    data = []
    for desc in mgr.list_engines():
        engine_name = desc["name"]
        entry: Dict[str, Any] = {
            # OpenAI-compatible fields (required by clients like Alpaca)
            "id": engine_name,
            "object": "model",
            "created": created,
"owned_by": "zen-llm-engine",
            # Legacy extra fields (kept for backward compat)
            "name": engine_name,
            "capabilities": _map_media_capabilities_to_model_caps(
                list(desc.get("media_capabilities", []))
            ),
        }
        # Use live engine data if the browser is already running, otherwise
        # fall back to the descriptor metadata (avoids opening browsers on probe).
        if engine_name in mgr.engines:
            eng = mgr.engines[engine_name]
            entry["limits"] = eng.get_interface_limits()
            entry["supported_models"] = eng.get_supported_models()
        else:
            desc_obj = mgr.get_descriptor(engine_name)
            if desc_obj:
                entry["limits"] = desc_obj.limits_dict()
                entry["supported_models"] = desc_obj.supported_models_list()
        data.append(LegacyModelEntry(**entry))
    return LegacyModelList(object="list", data=data)


@app.get("/models/{engine_name}")
async def model_info(engine_name: str) -> Dict[str, Any]:
    try:
        engine = EngineManager.get().get_engine(engine_name)
    except ValueError:
        raise HTTPException(status_code=404, detail="Engine not found")
    return {
        "engine": engine_name,
        "limits": engine.get_interface_limits(),
        "models": engine.get_supported_models(),
    }


@app.post("/login/{engine_name}")
async def login_engine(engine_name: str) -> Dict[str, Any]:
    try:
        engine = EngineManager.get().set_active_engine(engine_name)
    except ValueError:
        raise HTTPException(status_code=404, detail="Engine not found")

    # Someone is about to sign in by hand in this browser: stop requests from
    # navigating/resetting the tab meanwhile (resumes once the page looks signed in).
    mgr = EngineManager.get()
    mgr.pause_for_login(engine_name)
    result = await engine.start_login_flow()
    if result.get("login_state") == "logged":
        mgr.note_login_state(engine_name, True)
    return result


@app.get("/login/{engine_name}/state")
async def login_state(engine_name: str) -> Dict[str, Any]:
    try:
        engine = EngineManager.get().get_engine(engine_name)
    except ValueError:
        raise HTTPException(status_code=404, detail="Engine not found")

    mgr = EngineManager.get()
    if mgr.login_blocked(engine_name):
        # Breaker open for a missing login: the operator is probably signing in
        # in this very browser. check_login_state() may navigate away from the
        # login page, so only peek (read-only) here.
        peeked = await engine.peek_login_state()
        state = {
            "logged_in": peeked is True,
            "login_state": "logged" if peeked is True else ("unlogged" if peeked is False else "unknown"),
            "breaker": "open",
        }
    else:
        state = await engine.check_login_state()
    if state.get("login_state") == "logged":
        mgr.note_login_state(engine_name, True)
    return state


@app.post("/engine/{engine_name}/prompt", response_model=ChatCompletion)
async def engine_prompt(engine_name: str, req: Request) -> Any:
    """Dynamic prompt endpoint — works for any discovered engine."""
    if _rate_limit_exceeded(req):
        raise HTTPException(status_code=429, detail="Too many requests")
    mgr = EngineManager.get()
    try:
        canonical = mgr._resolve(engine_name)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Engine not found: {engine_name}")
    data = await _safe_parse_json(req)
    return await _prompt(
        canonical,
        req,
        explicit_prompt=data.get("prompt") or data.get("messages"),
        model_name=data.get("model", canonical),
        stream=bool(data.get("stream", False)),
        timeout=data.get("timeout"),
    )


# ---------------------------------------------------------------------------
# Legacy per-engine prompt endpoints — generated dynamically at startup
# ---------------------------------------------------------------------------


def _register_engine_routes(application: FastAPI) -> None:
    """Create /{engine_name}/prompt routes for every discovered engine."""
    mgr = EngineManager.get()
    for desc in mgr.list_engines():
        engine_name = desc["name"]

        # Build a closure that captures the canonical engine name
        def _make_handler(name: str):
            async def handler(req: Request) -> Any:
                if _rate_limit_exceeded(req):
                    raise HTTPException(status_code=429, detail="Too many requests")
                data = await _safe_parse_json(req)
                return await _prompt(
                    name,
                    req,
                    explicit_prompt=data.get("prompt") or data.get("messages"),
                    model_name=name,
                    stream=bool(data.get("stream", False)),
                    timeout=data.get("timeout"),
                )

            handler.__name__ = f"{name}_prompt"
            return handler

        application.add_api_route(
            f"/{engine_name}/prompt",
            _make_handler(engine_name),
            methods=["POST"],
            response_model=ChatCompletion,
        )
        logger.info(f"[app] Registered route POST /{engine_name}/prompt")


@app.get("/v1/models", response_model=ModelList)
async def v1_models() -> ModelList:
    """OpenAI-compatible model list. Returns one entry per provider (canonical name).
    Clients that send model='chatgpt' or model='gemini' will be routed correctly.
    Aliases and per-variant ids are intentionally excluded to maximise client compatibility."""
    mgr = EngineManager.get()
    created = int(time.time())
    entries: list[ModelEntry] = []
    for desc in mgr.list_engines():
        entries.append(
            ModelEntry(
                id=desc["name"],
                object="model",
                created=created,
                owned_by="selenium-llm-engine",
                capabilities=_map_media_capabilities_to_model_caps(
                    list(desc.get("media_capabilities", []))
                ),
            )
        )
    return ModelList(object="list", data=entries)


@app.get("/v1/models/{model_id:path}")
async def v1_model_detail(model_id: str) -> Dict[str, Any]:
    """OpenAI-compatible single model lookup. Supports 'provider' or 'provider:variant'."""
    mgr = EngineManager.get()
    provider = model_id.split(":")[0]
    try:
        mgr._resolve(provider)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    return {
        "id": model_id,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "zen-llm-engine",
    }


@app.post("/v1/chat/completions", response_model=ChatCompletion)
async def openai_chat(req: Request) -> Any:
    if _rate_limit_exceeded(req):
        raise HTTPException(status_code=429, detail="Too many requests")

    data = await _safe_parse_json(req)
    model = data.get("model")
    mgr = EngineManager.get()

    if not model:
        engine = mgr.get_default_engine()
        model = engine
    else:
        # Support provider:variant notation (e.g. "chatgpt:gpt-4o" -> engine="chatgpt")
        engine_hint = str(model).split(":")[0]
        try:
            engine = mgr._resolve(engine_hint)
        except ValueError:
            # Fall back to configured default engine for unrecognised names.
            # Announce it: routing a request to a different provider than the
            # caller asked for is otherwise indistinguishable from success.
            engine = mgr.get_default_engine()
            logger.warning(
                "[openai_compat] Unknown engine '%s' (from model '%s'); "
                "routing to default engine '%s' instead",
                engine_hint,
                model,
                engine,
            )
            model = engine

    if "prompt" in data:
        prompt_payload = data.get("prompt")
    elif "messages" in data:
        prompt_payload = data.get("messages")
    else:
        raise HTTPException(status_code=400, detail="Missing prompt/messages")

    unsupported_openai_params = ["logprobs", "top_logprobs", "logit_bias", "n", "presence_penalty", "frequency_penalty"]
    unsupported_present = [key for key in unsupported_openai_params if key in data]
    if unsupported_present:
        logger.warning(
            "[openai_compat] Ignoring unsupported OpenAI parameters for browser engines: %s",
            unsupported_present,
        )

    stream = bool(data.get("stream", False))

    # Detect the agentic (tool-calling) lane from OpenAI agentic fields.
    agent_ctx = detect_agent_context(data)
    if agent_ctx is not None:
        logger.info(
            "[agent] Agentic request detected (tools=%d, response_format=%s)",
            len(agent_ctx.get("tools") or []),
            bool(agent_ctx.get("response_format")),
        )

    return await _prompt(
        engine, req, explicit_prompt=prompt_payload, model_name=model, stream=stream,
        timeout=data.get("timeout"), agent_ctx=agent_ctx,
    )


# Legacy alias — some clients call /chat/completions without the /v1 prefix
@app.post("/chat/completions", response_model=ChatCompletion)
async def openai_chat_legacy(req: Request) -> Any:
    return await openai_chat(req)


async def _prompt(
    engine_name: str,
    req: Request,
    explicit_prompt: Any = None,
    model_name: str = "default",
    stream: bool = False,
    timeout: int | None = None,
    agent_ctx: Dict[str, Any] | None = None,
) -> Any:
    if RESET_IN_PROGRESS:
        raise HTTPException(
            status_code=503,
            detail="Service is resetting; please retry after a moment",
        )

    if _rate_limit_exceeded(req):
        raise HTTPException(status_code=429, detail="Too many requests")

    # Reject up front (clean 503 + Retry-After) while the engine's circuit
    # breaker is open, before any body parsing or a streaming response starts.
    EngineManager.get().assert_available(engine_name)

    if explicit_prompt is None:
        payload = await _safe_parse_json(req)
        prompt_payload = payload.get("prompt") or payload.get("messages")
    else:
        prompt_payload = explicit_prompt

    if prompt_payload is None:
        raise HTTPException(status_code=400, detail="Missing prompt/messages")

    prompt_text, media_items = _normalize_prompt_payload(prompt_payload)
    if not prompt_text and not media_items:
        raise HTTPException(status_code=400, detail="Missing prompt/messages")

    if not isinstance(prompt_text, str):
        prompt_text = str(prompt_text)

    # Agent mode: prepend the tool-calling harness so the browser-driven model
    # replies with parseable JSON. The scraped reply is parsed below with a
    # bounded reformulation retry.
    agent_mode = agent_ctx is not None
    has_tools = False
    if agent_mode:
        # Prepend the full harness AND append a short in-character reminder so
        # the roleplay contract is both the first and the LAST thing the model
        # reads — on long multi-turn histories the leading harness alone drifts
        # out of context and the model reverts to a plain "safe assistant".
        has_tools = bool(agent_ctx.get("tools"))
        prompt_text = (
            build_agent_system_prompt(agent_ctx)
            + prompt_text
            + build_agent_turn_reminder(has_tools, agent_ctx.get("tools"))
        )

    current_task = asyncio.current_task()
    if current_task is not None:
        _register_task(current_task)

    inc_requests()
    start = time.time()
    debug_mode.record_event(
        "input",
        engine_name,
        model=model_name,
        stream=stream,
        agent_mode=agent_mode,
        media_count=len(media_items),
        prompt=prompt_text,
    )
    try:
        mgr = EngineManager.get()

        if stream:

            async def generate_stream():
                try:
                    # Start engine work as a concurrent task so we can yield
                    # SSE heartbeats while waiting, keeping the connection alive.
                    result_future = asyncio.ensure_future(
                        mgr.enqueue(engine_name, prompt_text, media_items,
                                    timeout=timeout, agent_mode=agent_mode)
                    )

                    heartbeat_interval = 5.0  # seconds
                    while not result_future.done():
                        done, _ = await asyncio.wait(
                            {result_future},
                            timeout=heartbeat_interval,
                        )
                        if not done:
                            # SSE comment — keeps HTTP connection alive without
                            # injecting data into the content stream.
                            yield ": heartbeat\n\n"

                    result_obj = result_future.result()

                    # Agent mode over SSE: apply the SAME parse + bounded
                    # reformulation as the non-streaming path, otherwise the
                    # scraped JSON (e.g. a bare {"content": "..."} promise or a
                    # tool call) would be streamed back as raw text and the
                    # client would never receive structured tool_calls.
                    agent_parsed = None
                    if agent_mode:
                        agent_parsed = parse_agent_response(result_obj.text)
                        max_reformulations = 2
                        attempts = 0
                        while (
                            needs_reformulation(agent_parsed, has_tools=has_tools)
                            and attempts < max_reformulations
                        ):
                            attempts += 1
                            logger.warning(
                                "[agent] Stream reply needs reformulation, "
                                "attempt %d/%d",
                                attempts,
                                max_reformulations,
                            )
                            retry_obj = await mgr.enqueue(
                                engine_name,
                                build_reformulation_prompt(prompt_text),
                                media_items,
                                timeout=timeout,
                                agent_mode=True,
                            )
                            result_obj = retry_obj
                            agent_parsed = parse_agent_response(result_obj.text)

                    elapsed_ms = int((time.time() - start) * 1000)
                    if media_items:
                        inc_media_sent(len(media_items))
                    log_prompt(
                        engine_name,
                        result_obj.model_name,
                        prompt_text,
                        result_obj.text,
                        "ok",
                        elapsed_ms,
                    )
                    debug_mode.record_event(
                        "output",
                        engine_name,
                        model=result_obj.model_name,
                        stream=True,
                        elapsed_ms=elapsed_ms,
                        reply=result_obj.text,
                    )
                    inc_responses()
                    chunk_id = f"llm_{int(time.time())}"
                    if agent_parsed is not None and agent_parsed.tool_calls:
                        # Emit structured tool calls, then close with the
                        # OpenAI-mandated "tool_calls" finish reason.
                        yield _openai_chunk(
                            chunk_id,
                            result_obj.model_name,
                            "",
                            None,
                            tool_calls=to_openai_tool_calls(agent_parsed.tool_calls),
                        )
                        yield _openai_chunk(
                            chunk_id, result_obj.model_name, "", "tool_calls"
                        )
                    else:
                        # Final textual answer: prefer the parsed content over
                        # the raw scraped text so the client never sees the
                        # JSON wrapper.
                        out_text = result_obj.text
                        if agent_parsed is not None and agent_parsed.content:
                            out_text = agent_parsed.content
                        yield _openai_chunk(
                            chunk_id, result_obj.model_name, out_text, None
                        )
                        yield _openai_chunk(
                            chunk_id, result_obj.model_name, "", "stop"
                        )
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    if isinstance(e, EngineUnavailableError):
                        inc_errors()
                        raise
                    elapsed_ms = int((time.time() - start) * 1000)
                    log_prompt(
                        engine_name, "unknown", prompt_text, str(e), "error", elapsed_ms
                    )
                    debug_mode.record_event(
                        "error",
                        engine_name,
                        stream=True,
                        elapsed_ms=elapsed_ms,
                        error=str(e),
                    )
                    inc_errors()
                    raise HTTPException(status_code=500, detail=str(e))

            return StreamingResponse(generate_stream(), media_type="text/event-stream")

        result_obj = await mgr.enqueue(engine_name, prompt_text, media_items,
                                        timeout=timeout, agent_mode=agent_mode)

        agent_parsed = None
        if agent_mode:
            agent_parsed = parse_agent_response(result_obj.text)
            # Bounded reformulation retries when the reply is not parseable —
            # or when the model returned a valid {"content": ...} that is really
            # an out-of-character refusal to use the available tools.
            max_reformulations = 2
            attempts = 0
            while (
                needs_reformulation(agent_parsed, has_tools=has_tools)
                and attempts < max_reformulations
            ):
                attempts += 1
                logger.warning(
                    "[agent] Unparseable reply, reformulation attempt %d/%d",
                    attempts,
                    max_reformulations,
                )
                # Stateless engines refresh the page between turns, so re-send
                # the full harness + original request (not a bare nudge) to keep
                # the tool list and task in context on every retry.
                retry_obj = await mgr.enqueue(
                    engine_name, build_reformulation_prompt(prompt_text), media_items,
                    timeout=timeout, agent_mode=True,
                )
                result_obj = retry_obj
                agent_parsed = parse_agent_response(result_obj.text)

        duration_ms = int((time.time() - start) * 1000)
        if media_items:
            inc_media_sent(len(media_items))
        log_prompt(
            engine_name,
            result_obj.model_name,
            prompt_text,
            result_obj.text,
            "ok",
            duration_ms,
        )
        debug_mode.record_event(
            "output",
            engine_name,
            model=result_obj.model_name,
            stream=False,
            elapsed_ms=duration_ms,
            reply=result_obj.text,
        )
        inc_responses()

        return _openai_response(
            engine_name,
            result_obj.model_name,
            prompt_text,
            result_obj.text,
            duration_ms,
            agent_parsed=agent_parsed,
        )

    except asyncio.CancelledError:
        duration_ms = int((time.time() - start) * 1000)
        log_prompt(
            engine_name,
            "unknown",
            prompt_text,
            "cancelled due to reset",
            "error",
            duration_ms,
        )
        debug_mode.record_event(
            "error",
            engine_name,
            stream=stream,
            elapsed_ms=duration_ms,
            error="cancelled due to reset",
        )
        inc_errors()
        raise HTTPException(status_code=503, detail="Request cancelled due to reset")
    except HTTPException:
        raise
    except EngineUnavailableError:
        # Breaker opened while this request was queued: surface the 503 as-is,
        # without writing a (potentially huge) prompt row for every rejection.
        inc_errors()
        raise
    except Exception as e:
        duration_ms = int((time.time() - start) * 1000)
        log_prompt(engine_name, "unknown", prompt_text, str(e), "error", duration_ms)
        debug_mode.record_event(
            "error",
            engine_name,
            stream=stream,
            elapsed_ms=duration_ms,
            error=str(e),
        )
        inc_errors()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if current_task is not None:
            _unregister_task(current_task)


def _get_media_availability() -> Dict[str, int]:
    mgr = EngineManager.get()
    totals = {"unlogged": 0, "base": 0, "paid": 0}
    unlimited = {"unlogged": False, "paid": False}
    for desc in mgr._descriptors.values():
        media_support = getattr(desc, "media_support", {}) or {}
        if not isinstance(media_support, dict):
            continue
        engine_totals = {"unlogged": 0, "base": 0, "paid": 0}
        engine_unknown_base = False
        for cfg in media_support.values():
            if not isinstance(cfg, dict):
                continue
            limits = cfg.get("limits", {})
            if not isinstance(limits, dict):
                continue
            for tier in engine_totals:
                value = limits.get(tier)
                if value == -1:
                    if tier == "base":
                        engine_unknown_base = True
                    else:
                        unlimited[tier] = True
                elif isinstance(value, int) and value > 0:
                    engine_totals[tier] += value
        totals["base"] += engine_totals["base"]
        if engine_unknown_base:
            totals["base"] += 2
        totals["unlogged"] += engine_totals["unlogged"]
        totals["paid"] += engine_totals["paid"]
    result: Dict[str, int] = {}
    for tier, amount in totals.items():
        if tier in unlimited and unlimited[tier]:
            result[tier] = -1
        else:
            result[tier] = amount
    return result


@app.get("/stats")
async def stats() -> Dict[str, Any]:
    return {
        "stats": get_stats(),
        "media_sent_today": get_media_sent_today(),
        "media_availability": _get_media_availability(),
        "logged_engines": get_logged_engines(),
        "response_time": get_response_time_stats(),
    }


@app.get("/api/logs/app")
async def app_logs(since: int = 0) -> Dict[str, Any]:
    """Return application log entries with seq > since (incremental polling)."""
    with _LOG_BUFFER_LOCK:
        entries = [e for e in _LOG_BUFFER if e["seq"] > since]
    return {"entries": entries}


@app.get("/api/debug")
async def get_debug_state() -> Dict[str, Any]:
    """Return the current debug-mode state."""
    return debug_mode.status()


@app.post("/api/debug")
async def set_debug_state(body: Dict[str, Any]) -> Dict[str, Any]:
    """Enable or disable debug mode at runtime.

    Body: ``{"enabled": true|false}``.
    """
    enabled = body.get("enabled")
    if not isinstance(enabled, bool):
        raise HTTPException(
            status_code=400, detail="Body must contain boolean 'enabled'"
        )
    debug_mode.set_debug(enabled)
    logger.info("[debug] Debug mode %s via API", "enabled" if enabled else "disabled")
    return debug_mode.status()


@app.get("/api/debug/log")
async def debug_log(since: int = 0) -> Dict[str, Any]:
    """Return buffered debug-trace events with seq > since (incremental polling)."""
    return {"events": debug_mode.get_events(since), "enabled": debug_mode.debug_enabled()}


@app.post("/api/debug/clear")
async def debug_clear() -> Dict[str, Any]:
    """Clear the debug-trace event buffer."""
    debug_mode.clear_events()
    return {"status": "ok"}


@app.get("/api/engines/selector-hints")
async def selector_hints() -> Dict[str, Any]:
    """Return runtime-discovered best selectors for all active engine instances.

    Only engines that have processed at least one prompt will have cached
    selector data.  The UI can use this to suggest JSON reordering.
    """
    mgr = EngineManager.get()
    data: Dict[str, Any] = {}
    for name, engine in mgr.engines.items():
        data[name] = {
            "prompt_selector": getattr(engine, "_cached_prompt_selector", None),
            "send_selector": getattr(engine, "_cached_send_selector", None),
            "prompt_area_selectors": getattr(engine, "prompt_area_selectors", []),
            "send_button_selectors": getattr(engine, "send_button_selectors", []),
        }
    return {"data": data}


@app.post("/reset")
async def reset_state() -> Dict[str, Any]:
    global RESET_IN_PROGRESS
    manager = EngineManager.get()
    errors: list[str] = []

    RESET_IN_PROGRESS = True
    try:
        # Cancel in-flight prompt handlers (soft cancellation)
        try:
            await _cancel_inflight_tasks()
        except Exception as e:
            logger.warning(f"[reset] cancel_inflight_tasks error: {e}")
            errors.append(f"cancel: {e}")

        # Drain per-engine queues and cancel pending jobs
        try:
            await manager.drain_queues()
        except Exception as e:
            logger.warning(f"[reset] drain_queues error: {e}")
            errors.append(f"drain: {e}")

        try:
            await manager.stop_all()
        except Exception as e:
            logger.warning(f"[reset] stop_all error (continuing): {e}")
            errors.append(f"stop_all: {e}")

        manager.engines.clear()
        manager.active_engine = None
        rate_limit_store.clear()

        # Clear any runtime counters in DB so stats UI also resets
        clear_stats()
        clear_prompt_logs()

        message = (
            "Engine state cleared"
            if not errors
            else f"Engine state cleared (with errors: {'; '.join(errors)})"
        )
        return {"status": "ok", "message": message}

    finally:
        RESET_IN_PROGRESS = False


@app.post("/api/reset")
async def api_reset_state() -> Dict[str, Any]:
    return await reset_state()


@app.post("/api/session/reset")
async def reset_session() -> Dict[str, Any]:
    """Gracefully reset the browser session: cancel in-flight requests, drain
    every engine's queue, then cleanly restart the shared browser.

    Unlike ``/api/session/kill`` (SIGKILL — can lose whatever session/login
    state Chrome hadn't flushed to disk yet), this lets Chrome exit normally
    so the native profile keeps the login intact. Unlike ``/reset`` (which
    also wipes stats and prompt history), this only touches browser/queue
    state, so it's the right button for "something's stuck" without wanting
    to lose login or history.

    Use this e.g. after switching engines mid-request left the shared
    browser being driven by two requests at once.
    """
    global RESET_IN_PROGRESS
    manager = EngineManager.get()
    errors: list[str] = []

    RESET_IN_PROGRESS = True
    try:
        try:
            await _cancel_inflight_tasks()
        except Exception as e:
            logger.warning(f"[session_reset] cancel_inflight_tasks error: {e}")
            errors.append(f"cancel: {e}")

        try:
            await manager.drain_queues()
        except Exception as e:
            logger.warning(f"[session_reset] drain_queues error: {e}")
            errors.append(f"drain: {e}")

        try:
            await manager.stop_all()
        except Exception as e:
            logger.warning(f"[session_reset] stop_all error (continuing): {e}")
            errors.append(f"stop_all: {e}")

        manager.engines.clear()
        manager.active_engine = None

        message = (
            "Session reset — browser restarted gracefully, login preserved, queue cleared"
            if not errors
            else f"Session reset with errors: {'; '.join(errors)}"
        )
        return {"status": "ok", "message": message}
    finally:
        RESET_IN_PROGRESS = False


@app.post("/api/session/kill")
async def kill_session() -> Dict[str, Any]:
    """Force-kill the browser session immediately (SIGKILL).

    Use this when the browser is completely frozen and normal reset
    doesn't work.  Engine instances stay in memory and will
    auto-reinitialise the browser on the next request.
    """
    from core.zendriver_llm_base import force_kill_session

    manager = EngineManager.get()
    errors: list[str] = []

    # Drain queues so pending jobs don't pile up on the dead session
    try:
        await manager.drain_queues()
    except Exception as e:
        logger.warning(f"[kill_session] drain_queues error: {e}")
        errors.append(f"drain: {e}")

    # Invalidate driver references on all engine instances
    for name, engine in manager.engines.items():
        try:
            engine.driver = None
            engine._initialized = False
            engine._cookies_restored = False
        except Exception as e:
            logger.warning(f"[kill_session] clear engine {name}: {e}")
            errors.append(f"engine_{name}: {e}")

    # Force-kill browser processes
    try:
        await force_kill_session()
    except Exception as e:
        logger.error(f"[kill_session] force_kill_session error: {e}")
        errors.append(f"kill: {e}")

    message = (
        "Browser session killed — next request will start a new session"
        if not errors
        else f"Browser session killed (with errors: {'; '.join(errors)})"
    )
    logger.info(f"[kill_session] {message}")
    return {"status": "ok", "message": message}


@app.get("/logs")
async def logs(
    limit: int = 50,
    offset: int = 0,
    engine: str | None = None,
    model: str | None = None,
    status: str | None = None,
):
    return get_prompt_logs(
        limit=limit,
        offset=offset,
        engine=engine,
        model=model,
        status=status,
    )


@app.get("/api/history")
async def history(
    limit: int = 50,
    offset: int = 0,
    engine: str | None = None,
    model: str | None = None,
    status: str | None = None,
):
    return get_prompt_logs(
        limit=limit,
        offset=offset,
        engine=engine,
        model=model,
        status=status,
    )


@app.get("/ui", response_class=HTMLResponse)
async def ui() -> Any:
    html = Path("./web/index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)
