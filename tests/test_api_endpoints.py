from fastapi.testclient import TestClient
import asyncio
import json
import threading
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock
import pytest

import app as app_module
from app import app, _register_engine_routes
from core.engine_manager import EngineManager

client = TestClient(app)


def make_tab(**overrides):
    """A MagicMock standing in for a zendriver ``Tab``.

    All the async entry points ``ZendriverLLMBase`` calls on a Tab are
    pre-wired as ``AsyncMock`` so ``await tab.method(...)`` works out of the
    box; override any of them (or add attrs) via keyword arguments.
    """
    tab = MagicMock()
    tab.query_selector_all = AsyncMock(return_value=[])
    tab.xpath = AsyncMock(return_value=[])
    tab.get = AsyncMock(return_value=None)
    tab.evaluate = AsyncMock(return_value="")
    tab.reload = AsyncMock(return_value=None)
    tab.wait_for_ready_state = AsyncMock(return_value=True)
    tab.get_content = AsyncMock(return_value="")
    tab.url = "https://example.com"
    for key, value in overrides.items():
        setattr(tab, key, value)
    return tab


def make_element(*, displayed=True, enabled=True, tag="div", attrs=None, apply_result="", html=""):
    """A MagicMock standing in for a zendriver ``Element``.

    ``apply_result`` seeds ``element.apply(...)`` (used by the engine for
    innerText/textContent reads and arbitrary JS) — pass a callable to react
    to the JS source, or a plain value to always return it.
    """
    el = MagicMock()
    el.tag_name = tag
    attrs = dict(attrs or {})
    if not enabled:
        attrs.setdefault("disabled", "true")
    el.attrs = attrs
    position = MagicMock() if displayed else None
    el.get_position = AsyncMock(return_value=position)
    el.click = AsyncMock(return_value=None)
    if callable(apply_result) and not isinstance(apply_result, MagicMock):
        el.apply = AsyncMock(side_effect=apply_result)
    else:
        el.apply = AsyncMock(return_value=apply_result)
    el.send_keys = AsyncMock(return_value=None)
    el.send_file = AsyncMock(return_value=None)
    el.clear_input = AsyncMock(return_value=None)
    el.scroll_into_view = AsyncMock(return_value=None)
    el.get_html = AsyncMock(return_value=html)
    return el


class DummyEngine:
    def __init__(self):
        self.model = "default"
        self.last_media: list[Any] = []
        self.last_agent_mode: bool = False
        self.last_prompt: str = ""
        # When set, generate_response returns this instead of the default text.
        # May be a string or a list of strings (consumed one per call, useful
        # for simulating reformulation retries).
        self.next_response: Any = None

    def get_interface_limits(self):
        return {"max_prompt_chars": 1234, "model_name": "default"}

    def get_supported_models(self):
        return ["default"]

    async def start_login_flow(self):
        return {"logged_in": False, "login_state": "unlogged"}

    async def check_login_state(self):
        return {"logged_in": False, "login_state": "unlogged"}

    async def generate_response(self, prompt, media=None, timeout=None, agent_mode=False):
        self.last_media = media or []
        self.last_agent_mode = agent_mode
        self.last_prompt = prompt
        if isinstance(self.next_response, list) and self.next_response:
            return self.next_response.pop(0)
        if self.next_response is not None:
            return self.next_response
        return "dummy response"

    def get_current_model(self):
        return "default"


@pytest.fixture(autouse=True)
def setup_engine_manager(monkeypatch):
    """Replace the EngineManager singleton with a pre-loaded test instance."""
    # Mock DB calls so no filesystem writes happen during tests
    monkeypatch.setattr("app.inc_requests", lambda: None)
    monkeypatch.setattr("app.inc_responses", lambda: None)
    monkeypatch.setattr("app.inc_errors", lambda: None)
    monkeypatch.setattr("app.log_prompt", lambda *a, **kw: None)
    monkeypatch.setattr("app.inc_media_sent", lambda *a, **kw: None)

    # Reset the per-client rate-limit window so tests don't exhaust the shared
    # 20-req/60s budget across the (large) suite and get spurious 429s.
    app_module.rate_limit_store.clear()

    mgr = EngineManager.get()
    mgr.engines.clear()
    mgr.active_engine = None
    # Circuit-breaker state lives on the singleton: POST /login/<engine> pauses
    # the engine until a sign-in is seen, which must not leak into other tests.
    mgr._health.clear()

    # Inject two synthetic descriptors so /models and /api/engines work
    from core.engine_manager import EngineDescriptor

    chatgpt_desc = EngineDescriptor(
        name="chatgpt",
        aliases=["chatgpt", "openai", "gpt"],
        display_name="ChatGPT (test)",
        service_url="https://chat.openai.com",
        models={"default": 51000},
        default_model="default",
        source="builtin",
        source_path="<test>",
        media_capabilities=["image", "audio"],
    )
    gemini_desc = EngineDescriptor(
        name="gemini",
        aliases=["gemini", "google"],
        display_name="Gemini (test)",
        service_url="https://gemini.google.com",
        models={"default": 32000},
        default_model="default",
        source="builtin",
        source_path="<test>",
        media_capabilities=["image"],
    )
    mgr._descriptors = {"chatgpt": chatgpt_desc, "gemini": gemini_desc}
    mgr._alias_map = {
        "chatgpt": "chatgpt",
        "openai": "chatgpt",
        "gpt": "chatgpt",
        "gemini": "gemini",
        "google": "gemini",
    }

    # Re-register dynamic per-engine routes for this test fixture
    _register_engine_routes(app)

    # Pre-populate with DummyEngine instances so no real Selenium init happens
    mgr.engines["chatgpt"] = DummyEngine()
    mgr.engines["gemini"] = DummyEngine()

    yield

    mgr.engines.clear()
    mgr.active_engine = None
    mgr._health.clear()


def test_ping():
    response = client.get("/api/ping")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_models():
    """Legacy /models must return OpenAI-compatible format (id field required by clients like Alpaca)."""
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert "data" in data
    for entry in data["data"]:
        assert "id" in entry, "Each model entry must have an 'id' field"
        assert entry["id"] is not None
        assert entry["object"] == "model"
        # Legacy extra fields still present
        assert "name" in entry


def test_parse_media_part_accepts_generic_file():
    from app import _parse_media_part

    part = {
        "type": "input_file",
        "data": "data:text/plain;base64,Zm9vYmFy",
        "mime_type": "text/plain",
        "filename": "hello.txt",
    }
    item = _parse_media_part(part, 0)
    assert item.media_type == "document"
    assert item.mime_type == "text/plain"
    assert item.filename == "hello.txt"
    assert item.data == b"foobar"


def test_parse_media_part_openai_image_url_format():
    """image_url payload with nested {'image_url': {'url': '...'}} must be parsed."""
    from app import _parse_media_part

    image_data = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAA"
        "AAC0lEQVR4nGNgYAAAAAMAAWgmWQ0AAAAASUVORK5CYII="
    )
    part = {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{image_data}"},
    }
    item = _parse_media_part(part, 0)
    assert item.media_type == "image"
    assert item.mime_type == "image/png"


def test_multimodal_openai_vision_format():
    """End-to-end: OpenAI vision format {'image_url': {'url': 'data:...'}} reaches the engine."""
    image_data = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAA"
        "AAC0lEQVR4nGNgYAAAAAMAAWgmWQ0AAAAASUVORK5CYII="
    )
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "chatgpt",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "content": "Describe this image:"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_data}"},
                        },
                    ],
                }
            ],
        },
    )
    assert response.status_code == 200
    engine = EngineManager.get().engines["chatgpt"]
    assert len(engine.last_media) == 1
    assert engine.last_media[0].media_type == "image"
    assert engine.last_media[0].mime_type == "image/png"


def test_normalize_prompt_payload_includes_input_file():
    from app import _normalize_prompt_payload

    payload = [
        {
            "role": "user",
            "content": [
                {"type": "text", "content": "Please read this file."},
                {
                    "type": "input_file",
                    "data": "data:text/plain;base64,Zm9vYmFy",
                    "mime_type": "text/plain",
                    "filename": "hello.txt",
                },
            ],
        }
    ]
    prompt_text, media_items = _normalize_prompt_payload(payload)
    assert "Please read this file." in prompt_text
    assert len(media_items) == 1
    assert media_items[0].media_type == "document"


def test_legacy_chat_completions():
    """POST /chat/completions (without /v1) must work as alias."""
    response = client.post(
        "/chat/completions",
        json={"model": "chatgpt", "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"


def test_login_state():
    response = client.post("/login/chatgpt")
    assert response.status_code == 200
    assert response.json()["login_state"] == "unlogged"


def test_prompt_legacy_chatgpt():
    response = client.post("/chatgpt/prompt", json={"prompt": "Hello"})
    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["message"]["content"] == "dummy response"


def test_prompt_invalid_json_body():
    response = client.post(
        "/chatgpt/prompt",
        data="http://localhost:14848/v1/chat/completions",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 400
    assert "Invalid JSON body" in response.json()["detail"]


def test_prompt_dynamic_endpoint():
    response = client.post("/engine/chatgpt/prompt", json={"prompt": "Hello"})
    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["message"]["content"] == "dummy response"


def test_prompt_dynamic_endpoint_alias():
    """Engine aliases should work on the dynamic endpoint too."""
    response = client.post("/engine/openai/prompt", json={"prompt": "Hello"})
    assert response.status_code == 200


def test_multimodal_text_and_image():
    image_data = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAA"
        "AAC0lEQVR4nGNgYAAAAAMAAWgmWQ0AAAAASUVORK5CYII="
    )
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "chatgpt",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "content": "Here is an image:"},
                        {
                            "type": "image_url",
                            "url": f"data:image/png;base64,{image_data}",
                            "filename": "test.png",
                        },
                    ],
                }
            ],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    engine = EngineManager.get().engines["chatgpt"]
    assert len(engine.last_media) == 1
    assert engine.last_media[0].media_type == "image"


def test_multimodal_text_and_image_json_string_content():
    image_data = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAA"
        "AAC0lEQVR4nGNgYAAAAAMAAWgmWQ0AAAAASUVORK5CYII="
    )
    payload = {
        "model": "chatgpt",
        "messages": [
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "type": "text",
                        "content": "Here is an image:",
                        "attachments": [
                            {
                                "mime_type": "image/png",
                                "data": f"data:image/png;base64,{image_data}",
                                "filename": "test.png",
                            }
                        ],
                    }
                ),
            }
        ],
    }

    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    engine = EngineManager.get().engines["chatgpt"]
    assert len(engine.last_media) == 1
    assert engine.last_media[0].media_type == "image"
    assert engine.last_media[0].filename == "test.png"


def test_prompt_unknown_engine():
    response = client.post("/engine/nonexistent/prompt", json={"prompt": "Hello"})
    assert response.status_code == 404


def test_api_engines():
    response = client.get("/api/engines")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    names = [e["name"] for e in data["data"]]
    assert "chatgpt" in names
    assert "gemini" in names


def test_api_engines_include_media_capabilities():
    response = client.get("/api/engines")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    for entry in data["data"]:
        assert "media_capabilities" in entry
        assert isinstance(entry["media_capabilities"], list)

    chatgpt_entry = next((e for e in data["data"] if e["name"] == "chatgpt"), None)
    assert chatgpt_entry is not None
    assert "image" in chatgpt_entry["media_capabilities"]


def test_api_engines_reload():
    """Reload endpoint must return 200 and a valid data list."""
    response = client.post("/api/engines/reload")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert isinstance(data["data"], list)


def test_unlogged_flag_behavior():
    class FakeBaseEngine:
        def __init__(self, model_limits_map, default_model, allow_unlogged=False):
            self.model_limits_map = model_limits_map
            self.default_model = default_model
            self.allow_unlogged = allow_unlogged
            self._logged_in = True

        def is_user_logged_in(self):
            return self._logged_in

        def set_logged_in(self, value):
            self._logged_in = value

        def get_current_model(self):
            if not self.is_user_logged_in() and self.allow_unlogged and "unlogged" in self.model_limits_map:
                return "unlogged"
            return self.default_model

    base_engine = FakeBaseEngine(
        model_limits_map={"unlogged": 20000, "default": 50000},
        default_model="default",
    )
    base_engine.set_logged_in(False)
    assert base_engine.get_current_model() == "default"

    unlogged_engine = FakeBaseEngine(
        model_limits_map={"unlogged": 20000, "default": 50000},
        default_model="default",
        allow_unlogged=True,
    )
    unlogged_engine.set_logged_in(False)
    assert unlogged_engine.get_current_model() == "unlogged"


def test_media_limits_fallback_without_config():
    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    media_items = [type("M", (), {"media_type": "image"})()]
    assert engine._check_media_limits(media_items, "base") is None


def test_is_same_site_treats_in_site_navigation_as_not_a_redirect():
    """Some engines open a new conversation on send and navigate to a different
    path on the SAME host (e.g. '/c' -> '/chat/<id>'). That must not be flagged
    as a redirect-stall. Only a navigation to a DIFFERENT host is a real stall.
    """
    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://example.com/c",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    # Same host, different path -> in-site navigation, not a stall.
    assert engine._is_same_site("https://example.com/chat/abc-123") is True
    assert engine._is_same_site("https://example.com/") is True
    # Different host -> genuine off-site redirect (e.g. auth/login).
    assert engine._is_same_site("https://auth.example.org/login") is False
    # Empty/unknown URL is treated conservatively as same-site (no false stall).
    assert engine._is_same_site("") is True


def test_offsite_redirect_stall_is_distinguished_from_generic_stall():
    """A deterministic cross-host redirect (e.g. the service bounced to a
    login/auth domain) must be classified as an off-site stall so the retry
    loop can fail fast, while a generic redirect-stall must NOT be treated as
    off-site (it may be transient and worth retrying).
    """
    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://example.com/c",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    offsite = RuntimeError("redirect-stall (off-site): send not accepted after redirect")
    generic = RuntimeError("redirect-stall: send not accepted after redirect")

    # Off-site stall is recognised as off-site (fail fast).
    assert engine._is_offsite_redirect_stall(offsite) is True
    # Generic stall is NOT off-site (may retry).
    assert engine._is_offsite_redirect_stall(generic) is False
    # A generic stall is still a redirect-stall...
    assert engine._is_redirect_stall(generic) is True
    # ...whereas the off-site variant is not matched by the generic detector,
    # ensuring it takes the fail-fast branch instead of the retry branch.
    assert engine._is_redirect_stall(offsite) is False


def test_stepfun_audio_not_supported_by_model():
    from core.zendriver_llm_base import ZendriverLLMBase

    with open("engines/stepfun.json", encoding="utf-8") as fh:
        cfg = json.load(fh)

    engine = ZendriverLLMBase(
        service_url=cfg["service_url"],
        model_limits_map=cfg["models"],
        default_model=cfg.get("default_model", "default"),
    )
    engine.media_config = cfg.get("media_support", {})
    media_items = [type("M", (), {"media_type": "audio"})()]
    assert engine._check_media_limits(media_items, "paid") is None


def test_media_with_model_not_listed_is_allowed_to_try():
    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000, "other": 1000},
        default_model="other",
    )
    engine.media_config = {
        "audio": {
            "limits": {"default": -1},
            "supported_models": ["default"],
        }
    }
    media_items = [type("M", (), {"media_type": "audio"})()]
    assert engine._check_media_limits(media_items, "default") is None


def test_supported_models_all_allows_every_model():
    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000, "other": 1000},
        default_model="other",
    )
    engine.media_config = {
        "audio": {
            "limits": {"default": -1},
            "supported_models": ["all"],
        }
    }
    media_items = [type("M", (), {"media_type": "audio"})()]
    assert engine._check_media_limits(media_items, "default") is None


def test_supported_models_not_unlogged_allows_logged_models():
    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000, "unlogged": 1000},
        default_model="default",
    )
    engine.media_config = {
        "audio": {
            "limits": {"default": -1, "unlogged": 0},
            "supported_models": ["not-unlogged"],
        }
    }
    media_items = [type("M", (), {"media_type": "audio"})()]
    assert engine._check_media_limits(media_items, "default") is None


def test_upload_via_file_input_rejects_missing_value():
    """Pre-existing failure (unrelated to the zendriver migration): the real
    ``_upload_via_file_input`` trusts a successful send (``send_ok``) even
    when the 2s post-send poll never observes a value/files count, so with a
    mock ``send_file`` that raises nothing this returns True, not False. That
    mismatch between this test's expectation and the implementation predates
    this port (see CLAUDE.md/AGENTS.md task history) — kept as-is, still
    failing for the same underlying reason, not a new regression."""
    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    engine.media_config = {"audio": {"upload_selectors": ["input[type='file']"]}}
    mock_input = make_element(tag="input", attrs={"type": "file"}, apply_result=0)
    tab = make_tab(query_selector_all=AsyncMock(return_value=[mock_input]))

    result = asyncio.run(engine._upload_via_file_input(
        type("M", (), {"media_type": "audio"})(),
        "/tmp/dummy.mp3",
        tab,
    ))
    assert result is False


def test_upload_via_file_input_accepts_file_list():
    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    engine.media_config = {"audio": {"upload_selectors": ["input[type='file']"]}}
    mock_input = make_element(tag="input", attrs={"type": "file"}, apply_result=1)
    tab = make_tab(query_selector_all=AsyncMock(return_value=[mock_input]))

    result = asyncio.run(engine._upload_via_file_input(
        type("M", (), {"media_type": "audio"})(),
        "/tmp/dummy.mp3",
        tab,
    ))
    assert result is True


def test_upload_via_file_input_trusts_send_keys_when_spa_clears_files():
    """Regression: Angular/SPA resets .files/.value after processing; send_file success should be trusted."""
    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    engine.media_config = {"image": {"upload_selectors": ["input[type='file']"]}}

    # type check "file"; apply() (files.length) always 0 (SPA already cleared
    # the FileList) but send_file() succeeds without raising.
    mock_input = make_element(tag="input", attrs={"type": "file"}, apply_result=0)
    tab = make_tab(query_selector_all=AsyncMock(return_value=[mock_input]))

    result = asyncio.run(engine._upload_via_file_input(
        type("M", (), {"media_type": "image"})(),
        "/tmp/dummy.png",
        tab,
    ))

    assert result is True, "Should trust send_file success even when SPA clears .files"


def test_upload_via_clipboard_uses_popen_for_xclip():
    """Regression: xclip -i blocks until clipboard is read; must use Popen, not run."""
    import os
    import tempfile
    from core.zendriver_llm_base import ZendriverLLMBase
    from unittest.mock import patch

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    engine.prompt_area_selectors = ["textarea"]

    tab = make_tab()
    mock_input_el = make_element()
    engine._find_interactable_element = AsyncMock(return_value=mock_input_el)

    mock_proc = MagicMock()
    item = type("M", (), {"media_type": "image", "mime_type": "image/png"})()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.write(b"\x89PNG")
    tmp.close()

    try:
        with patch("shutil.which", side_effect=lambda cmd: cmd if cmd == "xclip" else None), \
             patch("subprocess.Popen", return_value=mock_proc) as mock_popen, \
             patch("subprocess.run") as mock_run:
            result = asyncio.run(engine._upload_via_clipboard(item, tmp.name, tab))
        assert result is True
        # Popen must be called (non-blocking); subprocess.run must NOT be called for xclip
        mock_popen.assert_called_once()
        mock_run.assert_not_called()
        # xclip process must be terminated after paste
        mock_proc.terminate.assert_called_once()
        # Paste is delivered as a real Ctrl+V CDP key combo, not a JS value write.
        assert mock_input_el.send_keys.called
    finally:
        os.unlink(tmp.name)


def test_upload_via_file_input_attempts_visibility_fallback_for_hidden_inputs():
    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    engine.media_config = {"image": {"upload_selectors": ["input[type='file']"]}}

    mock_input = make_element(tag="input", attrs={"type": "file"}, apply_result=1)
    mock_input.send_file = AsyncMock(side_effect=[Exception("element not interactable"), None])
    tab = make_tab(query_selector_all=AsyncMock(return_value=[mock_input]))

    result = asyncio.run(engine._upload_via_file_input(
        type("M", (), {"media_type": "image"})(),
        "/tmp/dummy.png",
        tab,
    ))

    assert result is True
    # The visibility-forcing JS ran as part of the send_file retry fallback.
    assert mock_input.apply.called


def test_upload_media_returns_false_when_all_upload_paths_fail():
    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    engine.media_config = {"audio": {"upload_selectors": ["input[type='file']"]}}
    tab = make_tab()  # query_selector_all() -> [] for every selector

    result = asyncio.run(engine._upload_media(
        [type("M", (), {"media_type": "audio", "mime_type": "audio/mpeg", "data": b"dummy"})()],
        tab,
    ))
    assert result is False


def test_upload_media_clicks_accept_buttons_after_successful_upload():
    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    engine.media_config = {"image": {"upload_selectors": ["input[type='file']"]}}

    mock_input = make_element(tag="input", attrs={"type": "file"}, apply_result=1)

    async def _query_side_effect(sel):
        return [mock_input] if sel == "input[type='file']" else []

    tab = make_tab(query_selector_all=AsyncMock(side_effect=_query_side_effect))
    engine._click_accept_buttons = AsyncMock()

    result = asyncio.run(engine._upload_media(
        [type("M", (), {"media_type": "image", "mime_type": "image/png", "data": b"dummy"})()],
        tab,
    ))

    assert result is True
    engine._click_accept_buttons.assert_called_once_with(tab, timeout=5.0)


def test_sync_generate_response_once_returns_error_when_send_button_not_ready_after_media_upload():
    import tempfile

    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    engine.driver = make_tab()
    engine.service_url = "https://example.com"
    engine._ensure_ready = AsyncMock(return_value=None)
    engine.refresh_login_state = AsyncMock(return_value=True)
    engine._click_accept_buttons = AsyncMock(return_value=None)
    engine._is_captcha_present = AsyncMock(return_value=False)
    engine._is_limit_present = AsyncMock(return_value=False)
    engine._check_account_tier = AsyncMock(return_value="base")
    engine._check_media_limits = lambda media, tier: None
    engine._upload_media = AsyncMock(return_value=True)
    engine._find_interactable_element = AsyncMock(return_value=make_element())
    engine._fill_input = AsyncMock(return_value=None)
    engine._wait_for_send_button_after_media_upload = AsyncMock(return_value=False)
    engine._click_send = AsyncMock(return_value=None)
    engine._post_send_check = AsyncMock(return_value=True)
    engine._wait_for_response = AsyncMock(return_value="response")

    result = asyncio.run(engine._generate_response_once("Hello", [type(
        "M", (), {"media_type": "audio", "mime_type": "audio/mpeg", "data": b"dummy"}
    )()]))
    assert result == "⚠️ Media upload failed. Please verify the file and try again."


def test_total_media_limit_applies_across_types():
    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    engine.media_config = {
        "total_limits": {"limits": {"default": 2}},
        "image": {"limits": {"default": -1}},
        "audio": {"limits": {"default": -1}},
    }
    media_items = [
        type("M", (), {"media_type": "image"})(),
        type("M", (), {"media_type": "audio"})(),
        type("M", (), {"media_type": "image"})(),
    ]

    assert engine._check_media_limits(media_items, "default") == (
        "⚠️ The use of 'media' is exhausted for today. Please try again tomorrow."
    )


def test_reset_state():
    manager = EngineManager.get()
    # set active engine then verify reset clears it
    manager.active_engine = manager.engines.get("chatgpt")
    assert manager.active_engine is not None

    response = client.post("/reset")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert manager.engines == {}
    assert manager.active_engine is None

    stats_res = client.get("/stats")
    assert stats_res.status_code == 200
    stats_data = stats_res.json()
    assert "stats" in stats_data
    # if DB is writable/clearable, stats may be empty; if readonly they may persist
    assert isinstance(stats_data["stats"], dict)
    assert "response_time" in stats_data
    assert "global_avg_ms" in stats_data["response_time"]
    assert "per_engine_avg_ms" in stats_data["response_time"]
    assert isinstance(stats_data["response_time"]["per_engine_avg_ms"], dict)


def test_media_counter_increments_on_prompt_with_media(monkeypatch):
    called = {"count": 0, "amount": 0}
    def record(amount):
        called["count"] += 1
        called["amount"] = amount
    monkeypatch.setattr("app.inc_media_sent", record)

    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "content": "hello"},
                    {"type": "image", "data": "data:image/png;base64,AAAA"},
                ],
            }
        ]
    }
    response = client.post("/engine/chatgpt/prompt", json=payload)
    assert response.status_code == 200
    assert called["count"] == 1
    assert called["amount"] == 1


def test_stats_returns_media_availability_by_tier():
    mgr = EngineManager.get()
    chatgpt_desc = mgr._descriptors["chatgpt"]
    gemini_desc = mgr._descriptors["gemini"]
    chatgpt_desc.media_support = {
        "image": {"limits": {"unlogged": 0, "base": -1, "paid": -1}}
    }
    gemini_desc.media_support = {
        "audio": {"limits": {"unlogged": 0, "base": 2, "paid": 0}}
    }

    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["media_sent_today"] == 0
    assert data["media_availability"]["unlogged"] == 0
    assert data["media_availability"]["base"] == 4
    assert data["media_availability"]["paid"] == -1


def test_api_reset_alias():
    manager = EngineManager.get()
    manager.active_engine = manager.engines.get("chatgpt")

    response = client.post("/api/reset")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_reset_session_clears_engines_but_not_stats_or_history(monkeypatch):
    """/api/session/reset clears engine/browser/queue state (same as /reset)
    but must NOT wipe stats or prompt history -- that's what distinguishes it
    from /reset, so it's safe to use as a "something's stuck" unblock without
    losing history. Also must not SIGKILL: stop_all() is the graceful path
    (see EngineManager.stop_all / shutdown_shared_driver), unlike
    /api/session/kill's force_kill_session()."""
    manager = EngineManager.get()
    manager.active_engine = manager.engines.get("chatgpt")

    stop_all_calls = []

    async def fake_stop_all():
        stop_all_calls.append(1)

    monkeypatch.setattr(manager, "stop_all", fake_stop_all)
    # drain_queues() itself is exercised by test_reset_state/test_api_reset_alias
    # already; stub it here so this test stays focused on (and immune to
    # ordering effects on) the stats/history/graceful-stop distinction it's
    # actually about.
    monkeypatch.setattr(manager, "drain_queues", AsyncMock())

    stats_cleared = []
    history_cleared = []
    monkeypatch.setattr("app.clear_stats", lambda: stats_cleared.append(1))
    monkeypatch.setattr("app.clear_prompt_logs", lambda: history_cleared.append(1))

    response = client.post("/api/session/reset")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "preserved" in body["message"].lower()

    assert manager.engines == {}
    assert manager.active_engine is None
    assert stop_all_calls == [1]  # graceful stop_all(), not force_kill_session()
    assert stats_cleared == []
    assert history_cleared == []


def test_reset_cancels_inflight_requests():
    class SlowDummyEngine:
        def __init__(self):
            self.model = "default"

        def get_interface_limits(self):
            return {"max_prompt_chars": 1234, "model_name": "default"}

        def get_supported_models(self):
            return ["default"]

        async def start_login_flow(self):
            return {"logged_in": False, "login_state": "unlogged"}

        async def check_login_state(self):
            return {"logged_in": False, "login_state": "unlogged"}

        async def generate_response(self, prompt):
            await asyncio.sleep(3)
            return "slow response"

        async def stop(self):
            return

        def get_current_model(self):
            return "default"

    manager = EngineManager.get()
    manager.engines["chatgpt"] = SlowDummyEngine()

    results = {}

    def call_prompt():
        try:
            r = client.post("/engine/chatgpt/prompt", json={"prompt": "hello"})
            results["response"] = r
        except Exception as e:
            results["error"] = e

    thread = threading.Thread(target=call_prompt)
    thread.start()

    # wait a moment for request to be in-flight
    time.sleep(0.1)

    response = client.post("/reset")
    assert response.status_code == 200

    thread.join(timeout=10)
    assert not thread.is_alive()

    assert "response" in results or "error" in results
    if "response" in results:
        assert results["response"].status_code in (503, 500)


def test_logs_history_endpoint():
    # ensure prompt logging endpoint is accessible and returns a list
    response = client.get("/logs?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_api_history_endpoint():
    response = client.get("/api/history?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_captcha_detection_short_circuit(monkeypatch):
    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://chat.openai.com",
        model_limits_map={"default": 50000},
        default_model="default",
    )

    async def _query_side_effect(sel):
        return [make_element()] if sel == "iframe#cf-chl-widget-ezspn" else []

    engine.driver = make_tab(
        query_selector_all=AsyncMock(side_effect=_query_side_effect),
        url="https://chat.openai.com",
    )
    engine._ensure_ready = AsyncMock(return_value=None)
    engine.refresh_login_state = AsyncMock(return_value=True)
    engine._click_accept_buttons = AsyncMock(return_value=None)

    result = asyncio.run(engine._generate_response_once("Hello"))
    assert "CAPTCHA" in result or "captcha" in result
    assert "Please complete" in result


def test_json_engine_loads_error_indicator_selectors():
    """Gemini renders transient failures ("Something went wrong (1076)") in an
    Angular Material snackbar outside the response area. Ensure the JSON
    ``error_indicators`` selectors are loaded so they can be detected."""
    from pathlib import Path
    from core.json_engine import JsonEngine

    engines_dir = Path(__file__).parent.parent / "engines"
    engine = JsonEngine(engines_dir / "gemini.json")

    assert "mat-snack-bar-container" in engine.error_indicator_selectors
    assert "simple-snack-bar" in " ".join(engine.error_indicator_selectors)


def test_gemini_authenticated_selectors_exclude_chat_ui():
    """Chat input/response classes must not be treated as a login signal.

    Regression: gemini.json's allow_unlogged=true means Gemini's own chat UI
    (div.assistant-message, .gemini-response, .chat-message.ai,
    div.chat-input-container) renders identically for anonymous and signed-in
    users, so using them as authenticated_css_selectors made the check assume
    "logged in" (via the fallback in _ensure_logged_in) even while a real
    "Sign in" button was visible on the page.
    """
    from pathlib import Path

    from core.json_engine import JsonEngine

    engines_dir = Path(__file__).parent.parent / "engines"
    engine = JsonEngine(engines_dir / "gemini.json")

    joined = " ".join(engine._login_cfg.get("authenticated_css_selectors", []))
    for weak in ("chat-input-container", "assistant-message", "gemini-response", "chat-message"):
        assert weak not in joined, f"{weak!r} renders in Gemini's unlogged mode too"


def test_gemini_login_detection_flags_signed_out_state_correctly():
    """Reproduces the exact page state observed live in production: the chat
    UI is present (Gemini's anonymous mode renders it too) alongside a real
    "Sign in" link -- must be detected as logged out, not fall through to
    the "assume logged in" default."""
    from pathlib import Path

    from core.json_engine import JsonEngine

    engines_dir = Path(__file__).parent.parent / "engines"
    engine = JsonEngine(engines_dir / "gemini.json")
    login_xpath = engine._login_cfg["login_button_xpath"]

    async def _xpath_side_effect(xp, timeout=2.5):
        return [make_element()] if xp == login_xpath else []

    tab = make_tab(url="https://gemini.google.com/", xpath=AsyncMock(side_effect=_xpath_side_effect))
    # No Google Account button / logout link -- signed out: query_selector_all
    # (used for authenticated_css_selectors) keeps the default empty result.

    assert asyncio.run(engine._ensure_logged_in(tab)) is False


def test_gemini_login_detection_flags_signed_in_state():
    """Uses the actual markup observed on a real, logged-in Gemini page:
    an <a aria-label="Google Account: ..."> (not a <button>) and a
    SignOutOptions link (not "logout") -- the selectors must match the real
    tag/wording, not a guess."""
    from pathlib import Path

    from core.json_engine import JsonEngine

    engines_dir = Path(__file__).parent.parent / "engines"
    engine = JsonEngine(engines_dir / "gemini.json")
    account_selector = engine._login_cfg["authenticated_css_selectors"][0]

    async def _query_side_effect(sel):
        return [make_element()] if sel == account_selector else []

    tab = make_tab(url="https://gemini.google.com/", query_selector_all=AsyncMock(side_effect=_query_side_effect))

    assert asyncio.run(engine._ensure_logged_in(tab)) is True


def test_error_indicator_selectors_default_empty():
    """Engines without an ``error_indicators`` key keep an empty list so the
    base engine behaviour is unchanged (engine-agnostic default)."""
    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://www.example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    assert engine.error_indicator_selectors == []


def test_get_error_indicator_text_returns_text_when_present():
    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://www.example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    engine.error_indicator_selectors = ["mat-snack-bar-container", ".error-message"]

    async def _query_side_effect(sel):
        if sel == "mat-snack-bar-container":
            return [make_element(apply_result="Something went wrong (1076)")]
        return []

    tab = make_tab(query_selector_all=AsyncMock(side_effect=_query_side_effect))

    text = asyncio.run(engine._get_error_indicator_text(tab))
    assert text == "Something went wrong (1076)"


def test_get_error_indicator_text_none_when_not_configured():
    from core.zendriver_llm_base import ZendriverLLMBase

    def _fail(sel):
        raise AssertionError("query_selector_all must not be called when unconfigured")

    engine = ZendriverLLMBase(
        service_url="https://www.example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    tab = make_tab(query_selector_all=AsyncMock(side_effect=_fail))
    # Default empty list -> no lookups, returns None.
    assert asyncio.run(engine._get_error_indicator_text(tab)) is None


def test_get_error_indicator_text_none_when_hidden():
    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://www.example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    engine.error_indicator_selectors = ["mat-snack-bar-container"]
    hidden = make_element(displayed=False, apply_result="Something went wrong (1076)")
    tab = make_tab(query_selector_all=AsyncMock(return_value=[hidden]))
    assert asyncio.run(engine._get_error_indicator_text(tab)) is None


def test_is_engine_error_response_detects_something_went_wrong():
    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://www.example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    assert engine._is_engine_error_response("Something went wrong (1076)") is True
    assert engine._is_engine_error_response("Error 1076") is True
    assert engine._is_engine_error_response("(1076)") is True


def test_is_engine_error_response_ignores_normal_reply_with_number():
    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://www.example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    # A legitimate short reply ending in a number must NOT be flagged.
    assert engine._is_engine_error_response("The answer is 1200") is False
    assert engine._is_engine_error_response("Room 404 is down the hall") is False


def test_check_login_state_no_browser_launch_when_uninitialized():
    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://chat.openai.com",
        model_limits_map={"default": 50000},
        default_model="default",
    )

    # If check_login_state is called before initialization, it must not cause browser init
    called = False

    async def fail_init():
        nonlocal called
        called = True
        raise RuntimeError("_ensure_ready should not be called")

    engine._ensure_ready = fail_init
    engine.driver = None

    state = asyncio.run(engine.check_login_state())
    assert state["login_state"] == "unlogged"
    assert state["logged_in"] is False
    assert called is False


# ---------------------------------------------------------------------------
# OpenAI-compatible /v1/* endpoint tests
# ---------------------------------------------------------------------------


def test_v1_models_list():
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert isinstance(data["data"], list)
    ids = [m["id"] for m in data["data"]]
    assert "chatgpt" in ids
    assert "gemini" in ids
    # only canonical names — no aliases, no provider:variant
    assert not any(":" in mid for mid in ids)
    for entry in data["data"]:
        assert entry["object"] == "model"
        assert entry["owned_by"] == "selenium-llm-engine"


def test_v1_models_single():
    response = client.get("/v1/models/chatgpt")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "chatgpt"
    assert data["object"] == "model"


def test_v1_models_variant():
    response = client.get("/v1/models/chatgpt:default")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "chatgpt:default"


def test_fill_input_contenteditable_triggers_extra_keystroke():
    from core.zendriver_llm_base import ZendriverLLMBase
    from zendriver.core.keys import SpecialKeys

    async def apply_side_effect(js):
        if "document.execCommand('insertText'" in js:
            return None
        if "el.innerText || el.textContent" in js:
            return "test"
        return None

    fake_el = make_element(tag="div", apply_result=apply_side_effect)
    tab = make_tab()

    engine = ZendriverLLMBase(
        service_url="https://www.example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    engine._ensure_ready = AsyncMock(return_value=None)
    engine.driver = tab

    asyncio.run(engine._fill_input(tab, fake_el, "test"))

    assert any(
        "document.execCommand('insertText'" in call.args[0] for call in fake_el.apply.await_args_list
    )
    send_keys_args = [call.args[0] for call in fake_el.send_keys.await_args_list]
    assert SpecialKeys.SPACE in send_keys_args
    assert SpecialKeys.BACKSPACE in send_keys_args


def test_fill_input_verifies_input_value_for_textarea():
    from core.zendriver_llm_base import ZendriverLLMBase

    async def apply_side_effect(js):
        if "el.innerText || el.textContent" in js:
            return "hello world"
        return None

    fake_el = make_element(tag="textarea", apply_result=apply_side_effect)
    tab = make_tab()

    engine = ZendriverLLMBase(
        service_url="https://www.example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    engine._ensure_ready = AsyncMock(return_value=None)
    engine.driver = tab

    asyncio.run(engine._fill_input(tab, fake_el, "hello world"))
    assert fake_el.send_keys.await_args_list[-1].args[0] == "hello world"


def test_fill_input_raises_when_verification_fails():
    from core.zendriver_llm_base import ZendriverLLMBase

    async def apply_side_effect(js):
        if "el.innerText || el.textContent" in js:
            return "wrong text"
        return None

    fake_el = make_element(tag="textarea", apply_result=apply_side_effect)
    tab = make_tab()

    engine = ZendriverLLMBase(
        service_url="https://www.example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    engine._ensure_ready = AsyncMock(return_value=None)
    engine.driver = tab

    with pytest.raises(RuntimeError, match="fill_input verification failed"):
        asyncio.run(engine._fill_input(tab, fake_el, "hello world"))


def test_v1_models_unknown():
    response = client.get("/v1/models/nonexistent_engine")
    assert response.status_code == 404


def test_v1_chat_completions_messages():
    response = client.post(
        "/v1/chat/completions",
        json={"model": "chatgpt", "messages": [{"role": "user", "content": "Hello"}]},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"] == "dummy response"


def test_v1_chat_null_model():
    """model=null must not crash — falls back to default engine."""
    response = client.post(
        "/v1/chat/completions",
        json={"model": None, "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["engine"] == "chatgpt"


def test_api_engines_default_setting():
    response = client.get("/api/engines/default")
    assert response.status_code == 200
    assert response.json()["default_engine"] == "chatgpt"

    response = client.post("/api/engines/default", json={"engine": "gemini"})
    assert response.status_code == 200
    assert response.json()["default_engine"] == "gemini"

    response = client.get("/api/engines/default")
    assert response.status_code == 200
    assert response.json()["default_engine"] == "gemini"

    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Hello"}]},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["engine"] == "gemini"


def test_v1_chat_provider_variant_model():
    """provider:variant notation must resolve to the correct engine."""
    response = client.post(
        "/v1/chat/completions",
        json={"model": "chatgpt:gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert response.status_code == 200


def test_token_count_nonzero():
    response = client.post("/chatgpt/prompt", json={"prompt": "Hello world"})
    assert response.status_code == 200
    usage = response.json()["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_v1_chat_completions_ignores_unsupported_openai_params():
    """Unsupported OpenAI parameters must be ignored rather than causing errors."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "chatgpt",
            "messages": [{"role": "user", "content": "Hello"}],
            "logprobs": True,
            "top_logprobs": 3,
            "n": 2,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["engine"] == "chatgpt"


def test_v1_streaming_sse_format():
    """stream=True must return SSE with chat.completion.chunk objects."""
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "chatgpt", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
    ) as resp:
        assert resp.status_code == 200
        lines = [line for line in resp.iter_lines() if line.startswith("data:")]
    assert any("[DONE]" in line for line in lines)
    data_lines = [line for line in lines if "[DONE]" not in line]
    assert len(data_lines) >= 1
    for line in data_lines:
        chunk = json.loads(line.removeprefix("data:").strip())
        assert chunk["object"] == "chat.completion.chunk"
        assert "choices" in chunk


# ---------------------------------------------------------------------------
# Selector hints endpoint and selector caching regression tests
# ---------------------------------------------------------------------------


def test_selector_hints_empty_when_no_prompts():
    """GET /api/engines/selector-hints returns an empty data dict before any prompt is sent."""
    mgr = EngineManager.get()
    mgr.engines.clear()
    response = client.get("/api/engines/selector-hints")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert data["data"] == {}


def test_selector_hints_structure_after_engine_loaded():
    """Once an engine instance is in the manager the hints endpoint must expose its selector lists."""
    engine = DummyEngine()
    engine.prompt_area_selectors = ["textarea", "div[contenteditable='true']"]
    engine.send_button_selectors = ["button[type='submit']", "button[aria-label*='Send']"]
    engine._cached_prompt_selector = None
    engine._cached_send_selector = None

    mgr = EngineManager.get()
    mgr.engines["chatgpt"] = engine

    response = client.get("/api/engines/selector-hints")
    assert response.status_code == 200
    data = response.json()["data"]
    assert "chatgpt" in data
    hints = data["chatgpt"]
    assert "prompt_selector" in hints
    assert "send_selector" in hints
    assert "prompt_area_selectors" in hints
    assert "send_button_selectors" in hints
    assert hints["prompt_selector"] is None
    assert hints["send_selector"] is None
    assert hints["prompt_area_selectors"] == engine.prompt_area_selectors
    assert hints["send_button_selectors"] == engine.send_button_selectors


def test_selector_hints_reflect_cached_values():
    """Cached selectors are included in the hints response after being set."""
    engine = DummyEngine()
    engine.prompt_area_selectors = ["textarea", "div[contenteditable='true']"]
    engine.send_button_selectors = ["button[type='submit']", "button[aria-label*='Send']"]
    engine._cached_prompt_selector = "div[contenteditable='true']"
    engine._cached_send_selector = "button[aria-label*='Send']"

    mgr = EngineManager.get()
    mgr.engines["gemini"] = engine

    response = client.get("/api/engines/selector-hints")
    assert response.status_code == 200
    hints = response.json()["data"]["gemini"]
    assert hints["prompt_selector"] == "div[contenteditable='true']"
    assert hints["send_selector"] == "button[aria-label*='Send']"


def test_find_interactable_element_caches_selector():
    """_find_interactable_element sets cache_attr to the found selector."""
    from core.zendriver_llm_base import ZendriverLLMBase

    base = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    assert base._cached_prompt_selector is None

    winning_selector = "div[contenteditable='true']"
    fake_el = make_element()

    async def query_side_effect(sel):
        return [fake_el] if sel == winning_selector else []

    tab = make_tab(query_selector_all=AsyncMock(side_effect=query_side_effect))
    selectors = ["textarea", winning_selector]
    result = asyncio.run(base._find_interactable_element(
        tab, selectors, timeout=3.0, cache_attr="_cached_prompt_selector"
    ))

    assert result is fake_el
    assert base._cached_prompt_selector == winning_selector


def test_find_interactable_element_tries_cached_first():
    """When a cached selector exists it is tried before others."""
    from core.zendriver_llm_base import ZendriverLLMBase

    base = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    cached_sel = "div[contenteditable='true']"
    base._cached_prompt_selector = cached_sel

    tried_order: list[str] = []
    fake_el = make_element()

    async def query_side_effect(sel):
        tried_order.append(sel)
        return [fake_el] if sel == cached_sel else []

    tab = make_tab(query_selector_all=AsyncMock(side_effect=query_side_effect))
    selectors = ["textarea", cached_sel, "input"]
    asyncio.run(base._find_interactable_element(
        tab, selectors, timeout=3.0, cache_attr="_cached_prompt_selector"
    ))

    assert tried_order[0] == cached_sel, "Cached selector must be tried first"


def test_find_interactable_element_falls_back_to_visible_non_clickable_element():
    from core.zendriver_llm_base import ZendriverLLMBase

    base = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )

    # Visible but disabled: never satisfies the "clickable" main loop, only
    # the final visible-only fallback pass.
    visible_element = make_element(displayed=True, enabled=False)
    tab = make_tab(query_selector_all=AsyncMock(return_value=[visible_element]))

    result = asyncio.run(base._find_interactable_element(
        tab,
        ["div[contenteditable='true']"],
        timeout=0.5,
        cache_attr="_cached_prompt_selector",
    ))

    assert result is visible_element
    assert base._cached_prompt_selector == "div[contenteditable='true']"


def test_find_interactable_element_handles_stale_cached_selector():
    """If the cached selector's query raises, the other selector is still tried."""
    from core.zendriver_llm_base import ZendriverLLMBase

    base = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    base._cached_prompt_selector = "textarea"

    fake_el = make_element()

    async def query_side_effect(sel):
        if sel == "textarea":
            raise Exception("could not find node with given id")
        if sel == "div[contenteditable='true']":
            return [fake_el]
        return []

    tab = make_tab(query_selector_all=AsyncMock(side_effect=query_side_effect))
    result = asyncio.run(base._find_interactable_element(
        tab,
        ["textarea", "div[contenteditable='true']"],
        timeout=3.0,
        cache_attr="_cached_prompt_selector",
    ))

    assert result is fake_el
    assert base._cached_prompt_selector == "div[contenteditable='true']"


def test_click_send_handles_stale_first_selector():
    """If first send selector's click fails (stale element), next selector should be used and cached."""
    from core.zendriver_llm_base import ZendriverLLMBase

    base = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    base.send_button_selectors = ["button.send", "button.send2"]
    base._cached_send_selector = "button.send"

    stale_btn = make_element()
    stale_btn.click = AsyncMock(side_effect=Exception("could not find node with given id"))
    stale_btn.apply = AsyncMock(side_effect=Exception("could not find node with given id"))
    good_btn = make_element()

    async def query_side_effect(sel):
        if sel == "button.send":
            return [stale_btn]
        if sel == "button.send2":
            return [good_btn]
        return []

    tab = make_tab(query_selector_all=AsyncMock(side_effect=query_side_effect))
    asyncio.run(base._click_send(tab, make_element()))

    assert base._cached_send_selector == "button.send2"
    assert good_btn.click.called


def test_fill_input_retries_on_stale_element():
    """_fill_input should recover from a stale input element by refinding it."""
    from core.zendriver_llm_base import ZendriverLLMBase

    base = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    first_input = make_element(tag="textarea")
    # The final, unguarded send_keys(text) call is where a stale reference
    # would surface (clear_input()'s own fallback swallows its failures).
    first_input.send_keys = AsyncMock(side_effect=Exception("could not find node with given id"))

    async def apply_side_effect(js):
        if "el.innerText || el.textContent" in js:
            return "hello world"
        return None

    second_input = make_element(tag="textarea", apply_result=apply_side_effect)
    tab = make_tab()

    base._find_interactable_element = AsyncMock(return_value=second_input)

    asyncio.run(base._fill_input(tab, first_input, "hello world"))

    assert second_input.clear_input.called
    assert second_input.send_keys.await_args_list[-1].args[0] == "hello world"


def test_wait_for_send_button_after_media_upload_returns_true_when_button_appears():
    import tempfile

    from core.zendriver_llm_base import ZendriverLLMBase

    base = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    fake_button = make_element()
    tab = make_tab(query_selector_all=AsyncMock(side_effect=[[], [fake_button]]))

    result = asyncio.run(base._wait_for_send_button_after_media_upload(tab, timeout=1.0))

    assert result is True
    assert tab.query_selector_all.await_count == 2


def test_wait_for_send_button_after_media_upload_times_out_when_button_never_appears():
    import tempfile

    from core.zendriver_llm_base import ZendriverLLMBase

    base = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    tab = make_tab()  # query_selector_all() -> [] always

    result = asyncio.run(base._wait_for_send_button_after_media_upload(tab, timeout=0.1))

    assert result is False


def test_wait_for_media_upload_complete_waits_for_selector_presence():
    from core.zendriver_llm_base import ZendriverLLMBase

    base = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    base.media_config = {
        "image": {
            "upload_complete_selectors": ["div.upload-preview"]
        }
    }
    fake_element = make_element()
    tab = make_tab(query_selector_all=AsyncMock(side_effect=[[], [fake_element]]))

    result = asyncio.run(base._wait_for_media_upload_complete(
        type("M", (), {"media_type": "image"})(),
        tab,
        timeout=1.0,
    ))

    assert result is True
    assert tab.query_selector_all.await_count == 2


def test_wait_for_media_upload_complete_waits_for_selector_absence():
    from core.zendriver_llm_base import ZendriverLLMBase

    base = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    base.media_config = {
        "image": {
            "upload_complete_selectors": ["!upload-image-disclaimer-dialog"]
        }
    }
    fake_element = make_element()
    tab = make_tab(query_selector_all=AsyncMock(side_effect=[[fake_element], []]))

    result = asyncio.run(base._wait_for_media_upload_complete(
        type("M", (), {"media_type": "image"})(),
        tab,
        timeout=1.0,
    ))

    assert result is True
    assert tab.query_selector_all.await_count == 2


def test_is_limit_present_detects_limit_warning():
    from core.zendriver_llm_base import ZendriverLLMBase

    base = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    base.limit_selectors = ["div.limit-warning"]

    fake_element = make_element()
    tab = make_tab(query_selector_all=AsyncMock(return_value=[fake_element]))

    assert asyncio.run(base._is_limit_present(tab)) is True


def test_is_limit_present_returns_false_when_no_limit_warning():
    from core.zendriver_llm_base import ZendriverLLMBase

    base = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    base.limit_selectors = ["div.limit-warning"]

    tab = make_tab()  # query_selector_all() -> [] always

    assert asyncio.run(base._is_limit_present(tab)) is False


# ---------------------------------------------------------------------------
# New endpoints: /api/logs/app and updated /stats
# ---------------------------------------------------------------------------


def test_app_logs_endpoint_returns_list():
    """GET /api/logs/app must return a JSON object with an 'entries' list."""
    response = client.get("/api/logs/app")
    assert response.status_code == 200
    data = response.json()
    assert "entries" in data
    assert isinstance(data["entries"], list)


def test_app_logs_since_parameter():
    """Passing since=<large_int> must return only newer entries (or an empty list)."""
    response = client.get("/api/logs/app?since=999999")
    assert response.status_code == 200
    data = response.json()
    assert data["entries"] == []


def test_stats_includes_logged_engines():
    """GET /stats must include a 'logged_engines' list instead of 'latest_logs'."""
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "stats" in data
    assert "logged_engines" in data
    assert isinstance(data["logged_engines"], list)
    assert "latest_logs" not in data


def test_stats_includes_response_time():
    """GET /stats must include response time averages."""
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "response_time" in data
    assert isinstance(data["response_time"], dict)
    assert "global_avg_ms" in data["response_time"]
    assert "per_engine_avg_ms" in data["response_time"]
    assert isinstance(data["response_time"]["per_engine_avg_ms"], dict)


# ---------------------------------------------------------------------------
# OpenAPI schema compliance tests (Pydantic response_model validation)
# ---------------------------------------------------------------------------


def test_openapi_schema_has_chat_completion_response():
    """The OpenAPI schema must document a response body for /v1/chat/completions."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    path = schema["paths"].get("/v1/chat/completions", {})
    post_op = path.get("post", {})
    responses = post_op.get("responses", {})
    assert "200" in responses, "POST /v1/chat/completions must have a 200 response schema"
    content = responses["200"].get("content", {})
    assert "application/json" in content, "Response must be application/json"


def test_openapi_schema_has_models_response():
    """The OpenAPI schema must document a response body for /v1/models."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    path = schema["paths"].get("/v1/models", {})
    get_op = path.get("get", {})
    responses = get_op.get("responses", {})
    assert "200" in responses
    content = responses["200"].get("content", {})
    assert "application/json" in content


def test_chat_completion_response_schema_fields():
    """POST /v1/chat/completions response must contain all required OpenAI-compatible fields."""
    response = client.post(
        "/v1/chat/completions",
        json={"model": "chatgpt", "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert response.status_code == 200
    data = response.json()
    required = {"id", "object", "created", "model", "choices", "usage", "engine", "prompt", "elapsed_ms"}
    assert required <= data.keys(), f"Missing fields: {required - data.keys()}"
    assert data["object"] == "chat.completion"
    assert isinstance(data["choices"], list)
    assert len(data["choices"]) > 0
    choice = data["choices"][0]
    assert "message" in choice
    assert choice["message"]["role"] == "assistant"
    assert isinstance(data["usage"]["total_tokens"], int)


def test_ping_response_schema():
    """GET /api/ping must return {status, service} — validated by PingResponse model."""
    response = client.get("/api/ping")
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) >= {"status", "service"}
    assert isinstance(data["status"], str)
    assert isinstance(data["service"], str)


def test_v1_models_response_schema_fields():
    """GET /v1/models entries must all carry the four required OpenAI model fields."""
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    required_entry_fields = {"id", "object", "created", "owned_by"}
    for entry in data["data"]:
        assert required_entry_fields <= entry.keys(), f"Missing: {required_entry_fields - entry.keys()}"
        assert isinstance(entry["created"], int)


def test_v1_models_include_capabilities():
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    for entry in data["data"]:
        assert "capabilities" in entry
        assert isinstance(entry["capabilities"], dict)


def test_legacy_models_response_schema_fields():
    """GET /models entries must have all OpenAI fields plus the legacy 'name' field."""
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    for entry in data["data"]:
        assert "id" in entry
        assert "object" in entry
        assert "name" in entry


# ---------------------------------------------------------------------------
# Redirect-stall detection tests
# ---------------------------------------------------------------------------


def test_post_send_check_returns_true_when_stop_button_visible():
    """_post_send_check must return True immediately when a stop button becomes visible."""
    import tempfile
    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    engine.stop_selectors = ["button[aria-label*='Stop']"]

    fake_btn = make_element()
    tab = make_tab(query_selector_all=AsyncMock(return_value=[fake_btn]), url="https://example.com")

    result = asyncio.run(engine._post_send_check(tab, timeout=2.0))
    assert result is True


def test_post_send_check_recognizes_mat_icon_stop_selector():
    """_post_send_check must detect a material icon stop indicator."""
    import tempfile
    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    engine.stop_selectors = ["mat-icon[fonticon='stop']"]

    fake_icon = make_element()
    tab = make_tab(query_selector_all=AsyncMock(return_value=[fake_icon]), url="https://example.com")

    result = asyncio.run(engine._post_send_check(tab, timeout=2.0))
    assert result is True


def test_post_send_check_returns_false_on_redirect():
    """_post_send_check must return False when timeout expires and URL has changed."""
    import tempfile

    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    engine.stop_selectors = ["button[aria-label*='Stop']"]
    engine.response_area_selectors = [".assistant-message"]

    # No stop button, no response text
    tab = make_tab(url="https://auth.example.com/login")

    result = asyncio.run(engine._post_send_check(tab, timeout=0.1))
    assert result is False


def test_get_latest_response_text_uses_first_matching_selector():
    """_get_latest_response_text should return text from the first selector that matches."""
    import tempfile

    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    engine.response_area_selectors = ["div.assistant", "div.alternate"]

    async def query_side_effect(sel):
        if sel == "div.assistant":
            return []
        if sel == "div.alternate":
            return [make_element(apply_result="Hello from assistant")]
        return []

    tab = make_tab(query_selector_all=AsyncMock(side_effect=query_side_effect))

    result = asyncio.run(engine._get_latest_response_text(tab))
    assert result == "Hello from assistant"


def test_get_latest_response_text_checks_prior_elements_when_last_is_empty():
    """_get_latest_response_text should use an earlier matching element when the last one is blank."""
    import tempfile

    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    engine.response_area_selectors = ["div.assistant"]

    empty_elem = make_element(apply_result="")
    filled_elem = make_element(apply_result="OK")

    tab = make_tab(query_selector_all=AsyncMock(return_value=[filled_elem, empty_elem]))

    result = asyncio.run(engine._get_latest_response_text(tab))
    assert result == "OK"


def test_get_latest_response_text_js_fallback_when_selectors_fail():
    """_get_latest_response_text should fall back to JS extraction when CSS selectors return nothing."""
    import tempfile

    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    engine.response_area_selectors = ["div.assistant", "div.alternate"]

    tab = make_tab(evaluate=AsyncMock(return_value="JS fallback text"))

    result = asyncio.run(engine._get_latest_response_text(tab))
    assert result == "JS fallback text"

def test_sync_generate_response_retries_on_redirect_stall():
    """The retry loop must retry once on redirect-stall without resetting the driver."""
    import tempfile
    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )

    call_count = 0

    async def fake_once(prompt, media=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("redirect-stall: send not accepted after redirect")
        return "ok response"

    engine._generate_response_once = fake_once
    reset_called = []

    async def fake_reset():
        reset_called.append(True)

    engine._reset_driver = fake_reset

    result = asyncio.run(engine._generate_response_retry_loop("hello"))
    assert result == "ok response"
    assert call_count == 2
    assert reset_called == [], "Driver must NOT be reset on redirect-stall"


def test_sync_generate_response_page_refresh_budget_persists_across_attempts():
    """A perpetually-stuck stop button must not cause an infinite refresh loop.

    Regression: _page_refresh_attempts was reset to 0 at the start of every
    single attempt, so the page-refresh budget replenished on each attempt and
    the paste → send → refresh cycle never terminated. The budget must be
    reset once per request and shared across attempts, so that once it is
    exhausted the engine falls back to a driver reset instead of looping.
    """
    import tempfile
    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    engine._max_page_refresh_attempts = 2

    call_count = 0
    reset_called = []

    async def fake_once(prompt, media=None):
        nonlocal call_count
        call_count += 1
        # Simulate _wait_for_response detecting a stuck stop button: it bumps the
        # refresh counter (up to the budget) then raises page_refresh_required.
        if engine._page_refresh_attempts < engine._max_page_refresh_attempts:
            engine._page_refresh_attempts += 1
        raise RuntimeError(
            "page_refresh_required: stop button stuck at start of "
            "_wait_for_response — page refreshed, retry without driver reset"
        )

    engine._generate_response_once = fake_once

    async def fake_reset():
        reset_called.append(True)
        # A real driver reset also clears the refresh counter; mirror that here.
        engine._page_refresh_attempts = 0
        # After the reset the page is clean, so the next attempt succeeds.
        async def _recovered(prompt, media=None):
            return "recovered"
        engine._generate_response_once = _recovered

    engine._reset_driver = fake_reset

    result = asyncio.run(engine._generate_response_retry_loop("hello"))

    assert result == "recovered"
    # Once the budget is exhausted the engine resets the driver exactly once.
    assert reset_called == [True], "Driver must be reset once budget is exhausted"
    # fake_once is invoked twice: attempt 1 (budget 0→1, retry) and attempt 2
    # (budget 1→2 → exhausted → driver reset). The reset swaps in the recovery
    # stub, so attempt 3 succeeds without incrementing call_count. Because the
    # budget persists across attempts, the loop terminates instead of spinning.
    assert call_count == 2


def test_sync_generate_response_retries_on_response_detection_timeout():
    """The retry loop retries with driver reset when response detection times out."""
    import tempfile
    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )

    call_count = 0

    async def fake_once(prompt, media=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError(
                "selenium_response_detection_timeout: no new response text appeared"
            )
        return "real response"

    engine._generate_response_once = fake_once
    reset_called = []

    async def fake_reset():
        reset_called.append(True)

    engine._reset_driver = fake_reset

    result = asyncio.run(engine._generate_response_retry_loop("hello"))
    assert result == "real response"
    assert call_count == 2
    assert reset_called == [True], "Driver MUST be reset on response detection timeout"


def test_sync_generate_response_once_retries_on_stale_element():
    """_generate_response_once should retry once when a stale element occurs.

    Pre-existing behavior note: the exact trigger point for a "stale element"
    differs from the old Selenium version (see core/zendriver_llm_base.py's
    ``_is_stale_element_error`` docstring — zendriver has no dedicated
    staleness exception type), so this simulates it via ``_click_send``
    raising a "could not find node" error, which is the realistic zendriver
    analogue."""
    import tempfile

    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    engine.driver = make_tab(url="https://example.com")
    engine._initialized = True
    engine._ensure_ready = AsyncMock(return_value=None)
    engine.refresh_login_state = AsyncMock(return_value=True)

    engine._find_interactable_element = AsyncMock(return_value=make_element())
    engine._fill_input = AsyncMock(return_value=None)
    engine._click_accept_buttons = AsyncMock(return_value=None)
    engine._is_captcha_present = AsyncMock(return_value=False)
    engine._is_limit_present = AsyncMock(return_value=False)
    engine._post_send_check = AsyncMock(return_value=True)
    engine._wait_for_response = AsyncMock(return_value="final response")
    engine._is_dead_session = lambda exc: False

    call_count = 0

    async def fake_click_send(tab, input_el):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise Exception("could not find node: stale element")
        return None

    engine._click_send = fake_click_send

    result = asyncio.run(engine._generate_response_once("hello"))
    assert result == "final response"
    assert call_count == 2


def test_wait_for_response_raises_on_detection_timeout(monkeypatch):
    """_wait_for_response raises RuntimeError when no new text is found within timeout."""
    import tempfile
    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    engine.response_area_selectors = ["div.response"]
    engine.stop_selectors = []

    # Tab always returns no elements (no stop buttons, no response elements)
    tab = make_tab(url="https://example.com")

    monkeypatch.setenv("SELENIUM_RESPONSE_MAX_WAIT", "1")
    monkeypatch.setattr("asyncio.sleep", AsyncMock())

    with pytest.raises(RuntimeError, match="selenium_response_detection_timeout"):
        asyncio.run(engine._wait_for_response(tab))


def test_wait_for_response_returns_best_effort_when_first_new_set(monkeypatch):
    """_wait_for_response returns best-effort text when new text appears before max_wait."""
    import tempfile
    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    engine.response_area_selectors = ["div.response"]
    engine.stop_selectors = []

    call_count = 0

    async def query_side_effect(sel):
        nonlocal call_count
        call_count += 1
        if sel == "div.response" and call_count > 2:
            return [make_element(apply_result="new response text")]
        return []

    tab = make_tab(query_selector_all=AsyncMock(side_effect=query_side_effect), url="https://example.com")

    monkeypatch.setenv("SELENIUM_RESPONSE_MAX_WAIT", "1")
    monkeypatch.setattr("asyncio.sleep", AsyncMock())

    result = asyncio.run(engine._wait_for_response(tab))
    # There is new text, so it should be returned (either from main loop or best-effort)
    # The exact return depends on timing, but it should not raise.
    assert result in ("new response text", "")


def test_wait_for_response_silent_freeze_triggers_page_refresh(monkeypatch):
    """_wait_for_response should attempt page refresh on silent freeze and raise page_refresh_required."""
    import tempfile
    from core.zendriver_llm_base import ZendriverLLMBase
    from unittest.mock import patch

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    engine.response_area_selectors = ["div.response"]
    engine.stop_selectors = ["button.stop"]
    engine._max_page_refresh_attempts = 2
    engine._page_refresh_attempts = 0

    # Simulate stop button visible but no response text ever appears
    async def query_side_effect(sel):
        if sel == "button.stop":
            return [make_element()]
        return []

    tab = make_tab(query_selector_all=AsyncMock(side_effect=query_side_effect), url="https://example.com")
    engine._wait_for_page_ready = AsyncMock(return_value=True)

    monkeypatch.setenv("SELENIUM_RESPONSE_MAX_WAIT", "300")
    # Low freeze threshold so the silent-freeze condition triggers quickly once
    # the (advancing) fake clock passes it.
    monkeypatch.setenv("SELENIUM_SILENT_FREEZE_THRESHOLD", "5")

    # Advance the fake clock a little on every time.time() call so the main
    # monitoring loop makes progress and eventually crosses the freeze threshold
    # while staying well under the (large) max_wait deadline.
    fake_time = [1000.0]

    def fake_time_func():
        fake_time[0] += 1.0
        return fake_time[0]

    with patch("time.time", side_effect=fake_time_func), \
         patch("asyncio.sleep", AsyncMock()):
        with pytest.raises(RuntimeError, match="page_refresh_required"):
            asyncio.run(engine._wait_for_response(tab))

    assert engine._page_refresh_attempts == 1
    tab.reload.assert_called_once()
    engine._wait_for_page_ready.assert_called_once_with(tab, timeout=30.0)


def test_wait_for_response_silent_freeze_resets_driver_after_max_refreshes(monkeypatch):
    """_wait_for_response should raise asyncio.TimeoutError after max page refresh attempts."""
    import tempfile
    from core.zendriver_llm_base import ZendriverLLMBase
    from unittest.mock import patch

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    engine.response_area_selectors = ["div.response"]
    engine.stop_selectors = ["button.stop"]
    engine._max_page_refresh_attempts = 1
    engine._page_refresh_attempts = 1  # already at max

    async def query_side_effect(sel):
        if sel == "button.stop":
            return [make_element()]
        return []

    tab = make_tab(query_selector_all=AsyncMock(side_effect=query_side_effect), url="https://example.com")

    monkeypatch.setenv("SELENIUM_RESPONSE_MAX_WAIT", "300")
    monkeypatch.setenv("SELENIUM_SILENT_FREEZE_THRESHOLD", "5")

    # Advance the fake clock on every call so the loop progresses and crosses the
    # freeze threshold while staying under max_wait.
    fake_time = [1000.0]

    def fake_time_func():
        fake_time[0] += 1.0
        return fake_time[0]

    with patch("time.time", side_effect=fake_time_func), \
         patch("asyncio.sleep", AsyncMock()):
        with pytest.raises(asyncio.TimeoutError, match="silent freeze detected"):
            asyncio.run(engine._wait_for_response(tab))

    # reload should NOT be called because we are already at max attempts
    tab.reload.assert_not_called()


def test_wait_for_response_error_indicator_navigates_to_service_home(monkeypatch):
    """Detecting an engine error toast must navigate to the service home.

    When an error indicator (e.g. Gemini "Something went wrong (1076)") is
    detected during response waiting, the watcher should return to the service
    home to start a clean chat (rather than a plain refresh that could restore
    the partial conversation) and raise page_refresh_required so the caller
    resends the full prompt without a full driver reset.
    """
    import tempfile
    from core.zendriver_llm_base import ZendriverLLMBase
    from unittest.mock import patch

    engine = ZendriverLLMBase(
        service_url="https://example.com/home",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    engine.response_area_selectors = ["div.response"]
    # No stop button so the silent-freeze / UI-stuck checks stay inactive and
    # the error-indicator branch is the one that fires.
    engine.stop_selectors = []
    engine.error_indicator_selectors = ["div.error-toast"]
    engine._max_page_refresh_attempts = 2
    engine._page_refresh_attempts = 0
    engine._skip_split_for_next = True

    # No response text ever appears, but an error toast is visible from the
    # first iteration so the error-indicator branch triggers immediately.
    async def query_side_effect(sel):
        if sel == "div.error-toast":
            return [make_element(apply_result="Something went wrong (1076)")]
        return []

    tab = make_tab(
        query_selector_all=AsyncMock(side_effect=query_side_effect),
        url="https://example.com/home",
    )
    engine._wait_for_page_ready = AsyncMock(return_value=True)

    monkeypatch.setenv("SELENIUM_RESPONSE_MAX_WAIT", "300")
    monkeypatch.setenv("SELENIUM_SILENT_FREEZE_THRESHOLD", "5")

    # Advance the fake clock only slightly per call so the error-indicator branch
    # fires before the no-progress freeze grace window elapses.
    fake_time = [1000.0]

    def fake_time_func():
        fake_time[0] += 0.05
        return fake_time[0]

    with patch("time.time", side_effect=fake_time_func), \
         patch("asyncio.sleep", AsyncMock()):
        with pytest.raises(RuntimeError, match="page_refresh_required"):
            asyncio.run(engine._wait_for_response(tab))

    assert engine._page_refresh_attempts == 1
    # Navigated to the service home (not a plain refresh).
    tab.get.assert_called_once_with("https://example.com/home")
    tab.reload.assert_not_called()
    engine._wait_for_page_ready.assert_called_once_with(tab, timeout=30.0)
    # Selector caches invalidated and full-resend forced for the next attempt.
    assert engine._cached_prompt_selector is None
    assert engine._cached_send_selector is None
    assert engine._skip_split_for_next is False


def test_wait_for_response_block_generation_with_stop_button_does_not_refresh(monkeypatch):
    """Block-generation engines must not trigger a page refresh at start.

    Regression: some engines (e.g. Gemini) render the whole answer in one block
    after a thinking delay, keeping the stop button visible the entire time with
    no incremental text. The old start-of-wait pre-check refreshed the page when
    the stop button stayed visible for a few seconds, discarding the in-progress
    response and causing an infinite paste → send → refresh loop.

    _wait_for_response must instead let the response arrive and return it,
    without ever calling tab.reload().
    """
    import tempfile
    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    engine.response_area_selectors = ["div.response"]
    engine.stop_selectors = ["button.stop"]
    engine.accept_button_selectors = []
    engine._click_accept_buttons = AsyncMock()
    engine._is_captcha_present = AsyncMock(return_value=False)
    engine._is_limit_present = AsyncMock(return_value=False)
    # Stop button stays visible throughout the whole block generation.
    engine._stop_button_present = AsyncMock(return_value=True)

    fake_element = make_element()
    # No text for the first few polls (thinking), then the whole block appears
    # and stays stable — mimicking non-incremental generation.
    stats = [(0, 0), (0, 0), (120, 3), (120, 3), (120, 3)]
    call_count = {"n": 0}

    async def fake_get_stats(_element):
        call_count["n"] += 1
        return stats[min(call_count["n"] - 1, len(stats) - 1)]

    engine._find_response_container_element = AsyncMock(return_value=(fake_element, "div.response"))
    engine._get_response_container_stats = fake_get_stats
    engine._extract_response_text_from_element = AsyncMock(return_value="block response")

    tab = make_tab(url="https://example.com")

    monkeypatch.setattr("asyncio.sleep", AsyncMock())
    result = asyncio.run(engine._wait_for_response(tab, max_wait=10))

    assert result == "block response"
    tab.reload.assert_not_called()
    assert engine._page_refresh_attempts == 0


def test_wait_for_response_watcher_stable_container_returns_text(monkeypatch):
    """_wait_for_response should return text after the generic container watcher sees stability."""
    import tempfile
    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    engine.response_area_selectors = ["div.response"]
    engine.accept_button_selectors = []
    engine._click_accept_buttons = AsyncMock()

    fake_element = make_element()
    stats = [(0, 0), (5, 1), (5, 1), (5, 1)]
    call_count = {"n": 0}

    async def fake_get_stats(_element):
        call_count["n"] += 1
        return stats[min(call_count["n"] - 1, len(stats) - 1)]

    engine._find_response_container_element = AsyncMock(return_value=(fake_element, "div.response"))
    engine._get_response_container_stats = fake_get_stats
    engine._extract_response_text_from_element = AsyncMock(return_value="final response")
    engine._is_captcha_present = AsyncMock(return_value=False)
    engine._is_limit_present = AsyncMock(return_value=False)

    tab = make_tab(url="https://example.com")

    monkeypatch.setattr("asyncio.sleep", AsyncMock())
    result = asyncio.run(engine._wait_for_response(tab, max_wait=10))

    assert result == "final response"


def test_wait_for_response_watcher_initial_stable_response_returns_text(monkeypatch):
    """_wait_for_response should return text when the response is already stable on first poll."""
    import tempfile
    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    engine.response_area_selectors = ["div.response"]
    engine.accept_button_selectors = []
    engine._click_accept_buttons = AsyncMock()

    fake_element = make_element()
    stats = [(5, 1), (5, 1), (5, 1)]
    call_count = {"n": 0}

    async def fake_get_stats(_element):
        call_count["n"] += 1
        return stats[min(call_count["n"] - 1, len(stats) - 1)]

    engine._find_response_container_element = AsyncMock(return_value=(fake_element, "div.response"))
    engine._get_response_container_stats = fake_get_stats
    engine._extract_response_text_from_element = AsyncMock(return_value="final response")
    engine._is_captcha_present = AsyncMock(return_value=False)
    engine._is_limit_present = AsyncMock(return_value=False)

    tab = make_tab(url="https://example.com")

    monkeypatch.setattr("asyncio.sleep", AsyncMock())
    result = asyncio.run(engine._wait_for_response(tab, max_wait=10))

    assert result == "final response"


# ---------------------------------------------------------------------------
# Prompt chunking tests
# ---------------------------------------------------------------------------


def test_should_split_prompt_below_limit():
    """_should_split_prompt must return False when the prompt fits within the limit."""
    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 100},
        default_model="default",
    )
    engine._split_prompt_parts = 3
    assert engine._should_split_prompt("x" * 100) is False
    assert engine._should_split_prompt("x" * 99) is False


def test_should_split_prompt_above_limit():
    """_should_split_prompt must return True when the prompt exceeds the limit."""
    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 100},
        default_model="default",
    )
    engine._split_prompt_parts = 3
    assert engine._should_split_prompt("x" * 101) is True


def test_should_split_prompt_disabled_when_parts_le_1():
    """_should_split_prompt must return False when SELENIUM_SPLIT_PROMPT_PARTS <= 1."""
    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 100},
        default_model="default",
    )
    engine._split_prompt_parts = 1
    assert engine._should_split_prompt("x" * 200) is False


def test_split_prompt_into_parts_count_and_coverage():
    """_split_prompt_into_parts must produce exactly n parts that together reconstruct the prompt."""
    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    prompt = "A" * 300
    parts = engine._split_prompt_into_parts(prompt, 3)
    assert len(parts) == 3
    assert "".join(parts) == prompt


def test_split_prompt_into_parts_chunks_within_limit():
    """Each chunk produced must be <= ceil(len/n) characters."""
    import math
    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    prompt = "B" * 301
    n = 3
    parts = engine._split_prompt_into_parts(prompt, n)
    max_chunk = math.ceil(len(prompt) / n)
    for part in parts:
        assert len(part) <= max_chunk


def test_split_prompt_keeps_tail_marker_in_final_chunk():
    """When the protected tail marker is present, everything after it must land
    intact in the LAST chunk and the marker itself must be stripped."""
    from core.zendriver_llm_base import ZendriverLLMBase
    from core.agent_protocol import AGENT_TAIL_MARKER

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    head = "H" * 300
    tail = "[REMINDER] call tool_x now"
    prompt = head + "\n\n" + AGENT_TAIL_MARKER + "\n" + tail
    parts = engine._split_prompt_into_parts(prompt, 3)
    # The tail must be the final chunk, whole, with no marker leaking through.
    assert parts[-1] == tail
    assert AGENT_TAIL_MARKER not in "".join(parts)
    # The head is spread across the earlier chunks.
    assert "".join(parts[:-1]).startswith("H")


def test_execute_chunked_send_invokes_driver_n_times():
    """_execute_chunked_send must call _fill_input and _click_send once per chunk."""
    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 100},
        default_model="default",
    )
    engine._split_prompt_parts = 3

    fake_el = make_element()
    fill_calls: list[str] = []
    click_calls: list[int] = []
    response_counter = [0]

    async def fake_find_interactable(*args, **kwargs):
        return fake_el

    async def fake_fill(driver, element, text):
        fill_calls.append(text)

    async def fake_click(driver, element):
        click_calls.append(1)

    async def fake_post_send_check(driver, **kwargs):
        return True

    async def fake_wait_response(driver, **kwargs):
        response_counter[0] += 1
        return f"OK part {response_counter[0]}"

    engine._find_interactable_element = fake_find_interactable
    engine._fill_input = fake_fill
    engine._click_send = fake_click
    engine._post_send_check = fake_post_send_check
    engine._wait_for_response = fake_wait_response
    # With pre-fill optimisation, _wait_for_send_ready replaces _wait_for_response
    # for intermediate chunks — mock it to return True immediately.
    engine._wait_for_send_ready = AsyncMock(return_value=True)
    engine._click_accept_buttons = AsyncMock()
    engine._stop_button_present = AsyncMock(return_value=False)

    # 301-char prompt with limit=100 → ceil(301/100)=4 parts min, but env_max=3
    # So n = min(3, max(ceil(301/100), 2)) = min(3, 4) = 3
    prompt = "Z" * 301
    result = asyncio.run(engine._execute_chunked_send(prompt, make_tab()))

    assert len(fill_calls) == 3
    assert len(click_calls) == 3
    # _wait_for_response is called only once (final chunk), not once per chunk.
    assert response_counter[0] == 1
    assert result  # non-empty response returned
    # The flag must be reset after completion
    assert engine._skip_split_for_next is False


def test_execute_chunked_send_intermediate_headers():
    """Intermediate chunks must carry the [PART {i}/{n}] header."""
    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 100},
        default_model="default",
    )
    engine._split_prompt_parts = 3

    fake_el = make_element()
    fill_calls: list[str] = []

    async def fake_find_interactable(*a, **kw):
        return fake_el

    async def fake_fill(d, e, text):
        fill_calls.append(text)

    engine._find_interactable_element = fake_find_interactable
    engine._fill_input = fake_fill
    engine._click_send = AsyncMock()
    engine._post_send_check = AsyncMock(return_value=True)
    engine._wait_for_response = AsyncMock(return_value="OK")
    engine._wait_for_send_ready = AsyncMock(return_value=True)
    engine._click_accept_buttons = AsyncMock()
    engine._stop_button_present = AsyncMock(return_value=False)

    prompt = "X" * 301
    asyncio.run(engine._execute_chunked_send(prompt, make_tab()))

    # Intermediate chunks (all but the last) must carry the header
    n = len(fill_calls)
    for i, text in enumerate(fill_calls[:-1], start=1):
        assert f"[PART {i}/{n}]" in text

    # The final chunk must NOT carry the header
    assert "[PART " not in fill_calls[-1]


def test_execute_chunked_send_prefill_before_wait():
    # Optimization was disabled intentionally
    pass


def test_skip_split_flag_prevents_recursion():
    """When _skip_split_for_next is True, _should_split_prompt is bypassed."""
    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 100},
        default_model="default",
    )
    engine._split_prompt_parts = 3
    engine._skip_split_for_next = True
    # Even though prompt is way over limit, _should_split_prompt returns True
    # but the flag prevents _execute_chunked_send from being called again.
    assert engine._should_split_prompt("X" * 500) is True
    # Verify the guard works inside _sync_generate_response_once by inspecting the
    # branch condition: not flag AND should_split → False when flag is True.
    assert not (not engine._skip_split_for_next and engine._should_split_prompt("X" * 500))


# ---------------------------------------------------------------------------
# FIFO queue + no-browser-probe regression tests
# ---------------------------------------------------------------------------


def test_models_no_browser_probe():
    """/models must not open any browsers when engines are not yet instantiated."""
    mgr = EngineManager.get()
    mgr.engines.clear()

    response = client.get("/models")
    assert response.status_code == 200

    # No engine instance should have been created
    assert mgr.engines == {}, "Engines should not be instantiated during /models probe"

    data = response.json()
    assert data["object"] == "list"
    for entry in data["data"]:
        assert "limits" in entry, "limits must be present even without a live browser"
        assert "supported_models" in entry, "supported_models must be present even without a live browser"
        assert isinstance(entry["limits"]["max_prompt_chars"], int)
        assert isinstance(entry["supported_models"], list)


def test_models_uses_live_data_if_engine_running():
    """/models must use live engine data when the engine browser is already running."""
    mgr = EngineManager.get()
    # DummyEngine is already in mgr.engines from the fixture
    assert "chatgpt" in mgr.engines

    response = client.get("/models")
    assert response.status_code == 200

    data = response.json()
    chatgpt_entry = next(e for e in data["data"] if e["id"] == "chatgpt")
    # DummyEngine.get_interface_limits() returns max_prompt_chars=1234
    assert chatgpt_entry["limits"]["max_prompt_chars"] == 1234


def test_max_workers_in_descriptor():
    """EngineDescriptor must expose max_workers (default 1) via to_dict()."""
    from core.engine_manager import EngineDescriptor

    desc = EngineDescriptor(
        name="my-engine",
        aliases=["my-engine"],
        display_name="My Engine",
        service_url="https://example.com",
        models={"default": 10000},
        default_model="default",
        source="json",
        source_path="<test>",
    )
    assert desc.max_workers == 1
    d = desc.to_dict()
    assert "max_workers" in d
    assert d["max_workers"] == 1

    desc2 = EngineDescriptor(
        name="my-engine",
        aliases=["my-engine"],
        display_name="My Engine",
        service_url="https://example.com",
        models={"default": 10000},
        default_model="default",
        source="json",
        source_path="<test>",
        max_workers=4,
    )
    assert desc2.to_dict()["max_workers"] == 4


def test_queue_fifo_serializes_requests():
    """Concurrent enqueue() calls on the same engine must be serialised FIFO."""
    from core.engine_manager import EngineManager

    execution_log: list[str] = []

    class OrderedDummyEngine:
        def get_current_model(self):
            return "default"

        async def generate_response(self, prompt: str) -> str:
            # Tiny yield so the event loop can interleave — but should NOT
            # because the queue serialises
            await asyncio.sleep(0)
            execution_log.append(prompt)
            return f"response-{prompt}"

    async def _run():
        mgr = EngineManager.get()
        mgr.engines["chatgpt"] = OrderedDummyEngine()
        # Clear queue state from previous tests without awaiting tasks that
        # belong to a different event loop (created by the TestClient).
        mgr._queue_workers.clear()
        mgr._job_queues.clear()
        # Submit three tasks concurrently
        results = await asyncio.gather(
            mgr.enqueue("chatgpt", "A"),
            mgr.enqueue("chatgpt", "B"),
            mgr.enqueue("chatgpt", "C"),
        )
        return results

    results = asyncio.run(_run())

    assert [r.text for r in results] == ["response-A", "response-B", "response-C"]
    assert execution_log == ["A", "B", "C"], f"FIFO order violated: {execution_log}"


# ---------------------------------------------------------------------------
# Fallback: send-button-based generation detection
# ---------------------------------------------------------------------------


def test_send_button_present_returns_true_when_visible():
    """_send_button_present must return True when a send button element is visible."""
    import tempfile

    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    engine.send_button_selectors = ["button[aria-label='Send message']"]

    tab = make_tab(query_selector_all=AsyncMock(return_value=[make_element()]))

    assert asyncio.run(engine._send_button_present(tab)) is True


def test_send_button_present_returns_false_when_absent():
    """_send_button_present must return False when no enabled send button is found."""
    import tempfile

    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    engine.send_button_selectors = ["button[aria-label='Send message']"]

    tab = make_tab()  # query_selector_all() -> [] always

    assert asyncio.run(engine._send_button_present(tab)) is False


def test_post_send_check_fallback_when_send_button_absent():
    """_post_send_check must return True (generation in progress) when:
    - no stop button is found
    - no new response text appeared
    - but the send button has disappeared (generation accepted by LLM)
    """
    import tempfile

    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    engine.stop_selectors = []          # primary check always skipped
    engine.send_button_selectors = ["button[aria-label='Send message']"]
    engine.response_area_selectors = []

    # No elements found for any selector → stop absent, text empty, send absent
    tab = make_tab(url="https://example.com")

    result = asyncio.run(engine._post_send_check(tab, timeout=1.0))
    # Send button and response area both absent → still no signal either
    # way, so the plain "URL looks ok, assume slow model" timeout path wins.
    assert result is True


def test_post_send_check_fallback_requires_response_area():
    """_post_send_check must recognize generation only when the response area exists."""
    import tempfile

    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    engine.stop_selectors = []
    engine.send_button_selectors = ["button.send"]
    engine.response_area_selectors = [".response"]

    async def query_side_effect(sel):
        if sel == "button.send":
            return []
        if sel == ".response":
            return [make_element()]
        return []

    tab = make_tab(query_selector_all=AsyncMock(side_effect=query_side_effect), url="https://example.com")

    assert asyncio.run(engine._post_send_check(tab, timeout=0.5)) is True


def test_wait_for_response_initial_phase_fallback_send_button_absent():
    """_wait_for_response must exit the initial wait when the send button disappears,
    signalling that the LLM has accepted the prompt and started generating.
    After that, once the response text is stable for 1 s (unchanged) and is
    different from baseline, the fallback logic must return the response
    without requiring any send-button check.
    """
    import tempfile
    from unittest.mock import patch

    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    engine.stop_selectors = []          # no stop selectors — fallback only
    engine.send_button_selectors = ["button[aria-label='Send message']"]
    engine.response_area_selectors = [".response"]
    engine.accept_button_selectors = []

    response_text = "Fallback response from LLM."

    # _get_latest_response_text: first call (baseline) = "", then stable response
    text_calls = [0]

    def _get_text(_tab: object) -> str:
        text_calls[0] += 1
        return "" if text_calls[0] == 1 else response_text

    # _send_button_present is used only in initial-phase fallback, not in Phase 2
    send_calls = [0]

    def _send_present(_tab: object) -> bool:
        send_calls[0] += 1
        # First two checks: absent (generating); from third onwards: present (done)
        return send_calls[0] > 2

    tab = make_tab(url="https://example.com")

    # patch.object auto-detects these are `async def` on the real class and
    # wraps side_effect in an AsyncMock, so a plain (non-async) side_effect
    # function works unchanged here.
    with (
        patch.object(engine, "_get_latest_response_text", side_effect=_get_text),
        patch.object(engine, "_send_button_present", side_effect=_send_present),
        patch("asyncio.sleep", AsyncMock()),
    ):
        result = asyncio.run(engine._wait_for_response(tab, max_wait=10))

    assert result == response_text


# ---------------------------------------------------------------------------
# Cookie persistence tests
#
# Persistence is intentionally a no-op here: it relies entirely on Chrome's
# native --user-data-dir profile, like a normal desktop browser. A CDP-based
# snapshot/restore layer was tried and removed -- replaying an old value of a
# short-lived, rotating anti-replay cookie (e.g. Google's __Secure-1PSIDTS)
# looked like session hijacking to the site's own security systems and
# triggered the very logout it was meant to prevent (observed live against
# Gemini). See the note above _save_cookies in core/zendriver_llm_base.py.
# ---------------------------------------------------------------------------


def test_save_cookies_is_a_noop(tmp_path):
    """No custom persistence — Chrome's own profile is the only mechanism."""
    from unittest.mock import MagicMock

    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=str(tmp_path),
    )
    engine.driver = MagicMock()

    engine._save_cookies()

    engine.driver.execute_cdp_cmd.assert_not_called()
    engine.driver.get_cookies.assert_not_called()
    assert list(tmp_path.iterdir()) == []


def test_restore_cookies_is_a_noop_but_marks_restored(tmp_path):
    from unittest.mock import MagicMock

    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=str(tmp_path),
    )
    engine.driver = MagicMock()

    engine._restore_cookies()

    engine.driver.execute_cdp_cmd.assert_not_called()
    engine.driver.add_cookie.assert_not_called()
    assert engine._cookies_restored is True


def test_maybe_save_cookies_is_a_noop_regardless_of_interval(tmp_path):
    from unittest.mock import MagicMock

    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=str(tmp_path),
    )
    engine.driver = MagicMock()
    engine._last_cookie_save = 0  # long ago -- would have tripped the old throttle

    engine._maybe_save_cookies()

    engine.driver.execute_cdp_cmd.assert_not_called()


def test_cookie_path_uses_engine_name(tmp_path):
    """_cookie_path includes ENGINE_NAME in the filename."""
    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=str(tmp_path),
    )
    engine.ENGINE_NAME = "my-engine"
    assert engine._cookie_path().endswith("cookies_my-engine.json")


def test_cookie_path_default_without_engine_name(tmp_path):
    """_cookie_path falls back to 'default' when ENGINE_NAME is not set."""
    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=str(tmp_path),
    )
    assert engine._cookie_path().endswith("cookies_default.json")


def test_remove_profile_lock_files_removes_dangling_symlink(tmp_path):
    """Regression: SingletonLock is a symlink to "<hostname>-<pid>", a target
    that never exists as a real path. os.path.exists() follows symlinks and
    returns False for a dangling one, so a prior version of this function
    guarded the removal with `if os.path.exists(path): os.remove(path)` and
    silently never removed it. Observed live: a lock file left in the
    (persistent-volume) profile dir by a container's previous instance --
    a different hostname -- blocked every browser relaunch after a rebuild
    with "profile appears to be in use by another Chromium process ... on
    another computer" until the container was manually cleaned up.
    """
    import core.zendriver_llm_base as zlb

    lock = tmp_path / "SingletonLock"
    lock.symlink_to("some-other-hostname-12345")  # dangling on purpose
    assert not lock.exists()  # exists() correctly reports False here
    assert lock.is_symlink()  # but the link itself is real and must be removed

    zlb._remove_profile_lock_files(str(tmp_path))

    assert not lock.is_symlink()


def test_remove_profile_lock_files_tolerates_missing_files(tmp_path):
    """No lock files present at all must not raise."""
    import core.zendriver_llm_base as zlb

    zlb._remove_profile_lock_files(str(tmp_path))  # must not raise


def test_protected_pids_empty_when_no_active_browser(monkeypatch):
    monkeypatch.setattr(
        "core.zendriver_llm_base.get_shared_browser_pid", lambda: None
    )
    mgr = EngineManager.get()
    assert mgr._protected_pids() == set()


def test_protected_pids_includes_root_and_descendants(monkeypatch):
    """The active browser's whole process tree must be protected, not just
    its root PID -- a long-open tab's renderer process has its own,
    independently old start time and would otherwise still look orphaned."""
    import core.engine_manager as em

    monkeypatch.setattr(
        "core.zendriver_llm_base.get_shared_browser_pid", lambda: 100
    )

    children = {100: [200, 201], 200: [300], 201: [], 300: []}

    def fake_run(cmd, **kwargs):
        result = MagicMock()
        if cmd[:2] == ["pgrep", "-P"]:
            parent = int(cmd[2])
            result.stdout = "\n".join(str(c) for c in children.get(parent, []))
        else:
            result.stdout = ""
        return result

    monkeypatch.setattr(em.subprocess, "run", fake_run)

    mgr = EngineManager.get()
    assert mgr._protected_pids() == {100, 200, 201, 300}


def test_cleanup_orphans_never_kills_protected_pid(monkeypatch):
    """Regression: _cleanup_orphans used to kill *any* chromium process older
    than SELENIUM_ORPHAN_AGE_THRESHOLD, with no notion of "in use" -- masked
    for a long time because the browser used to be recycled every ~300s by
    the (since-raised) session-hard-timeout default, always well under the
    600s orphan threshold. Once a long-lived shared browser became the
    normal state, this started SIGKILLing the live browser out from under
    in-flight requests every 5 minutes (observed live as an immediate
    "no close frame received or sent" CDP-transport error)."""
    import io
    import builtins

    import core.engine_manager as em

    mgr = EngineManager.get()
    monkeypatch.setattr(mgr, "_protected_pids", lambda: {4242})
    monkeypatch.setenv("SELENIUM_ORPHAN_AGE_THRESHOLD", "600")

    run_calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        run_calls.append(cmd)
        result = MagicMock()
        if cmd[:2] == ["pgrep", "-f"]:
            result.stdout = "4242\n9999\n"
        else:
            result.stdout = ""
        return result

    monkeypatch.setattr(em.subprocess, "run", fake_run)

    real_open = builtins.open

    def fake_open(path, *a, **kw):
        if path == "/proc/uptime":
            return io.StringIO("100000.0 90000.0\n")
        if isinstance(path, str) and path.startswith("/proc/") and path.endswith("/stat"):
            fields = ["x"] * 22
            fields[21] = "0"  # starttime=0 ticks -> looks maximally old
            return io.StringIO(" ".join(fields))
        return real_open(path, *a, **kw)

    monkeypatch.setattr(builtins, "open", fake_open)

    mgr._cleanup_orphans()

    kill_targets = [c[-1] for c in run_calls if c and c[0] == "kill"]
    assert "4242" not in kill_targets
    assert "9999" in kill_targets


def test_is_dead_session_recognises_closed_cdp_websocket(tmp_path):
    """Regression: zendriver's CDP transport raises "no close frame received
    or sent" (via the `websockets` library) when the browser process itself
    died and the underlying TCP connection dropped without a proper WS close
    handshake. This message wasn't in the dead-session marker list, so every
    retry kept hitting the same dead connection and failed in ~1ms instead of
    the browser ever getting recreated -- observed live."""
    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=str(tmp_path),
    )
    assert engine._is_dead_session(Exception("no close frame received or sent"))
    assert not engine._is_dead_session(Exception("could not find node: stale element"))


def test_generate_response_serializes_across_engines(tmp_path):
    """Regression: EngineManager gives each engine its own FIFO queue and
    worker task, which only serialises requests to the *same* engine --
    nothing stopped two different engines' workers from driving the one
    shared browser tab at the same time. Observed live: switching from a
    still-running Copilot request to ChatGPT mid-flight let both coroutines
    type/click/navigate the same tab concurrently, corrupting both and
    eventually forcing a nuclear browser reset. generate_response on two
    different engine instances must never interleave."""
    from core.zendriver_llm_base import ZendriverLLMBase

    engine_a = ZendriverLLMBase(
        service_url="https://a.example.com", model_limits_map={"default": 1000},
        default_model="default", profile_dir=str(tmp_path / "a"),
    )
    engine_b = ZendriverLLMBase(
        service_url="https://b.example.com", model_limits_map={"default": 1000},
        default_model="default", profile_dir=str(tmp_path / "b"),
    )

    events: list[str] = []

    async def fake_retry_loop_a(prompt, media):
        events.append("a-start")
        await asyncio.sleep(0.05)  # long enough that b would interleave if unlocked
        events.append("a-end")
        return "a-response"

    async def fake_retry_loop_b(prompt, media):
        events.append("b-start")
        await asyncio.sleep(0.01)
        events.append("b-end")
        return "b-response"

    engine_a._generate_response_retry_loop = fake_retry_loop_a
    engine_b._generate_response_retry_loop = fake_retry_loop_b

    async def run_both():
        return await asyncio.gather(
            engine_a.generate_response("hello", timeout=5),
            engine_b.generate_response("hi", timeout=5),
        )

    results = asyncio.run(run_both())

    assert results == ["a-response", "b-response"]
    assert events in (
        ["a-start", "a-end", "b-start", "b-end"],
        ["b-start", "b-end", "a-start", "a-end"],
    )


def test_build_options_includes_restore_session():
    """_chromium_args include --restore-last-session, and
    _ensure_profile_preferences writes the "clean exit" Preferences keys.

    Selenium's ``prefs`` experimental option (``profile.exit_type``,
    ``profile.exited_cleanly``) has no zendriver Config equivalent, so that
    part is now a separate step performed by writing the profile's
    Preferences JSON file directly — see core/zendriver_llm_base.py.
    """
    import json
    import os
    import tempfile

    from core.zendriver_llm_base import ZendriverLLMBase

    profile_dir = tempfile.mkdtemp()
    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=profile_dir,
    )
    args = engine._chromium_args(9222)
    assert "--restore-last-session" in args

    engine._ensure_profile_preferences()
    prefs_path = os.path.join(profile_dir, "Default", "Preferences")
    with open(prefs_path, encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["profile"]["exit_type"] == "Normal"
    assert data["profile"]["exited_cleanly"] is True
    # Browser-level sign-in stays off: a website login must be plain cookies.
    assert data["signin"]["allowed"] is False


def test_sync_generate_response_dynamic_chunking_retry():
    """Verify that the retry loop increments _split_prompt_parts on chunking failure."""
    from core.zendriver_llm_base import ZendriverLLMBase
    from unittest.mock import patch

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 100},
        default_model="default",
    )
    engine._split_prompt_parts = 2

    # Mocking _generate_response_once to fail with a chunking message on first call
    # and succeed on the second.
    call_count = 0

    async def mock_once(prompt, media=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("Send button did not become ready (UI freeze simulation)")
        return "dynamic result"

    engine._generate_response_once = mock_once
    engine._reset_driver = AsyncMock()

    prompt = "A" * 200  # Should trigger splitting

    with patch.object(engine, "_should_split_prompt", return_value=True):
        result = asyncio.run(engine._generate_response_retry_loop(prompt))

    assert result == "dynamic result"
    assert engine._split_prompt_parts == 3
    assert engine._reset_driver.call_count == 1


# ---------------------------------------------------------------------------
# Agentic (tool-calling) lane — /v1/chat/completions with tools
# ---------------------------------------------------------------------------


_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def test_agent_mode_returns_tool_calls():
    """A request with `tools` must yield an OpenAI tool_calls response."""
    engine = EngineManager.get().engines["chatgpt"]
    engine.next_response = (
        '```json\n{"tool_calls": [{"name": "get_weather", '
        '"arguments": {"city": "Rome"}}]}\n```'
    )
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "chatgpt",
            "messages": [{"role": "user", "content": "Weather in Rome?"}],
            "tools": [_WEATHER_TOOL],
        },
    )
    assert response.status_code == 200
    data = response.json()
    choice = data["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    tool_calls = choice["message"]["tool_calls"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "get_weather"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"city": "Rome"}
    # The engine must have been called in agent mode.
    assert engine.last_agent_mode is True
    # The harness must have been injected into the prompt.
    assert "[AGENT MODE" in engine.last_prompt
    # The trailing in-character reminder must also be appended on every turn.
    assert "[REMINDER]" in engine.last_prompt


def test_agent_mode_final_content_answer():
    """A JSON {content:...} reply must map to a normal stop response."""
    engine = EngineManager.get().engines["chatgpt"]
    engine.next_response = '```json\n{"content": "It is sunny."}\n```'
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "chatgpt",
            "messages": [{"role": "user", "content": "Weather?"}],
            "tools": [_WEATHER_TOOL],
        },
    )
    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["content"] == "It is sunny."


def test_agent_mode_reformulation_retry():
    """An unparseable first reply must trigger a bounded reformulation retry."""
    engine = EngineManager.get().engines["chatgpt"]
    # First reply is prose (unparseable), second is valid JSON.
    engine.next_response = [
        "Sure, I would call get_weather for Rome.",
        '```json\n{"tool_calls": [{"name": "get_weather", '
        '"arguments": {"city": "Rome"}}]}\n```',
    ]
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "chatgpt",
            "messages": [{"role": "user", "content": "Weather in Rome?"}],
            "tools": [_WEATHER_TOOL],
        },
    )
    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    # Both the initial reply and the reformulation prompt were consumed.
    assert engine.next_response == []


def test_agent_mode_reformulation_gives_up_gracefully():
    """After max retries with unparseable output, degrade to raw content."""
    engine = EngineManager.get().engines["chatgpt"]
    engine.next_response = "I cannot produce JSON, sorry."
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "chatgpt",
            "messages": [{"role": "user", "content": "Weather?"}],
            "tools": [_WEATHER_TOOL],
        },
    )
    assert response.status_code == 200
    choice = response.json()["choices"][0]
    # No empty response: raw text is returned as content.
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["content"] == "I cannot produce JSON, sorry."


def _collect_stream_chunks(resp):
    """Parse an SSE agent stream into its chat.completion.chunk objects."""
    chunks = []
    for line in resp.iter_lines():
        if not line.startswith("data:"):
            continue
        body = line.removeprefix("data:").strip()
        if body == "[DONE]":
            continue
        chunks.append(json.loads(body))
    return chunks


def test_agent_mode_streaming_emits_tool_calls():
    """stream=True in agent mode must emit structured tool_calls, not raw JSON."""
    engine = EngineManager.get().engines["chatgpt"]
    engine.next_response = (
        '```json\n{"tool_calls": [{"name": "get_weather", '
        '"arguments": {"city": "Rome"}}]}\n```'
    )
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "chatgpt",
            "messages": [{"role": "user", "content": "Weather in Rome?"}],
            "tools": [_WEATHER_TOOL],
            "stream": True,
        },
    ) as resp:
        assert resp.status_code == 200
        chunks = _collect_stream_chunks(resp)
    # A tool_calls delta and a tool_calls finish_reason must be present.
    tool_deltas = [
        c for c in chunks if c["choices"][0]["delta"].get("tool_calls")
    ]
    assert len(tool_deltas) == 1
    tc = tool_deltas[0]["choices"][0]["delta"]["tool_calls"][0]
    assert tc["function"]["name"] == "get_weather"
    finish_reasons = [c["choices"][0]["finish_reason"] for c in chunks]
    assert "tool_calls" in finish_reasons


def test_agent_mode_streaming_reformulates_content_with_tools():
    """stream=True: a first {content:...} promise while tools exist must be
    reformulated (same as the non-streaming path) into a real tool call."""
    engine = EngineManager.get().engines["chatgpt"]
    engine.next_response = [
        '```json\n{"content": "Procedo ad aggiornare la web UI."}\n```',
        '```json\n{"tool_calls": [{"name": "get_weather", '
        '"arguments": {"city": "Rome"}}]}\n```',
    ]
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "chatgpt",
            "messages": [{"role": "user", "content": "Make the UI purple."}],
            "tools": [_WEATHER_TOOL],
            "stream": True,
        },
    ) as resp:
        assert resp.status_code == 200
        chunks = _collect_stream_chunks(resp)
    # The promise was reformulated: both scripted replies were consumed and the
    # final stream carries a tool call, not the textual promise.
    assert engine.next_response == []
    tool_deltas = [
        c for c in chunks if c["choices"][0]["delta"].get("tool_calls")
    ]
    assert len(tool_deltas) == 1


def test_non_agent_request_unchanged():
    """A plain chat request (no tools) must NOT enter agent mode."""
    engine = EngineManager.get().engines["chatgpt"]
    engine.next_response = "dummy response"
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "chatgpt",
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )
    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["content"] == "dummy response"
    assert engine.last_agent_mode is False
    assert "[AGENT MODE]" not in engine.last_prompt


def test_agent_mode_empty_tools_list_is_not_agentic():
    """An empty tools list must not trigger agent mode (no harness injected)."""
    engine = EngineManager.get().engines["chatgpt"]
    engine.next_response = "dummy response"
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "chatgpt",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [],
        },
    )
    assert response.status_code == 200
    assert engine.last_agent_mode is False


def test_agent_mode_response_format_triggers_agent():
    """A response_format json_object must trigger agent mode without tools."""
    engine = EngineManager.get().engines["chatgpt"]
    engine.next_response = '```json\n{"content": "{\\"ok\\": true}"}\n```'
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "chatgpt",
            "messages": [{"role": "user", "content": "Give me JSON"}],
            "response_format": {"type": "json_object"},
        },
    )
    assert response.status_code == 200
    assert engine.last_agent_mode is True



def test_default_engine_survives_manager_restart():
    """A default chosen through the API must still be set after a restart.

    Regression: ``default_engine`` lived only in memory, so every boot fell back
    to whichever engine sorts first in ``engines/`` and silently discarded the
    operator's choice.
    """
    response = client.post("/api/engines/default", json={"engine": "gemini"})
    assert response.status_code == 200
    assert response.json()["default_engine"] == "gemini"

    original = EngineManager._instance
    try:
        # Simulate a process restart: drop the singleton and build a fresh one.
        EngineManager._instance = None
        assert EngineManager.get().get_default_engine() == "gemini"
    finally:
        EngineManager._instance = original


def test_unknown_model_logs_engine_fallback(caplog):
    """An unresolvable model must not be re-routed to another provider silently.

    Regression: a model name the manager could not resolve fell through to the
    default engine with no log line, so the request was answered by a different
    provider than the caller asked for.
    """
    import logging

    client.post("/api/engines/default", json={"engine": "gemini"})

    with caplog.at_level(logging.WARNING, logger="selenium-llm-api"):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gemini-2.5-flash",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

    assert response.status_code == 200
    assert response.json()["engine"] == "gemini"
    assert "routing to default engine" in caplog.text
    assert "gemini-2.5-flash" in caplog.text


def test_unlogged_prompt_failure_raises(monkeypatch):
    """A failed unlogged prompt must raise instead of returning the error text.

    Regression: the error was handed back as if it were the model's reply, so it
    was stored as a successful prompt and forwarded downstream as content.
    """
    import tempfile

    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    tab = make_tab(url="https://example.com")
    engine.driver = tab
    engine._initialized = True

    # Take the fast path so no real browser is ever started.
    engine._ensure_ready = AsyncMock(return_value=None)
    engine.refresh_login_state = AsyncMock(return_value=False)
    engine._is_dead_session = lambda exc: False
    engine._find_interactable_element = AsyncMock(return_value=make_element())
    engine._click_accept_buttons = AsyncMock(return_value=None)
    # A default mock tab answers every lookup falsily/emptily, which is what
    # keeps the captcha / usage-limit short circuits from tripping before the
    # prompt is typed.
    engine._is_captcha_present = AsyncMock(return_value=False)
    engine._is_limit_present = AsyncMock(return_value=False)

    async def _fail_fill_input(tab, element, prompt):
        raise RuntimeError(
            "[zendriver] fill_input verification failed: prompt content did not "
            "match expected text"
        )

    engine._fill_input = _fail_fill_input

    with pytest.raises(RuntimeError, match="Unlogged session"):
        asyncio.run(engine._generate_response_once("hello"))


def _stale_guard_engine(monkeypatch, stats_sequence, screen_text="OLD ANSWER"):
    """Build an engine whose page always shows ``screen_text``.

    ``stats_sequence`` drives ``_get_response_container_stats`` so a test can
    decide whether the watcher observes generation activity or a page that never
    moves. ``asyncio.sleep`` is neutralised so the poll loop runs instantly.
    """
    import tempfile

    from core.zendriver_llm_base import ZendriverLLMBase

    monkeypatch.setattr("asyncio.sleep", AsyncMock())

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    engine.response_area_selectors = ["div.response"]
    engine.stop_selectors = []

    container = make_element()
    stats = iter(stats_sequence)
    last_stat = stats_sequence[-1]

    async def _next_stats(_element):
        nonlocal last_stat
        last_stat = next(stats, last_stat)
        return last_stat

    engine._click_accept_buttons = AsyncMock()
    engine._stop_button_present = AsyncMock(return_value=False)
    # A default mock tab answers every lookup emptily, which would otherwise
    # trip the captcha / usage-limit short circuits inside the wait loop.
    engine._is_captcha_present = AsyncMock(return_value=False)
    engine._is_limit_present = AsyncMock(return_value=False)
    engine._log_response_container_diagnostics = lambda *a, **kw: None
    engine._find_response_container_element = AsyncMock(return_value=(container, "div.response"))
    engine._get_response_container_stats = _next_stats

    async def _get_latest(_tab):
        return screen_text

    async def _extract(_element):
        return screen_text

    engine._get_latest_response_text = _get_latest
    engine._extract_response_text_from_element = _extract
    return engine


def test_wait_for_response_rejects_stale_previous_answer(monkeypatch):
    """A stable page that never changed must not yield the previous answer.

    Regression: detection only asked "did the text stop changing?". A leftover
    answer from the previous turn is perfectly stable, so it was returned as if
    freshly generated -- the caller then received the prior turn's reply.
    """
    # Metrics never move: nothing was ever generated on the page.
    engine = _stale_guard_engine(monkeypatch, [(10, 1)])

    with pytest.raises(RuntimeError, match="Stale response"):
        asyncio.run(engine._wait_for_response(make_tab(), pre_send_text="OLD ANSWER"))


def test_wait_for_response_allows_identical_answer_after_real_generation(
    monkeypatch,
):
    """An identical answer that actually streamed in must still be returned.

    The stale guard keys on "no generation activity at all", so a model that
    legitimately repeats itself is not mistaken for a stale page.
    """
    # Metrics move first (generation observed), then settle.
    engine = _stale_guard_engine(monkeypatch, [(3, 1), (7, 1), (10, 1), (10, 1)])

    assert (
        asyncio.run(engine._wait_for_response(make_tab(), pre_send_text="OLD ANSWER"))
        == "OLD ANSWER"
    )


def test_click_accept_buttons_handles_multi_step_consent(monkeypatch):
    """A consent banner that must be expanded before it can be dismissed.

    Regression: the helper returned after the first successful click, so the
    button revealed by that click was never pressed and the banner kept
    blocking the page — the prompt was sent into a page that never generated,
    leaving the previous answer on screen.
    """
    import tempfile

    from core.zendriver_llm_base import ZendriverLLMBase

    monkeypatch.setattr("asyncio.sleep", AsyncMock())

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    engine.accept_button_selectors = ["button.expand", "button.dismiss"]

    clicked: list[str] = []
    expanded = {"value": False}

    def _element(name):
        el = make_element()

        async def _click():
            clicked.append(name)
            if name == "expand":
                expanded["value"] = True

        el.click = AsyncMock(side_effect=_click)
        return el

    async def _query_side_effect(sel):
        if sel == "button.expand":
            return [_element("expand")]
        # The dismiss button only exists once the banner has been expanded.
        if sel == "button.dismiss" and expanded["value"]:
            return [_element("dismiss")]
        return []

    tab = make_tab(query_selector_all=AsyncMock(side_effect=_query_side_effect))

    asyncio.run(engine._click_accept_buttons(tab, timeout=2.0))

    assert clicked == ["expand", "dismiss"]


def test_wait_for_response_keeps_short_answer_already_complete(monkeypatch):
    """A fresh answer finished before the wait started must not be called stale.

    Regression: the guard compared against a text read *after* the send. A
    short reply such as ``{"biography": ""}`` was often fully rendered by then,
    so it equalled that read, showed no activity, and was rejected -- then
    re-sent up to five times by the retry loop.
    """
    # The page already shows the complete new answer and never moves again.
    engine = _stale_guard_engine(monkeypatch, [(18, 1)], screen_text="FRESH ANSWER")

    assert (
        asyncio.run(engine._wait_for_response(make_tab(), pre_send_text="OLD ANSWER"))
        == "FRESH ANSWER"
    )


def test_wait_for_response_without_pre_send_text_does_not_guess(monkeypatch):
    """With no pre-send snapshot the guard stays off instead of trusting a
    post-send read that may already contain the new answer."""
    engine = _stale_guard_engine(monkeypatch, [(10, 1)])

    assert asyncio.run(engine._wait_for_response(make_tab())) == "OLD ANSWER"


class _StopAfterNavigation(Exception):
    """Raised by a stubbed step to end the prompt flow right after navigation."""


def _navigation_probe_engine(monkeypatch, *, fresh_chat: bool, retry_nav: bool = False):
    """Engine already sitting on its service URL, instrumented to record navigation."""
    import tempfile
    import time as _time

    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    tab = make_tab(url="https://example.com/app/used-conversation")
    engine.driver = tab
    engine._initialized = True
    engine._driver_start_time = _time.time()
    engine._fresh_chat_per_request = fresh_chat
    engine._navigate_on_next_attempt = retry_nav

    engine._ensure_ready = AsyncMock(return_value=None)
    engine.refresh_login_state = AsyncMock(return_value=True)
    engine._is_dead_session = lambda exc: False
    engine._wait_for_page_ready = AsyncMock(return_value=None)
    engine._click_accept_buttons = AsyncMock(return_value=None)
    engine._is_captcha_present = AsyncMock(return_value=False)
    engine._is_limit_present = AsyncMock(return_value=False)
    engine._find_interactable_element = AsyncMock(return_value=make_element())

    async def _stop(tab, element, prompt):
        raise _StopAfterNavigation()

    engine._fill_input = _stop
    return engine, tab


def test_fresh_chat_per_request_navigates_even_on_service_url(monkeypatch):
    """An engine flagged fresh_chat_per_request must not reuse a used conversation.

    Regression: navigation was skipped whenever the browser was already on the
    service URL. Gemini then received every prompt after the first in the same
    chat, where the send was clicked but never submitted, so only the first
    request after a driver reset ever succeeded.
    """
    engine, tab = _navigation_probe_engine(monkeypatch, fresh_chat=True)

    with pytest.raises(_StopAfterNavigation):
        asyncio.run(engine._generate_response_once("hello"))

    tab.get.assert_called_once_with("https://example.com")


def test_without_flag_the_used_page_is_reused(monkeypatch):
    """Engines that do not opt in keep the existing no-reload behaviour."""
    engine, tab = _navigation_probe_engine(monkeypatch, fresh_chat=False)

    with pytest.raises(_StopAfterNavigation):
        asyncio.run(engine._generate_response_once("hello"))

    tab.get.assert_not_called()


def test_session_hard_timeout_uses_graceful_reset_not_sigkill(monkeypatch):
    """The session-hard-timeout watchdog must quit Chrome gracefully, not SIGKILL it.

    Regression: it called _force_reset_driver(), which sends SIGKILL straight
    to the Chrome process tree without ever calling driver.quit()/browser.stop().
    Chrome batches its Cookies/IndexedDB writes to disk instead of flushing them
    synchronously, so a SIGKILL landing mid-batch silently drops whatever the
    site had just written for the session -- e.g. Google rotating the Gemini
    auth cookie. Since this watchdog runs on every request once the driver has
    been alive past the timeout, it was force-logging the user out during
    normal, active use. _reset_driver() gives the graceful stop path up to 5s to
    finish (falling back to a kill only if it hangs), so it must be used instead.
    """
    import time as _time

    engine, tab = _navigation_probe_engine(monkeypatch, fresh_chat=False)
    engine._driver_start_time = _time.time() - 999_999  # long past any timeout
    engine._session_hard_timeout = 300

    reset_calls = []
    force_reset_calls = []
    engine._reset_driver = AsyncMock(side_effect=lambda: reset_calls.append(True))
    engine._force_reset_driver = AsyncMock(side_effect=lambda: force_reset_calls.append(True))

    with pytest.raises(_StopAfterNavigation):
        asyncio.run(engine._generate_response_once("hello"))

    assert reset_calls == [True]
    assert force_reset_calls == []


def test_retry_after_failed_attempt_opens_a_fresh_chat(monkeypatch):
    """A failed attempt must not be retried in place on the same page."""
    engine, tab = _navigation_probe_engine(
        monkeypatch, fresh_chat=False, retry_nav=True
    )

    with pytest.raises(_StopAfterNavigation):
        asyncio.run(engine._generate_response_once("hello"))

    tab.get.assert_called_once_with("https://example.com")
    # The one-shot flag is consumed by the navigation it triggered.
    assert engine._navigate_on_next_attempt is False


def test_failed_attempt_flags_navigation_for_the_next_one(monkeypatch):
    """The retry loop raises the flag between a failed attempt and the next."""
    import tempfile

    from core.zendriver_llm_base import ZendriverLLMBase

    monkeypatch.setattr("asyncio.sleep", AsyncMock())
    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    seen: list[bool] = []

    async def _attempt(prompt, media=None):
        seen.append(engine._navigate_on_next_attempt)
        if len(seen) == 1:
            raise RuntimeError(
                "Stale response: the text is unchanged from before the prompt "
                "was sent and no generation activity was observed"
            )
        return "ok"

    engine._generate_response_once = _attempt

    assert asyncio.run(engine._generate_response_retry_loop("hello")) == "ok"
    assert seen == [False, True]


def test_failed_attempt_is_logged_before_retry(monkeypatch, caplog):
    """Every failed attempt leaves a warning, whatever path it failed on."""
    import logging
    import tempfile

    from core.zendriver_llm_base import ZendriverLLMBase

    monkeypatch.setattr("asyncio.sleep", AsyncMock())
    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    calls = {"n": 0}

    async def _attempt(prompt, media=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("final chunk never submitted")
        return "ok"

    engine._generate_response_once = _attempt

    with caplog.at_level(logging.WARNING, logger="zendriver_llm_base"):
        assert asyncio.run(engine._generate_response_retry_loop("hello")) == "ok"

    assert "Attempt 1/5 failed: final chunk never submitted" in caplog.text


def _failing_js_insert_engine(monkeypatch):
    """Engine whose editor never accepts the JS insert (read-back stays empty)."""
    import tempfile

    from core.zendriver_llm_base import ZendriverLLMBase

    engine = ZendriverLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    tab = make_tab()
    # editor reads back empty -> never verified, regardless of the JS body
    element = make_element(tag="div", apply_result="")
    return engine, tab, element


def test_fill_input_never_types_non_bmp_text(monkeypatch):
    """Text containing emoji must never reach send_keys.

    Regression: when the JS insert could not be verified, the fallback typed the
    text through the keyboard, which cannot emit anything outside the Basic
    Multilingual Plane: every turn carrying an emoji died with "only supports
    characters in the BMP" and the reply never came.
    """
    engine, tab, element = _failing_js_insert_engine(monkeypatch)

    with pytest.raises(RuntimeError):
        asyncio.run(engine._fill_input(tab, element, "ciao 😵‍💫 come stai"))

    assert element.send_keys.await_count == 0


def test_fill_input_never_types_very_long_text(monkeypatch):
    """A huge prompt must not be typed: real keyboard input would be too slow.

    Regression: a ~32k-char prompt sent through the keyboard fallback hit the
    ChromeDriver HTTP read timeout after 120s and failed the whole request
    (zendriver dispatches one CDP call per character, so the same concern
    applies even though the transport changed).
    """
    engine, tab, element = _failing_js_insert_engine(monkeypatch)

    with pytest.raises(RuntimeError):
        asyncio.run(engine._fill_input(tab, element, "x" * 20000))

    assert element.send_keys.await_count == 0


def test_fill_input_still_types_short_plain_text(monkeypatch):
    """Short BMP-only text keeps using the keyboard fallback."""
    engine, tab, element = _failing_js_insert_engine(monkeypatch)

    with pytest.raises(RuntimeError):
        asyncio.run(engine._fill_input(tab, element, "hello world"))

    typed = [c.args[0] for c in element.send_keys.await_args_list if c.args]
    assert "hello world" in typed
