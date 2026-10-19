"""JsonEngine — a ZendriverLLMBase subclass driven entirely by a JSON config dict.

Any LLM web interface that follows the common pattern:
  1. Navigate to a URL
  2. Type in a prompt area
  3. Click a send button
  4. Wait for the response area to stabilise

can be fully described with a JSON file.  No Python code required.

Schema (all fields except ``name`` and ``service_url`` are optional):

.. code-block:: json

    {
        "name": "my-engine",
        "display_name": "My Engine (Example)",
        "aliases": ["my-engine", "myengine"],
        "service_url": "https://www.example-llm.com",
        "default_model": "default",
        "models": {
            "default": 32000,
            "unlogged": 10000
        },
        "selectors": {
            "prompt_area": ["div[contenteditable='true']", "textarea"],
            "send_button": ["button[type='submit']"],
            "response_area": ["div.markdown"],
            "stop":         ["button[aria-label*='Stop']"]
        },
        "login_detection": {
            "url_prefix": "https://www.example-llm.com",
            "url_deny_keywords": ["login", "auth", "signin"],
            "login_button_xpath": "//button[contains(normalize-space(.), 'Log in')]",
            "authenticated_css_selectors": [
                "div[data-testid='conversation-panel']"
            ]
        }
    }
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from core.zendriver_llm_base import ZendriverLLMBase

logger = logging.getLogger("json_engine")

# Required top-level keys in every engine JSON file.
_REQUIRED_FIELDS: tuple[str, ...] = ("name", "service_url")


def _load_json(source: Path | dict[str, Any]) -> dict[str, Any]:
    """Accept either a file path or an already-parsed dict."""
    if isinstance(source, dict):
        return source
    with source.open(encoding="utf-8") as fh:
        return json.load(fh)


def _validate(config: dict[str, Any], source: str) -> None:
    missing = [k for k in _REQUIRED_FIELDS if not config.get(k)]
    if missing:
        raise ValueError(
            f"Engine JSON '{source}' is missing required field(s): {missing}"
        )


class JsonEngine(ZendriverLLMBase):
    """Zendriver engine whose entire configuration is read from a JSON dict/file.

    Parameters
    ----------
    source:
        Either a ``Path`` pointing to a ``.json`` file or a pre-parsed ``dict``.
    **kwargs:
        Forwarded verbatim to :class:`~core.zendriver_llm_base.ZendriverLLMBase`
        (e.g. ``headless``, ``profile_dir``).
    """

    def __init__(
        self,
        source: Path | dict[str, Any],
        **kwargs: Any,
    ) -> None:
        config = _load_json(source)
        source_label = str(source) if isinstance(source, Path) else "<dict>"
        _validate(config, source_label)

        self._config = config

        # ------------------------------------------------------------------ meta
        self.ENGINE_NAME: str = config["name"]
        self.ENGINE_ALIASES: list[str] = config.get("aliases", [config["name"]])
        self.display_name: str = config.get("display_name", config["name"])

        # ------------------------------------------------------------------ base
        super().__init__(
            service_url=config["service_url"],
            model_limits_map=dict(config.get("models", {"default": 10000})),
            default_model=config.get("default_model", "default"),
            allow_unlogged=bool(config.get("allow_unlogged", False)),
            **kwargs,
        )

        # ------------------------------------------------------------------ selectors
        sel = config.get("selectors", {})

        if "prompt_area" in sel:
            self.prompt_area_selectors = list(sel["prompt_area"])
        if "send_button" in sel:
            self.send_button_selectors = list(sel["send_button"])
        if "accept_button" in sel:
            self.accept_button_selectors = list(sel["accept_button"])
        if "response_area" in sel:
            self.response_area_selectors = list(sel["response_area"])
        if "stop" in sel:
            self.stop_selectors = list(sel["stop"])
        if "limit" in sel:
            self.limit_selectors = list(sel["limit"])
        if "error_indicators" in sel:
            self.error_indicator_selectors = list(sel["error_indicators"])
        if "send_button_blacklist" in sel:
            self.send_button_blacklist = list(sel["send_button_blacklist"])
        self.base_account_selector = sel.get("base_account_selector")

        # ------------------------------------------------------------------ login rules
        self._login_cfg: dict[str, Any] = config.get("login_detection", {})
        # True once the "state undetermined" fallback was logged for the
        # current streak; cleared whenever a definite state is observed.
        self._login_fallback_warned: bool = False

        # ------------------------------------------------------------------ response timing
        # Optional JSON key "response_max_wait" (seconds) overrides the built-in
        # 120-second stability-loop timeout in _wait_for_response.
        # Useful for slow/thinking models (e.g. Gemini 2.5 Flash) that can take
        # several minutes to produce a full response.
        if "response_max_wait" in config:
            self._response_max_wait = int(config["response_max_wait"])
        if "total_timeout" in config:
            self._total_timeout = int(config["total_timeout"])
        if "use_baseline_comparison" in config:
            self._use_baseline_comparison = bool(config["use_baseline_comparison"])
        # Optional JSON key "fresh_chat_per_request": open a new conversation for
        # every request instead of reusing the page already on the service URL.
        if "fresh_chat_per_request" in config:
            self._fresh_chat_per_request = bool(config["fresh_chat_per_request"])
        # Optional JSON key "silent_freeze_threshold" (seconds): how long the stop
        # button may remain visible with no response-text activity before the page
        # is considered stuck. Engines that emit the whole answer in one block
        # (no incremental streaming, e.g. Gemini) need a larger value so the
        # initial thinking delay isn't mistaken for a freeze.
        if "silent_freeze_threshold" in config:
            self._silent_freeze_threshold = float(config["silent_freeze_threshold"])
        if "vision_prompt_prefix" in config:
            self._vision_prompt_prefix = str(config["vision_prompt_prefix"])
        if "inline_response_prefix" in config:
            self._inline_response_prefix = str(config["inline_response_prefix"])

        # ------------------------------------------------------------------ media support
        self.media_config: dict[str, Any] = config.get("media_support", {})
        self.paid_account_selector = self.media_config.get("paid_account_selector")
        if self.base_account_selector is None:
            self.base_account_selector = self.media_config.get("base_account_selector")

    # ---------------------------------------------------------------------- login markers

    # One CDP round-trip instead of find-then-probe per element: no stale
    # node handles, and "visible" means a non-empty bounding box.
    _ANY_VISIBLE_JS = (
        "((sels) => sels.some((s) => { try { return Array.from("
        "document.querySelectorAll(s)).some((e) => { const r = "
        "e.getBoundingClientRect(); return r.width > 0 && r.height > 0; }); } "
        "catch (_) { return false; } }))(__SELECTORS__)"
    )

    async def _css_any_visible(self, tab: Any, selectors: list[str]) -> bool:
        """True if any CSS selector matches an element with a non-empty box."""
        if not selectors:
            return False
        expression = self._ANY_VISIBLE_JS.replace("__SELECTORS__", json.dumps(list(selectors)))
        try:
            return (await tab.evaluate(expression)) is True
        except Exception:
            return False

    async def _login_css_marker_visible(self, tab: Any) -> bool:
        """True if a ``login_button_css`` marker is visible right now.

        This is the *opt-in, high-confidence* signal: an engine author has
        declared exactly which elements mean "signed out". It alone may stop an
        engine (see ``LoginRequiredError``). Never navigates.
        """
        return await self._css_any_visible(tab, self._login_cfg.get("login_button_css", []))

    async def _login_marker_visible(self, tab: Any) -> bool:
        """True if any configured "please sign in" element is visible right now.

        Adds the legacy ``login_button_xpath`` on top of the CSS markers. XPath
        rules in existing engine files are broad, so this is only used for the
        informational login state, never to disable an engine.
        """
        if await self._login_css_marker_visible(tab):
            return True
        login_xpath: str = self._login_cfg.get("login_button_xpath", "")
        if login_xpath:
            for button in await self._xpath_all(tab, login_xpath):
                if await self._element_is_displayed(button):
                    return True
        return False

    def _on_service_host(self, url: str) -> bool:
        prefix = str(self._login_cfg.get("url_prefix") or self.service_url or "").lower()
        return bool(prefix) and url.lower().startswith(prefix)

    async def _peek_logged_out(self, tab: Any) -> bool:  # type: ignore[override]
        """Read-only: True only on positive evidence of a logged-out page.

        Evidence = a declared ``login_button_css`` marker visible on the service
        page. An off-service URL (login redirect, about:blank, ...) is *not*
        evidence here: redirects are handled by the redirect-stall path.
        """
        url = (await self._current_url(tab)).lower()
        if not self._on_service_host(url):
            return False
        return await self._login_css_marker_visible(tab)

    async def peek_login_state(self) -> bool | None:  # type: ignore[override]
        """Non-navigating probe used while the circuit breaker is open.

        False: logged-out marker visible. True: a configured authenticated
        element is visible, or the service page is fully loaded and no declared
        logged-out marker is visible. None: cannot tell (no driver, off the
        service host, page still loading, or no markers configured).
        """
        tab = self.driver
        if tab is None or not self._initialized:
            return None
        cfg = self._login_cfg
        try:
            url = (await self._current_url(tab)).lower()
            if not self._on_service_host(url):
                return None
            if await self._login_css_marker_visible(tab):
                return False
            if await self._css_any_visible(tab, cfg.get("authenticated_css_selectors", [])):
                return True
            if cfg.get("login_button_css"):
                ready = await tab.evaluate("document.readyState")
                return True if ready == "complete" else None
        except Exception:
            return None
        return None

    # ---------------------------------------------------------------------- login

    async def _ensure_logged_in(self, tab: Any) -> bool:  # type: ignore[override]
        """Generic login detection driven by ``login_detection`` config block.

        Detection steps (all optional — steps are skipped if the corresponding
        config key is absent):

        1. Navigate to ``url_prefix`` when the current URL doesn't start with it.
        2. Any of the ``url_deny_keywords`` in the current URL → **not logged in**.
        3. Any element matched by ``login_button_xpath`` is visible → **not logged in**.
        4. Any element matched by one of ``authenticated_css_selectors`` → **logged in**.
        5. Fallback → **logged in** (assume OK when no evidence of being locked out).
        """
        cfg = self._login_cfg
        try:
            current_url = (await self._current_url(tab)).lower()

            # Step 0 — authenticated element present on the current page
            auth_selectors: list[str] = cfg.get("authenticated_css_selectors", [])
            for css in auth_selectors:
                els = await self._query_all(tab, css)
                for e in els:
                    if await self._element_is_displayed(e):
                        self._login_fallback_warned = False
                        return True

            url_prefix: str = cfg.get("url_prefix", self.service_url)
            if url_prefix and not current_url.startswith(url_prefix):
                await tab.get(url_prefix)
                current_url = (await self._current_url(tab)).lower()

            # Step 2 — deny keywords in URL
            deny_keywords: list[str] = cfg.get("url_deny_keywords", [])
            if any(kw in current_url for kw in deny_keywords):
                return False

            # Steps 3 + 4 — visible login button / authenticated element. Single-page
            # apps draw these a moment after the page "loads": checking once, too
            # early, found neither and fell through to "assume logged in" (seen
            # live: the check ran ~1 s after launch). ``login_settle_seconds``
            # (opt-in, default 0 = single check) keeps looking for that long.
            loop = asyncio.get_running_loop()
            deadline = loop.time() + float(cfg.get("login_settle_seconds", 0) or 0)
            while True:
                if await self._login_marker_visible(tab):
                    self._login_fallback_warned = False
                    return False
                for css in auth_selectors:
                    els = await self._query_all(tab, css)
                    for e in els:
                        if await self._element_is_displayed(e):
                            self._login_fallback_warned = False
                            return True
                if loop.time() >= deadline:
                    break
                await asyncio.sleep(0.5)

            # Step 5 — fallback: no evidence either way. This used to be
            # silent, which is how a logged-out engine kept reporting "logged
            # in" for a day. Say so (once per streak) so it shows up in logs.
            if not self._login_fallback_warned:
                self._login_fallback_warned = True
                logger.warning(
                    "[json_engine:%s] login state undetermined: no authenticated "
                    "element and no login marker matched; assuming logged in. "
                    "Check login_detection selectors if this engine misbehaves.",
                    self.ENGINE_NAME,
                )
            return True

        except Exception as exc:
            logger.warning(
                f"[json_engine:{self.ENGINE_NAME}] login check failed: {exc}"
            )
            return False
