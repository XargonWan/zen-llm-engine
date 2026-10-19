"""Typed errors that let the engine stop touching the browser when it is pointless.

Both are engine-agnostic. ``LoginRequiredError`` is raised by the generate flow
when a failed attempt coincides with *positive* evidence that the service shows
a logged-out page; retrying (and above all resetting the browser) cannot fix
that, and every reset risks killing a login the operator is completing by hand.
``EngineUnavailableError`` is what :class:`core.engine_manager.EngineManager`
raises, instead of queueing, while an engine's circuit breaker is open.
"""

from __future__ import annotations


class LoginRequiredError(RuntimeError):
    """The service is showing a logged-out page; a human has to sign in."""


class EngineUnavailableError(RuntimeError):
    """The engine's circuit breaker is open: fail fast instead of queueing.

    ``reason`` is ``"login"`` (needs a manual sign-in) or ``"failures"`` (too
    many consecutive failed requests). ``retry_after`` is a hint in seconds for
    the HTTP ``Retry-After`` header.
    """

    def __init__(
        self,
        engine: str,
        reason: str,
        retry_after: int,
        detail: str | None = None,
    ) -> None:
        self.engine = engine
        self.reason = reason
        self.retry_after = max(1, int(retry_after))
        self.detail = detail or ""
        if reason == "login":
            message = f"Engine '{engine}' needs a manual sign-in (login required)"
        else:
            message = f"Engine '{engine}' is temporarily disabled after repeated failures"
        if self.detail:
            message += f": {self.detail}"
        super().__init__(message)
