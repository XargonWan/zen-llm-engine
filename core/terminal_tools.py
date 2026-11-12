"""OpenAI ``tools`` <-> terminal session adapter.

Chat UIs refuse a prompt that announces an agent or a tool harness, but answer
when the exchange looks like a shell session. A client that speaks the standard
OpenAI tool-calling API (``tools`` in the request, ``tool_calls`` in the reply)
therefore gets this treatment:

* the tools are presented as commands installed on the machine
  (``name --param <type>``);
* earlier tool calls and their results are replayed as ``$ command`` / output,
  because the engines are stateless and need the whole session every time;
* the model's reply is read as one command line and converted back into a
  standard ``tool_calls`` message; a lone ``----END----`` ends the session.

No wording sent to the model may mention agents, tools, modes or JSON: any such
word is what the chat UI objects to. The module is engine-agnostic and pure
(the orchestrator takes the send function as an argument).
"""

from __future__ import annotations

import json
import re
import shlex
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable, Sequence

from core.terminal_mode import END_MARKER, build_terminal_prompt

STYLE_CLI = "cli"
STYLE_BASH = "bash"
#: Final-answer tool of agent clients (Cline/Roo-style and SyntH): when the
#: client offers it, the end of the session is expressed as a call to it.
COMPLETION_TOOL = "attempt_completion"
DEFAULT_SUMMARY = "Done."
#: A longer reply is the model talking (leaked reasoning), not a command.
MAX_REPLY_LINES = 8
MAX_RETRIES = 2

#: Introduces the command list. The second sentence matters: told only that some
#: commands are "installed", the model improvises ordinary shell programs
#: (``ls``, ``python -c ...``) instead of using the listed ones.
CATALOG_HEADING = (
    "Only these commands are installed on this machine. Nothing else exists "
    "here: no ls, cat, grep, python or other standard programs."
)

_FENCE_RE = re.compile(r"```[a-zA-Z0-9_-]*\n?(.*?)```", re.DOTALL)
_PROMPT_PREFIX_RE = re.compile(r"^\s*(?:\$|>|#)\s+")
_COMMAND_RE = re.compile(r"^[A-Za-z0-9_./~-]+(\s|$)")
_DESCRIPTION_LIMIT = 110


def normalize_style(value: object) -> str:
    if isinstance(value, str) and value.strip().lower() == STYLE_BASH:
        return STYLE_BASH
    return STYLE_CLI


# ---------------------------------------------------------------------------
# Tool specs
# ---------------------------------------------------------------------------


@dataclass
class Param:
    name: str
    type: str = "string"
    required: bool = False
    enum: list[Any] | None = None


@dataclass
class Spec:
    name: str
    description: str = ""
    params: list[Param] = field(default_factory=list)


def parse_tools(tools: Any) -> list[Spec]:
    """Read an OpenAI ``tools`` list into :class:`Spec` objects."""
    specs: list[Spec] = []
    for tool in tools if isinstance(tools, list) else []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = fn.get("name")
        if not name:
            continue
        schema = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {}
        required = set(schema.get("required") or [])
        props = (
            schema.get("properties")
            if isinstance(schema.get("properties"), dict)
            else {}
        )
        params = []
        for key, meta in props.items():
            meta = meta if isinstance(meta, dict) else {}
            kind = meta.get("type")
            params.append(
                Param(
                    name=str(key),
                    type=kind if isinstance(kind, str) else "string",
                    required=key in required,
                    enum=meta.get("enum")
                    if isinstance(meta.get("enum"), list)
                    else None,
                )
            )
        specs.append(Spec(str(name), str(fn.get("description") or ""), params))
    return specs


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _param_hint(param: Param) -> str:
    kind = (
        "|".join(str(v) for v in param.enum) if param.enum else param.type or "string"
    )
    flag = f"--{param.name} <{kind}>"
    return flag if param.required else f"[{flag}]"


def render_catalog(specs: Iterable[Spec], style: str = STYLE_CLI) -> str:
    """List *specs* as installed commands; empty for the ``bash`` style."""
    if normalize_style(style) == STYLE_BASH:
        return ""
    lines = []
    for spec in specs:
        usage = " ".join([spec.name, *(_param_hint(p) for p in spec.params)])
        text = " ".join(spec.description.split())
        if len(text) > _DESCRIPTION_LIMIT:
            text = text[: _DESCRIPTION_LIMIT - 1].rstrip() + "…"
        lines.append(f"{usage}  # {text}" if text else usage)
    return f"{CATALOG_HEADING}\n" + "\n".join(lines) if lines else ""


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return shlex.quote(json.dumps(value, ensure_ascii=False))
    return shlex.quote(str(value))


def format_command(name: str, arguments: dict[str, Any] | None) -> str:
    parts = [name]
    for key, value in (arguments or {}).items():
        if value is not None:
            parts += [f"--{key}", _format_value(value)]
    return " ".join(parts)


def _loads(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw) if isinstance(raw, str) and raw.strip() else {}
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def split_conversation(
    messages: Sequence[Any], text_of: Callable[[Any], str]
) -> tuple[str, list[str]]:
    """Split OpenAI messages into ``(task, blocks)``.

    Everything before the first assistant turn is the task. After it, each
    assistant tool call becomes ``$ command`` and the matching ``tool`` message
    its output; other text is plain prose from the person.
    """
    task: list[str] = []
    blocks: list[str] = []
    index_by_call: dict[str, int] = {}
    started = False
    for message in messages:
        if not isinstance(message, dict):
            (blocks if started else task).append(text_of(message))
            continue
        role = str(message.get("role", "user"))
        text = text_of(message.get("content", "")).strip()
        if role == "assistant":
            started = True
            for call in message.get("tool_calls") or []:
                fn = call.get("function") or {}
                command = format_command(
                    str(fn.get("name", "")), _loads(fn.get("arguments"))
                )
                index_by_call[str(call.get("id"))] = len(blocks)
                blocks.append(f"$ {command}")
            if text and not message.get("tool_calls"):
                blocks.append(text)
        elif role == "tool":
            slot = index_by_call.get(str(message.get("tool_call_id")))
            if slot is not None:
                blocks[slot] += ("\n" + text) if text else ""
            elif text:
                blocks.append(text)
        elif text:
            (blocks if started else task).append(text)
    return "\n\n".join(task), blocks


def compose(task: str, catalog: str, transcript: str, closing: bool = True) -> str:
    parts = [p for p in (task.strip(), catalog.strip()) if p]
    if transcript.strip():
        parts.append("Session so far:\n" + transcript.strip())
        if closing:
            parts.append(
                f"If that already gets the task done, write only {END_MARKER}."
            )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


@dataclass
class Reply:
    #: ``call``, ``end``, ``unknown`` (command outside the catalog), ``prose``
    #: or ``empty``.
    kind: str
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    command: str = ""
    summary: str = ""


def _lines(text: str) -> list[str]:
    body = text.strip()
    fenced = _FENCE_RE.findall(body)
    if fenced:
        body = "\n".join(fenced)
    out = []
    for raw in body.splitlines():
        line = _PROMPT_PREFIX_RE.sub("", raw.strip()).strip().strip("`").strip()
        if line:
            out.append(line)
    return out


def looks_like_command(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 600 or not _COMMAND_RE.match(line):
        return False
    words = line.split()
    return not (len(words) > 8 and line.endswith((".", "!", "?")))


def _norm(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def _coerce(value: str, kind: str) -> Any:
    try:
        if kind == "integer":
            return int(value)
        if kind == "number":
            number = float(value)
            return int(number) if number.is_integer() else number
        if kind == "boolean":
            return value.strip().lower() in {"1", "true", "yes", "on"}
        if kind in {"array", "object"}:
            return json.loads(value)
    except (TypeError, ValueError):
        return value
    return value


def _parse_line(line: str, specs: dict[str, Spec]) -> Reply | None:
    try:
        tokens = shlex.split(line)
    except ValueError:
        return None
    spec = specs.get(_norm(tokens[0])) if tokens else None
    if spec is None:
        return None
    params = {p.name: p for p in spec.params}
    aliases = {_norm(n): n for n in params}
    args: dict[str, Any] = {}
    positional: list[str] = []
    rest, i = tokens[1:], 0
    while i < len(rest):
        token = rest[i]
        i += 1
        if token.startswith("--") and len(token) > 2:
            key, eq, inline = token[2:].partition("=")
            key = aliases.get(_norm(key), key)
            if eq:
                value: str | None = inline
            elif i < len(rest) and not rest[i].startswith("--"):
                value, i = rest[i], i + 1
            else:
                value = None
            kind = params[key].type if key in params else "string"
            args[key] = True if value is None else _coerce(value, kind)
        else:
            positional.append(token)
    if len(positional) == 1 and not args:
        try:
            loaded = json.loads(positional[0])
        except ValueError:
            loaded = None
        if isinstance(loaded, dict):
            return Reply("call", spec.name, loaded, line)
    for value in positional:
        free = next((n for n in params if n not in args), None)
        if free is None:
            break
        args[free] = _coerce(value, params[free].type)
    return Reply("call", spec.name, args, line)


def parse_reply(text: str | None, specs: Iterable[Spec]) -> Reply:
    """Interpret a model reply as a call, the end of the session, or neither."""
    specs = list(specs)
    if not text or not text.strip():
        return Reply("empty")
    lines = _lines(text)
    if not lines:
        return Reply("empty")
    # Only a closing marker counts: an echo of the instructions that quotes it
    # and then goes on to a command is not finished.
    if lines[-1] == END_MARKER:
        before = lines[:-1]
        # A command followed by the marker is a command not yet run, not an end.
        by_name = {_norm(s.name): s for s in specs}
        for line in before:
            parsed = _parse_line(line, by_name)
            if parsed is not None:
                return parsed
        return Reply("end", summary="\n".join(before).strip())
    if len(lines) > MAX_REPLY_LINES and not _FENCE_RE.search(text):
        return Reply("prose")
    by_name = {_norm(s.name): s for s in specs}
    for line in lines:
        parsed = _parse_line(line, by_name)
        if parsed is not None:
            return parsed
    if looks_like_command(lines[0]):
        return Reply("unknown", command=lines[0])
    return Reply("prose")


def not_found(command: str) -> str:
    first = command.split(None, 1)[0] if command.strip() else command
    return f"bash: {first}: command not found"


def openai_tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"call_{uuid.uuid4().hex[:12]}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


# ---------------------------------------------------------------------------
# Fitting a long session into the chat UI
# ---------------------------------------------------------------------------

_OLD_OUTPUT_CHARS = 400
_RECENT_OUTPUT_CHARS = 2500
_MIN_TASK_CHARS = 1500


def _clip(text: str, limit: int) -> str:
    """Keep the start of *text*; the start of a command output is what matters."""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + f"\n[... {len(text) - limit} characters omitted]"


def _clip_middle(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    return f"{text[:head].rstrip()}\n[... {len(text) - limit} characters omitted ...]\n{text[-tail:].lstrip()}"


def prompt_size(session: Session) -> int:
    """Length of the first prompt :func:`run_session` would send."""
    text = compose(
        session.task, render_catalog(session.specs, session.style), session.transcript
    )
    return len(build_terminal_prompt(text))


def fit_session(session: Session, budget: int) -> Session:
    """Shrink *session* until its prompt fits *budget* characters.

    The web UIs cap the prompt (and misbehave well before the cap when it is
    sent in several parts), while an agent keeps appending command output. In
    order: old outputs are cut to their first lines, recent ones to a longer
    head, the middle of the task text goes, and finally the oldest steps are
    dropped. The newest steps are always kept.
    """
    if budget <= 0 or prompt_size(session) <= budget:
        return session
    fitted = Session(session.specs, session.style, session.task, list(session.blocks))
    keep = 2
    for i in range(max(0, len(fitted.blocks) - keep)):
        if prompt_size(fitted) <= budget:
            return fitted
        fitted.blocks[i] = _clip(fitted.blocks[i], _OLD_OUTPUT_CHARS)
    for i in range(len(fitted.blocks)):
        if prompt_size(fitted) <= budget:
            return fitted
        fitted.blocks[i] = _clip(fitted.blocks[i], _RECENT_OUTPUT_CHARS)
    if prompt_size(fitted) > budget:
        room = len(fitted.task) - (prompt_size(fitted) - budget)
        # The omission marker adds a few dozen characters of its own.
        fitted.task = _clip_middle(fitted.task, max(_MIN_TASK_CHARS, room - 80))
    dropped = 0
    while prompt_size(fitted) > budget and len(fitted.blocks) > keep:
        fitted.blocks.pop(0)
        dropped += 1
    if dropped:
        fitted.blocks.insert(0, f"[{dropped} earlier steps omitted]")
    return fitted


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass
class Session:
    specs: list[Spec]
    style: str
    task: str
    #: One entry per command (with its output) or line of prose, oldest first.
    blocks: list[str]

    @property
    def transcript(self) -> str:
        return "\n\n".join(self.blocks)


@dataclass
class Outcome:
    #: Standard ``message.tool_calls`` entries, or ``None`` for a plain answer.
    tool_calls: list[dict[str, Any]] | None
    content: str | None
    #: Raw text of the last model reply and the last prompt sent (for logs).
    text: str
    prompt: str
    model_name: str


Send = Callable[[str], Awaitable[Any]]


async def run_session(
    send: Send, session: Session, max_retries: int = MAX_RETRIES
) -> Outcome:
    """Drive one turn: ask for a command, answering like a shell when it is wrong.

    *send* delivers a prompt to the engine and returns an object with ``text``
    and ``model_name``. A reply that is not a command is asked again; a command
    outside the catalog is answered ``command not found`` first.
    """
    catalog = render_catalog(session.specs, session.style)
    text = compose(session.task, catalog, session.transcript)
    names = {s.name for s in session.specs}
    framed, result = "", None
    for attempt in range(max_retries + 1):
        framed = build_terminal_prompt(text)
        result = await send(framed)
        reply = parse_reply(result.text, session.specs)
        if reply.kind == "call":
            call = openai_tool_call(reply.name, reply.arguments)
            return Outcome([call], None, result.text, framed, result.model_name)
        if (
            reply.kind == "end"
            and not session.blocks
            and not reply.summary
            and attempt < max_retries
        ):
            # Nothing has been run, so there is nothing to be done yet: the
            # model gave up on a long catalog. Ask again on a fresh chat.
            continue
        if reply.kind == "end":
            summary = reply.summary
            if not summary:
                # The marker carries no text, but a client expects an answer.
                recap = compose(session.task, "", session.transcript, closing=False)
                ask = (
                    "Using only the output shown above, tell me the final answer in "
                    "plain words. No commands, just the answer."
                    if session.blocks
                    else "Answer in plain words. No commands, just the answer."
                )
                answer = await send(f"{recap}\n\n{ask}")
                summary = (answer.text or "").strip() or DEFAULT_SUMMARY
            if COMPLETION_TOOL in names:
                call = openai_tool_call(COMPLETION_TOOL, {"summary": summary})
                return Outcome([call], None, result.text, framed, result.model_name)
            return Outcome(None, summary, result.text, framed, result.model_name)
        if attempt == max_retries:
            break
        if reply.kind == "unknown":
            text += f"\n\n$ {reply.command}\n{not_found(reply.command)}"
    assert result is not None
    return Outcome(None, result.text, result.text, framed, result.model_name)
