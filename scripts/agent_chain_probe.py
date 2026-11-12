"""Live probe: can a model drive a long chain of OpenAI tool calls through ZEN?

Plays the part of an agent client (Hermes, Cline, ...): it sends the standard
``tools`` request, executes whatever ``tool_calls`` come back against a small
simulated fleet, appends the results as ``tool`` messages and asks again, until
the model answers in plain text. The task needs many steps (list the servers,
read every error count, restart the worst one), so it exercises the session
replay across a long transcript. ``--pad-chars`` and ``--log-lines`` make the
prompt long enough to be sent in several chunks.

It talks to a real engine and a real chat UI: run it on purpose.

    uv run --no-project --with httpx python scripts/agent_chain_probe.py \
        --servers 8 --report /tmp/chain.json
    uv run --no-project --with httpx python scripts/agent_chain_probe.py \
        --servers 8 --pad-chars 12000 --log-lines 40
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_servers",
            "description": "List the names of all servers in the fleet.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_error_count",
            "description": "Errors seen by one server in the last hour.",
            "parameters": {
                "type": "object",
                "properties": {"server": {"type": "string"}},
                "required": ["server"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "restart_server",
            "description": "Restart one server.",
            "parameters": {
                "type": "object",
                "properties": {"server": {"type": "string"}},
                "required": ["server"],
            },
        },
    },
]

_HANDBOOK = (
    "Operations handbook. Servers are named srv-NN. Error counts are per hour. "
    "A restart interrupts traffic for about thirty seconds, so only restart a "
    "server when its error count is the highest of the fleet. Never restart two "
    "servers in the same task. Record nothing outside the tools. When the work is "
    "done, say in one sentence which server was restarted and why. "
)


@dataclass
class Fleet:
    errors: dict[str, int]
    log_lines: int = 0
    restarted: list[str] = field(default_factory=list)

    @classmethod
    def build(cls, size: int, log_lines: int, seed: int = 7) -> "Fleet":
        rng = random.Random(seed)
        counts = rng.sample(range(3, 90), size)
        return cls({f"srv-{i + 1:02d}": counts[i] for i in range(size)}, log_lines)

    @property
    def worst(self) -> str:
        return max(self.errors, key=lambda k: self.errors[k])

    def call(self, name: str, args: dict[str, Any]) -> str:
        if name == "list_servers":
            return ", ".join(self.errors)
        server = str(args.get("server", ""))
        if name not in {"get_error_count", "restart_server"}:
            return f"error: unknown command {name}"
        if server not in self.errors:
            return f"error: no such server {server!r}; known: {', '.join(self.errors)}"
        if name == "restart_server":
            self.restarted.append(server)
            return f"{server} restarted"
        lines = [
            f"2026-11-12T10:{i % 60:02d}:00Z {server} worker-{i % 4} request {1000 + i} "
            f"completed in {20 + (i * 7) % 90} ms status={'500' if i % 9 == 0 else '200'}"
            for i in range(self.log_lines)
        ]
        tail = ("\n" + "\n".join(lines)) if lines else ""
        return f"{server}: {self.errors[server]} errors in the last hour{tail}"


@dataclass
class Step:
    n: int
    seconds: float
    calls: list[str]
    prompt_chars: int


@dataclass
class Report:
    ok: bool = False
    reason: str = ""
    steps: list[Step] = field(default_factory=list)
    answer: str = ""
    worst: str = ""
    restarted: list[str] = field(default_factory=list)
    max_prompt_chars: int = 0


def run(
    base_url: str,
    model: str,
    fleet: Fleet,
    pad_chars: int,
    max_steps: int,
    timeout: int,
) -> Report:
    system = (
        (_HANDBOOK * (pad_chars // len(_HANDBOOK) + 1))[:pad_chars] if pad_chars else ""
    )
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append(
        {
            "role": "user",
            "content": "Find the server with the most errors in the last hour and "
            "restart it. Then tell me which one it was.",
        }
    )
    report = Report(worst=fleet.worst)
    for n in range(1, max_steps + 1):
        started = time.monotonic()
        response = httpx.post(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            json={
                "model": model,
                "timeout": timeout,
                "messages": messages,
                "tools": TOOLS,
            },
            timeout=timeout + 60,
        )
        if response.status_code >= 400:
            report.reason = (
                f"step {n}: HTTP {response.status_code}: {response.text[:200]}"
            )
            return report
        body = response.json()
        message = body["choices"][0]["message"]
        prompt_chars = len(body.get("prompt") or "")
        report.max_prompt_chars = max(report.max_prompt_chars, prompt_chars)
        calls = message.get("tool_calls") or []
        report.steps.append(
            Step(
                n,
                round(time.monotonic() - started, 1),
                [
                    f"{c['function']['name']} {c['function']['arguments']}"
                    for c in calls
                ],
                prompt_chars,
            )
        )
        print(
            f"  step {n}: {report.steps[-1].calls or 'ANSWER'} ({report.steps[-1].seconds}s)",
            flush=True,
        )
        if not calls:
            report.answer = str(message.get("content") or "")
            break
        messages.append({"role": "assistant", "content": None, "tool_calls": calls})
        for call in calls:
            args = json.loads(call["function"].get("arguments") or "{}")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": fleet.call(call["function"]["name"], args),
                }
            )
    else:
        report.reason = f"no answer within {max_steps} steps"
        return report

    report.restarted = list(fleet.restarted)
    if fleet.restarted != [fleet.worst]:
        report.reason = f"restarted {fleet.restarted}, expected [{fleet.worst!r}]"
    elif fleet.worst not in report.answer:
        report.reason = f"answer does not name {fleet.worst}: {report.answer[:120]!r}"
    else:
        report.ok = True
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-url", default="http://localhost:14848")
    parser.add_argument("--model", default="gemini")
    parser.add_argument("--servers", type=int, default=8)
    parser.add_argument(
        "--pad-chars", type=int, default=0, help="size of a long system message"
    )
    parser.add_argument(
        "--log-lines", type=int, default=0, help="log lines per error-count result"
    )
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--report")
    args = parser.parse_args()

    fleet = Fleet.build(args.servers, args.log_lines)
    print(
        f"{args.servers} servers, pad {args.pad_chars}, log lines {args.log_lines}; "
        f"worst is {fleet.worst} ({fleet.errors[fleet.worst]} errors)",
        flush=True,
    )
    report = run(
        args.base_url, args.model, fleet, args.pad_chars, args.max_steps, args.timeout
    )
    total = sum(s.seconds for s in report.steps)
    print(
        f"\n{'PASS' if report.ok else 'FAIL'}: {len(report.steps)} steps, {total:.0f}s, "
        f"longest prompt {report.max_prompt_chars} chars"
        + ("" if report.ok else f" — {report.reason}")
    )
    print(f"answer: {report.answer[:300]!r}")
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(asdict(report), fh, indent=2, ensure_ascii=False)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
