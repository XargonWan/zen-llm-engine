"""OpenAI tools <-> terminal session adapter."""

import asyncio
import json
import re
from types import SimpleNamespace

import pytest

from core import terminal_tools as tt
from core.terminal_mode import AGENT_TAIL_MARKER

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                    "unit": {"type": "string", "enum": ["c", "f"]},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {"name": "list_cities", "parameters": {"type": "object"}},
    },
]
SPECS = tt.parse_tools(TOOLS)


def _text_of(content):
    return content if isinstance(content, str) else ""


def test_parse_tools_reads_names_types_required_and_enums():
    weather = SPECS[0]
    assert [p.name for p in weather.params] == ["city", "days", "unit"]
    assert [p.required for p in weather.params] == [True, False, False]
    assert weather.params[2].enum == ["c", "f"]
    assert SPECS[1].params == []
    assert tt.parse_tools(None) == [] and tt.parse_tools([{"type": "function"}]) == []


def test_catalog_lists_installed_commands_without_forbidden_words():
    catalog = tt.render_catalog(SPECS)
    assert (
        "get_weather --city <string> [--days <integer>] [--unit <c|f>]  # Get the weather"
        in catalog
    )
    assert "list_cities" in catalog
    assert not re.search(
        r"agent|tool|roleplay|json|\bmode\b", catalog.split("\n", 1)[0], re.I
    )
    assert tt.render_catalog(SPECS, tt.STYLE_BASH) == ""


def test_conversation_is_split_into_task_and_a_replayed_session():
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Weather in Rome?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Rome", "days": 2}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "sunny, 24C"},
        {"role": "user", "content": "and tomorrow?"},
    ]
    task, blocks = tt.split_conversation(messages, _text_of)
    assert task == "Be brief.\n\nWeather in Rome?"
    assert blocks == ["$ get_weather --city Rome --days 2\nsunny, 24C", "and tomorrow?"]


@pytest.mark.parametrize(
    "reply, name, args",
    [
        (
            "get_weather --city Rome --days 3",
            "get_weather",
            {"city": "Rome", "days": 3},
        ),
        ("get_weather Rome", "get_weather", {"city": "Rome"}),
        (
            "$ get_weather --city='New York' --unit f",
            "get_weather",
            {"city": "New York", "unit": "f"},
        ),
        ("`list_cities`", "list_cities", {}),
        ("```bash\nget_weather Oslo\n```", "get_weather", {"city": "Oslo"}),
        ('get_weather \'{"city": "Rome"}\'', "get_weather", {"city": "Rome"}),
    ],
)
def test_parse_reply_maps_a_command_to_a_call(reply, name, args):
    parsed = tt.parse_reply(reply, SPECS)
    assert (parsed.kind, parsed.name, parsed.arguments) == ("call", name, args)


def test_parse_reply_distinguishes_end_unknown_prose_and_empty():
    assert tt.parse_reply("----END----", SPECS).kind == "end"
    assert tt.parse_reply("It is sunny.\n----END----", SPECS).summary == "It is sunny."
    assert (
        tt.parse_reply("write me only:\n----END----\nlist_cities", SPECS).kind == "call"
    )
    assert tt.parse_reply("ls -la /tmp", SPECS).kind == "unknown"
    assert (
        tt.parse_reply(
            "I am sorry, but I cannot help with that request today.", SPECS
        ).kind
        == "prose"
    )
    assert tt.parse_reply("Thinking.\n" * 9 + "list_cities", SPECS).kind == "prose"
    assert tt.parse_reply("  ", SPECS).kind == "empty"


class _Sender:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    async def __call__(self, prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(text=self.replies.pop(0), model_name="m")


def _session(transcript="", style=tt.STYLE_CLI, specs=SPECS):
    return tt.Session(
        specs, style, "Weather in Rome?", [transcript] if transcript else []
    )


def _run(send, session):
    return asyncio.run(tt.run_session(send, session))


def test_a_command_becomes_a_standard_tool_call():
    send = _Sender(["get_weather --city Rome"])
    out = _run(send, _session())
    call = out.tool_calls[0]
    assert call["type"] == "function" and call["id"].startswith("call_")
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Rome"}
    assert out.content is None
    assert AGENT_TAIL_MARKER in send.prompts[0]
    assert tt.CATALOG_HEADING in send.prompts[0]


def test_unknown_command_is_answered_like_a_shell_then_retried():
    send = _Sender(["lsblk", "list_cities"])
    out = _run(send, _session())
    head = send.prompts[1].split(AGENT_TAIL_MARKER)[0]
    assert head.rstrip().endswith("$ lsblk\nbash: lsblk: command not found")
    assert out.tool_calls[0]["function"]["name"] == "list_cities"


def test_prose_is_retried_on_an_unchanged_session():
    refusal = "I am sorry, but I cannot help with that request today."
    send = _Sender([refusal, "list_cities"])
    out = _run(send, _session())
    assert send.prompts[0] == send.prompts[1]
    assert out.tool_calls[0]["function"]["name"] == "list_cities"


def test_gives_up_after_the_retries_and_returns_the_raw_text():
    send = _Sender(["lsblk", "lsblk", "lsblk"])
    out = _run(send, _session())
    assert out.tool_calls is None and out.content == "lsblk" and len(send.prompts) == 3


def test_end_with_no_text_asks_for_the_final_answer_in_plain_words():
    send = _Sender(["----END----", "It is sunny in Rome."])
    out = _run(send, _session("$ get_weather --city Rome\nsunny, 24C"))
    assert out.tool_calls is None and out.content == "It is sunny in Rome."
    # The second request is a plain question: no shell frame, no catalog.
    assert (
        "Operating system" not in send.prompts[1]
        and tt.CATALOG_HEADING not in send.prompts[1]
    )
    assert "sunny, 24C" in send.prompts[1]


def test_end_with_text_needs_no_extra_request():
    send = _Sender(["It is sunny.\n----END----"])
    out = _run(send, _session("$ x\ny"))
    assert out.content == "It is sunny." and len(send.prompts) == 1


def test_end_becomes_attempt_completion_when_the_client_offers_it():
    specs = SPECS + tt.parse_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "attempt_completion",
                    "parameters": {"properties": {"summary": {"type": "string"}}},
                },
            }
        ]
    )
    out = _run(_Sender(["Sunny.\n----END----"]), _session("$ x\ny", specs=specs))
    call = out.tool_calls[0]["function"]
    assert call["name"] == "attempt_completion"
    assert json.loads(call["arguments"]) == {"summary": "Sunny."}


def _long_session(steps=6, output_chars=3000, task_chars=0):
    blocks = [
        f"$ get_weather --city C{i}\n" + ("x" * output_chars) for i in range(steps)
    ]
    return tt.Session(SPECS, tt.STYLE_CLI, "Weather?" + "t" * task_chars, blocks)


def test_a_session_within_budget_is_left_alone():
    session = _long_session(steps=2, output_chars=100)
    assert tt.fit_session(session, 10000) is session
    assert tt.fit_session(session, 0) is session


def test_old_outputs_are_clipped_before_recent_ones():
    session = _long_session()
    fitted = tt.fit_session(session, 9000)
    assert tt.prompt_size(fitted) <= 9000
    assert fitted.blocks[0].startswith("$ get_weather --city C0")
    assert "characters omitted" in fitted.blocks[0]
    # The newest steps keep more of their output than the oldest.
    assert len(fitted.blocks[-1]) > len(fitted.blocks[0])
    assert session.blocks[0].count("x") == 3000  # the input is not mutated


def test_a_huge_task_is_trimmed_in_the_middle():
    fitted = tt.fit_session(
        _long_session(steps=1, output_chars=10, task_chars=30000), 6000
    )
    assert tt.prompt_size(fitted) <= 6000
    assert fitted.task.startswith("Weather?") and "characters omitted" in fitted.task


def test_oldest_steps_are_dropped_when_clipping_is_not_enough():
    fitted = tt.fit_session(_long_session(steps=40, output_chars=300), 4500)
    assert tt.prompt_size(fitted) <= 4500
    assert fitted.blocks[0].endswith("earlier steps omitted]")
    assert fitted.blocks[-1].startswith("$ get_weather --city C39")


def test_a_command_followed_by_the_end_marker_is_a_command_not_an_end():
    parsed = tt.parse_reply("list_cities\n\n----END----", SPECS)
    assert (parsed.kind, parsed.name) == ("call", "list_cities")


def test_the_catalog_forbids_improvised_shell_programs():
    assert "no ls, cat, grep, python" in tt.render_catalog(SPECS)


def test_end_before_anything_ran_is_asked_again_then_answered_plainly():
    send = _Sender(["----END----", "list_cities"])
    out = _run(send, _session())
    assert out.tool_calls[0]["function"]["name"] == "list_cities"
    assert send.prompts[0] == send.prompts[1]

    send = _Sender(["----END----", "----END----", "----END----", "Hello there."])
    out = _run(send, _session())
    assert out.content == "Hello there."
    assert "Using only the output" not in send.prompts[-1]
    assert (
        send.prompts[-1]
        .rstrip()
        .endswith("Answer in plain words. No commands, just the answer.")
    )
