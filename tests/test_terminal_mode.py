"""Terminal mode: the prompt is framed as a plain shell session."""

import re

from core.terminal_mode import (
    AGENT_TAIL_MARKER,
    build_terminal_prompt,
    is_terminal_mode,
)


def test_is_terminal_mode_matches_only_the_terminal_value():
    assert is_terminal_mode("terminal")
    assert is_terminal_mode(" Terminal ")
    assert not is_terminal_mode("agent")
    assert not is_terminal_mode(None)
    assert not is_terminal_mode(True)


def test_frame_follows_the_task_and_ends_with_the_end_marker():
    prompt = build_terminal_prompt("Write a poem into a text file on the USB stick.")
    assert prompt.startswith("Write a poem into a text file on the USB stick.")
    assert "Operating system: Linux" in prompt
    assert "Language: bash" in prompt
    assert prompt.rstrip().endswith("----END----")


def test_frame_sits_behind_the_tail_marker():
    prompt = build_terminal_prompt("task")
    head, _, tail = prompt.partition(AGENT_TAIL_MARKER)
    assert head.strip() == "task"
    assert tail.lstrip().startswith("Operating system: Linux")


def test_frame_never_mentions_agents_tools_or_roleplay():
    # The tail marker is an internal boundary the sender strips before typing.
    prompt = build_terminal_prompt("task").replace(AGENT_TAIL_MARKER, "")
    assert not re.search(r"agent|tool|roleplay|json|\bmode\b", prompt, re.IGNORECASE)
