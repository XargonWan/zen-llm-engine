"""Boundary marker shared with the prompt chunker.

When a prompt is too long and has to be sent to the browser UI in several
sequential chunks, the chunker keeps everything from this marker onward in a
single, final chunk so the closing instructions are the LAST thing the model
reads before it answers. The engine layer treats the marker as an opaque string.
"""

AGENT_TAIL_MARKER = "<<<AGENT_TAIL>>>"

# Request field that switches a chat completion into terminal mode.
TERMINAL_MODE = "terminal"

# What the model writes, alone on a line, when it considers the session done.
END_MARKER = "----END----"

# Static frame, always English. It is what makes the model treat the exchange
# as a plain shell session (the caller pastes each command's output back) rather
# than as a tool-calling harness, which some chat UIs refuse outright.
_TERMINAL_FRAME = (
    "Operating system: Linux\n"
    "Language: bash\n"
    "\n"
    "Give me only the commands, a single command string at a time, that I can "
    "copy and paste into a terminal, nothing else. I can answer you with prose "
    "or output.\n"
    "When you think we are done, write me only:\n"
    "\n"
    f"{END_MARKER}"
)


def is_terminal_mode(mode: object) -> bool:
    """Return True when a request's ``mode`` field selects terminal mode."""
    return isinstance(mode, str) and mode.strip().lower() == TERMINAL_MODE


def build_terminal_prompt(task: str) -> str:
    """Wrap *task* (prose and/or a transcript) in the terminal-session frame.

    The frame sits behind ``AGENT_TAIL_MARKER`` so a chunked send keeps it, and
    therefore the instructions, in the last chunk the model reads.
    """
    return f"{task.rstrip()}\n\n{AGENT_TAIL_MARKER}\n{_TERMINAL_FRAME}"
