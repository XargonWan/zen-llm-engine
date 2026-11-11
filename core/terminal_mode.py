"""Boundary marker shared with the prompt chunker.

When a prompt is too long and has to be sent to the browser UI in several
sequential chunks, the chunker keeps everything from this marker onward in a
single, final chunk so the closing instructions are the LAST thing the model
reads before it answers. The engine layer treats the marker as an opaque string.
"""

AGENT_TAIL_MARKER = "<<<AGENT_TAIL>>>"
