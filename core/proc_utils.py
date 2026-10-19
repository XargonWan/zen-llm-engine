"""Small, dependency-free process/memory helpers (Linux-first, no-ops elsewhere).

Why this exists: the shared Chromium is a direct child of the server process
(``zendriver`` holds it as a ``subprocess.Popen``). When it is SIGKILLed without
anyone calling ``wait()`` it stays a zombie forever, and the periodic orphan
sweep kept "killing" the same zombies on every pass. Reaping them here is what
actually removes them from the process table.
"""

from __future__ import annotations

import ctypes
import os
import sys
import time

_LINUX = sys.platform.startswith("linux")


def _proc_state(pid: int) -> tuple[str, int] | None:
    """Return ``(state, ppid)`` from ``/proc/<pid>/stat`` or ``None``."""
    try:
        with open(f"/proc/{pid}/stat", "r") as fh:
            data = fh.read()
        # The comm field may contain spaces/parentheses: split after the last ')'.
        fields = data.rsplit(")", 1)[1].split()
        return fields[0], int(fields[1])
    except (OSError, IndexError, ValueError):
        return None


def process_state(pid: int) -> str | None:
    """Return the one-letter state of *pid* (``"Z"`` = zombie) or ``None``."""
    info = _proc_state(pid)
    return info[0] if info else None


def _all_pids() -> list[int]:
    try:
        return [int(name) for name in os.listdir("/proc") if name.isdigit()]
    except OSError:
        return []


def list_zombies(parent: int | None = None) -> list[int]:
    """PIDs of zombie processes, optionally only those whose parent is *parent*."""
    if not _LINUX:
        return []
    found: list[int] = []
    for pid in _all_pids():
        info = _proc_state(pid)
        if info and info[0] == "Z" and (parent is None or info[1] == parent):
            found.append(pid)
    return found


def count_zombies() -> int:
    return len(list_zombies())


def reap_pid(pid: int, timeout: float = 2.0) -> bool:
    """Reap *pid* if it is our child; wait up to *timeout* seconds for it to exit.

    Returns True when the process is gone/reaped (or was never our child), False
    if it was still running at the deadline. Safe to call concurrently with
    ``Popen.wait()``: losing that race just raises ``ChildProcessError``.
    """
    if not hasattr(os, "WNOHANG"):
        return True
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        try:
            done, _status = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            return True
        except OSError:
            return True
        if done == pid:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def reap_zombie_children() -> int:
    """Reap every zombie that is a direct child of this process; return the count."""
    if not _LINUX or not hasattr(os, "WNOHANG"):
        return 0
    reaped = 0
    for pid in list_zombies(parent=os.getpid()):
        if reap_pid(pid, timeout=0.0):
            reaped += 1
    return reaped


def rss_mb() -> float | None:
    """Resident set size of this process in MiB (Linux), else ``None``."""
    if not _LINUX:
        return None
    try:
        with open("/proc/self/status", "r") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024, 1)
    except (OSError, ValueError, IndexError):
        pass
    return None


def trim_memory() -> bool:
    """Ask glibc to return freed heap pages to the OS. No-op if unsupported.

    Resetting the browser frees large DOM/HTML strings; glibc keeps those arenas
    resident, so RSS only ever ratchets upwards without this.
    """
    if not _LINUX:
        return False
    try:
        return bool(ctypes.CDLL("libc.so.6").malloc_trim(0))
    except (OSError, AttributeError):
        return False
