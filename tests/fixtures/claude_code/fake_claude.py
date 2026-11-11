"""Stand-in executable for the subprocess launcher test (synthetic).

This is not Claude Code and makes no model calls. It reads the prompt on
stdin, prints a few stream-json shaped lines, and then waits to be stopped.

Modes (first argument):
    --version   print a version line and exit
    turns       print an init event and three tool turns, then wait; exit 130 on SIGINT
    ignore      like turns, but ignore SIGINT (the launcher must escalate)
    quiet       exit at once with status 3 and no output
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time


def emit(event: dict[str, object]) -> None:
    """Print one JSON line and flush it."""
    print(json.dumps(event), flush=True)


def stopped(signum: int, frame: object) -> None:
    """Exit the way a CLI interrupted by Ctrl-C does (status 130).

    The handler writes with ``os.write`` and leaves with ``os._exit``: printing
    through ``sys.stdout`` here could interrupt a print in progress and fail
    with a reentrant-call error, which would change the exit status.
    """
    line = json.dumps({"type": "system", "subtype": "stopped", "signal": signum}) + "\n"
    os.write(sys.stdout.fileno(), line.encode("utf-8"))
    os._exit(130)


def main() -> int:
    """Run the requested mode."""
    mode = sys.argv[1] if len(sys.argv) > 1 else "turns"
    if mode == "--version":
        print("2.1.309 (Claude Code)")
        return 0
    if mode == "quiet":
        return 3
    if mode == "ignore":
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    else:
        signal.signal(signal.SIGINT, stopped)
    prompt = sys.stdin.read()
    emit({"type": "system", "subtype": "init", "session_id": "fake", "prompt_chars": len(prompt)})
    print("not json", flush=True)
    for n in range(1, 4):
        use = {"type": "tool_use", "id": f"t{n}", "name": "Bash", "input": {"command": "true"}}
        emit({"type": "assistant", "message": {"id": f"m{n}", "content": [use]}})
        result = {"type": "tool_result", "tool_use_id": f"t{n}", "content": "ok"}
        emit({"type": "user", "message": {"content": [result]}})
        time.sleep(0.05)
    time.sleep(30)
    return 0


if __name__ == "__main__":
    sys.exit(main())
