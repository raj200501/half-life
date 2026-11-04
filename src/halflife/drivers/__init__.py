"""Drivers that run session plans on production agents (section 4 of the build brief).

* :mod:`halflife.drivers.claude_code`: headless Claude Code for E2, the
  ecological arm. One ``claude -p`` conversation per session, continued with
  ``--resume`` once per subtask message, stream-json parsing, forked recall
  probes, and the caps and billing guard of section 8.
* :mod:`halflife.drivers.snapshot_hook`: the PostToolUse hook command that
  hashes the workspace after every Claude Code tool call, so detectors see the
  same ``fs_after`` snapshots as in minimal-agent runs.
"""
