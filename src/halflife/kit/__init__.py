"""Mitigation kit for real agent tools (section 4 of the build brief, RQ5).

Every tool reads one portable constraint spec (:mod:`halflife.kit.spec`):

* :mod:`halflife.kit.guard`: ``halflife-guard``, a Claude Code PreToolUse hook
  that refuses tool calls breaking an enforced rule; its decision core also
  serves the minimal agent's ``guard`` arm.
* :mod:`halflife.kit.pin`: ``halflife-pin``, which writes the rules into
  CLAUDE.md or AGENTS.md (the ``pin`` mitigation for real tools).
* :mod:`halflife.kit.reinject`: the re-injection text and a SessionStart hook
  (matcher ``compact``) that adds the rules after compaction.
* :mod:`halflife.kit.compaction_prompt`: the constraint-preserving summary
  prompt (``preserve``) and Claude Code compact instructions.
* :mod:`halflife.kit.overhead`: latency and token measurements of the tools.

Install and usage: docs/MITIGATIONS.md.
"""
