"""The constraint-preserving compaction prompt (the ``preserve`` mitigation).

:func:`preserve_prompt` replaces :data:`halflife.compaction.FULL_SUMMARY_PROMPT`
in the minimal agent's ``preserve`` arm. It keeps the default prompt's wording
and adds one request: begin the summary with a "Rules in force" section that
copies every rule the user stated, word for word. The rules come first so a
summary cut short by the token cap still keeps them. The prompt never contains
the rules itself; the summarizer has to find them in the conversation, as a
real compactor would.

For Claude Code the same request goes into a "Compact Instructions" section of
CLAUDE.md, which Claude Code consults when it compacts, or into ``/compact``
with a focus ("To control what's preserved during compaction, add a 'Compact
Instructions' section to CLAUDE.md or run /compact with a focus",
https://code.claude.com/docs/en/how-claude-code-works, retrieved 2026-11-04).
There is no PreCompact hook here: a PreCompact hook can only block compaction
(exit 2 or ``"decision": "block"``), and Claude Code discards its
``systemMessage`` and ``continue`` fields, so it cannot add instructions
(https://code.claude.com/docs/en/hooks, retrieved 2026-11-04).
"""

from __future__ import annotations

__all__ = [
    "COMPACT_HEADING",
    "RULES_SECTION",
    "claude_compact_instructions",
    "compact_command",
    "compact_instructions_section",
    "preserve_prompt",
]

RULES_SECTION = "Rules in force"
"""Title of the section the summary must start with."""
COMPACT_HEADING = "## Compact Instructions"
"""The CLAUDE.md heading Claude Code reads compaction instructions from."""

_COPY_RULES = (
    f'Begin with a section titled "{RULES_SECTION}" that copies, word for word, every rule '
    "or constraint the user gave for this session, one per line, with any reason given for "
    "it. Do not shorten, merge, reword, or leave out any rule, even one that seems unrelated "
    "to the current request."
)


def preserve_prompt() -> str:
    """Summary instructions for the minimal agent's ``preserve`` arm.

    The same as :data:`halflife.compaction.FULL_SUMMARY_PROMPT` except that the
    summary must begin with a "Rules in force" section copying every rule word
    for word; the 200-word limit applies to the rest of the summary.
    """
    return (
        "The conversation below is about to be removed from your context to save space. "
        "Write a summary that lets you continue the work. "
        f"{_COPY_RULES} "
        "Then cover the project, the requests so far and their status, the current request, "
        "the files changed, and anything else you need to remember. "
        f'Plain text, at most 200 words after the "{RULES_SECTION}" section.'
    )


def claude_compact_instructions() -> str:
    """The same request phrased for Claude Code's compaction (CLAUDE.md or ``/compact``)."""
    return (
        "When you summarize this conversation for compaction, follow these instructions. "
        f"{_COPY_RULES} "
        "Include the rules from this file and any rule the user stated in the conversation. "
        "Then summarize the work as usual."
    )


def compact_instructions_section() -> str:
    """A "Compact Instructions" section for CLAUDE.md, ending with a newline."""
    return f"{COMPACT_HEADING}\n\n{claude_compact_instructions()}\n"


def compact_command() -> str:
    """A manual ``/compact`` command that asks Claude Code to keep the rules."""
    return f"/compact {claude_compact_instructions()}"
