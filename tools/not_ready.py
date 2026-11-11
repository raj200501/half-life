#!/usr/bin/env python3
"""Placeholder for a make target whose feature is not built yet.

It always exits non-zero, so a gate that depends on the target fails honestly
until the real implementation replaces this call in the Makefile.
"""

from __future__ import annotations

import sys


def main(argv: list[str]) -> int:
    """Print which target is missing and exit with status 1."""
    target = argv[1] if len(argv) > 1 else "target"
    what = argv[2] if len(argv) > 2 else ""
    detail = f" ({what})" if what else ""
    print(f"NOT IMPLEMENTED: make {target}{detail}. See docs/PLAN.md.")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
