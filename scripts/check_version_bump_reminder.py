#!/usr/bin/env python3
"""Remind to bump version when library source files change.

This pre-commit hook checks if floship_llm/ source files are staged
but none of the version-bearing files (pyproject.toml, __init__.py,
CHANGELOG.md, test_init.py) are also staged. If so, it prints a
reminder — but does NOT block the commit (exits 0).

To make this a hard gate, change the exit code to 1.
"""

from __future__ import annotations

import subprocess  # nosec B404 - git is a trusted binary
import sys

VERSION_FILES = {
    "pyproject.toml",
    "floship_llm/__init__.py",
    "tests/test_init.py",
    "CHANGELOG.md",
}

SOURCE_DIR = "floship_llm/"


def get_staged_files() -> list[str]:
    """Return list of staged file paths."""
    result = subprocess.run(  # nosec B603 B607 - hardcoded git command
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip().splitlines()


def main() -> int:
    """Check if source changes are missing a version bump."""
    staged = get_staged_files()

    source_changed = any(f.startswith(SOURCE_DIR) for f in staged)
    version_changed = bool(VERSION_FILES & set(staged))

    if source_changed and not version_changed:
        print(
            "\n⚠️  Reminder: floship_llm/ source files changed but version "
            "was not bumped."
        )
        print("   If this is a release, update ALL of:")
        for f in sorted(VERSION_FILES):
            print(f"     - {f}")
        print("   Then run: uv lock && python3 scripts/check_version.py")
        print("   After commit+push: git tag vX.Y.Z && git push origin vX.Y.Z")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
