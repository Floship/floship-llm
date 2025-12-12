#!/usr/bin/env python3
"""Check that version numbers are consistent across all files."""

import re
import sys
from pathlib import Path


def get_pyproject_version() -> str | None:
    """Extract version from pyproject.toml."""
    pyproject = Path("pyproject.toml")
    if not pyproject.exists():
        return None
    content = pyproject.read_text()
    match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
    return match.group(1) if match else None


def get_init_version() -> str | None:
    """Extract version from __init__.py."""
    init_file = Path("floship_llm/__init__.py")
    if not init_file.exists():
        return None
    content = init_file.read_text()
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', content, re.MULTILINE)
    return match.group(1) if match else None


def get_test_versions() -> list[str]:
    """Extract version assertions from test_init.py."""
    test_file = Path("tests/test_init.py")
    if not test_file.exists():
        return []
    content = test_file.read_text()
    # Find all version string assertions
    matches = re.findall(r'==\s*"(\d+\.\d+\.\d+)"', content)
    return matches


def get_changelog_version() -> str | None:
    """Extract latest version from CHANGELOG.md."""
    changelog = Path("CHANGELOG.md")
    if not changelog.exists():
        return None
    content = changelog.read_text()
    # Find first version header after [Unreleased]
    match = re.search(r'## \[(\d+\.\d+\.\d+)\]', content)
    return match.group(1) if match else None


def main() -> int:
    """Check version consistency."""
    pyproject_version = get_pyproject_version()
    init_version = get_init_version()
    changelog_version = get_changelog_version()
    test_versions = get_test_versions()

    errors = []

    if not pyproject_version:
        errors.append("❌ Could not find version in pyproject.toml")
    else:
        print(f"📦 pyproject.toml: {pyproject_version}")

    if not init_version:
        errors.append("❌ Could not find __version__ in floship_llm/__init__.py")
    else:
        print(f"🐍 __init__.py: {init_version}")

    if not changelog_version:
        errors.append("❌ Could not find version in CHANGELOG.md")
    else:
        print(f"📋 CHANGELOG.md: {changelog_version}")

    if test_versions:
        unique_test_versions = set(test_versions)
        print(f"🧪 test_init.py: {', '.join(unique_test_versions)}")
    else:
        errors.append("❌ Could not find version assertions in tests/test_init.py")

    # Check consistency
    if pyproject_version and init_version and pyproject_version != init_version:
        errors.append(
            f"❌ Version mismatch: pyproject.toml ({pyproject_version}) != __init__.py ({init_version})"
        )

    if pyproject_version and changelog_version and pyproject_version != changelog_version:
        errors.append(
            f"❌ Version mismatch: pyproject.toml ({pyproject_version}) != CHANGELOG.md ({changelog_version})"
        )

    if pyproject_version and test_versions:
        for tv in set(test_versions):
            if tv != pyproject_version:
                errors.append(
                    f"❌ Version mismatch: pyproject.toml ({pyproject_version}) != test_init.py ({tv})"
                )

    if errors:
        print("\n" + "\n".join(errors))
        print("\n💡 Update all version numbers to match before committing.")
        return 1

    print("\n✅ All versions are consistent!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
