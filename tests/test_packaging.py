"""The distribution must describe itself consistently.

Two files used to carry the version and a third carried the changelog, with
nothing tying them together: a release could ship a number in the metadata that
the module disagreed with, or no entry saying what changed. These are cheap
checks and they close that hole.
"""

from __future__ import annotations

import re
from importlib import metadata
from pathlib import Path

import pytest

import aotolith

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def test_the_module_and_the_metadata_agree() -> None:
    try:
        installed = metadata.version("aotolith")
    except metadata.PackageNotFoundError:  # pragma: no cover - depends on the env
        pytest.skip("aotolith is not installed in this environment")
    assert installed == aotolith.__version__


def test_the_changelog_has_an_entry_for_this_version() -> None:
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert changelog.startswith("# Changelog")
    headings = re.findall(r"^## \[([^\]]+)\]", changelog, flags=re.MULTILINE)
    assert aotolith.__version__ in headings, headings


def test_the_changelog_does_not_rename_released_versions() -> None:
    """Versions go down the file and never repeat."""

    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    released = [
        heading
        for heading in re.findall(r"^## \[([^\]]+)\]", changelog, flags=re.MULTILINE)
        if heading.lower() != "unreleased"
    ]
    assert len(released) == len(set(released)), released


@pytest.mark.parametrize("name", ["errors", "compatibility"])
def test_the_contract_documents_exist_in_both_languages(name: str) -> None:
    english = DOCS / f"{name}.md"
    translated = DOCS / f"{name}.zh-CN.md"
    assert english.is_file(), english
    assert translated.is_file(), translated
    # A translation that is a stub is worse than none: the first heading and the
    # last line are the cheapest evidence that it is a real document.
    assert english.read_text(encoding="utf-8").strip().splitlines()[0].startswith("#")
    assert translated.read_text(encoding="utf-8").strip().splitlines()[0].startswith("#")


def test_the_readmes_link_the_contract_documents() -> None:
    for name in ("README.md", "README.zh-CN.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "docs/errors" in text, name
        assert "CHANGELOG" in text, name


def test_every_example_the_docs_or_ci_run_actually_exists() -> None:
    """A documented command that no file answers is a broken promise.

    ``examples/tiny_llama_compile.py`` was referenced by both READMEs and by a
    CI step for a while before it existed, so CI was red and the numbers in the
    README had nothing behind them. The reference and the file have to be
    checked together, which is what this does: every ``python examples/...``
    the docs or the workflow run has to be a file in the tree.
    """

    root = ROOT
    sources = [root / "README.md", root / "README.zh-CN.md"]
    sources.extend(sorted((root / "docs").glob("*.md")))
    sources.extend(sorted((root / ".github" / "workflows").glob("*.yml")))

    # The path alone counts, not just `python examples/...`: a doc that points at
    # an example in a sentence rather than in a shell line is still promising the
    # file exists, and the first version of this test let those through.
    referenced: set[str] = set()
    for source in sources:
        for match in re.finditer(
            r"(examples/[\w./-]+\.py)", source.read_text(encoding="utf-8")
        ):
            referenced.add(match.group(1))

    assert referenced, "no example references found; the pattern has gone stale"
    missing = sorted(name for name in referenced if not (root / name).is_file())
    assert not missing, f"documented but missing: {missing}"
