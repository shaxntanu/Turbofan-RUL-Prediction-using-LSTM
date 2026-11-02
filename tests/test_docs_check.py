from pathlib import Path

from scripts.check_docs import find_broken_links


def test_existing_relative_markdown_target_is_accepted(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text("# Guide\n", encoding="utf-8")
    source = tmp_path / "README.md"
    source.write_text("[Guide](docs/guide.md)\n", encoding="utf-8")

    assert find_broken_links(tmp_path) == []


def test_missing_relative_target_reports_source_and_line(tmp_path: Path) -> None:
    source = tmp_path / "README.md"
    source.write_text("# Home\n\n[Guide](docs/missing.md)\n", encoding="utf-8")

    broken = find_broken_links(tmp_path)

    assert len(broken) == 1
    assert broken[0].source == Path("README.md")
    assert broken[0].line == 3
    assert broken[0].destination == "docs/missing.md"
    assert broken[0].reason == "target does not exist"


def test_target_outside_repository_is_reported(tmp_path: Path) -> None:
    (tmp_path.parent / "outside.md").write_text("# Outside\n", encoding="utf-8")
    source = tmp_path / "README.md"
    source.write_text("[Outside](../outside.md)\n", encoding="utf-8")

    broken = find_broken_links(tmp_path)

    assert len(broken) == 1
    assert broken[0].reason == "target resolves outside the repository"


def test_external_and_fragment_only_links_are_ignored(tmp_path: Path) -> None:
    source = tmp_path / "README.md"
    source.write_text(
        "[site](https://example.com/page) [section](#results)\n", encoding="utf-8"
    )

    assert find_broken_links(tmp_path) == []


def test_code_examples_are_not_treated_as_document_links(tmp_path: Path) -> None:
    source = tmp_path / "README.md"
    source.write_text(
        "```markdown\n[example](not-a-real-file.md)\n```\n",
        encoding="utf-8",
    )

    assert find_broken_links(tmp_path) == []


def test_image_targets_are_checked(tmp_path: Path) -> None:
    source = tmp_path / "README.md"
    source.write_text("![Plot](figures/missing.png)\n", encoding="utf-8")

    broken = find_broken_links(tmp_path)

    assert len(broken) == 1
    assert broken[0].destination == "figures/missing.png"
