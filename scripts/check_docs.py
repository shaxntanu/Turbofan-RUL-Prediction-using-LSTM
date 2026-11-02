"""Check local file targets in Markdown links and images."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urlsplit


LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\((<[^>]*>|(?:\\.|[^)])*)\)")
FENCED_CODE_PATTERN = re.compile(
    r"(?ms)^[ \t]{0,3}(`{3,}|~{3,})[^\n]*\n.*?^[ \t]{0,3}\1[ \t]*$"
)
INLINE_CODE_PATTERN = re.compile(r"(?<!`)`+[^`\n]*`+(?!`)")
EXCLUDED_DIRS = {
    ".git",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "models",
    "node_modules",
    "outputs",
    "tmp",
}


@dataclass(frozen=True)
class BrokenLink:
    source: Path
    line: int
    destination: str
    reason: str


def markdown_files(root: Path) -> list[Path]:
    """Return repository Markdown files, excluding generated and environment trees."""
    return sorted(
        path
        for path in root.rglob("*.md")
        if not any(part in EXCLUDED_DIRS for part in path.relative_to(root).parts)
    )


def _link_destination(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("<") and ">" in raw:
        return raw[1 : raw.index(">")]
    token = re.match(r"(?:\\.|[^\s])+", raw)
    return token.group(0) if token else ""


def find_broken_links(
    root: Path, files: Iterable[Path] | None = None
) -> list[BrokenLink]:
    """Find missing local file targets in inline Markdown links and images.

    External URLs and fragment-only links are intentionally not checked. Heading
    fragment spelling is renderer-specific and needs a separate Markdown parser.
    """
    root = root.resolve()
    candidates = list(files) if files is not None else markdown_files(root)
    broken: list[BrokenLink] = []

    for source in candidates:
        source = source if source.is_absolute() else root / source
        text = source.read_text(encoding="utf-8-sig")
        text = FENCED_CODE_PATTERN.sub(lambda match: "\n" * match.group(0).count("\n"), text)
        text = INLINE_CODE_PATTERN.sub("", text)

        for match in LINK_PATTERN.finditer(text):
            destination = _link_destination(match.group(1))
            parsed = urlsplit(destination)
            if not destination or parsed.scheme or not parsed.path:
                continue

            target = (source.parent / unquote(parsed.path)).resolve()
            try:
                target.relative_to(root)
            except ValueError:
                line = text.count("\n", 0, match.start()) + 1
                broken.append(
                    BrokenLink(
                        source=source.relative_to(root),
                        line=line,
                        destination=destination,
                        reason="target resolves outside the repository",
                    )
                )
                continue
            if not target.exists():
                line = text.count("\n", 0, match.start()) + 1
                broken.append(
                    BrokenLink(
                        source=source.relative_to(root),
                        line=line,
                        destination=destination,
                        reason="target does not exist",
                    )
                )

    return broken


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    files = markdown_files(root)
    broken = find_broken_links(root, files)
    if broken:
        for item in broken:
            print(f"{item.source}:{item.line}: {item.reason}: {item.destination}")
        print(f"Found {len(broken)} broken local Markdown target(s) in {len(files)} files.")
        return 1

    print(f"Checked local Markdown targets in {len(files)} files. No missing targets found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
