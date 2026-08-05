#!/usr/bin/env python3
"""Merge the converted paper Markdown files into one volume."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)
MARKDOWN_FORMATTING = re.compile(r"[*_`\[\]]")
NON_SLUG_CHARACTER = re.compile(r"[^a-z0-9 -]")

DEFAULT_TITLE = "Database Systems Papers"
DEFAULT_SUBTITLE = (
    "Combined text-only Markdown volume. Images and diagrams are omitted; "
    "the source PDFs remain authoritative."
)


def paper_title(markdown: str, source: Path) -> str:
    """Return a readable title from the first heading or filename."""
    match = HEADING.search(markdown)
    if not match:
        return source.stem.replace("-", " ").title()
    return MARKDOWN_FORMATTING.sub("", match.group(1)).strip()


def slugify(value: str) -> str:
    """Create a stable GitHub-style fragment for a paper boundary."""
    value = NON_SLUG_CHARACTER.sub("", value.lower())
    return re.sub(r"[ -]+", "-", value).strip("-")


def merge_markdown(
    source_directory: Path,
    output: Path,
    title: str = DEFAULT_TITLE,
    subtitle: str = DEFAULT_SUBTITLE,
) -> None:
    """Merge source Markdown files in filename order, excluding the output."""
    output_path = output.resolve()
    sources = [
        path
        for path in sorted(source_directory.glob("*.md"))
        if path.resolve() != output_path
    ]
    if not sources:
        raise ValueError(f"No Markdown files found in {source_directory}")

    papers = []
    for source in sources:
        markdown = source.read_text(encoding="utf-8").strip()
        heading = paper_title(markdown, source)
        papers.append((heading, markdown))

    contents = [
        f"- [{title}](#paper-{slugify(title)})"
        for title, _ in papers
    ]
    sections = [
        f"## Paper: {title}\n\n{markdown}"
        for title, markdown in papers
    ]
    merged = "\n".join(
        [
            f"# {title}",
            "",
            f"> {subtitle}",
            "",
            "## Contents",
            "",
            *contents,
            "",
            "---",
            "",
            "\n\n---\n\n".join(sections),
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(merged, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("original"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("original/all-papers.md"),
    )
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--subtitle", default=DEFAULT_SUBTITLE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge_markdown(args.source, args.output, args.title, args.subtitle)
    print(f"Merged Markdown papers into {args.output}")


if __name__ == "__main__":
    main()
