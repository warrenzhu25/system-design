#!/usr/bin/env python3
"""Convert the repository's source PDFs to text-only Markdown files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Callable, Iterable


Extractor = Callable[[Path], str]

MARKDOWN_IMAGE = re.compile(
    r"!?\[[^\]]*\]\([^\n)]*\.(?:avif|gif|jpe?g|png|svg|webp)[^\n)]*\)",
    re.I,
)
HTML_IMAGE = re.compile(r"<img\b[^>]*>", re.I | re.S)
EXCESS_BLANK_LINES = re.compile(r"\n{3,}")
FIGURE_CAPTION = re.compile(r"^\s*(?:figure|fig[.])\s*\d", re.I)


def clean_markdown(markdown: str) -> str:
    """Remove visual embeds and normalize common PDF extraction artifacts."""
    markdown = markdown.replace("\r\n", "\n").replace("\r", "\n")
    markdown = markdown.replace("\x00", "").replace("\ufeff", "")
    markdown = markdown.replace("\u00a0", " ").replace("\u200b", "")
    markdown = markdown.replace("\f", "\n\n")
    markdown = MARKDOWN_IMAGE.sub("", markdown)
    markdown = HTML_IMAGE.sub("", markdown)
    markdown = "\n".join(line.rstrip() for line in markdown.splitlines())
    markdown = EXCESS_BLANK_LINES.sub("\n\n", markdown).strip()
    return f"{markdown}\n" if markdown else ""


def _horizontal_overlap(first, second) -> float:
    overlap = max(0.0, min(first.x1, second.x1) - max(first.x0, second.x0))
    narrower_width = min(first.width, second.width)
    return overlap / narrower_width if narrower_width else 0.0


def find_figure_regions(page) -> list:
    """Locate vector or raster visual regions adjacent to figure captions."""
    captions = []
    for block in page.get_text("blocks", sort=True):
        text = " ".join(block[4].split())
        if FIGURE_CAPTION.match(text):
            captions.append(page.rect.__class__(block[:4]))

    if not captions:
        return []

    visual_regions = list(page.cluster_drawings())
    for image in page.get_image_info(xrefs=True):
        visual_regions.append(page.rect.__class__(image["bbox"]))

    page_area = page.rect.width * page.rect.height
    figure_regions = []
    for caption in captions:
        for region in visual_regions:
            region_area = region.width * region.height
            if region_area < 100 or region_area > page_area * 0.75:
                continue
            if _horizontal_overlap(caption, region) < 0.2:
                continue

            gap_above = caption.y0 - region.y1
            gap_below = region.y0 - caption.y1
            if -2 <= gap_above <= 60 or -2 <= gap_below <= 60:
                figure_regions.append(region)

    return figure_regions


def suppress_figure_text(document) -> None:
    """Redact text labels inside detected visuals, leaving captions untouched."""
    for page in document:
        regions = find_figure_regions(page)
        for region in regions:
            page.add_redact_annot(region, fill=False, cross_out=False)
        if regions:
            page.apply_redactions(images=0, graphics=0, text=0)


def extract_text_markdown(source: Path) -> str:
    """Extract reading-order-aware Markdown while suppressing visual objects."""
    try:
        import pymupdf
        import pymupdf4llm
    except ImportError as error:
        raise RuntimeError(
            "pymupdf4llm is required; install it with "
            "`python3 -m pip install pymupdf4llm`."
        ) from error

    document = pymupdf.open(source)
    try:
        suppress_figure_text(document)
        return pymupdf4llm.to_markdown(
            document,
            write_images=False,
            embed_images=False,
            ignore_images=True,
            ignore_graphics=True,
            force_text=True,
            page_chunks=False,
            page_separators=False,
            show_progress=False,
        )
    finally:
        document.close()


def add_source_note(markdown: str, source_name: str) -> str:
    """Add the conversion notice immediately after the document title."""
    note = (
        "> Text-only Markdown conversion of the "
        f"[source PDF](../pdf/{source_name}). Images and diagrams are omitted; "
        "the PDF remains authoritative."
    )
    lines = markdown.rstrip().splitlines()
    if lines and lines[0].startswith("#"):
        return "\n".join([lines[0], "", note, "", *lines[1:]]).strip() + "\n"
    return f"{note}\n\n{markdown.rstrip()}\n"


def convert_pdf(
    source: Path,
    destination: Path,
    extractor: Extractor = extract_text_markdown,
) -> None:
    """Convert one PDF and write its cleaned Markdown output."""
    extracted = extractor(source)
    markdown = clean_markdown(extracted)
    markdown = add_source_note(markdown, source.name)
    markdown = clean_markdown(markdown)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(markdown, encoding="utf-8")


def find_pdfs(source_directory: Path) -> Iterable[Path]:
    """Return source PDFs in stable filename order."""
    return sorted(source_directory.glob("*.pdf"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("pdf"))
    parser.add_argument("--output", type=Path, default=Path("original"))
    parser.add_argument("paths", nargs="*", type=Path, help="Optional PDFs to convert")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = args.paths or list(find_pdfs(args.source))
    if not sources:
        raise SystemExit(f"No PDF files found in {args.source}")

    for source in sources:
        destination = args.output / f"{source.stem}.md"
        convert_pdf(source, destination)
        print(f"Converted {source} -> {destination}")


if __name__ == "__main__":
    main()
