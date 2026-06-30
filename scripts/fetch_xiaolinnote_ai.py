#!/usr/bin/env python3
"""Archive a xiaolinnote.com content area into markdown.

Fetches every article under a given root (e.g. /ai/ or /agent/), isolates the
main content, strips site chrome, converts HTML -> GFM via pandoc, and writes
one consolidated markdown file per subsection into an output dir. Images are
kept as remote cdn.xiaolincoding.com links.

Usage:
    python3 fetch_xiaolinnote_ai.py [ROOT] [OUT_DIR]
        ROOT     site path to archive, default "/ai/"
        OUT_DIR  output folder, default "xiaolinnote_<root>" next to the repo

Stdlib only (urllib/re/subprocess) plus the pandoc CLI.
"""

import os
import re
import subprocess
import sys
import time
import urllib.request

BASE = "https://xiaolinnote.com"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# Optional pretty section titles; any subsection not listed falls back to its slug.
SECTION_TITLES = {
    "agent": "Agent 面试题", "llm": "LLM 大模型面试题",
    "rag": "RAG 面试题", "tools": "LLM Tool Calling / MCP 面试题",
    "concept": "概念", "engineering": "工程实践",
}


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def article_links(index_html, root):
    """Return {section: [paths in document order]} for <root><section>/*.html.

    Sections (subdirectories) are discovered in the order they first appear.
    Within a section the *_info.html landing page is placed first.
    """
    links = {}
    seen = set()
    pat = re.escape(root) + r'([a-z0-9_-]+)/[^"]+\.html'
    for m in re.finditer(r'href="(' + pat + r')"', index_html):
        path, section = m.group(1), m.group(2)
        if path in seen:
            continue
        seen.add(path)
        links.setdefault(section, []).append(path)
    for section in links:
        info = [p for p in links[section] if p.endswith("_info.html")]
        rest = [p for p in links[section] if not p.endswith("_info.html")]
        links[section] = info + rest
    return links


def isolate_main(html):
    """Extract the article body (<div id="markdown-content">) and strip chrome."""
    start = html.find('id="markdown-content"')
    if start == -1:
        m = re.search(r'<main[^>]*id="main-content".*?</main>', html, re.S)
        body = m.group(0) if m else html
    else:
        body = html[html.rfind("<div", 0, start):]
        # Cut everything from the first footer / prev-next nav / comment widget
        # (these are siblings that follow the article body).
        for marker in ('<nav class="page-nav"', "<footer", '<div class="page-comment"',
                       "giscus", '<div class="contributors"'):
            i = body.find(marker)
            if i != -1:
                body = body[:i]

    # Drop scripts and inline base64 data-URI images (decorative SVG icons).
    body = re.sub(r'<script.*?</script>', '', body, flags=re.S)
    body = re.sub(r'<img[^>]*src="data:[^"]*"[^>]*/?>', '', body, flags=re.S)
    # Unwrap header-anchor links so headings stay plain text, not links.
    body = re.sub(r'<a [^>]*header-anchor[^>]*>(.*?)</a>', r'\1', body, flags=re.S)
    return body


def to_markdown(html_fragment):
    proc = subprocess.run(
        ["pandoc", "-f", "html", "-t", "gfm", "--wrap=none"],
        input=html_fragment.encode("utf-8"),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    md = proc.stdout.decode("utf-8", errors="replace")
    out = []
    for ln in md.splitlines():
        s = ln.strip()
        # Drop pandoc's leftover raw wrapper tags (div/span on their own line).
        if re.fullmatch(r'</?(div|span)(\s[^>]*)?>', s):
            continue
        if "](data:" in ln or 'src="data:' in ln:  # residual base64 images
            continue
        # Normalize shiki-highlighted fences to plain fenced code.
        ln = re.sub(r'^(\s*```)\s*shiki\s*$', r'\1', ln)
        out.append(ln)
    md = re.sub(r'\n{3,}', '\n\n', "\n".join(out)).strip()
    return md


def assemble_article(md, url):
    """Demote every heading one level (so the article title sits under the
    section H1), skipping '#' lines inside fenced code, and inject a source line."""
    lines, in_fence, injected = [], False, False
    for ln in md.splitlines():
        if ln.lstrip().startswith("```"):
            in_fence = not in_fence
            lines.append(ln)
            continue
        m = re.match(r'^(#{1,6})(\s+.*)$', ln)
        if m and not in_fence:
            ln = "#" + m.group(1) + m.group(2)  # demote one level
            lines.append(ln)
            if not injected:
                lines.append(f"\n> Source: {url}")
                injected = True
            continue
        lines.append(ln)
    return "\n".join(lines)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/ai/"
    if not root.startswith("/"):
        root = "/" + root
    if not root.endswith("/"):
        root += "/"
    default_out = os.path.join(REPO, "xiaolinnote_" + root.strip("/").replace("/", "_"))
    out_dir = sys.argv[2] if len(sys.argv) > 2 else default_out
    index = BASE + root

    os.makedirs(out_dir, exist_ok=True)
    print(f"Fetching index {index}")
    links = article_links(fetch(index), root)
    total = sum(len(v) for v in links.values())
    print("Links per section:", {s: len(v) for s, v in links.items()}, "total", total)

    ok = 0
    for section, paths in links.items():
        title = SECTION_TITLES.get(section, section.replace("-", " ").title())
        parts = [f"# {title}\n\n> Archived from {index} ({section}). Personal study copy.\n"]
        for path in paths:
            url = BASE + path
            try:
                html = fetch(url)
            except Exception as e:  # noqa: BLE001
                print(f"  FAIL {url}: {e}", file=sys.stderr)
                continue
            md = to_markdown(isolate_main(html))
            parts.append(assemble_article(md, url))
            ok += 1
            print(f"  ok  {url}  ({len(md)//1024} KB)")
            time.sleep(0.5)
        out = os.path.join(out_dir, f"{section}.md")
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n\n".join(parts) + "\n")
        print(f"Wrote {out} ({os.path.getsize(out)//1024} KB)")

    print(f"Done: {ok}/{total} pages fetched.")


if __name__ == "__main__":
    main()
