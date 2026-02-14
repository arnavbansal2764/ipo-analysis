#!/usr/bin/env python3
"""
Convert IPO metadata JSONs to well-structured Markdown files for RAG,
convert RHP PDFs to Markdown using pymupdf4llm, and extract URLs into
a separate metadata JSON per IPO.

Output per IPO folder:
    ipo_data/<folder>/
    ├── metadata.json               # Original scraped data (untouched)
    ├── <ipo-name>.md               # Metadata converted to structured markdown
    ├── <ipo-name>.metadata.json    # Extracted URLs and identifiers
    └── rhp_docs/
        ├── *.pdf                   # Original PDFs (untouched)
        └── *.md                    # PDF → Markdown via pymupdf4llm

Usage:
    python convert_to_markdown.py                     # Process all IPOs
    python convert_to_markdown.py --ipo <folder>      # Single IPO folder
    python convert_to_markdown.py --skip-pdf           # Skip PDF conversion
    python convert_to_markdown.py --force              # Re-convert even if .md exists
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import pymupdf4llm
from pypdf import PdfReader, PdfWriter


# --- Configuration ---
BASE_OUTPUT_DIR = "./ipo_data"


# ============================================================
# Helpers
# ============================================================

def log(msg: str, **kw):
    extra = " ".join(f"{k}={v}" for k, v in kw.items())
    print(f"  [INFO] {msg} {extra}".rstrip())


def slugify(name: str) -> str:
    """Turn an IPO name into a clean filename slug."""
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"[\s]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s


# ============================================================
# Part 1 – Metadata JSON → Markdown
# ============================================================

def _fmt(v) -> str:
    """Format a value for markdown, escaping pipe characters."""
    if v is None or v == "":
        return "N/A"
    if isinstance(v, bool):
        return "Yes" if v else "No"
    return str(v).replace("|", "\\|")


def _dict_to_table(d: dict) -> str:
    """Flat dict → 2-column markdown table."""
    if not d:
        return ""
    lines = ["| Field | Value |", "|---|---|"]
    for k, v in d.items():
        lines.append(f"| {_fmt(k)} | {_fmt(v)} |")
    return "\n".join(lines)


def _rows_to_table(items: list) -> str:
    """List of dicts (or list of lists) → markdown table."""
    if not items:
        return ""

    # Separate dict rows from bare-list footnotes (e.g. ["Amount in ₹ Crore"])
    dict_items = [i for i in items if isinstance(i, dict)]
    list_items = [i for i in items if isinstance(i, list)]

    footnotes = ""
    if list_items:
        footnotes = "\n\n" + "\n".join(
            "*" + " | ".join(str(c) for c in row) + "*" for row in list_items
        )

    if dict_items:
        headers = list(dict_items[0].keys())
        lines = ["| " + " | ".join(_fmt(h) for h in headers) + " |"]
        lines.append("|" + "---|" * len(headers))
        for item in dict_items:
            lines.append("| " + " | ".join(_fmt(item.get(h)) for h in headers) + " |")
        return "\n".join(lines) + footnotes

    # Pure list-of-lists
    if list_items:
        cols = max(len(r) for r in list_items)
        lines = []
        for i, row in enumerate(list_items):
            lines.append("| " + " | ".join(_fmt(c) for c in row) + " |")
            if i == 0:
                lines.append("|" + "---|" * cols)
        return "\n".join(lines)

    return ""


def metadata_to_markdown(meta: dict) -> str:
    """Convert the metadata dict to a well-structured Markdown string."""
    parts: list[str] = []

    name = meta.get("name", "Unknown IPO")

    # --- YAML Frontmatter ---
    parts.append("---")
    parts.append(f'ipo_id: "{meta.get("id", "")}"')
    parts.append(f'company_name: "{name}"')
    parts.append(f'open_date: "{meta.get("open_date", "")}"')
    parts.append(f'close_date: "{meta.get("close_date", "")}"')
    parts.append(f'ipo_period: "{meta.get("ipo_period", "")}"')
    parts.append(f'is_open: {str(meta.get("is_open", False)).lower()}')
    parts.append(f'source: "chittorgarh"')
    parts.append(f'doc_type: "ipo_metadata"')
    parts.append("---\n")

    # --- Title ---
    parts.append(f"# {name}\n")

    # --- Summary Cards ---
    cards = meta.get("summary_cards", {})
    if cards:
        parts.append("## Summary\n")
        parts.append(_dict_to_table(cards) + "\n")

    # --- IPO Details ---
    details = meta.get("ipo_details", {})
    if details:
        parts.append("## IPO Details\n")
        parts.append(_dict_to_table(details) + "\n")

    # --- Timetable ---
    tt = meta.get("timetable", {})
    if tt:
        parts.append("## Timetable\n")
        parts.append(_dict_to_table(tt) + "\n")

    # --- Reservation ---
    res = meta.get("reservation", {})
    if res:
        parts.append("## Reservation\n")
        if isinstance(res, dict):
            s = res.get("summary", "")
            if s:
                parts.append(f"{s}\n")
            t = res.get("table", [])
            if t:
                parts.append(_rows_to_table(t) + "\n")
        elif isinstance(res, list):
            parts.append(_rows_to_table(res) + "\n")

    # --- Lot Size ---
    ls = meta.get("lot_size", [])
    if ls:
        parts.append("## Lot Size\n")
        if isinstance(ls, list):
            parts.append(_rows_to_table(ls) + "\n")
        elif isinstance(ls, dict):
            parts.append(_dict_to_table(ls) + "\n")

    # --- Financials ---
    fin = meta.get("financials", [])
    if fin:
        parts.append("## Financials\n")
        if isinstance(fin, list):
            parts.append(_rows_to_table(fin) + "\n")
        elif isinstance(fin, dict):
            parts.append(_dict_to_table(fin) + "\n")

    # --- KPIs ---
    kpis = meta.get("kpis", {})
    if kpis and isinstance(kpis, dict):
        parts.append("## Key Performance Indicators\n")
        for tbl_name, tbl_data in kpis.items():
            if isinstance(tbl_data, list):
                parts.append(f"### {tbl_name.replace('_', ' ').title()}\n")
                parts.append(_rows_to_table(tbl_data) + "\n")

    # --- Subscription Details ---
    sub = meta.get("subscription_details", {})
    if sub:
        parts.append("## Subscription Status\n")
        if isinstance(sub, dict):
            s = sub.get("summary", "")
            if s:
                if isinstance(s, dict):
                    parts.append(_dict_to_table(s) + "\n")
                else:
                    parts.append(f"{s}\n")
            t = sub.get("table", [])
            if t:
                parts.append(_rows_to_table(t) + "\n")

    # --- Objects of Issue ---
    obj = meta.get("objects_of_issue", [])
    if obj:
        parts.append("## Objects of the Issue\n")
        if isinstance(obj, list):
            parts.append(_rows_to_table(obj) + "\n")

    # --- Peer Comparison ---
    pc = meta.get("peer_comparison", [])
    if pc:
        parts.append("## Peer Comparison\n")
        if isinstance(pc, list):
            parts.append(_rows_to_table(pc) + "\n")

    # --- Recommendations ---
    rec = meta.get("recommendations", [])
    if rec:
        parts.append("## Recommendations\n")
        if isinstance(rec, list):
            parts.append(_rows_to_table(rec) + "\n")

    # --- Promoter Info ---
    pi = meta.get("promoter_info", "")
    if pi:
        parts.append("## Promoter Information\n")
        parts.append(f"{pi}\n")

    return "\n".join(parts)


# ============================================================
# Part 2 – Extract URLs → <ipo-name>.metadata.json
# ============================================================

_URL_RE = re.compile(r'https?://[^\s"\'<>]+')


def extract_urls_metadata(meta: dict) -> dict:
    """Extract all URLs and key identifiers from metadata into a compact dict."""
    urls: dict[str, str | list[str]] = {}

    # Explicitly known URL keys
    if meta.get("rhp_link"):
        urls["rhp_link"] = meta["rhp_link"]
    if meta.get("gmp_link"):
        urls["gmp_link"] = meta["gmp_link"]
    if meta.get("rhp_local_paths"):
        urls["rhp_local_paths"] = meta["rhp_local_paths"]

    # Scan all string values for any other embedded URLs
    other_urls: list[str] = []
    _scan_for_urls(meta, other_urls, skip_keys={"rhp_link", "gmp_link", "rhp_local_paths"})
    if other_urls:
        urls["other_urls"] = sorted(set(other_urls))

    return {
        "id": meta.get("id", ""),
        "name": meta.get("name", ""),
        "folder": meta.get("folder", ""),
        "ipo_period": meta.get("ipo_period", ""),
        "open_date": meta.get("open_date", ""),
        "close_date": meta.get("close_date", ""),
        "order_date": meta.get("order_date", ""),
        "is_open": meta.get("is_open", False),
        "urls": urls,
    }


def _scan_for_urls(obj, found: list[str], skip_keys: set | None = None, _current_key: str = ""):
    """Recursively scan a JSON-like object for URL strings."""
    if skip_keys and _current_key in skip_keys:
        return
    if isinstance(obj, str):
        for m in _URL_RE.finditer(obj):
            found.append(m.group(0))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _scan_for_urls(v, found, skip_keys, _current_key=k)
    elif isinstance(obj, list):
        for item in obj:
            _scan_for_urls(item, found, skip_keys, _current_key=_current_key)


# ============================================================
# Part 3 – PDF → Markdown (reuses pdf_to_markdown.py logic)
# ============================================================

def convert_pdf_to_markdown(pdf_path: str, output_path: str, chunk_size: int = 20) -> str:
    """Convert a PDF to markdown using pymupdf4llm with chunked processing."""
    if not os.path.isfile(pdf_path):
        log(f"PDF not found, skipping: {pdf_path}")
        return ""

    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)
    log("Converting PDF", path=os.path.basename(pdf_path), pages=num_pages)

    start_time = time.time()

    if num_pages <= chunk_size:
        md_text = pymupdf4llm.to_markdown(pdf_path)
        log("Converted in single chunk", pages=num_pages)
    else:
        md_parts: list[str] = []
        for start in range(0, num_pages, chunk_size):
            end = min(start + chunk_size, num_pages)
            temp_path = f"/tmp/_chunk_{start + 1}_{end}.pdf"

            writer = PdfWriter()
            for i in range(start, end):
                writer.add_page(reader.pages[i])
            with open(temp_path, "wb") as f:
                writer.write(f)

            log("Processing pages", start=start + 1, end=end, total=num_pages)
            part_md = pymupdf4llm.to_markdown(temp_path)
            md_parts.append(part_md)

            try:
                os.remove(temp_path)
            except OSError:
                pass

        md_text = "\n\n".join(md_parts)

    elapsed = time.time() - start_time
    log("PDF conversion complete", elapsed=f"{elapsed:.1f}s", pages=num_pages)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    log("Saved PDF markdown", output=os.path.basename(output_path))
    return md_text


# ============================================================
# Part 4 – Orchestration
# ============================================================

def process_single_ipo(ipo_folder: str, skip_pdf: bool = False, force: bool = False) -> bool:
    """Process one IPO folder:
    1. metadata.json → <ipo-name>.md
    2. metadata.json → <ipo-name>.metadata.json  (URLs only)
    3. rhp_docs/*.pdf → rhp_docs/*.md              (via pymupdf4llm)
    """
    folder_path = Path(ipo_folder)
    if not folder_path.is_dir():
        print(f"[SKIP] Not a directory: {ipo_folder}")
        return False

    metadata_file = folder_path / "metadata.json"
    if not metadata_file.exists():
        print(f"[SKIP] No metadata.json in {folder_path.name}")
        return False

    with open(metadata_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    ipo_name = meta.get("name", folder_path.name)
    slug = slugify(ipo_name)

    print(f"\n{'='*60}")
    print(f"  {ipo_name}")
    print(f"{'='*60}")

    # --- 1. Metadata → Markdown ---
    md_path = folder_path / f"{slug}.md"
    if md_path.exists() and not force:
        print(f"  [SKIP] Markdown exists: {md_path.name}")
    else:
        md_text = metadata_to_markdown(meta)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_text)
        print(f"  [OK]   Metadata markdown → {md_path.name}")

    # --- 2. URLs → <ipo-name>.metadata.json ---
    urls_path = folder_path / f"{slug}.metadata.json"
    if urls_path.exists() and not force:
        print(f"  [SKIP] URL metadata exists: {urls_path.name}")
    else:
        urls_data = extract_urls_metadata(meta)
        with open(urls_path, "w", encoding="utf-8") as f:
            json.dump(urls_data, f, indent=2, ensure_ascii=False)
        print(f"  [OK]   URL metadata → {urls_path.name}")

    # --- 3. RHP PDFs → Markdown ---
    if skip_pdf:
        print(f"  [SKIP] PDF conversion skipped (--skip-pdf)")
    else:
        rhp_dir = folder_path / "rhp_docs"
        if rhp_dir.is_dir():
            pdfs = sorted(rhp_dir.glob("*.pdf")) + sorted(rhp_dir.glob("*.PDF"))
            if pdfs:
                for pdf in pdfs:
                    md_out = rhp_dir / (pdf.stem + ".md")
                    if md_out.exists() and not force:
                        print(f"  [SKIP] Already converted: {pdf.name} → {md_out.name}")
                        continue
                    try:
                        convert_pdf_to_markdown(str(pdf), str(md_out))
                    except Exception as e:
                        print(f"  [ERR]  Failed to convert {pdf.name}: {e}")
            else:
                print(f"  [INFO] No PDFs in rhp_docs/")
        else:
            print(f"  [INFO] No rhp_docs/ directory")

    return True


def process_all_ipos(skip_pdf: bool = False, force: bool = False):
    """Process every IPO folder under BASE_OUTPUT_DIR."""
    base = Path(BASE_OUTPUT_DIR)
    if not base.is_dir():
        print(f"[ERR] Directory not found: {BASE_OUTPUT_DIR}")
        print(f"      Run ipo_local_scraper.py first to fetch IPO data.")
        return

    folders = sorted(f for f in base.iterdir() if f.is_dir())
    if not folders:
        print(f"[ERR] No IPO folders found in {BASE_OUTPUT_DIR}")
        return

    total = len(folders)
    print(f"Found {total} IPO folders in {BASE_OUTPUT_DIR}\n")

    ok = 0
    fail = 0
    for i, folder in enumerate(folders, 1):
        try:
            print(f"[{i}/{total}]", end="")
            if process_single_ipo(str(folder), skip_pdf=skip_pdf, force=force):
                ok += 1
            else:
                fail += 1
        except Exception as e:
            print(f"  [ERR] {folder.name}: {e}")
            fail += 1

    print(f"\n{'='*60}")
    print(f"  Done — {ok} succeeded, {fail} failed out of {total}")
    print(f"{'='*60}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert IPO metadata and RHP PDFs to Markdown for RAG"
    )
    parser.add_argument(
        "--ipo", type=str,
        help="Process a single IPO folder (name or full path)"
    )
    parser.add_argument(
        "--skip-pdf", action="store_true",
        help="Skip PDF→Markdown conversion (only process metadata)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-convert files even if output already exists"
    )
    args = parser.parse_args()

    if args.ipo:
        path = args.ipo
        if not os.path.isabs(path):
            candidate = os.path.join(BASE_OUTPUT_DIR, path)
            if os.path.isdir(candidate):
                path = candidate
        process_single_ipo(path, skip_pdf=args.skip_pdf, force=args.force)
    else:
        process_all_ipos(skip_pdf=args.skip_pdf, force=args.force)


if __name__ == "__main__":
    main()
