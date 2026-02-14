#!/usr/bin/env python3
"""Convert a PDF file to Markdown, preserving tables."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import pymupdf4llm
from pypdf import PdfReader, PdfWriter


def log(message, **kwargs):
    print(f"[INFO] {message}", *(f"{k}={v}" for k, v in kwargs.items()))


def convert_pdf_to_markdown(pdf_path: str, output_path: str | None = None, chunk_size: int = 20) -> str:
    """Convert a PDF to Markdown and return the text.

    Args:
        pdf_path: Path to the input PDF file.
        output_path: Optional path for the output .md file.
                     Defaults to the same name/location as the PDF with a .md extension.
        chunk_size: Number of pages per processing chunk (default 20).

    Returns:
        The generated Markdown text.
    """
    if not os.path.isfile(pdf_path):
        print(f"Error: file not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    if output_path is None:
        output_path = os.path.splitext(pdf_path)[0] + ".md"

    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)
    log("Opened document", path=pdf_path, pages=num_pages)

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
    log("Conversion complete", elapsed=f"{elapsed:.1f}s", pages=num_pages)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    log("Saved markdown", output=output_path)
    return md_text


def main():
    parser = argparse.ArgumentParser(description="Convert a PDF to Markdown (preserves tables).")
    parser.add_argument("pdf", help="Path to the input PDF file.")
    parser.add_argument("-o", "--output", default=None, help="Output .md file path (default: <input>.md).")
    parser.add_argument("--chunk-size", type=int, default=20, help="Pages per processing chunk (default: 20).")
    args = parser.parse_args()

    convert_pdf_to_markdown(args.pdf, args.output, args.chunk_size)


if __name__ == "__main__":
    main()
