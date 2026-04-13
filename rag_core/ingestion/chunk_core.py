"""
chunk_core.py

Creates retrievable text chunks from parsed PDF pages.

Input (from parse_core.py):
[
  {
    "filename": "...",
    "page_number": 1,
    "text": "...",
    "section_header": "..."
  },
  ...
]

Output format:
[
  {
    "chunk_id": "...",
    "filename": "...",
    "page_number": ...,
    "section_header": "...",
    "text": "chunk text..."
  },
  ...
]

Supports:
- page-based chunking
- fixed-size chunking with overlap
- adaptive paragraph-aware chunking
"""

from __future__ import annotations
from typing import List, Dict, Optional
from pathlib import Path
import uuid
import logging

logger = logging.getLogger(__name__)


# ======================================
# Helper: create a globally unique chunk ID
# ======================================
def create_chunk_id(filename: str, page: int, index: int) -> str:
    return f"{Path(filename).stem}_p{page}_c{index}_{uuid.uuid4().hex[:8]}"


# ======================================
# Strategy A: Page-based chunking
# ======================================
def chunk_page_based(parsed_pages: List[Dict]) -> List[Dict]:
    chunks = []
    for page in parsed_pages:
        chunk = {
            "chunk_id": create_chunk_id(page["filename"], page["page_number"], 0),
            "filename": page["filename"],
            "page_number": page["page_number"],
            "section_header": page.get("section_header"),
            "text": page["text"],
        }
        chunks.append(chunk)
    return chunks


# ======================================
# Strategy B: Fixed-size chunking with overlap
# ======================================
def chunk_fixed_size(
    parsed_pages: List[Dict],
    chunk_size: int = 300,
    overlap: int = 50,
) -> List[Dict]:
    """
    Splits each page into overlapping chunks.

    Example config:
        chunk_size = 300 tokens (or characters)
        overlap = 50
    """
    chunks = []

    for page in parsed_pages:
        text = page["text"]
        n = len(text)

        if n == 0:
            continue

        start = 0
        index = 0

        while start < n:
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            chunk = {
                "chunk_id": create_chunk_id(page["filename"], page["page_number"], index),
                "filename": page["filename"],
                "page_number": page["page_number"],
                "section_header": page.get("section_header"),
                "text": chunk_text,
            }

            chunks.append(chunk)

            index += 1
            start = end - overlap  # move window with overlap

            if start < 0:
                start = 0

    return chunks


# ======================================
# Strategy C: Paragraph-aware chunking
# ======================================
def chunk_paragraph_based(
    parsed_pages: List[Dict],
    max_chunk_chars: int = 1000,
) -> List[Dict]:
    """
    Splits pages based on paragraph boundaries, up to max_chunk_chars.
    Produces cleaner chunks than fixed-size.
    """

    chunks = []

    for page in parsed_pages:
        paragraphs = [p.strip() for p in page["text"].split("\n\n") if p.strip()]
        if not paragraphs:
            continue

        buffer = ""
        index = 0

        for p in paragraphs:
            # If adding the paragraph exceeds limit → flush buffer
            if len(buffer) + len(p) > max_chunk_chars:
                chunk = {
                    "chunk_id": create_chunk_id(page["filename"], page["page_number"], index),
                    "filename": page["filename"],
                    "page_number": page["page_number"],
                    "section_header": page.get("section_header"),
                    "text": buffer.strip(),
                }
                chunks.append(chunk)
                index += 1
                buffer = p
            else:
                buffer += ("\n\n" + p)

        # Flush remainder
        if buffer:
            chunk = {
                "chunk_id": create_chunk_id(page["filename"], page["page_number"], index),
                "filename": page["filename"],
                "page_number": page["page_number"],
                "section_header": page.get("section_header"),
                "text": buffer.strip(),
            }
            chunks.append(chunk)

    return chunks


# ======================================
# Public API: chunk_pages
# ======================================
def chunk_pages(
    parsed_pages: List[Dict],
    strategy: str = "fixed",
    chunk_size: int = 300,
    overlap: int = 50,
    max_chunk_chars: int = 1000,
) -> List[Dict]:
    """
    Unified chunking interface.

    strategy:
        "page"  → one chunk per page
        "fixed" → fixed-size chunks with overlap
        "paragraph" → chunk by paragraph boundaries

    Returns list of chunk dicts.
    """

    if strategy == "page":
        logger.info("[CHUNK] Using page-based chunking.")
        return chunk_page_based(parsed_pages)

    elif strategy == "fixed":
        logger.info(
            f"[CHUNK] Using fixed-size chunking (chunk_size={chunk_size}, overlap={overlap})."
        )
        return chunk_fixed_size(parsed_pages, chunk_size, overlap)

    elif strategy == "paragraph":
        logger.info(
            f"[CHUNK] Using paragraph-based chunking (max_chunk_chars={max_chunk_chars})."
        )
        return chunk_paragraph_based(parsed_pages, max_chunk_chars)

    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
