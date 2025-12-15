"""
parse_core.py

Provides PDF → structured text parsing using PyMuPDF (fitz).
Extracts:
- Page-level text
- Optional section headers (based on font size heuristics)
- Metadata for downstream chunking

Output Format:
[
    {
        "filename": "...",
        "page_number": 1,
        "text": "...",
        "section_header": "optional"
    },
    ...
]
"""

import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


# ==========================
# Section Header Heuristic
# ==========================
def extract_section_header(blocks: List[dict]) -> Optional[str]:
    """
    Heuristic: Find the line of text with the largest font size on the page.
    Often corresponds to section title / header.
    """
    max_size = 0
    header_text = None

    for block in blocks:
        if block["type"] != 0:  # ignore images etc
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                size = span.get("size", 0)
                text = span.get("text", "").strip()

                if size > max_size and len(text) > 3:  # Filter tiny fragments
                    max_size = size
                    header_text = text

    return header_text


# ==========================
# Core Function: parse_pdf
# ==========================
def download_parse_pdf_naive(url) -> str:
    # download
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    
    pdf_content = response.content # type: bytes
    
    # Validate that the content is a PDF
    if not pdf_content.startswith(b'%PDF'):
        raise ValueError(
            f"URL does not point to a valid PDF file. "
            f"Content type: {response.headers.get('content-type', 'unknown')}. "
            f"Please provide a direct link to a PDF document."
        )
    
    # parse
    from pypdf import PdfReader
    from io import BytesIO
    reader = PdfReader(BytesIO(pdf_content))
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text
    
def parse_pdf(file_path: str) -> List[Dict]:
    """
    Parse a PDF into structured page-level records.
    
    Returns list of dicts:
    [
        {
            "filename": "...",
            "page_number": 1,
            "text": "...",
            "section_header": "optional"
        },
        ...
    ]
    """
    file_path = Path(file_path)

    logger.info(f"[PARSE] Parsing PDF: {file_path}")

    try:
        doc = fitz.open(file_path)
    except Exception as e:
        logger.error(f"[ERROR] Could not open PDF: {file_path}. Reason: {e}")
        return []

    results = []

    for page_index, page in enumerate(doc):
        page_number = page_index + 1

        # Extract text (simplified)
        text = page.get_text("text")

        # Extract layout blocks (if needed for headers)
        blocks = page.get_text("dict").get("blocks", [])

        # Heuristic extraction of section header
        section_header = extract_section_header(blocks)

        # Clean text
        cleaned = text.replace("\x00", "").strip()

        # Append structured page record
        results.append(
            {
                "filename": file_path.name,
                "page_number": page_number,
                "text": cleaned,
                "section_header": section_header,
            }
        )

    logger.info(f"[PARSE] Completed parsing {file_path.name}: {len(results)} pages processed.")
    return results


# ==========================
# Public API: process_pdf
# ==========================
def process_pdf(file_path: str) -> Optional[List[dict]]:
    """
    Wrapper that checks file existence and logs cleanly.
    """
    path = Path(file_path)
    if not path.exists():
        logger.error(f"[ERROR] PDF not found: {file_path}")
        return None

    return parse_pdf(file_path)
