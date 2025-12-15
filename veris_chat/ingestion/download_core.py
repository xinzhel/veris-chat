"""
download_core.py

Handles downloading PDFs from URLs with:
- Retry logic + exponential backoff
- SHA256 checksum for deduplication
- Automatic filename detection
- Validation for PDF content type
- Clean directory handling

Used by the ingestion pipeline to download PDF files before parsing/chunking.
"""

import os
import time
import hashlib
import logging
import requests
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse, unquote

# Configure logger
logger = logging.getLogger(__name__)


# ==============================
# Utility: Compute SHA256 checksum
# ==============================
def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file for deduplication."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha.update(chunk)
    return sha.hexdigest()


# ==============================
# Utility: Extract filename from URL
# ==============================
def derive_filename_from_url(url: str) -> str:
    """Try to extract a clean PDF filename from the URL."""
    parsed = urlparse(url)
    filename = os.path.basename(parsed.path)

    if not filename.lower().endswith(".pdf"):
        # fallback filename
        filename = f"document_{int(time.time())}.pdf"

    return unquote(filename)


# ==============================
# Core Function: download_pdf
# ==============================
def download_pdf(
    url: str,
    output_dir: str,
    max_retries: int = 3,
    timeout: int = 15,
    overwrite: bool = False,
) -> Tuple[Optional[Path], Optional[str]]:
    """
    Download a PDF from a URL with retry + dedup + validation.

    Returns:
        (file_path, sha256_hash) if success
        (None, None) if failed
    """

    # Ensure directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Deduce filename
    filename = derive_filename_from_url(url)
    file_path = output_path / filename

    # Skip if exists
    if file_path.exists() and not overwrite:
        logger.info(f"[SKIP] File already exists locally: {file_path}")
        sha256 = compute_sha256(file_path)
        return file_path, sha256

    # Retry loop
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"[DOWNLOAD] Attempt {attempt}/{max_retries}: {url}")

            response = requests.get(url, timeout=timeout, stream=True)

            if response.status_code != 200:
                raise Exception(f"Status code: {response.status_code}")

            # Validate content
            content_type = response.headers.get("Content-Type", "").lower()
            if "pdf" not in content_type:
                logger.warning(
                    f"[WARN] URL {url} returned non-PDF content-type {content_type}"
                )

            # Save file
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            # Compute checksum
            sha256 = compute_sha256(file_path)

            logger.info(
                f"[SUCCESS] Downloaded PDF: {file_path.name} (SHA256={sha256[:12]}...)"
            )
            return file_path, sha256

        except Exception as e:
            logger.error(f"[ERROR] Failed to download {url}: {e}")

            if attempt < max_retries:
                sleep_time = 2 ** attempt
                logger.info(f"Retrying after {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                logger.error(f"[FAIL] Exhausted retries for {url}")

    return None, None


# ==============================
# Public API: process_url
# ==============================
def process_url(url: str, output_dir: str) -> Optional[dict]:
    """
    Wrapper that downloads a PDF and returns structured metadata.
    
    Returns:
        {
            "url": ...,
            "local_path": ...,
            "filename": ...,
            "sha256": ...
        }
        or None if failed.
    """
    file_path, sha256 = download_pdf(url, output_dir)

    if file_path is None:
        return None

    return {
        "url": url,
        "local_path": str(file_path),
        "filename": file_path.name,
        "sha256": sha256,
    }
