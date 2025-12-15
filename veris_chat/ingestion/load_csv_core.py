"""Core logic for loading PDF URLs from CSV files.

This module provides functionality to read CSV files containing PDF URLs
and extract them for further processing in the ingestion pipeline.
"""

import csv
from typing import List
from pathlib import Path


def load_pdf_urls_from_csv(csv_file_path: str) -> List[str]:
    """Read CSV file and extract PDF URLs, skipping the header row.
    
    Args:
        csv_file_path: Path to the CSV file containing PDF URLs
        
    Returns:
        List of PDF URLs extracted from the CSV file
        
    Raises:
        FileNotFoundError: If the CSV file does not exist
        ValueError: If the CSV file is empty or has no data rows
        
    Example:
        >>> urls = load_pdf_urls_from_csv("data/csv/epa_licence_point.csv")
        >>> print(f"Found {len(urls)} PDF URLs")
    """
    csv_path = Path(csv_file_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
    
    pdf_urls = []
    
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        
        # Skip the header row
        next(csv_reader, None)
        
        # Extract URLs from remaining rows
        for row in csv_reader:
            if row and row[0].strip():  # Check if row exists and first column is not empty
                pdf_urls.append(row[0].strip())
    
    if not pdf_urls:
        raise ValueError(f"No PDF URLs found in CSV file: {csv_file_path}")
    
    return pdf_urls
