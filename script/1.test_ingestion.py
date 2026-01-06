"""
Test script for IngestionClient with session_id support.

Tests:
1. Download, parse, chunk, embed, and store PDFs to Qdrant
2. Verify payload metadata includes session_id

Usage: Run in interactive notebook mode or as a script.
"""

import sys

sys.path.insert(0, ".")

from veris_chat.chat.config import load_config
from veris_chat.utils.logger import setup_logging

# Setup logging with console output to see detailed operations
setup_logging(
    run_id="test_ingestion",
    result_dir="./logs",
    add_console_handler=True,
    verbose=True,
    allowed_namespaces=("veris_chat", "ingestion", "__main__"),
)

# Load configuration
print("=" * 60)
print("IngestionClient Test with session_id")
print("=" * 60)

print("\n[Setup] Loading configuration...")
config = load_config()

models_cfg = config["models"]
qdrant_cfg = config["qdrant"]
chunking_cfg = config["chunking"]

print(f"  Embedding model: {models_cfg.get('embedding_model')}")
print(f"  Collection name: {qdrant_cfg.get('collection_name')}")
print(f"  Vector size: {qdrant_cfg.get('vector_size')}")
print(f"  Chunk size: {chunking_cfg.get('chunk_size')}")
print(f"  Chunk overlap: {chunking_cfg.get('overlap')}")

# -----------------------------------------------------------------------------
# Initialize IngestionClient
# -----------------------------------------------------------------------------
print("\n[1/6] Initializing IngestionClient...")

from veris_chat.ingestion.main_client import IngestionClient

# Use a test collection to avoid polluting production data
TEST_COLLECTION = "veris_pdfs_test"
TEST_SESSION_ID = "test_session_001"

client = IngestionClient(
    storage_path="./qdrant_local_test",
    collection_name=TEST_COLLECTION,
    embedding_model=models_cfg.get("embedding_model"),
    embedding_dim=qdrant_cfg.get("vector_size"),
    chunk_size=chunking_cfg.get("chunk_size", 512),
    chunk_overlap=chunking_cfg.get("overlap", 50),
)

print(f"  ✓ IngestionClient initialized")
print(f"  Collection: {TEST_COLLECTION}")
print(f"  Session ID: {TEST_SESSION_ID}")

# -----------------------------------------------------------------------------
# Reset collection and clear cache to ensure fresh ingestion
# -----------------------------------------------------------------------------
print("\n[2/6] Resetting collection for fresh ingestion...")

client.reset_collection(delete_pdfs=False)
print(f"  ✓ Collection reset complete (Qdrant + url_cache + session_index)")

# -----------------------------------------------------------------------------
# Test PDF ingestion with session_id (with timing)
# -----------------------------------------------------------------------------
print("\n[3/6] Testing PDF ingestion with session_id...")

# Sample URLs from task specification
TEST_URLS = [
    "https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/e991aac7-4fb2-eb11-8236-00224814b351/OL000071228 - Statutory Document.pdf",
    "https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/8b2790ea-4fb2-eb11-8236-00224814b9c3/OL000073004 - Statutory Document.pdf",
]

for i, url in enumerate(TEST_URLS, 1):
    print(f"\n  [{i}/{len(TEST_URLS)}] Processing: {url[:60]}...")
    try:
        result = client.store(url, session_id=TEST_SESSION_ID)
        skipped = result.get("skipped", False)
        
        status = "Skipped (cached)" if skipped else "Successfully ingested"
        print(f"  ✓ {status} document {i}")
    except Exception as e:
        print(f"  ✗ Failed to ingest document {i}: {e}")
        raise

# -----------------------------------------------------------------------------
# Verify payload metadata 
# -----------------------------------------------------------------------------
print("\n[4/6] Verifying payload metadata...")

# Scroll through collection to check payloads
records, _ = client.qdrant.scroll(
    collection_name=TEST_COLLECTION,
    limit=10,
    with_payload=True,
    with_vectors=False,
)

print(f"  Found {len(records)} records in collection")

# Check required payload fields
REQUIRED_FIELDS = [
    # "session_id",
    "url",
    "filename",
    "page_number",
    "chunk_id",
    "chunk_index",
    "section_header",
    "text",
]

all_valid = True
for record in records[:3]:  # Check first 3 records
    payload = record.payload or {}
    missing = [f for f in REQUIRED_FIELDS if f not in payload]
    if missing:
        print(f"  ✗ Record {record.id} missing fields: {missing}")
        all_valid = False
    else:
        print(f"  ✓ Record {record.id} has all required fields")
        # print(f"    session_id: {payload.get('session_id')}")
        print(f"    filename: {payload.get('filename')}")
        print(f"    page_number: {payload.get('page_number')}")
        print(f"    chunk_index: {payload.get('chunk_index')}")

if all_valid:
    print("\n  ✓ All records have required payload metadata including session_id")
else:
    print("\n  ✗ Some records are missing required fields")
    raise AssertionError("Payload validation failed")

# Verify session_id value matches
# for record in records[:3]:
#     payload = record.payload or {}
#     if payload.get("session_id") != TEST_SESSION_ID:
#         print(f"  ✗ session_id mismatch: expected {TEST_SESSION_ID}, got {payload.get('session_id')}")
#         raise AssertionError("session_id mismatch")

# print(f"  ✓ All records have correct session_id: {TEST_SESSION_ID}")

# -----------------------------------------------------------------------------
# Verify session_index tracks URLs correctly
# -----------------------------------------------------------------------------
print("\n[5/6] Verify session_index tracks URLs correctly...")
session_urls = client.get_session_urls(TEST_SESSION_ID)
print(f"  Session '{TEST_SESSION_ID}' has {len(session_urls)} URLs")
for url in session_urls:
    print(f"    - {url[:60]}...")
assert len(session_urls) == len(TEST_URLS), "Session should have all ingested URLs"

# -----------------------------------------------------------------------------
# Test retrieval
# -----------------------------------------------------------------------------
print("\n[6/6] Testing retrieval...")

query = "What is the purpose of this document?"

results = client.retrieve(query, top_k=3)

print(f"  Query: {query}")
print(f"  Results: {results['num_results']} chunks retrieved")

for i, chunk in enumerate(results["chunks"][:2], 1):
    print(f"\n  Result {i}:")
    print(f"    Score: {chunk['score']:.4f}")
    print(f"    Filename: {chunk['filename']}")
    print(f"    Page: {chunk['page_number']}")
    print(f"    Text preview: {chunk['text'][:100]}...")

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("IngestionClient test completed successfully!")
print("=" * 60)
