"""
Test script for IngestionClient with session_id support.

Tests:
1. Download, parse, chunk, embed, and store PDFs to Qdrant
2. Verify payload metadata includes session_id

Usage: Run in interactive notebook mode or as a script.
"""

import sys
import time
import logging

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

# Get logger for timing
timing_logger = logging.getLogger("veris_chat.timing")

# Timing results storage
timing_results = {}

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
print("\n[1/5] Initializing IngestionClient...")

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
# Delete existing collection and clear cache to ensure fresh ingestion
# -----------------------------------------------------------------------------
print("\n[2/5] Deleting existing collection for fresh ingestion...")

t_collection_start = time.perf_counter()

try:
    client.qdrant.delete_collection(collection_name=TEST_COLLECTION)
    print(f"  ✓ Deleted existing collection: {TEST_COLLECTION}")
except Exception as e:
    print(f"  ⚠ Collection may not exist: {e}")

# Clear the URL cache to force re-ingestion
client.url_cache.clear()
print(f"  ✓ Cleared URL cache to force fresh ingestion")

# Recreate the collection
from qdrant_client.http import models as qdrant_models

client.qdrant.create_collection(
    collection_name=TEST_COLLECTION,
    vectors_config=qdrant_models.VectorParams(
        size=qdrant_cfg.get("vector_size", 1024),
        distance=qdrant_models.Distance.COSINE,
    ),
)
print(f"  ✓ Created fresh collection: {TEST_COLLECTION}")

# Create payload index for session_id (required for filtering in Qdrant Cloud)
client.qdrant.create_payload_index(
    collection_name=TEST_COLLECTION,
    field_name="session_id",
    field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
)
print(f"  ✓ Created payload index for session_id")

timing_results["collection_setup"] = time.perf_counter() - t_collection_start
print(f"  ⏱ Collection setup time: {timing_results['collection_setup']:.3f}s")

# -----------------------------------------------------------------------------
# Test PDF ingestion with session_id (with timing)
# -----------------------------------------------------------------------------
print("\n[3/5] Testing PDF ingestion with session_id...")

# Sample URLs from task specification
TEST_URLS = [
    "https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/e991aac7-4fb2-eb11-8236-00224814b351/OL000071228 - Statutory Document.pdf",
    "https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/8b2790ea-4fb2-eb11-8236-00224814b9c3/OL000073004 - Statutory Document.pdf",
]

# Time the entire ingestion process
t_ingestion_start = time.perf_counter()

for i, url in enumerate(TEST_URLS, 1):
    print(f"\n  [{i}/{len(TEST_URLS)}] Processing: {url[:60]}...")
    try:
        t_doc_start = time.perf_counter()
        client.store(url, session_id=TEST_SESSION_ID)
        t_doc_elapsed = time.perf_counter() - t_doc_start
        print(f"  ✓ Successfully ingested document {i}")
        print(f"  ⏱ Document {i} ingestion time: {t_doc_elapsed:.3f}s")
        timing_results[f"doc_{i}_ingestion"] = t_doc_elapsed
    except Exception as e:
        print(f"  ✗ Failed to ingest document {i}: {e}")
        raise

timing_results["total_ingestion"] = time.perf_counter() - t_ingestion_start
print(f"\n  ⏱ Total ingestion time: {timing_results['total_ingestion']:.3f}s")

# -----------------------------------------------------------------------------
# Verify payload metadata includes session_id
# -----------------------------------------------------------------------------
print("\n[4/5] Verifying payload metadata includes session_id...")

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
    "session_id",
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
        print(f"    session_id: {payload.get('session_id')}")
        print(f"    filename: {payload.get('filename')}")
        print(f"    page_number: {payload.get('page_number')}")
        print(f"    chunk_index: {payload.get('chunk_index')}")

if all_valid:
    print("\n  ✓ All records have required payload metadata including session_id")
else:
    print("\n  ✗ Some records are missing required fields")
    raise AssertionError("Payload validation failed")

# Verify session_id value matches
for record in records[:3]:
    payload = record.payload or {}
    if payload.get("session_id") != TEST_SESSION_ID:
        print(f"  ✗ session_id mismatch: expected {TEST_SESSION_ID}, got {payload.get('session_id')}")
        raise AssertionError("session_id mismatch")

print(f"  ✓ All records have correct session_id: {TEST_SESSION_ID}")

# -----------------------------------------------------------------------------
# Test retrieval
# -----------------------------------------------------------------------------
print("\n[5/5] Testing retrieval...")

query = "What is the purpose of this document?"

t_retrieval_start = time.perf_counter()
results = client.retrieve(query, top_k=3)
timing_results["retrieval"] = time.perf_counter() - t_retrieval_start

print(f"  Query: {query}")
print(f"  Results: {results['num_results']} chunks retrieved")
print(f"  ⏱ Retrieval time: {timing_results['retrieval']:.3f}s")

for i, chunk in enumerate(results["chunks"][:2], 1):
    print(f"\n  Result {i}:")
    print(f"    Score: {chunk['score']:.4f}")
    print(f"    Filename: {chunk['filename']}")
    print(f"    Page: {chunk['page_number']}")
    print(f"    Text preview: {chunk['text'][:100]}...")

# -----------------------------------------------------------------------------
# Log timing summary
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Timing Summary")
print("=" * 60)

timing_logger.info("=" * 60)
timing_logger.info("TIMING SUMMARY - Ingestion Test")
timing_logger.info("=" * 60)

# Collection setup time
if "collection_setup" in timing_results:
    msg = f"1) Collection Setup (delete + create): {timing_results['collection_setup']:.3f}s"
    print(f"  {msg}")
    timing_logger.info(msg)

# Per-document ingestion times
for i in range(1, len(TEST_URLS) + 1):
    key = f"doc_{i}_ingestion"
    if key in timing_results:
        msg = f"   Document {i} Ingestion: {timing_results[key]:.3f}s"
        print(f"  {msg}")
        timing_logger.info(msg)

if "total_ingestion" in timing_results:
    msg = f"2) Total Ingestion ({len(TEST_URLS)} docs): {timing_results['total_ingestion']:.3f}s"
    print(f"\n  {msg}")
    timing_logger.info(msg)

if "retrieval" in timing_results:
    msg = f"3) Retrieval: {timing_results['retrieval']:.3f}s"
    print(f"  {msg}")
    timing_logger.info(msg)

# Calculate total
total_time = (
    timing_results.get("collection_setup", 0)
    + timing_results.get("total_ingestion", 0)
    + timing_results.get("retrieval", 0)
)
msg = f"TOTAL: {total_time:.3f}s"
print(f"\n  {msg}")
timing_logger.info(msg)
timing_logger.info("=" * 60)

print("\n" + "=" * 60)
print("IngestionClient test completed successfully!")
print("=" * 60)
