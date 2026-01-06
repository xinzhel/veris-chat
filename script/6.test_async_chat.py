"""
Test script for async streaming chat service.

Tests:
1. Async streaming chat with document ingestion
2. Token-by-token streaming output
3. Citation extraction after streaming
4. Timing comparison with sync chat

Prerequisites: Qdrant must be accessible (local or cloud).

Usage: python script/6.test_async_chat.py
"""

import asyncio
import os
import sys
import time

# Set AWS_REGION before imports
os.environ["AWS_REGION"] = "ap-southeast-2"

sys.path.insert(0, ".")

from veris_chat.chat.config import load_config
from veris_chat.utils.logger import setup_logging

# Setup logging
logger = setup_logging(
    run_id="test_async_chat",
    result_dir="./logs",
    add_console_handler=True,
    verbose=True,
    allowed_namespaces=("veris_chat", "__main__"),
)


def log_print(msg: str):
    """Print to console and log to file."""
    print(msg)
    logger.info(msg)


# Test constants
TEST_SESSION_ID = "test_async_chat_001"
TEST_URLS = [
    "https://drapubcdnprd.azureedge.net/publicregister/attachments/permissions/e991aac7-4fb2-eb11-8236-00224814b351/OL000071228 - Statutory Document.pdf",
]


async def test_async_chat_streaming():
    """
    Test async streaming chat with document ingestion.
    
    This test:
    1. Ingests a document
    2. Streams the response token-by-token
    3. Measures time to first token
    4. Extracts citations at the end
    """
    log_print("=" * 60)
    log_print("Test 1: Async Streaming Chat")
    log_print("=" * 60)
    
    from veris_chat.chat.service import async_chat
    from veris_chat.ingestion.main_client import IngestionClient
    from veris_chat.chat.config import load_config

    # -----------------------------------------------------------------------------
    # Reset collection for clean test state
    # -----------------------------------------------------------------------------
    log_print("\n[Reset] Resetting Qdrant collection and session index...")
    config = load_config()
    models_cfg = config["models"]
    qdrant_cfg = config["qdrant"]
    chunking_cfg = config["chunking"]
    client = IngestionClient(
        embedding_model=models_cfg.get("embedding_model"),
        embedding_dim=qdrant_cfg.get("vector_size"),
        chunk_size=chunking_cfg.get("chunk_size", 512),
        chunk_overlap=chunking_cfg.get("overlap", 50),
    )

    reset_result = client.reset_collection(delete_pdfs=False)
    log_print(f"  ✓ Collection reset in {reset_result['elapsed_time']:.2f}s")
    log_print(f"  Deleted PDFs: {reset_result['deleted_pdfs_count']}")
    
    message = "What is the purpose of this document? Provide a brief summary."
    
    log_print(f"\nQuery: {message}")
    log_print(f"Session ID: {TEST_SESSION_ID}")
    log_print(f"Document URLs: {TEST_URLS}")
    log_print("\n" + "-" * 40)
    log_print("Streaming Response:")
    log_print("-" * 40)
    
    t_start = time.perf_counter()
    first_token_time = None
    token_count = 0
    full_response = ""
    final_result = None
    
    async for chunk in async_chat(
        session_id=TEST_SESSION_ID,
        message=message,
        document_urls=TEST_URLS,
        top_k=5,
        use_memory=False,  # Disable memory for simpler test
        citation_style="markdown_link",
    ):
        if chunk["type"] == "token":
            if first_token_time is None:
                first_token_time = time.perf_counter() - t_start
            print(chunk["content"], end="", flush=True)
            full_response += chunk["content"]
            token_count += 1
        elif chunk["type"] == "done":
            final_result = chunk
        elif chunk["type"] == "error":
            log_print(f"\nError: {chunk['content']}")
            return
    
    total_time = time.perf_counter() - t_start
    
    log_print("\n" + "-" * 40)
    log_print("\nResults:")
    log_print(f"  Time to first token: {first_token_time:.2f}s")
    log_print(f"  Total time: {total_time:.2f}s")
    log_print(f"  Tokens streamed: {token_count}")
    log_print(f"  Response length: {len(full_response)} chars")
    
    if final_result:
        log_print(f"\nTiming breakdown:")
        timing = final_result.get("timing", {})
        log_print(f"  Ingestion: {timing.get('ingestion', 0):.2f}s")
        log_print(f"  Retrieval: {timing.get('retrieval', 0):.2f}s")
        log_print(f"  Generation: {timing.get('generation', 0):.2f}s")
        log_print(f"  Memory: {timing.get('memory', 0):.2f}s")
        
        log_print(f"\nCitations ({len(final_result.get('citations', []))}):")
        for i, citation in enumerate(final_result.get("citations", [])[:3], 1):
            log_print(f"  {i}. {citation}")
        
        log_print(f"\nSources ({len(final_result.get('sources', []))}):")
        for i, source in enumerate(final_result.get("sources", [])[:3], 1):
            log_print(f"  {i}. {source.get('file', 'unknown')} (p.{source.get('page', '?')})")


async def test_follow_up_streaming():
    """
    Test follow-up streaming chat (no new documents).
    
    Uses existing session to verify retrieval from previously ingested docs.
    """
    log_print("\n" + "=" * 60)
    log_print("Test 2: Follow-up Streaming Chat (no ingestion)")
    log_print("=" * 60)
    
    from veris_chat.chat.service import async_chat
    
    message = "What specific permits or licenses are mentioned in the document?"
    
    log_print(f"\nQuery: {message}")
    log_print(f"Session ID: {TEST_SESSION_ID}")
    log_print("\n" + "-" * 40)
    log_print("Streaming Response:")
    log_print("-" * 40)
    
    t_start = time.perf_counter()
    first_token_time = None
    token_count = 0
    
    async for chunk in async_chat(
        session_id=TEST_SESSION_ID,
        message=message,
        document_urls=None,  # No new documents
        top_k=5,
        use_memory=False,
        citation_style="markdown_link",
    ):
        if chunk["type"] == "token":
            if first_token_time is None:
                first_token_time = time.perf_counter() - t_start
            print(chunk["content"], end="", flush=True)
            token_count += 1
        elif chunk["type"] == "done":
            final_result = chunk
    
    total_time = time.perf_counter() - t_start
    
    log_print("\n" + "-" * 40)
    log_print(f"\nResults:")
    if first_token_time is not None:
        log_print(f"  Time to first token: {first_token_time:.2f}s")
    else:
        log_print(f"  Time to first token: N/A (no tokens received)")
    log_print(f"  Total time: {total_time:.2f}s")
    log_print(f"  Tokens streamed: {token_count}")


async def test_comparison_with_sync():
    """
    Compare async streaming with sync chat.
    
    Both should produce similar answers from the same context.
    """
    log_print("\n" + "=" * 60)
    log_print("Test 3: Comparison - Async vs Sync")
    log_print("=" * 60)
    
    from veris_chat.chat.service import async_chat, chat
    
    message = "What is the license number?"
    log_print(f"\nQuery: {message}")
    
    # Async streaming
    log_print("\n--- Async Streaming ---")
    t_async_start = time.perf_counter()
    async_response = ""
    async_first_token = None
    
    async for chunk in async_chat(
        session_id=TEST_SESSION_ID,
        message=message,
        document_urls=None,
        top_k=3,
        use_memory=False,
    ):
        if chunk["type"] == "token":
            if async_first_token is None:
                async_first_token = time.perf_counter() - t_async_start
            async_response += chunk["content"]
    
    t_async_total = time.perf_counter() - t_async_start
    
    # Sync chat
    log_print("--- Sync Chat ---")
    t_sync_start = time.perf_counter()
    sync_result = chat(
        session_id=TEST_SESSION_ID,
        message=message,
        document_urls=None,
        top_k=3,
        use_memory=False,
    )
    t_sync_total = time.perf_counter() - t_sync_start
    
    log_print(f"\nAsync: First token at {async_first_token:.2f}s, Total: {t_async_total:.2f}s")
    log_print(f"Sync:  Total: {t_sync_total:.2f}s")
    log_print(f"\nAsync response preview: {async_response[:200]}...")
    log_print(f"Sync response preview:  {sync_result['answer'][:200]}...")


async def main():
    log_print("=" * 60)
    log_print("Async Streaming Chat Service Test")
    log_print("=" * 60)
    
    await test_async_chat_streaming()
    await test_follow_up_streaming()
    await test_comparison_with_sync()
    
    log_print("\n" + "=" * 60)
    log_print("All tests completed!")
    log_print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
