"""
Test script for conversation memory using Mem0Memory.

This test is completely independent from the Task 7.4 redesign.
❌ session_index (session → URLs mapping)
❌ url_cache (URL → ingestion metadata)
❌ IngestionClient or document ingestion
❌ veris_pdfs collection (document chunks)
❌ URL-based filtering (MatchAny filter)
❌ retrieve_with_url_filter() or get_url_filtered_retriever()


Tests:
1. Memory persistence - messages are stored and retrievable
2. Session isolation - different sessions have separate memories

Prerequisites: Qdrant must be accessible (local or cloud).

Usage: Run in interactive notebook mode or as a script.
"""

import os
import sys


# Set AWS_REGION BEFORE any imports that might use Bedrock
# IMPORTANT: This is required because Mem0's factory uses BaseLlmConfig which
# doesn't accept aws_region param - it reads from env var instead.
# Claude 3.5 Sonnet v2 supports ON_DEMAND in ap-southeast-2
os.environ["AWS_REGION"] = "ap-southeast-2"

sys.path.insert(0, ".")

from veris_chat.chat.config import load_config
from veris_chat.utils.logger import setup_logging

# Setup logging
logger = setup_logging(
    run_id="test_memory",
    result_dir="./logs",
    add_console_handler=True,
    verbose=True,
    allowed_namespaces=("veris_chat", "__main__"),
)

logger.info("=" * 60)
logger.info("Conversation Memory Test (Mem0Memory)")
logger.info("=" * 60)

# -----------------------------------------------------------------------------
# Load configuration
# -----------------------------------------------------------------------------
logger.info("[Setup] Loading configuration...")
config = load_config()
logger.info("  ✓ Configuration loaded")

# Test constants
TEST_SESSION_ID_1 = "test_memory_session_001"
TEST_SESSION_ID_2 = "test_memory_session_002"

logger.info(f"  Test session 1: {TEST_SESSION_ID_1}")
logger.info(f"  Test session 2: {TEST_SESSION_ID_2}")

# -----------------------------------------------------------------------------
# Cleanup stale memory collections (dimension mismatch fix)
# -----------------------------------------------------------------------------
logger.info("[Cleanup] Deleting stale memory collections if they exist...")
import os
import qdrant_client

qdrant_url = config["qdrant"].get("url") or os.getenv("QDRANT_URL")
qdrant_api_key = config["qdrant"].get("api_key") or os.getenv("QDRANT_API_KEY")

if qdrant_url:
    client = qdrant_client.QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    for session_id in [TEST_SESSION_ID_1, TEST_SESSION_ID_2]:
        collection_name = f"mem0_memory_{session_id}"
        try:
            client.delete_collection(collection_name)
            logger.info(f"  ✓ Deleted stale collection: {collection_name}")
        except Exception:
            logger.info(f"  - Collection {collection_name} does not exist (OK)")
    client.close()

# -----------------------------------------------------------------------------
# Test get_session_memory
# -----------------------------------------------------------------------------
logger.info("[1/4] Testing get_session_memory...")

from veris_chat.chat.retriever import get_session_memory
from llama_index.core.base.llms.types import ChatMessage, MessageRole

memory1 = get_session_memory(TEST_SESSION_ID_1)
logger.info(f"  ✓ Mem0Memory created for session: {TEST_SESSION_ID_1}")
logger.info(f"  Memory type: {type(memory1).__name__}")
logger.info(f"  Context: {memory1.context.get_context()}")

# -----------------------------------------------------------------------------
# Test memory persistence - add messages
# -----------------------------------------------------------------------------
logger.info("[2/4] Testing memory persistence - adding messages...")

# Reset memory first to ensure clean state
memory1.reset()
logger.info("  ✓ Memory reset")

# Add user message
user_msg = ChatMessage(
    role=MessageRole.USER,
    content="My name is Alice and I'm working on the VERIS project."
)
memory1.put(user_msg)
logger.info(f"  ✓ Added user message: {user_msg.content[:50]}...")

# Add assistant message
assistant_msg = ChatMessage(
    role=MessageRole.ASSISTANT,
    content="Hello Alice! I'd be happy to help you with the VERIS project."
)
memory1.put(assistant_msg)
logger.info(f"  ✓ Added assistant message: {assistant_msg.content[:50]}...")

# Add another exchange
user_msg2 = ChatMessage(
    role=MessageRole.USER,
    content="Can you tell me about site contamination regulations?"
)
memory1.put(user_msg2)
logger.info(f"  ✓ Added user message: {user_msg2.content[:50]}...")

# Verify messages are stored
all_messages = memory1.get_all()
logger.info(f"  ✓ Total messages in memory: {len(all_messages)}")

if len(all_messages) >= 3:
    logger.info("  ✓ Memory persistence verified - messages stored correctly")
else:
    logger.warning(f"  ⚠ Expected at least 3 messages, got {len(all_messages)}")

# -----------------------------------------------------------------------------
# Test memory retrieval with context
# -----------------------------------------------------------------------------
logger.info("[3/4] Testing memory retrieval with context...")

# Wait for Mem0 to process and extract facts from the conversation
# Mem0 uses an LLM to extract semantic facts asynchronously
import time
logger.info("  ⏳ Waiting for Mem0 to extract facts from conversation (5s)...")
time.sleep(5)

# Get messages with memory context injection
messages_with_context = memory1.get(input="What is my name?")
logger.info(f"  ✓ Retrieved {len(messages_with_context)} messages with context")

# Check if system message with memory context is present
if messages_with_context and messages_with_context[0].role == MessageRole.SYSTEM:
    system_content = messages_with_context[0].content
    logger.info("  ✓ System message with memory context present")
    logger.info(f"  System message preview: {system_content[:200]}...")
else:
    logger.warning("  ⚠ No system message with memory context found")

# -----------------------------------------------------------------------------
# Test session isolation
# -----------------------------------------------------------------------------
logger.info("[4/4] Testing session isolation...")

# Create memory for a different session
memory2 = get_session_memory(TEST_SESSION_ID_2)
logger.info(f"  ✓ Mem0Memory created for session: {TEST_SESSION_ID_2}")

# Reset and add different content
memory2.reset()
user_msg_session2 = ChatMessage(
    role=MessageRole.USER,
    content="My name is Bob and I'm reviewing environmental permits."
)
memory2.put(user_msg_session2)
logger.info(f"  ✓ Added message to session 2: {user_msg_session2.content[:50]}...")

# Verify session 1 still has its own messages
memory1_messages = memory1.get_all()
memory2_messages = memory2.get_all()

logger.info(f"  Session 1 messages: {len(memory1_messages)}")
logger.info(f"  Session 2 messages: {len(memory2_messages)}")

# Check content isolation
session1_has_alice = any("Alice" in str(msg.content) for msg in memory1_messages)
session2_has_bob = any("Bob" in str(msg.content) for msg in memory2_messages)
session1_has_bob = any("Bob" in str(msg.content) for msg in memory1_messages)
session2_has_alice = any("Alice" in str(msg.content) for msg in memory2_messages)

if session1_has_alice and not session1_has_bob:
    logger.info("  ✓ Session 1 correctly contains Alice, not Bob")
else:
    logger.warning("  ⚠ Session 1 isolation issue")

if session2_has_bob and not session2_has_alice:
    logger.info("  ✓ Session 2 correctly contains Bob, not Alice")
else:
    logger.warning("  ⚠ Session 2 isolation issue")

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------
logger.info("[Cleanup] Resetting test memories...")
memory1.reset(reset_mem0=True)
memory2.reset(reset_mem0=True)
logger.info("  ✓ Test memories reset")

logger.info("=" * 60)
logger.info("Conversation memory test completed!")
logger.info("=" * 60)
