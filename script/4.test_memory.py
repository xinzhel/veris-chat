"""
Test script for conversation memory using Mem0Memory.

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
setup_logging(
    run_id="test_memory",
    result_dir="./logs",
    add_console_handler=True,
    verbose=True,
    allowed_namespaces=("veris_chat", "__main__"),
)

print("=" * 60)
print("Conversation Memory Test (Mem0Memory)")
print("=" * 60)

# -----------------------------------------------------------------------------
# Load configuration
# -----------------------------------------------------------------------------
print("\n[Setup] Loading configuration...")
config = load_config()
print("  ✓ Configuration loaded")

# Test constants
TEST_SESSION_ID_1 = "test_memory_session_001"
TEST_SESSION_ID_2 = "test_memory_session_002"

print(f"  Test session 1: {TEST_SESSION_ID_1}")
print(f"  Test session 2: {TEST_SESSION_ID_2}")

# -----------------------------------------------------------------------------
# Cleanup stale memory collections (dimension mismatch fix)
# -----------------------------------------------------------------------------
print("\n[Cleanup] Deleting stale memory collections if they exist...")
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
            print(f"  ✓ Deleted stale collection: {collection_name}")
        except Exception:
            print(f"  - Collection {collection_name} does not exist (OK)")
    client.close()

# -----------------------------------------------------------------------------
# Test get_session_memory
# -----------------------------------------------------------------------------
print("\n[1/4] Testing get_session_memory...")

from veris_chat.chat.retriever import get_session_memory
from llama_index.core.base.llms.types import ChatMessage, MessageRole

memory1 = get_session_memory(TEST_SESSION_ID_1)
print(f"  ✓ Mem0Memory created for session: {TEST_SESSION_ID_1}")
print(f"  Memory type: {type(memory1).__name__}")
print(f"  Context: {memory1.context.get_context()}")

# -----------------------------------------------------------------------------
# Test memory persistence - add messages
# -----------------------------------------------------------------------------
print("\n[2/4] Testing memory persistence - adding messages...")

# Reset memory first to ensure clean state
memory1.reset()
print("  ✓ Memory reset")

# Add user message
user_msg = ChatMessage(
    role=MessageRole.USER,
    content="My name is Alice and I'm working on the VERIS project."
)
memory1.put(user_msg)
print(f"  ✓ Added user message: {user_msg.content[:50]}...")

# Add assistant message
assistant_msg = ChatMessage(
    role=MessageRole.ASSISTANT,
    content="Hello Alice! I'd be happy to help you with the VERIS project."
)
memory1.put(assistant_msg)
print(f"  ✓ Added assistant message: {assistant_msg.content[:50]}...")

# Add another exchange
user_msg2 = ChatMessage(
    role=MessageRole.USER,
    content="Can you tell me about site contamination regulations?"
)
memory1.put(user_msg2)
print(f"  ✓ Added user message: {user_msg2.content[:50]}...")

# Verify messages are stored
all_messages = memory1.get_all()
print(f"  ✓ Total messages in memory: {len(all_messages)}")

if len(all_messages) >= 3:
    print("  ✓ Memory persistence verified - messages stored correctly")
else:
    print(f"  ⚠ Expected at least 3 messages, got {len(all_messages)}")

# -----------------------------------------------------------------------------
# Test memory retrieval with context
# -----------------------------------------------------------------------------
print("\n[3/4] Testing memory retrieval with context...")

# Wait for Mem0 to process and extract facts from the conversation
# Mem0 uses an LLM to extract semantic facts asynchronously
import time
print("  ⏳ Waiting for Mem0 to extract facts from conversation (5s)...")
time.sleep(5)

# Get messages with memory context injection
messages_with_context = memory1.get(input="What is my name?")
print(f"  ✓ Retrieved {len(messages_with_context)} messages with context")

# Check if system message with memory context is present
if messages_with_context and messages_with_context[0].role == MessageRole.SYSTEM:
    system_content = messages_with_context[0].content
    print(f"  ✓ System message with memory context present")
    print(f"  System message preview: {system_content[:200]}...")
else:
    print("  ⚠ No system message with memory context found")

# -----------------------------------------------------------------------------
# Test session isolation
# -----------------------------------------------------------------------------
print("\n[4/4] Testing session isolation...")

# Create memory for a different session
memory2 = get_session_memory(TEST_SESSION_ID_2)
print(f"  ✓ Mem0Memory created for session: {TEST_SESSION_ID_2}")

# Reset and add different content
memory2.reset()
user_msg_session2 = ChatMessage(
    role=MessageRole.USER,
    content="My name is Bob and I'm reviewing environmental permits."
)
memory2.put(user_msg_session2)
print(f"  ✓ Added message to session 2: {user_msg_session2.content[:50]}...")

# Verify session 1 still has its own messages
memory1_messages = memory1.get_all()
memory2_messages = memory2.get_all()

print(f"  Session 1 messages: {len(memory1_messages)}")
print(f"  Session 2 messages: {len(memory2_messages)}")

# Check content isolation
session1_has_alice = any("Alice" in str(msg.content) for msg in memory1_messages)
session2_has_bob = any("Bob" in str(msg.content) for msg in memory2_messages)
session1_has_bob = any("Bob" in str(msg.content) for msg in memory1_messages)
session2_has_alice = any("Alice" in str(msg.content) for msg in memory2_messages)

if session1_has_alice and not session1_has_bob:
    print("  ✓ Session 1 correctly contains Alice, not Bob")
else:
    print("  ⚠ Session 1 isolation issue")

if session2_has_bob and not session2_has_alice:
    print("  ✓ Session 2 correctly contains Bob, not Alice")
else:
    print("  ⚠ Session 2 isolation issue")

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------
print("\n[Cleanup] Resetting test memories...")
memory1.reset()
memory2.reset()
print("  ✓ Test memories reset")

print("\n" + "=" * 60)
print("Conversation memory test completed!")
print("=" * 60)
