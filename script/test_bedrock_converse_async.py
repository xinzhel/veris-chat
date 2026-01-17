"""
Test BedrockConverse async streaming functionality.

Tests:
1. astream_chat() - async streaming generation
2. achat() - async non-streaming generation
3. Comparison with sync methods
"""

import asyncio
import os
import sys
import time

# Add project root to path for config import
sys.path.insert(0, ".")

from veris_chat.chat.config import load_config, get_bedrock_kwargs
from llama_index.llms.bedrock_converse import BedrockConverse
from llama_index.core.llms import ChatMessage

# Load configuration from config.yaml and .env
config = load_config()
bedrock_kwargs = get_bedrock_kwargs(config)

# For BedrockConverse API in ap-southeast-2, Claude 3.5 Sonnet v2 requires
# a cross-region inference profile. Use the us. prefix for cross-region access.

# Problem: The BedrockConverse API (Converse API) doesn't support on-demand invocation 
# of anthropic.claude-3-5-sonnet-20241022-v2:0 directly in ap-southeast-2.
# Solution: Use the cross-region inference profile prefix us. 
# → us.anthropic.claude-3-5-sonnet-20241022-v2:0
MODEL = config["models"]["streaming_model"]
print(f"Testing with model: {MODEL}")


async def test_astream_chat():
    """
    Test async streaming chat with BedrockConverse.astream_chat().
    
    Key metrics:
    - Time to first token: How quickly streaming begins
    - Total time: End-to-end latency
    - Token count: Number of chunks received
    
    How streaming works:
        Client                          Bedrock Server
        |                                  |
        |--- POST /converse (stream) ----->|  (1 request)
        |                                  |
        |<--- chunk: "1" ------------------|
        |<--- chunk: "\n2" ----------------|  (same connection)
        |<--- chunk: "\n3" ----------------|
        |<--- chunk: "\n4" ----------------|
        |<--- chunk: "\n5" ----------------|
        |<--- [stream end] ----------------|
        |                                  |

    
    Why async for streaming:
    - Single request: async provides no speed benefit
    - Many concurrent users
        Async event loop (1 thread):

        User A: [wait chunk]----[send]----[wait chunk]----[send]----
        User B: ----[wait chunk]----[send]----[wait chunk]----[send]
        User C: [wait chunk]----[send]----[wait chunk]----[send]----
                    ↑
                Thread not blocked, interleaves all users

    
    Use case: FastAPI /chat/stream/ endpoint where tokens are sent to client as they arrive.
    """
    print("=" * 60)
    print("Test 1: astream_chat() - Async Streaming")
    print("=" * 60)
    
    llm = BedrockConverse(model=MODEL, **bedrock_kwargs)
    messages = [
        ChatMessage(role="user", content="Count from 1 to 5, one number per line."),
    ]
    
    t_start = time.perf_counter()
    token_count = 0
    first_token_time = None
    
    print("Response: ", end="", flush=True)
    stream = await llm.astream_chat(messages)
    async for chunk in stream:
        if first_token_time is None:
            first_token_time = time.perf_counter() - t_start
        print(chunk.delta, end="", flush=True)
        token_count += 1
    
    total_time = time.perf_counter() - t_start
    print(f"\n\nTime to first token: {first_token_time:.2f}s")
    print(f"Total time: {total_time:.2f}s")
    print(f"Tokens streamed: {token_count}")


async def test_achat():
    """
    Test async non-streaming chat with BedrockConverse.achat().
    
    Returns complete response after LLM finishes generating.
    Thread is released during I/O wait, allowing event loop to handle other tasks.
    
    Use case: Async endpoint where you need full response before processing (e.g., JSON parsing).
    """
    print("\n" + "=" * 60)
    print("Test 2: achat() - Async Non-Streaming")
    print("=" * 60)
    
    llm = BedrockConverse(model=MODEL, **bedrock_kwargs)
    messages = [
        ChatMessage(role="user", content="What is 2+2? Answer in one word."),
    ]
    
    t_start = time.perf_counter()
    response = await llm.achat(messages)
    total_time = time.perf_counter() - t_start
    
    print(f"Response: {response.message.content}")
    print(f"Total time: {total_time:.2f}s")


def test_sync_chat():
    """
    Test sync chat with BedrockConverse.chat() for comparison.
    
    Blocks the thread until response is complete.
    In FastAPI, sync endpoints run in thread pool (limited concurrency).
    
    Use case: Simple scripts, testing, or when async isn't needed.
    """
    print("\n" + "=" * 60)
    print("Test 3: chat() - Sync Non-Streaming (comparison)")
    print("=" * 60)
    
    llm = BedrockConverse(model=MODEL, **bedrock_kwargs)
    messages = [
        ChatMessage(role="user", content="What is 2+2? Answer in one word."),
    ]
    
    t_start = time.perf_counter()
    response = llm.chat(messages)
    total_time = time.perf_counter() - t_start
    
    print(f"Response: {response.message.content}")
    print(f"Total time: {total_time:.2f}s")


async def test_concurrent_requests():
    """
    Test concurrent async requests with asyncio.gather().
    
    Demonstrates async concurrency: 3 requests run in parallel on single thread.
    Total time should be ~same as single request (not 3x) because I/O waits overlap.
    
    Use case: Batch processing, parallel API calls, high-concurrency FastAPI endpoints.
    """
    print("\n" + "=" * 60)
    print("Test 4: Concurrent Async Requests")
    print("=" * 60)
    
    llm = BedrockConverse(model=MODEL, **bedrock_kwargs)
    
    async def single_request(name: str, question: str):
        messages = [ChatMessage(role="user", content=question)]
        t_start = time.perf_counter()
        response = await llm.achat(messages)
        elapsed = time.perf_counter() - t_start
        return name, response.message.content.strip(), elapsed
    
    t_start = time.perf_counter()
    results = await asyncio.gather(
        single_request("A", "What is 1+1? Answer in one word."),
        single_request("B", "What is 2+2? Answer in one word."),
        single_request("C", "What is 3+3? Answer in one word."),
    )
    total_time = time.perf_counter() - t_start
    
    for name, answer, elapsed in results:
        print(f"  Request {name}: {answer} ({elapsed:.2f}s)")
    print(f"Total time for 3 concurrent requests: {total_time:.2f}s")
    print("(Should be ~same as single request if truly concurrent)")


async def main():
    await test_astream_chat()
    await test_achat()
    test_sync_chat()
    await test_concurrent_requests()
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
