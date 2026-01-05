import os
import asyncio
os.environ["AWS_REGION"] = "ap-southeast-2"

from llama_index.llms.bedrock_converse import BedrockConverse
from llama_index.core.llms import ChatMessage

MODEL = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
llm = BedrockConverse(model=MODEL, region_name="ap-southeast-2")

print(f"LLM model: {llm.model}")

async def test():
    messages = [ChatMessage(role="user", content="Count to 3")]
    stream = await llm.astream_chat(messages)
    async for chunk in stream:
        if chunk.delta:
            print(chunk.delta, end="", flush=True)
    print()

asyncio.run(test())
