"""
Test script for AWS Bedrock connectivity via LlamaIndex.

Tests:
1. BedrockEmbedding - text embedding generation
2. Bedrock LLM - chat completion (Claude Opus 4.5)

Prerequisites:
- AWS SSO login: `aws sso login --profile <your-profile>`
- AWS_REGION=us-east-1 in .env (required for Opus 4.5 with us.* prefix)
- Model ID: us.anthropic.claude-opus-4-5-20251101-v1:0 in config.yaml

Usage: python script/test_bedrock_llamaindex.py
conda run -n veris_vectordb python script/test_bedrock_llamaindex.py
"""

import sys
sys.path.insert(0, ".")

from veris_chat.chat.config import load_config, get_bedrock_kwargs

# Load configuration
print("=" * 60)
print("Bedrock Connectivity Test")
print("=" * 60)

print("\n[Setup] Loading configuration...")
config = load_config()

models_cfg = config["models"]
aws_cfg = config["aws"]

print(f"  Embedding model: {models_cfg.get('embedding_model')}")
print(f"  Generation model: {models_cfg.get('generation_model')}")
print(f"  AWS Region: {aws_cfg.get('region')}")
print(f"  Using SSO: {aws_cfg.get('use_sso')}")

# Build Bedrock kwargs
bedrock_kwargs = get_bedrock_kwargs(config)
print(f"  Bedrock kwargs: {list(bedrock_kwargs.keys())}")

# -----------------------------------------------------------------------------
# Test 1: BedrockEmbedding
# -----------------------------------------------------------------------------
print("\n[1/2] Testing BedrockEmbedding...")

from llama_index.embeddings.bedrock import BedrockEmbedding

try:
    embed_model = BedrockEmbedding(
        model_name=models_cfg.get("embedding_model"),
        **bedrock_kwargs,
    )
    print("  ✓ BedrockEmbedding initialized")

    # Test single embedding
    test_text = "This is a test document for embedding."
    embedding = embed_model.get_text_embedding(test_text)
    print(f"  ✓ Single embedding generated: dim={len(embedding)}")

    # Test batch embedding
    batch_texts = [
        "First test document.",
        "Second test document.",
    ]
    embeddings = embed_model.get_text_embedding_batch(batch_texts)
    print(f"  ✓ Batch embeddings generated: count={len(embeddings)}, dim={len(embeddings[0])}")

except Exception as e:
    print(f"  ✗ BedrockEmbedding test failed: {e}")
    raise

# -----------------------------------------------------------------------------
# Test 2: Bedrock LLM
# -----------------------------------------------------------------------------
print("\n[2/2] Testing Bedrock LLM...")

from llama_index.llms.bedrock import Bedrock
from llama_index.core.llms import ChatMessage

try:
    # Note: Bedrock class works but shows deprecation warning.
    # BedrockConverse is recommended but requires llama-index-llms-bedrock-converse.
    llm = Bedrock(
        model=models_cfg.get("generation_model"),
        context_size=1000,
        **bedrock_kwargs,
    )
    print("  ✓ Bedrock LLM initialized")

    # Test chat completion
    messages = [
        ChatMessage(role="system", content="You are a helpful assistant. Be brief."),
        ChatMessage(role="user", content="Say hello in one sentence."),
    ]

    response = llm.chat(messages)
    print(f"  ✓ Chat completion successful")
    print(f"    Response: {response.message.content[:100]}...")

except Exception as e:
    print(f"  ✗ Bedrock LLM test failed: {e}")
    raise

print("\n" + "=" * 60)
print("Bedrock connectivity test completed successfully!")
print("=" * 60)
