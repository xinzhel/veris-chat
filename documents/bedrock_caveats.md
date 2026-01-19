### Debugging Notes
- **AWS_REGION issue**: `chat_api.py` line 27 has `os.environ.setdefault("AWS_REGION", "us-east-1")` 
  - **Why setdefault?** For Sonnet 3.5 v2, we must NOT pass `region_name` explicitly to BedrockConverse - let boto3 use env/default credentials. The `setdefault` ensures `AWS_REGION` is set for boto3 to use.
  - **Why setdefault will lead to a problem? In one sentence: `setdefault` in `chat_api.py` takes precedence over `.env` file**: `setdefault` runs at module import time, before any function calls. `load_config()` is called later in `veris_chat/chat/service.py` (in `chat()` & `async_chat()`). Hence, `load_dotenv()` inside `load_config()` does NOT override existing env vars. Hence, this can casue weired errors, e.g., **Model validation error**: "The provided model identifier is invalid". Because AWS_REGION in .env does not work
  - **How does AWS_REGION env var affect BedrockConverse without passing region_name?** When `BedrockConverse(model=...)` is created without `region_name`, boto3's default credential chain picks up `AWS_REGION` from environment. Verified: `llm.region_name` = `None`, but `llm._client.meta.region_name` = `us-east-1` (from env var).
    - `llm._client` is botocore.client.BedrockRuntime - the boto3 client for Bedrock Runtime API. This is the low-level client that makes the actual ConverseStream API calls to AWS.

- **context_size issue**: Error "`context_size` argument not provided and model provided refers to a non-foundation model"
  - **Key issue**: The `Bedrock` class (old InvokeModel API) requires `context_size` for non-foundation models like Opus 4.5 with `us.*` prefix. `BedrockConverse` does NOT have this issue.
  - **Fix**: Added `context_size=200000` to `Bedrock` class in `_get_models()` (`service.py` line ~300).