"""Tests Mem0's fact extraction prompt with Bedrock LLM.

**Use when:** Mem0 memory isn't extracting facts from conversations (empty `search()` results).

```bash
python script/test_mem0_fact_extraction.py
```

## Common Issues

1. **"on-demand throughput isn't supported"** - Model requires INFERENCE_PROFILE. Use `list_bedrock_models.py` to find ON_DEMAND models or `list_inference_profiles.py` to get profile IDs.

2. **Empty fact extraction** - Mem0's LLM call may be failing silently. Use `test_mem0_fact_extraction.py` to debug.

3. **Region mismatch** - Set `AWS_REGION` env var before importing mem0 modules.
"""

import os
os.environ['AWS_REGION'] = 'us-east-1'

from mem0.llms.aws_bedrock import AWSBedrockLLM
from mem0.memory.utils import get_fact_retrieval_messages

llm = AWSBedrockLLM(config={
    'model': 'anthropic.claude-3-5-sonnet-20240620-v1:0',
    'temperature': 0.1,
    'max_tokens': 2000,
})

system_prompt, user_prompt = get_fact_retrieval_messages('User: My name is Alice and I love Python programming.', False)

messages = [
    {'role': 'system', 'content': system_prompt},
    {'role': 'user', 'content': user_prompt},
]

print('Testing with Mem0 prompt...')
response = llm.generate_response(messages)
print(f'Response: {response}')