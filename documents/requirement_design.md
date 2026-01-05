# ✅ **Document-Grounded Conversational System**

We are developping an interactive system where, for each **conversation session**, the user supplies a **small set of task-relevant document URLs**. These documents are:

1. downloaded and parsed on-demand,
2. chunked and embedded on-the-fly,
3. stored in a **session-scoped ephemeral vector memory**, and
4. queried repeatedly across multi-turn conversation with **conversation memory + citations**.


* Request: Each request includes `session_id`.
    ```json
    {
    "session_id": "a157",
    "document_urls": ["https://...pdf", "https://...pdf"],
    "message": "..."
    }
    ```
* The system maintains:
    * **Session memory** (chat history / mem0-style)
    * **Session vector store** (documents attached to that session)

* Structure of the repository:
    ```
    /documents/
        requirement_design.md (this file)
        TODO.md (ignored)

    /veris_chat/ 
        chat/
        ingestion/
        utils/
            citation_query_engine.py
            memory.py

    /script/
        (for each module in veris_rag, add a simple test script, which should be a naive Python code without testing wrapper or argparse, so that I can run it in an interactive notebook mode)

    /app/
        chat_api.py  (FastAPI or similar)

    config.yaml

    environment.yaml
    ```

---

Below are the Core Capabilities.

## **1 Session-Attached, Multi-PDF Document Ingestion**
 If `document_urls` are provided for a session:

* download PDFs (with local cache)
* parse into per-page text + page_number
* chunk into retrievable chunks
* embed chunks
* store vectors into Qdrant under session scope (collection or payload filter)
* Required payload metadata per chunk
    * `session_id`
    * `url`
    * `filename`
    * `page_number`
    * `chunk_id`
    * `chunk_index`
    * `section_header` (optional)
    * `text`
This part has been implemented in `veris_chat/ingestion`, exposing `IngestionClient` as the one-stop interface for ingestion.

```python
from veris_chat.ingestion.main_client import IngestionClient
# Initialize Client with config settings
client = IngestionClient(
    storage_path="./qdrant_local",
    collection_name=qdrant_cfg["collection_name"],
    embedding_model=embedding_model,
    embedding_dim=qdrant_cfg["vector_size"],
    chunk_size=chunking_cfg["chunk_size"],
    chunk_overlap=chunking_cfg["overlap"],
)
client.store(url)
```

Note: the ingestion code is independent with the rest below. The rest of system is implemented directly under LlamaIndex without the complicated Python wrapper, specifically including the LlamaIndex support below: 
* AWS Bedrock LLMs/embeddings,
* Qdrant DB, 
* conversation memory (source code is copied into `utils/memory.py` for more customization), and 
* citation-grounded generation (source code is copied into `utils/citation_query_engine.py.py` for more customization).

## **2 Retrieval-Augmented Generation**

### **2.1 Retrieval**
Re-connect to Qdrant VectorDB via llamaindex interface as VectoStoreIndex, as shown below
```python
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore

import qdrant_client

client = qdrant_client.QdrantClient(
    "<qdrant-url>",
    api_key="<qdrant-api-key>", # For Qdrant Cloud, None for local instance
)

vector_store = QdrantVectorStore(client=client, collection_name="documents")
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
```

* Retrieval --- For each query request:
    1. Determine the session’s vector memory:
        * index-key filtering based on a filter `session_id == session_id`
    2. Perform semantic search (top-K)
    3. Return chunks + payload metadata

### **2.2 LLM Genration**

Inject:
    * conversation memory context (session-scoped): see Section 4. Conversation Memory for detail.
    * retrieved chunks (session-scoped)
    * user query

Requiring **Citation-Grounded Generation**, i.e., Responses **must include citations** referencing the source documents.
* Each answer must reference the PDFs used during reasoning. Citations must be generated from metadata (the payload of each chunck)
  * `filename`
  * `page_number`
  * `url`
  * `chunk_index`
* Citation format can be inline, e.g.,
  `… as noted in {filename}.pdf (p. {page_number}).`
* Must support multiple citation formats (inline / bracket / footnotes), but default is inline.
* The system must expose the full set of source nodes used for each answer.

### **2.3 AWS Bedrock Integration under LlamaIndex**
 We need to connect to Bedrock under the support of both `aws sso login` and Access Keys. If the following environment variables are empty (which is set in `.env`), rely on `aws sso login`.

AWS_ACCESS_KEY_ID=""
AWS_SECRET_ACCESS_KEY=""
AWS_SESSION_TOKEN=""

NOTE: In `veris_chat/chat`, reduce wrapper functions like get_llm(), get_embed_model(), get_qdrant_client() since LlamaIndex already provides native support for these components 


#### **2.3.1 Embedding model via AWS Bedrock**

See the config.yaml. Below is the content of initial config.yaml:
```yaml
# Model settings
models:
  embedding_model: "cohere.embed-english-v3"  # AWS Bedrock embedding model
  generation_model: "anthropic.claude-3-5-sonnet-20240620-v1:0"

# Qdrant settings
qdrant:
  collection_name: "veris_pdfs"
  vector_size: 1024  # Cohere embed-v4 dimension

# Logging settings
logging:
  level: "INFO"  # Options: DEBUG, INFO, WARNING, ERROR
  log_prefix: "rag" # followed by date; log is date-wise
  console_output: true
```

Example Python code:

```python
from llama_index.embeddings.bedrock import BedrockEmbedding
embed_model = BedrockEmbedding(
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
    region_name="<aws-region>",
    profile_name="<aws-profile>",
)
# or simply
# model = BedrockEmbedding(model_name="cohere.embed-english-v3")
coherePayload = ["This is a test document", "This is another test document"]

embed1 = model.get_text_embedding("This is a test document")
print(embed1)

embeddings = model.get_text_embedding_batch(coherePayload)
print(embeddings)
```

#### **2.3.2 LLM Generation**

**Sync Generation** (using deprecated `Bedrock` class):

```python
from llama_index.core.llms import ChatMessage
from llama_index.llms.bedrock import Bedrock

llm = Bedrock(model="anthropic.claude-3-5-sonnet-20241022-v2:0", profile_name=profile_name)
messages = [
    ChatMessage(role="system", content="You are a helpful assistant"),
    ChatMessage(role="user", content="Tell me a story"),
]
resp = llm.chat(messages)
resp = llm.stream_chat(messages)  # sync streaming works
```

**Async Streaming** (using `BedrockConverse` - recommended):

```python
from llama_index.llms.bedrock_converse import BedrockConverse

llm = BedrockConverse(model="anthropic.claude-3-5-sonnet-20241022-v2:0", profile_name=profile_name)
messages = [
    ChatMessage(role="system", content="You are a helpful assistant"),
    ChatMessage(role="user", content="Tell me a story"),
]

# Async streaming (for FastAPI /chat/stream/ endpoint)
async for chunk in llm.astream_chat(messages):
    print(chunk.delta, end="")
```

**Note**: `llama_index.llms.bedrock.Bedrock` has `astream_chat()` but raises `NotImplementedError`. Use `BedrockConverse` for async streaming. See `documents/async.md` for concurrency details.

#### **2.3.3 Citation-Grounded Generation**
```python
query_engine = CitationQueryEngine.from_args(
    index,
    similarity_top_k=3,
    # here we can control how granular citation sources are, the default is 512
    citation_chunk_size=512,
)
```

**Note**: the QueryEngine must be constructed from the **session’s index**.

You may need to modify the source code at [the source code](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/query_engine/citation_query_engine.py) has been copied to the local module: `utils/citation_query_engine.py` for easy import and customization
---

# **3. Conversation Memory**

The system must support **conversation memory** alongside retrieval:

Use mem0 integration, [the source code](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/memory/llama-index-memory-mem0/llama_index/memory/mem0/base.py) has been copied to the local module: `utils/memory.py` for easy import and customization.

### Behavior:

* memory store is session-keyed
* memory is persisted via Qdrant
* memory content must be injected into the LLM prompt for each new query, along with retrieved context:
  ```
  prompt = memory_context + retrieved_chunks + user_query
  ```

---

# **4. FastAPI**

Each user request contains:
```json
{
  "message": "Given the site located at 322 New Street, Brighton 3186, is the site a priority site?",
  "session_id": "a157",
  "document_urls": [
    "https://.../doc1.pdf",
    "https://.../doc2.pdf"
  ]
}
```

`session_id` is used to retrieve the memory.

Two endpoints:
- `/chat/` - Sync endpoint (uses thread pool for concurrency)
- `/chat/stream/` - Async streaming endpoint (uses `BedrockConverse.astream_chat()`)

Response format:
```json
{
    "response": "...",
    "session_id": "a157",
    "citations": [
        {"file": "xxx.pdf", "page": 12, "chunk_id": "c_14", "url": "xxx"},
        ...
    ],
}
```

Streaming format (Server-Sent Events):
```
{"type": "token", "content": "The site"}
{"type": "token", "content": " is classified"}
...
{"type": "done", "citations": [...], "sources": [...]}
```

See `documents/async.md` for details on sync vs async concurrency patterns.
---
 
