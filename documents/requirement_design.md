# ✅ **Multi-PDF RAG System against Veris Documents**

This system runs on VERIS documents, is implemented under LlamaIndex, with its native support of 
* AWS Bedrock LLMs/embeddings,
* Qdrant DB, 
* conversation memory (source code is copied into `utils/memory.py` for more customization), and 
* citation-grounded generation (source code is copied into `utils/citation_query_engine.py.py` for more customization).

The structure of the repository:
```
/documents/
    requirement_design.md (this file)
    TODO.md (ignored)

/veris_rag/
    (put your implmentation of each module)

/utils/
    citation_query_engine.py
    memory.py

/script/
    (for each module in veris_rag, add a simple test script, which should be a naive Python code without testing wrapper or argparse, so that I can run it in an interactive notebook mode)

/app/
    chat_api.py  (FastAPI or similar)

config.yaml

environment.yaml
```

The system must support the following capabilities
---

# **1. Core Capabilities**

### **1.1 Multi-PDF Document Ingestion**

* The system must load **multiple PDF documents** from a directory.
* Each PDF should be parsed into text, chunked, and embedded.
* Metadata must include:

  * filename
  * page number
  * chunk ID
  * optional section header (if extractable)

(I have ingested the PDF documents onto VectorDB, which can be accessed via QDRANT_URL, QDRANT_API_KEY )

This part can fit in llamaindex interface as VectoStoreIndex, as shown below
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

### **1.2 Dynamic RAG**
* Retrieval 
    0. Input from a request: {"PID": "", "user_query": ""} 
    2. perform index-key filtering based on the given "PID"
    3. perform semantic search (vector-similarity) **dynamically**. The system must retrieve top-K relevant chunks attached to the given "PID".
* LLM genration
    0. Input: Query + Retrieved chunks must be injected into the LLM prompt 
    1. RAG must support long-form queries that require reasoning across multiple documents.

---

# **2. Citation-Grounded Generation (NotebookLM-Style)**

Responses **must include citations** referencing the source documents.

Requirements:

* Each answer must reference the PDFs used during reasoning. Citations must be generated from metadata (the payload of each chunck)
  * `filename`
  * `page_number`
  * `url`
  * `chunk_index`
* Citation format can be inline, e.g.,
  `… as noted in {filename}.pdf (p. {page_number}).`
* Must support multiple citation formats (inline / bracket / footnotes), but default is inline.
* The system must expose the full set of source nodes used for each answer.

```python
query_engine = CitationQueryEngine.from_args(
    index,
    similarity_top_k=3,
    # here we can control how granular citation sources are, the default is 512
    citation_chunk_size=512,
)
You may need to modify the source code at 
```


[the source code](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/query_engine/citation_query_engine.py) has been copied to the local module: `utils/citation_query_engine.py` for easy import and customization

---

# **3. Conversation Memory**

The system must support **conversation memory** alongside retrieval:

Use mem0 integration, [the source code](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/memory/llama-index-memory-mem0/llama_index/memory/mem0/base.py) has been copied to the local module: `utils/memory.py` for easy import and customization.

### Behavior:

* The memory content must be injected into the LLM prompt for each new query.
* Memory must **coexist** with RAG retrieval:

  ```
  prompt = memory_context + retrieved_chunks + user_query
  ```

---

# **4. AWS Bedrock Integration**

The system must use AWS Bedrock for both. We need to connect to Bedrock under the support of both `aws sso login` and Access Keys 
* If the following environment variables are empty (which is set in `.env`), rely on `aws sso login`.

AWS_ACCESS_KEY_ID=""
AWS_SECRET_ACCESS_KEY=""
AWS_SESSION_TOKEN=""

### **4.1 LLM generation**
Example Python code:

```python
from llama_index.core.llms import ChatMessage
from llama_index.llms.bedrock import Bedrock

messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="Tell me a story"),
]

resp = Bedrock(
    model="amazon.titan-text-express-v1", profile_name=profile_name
).chat(messages)
```

Streaming:

```python
from llama_index.llms.bedrock import Bedrock
llm = Bedrock(
    model="amazon.titan-text-express-v1",
    aws_access_key_id="AWS Access Key ID to use",
    aws_secret_access_key="AWS Secret Access Key to use",
    aws_session_token="AWS Session Token to use",
    region_name="AWS Region to use, eg. us-east-1",
)
# or simply
llm = Bedrock(model="amazon.titan-text-express-v1", profile_name=profile_name)
messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="Tell me a story"),
]
resp = llm.stream_chat(messages)
```


### **4.2 Embedding model via AWS Bedrock**

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

---

# **5. FastAPI**

Each user request contains:
```json
{
  "message": "Given the site located at 322 New Street, Brighton 3186, is the site a priority site?",
  "session_id": "a157"
}
```
`session_id` is used to retrieve the memory.


Two endpoints are `/chat/` and `/chat/stream/` (streaming).

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
---
 
