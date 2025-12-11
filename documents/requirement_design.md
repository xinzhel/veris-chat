# ✅ **Multi-PDF RAG System against Veris Documents**

This system runs on VERIS documents, is implemented under LlamaIndex, with its native support of **AWS Bedrock LLMs/embeddings, Qdrant DB, conversation memory (source code is copied into `utils/memory.py` for more customization), and citation-grounded generation (source code is copied into `utils/citation_query_engine.py.py` for more customization)**.

The system must support:

* **Chat interface** (via API)
* **Citation-grounded responses**
* **Multi-document summarization**
* **Cross-document reasoning**
* **Conversation continuity via memory**
* **Dynamic retrieval for each query**

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

The system must use AWS Bedrock for both:

### **4.1 LLM generation**

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

---

# **5. Query Engine Behavior**

For each user query:

1. Retrieve relevant chunks from vector store
2. Gather conversation memory context
3. Format prompt (RAG + memory + user question)
4. Call Bedrock LLM
5. Parse output + attach citation metadata
6. Return:

   ```json
   {
       "answer": "...",
       "citations": [
           {"file": "xxx.pdf", "page": 12, "chunk_id": "c_14"},
           ...
       ]
   }
   ```

---
 
