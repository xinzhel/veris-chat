# Important and Emergent Tasks

## Chat Service Memory Integration
* `CitationQueryEngine.query()` only accepts query string, not chat message list
* `memory.get()` returns system message + chat history, but only memory context is prepended to query string
* Full chat history structure is lost when calling engine.query()
* Current workaround in `service.py`:
  ```python
  # memory.get() returns [SystemMessage, ...chat_history]
  messages_with_context = memory.get(input=message)
  memory_context = messages_with_context[0].content  # Extract system message only
  query_text = f"Context from previous conversations:\n{memory_context}\n\nCurrent question: {message}"
  response = engine.query(query_text)  # Chat history lost here
  ```
* Options: (1) Modify CitationQueryEngine to accept system prompt, or (2) Use LLM chat directly with full message history

## URL Cache Not Session-Aware
* `IngestionClient.url_cache` tracks URLs globally, not per-session
* If URL is ingested for session A, it's skipped for session B and later the retrieval won't retrive for this url since all chunks for the url is attached to Session A

## Async Streaming: Context Overflow Not Fully Handled
* **Location**: `CitationQueryEngine.prepare_streaming_context()`
* **Issue**: When retrieved context exceeds LLM's context window, `CompactAndRefine` makes multiple LLM calls with a refinement loop:
  ```
  Call 1: QA_TEMPLATE + packed_chunk[0] → initial_answer
  Call 2: REFINE_TEMPLATE + initial_answer + packed_chunk[1] → refined_answer
  Call N: REFINE_TEMPLATE + prev_answer + packed_chunk[N-1] → final_answer
  ```
* **Current workaround**: For async streaming, we only use `packed_chunks[0]` and log a warning. This means some retrieved context may be dropped.
* **Why it's usually OK**: Claude 3.5 has 200K context window. With typical settings (top_k=5-10, chunk_size=512), overflow is rare.
* **Full fix options**:
  1. Implement async refinement loop (complex: stream first chunk, collect, stream refinement)
  2. Reduce top_k dynamically if overflow detected
  3. Use tree_summarize to compress context before streaming





# Important but Not Emergent Tasks
## Agentic RAG
LLM 决定检索策略、多轮检索： 
* query 改写 
* 检索策略自适应、反思回环 

Step 1: LLM 分析用户问题 → 决定需要检索吗？ 
Step 2: 如果需要： → 生成查询语句（可能多个 rewrites） → 多轮检索 Qdrant → 整合结果 
Step 3: LLM 基于检索结果生成回答 
Step 4: LLM 自我反思（self-critique） → 是否需要额外检索？ → 如果需要回到 Step 2 
Step 5: 输出最终回答


目前没有开箱即用、完整实现 Agentic RAG Loop 的框架

## Batch embedding
 since Bedrock embeddings are called during session ingestion, ingestion latency becomes part of user interaction. 

## Caching
do not re-embed same chunk text if re-attached within a session or other sessions

<!-- ====== Finished =======

## Task 2
* citation footnotes: No need. Fully in-text citation
* URL link should be inserted in a specific format, determined by Ozzy for rendering 

## Begin Task 3, 4 in requirments_design.md

## Check (Not Systematic Evaluation)
* time usage of 1) Ingestion; 2) Retrieval; 2) Citation-Grounded Generation; 3) Memory Retrieval
* the accuracy of references across multiple documents
======================= -->

# Not Important and Not Emergent
A few considerations:

**Works fine as-is:**
- API access via `http://<ec2-ip>:8000`
- Instance Profile handles Bedrock auth
- systemd keeps the app running

**Optional improvements for customer-facing:**

| Concern | Simple Fix |
|---------|------------|
| HTTPS | Add nginx reverse proxy + Let's Encrypt SSL |
| Domain name | Point DNS to EC2 Elastic IP |
| Downtime during updates | Use rolling restart or brief maintenance window |
| High traffic | Increase instance size (t3.large → t3.xlarge) |

**Not needed unless scaling:**
- Load balancer (single instance is fine for moderate traffic)
- Auto-scaling (manual scaling works for predictable load)
- Docker/ECS (overkill for single app)

For a research project or limited customer base, EC2 + systemd + optional nginx is perfectly adequate. Want me to proceed with the startup script?