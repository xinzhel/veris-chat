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
## Finish Task 2
* citation footnotes: No need. Fully in-text citation
* URL link should be inserted in a specific format, determined by Ozzy for rendering 

## Begin Task 3, 4 in requirments_design.md

## Check (Not Systematic Evaluation)
* time usage of 1) Ingestion; 2) Retrieval; 2) Citation-Grounded Generation; 3) Memory Retrieval
* the accuracy of references across multiple documents

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

