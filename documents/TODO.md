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

