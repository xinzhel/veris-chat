# Environment Setup

## Python Version

This project requires Python >= 3.11 (upgraded from 3.10 on 2026-04-14).

Reason: `lits-llm` (the LLM agent framework used for the ReAct pipeline) requires Python >= 3.11.

Note: The RAG pipeline (`rag_app/`) was originally developed on Python 3.10. If any 3.11 incompatibility arises in RAG dependencies, it can be addressed when needed — the RAG pipeline is stable and rarely modified.

## Package Management

Uses `uv` for fast dependency resolution.

### Fresh setup

```bash
# Create venv with Python 3.11
uv venv .venv --python 3.11

# Activate
source .venv/bin/activate

# Install project dependencies (from pyproject.toml)
uv pip install -r pyproject.toml

# Install lits-llm in editable mode (symlinked during dev, copied for EC2 deploy)
uv pip install -e prev_projects_repo/lits_llm
```

### Recreating venv (e.g., after Python version change)

```bash
uv venv .venv --python 3.11 --clear
source .venv/bin/activate
uv pip install -r pyproject.toml
uv pip install -e prev_projects_repo/lits_llm
```

## lits-llm Dependency

`prev_projects_repo/lits_llm` is a symlink to the local lits-llm repo during development:

```
prev_projects_repo/lits_llm -> /Users/xinzheli/git_repo/tree_search/lits_llm
```

This allows editable development — changes to lits-llm are immediately available without reinstalling.

For EC2 deployment, the symlink should be replaced with an actual copy of the lits-llm directory so the code is self-contained.

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `boto3` / `aioboto3` | AWS Bedrock API (sync / async) |
| `llama-index-*` | RAG pipeline (CitationQueryEngine, embeddings) |
| `qdrant-client` | Vector store |
| `neo4j` | Knowledge graph client |
| `mem0ai` | Conversation memory (RAG pipeline only) |
| `fastapi` / `uvicorn` | API server |
| `lits-llm` | LLM agent framework (ReAct pipeline) |
