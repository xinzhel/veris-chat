# Rename veris_chat/ → rag_core/

Rename the core library package from `veris_chat/` to `rag_core/` since it's an application-agnostic RAG framework. Veris/parcel-specific logic stays in `app/`.

## Tasks

- [x] Task 1: Rename package directory
  - [x] `mv veris_chat/ rag_core/`
  - [x] Update all internal imports (bulk sed: `veris_chat.` → `rag_core.`)
  - [x] Update `app/chat_api.py`, `script/*.py`, `.kiro/specs/`, `documents/` references

- [x] Task 2: Verify nothing is broken
  - [x] `python -c "from rag_core.chat.service import chat"` ✓
  - [x] `python -c "from rag_core.kg import get_kg_client"` ✓
  - [x] `python -c "from app.chat_api import app"` ✓
  - [x] `grep -r "veris_chat" --include="*.py"` → no results ✓

- [x] Task 3: Committed
