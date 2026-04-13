# Rename rag_core/ → rag_core/

Rename the core library package from `rag_core/` to `rag_core/` since it's an application-agnostic RAG/agent framework. Veris/parcel-specific logic stays in `app/`.

## Tasks

- [ ] Task 1: Rename package directory
  - [ ] `mv rag_core/ rag_core/`
  - [ ] Update all internal imports within `rag_core/` (e.g., `from veris_chat.chat.config` → `from rag_core.chat.config`)
  - [ ] Update `app/chat_api.py` imports
  - [ ] Update `script/*.py` imports
  - [ ] Update `deploy/user_data.sh` if it references `veris_chat`
  - [ ] Update `pyproject.toml` if package name is referenced
  - [ ] Update `.kiro/specs/` docs that reference `rag_core/`
  - [ ] Update `documents/requirement_design.md` repo structure

- [ ] Task 2: Verify nothing is broken
  - [ ] Run: `python -c "from rag_core.chat.service import chat"`
  - [ ] Run: `python -c "from rag_core.kg import get_kg_client"`
  - [ ] Run: `python -c "from app.chat_api import app"`
  - [ ] Check no remaining `veris_chat` references: `grep -r "veris_chat" --include="*.py"`

- [ ] Task 3: Commit and update specs
  - [ ] `git add -A && git commit -m "refactor: rename rag_core/ → rag_core/"`
  - [ ] Mark this spec as completed
