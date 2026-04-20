#!/bin/bash
# Push latest code to deploy-clean branch (single commit, no history)
# Usage: ./deploy/push_clean.sh
#
# Symlinks (chore/, prev_projects_repo/) are committed as-is (text files with paths).
# They'll be broken symlinks on EC2 — that's fine, they're not imported.
# lits-llm is installed via pip on EC2 (see user_data.sh).

set -e

CURRENT_BRANCH=$(git branch --show-current)

# Stash any uncommitted changes first
STASH_RESULT=$(git stash push -m "push_clean temp stash" 2>&1) || true

git checkout --orphan temp-deploy

# Copy lits package from symlinked lits_llm into project root for EC2
# (EC2 won't have the symlink target, so we need the actual files)
if [ -d "prev_projects_repo/lits_llm/lits" ]; then
    cp -r prev_projects_repo/lits_llm/lits ./lits
    echo "Copied lits/ package to project root"
elif [ -L "prev_projects_repo/lits_llm" ]; then
    LITS_TARGET=$(readlink prev_projects_repo/lits_llm)
    cp -r "$LITS_TARGET/lits" ./lits
    echo "Copied lits/ from symlink target $LITS_TARGET"
fi

git add -A
# Exclude files not needed on EC2
git reset HEAD -- .kiro deploy/push_clean.sh backups/ llm_chat/ unit_test/ prev_projects_repo/ 2>/dev/null || true
git commit -m "Deployment $(date +%Y-%m-%d)"
git branch -D deploy-clean 2>/dev/null || true
git branch -m deploy-clean
git push -f agent deploy-clean
git checkout -f "$CURRENT_BRANCH"

# Restore stashed changes if any
if [[ "$STASH_RESULT" != *"No local changes"* ]]; then
    git stash pop 2>/dev/null || true
fi

echo "✓ Pushed to agent:deploy-clean"
