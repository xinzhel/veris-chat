#!/bin/bash
# Push latest code to deploy-clean branch (single commit, no history)
# Usage: ./deploy/push_clean.sh
#
# lits/ is already in the repo as a hard copy — no special handling needed.
# Symlinks (chore/, prev_projects_repo/) are committed as-is (broken on EC2, not imported).

set -e

CURRENT_BRANCH=$(git branch --show-current)

# Stash any uncommitted changes first
STASH_RESULT=$(git stash push -m "push_clean temp stash" 2>&1) || true

git checkout --orphan temp-deploy
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
