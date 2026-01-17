#!/bin/bash
# Push latest code to deploy-clean branch (single commit, no history)
# Usage: ./deploy/push_clean.sh

set -e

CURRENT_BRANCH=$(git branch --show-current)

# Stash any uncommitted changes first
STASH_RESULT=$(git stash push -m "push_clean temp stash" 2>&1) || true

git checkout --orphan temp-deploy
git add -A
git commit -m "Deployment $(date +%Y-%m-%d)"
git branch -D deploy-clean 2>/dev/null || true
git branch -m deploy-clean
git push -f deploy deploy-clean
git checkout "$CURRENT_BRANCH"

# Restore stashed changes if any
if [[ "$STASH_RESULT" != *"No local changes"* ]]; then
    git stash pop 2>/dev/null || true
fi

echo "✓ Pushed to deploy:deploy-clean"
