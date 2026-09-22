#!/usr/bin/env bash
# Run ONCE, right after the initial push to main, by whoever owns the GitHub repo.
# Creates one branch per model owner off main and pushes them, so every groupmate can
# `git clone` and immediately `git checkout` straight into their own branch.
#
# Usage:
#   chmod +x scripts/setup_branches.sh
#   ./scripts/setup_branches.sh

set -e

BRANCHES=(
  "model/bilstm"             # Teammate A — M1
  "model/textcnn"            # Teammate B — M2
  "model/bilstm-attention"   # Teammate C — M3
  "model/distilbert"         # DEEPDEV    — M4
)

git checkout main
git pull origin main

for branch in "${BRANCHES[@]}"; do
  if git show-ref --verify --quiet "refs/heads/$branch"; then
    echo "Branch $branch already exists locally, skipping create."
  else
    git checkout -b "$branch" main
  fi
  git push -u origin "$branch"
  git checkout main
done

echo ""
echo "Done. Branches pushed: ${BRANCHES[*]}"
echo "Each teammate should now run:"
echo "  git clone <repo-url>"
echo "  git checkout <their model/... branch>"
