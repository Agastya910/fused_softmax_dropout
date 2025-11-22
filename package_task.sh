#!/usr/bin/env bash
set -e
PKG="fused_softmax_dropout_task.zip"
rm -f $PKG
# include visible files + tests + docs + gold (for reviewers)
zip -r $PKG docs fused_softmax_dropout tests task.json run_tests_and_score.py README_task.md
# add gold files (kept for reviewers)
if [ -d gold ]; then
  zip -ur $PKG gold
fi
echo "Created $PKG"
