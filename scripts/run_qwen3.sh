#!/usr/bin/env bash
# Settings behind the Qwen3 rows of Table 3 (LR / c0 from Table 7; ProD trains a 2-epoch schedule and
# the listed epoch's checkpoint is used).
#
#   bash scripts/run_qwen3.sh math Qwen3-4B
set -euo pipefail
TASK=$1; SIZE=$2
case "$TASK/$SIZE" in
  math/Qwen3-4B)       LR=5e-6; C0=0.5; EP=1 ;;
  math/Qwen3-1.7B)     LR=5e-6; C0=0.5; EP=2 ;;
  hotpotqa/Qwen3-4B)   LR=1e-6; C0=0.7; EP=2 ;;
  hotpotqa/Qwen3-1.7B) LR=1e-5; C0=0.5; EP=1 ;;
  *) echo "unknown setting $TASK/$SIZE"; exit 1 ;;
esac
bash "$(dirname "$0")/run_student.sh" "$TASK" "Qwen/$SIZE" "$LR" "$C0" "$EP" "$SIZE"
