#!/usr/bin/env bash
# Collect GPT-4.1 teacher trajectories on the training set and build the SFT distillation set.
# Skip this step by downloading our teacher trajectories instead (scripts/download_data.sh).
#
#   OPENAI_API_KEY=... bash scripts/run_teacher.sh math|hotpotqa
set -euo pipefail
TASK=$1
OUT=${OUTPUT_ROOT:-outputs}

python -m prod.rollout --task "$TASK" --split train --backend litellm --model gpt-4.1-2025-04-14 --tag gpt-4.1 --output_root "$OUT"
python -m prod.judge --task "$TASK" --split train --tag gpt-4.1 --output_root "$OUT"
python -m prod.stagnation --task "$TASK" --split train --tag gpt-4.1 --output_root "$OUT"
python -m prod.build_sft_data --task "$TASK" --teacher_tag gpt-4.1 --output_root "$OUT"
