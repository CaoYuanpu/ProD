#!/usr/bin/env bash
# Full student pipeline on one task: SFT distillation -> student rollouts on the training set ->
# ProD pair data -> ProD training -> evaluation of the SFT and ProD students.
# Needs the teacher outputs from scripts/run_teacher.sh (or scripts/download_data.sh).
#
#   bash scripts/run_student.sh TASK MODEL PROD_LR C0 PROD_EPOCH [NAME]
#   e.g. bash scripts/run_student.sh math Qwen/Qwen3-1.7B 5e-6 0.5 2
#
# SFT trains 5 epochs and keeps the epoch-3 checkpoint; ProD trains a 2-epoch schedule and keeps
# the PROD_EPOCH checkpoint (see scripts/run_qwen3.sh for the settings used in the paper).
# For hotpotqa, start the retriever first (retriever/README.md).
set -euo pipefail
TASK=$1; MODEL=$2; PROD_LR=$3; C0=$4; PROD_EPOCH=$5
NAME=${6:-$(basename "$MODEL")}
OUT=${OUTPUT_ROOT:-outputs}
SFT_EPOCH=${SFT_EPOCH:-3}
EXTRA=()
# vLLM shares the GPUs with the retriever on hotpotqa
[ "$TASK" = hotpotqa ] && EXTRA=(--gpu_memory_utilization 0.75)

# Path of the checkpoint saved at the end of epoch $2 in output dir $1 (one checkpoint per epoch).
ckpt() { ls -d "$1"/checkpoint-* | sort -t- -k2 -n | sed -n "${2}p"; }

evaluate() {  # tag lora
  local TAG=$1 LORA=$2 SPLIT DATASET
  mapfile -t SETS < <(python -c "
from prod.tasks import get_task
for parts in get_task('$TASK')['eval_sets'].values():
    for split, dataset in parts: print(split, dataset)")
  for SET in "${SETS[@]}"; do
    read -r SPLIT DATASET <<< "$SET"
    python -m prod.rollout --task "$TASK" --split "$SPLIT" --dataset "$DATASET" --model "$MODEL" --lora "$LORA" \
      --tag "$TAG" --output_root "$OUT" "${EXTRA[@]}" < /dev/null
    python -m prod.judge --task "$TASK" --split "$SPLIT" --dataset "$DATASET" --tag "$TAG" --output_root "$OUT"
  done
  python -m prod.evaluate --task "$TASK" --tag "$TAG" --output_root "$OUT"
}

# 1. SFT distillation
SFT_DIR=$OUT/$TASK/ckpt/sft_$NAME
[ -d "$SFT_DIR" ] || python -m prod.train_sft --task "$TASK" --model "$MODEL" --output_dir "$SFT_DIR" --output_root "$OUT"
SFT_LORA=$(ckpt "$SFT_DIR" "$SFT_EPOCH")

# 2-3. student rollouts on the training questions + correctness + stagnation -> ProD pair data
#      (skipped when the pair data already exists, e.g. downloaded with scripts/download_data.sh)
PAIRS=$OUT/$TASK/data/prod_sft_$NAME.json
if [ ! -f "$PAIRS" ]; then
  python -m prod.rollout --task "$TASK" --split train --model "$MODEL" --lora "$SFT_LORA" --tag "sft_$NAME" --output_root "$OUT" "${EXTRA[@]}"
  python -m prod.judge --task "$TASK" --split train --tag "sft_$NAME" --output_root "$OUT"
  python -m prod.stagnation --task "$TASK" --split train --tag "sft_$NAME" --output_root "$OUT"
  python -m prod.build_prod_data --task "$TASK" --student_tag "sft_$NAME" --output_root "$OUT"
fi

# 3. ProD training
PROD_DIR=$OUT/$TASK/ckpt/prod_$NAME
[ -d "$PROD_DIR" ] || python -m prod.train_prod --task "$TASK" --model "$MODEL" --init_lora "$SFT_LORA" \
  --data "$PAIRS" --output_dir "$PROD_DIR" --lr "$PROD_LR" --c0 "$C0"
PROD_LORA=$(ckpt "$PROD_DIR" "$PROD_EPOCH")

# 4. evaluation (in-domain + out-of-domain)
evaluate "sft_$NAME" "$SFT_LORA"
evaluate "prod_$NAME" "$PROD_LORA"
