#!/usr/bin/env bash
# Unpack the released data into outputs/ (or $OUTPUT_ROOT):
#   teacher_gpt-4.1_{math,hotpotqa}.tar.gz   GPT-4.1 training-set trajectories, judgments, stagnation
#                                            scores and the SFT set (outputs/<task>/data/sft_gpt-4.1.json)
#   prod_<task>_sft_Qwen3-<size>.json.gz     ProD pair data of the Qwen3-1.7B / 4B SFT students
#
#   bash scripts/unpack_data.sh [dir with the files, default: data/release]
set -euo pipefail
SRC=${1:-$(dirname "$0")/../data/release}
OUT=${OUTPUT_ROOT:-outputs}
mkdir -p "$OUT"
for f in "$SRC"/teacher_gpt-4.1_*.tar.gz; do
  tar -xzf "$f" -C "$OUT" --strip-components=1
done
for f in "$SRC"/prod_*_sft_*.json.gz; do
  name=$(basename "$f" .json.gz)            # prod_<task>_sft_Qwen3-<size>
  task=$(echo "$name" | cut -d_ -f2)
  mkdir -p "$OUT/$task/data"
  gunzip -c "$f" > "$OUT/$task/data/prod_${name#prod_${task}_}.json"
done
find "$OUT" -maxdepth 3 -name "*.json" -path "*/data/*" | sort
