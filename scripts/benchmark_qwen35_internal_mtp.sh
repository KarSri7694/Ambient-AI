#!/usr/bin/env bash
set -euo pipefail

# Dedicated benchmark launcher for:
#   Qwen3.6-35B-A3B-UD-Q6_K.gguf
#
# This model has the MTP draft head inside the main GGUF, so do not pass
# --model-draft. Internal MTP is enabled with:
#   --spec-type draft-mtp --spec-draft-n-max N
#
# Example quick run:
#   RUNS=1 WARMUPS=0 MAX_TOKENS=2048 bash scripts/benchmark_qwen35_internal_mtp.sh
#
# Example full run:
#   RUNS=3 WARMUPS=1 MAX_TOKENS=4096 bash scripts/benchmark_qwen35_internal_mtp.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL="${MODEL:-/persistent/models/Qwen35B/Qwen3.6-35B-A3B-UD-Q6_K.gguf}"
export MMPROJ="${MMPROJ:-/persistent/models/Qwen35B/mmproj-F16.gguf}"

# Important: keep empty. This model uses internal MTP from the main GGUF.
export MTP_MODEL=""

export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-8092}"
export OUT_DIR="${OUT_DIR:-.ambient_data/benchmarks/qwen35}"

export RUNS="${RUNS:-1}"
export WARMUPS="${WARMUPS:-0}"
export MAX_TOKENS="${MAX_TOKENS:-2048}"

QWEN35_CONTEXT="${QWEN35_CONTEXT:-150000}"
QWEN35_NGL="${QWEN35_NGL:-999}"
QWEN35_CTK="${QWEN35_CTK:-q8_0}"
QWEN35_CTV="${QWEN35_CTV:-q8_0}"
QWEN35_FA="${QWEN35_FA:-on}"

BATCH_START="${BATCH_START:-2048}"
BATCH_MAX="${BATCH_MAX:-4096}"
UBATCH_START="${UBATCH_START:-512}"
UBATCH_MAX="${UBATCH_MAX:-2048}"
BATCH_STEP="${BATCH_STEP:-1024}"
MTP_DRAFT_MIN="${MTP_DRAFT_MIN:-1}"
MTP_DRAFT_MAX="${MTP_DRAFT_MAX:-2}"

if [[ ! -f "$MODEL" ]]; then
  echo "Model file not found: $MODEL" >&2
  exit 1
fi

if [[ -n "$MMPROJ" && ! -f "$MMPROJ" ]]; then
  echo "mmproj file not found: $MMPROJ" >&2
  exit 1
fi

CONFIGS=()

for ((batch_size=BATCH_START; batch_size<=BATCH_MAX; batch_size+=BATCH_STEP)); do
  for ((ubatch_size=UBATCH_START; ubatch_size<=UBATCH_MAX; ubatch_size+=BATCH_STEP)); do
    if (( ubatch_size > batch_size )); then
      continue
    fi
    CONFIGS+=(
      "qwen35-base-b${batch_size}-ub${ubatch_size}::-c ${QWEN35_CONTEXT} -ngl ${QWEN35_NGL} -fa ${QWEN35_FA} -ctk ${QWEN35_CTK} -ctv ${QWEN35_CTV} -b ${batch_size} -ub ${ubatch_size}"
    )
  done
done

for ((batch_size=BATCH_START; batch_size<=BATCH_MAX; batch_size+=BATCH_STEP)); do
  for ((ubatch_size=UBATCH_START; ubatch_size<=UBATCH_MAX; ubatch_size+=BATCH_STEP)); do
    if (( ubatch_size > batch_size )); then
      continue
    fi
    for ((draft_tokens=MTP_DRAFT_MIN; draft_tokens<=MTP_DRAFT_MAX; draft_tokens++)); do
      CONFIGS+=(
        "qwen35-mtp-d${draft_tokens}-b${batch_size}-ub${ubatch_size}::-c ${QWEN35_CONTEXT} -ngl ${QWEN35_NGL} -fa ${QWEN35_FA} -ctk ${QWEN35_CTK} -ctv ${QWEN35_CTV} -b ${batch_size} -ub ${ubatch_size} --spec-type draft-mtp --spec-draft-n-max ${draft_tokens}"
      )
    done
  done
done

echo "Benchmarking Qwen3.6-35B with internal MTP"
echo "MODEL=$MODEL"
echo "MMPROJ=$MMPROJ"
echo "OUT_DIR=$OUT_DIR"
echo "RUNS=$RUNS WARMUPS=$WARMUPS MAX_TOKENS=$MAX_TOKENS"
echo "Configs: ${#CONFIGS[@]}"

# Source instead of executing so the CONFIGS bash array remains available to
# benchmark_llama_server_matrix.sh.
# shellcheck source=scripts/benchmark_llama_server_matrix.sh
source "$SCRIPT_DIR/benchmark_llama_server_matrix.sh"
