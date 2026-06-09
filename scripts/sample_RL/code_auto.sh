#!/usr/bin/env bash
set -euo pipefail

PY=python

# Activate your TTS venv before running this script


export OPENAI_API_BASE="http://localhost:8000/v1"
export OPENAI_API_KEY=your-token-abc123
export MODEL_NAME="${EVAL_MODEL_NAME}-1"

CODE_PKL=${BASE_DIR}/Skywork-OR1/or1_data/train/train_7b_code.pkl
OUT_DIR=${EVAL_BASE_DIR}
mkdir -p "${OUT_DIR}"
CODE_LOG_FILE="$OUT_DIR/run_code_sample.log"

${PY} main_sample_code.py \
  --code_pkl "${CODE_PKL}" \
  --output_jsonl "${OUT_DIR}/code_two_part.jsonl" \
  --output_pkl   "${OUT_DIR}/code_two_part.pkl" \
  --max_items 0 \
  --num_workers 8 \
  --model_type vllm \
  --model_name $MODEL_NAME \
  --temperature 1.0 \
  --max_tokens 16384 \
  --max_retries 8 \
  --skip_if_initial_correct \
  > >(tee -a "$CODE_LOG_FILE") 2>&1

# Log file
MATH_LOG_FILE="$OUT_DIR/run_math_sample_1.log"

# Launch parameters
MAIN=${BASE_DIR}/TTS_codebase/main_sample_math_chunked.py


${PY} $MAIN \
  --math_pkl ${BASE_DIR}/Skywork-OR1/or1_data/train/train_7b_math.pkl \
  --output_jsonl ${OUT_DIR}/math_world_2_id_1.jsonl \
  --output_pkl   ${OUT_DIR}/math_world_2_id_1.pkl \
  --num_workers 8 \
  --max_retries 8 \
  --skip_if_initial_correct \
  --model_type vllm \
  --model_name $MODEL_NAME \
  --temperature 1.0 \
  --max_tokens 16384 \
  --total_chunks 2 \
  --chunk_id "1" \
  > >(tee -a "$MATH_LOG_FILE") 2>&1

tmux wait-for -S code_done