#!/bin/bash
# ============================
# Run math reflection pipeline
# ============================

set -e  # Exit on any command failure
set -u  # Error on undefined variables

# Activate your TTS venv before running this script

# Output directory
OUT_DIR=${EVAL_BASE_DIR}
mkdir -p "$OUT_DIR"

# Log file
MATH_LOG_FILE="$OUT_DIR/run_math_sample_2.log"

# Launch parameters
PYTHON=python
MAIN=${BASE_DIR}/TTS_codebase/main_sample_math_chunked.py

export OPENAI_API_BASE="http://localhost:8001/v1"
export OPENAI_API_KEY=your-token-abc1234
export MODEL_NAME="${EVAL_MODEL_NAME}-2"

$PYTHON $MAIN \
  --math_pkl ${BASE_DIR}/Skywork-OR1/or1_data/train/train_7b_math.pkl \
  --output_jsonl ${OUT_DIR}/math_world_2_id_2.jsonl \
  --output_pkl   ${OUT_DIR}/math_world_2_id_2.pkl \
  --num_workers 8 \
  --max_retries 8 \
  --skip_if_initial_correct \
  --model_type vllm \
  --model_name $MODEL_NAME \
  --temperature 1.0 \
  --max_tokens 16384 \
  --total_chunks 2 \
  --chunk_id "2" \
  > >(tee -a "$MATH_LOG_FILE") 2>&1

tmux wait-for -S math_done