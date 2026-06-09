#!/usr/bin/env bash
set -euo pipefail
set -x

tmux kill-session -t 0 || true
tmux kill-session -t 1 || true
# ========= USER CONFIG =========
export EVAL_MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct   # HuggingFace model path
export EVAL_MODEL_NAME=Qwen2.5-1.5B                   # Short name for served model
export BASE_DIR=$(pwd)/..                              # Parent dir containing both repos
export EVAL_BASE_DIR="${BASE_DIR}/rl_data_output"      # Output directory for generated data
export TTS_VENV=""                                     # Path to TTS venv activate script (leave empty if not using venv)
export VERL_VENV=""                                    # Path to verl-orz venv activate script (leave empty if not using venv)
# ================================

mkdir -p "${EVAL_BASE_DIR}"

activate_tts() {
  if [ -n "$TTS_VENV" ]; then source "$TTS_VENV"; fi
}

# Start the first vLLM instance (GPUs 0-3, port 8000)
tmux new-session -d -s 0 "
export CUDA_VISIBLE_DEVICES=0,1,2,3;
$([ -n \"$TTS_VENV\" ] && echo \"source $TTS_VENV;\")
exec python -m vllm.entrypoints.openai.api_server \
  --model $EVAL_MODEL_PATH \
  --served-model-name ${EVAL_MODEL_NAME}-1 \
  --dtype auto \
  --tensor-parallel-size 4 \
  --api-key your-token-abc123 \
  --port 8000
"
echo "[tmux:0] launched ${EVAL_MODEL_NAME}-1 on port 8000 (GPUs 0-3)."

sleep 300

# Start the second vLLM instance (GPUs 4-7, port 8001)
tmux new-session -d -s 1 "
export CUDA_VISIBLE_DEVICES=4,5,6,7;
$([ -n \"$TTS_VENV\" ] && echo \"source $TTS_VENV;\")
python -m vllm.entrypoints.openai.api_server \
  --model $EVAL_MODEL_PATH \
  --served-model-name ${EVAL_MODEL_NAME}-2 \
  --dtype auto \
  --tensor-parallel-size 4 \
  --api-key your-token-abc1234 \
  --port 8001
"
echo "[tmux:1] launched ${EVAL_MODEL_NAME}-2 on port 8001 (GPUs 4-7)."
sleep 300

activate_tts
cd ${BASE_DIR}/TTS_codebase

tmux kill-session -t 5 || true
tmux kill-session -t 6 || true

# Launch code sampling
tmux new-session -d -s 5 "
export EVAL_MODEL_NAME=$EVAL_MODEL_NAME;
export EVAL_BASE_DIR=$EVAL_BASE_DIR;
export BASE_DIR=$BASE_DIR;
$([ -n \"$TTS_VENV\" ] && echo \"source $TTS_VENV;\")
bash ./scripts/sample_RL/code_auto.sh
"

# Launch math sampling
tmux new-session -d -s 6 "
export EVAL_MODEL_NAME=$EVAL_MODEL_NAME;
export EVAL_BASE_DIR=$EVAL_BASE_DIR;
export BASE_DIR=$BASE_DIR;
$([ -n \"$TTS_VENV\" ] && echo \"source $TTS_VENV;\")
bash ./scripts/sample_RL/math_auto.sh
"

tmux wait-for code_done
tmux wait-for math_done

# Convert to PKL format for training
python ./tools/format_change.py \
  --input_json ${EVAL_BASE_DIR}/math_two_part.jsonl \
  --output_pkl ${EVAL_BASE_DIR}/math_two_part_formated.pkl \
  --allow_jsonc

python ./tools/format_change.py \
  --input_json ${EVAL_BASE_DIR}/code_two_part.jsonl \
  --output_pkl ${EVAL_BASE_DIR}/code_two_part_formated.pkl \
  --allow_jsonc

# Switch to verl-orz env and start RL training
cd ${BASE_DIR}
if [ -n "$VERL_VENV" ]; then source "$VERL_VENV"; fi
bash ${BASE_DIR}/Skywork-OR1/run_RL.sh
