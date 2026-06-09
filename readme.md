# REVE

**REVE**S: **RE**vision and **VE**rification-Augmented Training for Test-Time Scaling

This repository contains the implementation of REVES, a framework that augments RL training data with revision-verification loops and provides a unified evaluation suite for inference-time search strategies.

---

## Overview

REVES operates in two phases:

1. **RL Data Augmentation** -- Given a base model and seed dataset (e.g., [Skywork-OR1-RL-Data](https://huggingface.co/datasets/Skywork/Skywork-OR1-RL-Data)), the model attempts each problem, and upon failure, enters a *revision-verification loop*: it revises its answer given binary feedback ("incorrect"), then a local verifier judges correctness. Each revision and verification step is recorded as a new training sample. This produces diverse, on-policy training data for GRPO.

2. **Inference-Time Search (Evaluation)** -- At test time, the trained model is evaluated using various search strategies (vanilla, Best-of-N, reflection, MCTS, AB-MCTS, Mind Evolution, etc.) across multiple benchmarks (LiveCodeBench, Countdown, N-Queens, Mini Sudoku, ARC-AGI, TravelPlanner).

---

## Installation

### 1. Clone

```bash
git clone <this-repo-url> REVE
git clone https://github.com/SkyworkAI/Skywork-OR1.git
```

### 2. Data Augmentation Environment

```bash
python -m venv ./reve-env
source ./reve-env/bin/activate
cd REVE
pip install -r requirements-TTS.txt
pip install -e .
```

### 3. RL Training Environment (Skywork-OR1)

```bash
python -m venv ./verl-env
source ./verl-env/bin/activate
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install flash-attn --no-build-isolation
cd Skywork-OR1
pip install -e .
```

---

## RL Data Augmentation

### Step 1: Download Seed Data

```bash
source ./verl-env/bin/activate
cd Skywork-OR1

# Choose model_size: 1p5b, 7b
python ./or1_scripts/data_preprocess/download_and_filter_data_7b.py \
  --local_dir ./or1_data/train
```

This downloads and filters the [Skywork-OR1-RL-Data](https://huggingface.co/datasets/Skywork/Skywork-OR1-RL-Data) into `train_7b_math.pkl` and `train_7b_code.pkl`.

### Step 2: Serve the Model

Launch a vLLM OpenAI-compatible server:

```bash
source ./reve-env/bin/activate

# Single-GPU example
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --served-model-name Qwen2.5-7B-1 \
  --dtype auto \
  --tensor-parallel-size 1 \
  --api-key your-token \
  --port 8000
```

For multi-GPU, see `sample_RL_data.sh` which launches two instances on GPUs 0-3 and 4-7.

### Step 3: Generate Augmented Data

```bash
source ./reve-env/bin/activate
cd REVE

export OPENAI_API_BASE="http://localhost:8000/v1"
export OPENAI_API_KEY=your-token

# Math data
python main_sample_math.py \
  --math_pkl ../Skywork-OR1/or1_data/train/train_7b_math.pkl \
  --output_jsonl ./output/math_augmented.jsonl \
  --output_pkl   ./output/math_augmented.pkl \
  --num_workers 8 \
  --model_type vllm \
  --model_name Qwen2.5-7B-1 \
  --temperature 1.0 \
  --max_tokens 16384 \
  --max_retries 8 \
  --skip_if_initial_correct

# Code data
python main_sample_code.py \
  --code_pkl ../Skywork-OR1/or1_data/train/train_7b_code.pkl \
  --output_jsonl ./output/code_augmented.jsonl \
  --output_pkl   ./output/code_augmented.pkl \
  --num_workers 8 \
  --model_type vllm \
  --model_name Qwen2.5-7B-1 \
  --temperature 1.0 \
  --max_tokens 16384 \
  --max_retries 8 \
  --skip_if_initial_correct
```

For large-scale runs, use `main_sample_math_chunked.py` with `--total_chunks` and `--chunk_id` to parallelize across multiple machines.

### Step 4: Convert to Training Format

```bash
python tools/format_change.py \
  --input_json ./output/math_augmented.jsonl \
  --output_pkl ./output/math_augmented_formatted.pkl \
  --allow_jsonc

python tools/format_change.py \
  --input_json ./output/code_augmented.jsonl \
  --output_pkl ./output/code_augmented_formatted.pkl \
  --allow_jsonc
```

### Step 5: Run RL Training

```bash
source ./verl-env/bin/activate
cd Skywork-OR1
bash run_RL.sh
```

See the [Skywork-OR1](https://github.com/SkyworkAI/Skywork-OR1) repo for training configuration details.

**One-click pipeline:** `sample_RL_data.sh` automates Steps 2-5 (edit the config section at the top before running).

---

## Evaluation (Inference-Time Search)

### Step 1: Serve the Model

```bash
python -m vllm.entrypoints.openai.api_server \
  --model <model-path-or-hf-id> \
  --served-model-name <model-name> \
  --dtype auto \
  --tensor-parallel-size <num-gpus> \
  --api-key your-token \
  --port 8000
```

### Step 2: Run Evaluation

```bash
export OPENAI_API_BASE="http://localhost:8000/v1"
export OPENAI_API_KEY=your-token

python main_parallel_mp.py \
  --task_name <task> \
  --planner <planner> \
  --model_type vllm \
  --model_name <model-name> \
  --temperature 0.0 \
  --max_tokens 4096 \
  --output_dir ./results/ \
  --num_workers 8
```

### Supported Tasks

| Task | `--task_name` | Source |
|------|--------------|--------|
| LiveCodeBench | `LiveCodeBench` | HuggingFace (auto-download) |
| Countdown | `Countdown` | `reasoning_gym` (pip install) |
| N-Queens | `n_queens` | `reasoning_gym` |
| Mini Sudoku | `mini_sudoku` | `reasoning_gym` |
| ARC-AGI | `ARC-AGI` | Local JSONL (set `ARC_AGI_DATA` env var) |
| TravelPlanner | `TravelPlanner` | HuggingFace (auto-download) |
| CodeContest | `CodeContest` | Local JSONL (set `CODECONTEST_JSONL` env var) |

### Supported Planners

| Planner | `--planner` | Key Args |
|---------|------------|----------|
| Vanilla (direct generation) | `vanilla` | -- |
| Best-of-N | `bon` | `--n`, `--num_per_generation` |
| Reflection | `reflection` | `--max_rounds`, `--multiple_reflection` |
| Reflection w/ Confidence | `reflection_with_conf` | `--max_rounds` |
| MCTS | `MCTS` | `--simulations`, `--samples_per_action`, `--multiple_reflection` |
| AB-MCTS (variant A) | `AB_MCTS_A` | `--simulations`, `--dist_type {beta,gaussian}`, `--multiple_reflection` |
| AB-MCTS (variant M) | `AB_MCTS_M` | `--simulations`, `--multiple_reflection` |
| Mind Evolution | `mind_evolution` | `--hyperparams '{"key": val}'` |
| Context Reflection | `ContextReflection` | `--max_rounds`, `--context_window` |

### Examples

```bash
# Best-of-64 on LiveCodeBench
python main_parallel_mp.py \
  --task_name LiveCodeBench \
  --planner bon \
  --n 64 --num_per_generation 64 \
  --model_type vllm --model_name Qwen2.5-7B-1 \
  --temperature 0.6 --max_tokens 16384 \
  --output_dir ./results/ --num_workers 8

# Reflection (up to 32 rounds) on Countdown
python main_parallel_mp.py \
  --task_name Countdown \
  --planner reflection \
  --max_rounds 32 \
  --model_type vllm --model_name Qwen2.5-7B-1 \
  --temperature 0.0 --max_tokens 4096 \
  --output_dir ./results/ --num_workers 8

# AB-MCTS-A (16 simulations, beta prior) on N-Queens
python main_parallel_mp.py \
  --task_name n_queens \
  --planner AB_MCTS_A \
  --simulations 16 --dist_type beta --multiple_reflection \
  --model_type vllm --model_name Qwen2.5-7B-1 \
  --temperature 0.6 --max_tokens 4096 \
  --output_dir ./results/ --num_workers 8
```

---

## Project Structure

```
REVE/
├── main_sample_code.py           # RL data augmentation: code (revision-verification loop)
├── main_sample_math.py           # RL data augmentation: math
├── main_sample_math_chunked.py   # Chunked version for distributed math augmentation
├── main_parallel_mp.py           # Evaluation entry point (inference-time search)
├── sample_RL_data.sh             # One-click: augment data + launch RL training
│
├── planner/                      # Search / planning strategies
│   ├── vanilla_planner.py
│   ├── bon_planner.py            # Best-of-N
│   ├── reflection_planner.py     # Iterative reflection
│   ├── code_reflection_planner.py
│   ├── math_reflection_planner.py
│   ├── MCTS_planner.py           # Monte Carlo Tree Search
│   ├── AB_MCTS_A_planner.py      # Alpha-Beta MCTS (A)
│   ├── AB_MCTS_M_planner.py      # Alpha-Beta MCTS (M)
│   ├── mind_evolution_planner.py
│   └── ...
│
├── evaluator/                    # Task-specific evaluators & scorers
│   ├── livecodebench/            # LiveCodeBench code execution & scoring
│   ├── TravelPlanner/            # TravelPlanner constraint checking
│   ├── eval_countdown.py
│   ├── eval_n_queens.py
│   ├── eval_mini_sudoku.py
│   └── ...
│
├── prompt/                       # Task-specific prompt templates
│   ├── code_pmt.py
│   ├── math_pmt.py
│   ├── LiveCodeBench_pmt.py
│   └── ...
│
├── utils/                        # Shared utilities
│   ├── apis.py                   # LLM API wrapper (OpenAI, vLLM, Gemini, etc.)
│   ├── ab_mcts_a/                # AB-MCTS-A algorithm
│   ├── ab_mcts_m/                # AB-MCTS-M algorithm
│   ├── mcts/                     # Standard MCTS
│   └── ...
│
├── tools/                        # Data processing utilities
│   ├── format_change.py          # JSONL/JSON -> PKL converter
│   └── ...
│
├── scripts/sample_RL/            # Shell scripts for distributed data generation
├── requirements-TTS.txt
└── setup.py
```
