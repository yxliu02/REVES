import sys
import os
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, Tuple
from datasets import load_dataset
from tqdm import tqdm
from utils.utils import extract_plan_obj, extract_code_block, extract_countdown_answer, extract_n_queens_board, extract_mini_sudoku_board

# Ensure project root is on path
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_dir)


def load_data(task_name: str):
    if task_name == "TravelPlanner":
        return load_dataset("osunlp/TravelPlanner", "validation")["validation"]
    if task_name == "NaturalPlan-meeting":
        with open("natural-plan-data/meeting_planning.json", "r", encoding="utf-8") as f:
            return json.load(f)
    if task_name == "NaturalPlan-trip":
        with open("natural-plan-data/trip_planning.json", "r", encoding="utf-8") as f:
            return json.load(f)
    if task_name == "LiveCodeBench":
        from utils.code_generation import load_code_generation_dataset
        return load_code_generation_dataset(start_date="2024-08-01", end_date="2025-01-01",release_version="release_v6")
    if task_name == "CodeContest":
        from utils.code_generation import load_code_generation_dataset_from_jsonl
        return load_code_generation_dataset_from_jsonl(
            os.environ.get("CODECONTEST_JSONL", "codecontests_lcb_test.jsonl")
        )
    if task_name == "Countdown":
        import reasoning_gym
        return reasoning_gym.create_dataset(
            name="countdown",
            size=100,
            seed=42                  
        )
    if task_name == "n_queens":
        import reasoning_gym
        return reasoning_gym.create_dataset(
            name="n_queens",
            size=100,
            seed=42,
            n=6,
            min_remove=1,
            max_remove=3,  # must be <= n
        )
    if task_name == "mini_sudoku":
        import reasoning_gym
        return reasoning_gym.create_dataset(
            name="mini_sudoku",
            size=100,        # dataset size
            seed=42,         # random seed
            min_empty=6,     # minimum empty cells (lower difficulty bound)
            max_empty=12,    # maximum empty cells (upper difficulty bound)
        )
    if task_name == "ARC-AGI":
        with open(os.environ.get("ARC_AGI_DATA", "arc_agi_merged.json"), "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    raise ValueError("Unsupported task_name")


def make_planner(args, llm_cfg) -> Any:
    """Instantiate the appropriate planner based on args."""
    if args.planner == "vanilla":
        from planner.vanilla_planner import Planner
        return Planner(args.task_name, prev_response="", llm_config=llm_cfg)
    if args.planner == "bon":
        from planner.bon_planner import BoNPlanner
        return BoNPlanner(args.task_name, n=args.n, num_per_generation=args.num_per_generation,
                          prev_response="", llm_config=llm_cfg)
    if args.planner == "reflection":
        from planner.reflection_planner import ReflectionPlanner
        return ReflectionPlanner(args.task_name, max_rounds=args.max_rounds, multiple_reflection=args.multiple_reflection,
                                 prev_response="", llm_config=llm_cfg)
    if args.planner == "reflection_with_conf":
        from planner.reflection_planner_with_conf import ReflectionPlanner
        return ReflectionPlanner(args.task_name, max_rounds=args.max_rounds, multiple_reflection=args.multiple_reflection,
                                 prev_response="", llm_config=llm_cfg)
    if args.planner == "MCTS":
        from planner.MCTS_planner import MCTSPlanner
        return MCTSPlanner(args.task_name, simulations=args.simulations, samples_per_action=args.samples_per_action, 
                           multiple_reflection=args.multiple_reflection, prev_response="", llm_config=llm_cfg)
    if args.planner == "mind_evolution":
        from planner.mind_evolution_planner import MindEvolutionPlanner
        return MindEvolutionPlanner(args.task_name, prev_response="", hyperparams=args.hyperparams, llm_config=llm_cfg)
    if args.planner == "AB_MCTS_M":
        from planner.AB_MCTS_M_planner import AB_MCTS_MPlanner
        return AB_MCTS_MPlanner(args.task_name, simulations=args.simulations, multiple_reflection=args.multiple_reflection,
                               prev_response="", llm_config=llm_cfg)
    if args.planner == "AB_MCTS_A":
        from planner.AB_MCTS_A_planner import AB_MCTS_APlanner
        return AB_MCTS_APlanner(args.task_name, simulations=args.simulations, dist_type=args.dist_type, multiple_reflection=args.multiple_reflection,
                               prev_response="", llm_config=llm_cfg)
    if args.planner == "test1":
        from planner.test1_planner import TestPlanner1
        return TestPlanner1(args.task_name, max_rounds=args.max_rounds, n=args.n, multiple_reflection=args.multiple_reflection,
                                 prev_response="", llm_config=llm_cfg)
    if args.planner == "ContextReflection":
        from planner.context_reflection_palnner import ContextReflectionPlanner
        return ContextReflectionPlanner(args.task_name, max_rounds=args.max_rounds, context_window=args.context_window,
                                 prev_response="", llm_config=llm_cfg)
    if args.planner == "test2":
        from planner.test2_planner import TestPlanner2
        return TestPlanner2(args.task_name, max_rounds=args.max_rounds,
                                 prev_response="", llm_config=llm_cfg)
    if args.planner == "test3":
        from planner.test3_planner import TestPlanner3
        return TestPlanner3(args.task_name, max_rounds=args.max_rounds,
                                 prev_response="", llm_config=llm_cfg)
    if args.planner == "test4":
        from planner.test4_planner import TestPlanner4
        return TestPlanner4(args.task_name, max_rounds=args.max_rounds,
                                 prev_response="", llm_config=llm_cfg)

    raise ValueError("Unsupported planner type")


def process_entry(params: Tuple[int, Dict[str, Any], Dict[str, Any], argparse.Namespace]) -> Dict[str, Any] | None:
    """
    params: (idx, entry, llm_cfg, args_dict)
    args_dict must include 'planner', 'task_name', 'n', 'num_responses',
    'max_rounds', 'depth', 'width', and 'completed_indices'.
    """
    idx, entry, llm_cfg, args_dict = params
    if idx in args_dict['completed_indices']:
        return None

    try:
        # Reconstruct args Namespace minimally
        class TempArgs:
            pass
        targs = TempArgs()
        for k, v in args_dict.items():
            setattr(targs, k, v)

        planner = make_planner(targs, llm_cfg)
        if targs.task_name == "TravelPlanner":
            plan_txt, usage, attempt_count = planner.run(entry)
            plan_obj = extract_plan_obj(plan_txt)
            return {
                "idx": idx,
                "query": entry.get("query"),
                "plan": plan_obj,
                "task_name": targs.task_name,
                "usage": usage,
                "attempt_count": attempt_count
            }
        elif targs.task_name == "LiveCodeBench" or targs.task_name == "CodeContest":
            # entry is a CodeGenerationProblem dataclass; generate and extract code
            # print(planner.run(entry))
            # print(entry)
            raw_txt, usage, attempt_count = planner.run(entry)
            code = extract_code_block(raw_txt)
            # Safely get question_id via getattr fallback
            qid = getattr(entry, "question_id", None) or (entry.get("question_id") if isinstance(entry, dict) else None)
            return {
                "idx": idx,
                "question_id": qid,
                "code": code,
                "task_name": targs.task_name,
                "usage": usage,
                "attempt_count": attempt_count
            }
        elif targs.task_name == "ARC-AGI":
            raw_txt, usage, attempt_count = planner.run(entry)
            code = extract_code_block(raw_txt)
            # Safely get question_id via getattr fallback
            return {
                "idx": idx,
                "code": code,
                "task_name": targs.task_name,
                "usage": usage,
                "attempt_count": attempt_count
            }
        elif targs.task_name == "Countdown":
            raw_txt, usage, attempt_count = planner.run(entry)
            answer = extract_countdown_answer(raw_txt)
            return {
                "idx": idx,
                "raw_txt": raw_txt,
                "answer": answer,
                "task_name": targs.task_name,
                "usage": usage,
                "attempt_count": attempt_count
            }
        elif targs.task_name == "n_queens":
            raw_txt, usage, attempt_count = planner.run(entry)
            board = extract_n_queens_board(raw_txt)
            from evaluator.eval_n_queens import evaluate_n_queens
            score_dict = evaluate_n_queens(entry, board)
            correctness = True if score_dict["score"] == 0 else False
            return {
                "idx": idx,
                "raw_txt": raw_txt,
                "board": board,
                "correctness": correctness,
                "task_name": targs.task_name,
                "usage": usage,
                "attempt_count": attempt_count
            }
        elif targs.task_name == "mini_sudoku":
            raw_txt, usage, attempt_count = planner.run(entry)
            board = extract_mini_sudoku_board(raw_txt)
            from evaluator.eval_mini_sudoku import evaluate_mini_sudoku
            score_dict = evaluate_mini_sudoku(entry, board)
            correctness = True if score_dict["score"] == 0 else False
            return {
                "idx": idx,
                "raw_txt": raw_txt,
                "board": board,
                "correctness": correctness,
                "task_name": targs.task_name,
                "usage": usage,
                "attempt_count": attempt_count
            }
        else:
            raise ValueError("Unsupported task_name")
    except Exception as e:
        # Catch all exceptions to prevent pickle errors with APIStatusError etc.
        print(f"[ERROR] process_entry idx={idx} failed: {type(e).__name__}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Batch-run planner over validation split with resume and in-order output."
    )

    # I/O + core options
    parser.add_argument("--output_dir", required=True, help="Directory to write output JSONL files into.")
    parser.add_argument("--task_name", default="TravelPlanner", help="Task name used for filename and records.")

    # LLM options
    parser.add_argument("--model_type", default="gemini", help="LLM API type: openai, gemini, or vllm.")
    parser.add_argument("--model_name", default="gemini-1.5-flash", help="LLM model name to use.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for LLM.")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max tokens for LLM generation.")

    # Planner family and hyper‑params
    parser.add_argument("--planner", default="vanilla", help="Planner type")
    parser.add_argument("--n", type=int, default=64, help="Total responses required (planner==bon)")
    parser.add_argument("--num_per_generation", type=int, default=64, help="Num per generation (planner==bon)")
    parser.add_argument("--max_rounds", type=int, default=64, help="Max rounds for reflection (planner==reflection)")
    parser.add_argument("--simulations", type=int, default=16, help="Simulations for MCTS planner (planner==MCTS/AB_MCTS)")
    parser.add_argument("--samples_per_action", type=int, default=4, help="samples_per_action for MCTS planner (planner==MCTS)")
    parser.add_argument("--dist_type", type=str, default="beta", help="choose from {beta, gaussian} (planner==AB_MCTS_A)")
    parser.add_argument("--multiple_reflection", action="store_true", help="If set, enables multiple reflection mode. (planner == AB_MCTS/MCTS)")
    parser.add_argument("--context_window", type=int, default=-1, help="How many history response should be included in the reflection, -1 means all the histroy")
    parser.add_argument("--hyperparams", type=json.loads, default={}, help="The parameters for Mind Evolution")

    # Concurrency
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of processes (0 or 1 -> sequential)")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Base filenames
    base_filename = f"{args.model_name}_{args.task_name}_{args.planner}_{args.temperature}_{args.max_tokens}"
    if args.planner in ["reflection", "MCTS", "AB_MCTS_M", "AB_MCTS_A", "test1"]:
        base_filename = base_filename + f"_multiple_reflection_{args.multiple_reflection}"
    if args.planner == "AB_MCTS_A":
        base_filename = base_filename + f"_{args.dist_type}" + f"_{args.simulations}"
    if args.planner == "bon":
        base_filename = base_filename + f"_{args.n}"
    if args.planner == "reflection" or args.planner == "reflection_with_conf":
        base_filename = base_filename + f"_{args.max_rounds}"
    if args.planner == "MCTS":
        base_filename = base_filename + f"_{args.simulations}_{args.samples_per_action}"
    if args.planner == "AB_MCTS_M":
        base_filename = base_filename + f"_{args.simulations}"
    if args.planner == "ContextReflection":
        base_filename = base_filename + f"_context_window_{args.context_window}"

    output_path = os.path.join(args.output_dir, base_filename + ".jsonl")
    ordered_path = os.path.join(args.output_dir, base_filename + "_in_order.jsonl")

    data_split = load_data(args.task_name)

    # Prepare LLM config
    llm_cfg: Dict[str, Any] = {
        "model_type": args.model_type,
        "model_name": args.model_name,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }

    # Load existing indices if resuming
    completed_indices = set()
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    completed_indices.add(rec.get('idx'))
                except json.JSONDecodeError:
                    continue
        print(f"Resuming: found {len(completed_indices)} completed entries, skipping them.")

    # Pack args into a simple dict for pickling
    args_dict = {**vars(args), 'completed_indices': completed_indices}

    # Prepare tasks
    tasks = [
        (i, e, llm_cfg, args_dict)
        for i, e in enumerate(data_split, start=1)
        if i not in completed_indices
    ]

    # Run and write
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as fout:
        num_workers = max(1, args.num_workers)
        if num_workers <= 1:
            for rec in tqdm((process_entry(t) for t in tasks), total=len(tasks), desc="Processing entries"):
                if rec:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        else:
            with ProcessPoolExecutor(max_workers=num_workers) as pool:
                futures = {pool.submit(process_entry, t): t[0] for t in tasks}
                for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing entries"):
                    rec = fut.result()
                    if rec:
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Final in-order output
    print("Generating in-order output...")
    all_recs = []
    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                all_recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    all_recs.sort(key=lambda x: x.get('idx', 0))
    with open(ordered_path, 'w', encoding='utf-8') as fout:
        for rec in all_recs:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_recs)} records to {output_path} and {ordered_path}")


if __name__ == "__main__":
    main()
