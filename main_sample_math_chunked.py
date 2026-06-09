# main_sample_math_chunked.py
# -*- coding: utf-8 -*-
import os
import json
import pickle
import argparse
import random
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from planner.math_reflection_planner import MathReflectionPlanner


def _load_pkl(path: str) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        return pickle.load(f)


def _ensure_dir(p: str):
    d = os.path.dirname(p)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _item_to_source(item: Dict[str, Any]) -> Optional[str]:
    for k in ["data_source", "source", "dataset", "name"]:
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None

def _records_to_or1_items(
    recs: List[Dict[str, Any]],
    idx: int,
    ability: str = "math",
    data_source: Optional[str] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rec in recs:
        prm = rec.get("prompt", [])
        rm = rec.get("reward_model", {})
        gt = None
        if isinstance(rm, dict):
            gt = rm.get("ground_truth")  # Note: this is a JSON string, e.g. "[\"True\"]" or "[\"15625\"]"

        # Filter to keep only user messages
        filtered_prompt = [
            m for m in prm
            if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str)
        ]

        if not filtered_prompt or not isinstance(gt, str) or not gt.strip():
            continue

        one_item = {
            "prompt": filtered_prompt,
            # Keep the reward_model structure and string ground_truth consistent with Skywork-OR1
            "reward_model": {"ground_truth": gt, "style": "rule"},
            "extra_info": {"index": idx, "ability": ability},
        }
        if data_source:
            one_item["extra_info"]["data_source"] = data_source
        out.append(one_item)
    return out
    
def _one_item_worker(args: Tuple[int, Dict[str, Any], Dict[str, Any], int, bool]) -> Tuple[int, Optional[List[Dict[str, Any]]], Optional[str]]:
    idx, item, llm_cfg, max_retries, skip_if_initial_correct = args
    try:
        planner = MathReflectionPlanner(llm_config=llm_cfg, max_retries=max_retries)
        recs = planner.run(item, skip_if_initial_correct=skip_if_initial_correct)
        return idx, recs, _item_to_source(item)
    except Exception:
        return idx, None, _item_to_source(item)


def main():
    parser = argparse.ArgumentParser(description="Reflection+verification data builder for MATH (OR1 schema) - Chunked version")
    parser.add_argument("--math_pkl", required=True, help="Path to sampled MATH PKL.")
    parser.add_argument("--output_jsonl", required=True, help="Output JSONL (OR1 records).")
    parser.add_argument("--output_pkl", required=True, help="Output PKL (OR1 records).")
    parser.add_argument("--max_items", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--model_type", default="vllm")
    parser.add_argument("--model_name", default="qwen-7b")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=4096)

    parser.add_argument("--max_retries", type=int, default=8)
    parser.add_argument("--skip_if_initial_correct", action="store_true")
    
    # New chunk-related arguments
    parser.add_argument("--total_chunks", type=int, required=True, help="Total number of chunks this dataset is split into")
    parser.add_argument("--chunk_id", required=True, help="Which chunk this script is processing (e.g., '3a', '2b')")

    args = parser.parse_args()

    _ensure_dir(args.output_jsonl)
    _ensure_dir(args.output_pkl)

    items_all = _load_pkl(args.math_pkl)
    items = [it for it in items_all if isinstance(it, dict) and it.get("ability", "").lower() == "math"]

    if args.shuffle:
        random.shuffle(items)
    
    # Calculate chunk boundaries
    total_items = len(items)
    chunk_size = total_items // args.total_chunks
    remainder = total_items % args.total_chunks
    
    # Parse chunk_id to get the chunk number (e.g., "3a" -> 3, "2b" -> 2)
    chunk_num = int(''.join(filter(str.isdigit, args.chunk_id)))
    
    if chunk_num < 1 or chunk_num > args.total_chunks:
        raise ValueError(f"chunk_id {args.chunk_id} is invalid. Must be between 1 and {args.total_chunks}")
    
    # Calculate start and end indices for this chunk
    start_idx = 0
    for i in range(1, chunk_num):
        current_chunk_size = chunk_size + (1 if i <= remainder else 0)
        start_idx += current_chunk_size
    
    current_chunk_size = chunk_size + (1 if chunk_num <= remainder else 0)
    end_idx = start_idx + current_chunk_size
    
    # Extract this chunk's data
    chunk_items = items[start_idx:end_idx]
    
    print(f"Processing chunk {args.chunk_id}/{args.total_chunks}")
    print(f"Total items: {total_items}")
    print(f"Chunk {chunk_num} range: {start_idx} to {end_idx-1} ({len(chunk_items)} items)")
    
    if args.max_items > 0:
        chunk_items = chunk_items[:args.max_items]
        print(f"Limited to {args.max_items} items")

    llm_cfg = {
        "model_type": args.model_type,
        "model_name": args.model_name,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }

    tasks: List[Tuple[int, Dict[str, Any], Dict[str, Any], int, bool]] = [
        (i, item, llm_cfg, args.max_retries, args.skip_if_initial_correct) for i, item in enumerate(chunk_items, start=start_idx+1)
    ]

    kept_items, total_items = 0, 0
    all_or1_records: List[Dict[str, Any]] = []

    with open(args.output_jsonl, "w", encoding="utf-8") as f_jsonl:
        with ProcessPoolExecutor(max_workers=max(1, args.num_workers)) as pool:
            futures = {pool.submit(_one_item_worker, t): t[0] for t in tasks}
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Building math records (chunk {args.chunk_id})"):
                total_items += 1
                idx, recs, data_source = fut.result()
                if not recs:
                    continue
                or1_items = _records_to_or1_items(recs, idx=idx, ability="math", data_source=data_source)
                if not or1_items:
                    continue
                kept_items += 1
                all_or1_records.extend(or1_items)
                for it in or1_items:
                    f_jsonl.write(json.dumps(it, ensure_ascii=False) + "\n")

    with open(args.output_pkl, "wb") as f_pkl:
        pickle.dump(all_or1_records, f_pkl)

    print(f"[DONE] Chunk {args.chunk_id}: kept {kept_items} / {total_items} math items")
    print(f"[DONE] jsonl -> {args.output_jsonl}")
    print(f"[DONE] pkl   -> {args.output_pkl}")


if __name__ == "__main__":
    main()
