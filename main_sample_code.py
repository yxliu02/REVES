# -*- coding: utf-8 -*-
import os
import json
import pickle
import argparse
import random
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from planner.code_reflection_planner import CodeReflectionPlanner


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
    ability: str = "code",
    data_source: Optional[str] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rec in recs:
        prm = rec.get("prompt", [])
        rm = rec.get("reward_model", {})
        gt = rm.get("ground_truth")
        if not isinstance(prm, list) or not isinstance(gt, str) or not gt:
            continue

        filtered_prompt = [
            m for m in prm if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str)
        ]
        if not filtered_prompt:
            continue

        one_item = {
            "prompt": filtered_prompt,
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
        planner = CodeReflectionPlanner(llm_config=llm_cfg, max_retries=max_retries)
        recs = planner.run(item, skip_if_initial_correct=skip_if_initial_correct)
        return idx, recs, _item_to_source(item)
    except Exception:
        return idx, None, _item_to_source(item)


def main():
    parser = argparse.ArgumentParser(description="Reflection+verification data builder for CODE (OR1 schema)")
    parser.add_argument("--code_pkl", required=True, help="Path to sampled CODE PKL.")
    parser.add_argument("--output_jsonl", required=True, help="Output JSONL (OR1 records).")
    parser.add_argument("--output_pkl", required=True, help="Output PKL (OR1 records).")
    parser.add_argument("--max_items", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--num_workers", type=int, default=8)

    # LLM
    parser.add_argument("--model_type", default="vllm")
    parser.add_argument("--model_name", default="qwen-7b")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=4096)

    # Planner hyperparameters
    parser.add_argument("--max_retries", type=int, default=8)
    parser.add_argument("--skip_if_initial_correct", action="store_true")

    args = parser.parse_args()

    _ensure_dir(args.output_jsonl)
    _ensure_dir(args.output_pkl)

    items_all = _load_pkl(args.code_pkl)
    # Only keep items with ability == "code"
    items = [it for it in items_all if isinstance(it, dict) and it.get("ability", "").lower() == "code"]

    if args.shuffle:
        random.shuffle(items)
    if args.max_items > 0:
        items = items[:args.max_items]

    llm_cfg = {
        "model_type": args.model_type,
        "model_name": args.model_name,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }

    tasks: List[Tuple[int, Dict[str, Any], Dict[str, Any], int, bool]] = [
        (i, item, llm_cfg, args.max_retries, args.skip_if_initial_correct) for i, item in enumerate(items, start=1)
    ]

    kept_items, total_items = 0, 0
    all_or1_records: List[Dict[str, Any]] = []

    with open(args.output_jsonl, "w", encoding="utf-8") as f_jsonl:
        with ProcessPoolExecutor(max_workers=max(1, args.num_workers)) as pool:
            futures = {pool.submit(_one_item_worker, t): t[0] for t in tasks}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Building code records"):
                total_items += 1
                idx, recs, data_source = fut.result()
                if not recs:
                    continue
                or1_items = _records_to_or1_items(recs, idx=idx, ability="code", data_source=data_source)
                if not or1_items:
                    continue
                kept_items += 1
                all_or1_records.extend(or1_items)
                for it in or1_items:
                    f_jsonl.write(json.dumps(it, ensure_ascii=False) + "\n")

    with open(args.output_pkl, "wb") as f_pkl:
        pickle.dump(all_or1_records, f_pkl)

    print(f"[DONE] kept {kept_items} / {total_items} code items")
    print(f"[DONE] jsonl -> {args.output_jsonl}")
    print(f"[DONE] pkl   -> {args.output_pkl}")


if __name__ == "__main__":
    main()
