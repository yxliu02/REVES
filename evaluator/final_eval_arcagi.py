# -*- coding: utf-8 -*-
from __future__ import annotations
import json
import argparse
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple, Optional
from collections import Counter

###############################################################################
# IO helpers (supports JSON / JSONL)
###############################################################################

def read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
    if not txt:
        return []
    # If it starts with '[', it is likely a JSON array; otherwise parse as JSONL line by line
    if txt[0] == "[":
        data = json.loads(txt)
        if isinstance(data, list):
            return data
        raise ValueError(f"{path} is a JSON object, expected list.")
    # JSONL
    return [json.loads(line) for line in txt.splitlines() if line.strip()]

###############################################################################
# ARC comparison and subprocess execution
###############################################################################

def grids_equal(a: Any, b: Any) -> bool:
    # ARC grids are list[list[int]]; == compares element-wise directly
    return a == b

def _worker(code: str, inputs: List[List[List[int]]], ret):
    try:
        g, l = {}, {}
        exec(code, g, l)
        f = l.get("transform") or g.get("transform")
        if not callable(f):
            ret["err"] = "No callable transform()"
            return
        ret["preds"] = [f(x) for x in inputs]
    except Exception as e:
        ret["err"] = repr(e)

def run_code(code: str, inputs: List[List[List[int]]], timeout: int = 8) -> Tuple[Optional[List[Any]], Optional[str]]:
    ctx = mp.get_context("spawn")
    mgr = ctx.Manager()
    ret = mgr.dict()
    p = ctx.Process(target=_worker, args=(code, inputs, ret))
    p.start()
    p.join(timeout=timeout)
    if p.is_alive():
        p.kill()
        p.join()
        return None, "Timeout"
    if "err" in ret:
        return None, ret["err"]
    return ret.get("preds"), None

###############################################################################
# Single-problem evaluation: solved only if all train examples pass
###############################################################################

def eval_one_problem(problem: Dict[str, Any], code: str, timeout: int = 8) -> Tuple[bool, Optional[str]]:
    train = problem.get("train", []) or []
    inputs  = [ex["input"]  for ex in train]
    outputs = [ex["output"] for ex in train]
    preds, err = run_code(code, inputs, timeout=timeout)
    if err or preds is None:
        return False, err or "UnknownError"
    return all(grids_equal(p, gt) for p, gt in zip(preds, outputs)), None

###############################################################################
# Main logic: load ARC data and code outputs, build a multi-key mapping, and compute accuracy
###############################################################################

def main():
    ap = argparse.ArgumentParser(description="Evaluate ARC-AGI solutions (train-only exact match).")
    ap.add_argument("--data_json", required=True, help="ARC-AGI dataset (JSON or JSONL; for JSONL, one problem dict per line)")
    ap.add_argument("--codes_file", required=True, help="Model outputs (JSON or JSONL; each entry must contain at least idx and code)")
    ap.add_argument("--timeout", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--debug", action="store_true", help="Print unmatched/error statistics for debugging")
    args = ap.parse_args()

    # Load dataset
    problems = read_json_or_jsonl(args.data_json)
    if not problems:
        print("Solved: 0/0 = 0.00%")
        return

    # Build a multi-key mapping from idx -> problem:
    # Supports sequential index (1-based), problem['idx'], and problem['id']; both raw and str() values are mapped.
    idx2prob: Dict[Any, Dict[str, Any]] = {}
    for i, prob in enumerate(problems, start=1):
        keys = {i, str(i)}  # Sequential index is always available
        if "idx" in prob:
            keys.add(prob["idx"])
            keys.add(str(prob["idx"]))
        if "id" in prob:
            keys.add(prob["id"])
            keys.add(str(prob["id"]))
        for k in keys:
            idx2prob[k] = prob

    # Load code results (supports JSON/JSONL)
    recs = read_json_or_jsonl(args.codes_file)

    tasks: List[Tuple[Any, Dict[str, Any], str]] = []
    miss = 0
    for rec in recs:
        if not isinstance(rec, dict):
            continue
        key = rec.get("idx", None)
        code = rec.get("code", "")
        if key is None or not code:
            continue
        prob = idx2prob.get(key) or idx2prob.get(str(key))
        if prob is None:
            miss += 1
            continue
        tasks.append((key, prob, code))

    total = len(tasks)
    if total == 0:
        if args.debug:
            print(f"[debug] No matched tasks. Unmatched outputs: {miss}")
        print("Solved: 0/0 = 0.00%")
        return

    solved = 0
    err_stats = Counter()

    # Concurrent evaluation
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        fut2idx = {ex.submit(eval_one_problem, prob, code, args.timeout): key for (key, prob, code) in tasks}
        for fut in as_completed(fut2idx):
            ok = False
            try:
                ok, err = fut.result()
                if err:
                    err_stats[err] += 1
            except Exception as e:
                err_stats["ExecutorError"] += 1
                ok = False
            solved += 1 if ok else 0

    acc = solved / total if total else 0.0
    if args.debug:
        print(f"[debug] total matched: {total}, unmatched outputs: {miss}")
        if err_stats:
            print(f"[debug] error stats: {dict(err_stats)}")
    print(f"Solved: {solved}/{total} = {acc:.2%}")

if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    main()
