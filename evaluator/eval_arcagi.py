# -*- coding: utf-8 -*-
from __future__ import annotations
import json
import sys
import argparse
import multiprocessing as mp
from typing import Any, Dict, List, Tuple, Optional

###############################################################################
# Utils
###############################################################################

def _grids_equal(a: Any, b: Any) -> bool:
    """Strict equality for nested list[int] grids."""
    if not (isinstance(a, list) and isinstance(b, list)):
        return False
    if len(a) != len(b):
        return False
    for r1, r2 in zip(a, b):
        if not (isinstance(r1, list) and isinstance(r2, list)):
            return False
        if len(r1) != len(r2):
            return False
        for x, y in zip(r1, r2):
            if x != y:
                return False
    return True

def _extract_train_test(data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    train = data.get("train", []) or []
    test = data.get("test", []) or []
    return train, test

###############################################################################
# Worker (runs in an isolated subprocess)
###############################################################################

def _worker_run(code: str, inputs: List[List[List[int]]], ret_dict):
    """
    Execute `code` (should define transform(grid) -> grid) and run on inputs.
    Returns via ret_dict:
      {
        "ok": bool,
        "preds": List[grid] | None,
        "error_code": Optional[int],
        "error_message": Optional[str],
        "parse_error": Optional[str],
        "runtime_error": Optional[str],
      }
    """
    result: Dict[str, Any] = {
        "ok": False, "preds": None,
        "error_code": None, "error_message": None,
        "parse_error": None, "runtime_error": None,
    }
    try:
        # Execute user code in a fresh namespace
        g: Dict[str, Any] = {}
        l: Dict[str, Any] = {}
        exec(code, g, l)
        transform = l.get("transform") or g.get("transform")
        if not callable(transform):
            result["parse_error"] = "No callable `transform` found in code."
            ret_dict.update(result)
            return

        preds: List[Any] = []
        for idx, grid in enumerate(inputs):
            try:
                out = transform(grid)
            except Exception as e:
                result["runtime_error"] = f"Exception on case {idx}: {e!r}"
                ret_dict.update(result)
                return
            preds.append(out)

        result["ok"] = True
        result["preds"] = preds
        ret_dict.update(result)
    except SyntaxError as e:
        result["parse_error"] = f"SyntaxError: {e!r}"
        ret_dict.update(result)
    except Exception as e:
        result["error_code"] = -4
        result["error_message"] = f"Worker crashed: {e!r}"
        ret_dict.update(result)

def _run_in_subprocess(code: str, inputs: List[List[List[int]]], timeout: int = 8) -> Dict[str, Any]:
    """
    Spawn a subprocess to run user code on given inputs, with a wall-clock timeout.
    """
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    ret_dict = manager.dict()

    p = ctx.Process(target=_worker_run, args=(code, inputs, ret_dict))
    p.start()
    p.join(timeout=timeout)
    if p.is_alive():
        p.kill()
        p.join()
        return {
            "ok": False, "preds": None,
            "error_code": -5,
            "error_message": "GlobalTimeoutOrCrash",
            "parse_error": None, "runtime_error": None,
        }

    # Convert manager.dict() -> plain dict
    return dict(ret_dict) if ret_dict else {
        "ok": False, "preds": None,
        "error_code": -4,
        "error_message": "Empty worker result",
        "parse_error": None, "runtime_error": None,
    }

###############################################################################
# Public Evaluators
###############################################################################

def eval_arc_train_only(query: Dict[str, Any], code: str, timeout: int = 8) -> Dict[str, Any]:
    """
    Evaluate ARC-AGI code on TRAIN pairs only.
    Returns a dict shaped for issues_from_eval(task_name="ARC-AGI").
      {
        "score": float,              # accuracy - 1.0
        "train_results": [           # per-example correctness + preds
            {"index": i, "is_correct": bool, "pred": grid, "expected": grid},
            ...
        ],
        # Optional diagnostics:
        "tests_pred": [ {"index": j, "pred": grid or None}, ... ],   # (not used for scoring; only if you want)
        "error_code": ..., "error_message": ..., "parse_error": ..., "runtime_error": ...,
      }
    """
    train, _ = _extract_train_test(query)
    inputs = [ex["input"] for ex in train]
    expected = [ex["output"] for ex in train]

    # Run user code in a clean subprocess
    worker_res = _run_in_subprocess(code, inputs, timeout=timeout)

    # Handle worker failures
    if not worker_res.get("ok"):
        # Score: severe failure gets a very low score (consistent with LCB's -100.0)
        out = {
            "score": -100.0,
            "train_results": [],
        }
        for k in ("error_code", "error_message", "parse_error", "runtime_error"):
            if worker_res.get(k) is not None:
                out[k] = worker_res.get(k)
        return out

    preds = worker_res["preds"] or []
    train_results: List[Dict[str, Any]] = []
    correct = 0
    for i, (pred, exp) in enumerate(zip(preds, expected)):
        ok = _grids_equal(pred, exp)
        if ok:
            correct += 1
        train_results.append({
            "index": i,
            "is_correct": bool(ok),
            "pred": pred,
            "expected": exp,
        })

    total = len(expected) if expected else 1
    acc = correct / total
    score = acc - 1.0
    return {
        "score": float(score),
        "train_results": train_results,
    }

def eval_arc_full(query: Dict[str, Any], code: str, timeout: int = 8) -> Dict[str, Any]:
    """
    Evaluate on TRAIN (scored) and also produce TEST predictions (display only).
    Shape matches issues_from_eval(task_name="ARC-AGI") expectations.
    """
    train, test = _extract_train_test(query)

    # 1) TRAIN (scored)
    res_train = eval_arc_train_only(query, code, timeout=timeout)

    # If train already reported a fatal error, return immediately
    if "error_code" in res_train or "parse_error" in res_train or "runtime_error" in res_train:
        return res_train

    # 2) TEST (not scored, display only)
    test_inputs = [ex["input"] for ex in test] if test else []
    tests_pred: List[Dict[str, Any]] = []
    if test_inputs:
        worker_res = _run_in_subprocess(code, test_inputs, timeout=timeout)
        if worker_res.get("ok"):
            preds = worker_res["preds"] or []
            for j, p in enumerate(preds):
                tests_pred.append({"index": j, "pred": p})
        else:
            # Test phase crash does not affect train score, but provide a notice
            tests_pred.append({"index": 0, "pred": None})
            res_train["error_code"] = worker_res.get("error_code", -4)
            res_train["error_message"] = worker_res.get("error_message")
            if worker_res.get("parse_error"):
                res_train["parse_error"] = worker_res["parse_error"]
            if worker_res.get("runtime_error"):
                res_train["runtime_error"] = worker_res["runtime_error"]

    if tests_pred:
        res_train["tests_pred"] = tests_pred
    return res_train

###############################################################################
# CLI (optional)
###############################################################################

def _load_json(fp: str) -> Dict[str, Any]:
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Evaluate ARC-AGI code on train (and optionally show test preds).")
    parser.add_argument("--data", type=str, required=True, help="Path to ARC-AGI JSON with keys train/test.")
    parser.add_argument("--code", type=str, required=True, help="Path to a .py file containing transform(grid).")
    parser.add_argument("--timeout", type=int, default=8)
    parser.add_argument("--full", action="store_true", help="If set, also run on test and include tests_pred.")
    args = parser.parse_args()

    data = _load_json(args.data)
    with open(args.code, "r", encoding="utf-8") as f:
        code = f.read()

    if args.full:
        eva = eval_arc_full(data, code, timeout=args.timeout)
    else:
        eva = eval_arc_train_only(data, code, timeout=args.timeout)

    acc = (eva.get("score", -1.0) + 1.0)
    print(f"Accuracy on train: {acc:.2%}")   # Only the accuracy

    print(json.dumps(eva, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    main()
