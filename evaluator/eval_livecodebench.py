# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from tqdm import tqdm
import json, base64, subprocess, sys
import numpy as np
# Custom data structures and loading
from utils.code_generation import (
    CodeGenerationProblem,
    load_code_generation_dataset,
)



# Call utils in the same way as the original version using your path
# (If your project uses lcb_runner.evaluation.testing_util, change to that)
from evaluator.LiveCodeBench.testing_util import run_test


def _is_case_pass(e):
    if isinstance(e, (bool, np.bool_)):
        return bool(e)
    if isinstance(e, (int, np.integer)):
        return e > 0
    return False

def _normalize_results(res):
    fixed = []
    for e in res:
        if isinstance(e, np.ndarray):
            try:
                e = e.item()
            except Exception:
                pass
        if isinstance(e, np.bool_):
            e = bool(e)
        fixed.append(e)
    return fixed

# ========== Build sample ==========
def _build_input_output(problem: CodeGenerationProblem, public_only: bool) -> Dict[str, str]:
    tests = problem.public_test_cases if public_only else (problem.public_test_cases + problem.private_test_cases)
    return {
        "input_output": json.dumps(
            {
                "inputs": [t.input for t in tests],
                "outputs": [t.output for t in tests],
                "fn_name": problem.metadata.get("func_name", None),
            }
        )
    }


# ========== Original approach: run run_test in an isolated subprocess per problem ==========
def _temp_run(sample, code, debug, result_list, meta_list, timeout):
    res, metadata = run_test(sample, test=code, debug=debug, timeout=timeout)
    res = _normalize_results(res)
    result_list.append(res)
    meta_list.append(metadata)


def check_correctness(sample, code, timeout: int, debug: bool = False):
    “””
    Execute run_test in a fresh subprocess, with a global timeout as a safety net.
    Returns (results, metadata).
    “””
    ctx = mp.get_context("spawn")  # spawn is more stable; avoids inheriting signals/handles
    manager = ctx.Manager()
    result = manager.list()
    meta = manager.list()

    p = ctx.Process(target=_temp_run, args=(sample, code, debug, result, meta, timeout))
    p.start()
    # Global timeout: timeout seconds per case + a small buffer
    in_outs = json.loads(sample["input_output"])
    per_case = len(in_outs.get("inputs", [])) or 1
    p.join(timeout=(timeout + 1) * per_case + 5)

    if p.is_alive():
        p.kill()
        p.join()

    if not result:
        # Subprocess crashed or was killed; treat all cases as failed
        return ([-1] * per_case), {"error_code": -5, "error_message": "GlobalTimeoutOrCrash"}

    return result[0], meta[0]


# ========== Public-only evaluation functions ==========

def _call_worker_subprocess(sample: dict, code: str, timeout: int = 6) -> tuple[list, dict]:
    """
    Run worker (evaluator.LiveCodeBench.worker_entry) in an isolated Python process, returns (results, metadata).
    - Per-case timeout is handled by SIGALRM inside the worker (in seconds, equals timeout).
    - This implements an outer safety net: scaled by number of cases, kills subprocess on timeout
      and returns a controlled error instead of raising an exception.
    """
    import base64, json, sys, subprocess, os, signal

    # ---- Encode sample / code ----
    try:
        sample_b64 = base64.b64encode(
            json.dumps(sample, ensure_ascii=False).encode("utf-8")
        ).decode("ascii")
    except Exception as e:
        return [-4], {
            "error_code": -4,
            "error_message": f"encode sample failed: {e}",
        }

    try:
        code_b64 = base64.b64encode(code.encode("utf-8")).decode("ascii")
    except Exception as e:
        return [-4], {
            "error_code": -4,
            "error_message": f"encode code failed: {e}",
        }

    # ---- Scale outer safety-net timeout by number of test cases ----
    try:
        in_outs = json.loads(sample.get("input_output", "{}"))
        num_cases = len(in_outs.get("inputs", [])) or 1
    except Exception:
        num_cases = 1

    # Baseline: timeout*20; fallback: scale linearly by case count; minimum 30s buffer
    # NOTE: eval_all_tests includes private cases, so num_cases can be very large => outer_timeout
    #       becomes large and may appear to hang. An env-var hard cap (seconds) is supported here.
    outer_timeout = max(timeout * 20, timeout * (num_cases + 2), 30)
    cap_s = os.getenv("LCB_WORKER_OUTER_TIMEOUT_CAP_S")
    if cap_s:
        try:
            outer_timeout = min(outer_timeout, int(cap_s))
        except Exception:
            pass

    cmd = [
        sys.executable,
        "-m", "evaluator.LiveCodeBench.worker_entry",
        "--sample_b64", sample_b64,
        "--code_b64", code_b64,
        "--timeout", str(timeout),
    ]

    # ---- Run subprocess (controlled timeout + kill + collect output) ----
    # Using Popen+communicate instead of subprocess.run for manual termination and stdout/stderr collection on timeout
    try:
        # start_new_session=True: make the worker a new process group leader so we can kill
        # the entire group on timeout, preventing orphan/zombie processes from forked children.
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as e:
        return [-4], {
            "error_code": -4,
            "error_message": f"spawn worker failed: {e}",
        }

    try:
        stdout, stderr = proc.communicate(timeout=outer_timeout)
    except subprocess.TimeoutExpired:
        # Timeout: force-kill the entire process group and try to collect remaining output.
        # Note: communicate(timeout=...) after kill can still raise a second TimeoutExpired;
        # not catching it would crash the caller (this is where “timed out after 5 seconds” comes from).
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            # Process was killed but pipe/wait is still stuck: abandon output and return a controlled error
            stdout, stderr = b"", b""
        except Exception:
            stdout, stderr = b"", b""

        return [-4], {
            "error_code": -4,
            "error_message": f"worker outer-timeout after {outer_timeout}s",
            "stdout_head": (stdout[:1000].decode("utf-8", "ignore") if stdout else None),
            "stderr_head": (stderr[:1000].decode("utf-8", "ignore") if stderr else None),
            "num_cases": num_cases,
            "timeout_per_case": timeout,
        }
    except Exception as e:
        # Other runtime exceptions
        try:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                proc.kill()
        except Exception:
            pass
        return [-4], {
            "error_code": -4,
            "error_message": f"worker communicate failed: {e}",
        }

    # ---- Check return code ----
    if proc.returncode != 0:
        return [-4], {
            "error_code": -4,
            "error_message": f"worker failed: rc={proc.returncode}",
            "stdout_head": (stdout[:1000].decode("utf-8", "ignore") if stdout else None),
            "stderr_head": (stderr[:1000].decode("utf-8", "ignore") if stderr else None),
        }

    # ---- Parse and validate payload ----
    try:
        payload_text = stdout.decode("utf-8", "ignore") if stdout else ""
        payload = json.loads(payload_text) if payload_text else {}
    except Exception as e:
        return [-4], {
            "error_code": -4,
            "error_message": f"bad worker output: {e}",
            "stdout_head": (payload_text[:1000] if payload_text else None),
            "stderr_head": (stderr[:1000].decode("utf-8", "ignore") if stderr else None),
        }

    results = payload.get("results", None)
    metadata = payload.get("metadata", {})

    # Fallback for result type and content
    if not isinstance(results, list) or len(results) == 0:
        # If worker did not produce valid results, treat as failure
        return [-4], {
            "error_code": -4,
            "error_message": "missing or invalid results from worker",
            "metadata": metadata,
            "stdout_head": (payload_text[:1000] if stdout else None),
            "stderr_head": (stderr[:1000].decode("utf-8", "ignore") if stderr else None),
        }

    if not isinstance(metadata, dict):
        metadata = {"_raw_metadata": metadata}

    return results, metadata


# def _call_worker_subprocess(sample: dict, code: str, timeout: int = 6) -> tuple[list, dict]:
#     """Run utils.run_test in an isolated Python process, returns (results, metadata)."""
#     sample_b64 = base64.b64encode(json.dumps(sample, ensure_ascii=False).encode("utf-8")).decode("ascii")
#     code_b64 = base64.b64encode(code.encode("utf-8")).decode("ascii")

#     cmd = [
#         sys.executable,
#         "-m", "evaluator.LiveCodeBench.worker_entry",  # the worker module
#         "--sample_b64", sample_b64,
#         "--code_b64", code_b64,
#         "--timeout", str(timeout),
#     ]
#     # Timeout is handled by SIGALRM inside the worker; this is just the outer safety net (slightly larger)
#     proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout * 20)
#     if proc.returncode != 0:
#         # Worker crashed / RE / segfault etc., treat all as failed
#         return [-4], {"error_code": -4, "error_message": f"worker failed: rc={proc.returncode}, stderr={proc.stderr.decode('utf-8', 'ignore')}"}

#     try:
#         payload = json.loads(proc.stdout.decode("utf-8", "ignore"))
#         return payload["results"], payload.get("metadata", {})
#     except Exception as e:
#         return [-4], {"error_code": -4, "error_message": f"bad worker output: {e}"}

def eval_public_only(query: CodeGenerationProblem, code: str, timeout: int = 6) -> Dict[str, Any]:
    sample = _build_input_output(query, public_only=True)
    # Run via subprocess
    results, details = _call_worker_subprocess(sample, code, timeout=timeout)
    total = len(results)
    passed = sum(1 for x in results if _is_case_pass(x))
    pass_rate = (passed / total) if total > 0 else 0.0
    score = pass_rate - 1.0  # Definition: score = pass_rate - 1

    errors = []
    if any(isinstance(x, int) and x == -4 for x in results):
        score = -100.0
    # The modified utils puts all failed cases in metadata['failures']; the original gives a single error.
    if isinstance(details, dict):
        fails = details.get("failures")
        if isinstance(fails, list):
            for d in fails:
                errors.append({
                    "inputs": d.get("inputs"),
                    "expected": d.get("expected"),
                    "output": d.get("output"),
                    "error_message": d.get("error_message"),
                })
        else:
            # Backwards-compatible with original single-error format
            if details.get("error_message") or details.get("error"):
                errors.append({
                    "inputs": details.get("inputs"),
                    "expected": details.get("expected"),
                    "output": details.get("output"),
                    "error_message": details.get("error_message") or details.get("error"),
                })

    return {"score": score, "errors": errors}


def eval_all_score_public_errors(query: CodeGenerationProblem, code: str, timeout: int = 6) -> Dict[str, Any]:
    """
    Evaluate on ALL test cases (public + private) for score, but only expose PUBLIC failures in `errors`.

    This is useful when you want a strict score while keeping error details limited to what is publicly available.
    """
    sample = _build_input_output(query, public_only=False)
    results, details = _call_worker_subprocess(sample, code, timeout=timeout)

    total = len(results)
    passed = sum(1 for x in results if _is_case_pass(x))
    pass_rate = (passed / total) if total > 0 else 0.0
    score = pass_rate - 1.0

    try:
        num_public = len(query.public_test_cases)
    except Exception:
        num_public = 0

    errors = []
    if any(isinstance(x, int) and x == -4 for x in results):
        score = -100.0

    if isinstance(details, dict):
        fails = details.get("failures")
        if isinstance(fails, list):
            for d in fails:
                case_idx = d.get("case_index")
                if isinstance(case_idx, int) and num_public > 0 and case_idx >= num_public:
                    continue
                errors.append({
                    "inputs": d.get("inputs"),
                    "expected": d.get("expected"),
                    "output": d.get("output"),
                    "error_message": d.get("error_message"),
                })
        else:
            # Non per-case error (compile/import crash etc.): keep as a generic error
            if details.get("error_message") or details.get("error"):
                errors.append({
                    "inputs": details.get("inputs"),
                    "expected": details.get("expected"),
                    "output": details.get("output"),
                    "error_message": details.get("error_message") or details.get("error"),
                })

    return {"score": score, "errors": errors}


def eval_all_tests(query: CodeGenerationProblem, code: str, timeout: int = 6) -> Dict[str, Any]:
    """
    Same interface as eval_public_only, but evaluates all test cases (public + private).
    Returns {"score": pass_rate - 1.0, "errors": [...]}.
    """
    sample = _build_input_output(query, public_only=False)
    results, details = _call_worker_subprocess(sample, code, timeout=timeout)

    total = len(results)
    passed = sum(1 for x in results if _is_case_pass(x))
    pass_rate = (passed / total) if total > 0 else 0.0
    score = pass_rate - 1.0

    errors = []
    if any(isinstance(x, int) and x == -4 for x in results):
        score = -100.0

    if isinstance(details, dict):
        fails = details.get("failures")
        if isinstance(fails, list):
            for d in fails:
                errors.append({
                    "inputs": d.get("inputs"),
                    "expected": d.get("expected"),
                    "output": d.get("output"),
                    "error_message": d.get("error_message"),
                })
        else:
            if details.get("error_message") or details.get("error"):
                errors.append({
                    "inputs": details.get("inputs"),
                    "expected": details.get("expected"),
                    "output": details.get("output"),
                    "error_message": details.get("error_message") or details.get("error"),
                })

    return {"score": score, "errors": errors}

# ========== Main evaluation: all public+private cases must pass to count as correct ==========
def evaluate_file(
    outputs_file: str,
    release_version: str,
    start_date: str,
    end_date: str,
    workers: int = 8,
    timeout: int = 6,
    debug: bool = False,
) -> Tuple[int, int]:
    dataset = load_code_generation_dataset(
        release_version=release_version,
        start_date=start_date,
        end_date=end_date,
    )
    problem_by_qid: Dict[str, CodeGenerationProblem] = {p.question_id: p for p in dataset}

    # Read submissions
    recs: List[Dict[str, Any]] = []
    with open(outputs_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            # Only collect LiveCodeBench tasks
            if rec.get("task_name") == "LiveCodeBench":
                recs.append(rec)

    # Build tasks
    tasks = []
    for rec in recs:
        qid = rec.get("question_id")
        code = rec.get("code", "")
        if not qid or not code:
            continue
        prob = problem_by_qid.get(qid)
        if prob is None:
            # Not within the time window / release version
            continue
        sample = _build_input_output(prob, public_only=False)
        tasks.append((qid, sample, code))

    total = len(tasks)
    passed = 0

    # Concurrent: use thread pool for scheduling; each task spawns its own subprocess for actual evaluation
    results: List[Tuple[str, bool]] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut2qid = {ex.submit(check_correctness, sample, code, timeout, debug): qid for (qid, sample, code) in tasks}
        for fut in tqdm(as_completed(fut2qid), total=len(fut2qid), desc="Evaluating", unit="prob"):
            qid = fut2qid[fut]
            ok = False
            try:
                res, _ = fut.result()
                res = _normalize_results(res)
                ok = all(_is_case_pass(x) for x in res)
            except Exception:
                ok = False
            results.append((qid, ok))

    passed = sum(1 for _, ok in results if ok)
    return passed, total


def main():
    parser = argparse.ArgumentParser(description="Evaluate LiveCodeBench (public+private: all-pass => correct)")
    parser.add_argument("--outputs_file", type=str, required=True)
    parser.add_argument("--release_version", type=str, required=True)
    parser.add_argument("--start_date", type=str, required=True)
    parser.add_argument("--end_date", type=str, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=6)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    passed, total = evaluate_file(
        outputs_file=args.outputs_file,
        release_version=args.release_version,
        start_date=args.start_date,
        end_date=args.end_date,
        workers=args.workers,
        timeout=args.timeout,
        debug=args.debug,
    )
    rate = (passed / total) if total > 0 else 0.0
    print(f"Pass: {passed}/{total} ({rate:.2%})")


if __name__ == "__main__":
    # Works on Windows/macOS too; explicitly setting spawn is more stable
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    main()
