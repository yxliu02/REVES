# evaluator/worker_entry.py
from __future__ import annotations
import argparse, json, base64, sys
from evaluator.LiveCodeBench.testing_util import run_test  # reuse utils.run_test

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample_b64", required=True, help="base64-encoded JSON with {'input_output': ...}")
    ap.add_argument("--code_b64", required=True, help="base64-encoded utf-8 python code")
    ap.add_argument("--timeout", type=int, default=6)
    args = ap.parse_args()

    sample = json.loads(base64.b64decode(args.sample_b64).decode("utf-8"))
    code = base64.b64decode(args.code_b64).decode("utf-8")

    # Still using utils.run_test (uses signal/sandbox internally)
    results, metadata = run_test(sample, test=code, debug=False, timeout=args.timeout)

    # Output as a single-line JSON
    print(json.dumps({"results": results, "metadata": metadata}, ensure_ascii=False))

if __name__ == "__main__":
    main()
