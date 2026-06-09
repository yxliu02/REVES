#!/usr/bin/env python3
import json
import argparse
from math_verify import verify, parse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", help="Path to JSON or JSONL file")
    args = parser.parse_args()
    # Read data
    with open(args.json_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            data = [json.loads(line) for line in f if line.strip()]

    if isinstance(data, dict):
        data = [data]

    total = 0
    correct = 0
    for rec in data:
        if "raw_txt" not in rec or "gold_answer" not in rec:
            print("ERROR")
        pred = parse(rec["raw_txt"])
        gold = parse(rec["gold_answer"])
        total += 1
        if verify(pred, gold):
            correct += 1

    acc = correct / total if total > 0 else 0.0
    print(f"pass/total = {correct}/{total} = {acc:.4f}")

if __name__ == "__main__":
    main()
