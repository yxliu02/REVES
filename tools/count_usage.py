#!/usr/bin/env python3
import json
import sys

if len(sys.argv) != 2:
    print("Usage: sum_tokens.py <json_file>")
    sys.exit(1)

file_path = sys.argv[1]

# Determine whether the file is JSON or JSONL
try:
    with open(file_path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            data = json.load(f)
        else:
            data = [json.loads(line) for line in f if line.strip()]
except Exception as e:
    print(f"Error reading {file_path}: {e}")
    sys.exit(1)

total_input_tokens = 0
total_output_tokens = 0
total_attempts = 0

for item in data:
    usage = item.get("usage", {})
    total_input_tokens += usage.get("input_token", 0)
    total_output_tokens += usage.get("output_token", 0)
    total_attempts += item.get("attempt_count", 0)

print(f"{file_path}:")
print(f"input_token: {total_input_tokens}")
print(f"output_token: {total_output_tokens}")
print(f"attempt_count: {total_attempts}")
print("-------")
