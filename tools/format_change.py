# format_change.py
# -*- coding: utf-8 -*-
"""
Robust JSON/JSONL -> PKL converter with field hoisting

Features:
- Supports .json / .jsonl / .json.gz / .jsonl.gz
- Optional JSONC preprocessing: remove // and /* */ comments, and trailing commas
- JSONL: skip bad lines in non-strict mode (default) and write them to <output_pkl>.bad.jsonl (configurable)
- Fill or overwrite top-level fields: ability, data_source
- HOIST: promote extra_info.ability / extra_info.data_source to top-level (same level as prompt)
- Optionally keep non-dict items by wrapping them
"""

import os
import re
import gzip
import json
import pickle
import argparse
from typing import List, Dict, Any, Optional, Tuple

# ----------------------- Small utils -----------------------

def _ensure_dir(p: str):
    d = os.path.dirname(p)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _open_text(path: str):
    """Open text (UTF-8) with optional .gz support."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return open(path, "r", encoding="utf-8", newline="")

def _open_badfile(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "wt", encoding="utf-8", newline="")
    return open(path, "w", encoding="utf-8", newline="")

def _strip_bom(s: str) -> str:
    return s.lstrip("\ufeff")

# Remove //... and /* ... */ comments (JSONC-like). Pure Python re; no recursion.
_JSONC_COMMENT_RE = re.compile(
    r"(//[^\r\n]*$)|(/\*.*?\*/)",
    re.MULTILINE | re.DOTALL,
)

def _remove_comments(text: str) -> str:
    return _JSONC_COMMENT_RE.sub("", text)

def _remove_trailing_commas(text: str) -> str:
    """
    Best-effort removal of trailing commas before ] or }.
    Examples:
      [1,2,] -> [1,2]
      {"a":1,} -> {"a":1}
    """
    return re.sub(r",(?=\s*[\]\}])", "", text)

def _preprocess_jsonc(text: str, allow_jsonc: bool) -> str:
    text = _strip_bom(text)
    if allow_jsonc:
        text = _remove_comments(text)
        text = _remove_trailing_commas(text)
    return text

def _detect_is_array(path: str) -> bool:
    """
    Peek the first non-whitespace character to decide if it's a JSON array file.
    True  -> treat as a single JSON array
    False -> treat as JSONL
    """
    with _open_text(path) as f:
        while True:
            ch = f.read(1)
            if not ch:
                break
            if not ch.isspace():
                return ch == "["
    return False  # empty -> treat as JSONL 0 items

def _fix_common_line_issues(line: str) -> str:
    """Trim, drop trailing comma at end of the line (JSONL artifacts), strip BOM."""
    line = _strip_bom(line).strip()
    if line.endswith(","):
        line = line[:-1].rstrip()
    return line

# ----------------------- Loaders -----------------------

def _safe_json_loads(s: str, allow_jsonc: bool) -> Any:
    s = _preprocess_jsonc(s, allow_jsonc=allow_jsonc)
    return json.loads(s)

def _load_json_array(path: str, allow_jsonc: bool) -> List[Any]:
    with _open_text(path) as f:
        text = f.read()
    text = _preprocess_jsonc(text, allow_jsonc=allow_jsonc)
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("JSON file is not an array.")
    return data

def _iter_jsonl(path: str, allow_jsonc: bool, strict: bool, bad_sink) -> Tuple[int, List[Any]]:
    """
    Iterate JSONL; on error:
      - strict=True  -> raise immediately
      - strict=False -> skip and write the line to bad_sink
    Returns (bad_count, items)
    """
    items: List[Any] = []
    bad_count = 0
    with _open_text(path) as f:
        for lineno, raw in enumerate(f, start=1):
            line = _fix_common_line_issues(raw)
            if not line:
                continue
            try:
                obj = _safe_json_loads(line, allow_jsonc=allow_jsonc)
                items.append(obj)
            except Exception as e:
                bad_count += 1
                if strict:
                    raise ValueError(
                        f"JSONL parse failed (line {lineno}): {e}\nRaw line: {raw}"
                    ) from e
                if bad_sink is not None:
                    rec = {
                        "line": lineno,
                        "error": str(e),
                        "raw": raw.rstrip("\r\n"),
                    }
                    bad_sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return bad_count, items

def _load_items(path: str, allow_jsonc: bool, strict: bool, bad_path: Optional[str]) -> Tuple[List[Any], int]:
    if _detect_is_array(path):
        return _load_json_array(path, allow_jsonc=allow_jsonc), 0
    # JSONL path
    bad_count = 0
    items: List[Any] = []
    bad_sink = None
    try:
        if not strict and bad_path:
            _ensure_dir(bad_path)
            bad_sink = _open_badfile(bad_path)
        bad_count, items = _iter_jsonl(path, allow_jsonc=allow_jsonc, strict=strict, bad_sink=bad_sink)
    finally:
        if bad_sink:
            bad_sink.close()
    return items, bad_count

# ----------------------- Transform -----------------------

def _normalize_item(
    obj: Any,
    default_ability: str,
    default_data_source: str,
    force_overwrite: bool,
    keep_non_dict: bool,
) -> Optional[Dict[str, Any]]:
    """
    Hoist extra_info.ability / extra_info.data_source to the top level and fill defaults as needed.
    - If keep_non_dict=False and obj is not a dict, discard it.
    - Only overwrite existing top-level values when force_overwrite=True.
    - After hoisting, remove the corresponding keys from extra_info to avoid duplication.
    """
    if not isinstance(obj, dict):
        if keep_non_dict:
            return {"_value": obj, "ability": default_ability, "data_source": default_data_source}
        return None

    extra = obj.get("extra_info")
    if not isinstance(extra, dict):
        extra = {}

    # ---- Hoist ability ----
    extra_ability = extra.get("ability", None)
    if extra_ability is not None and (force_overwrite or obj.get("ability") is None):
        obj["ability"] = extra_ability
        extra.pop("ability", None)

    # If top-level ability is still missing, fill with default
    if obj.get("ability") is None or (force_overwrite and extra_ability is None):
        # When force_overwrite=True and extra doesn't provide ability, also overwrite with default
        obj["ability"] = default_ability

    # ---- Hoist data_source ----
    extra_ds = extra.get("data_source", None)
    if extra_ds is not None and (force_overwrite or obj.get("data_source") is None):
        obj["data_source"] = str(extra_ds)
        extra.pop("data_source", None)

    # If top-level data_source is still missing, fill with default
    if obj.get("data_source") is None or (force_overwrite and extra_ds is None):
        obj["data_source"] = str(default_data_source)

    # Clean up extra_info (comment out the next two lines to keep empty dicts)
    if extra:
        obj["extra_info"] = extra
    elif "extra_info" in obj:
        obj.pop("extra_info", None)

    return obj

# ----------------------- Main -----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True, help="Path to input .json/.jsonl[.gz]")
    parser.add_argument("--output_pkl", required=True, help="Path to output PKL file")
    parser.add_argument("--data_source", default="train-code-generic", help="Default top-level data_source if missing")
    parser.add_argument("--ability", default="code", help="Default top-level ability if missing")
    parser.add_argument("--force_overwrite", action="store_true",
                        help="If set, overwrite existing top-level ability/data_source (and default if extra_info missing)")
    parser.add_argument("--strict", action="store_true",
                        help="Strict mode: stop on first JSON parse error")
    parser.add_argument("--allow_jsonc", action="store_true",
                        help="Allow JSONC-like input (remove comments + trailing commas)")
    parser.add_argument("--keep_non_dict", action="store_true",
                        help="Keep non-dict items by wrapping them under key '_value'")
    parser.add_argument("--bad_lines_out",
                        help="Where to write bad JSONL lines (default: <output_pkl>.bad.jsonl). Supports .gz")
    args = parser.parse_args()

    bad_out = args.bad_lines_out or (args.output_pkl + ".bad.jsonl")

    # Load
    try:
        items, bad_count = _load_items(
            args.input_json,
            allow_jsonc=args.allow_jsonc,
            strict=args.strict,
            bad_path=None if args.strict else bad_out
        )
    except Exception as e:
        raise SystemExit(f"[ERROR] Read/parse failed: {e}")

    # Normalize
    fixed: List[Dict[str, Any]] = []
    non_dict_skipped = 0
    for obj in items:
        norm = _normalize_item(
            obj,
            default_ability=args.ability,
            default_data_source=args.data_source,
            force_overwrite=args.force_overwrite,
            keep_non_dict=args.keep_non_dict,
        )
        if norm is None:
            non_dict_skipped += 1
            continue
        fixed.append(norm)

    # Dump
    _ensure_dir(args.output_pkl)
    with open(args.output_pkl, "wb") as f:
        pickle.dump(fixed, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Logs
    print(f"[DONE] Converted {len(fixed)} items -> {args.output_pkl}")
    if bad_count and not args.strict:
        print(f"[WARN] Skipped {bad_count} bad JSONL line(s). See: {bad_out}")
    if non_dict_skipped:
        print(f"[INFO] Skipped {non_dict_skipped} non-dict item(s). Use --keep_non_dict to retain them.")

if __name__ == "__main__":
    main()
