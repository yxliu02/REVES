#!/usr/bin/env python3
import pickle
import json
import sys
from pathlib import Path

try:
    import pandas as pd  # Optional, for DataFrame pkl compatibility
except Exception:
    pd = None

def try_json_loads(x):
    """Parse if it's a JSON string, otherwise return as-is."""
    if isinstance(x, str):
        s = x.strip()
        # Rough check for JSON-like strings
        if (s.startswith('{') and s.endswith('}')) or (s.startswith('[') and s.endswith(']')) or s.lower() in ('true','false'):
            try:
                return json.loads(s)
            except Exception:
                return x
    return x

def is_bool_like(v):
    if isinstance(v, bool):
        return True
    if isinstance(v, str) and v.strip().lower() in ('true','false'):
        return True
    return False

def list_all_bool_like(v):
    return isinstance(v, list) and len(v) > 0 and all(is_bool_like(x) for x in v)

def classify(gt_raw):
    """
    Returns ('verification'|'reflection', normalized_gt)
    - verification: ground_truth is bool / 'True'/'False' / [bool...]
    - reflection: everything else (including dict/free text/IO pairs)
    """
    gt = try_json_loads(gt_raw)

    # Single bool or bool-like string
    if is_bool_like(gt):
        return 'verification', (gt if isinstance(gt, bool) else gt.strip().lower() == 'true')

    # List of all bool/bool-like strings
    if list_all_bool_like(gt):
        norm = [(x if isinstance(x, bool) else x.strip().lower() == 'true') for x in gt]
        return 'verification', norm

    # Everything else is treated as reflection (including {"inputs":[...],"outputs":[...]})
    return 'reflection', gt

def extract_ground_truth(entry):
    """Generic path: entry['reward_model']['ground_truth']"""
    rm = entry.get('reward_model', {}) if isinstance(entry, dict) else {}
    return rm.get('ground_truth', None)

def to_records(obj):
    """Normalize the top-level object into list[dict]."""
    if isinstance(obj, list):
        # Ensure elements are dicts
        out = []
        for x in obj:
            if isinstance(x, dict):
                out.append(x)
            elif pd is not None and isinstance(x, pd.Series):
                out.append(x.to_dict())
            else:
                raise TypeError(f"Unsupported element type in list: {type(x)}")
        return out

    if isinstance(obj, dict):
        # Some pipelines store samples in a dict; if its values are lists, try to extract the main list
        # Prioritize common keys
        for key in ('data', 'samples', 'records'):
            if key in obj and isinstance(obj[key], list):
                return to_records(obj[key])
        # Otherwise treat the entire dict as a single sample
        return [obj]

    if pd is not None and isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')

    raise TypeError(f"Unsupported top-level type in pkl: {type(obj)}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python split_pkl.py <input_pkl_path>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        sys.exit(1)

    with open(input_path, 'rb') as f:
        data_obj = pickle.load(f)

    records = to_records(data_obj)

    verification, reflection = [], []
    for rec in records:
        gt_raw = extract_ground_truth(rec)
        label, gt_norm = classify(gt_raw)

        # Attach a parsed view for easier downstream inspection (without modifying the original structure)
        out = dict(rec)
        parsed = out.get('_parsed', {})
        parsed['type'] = label
        parsed['ground_truth'] = gt_norm
        out['_parsed'] = parsed

        if label == 'verification':
            verification.append(out)
        else:
            reflection.append(out)

    base = input_path.with_suffix('')
    ver_path = base.with_name(base.name + "_verification.pkl")
    ref_path = base.with_name(base.name + "_reflection.pkl")

    with open(ver_path, 'wb') as f:
        pickle.dump(verification, f)
    with open(ref_path, 'wb') as f:
        pickle.dump(reflection, f)

    # As requested: bash outputs “a single pkl file path”; here we output the verification path.
    # If you want to output reflection instead, replace the line below with print(ref_path)
    print(str(ver_path))

    # Also print a summary to stderr (without affecting the single-line stdout output)
    sys.stderr.write(f"Total: {len(records)} | verification: {len(verification)} | reflection: {len(reflection)}\n")
    sys.stderr.write(f"Saved:\n  {ver_path}\n  {ref_path}\n")

if __name__ == "__main__":
    main()
