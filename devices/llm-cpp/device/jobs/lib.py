"""Shared helpers for llm-cpp device jobs.

Every measurement here comes from a real operation timed with the monotonic
clock. Nothing is estimated or synthesized.
"""
import json
import os
import time


def now():
    return time.monotonic()


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def mb_per_s(nbytes, seconds):
    if seconds <= 0:
        return 0.0
    return round((nbytes / (1024 * 1024)) / seconds, 2)


def record(name, stage, payload):
    """Append a job record to the device metrics ledger."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ledger = os.path.join(root, "metrics", "ledger.json")
    data = read_json(ledger, {"device": "llm-cpp", "jobs": []})
    data["jobs"].append({
        "job": name,
        "stage": stage,
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **payload,
    })
    write_json(ledger, data)
    return data
