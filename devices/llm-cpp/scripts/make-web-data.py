#!/usr/bin/env python3
"""Generate web/device.json from real device metrics.

Run after device/orchestrate.py so the website only ever shows measured values.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name):
    p = os.path.join(ROOT, "metrics", name)
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)


def main():
    storage = load("storage.json")
    network = load("network.json")
    compute = load("compute.json")
    out = {
        "device": "llm-cpp",
        "generated_from": "metrics/*.json (real measurements)",
        "model": {
            "id": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            "file": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
            "params": "0.49B",
            "quant": "Q4_K_M",
            "size_mb": 469,
        },
        "runtime": {
            "engine": "llama.cpp",
            "commit": "dd099a6",
            "build": "static (no glibc dependency)",
            "version": "1.0.0-dev",
        },
        "capabilities": {
            "storage": {
                "write_mb_s": storage.get("write_mb_s"),
                "read_mb_s": storage.get("read_mb_s"),
                "fsync_mb_s": storage.get("fsync_write_mb_s"),
                "volume_mb": storage.get("volume_mb"),
            },
            "network": {
                "throughput_mb_s": network.get("throughput_mb_s"),
                "throughput_mbit_s": network.get("throughput_mbit_s"),
                "integrity": network.get("integrity_sha256_match"),
                "endpoint": network.get("device_endpoint"),
            },
            "compute": {
                "decode_tok_s": compute.get("decode_tok_s_mean"),
                "prefill_tok_s": compute.get("prefill_tok_s_mean"),
                "cpu_set": compute.get("allocation", {}).get("cpu_set"),
                "mem_cap_gb": compute.get("allocation", {}).get("memory_cap_gb"),
            },
        },
        "measured_at": compute.get("runs") and __import__("time").strftime(
            "%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()
        ) or None,
    }
    dest = os.path.join(ROOT, "web", "device.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", dest)


if __name__ == "__main__":
    main()
