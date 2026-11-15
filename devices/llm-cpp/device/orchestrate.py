#!/usr/bin/env python3
"""llm-cpp device orchestrator.

Runs the device jobs in series, then writes the device report. Each job
manufactures one capability of the device and records real measured results.
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOBS = ["job_storage.py", "job_network.py", "job_compute.py"]


def run_job(name):
    path = os.path.join(ROOT, "device", "jobs", name)
    print(f"\n=== SERIES -> {name} ===", flush=True)
    r = subprocess.run([sys.executable, path], cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"job failed: {name} (exit {r.returncode})")


def build_report():
    metrics = {}
    for f in ("storage", "network", "compute"):
        p = os.path.join(ROOT, "metrics", f"{f}.json")
        if os.path.exists(p):
            with open(p) as fh:
                metrics[f] = json.load(fh)
    report = {
        "device": "llm-cpp",
        "runtime": "llama.cpp (static build)",
        "capabilities": {
            "storage": metrics.get("storage"),
            "network": metrics.get("network"),
            "compute": metrics.get("compute"),
        },
    }
    out = os.path.join(ROOT, "metrics", "device-report.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    return report


def main():
    for j in JOBS:
        run_job(j)
    report = build_report()
    c = report["capabilities"]
    print("\n===== DEVICE REPORT (real measurements) =====")
    if c["storage"]:
        print(f"storage : write {c['storage']['write_mb_s']} MB/s | read {c['storage']['read_mb_s']} MB/s | fsync {c['storage']['fsync_write_mb_s']} MB/s")
    if c["network"]:
        print(f"network : {c['network']['throughput_mb_s']} MB/s ({c['network']['throughput_mbit_s']} Mbit/s) integrity={c['network']['integrity_sha256_match']}")
    if c["compute"]:
        print(f"compute : decode {c['compute']['decode_tok_s_mean']} tok/s | prefill {c['compute']['prefill_tok_s_mean']} tok/s")
    print("===========================================")


if __name__ == "__main__":
    main()
