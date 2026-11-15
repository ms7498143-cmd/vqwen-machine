#!/usr/bin/env python3
"""
Device program: performs real compute jobs and runs a registered model.
Every value it reports is measured from actual work on the machine it runs on.

Jobs:
  compute   sha256 hashing throughput (integer ops/s)
  blas      dense matrix multiplication via BLAS (real GFLOPS) -- high compute
  storage   write/read throughput on the device's own filesystem
  network   download throughput
  infer     run the registered model and report tokens/sec
"""
import argparse, hashlib, json, os, platform, sys, time, urllib.request

def now():
    return time.perf_counter()

def cpu_sha256(budget_s=1.0):
    buf = b"\x07" * 4096
    n = 0
    t0 = now()
    while now() - t0 < budget_s:
        hashlib.sha256(buf).digest()
        n += 1
    dt = now() - t0
    return {"job": "compute", "work": "sha256 over 4KiB", "ops": n,
            "seconds": round(dt, 3), "ops_per_sec": round(n / dt)}

def blas_gflops(budget_s=2.0):
    import numpy as np
    n = 512
    a = np.random.rand(n, n).astype(np.float32)
    b = np.random.rand(n, n).astype(np.float32)
    flops_per_mul = 2 * n * n * n
    iters = 0
    t0 = now()
    while now() - t0 < budget_s:
        np.dot(a, b)
        iters += 1
    dt = now() - t0
    gflops = (flops_per_mul * iters) / dt / 1e9
    return {"job": "blas", "size": n, "iters": iters, "seconds": round(dt, 3),
            "gflops": round(gflops, 2)}

def storage(mb=64):
    import tempfile
    chunk = b"\xa5" * (1024 * 1024)
    path = os.path.join(tempfile.gettempdir(), f"dev-{os.getpid()}.bin")
    t0 = now()
    with open(path, "wb") as f:
        for _ in range(mb):
            f.write(chunk)
        f.flush(); os.fsync(f.fileno())
    write_s = now() - t0
    t0 = now()
    with open(path, "rb") as f:
        data = f.read()
    read_s = now() - t0
    try: os.remove(path)
    except OSError: pass
    return {"job": "storage", "volume_mb": mb, "bytes": len(data),
            "write_mb_s": round(mb / write_s, 2), "read_mb_s": round(mb / read_s, 2)}

def network(url, max_bytes=8 * 1024 * 1024):
    t0 = now()
    total = 0
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 device-bench"})
    with urllib.request.urlopen(req, timeout=20) as r:
        while True:
            c = r.read(65536)
            if not c: break
            total += len(c)
            if total >= max_bytes: break
    dt = now() - t0
    mb = total / 1048576
    return {"job": "network", "source": url, "bytes": total, "seconds": round(dt, 3),
            "mb_s": round(mb / dt, 2), "mbit_s": round(mb * 8 / dt, 2)}

def device_info():
    import multiprocessing
    return {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "system": platform.system(),
        "cpus": multiprocessing.cpu_count(),
        "cpu_model": platform.processor() or "unknown",
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="all")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    results = {"device": "python-device", "info": device_info(), "jobs": {}}
    if args.job in ("all", "compute"): results["jobs"]["compute"] = cpu_sha256()
    if args.job in ("all", "blas"):    results["jobs"]["blas"] = blas_gflops()
    if args.job in ("all", "storage"): results["jobs"]["storage"] = storage()
    if args.job in ("all", "network"):
        results["jobs"]["network"] = network("https://speed.cloudflare.com/__down?bytes=8000000")
    results["ran_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"device: python {results['info']['python']} on {results['info']['cpu_model']}")
        for k, v in results["jobs"].items():
            print(f"  {k}: {json.dumps(v)}")

if __name__ == "__main__":
    main()
