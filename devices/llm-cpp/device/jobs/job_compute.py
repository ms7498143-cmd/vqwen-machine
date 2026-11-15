"""Device job: COMPUTE.

Manufactures the llm-cpp device's own compute allocation and runs the model
inside it. The allocation is enforced by the kernel:

  * CPU set     -> taskset pins the process to specific cores
  * Memory cap  -> RLIMIT_AS caps address space (SIGKILL/ENOMEM beyond it)

The model is read from the device's own storage volume, and the server listens
on the device's own network endpoint. Throughput is measured from the server's
own reported timings for real generations.
"""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import now, record, write_json

DEVICE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SERVER = os.path.join(DEVICE_ROOT, "bin", "llama-server")
MODEL = os.environ.get(
    "DEVICE_MODEL",
    os.path.join(DEVICE_ROOT, "models", "qwen2.5-0.5b-instruct-q4_k_m.gguf"),
)
PORT = int(os.environ.get("DEVICE_PORT", "18080"))
CORES = os.environ.get("DEVICE_CORES", "0-3")
MEM_CAP_GB = int(os.environ.get("DEVICE_MEM_GB", "6"))
PROMPT = "Write a short paragraph explaining why the sky is blue."
MAX_TOKENS = 256
RUNS = 3


def start_server():
    cores = CORES.replace("-", ",").split(",")
    cpu_ids = []
    for c in cores:
        cpu_ids.append(int(c))

    def limit():
        # Memory cap: address space the device process may use.
        import resource
        cap = MEM_CAP_GB * 1024 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
        # CPU set: pin the device process to its own cores.
        os.sched_setaffinity(0, set(cpu_ids))

    cmd = [
        SERVER,
        "-m", MODEL,
        "--host", "127.0.0.1", "--port", str(PORT),
        "-t", str(len(cpu_ids)),
        "-c", "4096",
    ]
    log = open(os.path.join(DEVICE_ROOT, "metrics", "compute-server.log"), "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, preexec_fn=limit)
    for _ in range(60):
        time.sleep(1)
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
            return proc
        except Exception:
            if proc.poll() is not None:
                raise RuntimeError("server exited early; see compute-server.log")
    raise RuntimeError("server did not become healthy in 60s")


def generate(seed):
    body = json.dumps({
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS, "temperature": 0.7, "seed": seed,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"},
    )
    t0 = now()
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.load(r)
    return data, now() - t0


def main():
    if not os.path.exists(MODEL):
        raise SystemExit(f"model not on device storage: {MODEL}")
    alloc = {
        "cpu_set": CORES,
        "memory_cap_gb": MEM_CAP_GB,
        "model": os.path.relpath(MODEL, DEVICE_ROOT),
        "endpoint": f"127.0.0.1:{PORT}",
        "enforcement": "taskset (CPU affinity) + RLIMIT_AS (address space cap)",
    }
    proc = start_server()
    runs = []
    try:
        for i in range(RUNS):
            data, wall = generate(seed=2000 + i)
            t = data["timings"]
            runs.append({
                "prompt_tokens": t["prompt_n"],
                "prefill_tok_s": round(t["prompt_per_second"], 2),
                "decode_tokens": t["predicted_n"],
                "decode_tok_s": round(t["predicted_per_second"], 2),
                "wall_s": round(wall, 2),
            })
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(15)
        except subprocess.TimeoutExpired:
            proc.kill()

    decode = [r["decode_tok_s"] for r in runs]
    result = {
        "allocation": alloc,
        "runs": runs,
        "decode_tok_s_mean": round(sum(decode) / len(decode), 2),
        "decode_tok_s_min": min(decode),
        "decode_tok_s_max": max(decode),
        "prefill_tok_s_mean": round(sum(r["prefill_tok_s"] for r in runs) / len(runs), 2),
    }
    write_json(os.path.join(DEVICE_ROOT, "metrics", "compute.json"), result)
    record("compute", "manufactured", result)
    print(f"[compute] allocation cpu={CORES} mem={MEM_CAP_GB}GB")
    print(f"[compute] decode {result['decode_tok_s_mean']} tok/s (min {result['decode_tok_s_min']} / max {result['decode_tok_s_max']})")
    print(f"[compute] prefill {result['prefill_tok_s_mean']} tok/s")


if __name__ == "__main__":
    main()
