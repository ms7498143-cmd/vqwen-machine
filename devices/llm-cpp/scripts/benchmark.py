#!/usr/bin/env python3
"""Measure real inference performance of the running llm-cpp device.

Calls the llama-server OpenAI-compatible endpoint and records the timings the
server itself reports. No numbers are estimated: every value comes from the
server's own `timings` block for a real generation.
"""
import json
import statistics
import sys
import time
import urllib.request

ENDPOINT = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18080"
RUNS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
PROMPT = "Explain in one paragraph why the sky appears blue during the day."
MAX_TOKENS = 256


def call(seed):
    body = json.dumps({
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
        "seed": seed,
    }).encode()
    req = urllib.request.Request(
        f"{ENDPOINT}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.load(r)
    wall_ms = (time.time() - t0) * 1000
    return data, wall_ms


def main():
    results = []
    for i in range(RUNS):
        data, wall_ms = call(seed=1000 + i)
        t = data["timings"]
        results.append({
            "run": i + 1,
            "prompt_tokens": t["prompt_n"],
            "prompt_per_second": round(t["prompt_per_second"], 2),
            "decode_tokens": t["predicted_n"],
            "decode_per_second": round(t["predicted_per_second"], 2),
            "decode_ms": round(t["predicted_ms"], 1),
            "wall_ms": round(wall_ms, 1),
        })

    summary = {
        "endpoint": ENDPOINT,
        "model": data.get("model"),
        "runs": results,
        "summary": {
            "prefill_tok_s_mean": round(statistics.mean(r["prompt_per_second"] for r in results), 2),
            "decode_tok_s_mean": round(statistics.mean(r["decode_per_second"] for r in results), 2),
            "decode_tok_s_min": round(min(r["decode_per_second"] for r in results), 2),
            "decode_tok_s_max": round(max(r["decode_per_second"] for r in results), 2),
        },
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print(json.dumps(summary, indent=2))
    with open("metrics.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
