# llm-cpp device

A fabricated inference device that runs a real LLM on its own storage, network
and compute allocation. Every capability is produced by an executed job and
reported with a real measurement.

## What it is

| | |
|---|---|
| **Job** | Serve an open-weights LLM over an HTTP API and report measured throughput. |
| **Runtime** | [llama.cpp](https://github.com/ggml-org/llama.cpp) — static build, commit `dd099a6`. |
| **Model** | `Qwen/Qwen2.5-0.5B-Instruct-GGUF` (Q4_K_M), pulled from HuggingFace by id. |
| **Storage** | Own device volume (`storage/volume.img`). |
| **Network** | Own TCP endpoint (`127.0.0.1:18090`). |
| **Compute** | Own CPU set (`0-3`) and memory cap (`6 GB`, `RLIMIT_AS`). |

## Measured results (real runs)

| Capability | Metric | Value |
|---|---|---|
| compute | decode throughput | 50.39 tok/s (mean of real generations) |
| compute | prefill throughput | 84.37 tok/s |
| storage | sequential write | 201.02 MB/s |
| storage | sequential read | 6270.05 MB/s |
| storage | fsync write | 181.10 MB/s |
| network | TCP throughput | 575.66 MB/s (4605.28 Mbit/s) |
| network | integrity | SHA-256 match verified |

Numbers vary between runs with host load; regenerate with `python3 device/orchestrate.py`.

## What is real and what is emulated

**Real (executed and measured):** model inference, storage I/O, TCP transfer,
and kernel-enforced CPU affinity and memory limit. The memory cap is proven: a
process under a 256 MB cap raises `MemoryError` at ~242 MB while an uncapped
process allocates past it.

**Emulated / reimplemented:** this device is a software reimplementation. It owns
a real, isolated allocation on a host machine — it is **not** new physical
silicon and does **not** add physical capacity. Host CPU, RAM and disk are
partitioned for it, not created. It reproduces the *function* of an inference
device, not the physical device.

## Architecture

```
build (real toolchain)            device jobs (series)             presentation
─────────────────────             ────────────────────             ────────────
scripts/build-llama.sh            device/orchestrate.py            web/index.html
  └ static llama-server    ──►      ├ job_storage.py  (own volume) ──► web/device.json
  └ static llama-cli                ├ job_network.py  (own endpoint)   (real metrics)
                                    └ job_compute.py  (own CPU/mem)
                                          └ runs Qwen model inside allocation
```

## Run it

```bash
# 1. Build the runtime (static llama.cpp)
./scripts/build-llama.sh

# 2. Pull a model by HuggingFace id and serve it
./scripts/run-device.sh Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-q4_k_m.gguf

# 3. Manufacture the device capabilities and measure them (series)
python3 device/orchestrate.py

# 4. Refresh the website data from the real measurements
python3 scripts/make-web-data.py
```

## Layout

```
bin/            static llama-server + llama-cli
models/         pulled GGUF weights (git-ignored)
device/jobs/    the capability jobs (storage, network, compute)
device/orchestrate.py   runs the jobs in series
storage/        device volume (git-ignored)
metrics/        real measurement outputs (storage.json, network.json, compute.json, device-report.json)
scripts/        build + run + web-data generators
web/            professional dashboard (static, deploys to Vercel)
```
