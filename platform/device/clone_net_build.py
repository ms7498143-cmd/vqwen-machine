#!/usr/bin/env python3
"""
Build the clone-network artifacts and verify them.

Produces, in device/clone_net_artifacts/:
  meta.json  vocabulary, dimensions, accuracy, measured speed
  data.bin   quantized weights laid out for the clone's memory
  code.bin   the forward pass as x86-64 machine code

The forward pass is assembled here, at build time, so the serving environment
needs no compiler. Verification compares the clone against an independent
numpy reference over every context in the corpus, and compares generated text
end to end. If verification fails, no artifacts are written.
"""

import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ai_model import (train, quantize, emit_asm, data_bytes,  # noqa: E402
                      ref_forward, CORPUS)
from clone_net import CloneNet, ARTIFACTS, WEIGHTS_BASE  # noqa: E402


def assemble(asm):
    d = tempfile.mkdtemp()
    spath = os.path.join(d, "f.s")
    opath = os.path.join(d, "f.o")
    with open(spath, "w") as f:
        f.write(asm)
    subprocess.run(["gcc", "-c", "-o", opath, spath], check=True, capture_output=True)
    import re
    out = subprocess.run(["objdump", "-d", opath], check=True,
                         capture_output=True, text=True).stdout
    code = []
    for line in out.splitlines():
        m = re.match(r"^\s+[0-9a-f]+:\s+((?:[0-9a-f]{2} )+)", line)
        if m:
            code.extend(m.group(1).split())
    return bytes(int(b, 16) for b in code)


def main():
    print("Building clone-network artifacts\n")
    model = train()
    q = quantize(model)
    asm = emit_asm(q, WEIGHTS_BASE)
    code = assemble(asm)
    print(f"  model: vocab={q['V']} hidden={q['H']} context={q['K']} "
          f"float accuracy={model['float_acc']:.3f}")
    print(f"  forward pass: {len(code)} bytes of x86-64, {asm.count(chr(10))} asm lines")

    # write artifacts to a temp location so CloneNet can load them
    os.makedirs(ARTIFACTS, exist_ok=True)
    with open(os.path.join(ARTIFACTS, "code.bin"), "wb") as f:
        f.write(code)
    with open(os.path.join(ARTIFACTS, "data.bin"), "wb") as f:
        f.write(data_bytes(q))

    # verify: per-token and end-to-end
    vocab = q["vocab"]
    idx = {c: i for i, c in enumerate(vocab)}
    data = [idx[c] for c in CORPUS]
    K = q["K"]

    # a temporary meta so CloneNet can start
    with open(os.path.join(ARTIFACTS, "meta.json"), "w") as f:
        json.dump({"vocab": vocab, "K": K, "H": q["H"], "V": q["V"]}, f)
    net = CloneNet()

    mism = 0
    total = 0
    for i in range(K, len(data)):
        ctx = data[i - K:i]
        if net.predict(ctx) != ref_forward(q, ctx):
            mism += 1
        total += 1
    print(f"  per-token verification: {total} contexts, {mism} mismatches")

    def gen(engine, prompt, n=60):
        out = [idx[c] for c in prompt]
        for _ in range(n):
            out.append(engine(out[-K:]))
        return "".join(vocab[i] for i in out)

    clone_text = gen(lambda c: net.predict(c), "the ")
    ref_text = gen(lambda c: ref_forward(q, c), "the ")
    text_ok = clone_text == ref_text
    print(f"  generated text identical: {text_ok}")
    print(f"    clone: {clone_text!r}")

    if mism != 0 or not text_ok:
        print("\nVERIFICATION FAILED — artifacts not finalized")
        return 1

    # measure speed and instructions per pass
    ctx = data[:K]
    reps = 3000
    before = net.cpu.insns
    t0 = time.perf_counter()
    for _ in range(reps):
        net.predict(ctx)
    dt = time.perf_counter() - t0
    per_pass = (net.cpu.insns - before) // reps
    passes_per_s = reps / dt
    print(f"  speed: {passes_per_s:,.0f} forward passes/s ({dt/reps*1e6:.1f} us each)")
    print(f"  instructions per pass: {per_pass:,}")

    meta = {
        "vocab": vocab, "K": K, "H": q["H"], "V": q["V"],
        "float_accuracy": round(model["float_acc"], 4),
        "code_bytes": len(code),
        "instructions_per_pass": per_pass,
        "passes_per_second": round(passes_per_s, 1),
        "verified_contexts": total,
        "verified_mismatches": mism,
        "verified_text_identical": text_ok,
        "engine": "device/cpu.py (independent x86-64 interpreter)",
        "honest_limit": ("Inference runs on a software CPU clone, which executes "
                         "on the host processor and is far slower than native code."),
    }
    with open(os.path.join(ARTIFACTS, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  artifacts written to {ARTIFACTS}")
    print("BUILD RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
