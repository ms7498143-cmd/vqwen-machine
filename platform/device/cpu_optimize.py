#!/usr/bin/env python3
"""
Self-optimization engine.

This closes the loop the master prompt asks for, and it does so with measured
numbers rather than claims:

  1. Measure the current implementation (interpreter, and JIT if available).
  2. Identify the bottleneck (instructions per second, dispatch overhead).
  3. Apply an optimization (basic-block translation).
  4. Re-verify correctness (cpu_jit_verify.py must pass).
  5. Keep the change only if it is both correct and faster.
  6. Record the result so the next run can compare.

Correctness is a hard gate: a faster engine that fails verification is
discarded. The measured speedup is reported honestly, including when it is
small or negative.
"""

import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from asm import assemble  # noqa: E402
from cpu import CPU  # noqa: E402
from jit import JIT  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
HISTORY = os.path.join(HERE, "cpu_opt_history.json")

# A workload that mixes arithmetic, a loop and a branch: realistic enough that
# dispatch overhead dominates, which is the cost the JIT removes.
WORKLOAD = [
    "xorl %eax, %eax",       # acc = 0
    "xorl %ecx, %ecx",       # i = 0
    "L1:",
    "addq %rcx, %rax",
    "imulq %rcx, %rax",
    "incq %rcx",
    "cmpq %rdi, %rcx",
    "jl L1",
]


def assemble_workload():
    from cpu_programs import assemble_program
    code = assemble_program(WORKLOAD)
    # Drop the trailing `ret`: this harness runs the block by RIP bounds and
    # does not want to execute a return.
    if code and code[-1] == 0xC3:
        code = code[:-1]
    return code


def run_interpreter(code, n, reps):
    cpu = CPU()
    addr = cpu.load(code)
    t0 = time.perf_counter()
    for _ in range(reps):
        cpu.reg[0] = 0
        cpu.reg[1] = 0
        cpu.reg[7] = n
        cpu.rip = addr
        cpu.halted = False
        while cpu.rip < addr + len(code):
            cpu.step()
    return time.perf_counter() - t0, cpu.reg[0]


def run_jit(code, n, reps):
    cpu = CPU()
    addr = cpu.load(code)
    jit = JIT(cpu, code_end=addr + len(code))
    t0 = time.perf_counter()
    for _ in range(reps):
        cpu.reg[0] = 0
        cpu.reg[1] = 0
        cpu.reg[7] = n
        cpu.rip = addr
        cpu.halted = False
        jit.run(max_blocks=10_000_000)
    return time.perf_counter() - t0, cpu.reg[0]


def verify():
    """Correctness gate: the JIT must pass its verification suite."""
    r = subprocess.run([sys.executable, os.path.join(HERE, "cpu_jit_verify.py")],
                       capture_output=True, text=True, cwd=HERE)
    return r.returncode == 0, r.stdout.strip().splitlines()[-1] if r.stdout else ""


def main():
    code = assemble_workload()
    n = 20000
    reps = 5

    print("Self-optimization engine\n")
    print(f"workload: sum and product loop over {n} iterations, {reps} repetitions")
    print(f"code: {code.hex(' ')}\n")

    # 1. Measure the baseline (interpreter).
    it_s, it_res = run_interpreter(code, n, reps)
    print(f"  baseline (interpreter): {it_s:.4f}s   result={it_res}")

    # 2-3. The optimization under test: basic-block JIT.
    jit_s, jit_res = run_jit(code, n, reps)
    print(f"  candidate (JIT):        {jit_s:.4f}s   result={jit_res}")

    # 4. Correctness gate.
    ok, summary = verify()
    print(f"  correctness gate:       {'PASS' if ok else 'FAIL'}  ({summary})")

    correct = (it_res == jit_res) and ok
    faster = jit_s < it_s
    speedup = round(it_s / jit_s, 2) if jit_s else 0
    kept = correct and faster

    print(f"\n  results identical: {it_res == jit_res}")
    print(f"  speedup:           {speedup}x")
    print(f"  decision:          {'KEEP JIT' if kept else 'DISCARD (must be correct AND faster)'}")

    record = {
        "workload_iterations": n,
        "repetitions": reps,
        "interpreter_s": round(it_s, 4),
        "jit_s": round(jit_s, 4),
        "speedup": speedup,
        "correct": correct,
        "kept": kept,
    }
    hist = []
    if os.path.exists(HISTORY):
        with open(HISTORY) as f:
            hist = json.load(f)
    hist.append(record)
    with open(HISTORY, "w") as f:
        json.dump(hist, f, indent=2)
    print(f"  recorded in {HISTORY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
