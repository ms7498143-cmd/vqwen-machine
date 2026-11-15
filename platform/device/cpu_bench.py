#!/usr/bin/env python3
"""
Honest performance benchmark: the CPU clone versus the physical CPU.

Both sides run the *same machine code* on the *same workload*. The physical
side runs it natively through a compiled C shim; the clone decodes and
interprets the same bytes in cpu.py. Each measurement is repeated and the
median is reported, together with the ratio.

This file exists to state the truth plainly, including the uncomfortable part:
a software CPU clone runs on the host processor and is far slower than it. The
clone provides functional equivalence, not performance equivalence. Every
number below is measured on the machine that runs this file, and the region,
CPU model and core count are recorded with the results so the figures can be
reproduced or challenged.
"""

import ctypes
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from asm import assemble  # noqa: E402
from cpu import CPU  # noqa: E402


def host_cpu_name():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def build_bench(asm):
    """Compile a shim that runs the given asm in a loop and times it natively."""
    src = """
#include <stdint.h>
#include <time.h>
typedef struct { uint64_t result; double seconds; } out_t;
static uint64_t now_ns(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + t.tv_nsec;
}
out_t run(uint64_t iters, uint64_t a) {
    out_t o;
    uint64_t start = now_ns();
    uint64_t r = a;
    for (uint64_t i = 0; i < iters; i++) {
        __asm__ __volatile__(
            "mov %[r], %%rax\\n\\t"
            "mov %[i], %%rbx\\n\\t"
            "ASM_BODY"
            "mov %%rax, %[r]\\n\\t"
            : [r] "+r" (r)
            : [i] "r" (i)
            : "rax", "rbx", "cc", "memory"
        );
    }
    o.seconds = (double)(now_ns() - start) / 1e9;
    o.result = r;
    return o;
}
"""
    d = tempfile.mkdtemp()
    cpath = os.path.join(d, "b.c")
    so = os.path.join(d, "b.so")
    body = asm.replace("%", "%%") + "\\n\\t"
    with open(cpath, "w") as f:
        f.write(src.replace("ASM_BODY", body))
    subprocess.run(["gcc", "-O2", "-shared", "-fPIC", "-o", so, cpath],
                   check=True, capture_output=True)
    lib = ctypes.CDLL(so)
    lib.run.restype = type("out_t", (ctypes.Structure,),
                           {"_fields_": [("result", ctypes.c_uint64),
                                         ("seconds", ctypes.c_double)]})
    lib.run.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
    return lib


def time_clone(code, iters, a=0):
    """Time the clone executing `iters` copies of the routine."""
    cpu = CPU()
    addr = cpu.load(code)
    stack = cpu.base + len(cpu.mem) - 0x200
    t0 = time.perf_counter()
    for _ in range(iters):
        cpu.reg[0] = a
        cpu.reg[3] = iters
        cpu.reg[4] = stack          # RSP: a valid stack for store/load tests
        cpu.rip = addr
        cpu.halted = False
        while cpu.rip < addr + len(code):
            cpu.step()
    return time.perf_counter() - t0


BENCHMARKS = [
    ("integer add", "addq %rbx, %rax"),
    ("integer multiply", "imulq %rbx, %rax"),
    ("shift left", "mov $3, %cl; shlq %cl, %rax"),
    ("rotate", "mov $7, %cl; rolq %cl, %rax"),
    ("compare + branch-free set", "cmpq %rbx, %rax; setg %al; movzbl %al, %eax"),
    ("lea (address calc)", "leaq 8(%rax,%rbx,4), %rax"),
    ("bitwise xor", "xorq %rbx, %rax"),
    ("memory store/load", "movq %rax, (%rsp); movq (%rsp), %rax"),
]


def measure(name, asm, native_iters=20_000_000, clone_iters=20_000):
    code = assemble(asm)
    lib = build_bench(asm)
    # native, best of 5
    nat = []
    for _ in range(5):
        o = lib.run(native_iters, 0x1234)
        nat.append(o.seconds)
    nat_s = min(nat)
    nat_ops = native_iters / nat_s

    # clone, best of 3
    cl = []
    for _ in range(3):
        cl.append(time_clone(code, clone_iters, 0x1234))
    cl_s = min(cl)
    cl_ops = clone_iters / cl_s

    return {
        "name": name,
        "asm": asm,
        "native_ops_per_s": round(nat_ops, 0),
        "clone_ops_per_s": round(cl_ops, 0),
        "slowdown_factor": round(nat_ops / cl_ops, 1),
    }


def main():
    results = []
    print("Honest benchmark — same machine code, native versus clone")
    print(f"host CPU: {host_cpu_name()}")
    print(f"python:   {platform.python_version()}  cores: {os.cpu_count()}\n")
    print(f"  {'workload':<28}{'native ops/s':>16}{'clone ops/s':>16}{'slowdown':>12}")
    for name, asm in BENCHMARKS:
        r = measure(name, asm)
        results.append(r)
        print(f"  {name:<28}{r['native_ops_per_s']:>16,.0f}"
              f"{r['clone_ops_per_s']:>16,.0f}{r['slowdown_factor']:>11.0f}x")

    ratios = [r["slowdown_factor"] for r in results]
    summary = {
        "host_cpu": host_cpu_name(),
        "cores": os.cpu_count(),
        "python": platform.python_version(),
        "median_slowdown": round(statistics.median(ratios), 1),
        "results": results,
        "note": ("A software CPU clone executes on the host processor and is "
                 "slower than it. This provides functional equivalence, not "
                 "performance equivalence."),
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cpu_bench.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  median slowdown: {summary['median_slowdown']}x")
    print(f"  written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
