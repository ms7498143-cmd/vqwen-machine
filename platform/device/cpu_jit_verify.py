#!/usr/bin/env python3
"""
Verify the JIT against the interpreter and the physical CPU.

The JIT is only useful if it is correct. Two checks are run:

  1. Interpreter versus JIT: the same routine, executed by both engines, must
     leave every register and every flag identical.
  2. JIT versus the physical CPU: the same machine code run natively must give
     the same result as the JIT.

If either check fails, the JIT must not be used. Correctness is never traded
for speed.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from asm import assemble  # noqa: E402
from cpu import CPU  # noqa: E402
from cpu_diff import build_harness, flags_from_word, HOST_FLAGS  # noqa: E402
from jit import JIT  # noqa: E402


def run_interp(code, a, b):
    c = CPU()
    addr = c.load(code)
    c.reg[0], c.reg[3] = a, b
    c.rip = addr
    while c.rip < addr + len(code):
        c.step()
    return c.reg[0] & ((1 << 64) - 1), c.flags_snapshot()


def run_jit(code, a, b):
    c = CPU()
    addr = c.load(code)
    c.reg[0], c.reg[3] = a, b
    c.rip = addr
    j = JIT(c, code_end=addr + len(code))
    j.run(max_blocks=2_000_000)
    return c.reg[0] & ((1 << 64) - 1), c.flags_snapshot()


CASES = [
    ("add rax,rbx", "addq %rbx, %rax", HOST_FLAGS),
    ("sub rax,rbx", "subq %rbx, %rax", HOST_FLAGS),
    ("and rax,rbx", "andq %rbx, %rax", HOST_FLAGS),
    ("xor rax,rbx", "xorq %rbx, %rax", HOST_FLAGS),
    ("or  rax,rbx", "orq %rbx, %rax", HOST_FLAGS),
    ("inc rax", "incq %rax", ["PF", "AF", "ZF", "SF", "OF"]),
    ("dec rax", "decq %rax", ["PF", "AF", "ZF", "SF", "OF"]),
    ("mov rax,rbx", "movq %rbx, %rax", []),
    ("mov rax,imm", "movq $0x1234, %rax", []),
    ("lea", "leaq 8(%rax,%rbx,4), %rax", []),
    ("loop sum", "xorl %eax,%eax; L1: addq %rbx,%rax; decq %rbx; jnz L1", HOST_FLAGS),
]

# Operands for the looping case must be small: a counter of 0 would wrap and
# run for 2^64 iterations.
LOOP_OPERANDS = [(0, 1), (0, 3), (0, 10), (0, 100), (5, 7), (0, 255)]

OPERANDS = [(0, 0), (1, 1), (0xFFFFFFFFFFFFFFFF, 1), (0x7FFFFFFFFFFFFFFF, 1),
            (0x8000000000000000, 1), (0x123456789, 0x9ABCDEF), (1000, 7), (0, 5)]


def main():
    ok = True
    print("JIT verification\n")
    for name, asm, defined in CASES:
        code = assemble(asm)
        lib = build_harness(asm)
        bad = 0
        ops = LOOP_OPERANDS if name.startswith("loop") else OPERANDS
        for (a, b) in ops:
            ri, fi = run_interp(code, a, b)
            rj, fj = run_jit(code, a, b)
            hw = lib.run_case(a, b, 0)
            hr = hw.result & ((1 << 64) - 1)
            hf = flags_from_word(hw.flags)

            interp_jit_ok = (ri == rj) and all(fi[f] == fj[f] for f in HOST_FLAGS)
            jit_host_ok = (rj == hr) and all(hf[f] == fj[f] for f in defined)
            if not (interp_jit_ok and jit_host_ok):
                bad += 1
                if bad == 1:
                    print(f"  mismatch a={a:#x} b={b:#x}: interp={ri:#x} jit={rj:#x} host={hr:#x}")
        status = "PASS" if bad == 0 else "FAIL"
        ok = ok and bad == 0
        print(f"  {status}  {name:<16} interpreter == JIT == physical CPU "
              f"({len(ops)} operand pairs)")
    print("\nJIT RESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
