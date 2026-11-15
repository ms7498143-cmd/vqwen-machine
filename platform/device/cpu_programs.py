#!/usr/bin/env python3
"""
Real-program execution tests for the CPU clone.

These are not single instructions. Each test is a small program with loops,
memory traffic, function calls and a stack, and each is checked against an
independent reference computed in Python. They show that the clone executes
whole routines correctly, not just isolated opcodes.

The programs are built from bytes assembled by the host assembler where
possible, so the code under test is genuine compiler output.
"""

import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from asm import assemble  # noqa: E402
from cpu import CPU  # noqa: E402


def assemble_program(lines):
    """Assemble a full function body (list of instruction strings) to bytes."""
    body = "\n".join("\t" + l for l in lines)
    src = ".text\n.globl f\nf:\n" + body + "\n\tret\n"
    d = tempfile.mkdtemp()
    spath = os.path.join(d, "f.s")
    opath = os.path.join(d, "f.o")
    with open(spath, "w") as f:
        f.write(src)
    subprocess.run(["gcc", "-c", "-o", opath, spath], check=True, capture_output=True)
    import re
    out = subprocess.run(["objdump", "-d", opath], check=True,
                         capture_output=True, text=True).stdout
    code = []
    for line in out.splitlines():
        m = re.match(r"^\s+[0-9a-f]+:\s+((?:[0-9a-f]{2} )+)", line)
        if m:
            code.extend(m.group(1).split())
    # Keep the trailing `ret`: for whole programs it is the function return.
    return bytes(int(b, 16) for b in code)


def run_program(code, args, mem_bytes=None):
    """Load a program into a fresh CPU, call it with args, return RAX."""
    cpu = CPU()
    if mem_bytes is not None:
        base, data = mem_bytes
        cpu.mem[base - cpu.base: base - cpu.base + len(data)] = data
    addr = cpu.load(code)
    return cpu.call(addr, *args), cpu


def check(name, got, want, extra=""):
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<40} clone={got}{'' if ok else f' expected={want}'} {extra}")
    return ok


def main():
    all_ok = True
    print("Real-program execution tests (clone runs the whole routine)\n")

    # 1. Iterative sum 1..n with a real loop.
    prog = assemble_program([
        "xorl %eax, %eax",
        "movl $1, %ecx",
        "L1:",
        "addq %rcx, %rax",
        "incq %rcx",
        "cmpq %rdi, %rcx",
        "jle L1",
    ])
    for n in (0, 1, 10, 1000):
        # The loop is do-while: the body runs once before the first test, so
        # for n = 0 the result is 1.
        want = sum(range(1, n + 1)) if n >= 1 else 1
        got, _ = run_program(prog, [n])
        all_ok &= check(f"loop sum 1..{n}", got, want)

    # 2. Sum an array in memory (loads and stores through pointers).
    prog = assemble_program([
        "xorl %eax, %eax",
        "xorl %ecx, %ecx",
        "L1:",
        "cmpq %rdi, %rcx",
        "jge L2",
        "addq (%rsi,%rcx,8), %rax",
        "incq %rcx",
        "jmp L1",
        "L2:",
    ])
    vals = [3, 1, 4, 1, 5, 9, 2, 6]
    import struct
    data = b"".join(struct.pack("<q", v) for v in vals)
    mem = (0x500000, data)
    got, _ = run_program(prog, [len(vals), 0x500000], mem_bytes=mem)
    all_ok &= check("sum array of %d from memory" % len(vals), got, sum(vals))

    # 3. Write an array to memory, then read it back and checksum.
    prog = assemble_program([
        "xorl %eax, %eax",       # running checksum
        "xorl %ecx, %ecx",
        "L1:",
        "cmpq %rdi, %rcx",
        "jge L2",
        "movq %rcx, %rdx",
        "imulq %rcx, %rdx",      # rdx = i*i
        "movq %rdx, (%rsi,%rcx,8)",
        "addq %rdx, %rax",
        "incq %rcx",
        "jmp L1",
        "L2:",
    ])
    n = 50
    got, cpu = run_program(prog, [n, 0x600000])
    want = sum(i * i for i in range(n))
    all_ok &= check("store i*i to memory + checksum (n=%d)" % n, got, want)

    # Verify the memory actually holds the squares.
    stored = struct.unpack("<50q", bytes(cpu.mem[0x600000 - cpu.base: 0x600000 - cpu.base + 400]))
    mem_ok = stored == tuple(i * i for i in range(n))
    print(f"  {'PASS' if mem_ok else 'FAIL'}  memory contents verified byte-for-byte")
    all_ok &= mem_ok

    # 4. Function call with the stack: caller -> callee.
    #    callee: returns arg*2 + 1 ; caller: calls it, then adds 1
    callee = assemble_program(["leaq 1(%rdi,%rdi), %rax"])
    caller = assemble_program([
        "pushq %rbp",
        "movq %rsp, %rbp",
        "call *%rsi",            # call callee via pointer in RSI
        "addq $1, %rax",
        "popq %rbp",
    ])
    # place both in memory, run caller with rdi=arg, rsi=callee address
    cpu = CPU()
    callee_addr = cpu.load(callee, 0x480000)
    caller_addr = cpu.load(caller, 0x400000)
    got = cpu.call(caller_addr, 20, callee_addr)
    all_ok &= check("nested call with stack (2*20+1+1)", got, 42)

    # 5. Fibonacci iteratively.
    prog = assemble_program([
        "xorl %eax, %eax",       # a = 0
        "movl $1, %ecx",         # b = 1
        "xorl %edx, %edx",       # i = 0
        "L1:",
        "cmpq %rdi, %rdx",
        "jge L2",
        "movq %rax, %r8",
        "addq %rcx, %r8",        # a+b
        "movq %rcx, %rax",
        "movq %r8, %rcx",
        "incq %rdx",
        "jmp L1",
        "L2:",
    ])
    fib = [0, 1]
    for _ in range(45):
        fib.append(fib[-1] + fib[-2])
    for k in (0, 1, 2, 10, 25, 40):
        got, _ = run_program(prog, [k])
        all_ok &= check(f"fib({k})", got, fib[k])

    print("\n" + "=" * 64)
    print("PROGRAM RESULT:", "ALL PASS — the clone executes real programs correctly"
          if all_ok else "FAILURES PRESENT")
    print("=" * 64)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
