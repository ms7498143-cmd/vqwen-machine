#!/usr/bin/env python3
"""
Assemble an instruction with the host assembler and return its machine code.

Hand-written opcode bytes are a source of test error: a wrong byte means the
test compares two different instructions. This helper removes that risk by
letting the real assembler produce the bytes, so the native run and the clone
run are always the same instruction.

Returns the bytes of the instruction only (the trailing `ret` is stripped).
"""

import os
import re
import subprocess
import tempfile


def assemble(asm, strip_ret=True):
    """Assemble one AT&T-syntax instruction and return its machine code bytes."""
    src = ".text\n.globl f\nf:\n\t" + asm.strip() + "\n\tret\n"
    d = tempfile.mkdtemp()
    spath = os.path.join(d, "f.s")
    opath = os.path.join(d, "f.o")
    with open(spath, "w") as f:
        f.write(src)
    subprocess.run(["gcc", "-c", "-o", opath, spath], check=True,
                   capture_output=True)
    out = subprocess.run(["objdump", "-d", opath], check=True,
                         capture_output=True, text=True).stdout
    # Collect the hex bytes of the disassembly, which includes our instruction
    # followed by the trailing `ret`.
    code = []
    started = False
    for line in out.splitlines():
        m = re.match(r"^\s+[0-9a-f]+:\s+((?:[0-9a-f]{2} )+)", line)
        if not m:
            continue
        code.extend(m.group(1).split())
    if not code:
        raise RuntimeError("could not assemble: " + asm)
    if strip_ret and code[-1] == "c3":
        code = code[:-1]
    return bytes(int(b, 16) for b in code)


def verify_consistency(cases):
    """Sanity check: print each instruction with its assembled bytes."""
    for name, asm in cases:
        print(f"  {name:<28} {asm:<34} -> {assemble(asm).hex(' ')}")


if __name__ == "__main__":
    verify_consistency([
        ("ADD", "addq %rbx, %rax"),
        ("LEA idx*8", "leaq 16(%rax,%rbx,8)"),
        ("IMUL 3op", "imulq $7, %rbx, %rax"),
        ("ROR cl", "mov $13, %cl; rorq %cl, %rax"),
    ])
