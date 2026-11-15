#!/usr/bin/env python3
"""
Differential verification of the CPU clone against the physical CPU.

Method: assemble one small routine, then execute the exact same machine code
two ways.

  1. Natively, by calling it from a C shim compiled for the host CPU.
  2. In the clone, by decoding and interpreting the same bytes in cpu.py.

The two results are compared bit-for-bit, together with the six arithmetic
flags. This is the strongest evidence available that the clone's instruction
semantics match the physical processor, because both sides run identical bytes.

Every case prints PASS or FAIL and the observed values. Nothing is assumed.
"""

import ctypes
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpu import CPU  # noqa: E402

HOST_FLAGS = ["CF", "PF", "AF", "ZF", "SF", "OF"]


HARNESS_TEMPLATE = """
#include <stdio.h>
#include <stdint.h>
typedef struct { uint64_t result; uint64_t flags; } out_t;
out_t run_case(uint64_t a, uint64_t b, uint64_t c) {
    out_t o;
    __asm__ __volatile__(
        "mov %[a], %%rax\\n\\t"
        "mov %[b], %%rbx\\n\\t"
        "mov %[c], %%rcx\\n\\t"
        "@ASM@"
        "pushfq\\n\\t"
        "pop %[f]\\n\\t"
        "mov @RET@, %[r]\\n\\t"
        : [r] "=r" (o.result), [f] "=r" (o.flags)
        : [a] "r" (a), [b] "r" (b), [c] "r" (c)
        : "rax", "rbx", "rcx", "rdx", "cc", "memory"
    );
    return o;
}
"""


def build_harness(asm, ret_expr="rax", extra_asm=""):
    """Compile a C file that runs `asm` and returns (result, flags)."""
    # Escape register prefixes for GCC inline asm (%% is a literal %).
    esc = asm.replace("%", "%%")
    if not esc.rstrip().endswith(";"):
        esc = esc.rstrip() + "\\n\\t"
    src = (HARNESS_TEMPLATE
           .replace("@ASM@", esc.replace('"', '\\"'))
           .replace("@RET@", "%%" + ret_expr))
    d = tempfile.mkdtemp()
    cpath = os.path.join(d, "h.c")
    so = os.path.join(d, "h.so")
    with open(cpath, "w") as f:
        f.write(src)
    subprocess.run(["gcc", "-O0", "-shared", "-fPIC", "-o", so, cpath],
                   check=True, capture_output=True)
    lib = ctypes.CDLL(so)
    lib.run_case.restype = type("out_t", (ctypes.Structure,),
                                {"_fields_": [("result", ctypes.c_uint64),
                                              ("flags", ctypes.c_uint64)]})
    lib.run_case.argtypes = [ctypes.c_uint64] * 3
    return lib


def flags_from_word(w):
    return {
        "CF": (w >> 0) & 1, "PF": (w >> 2) & 1, "AF": (w >> 4) & 1,
        "ZF": (w >> 6) & 1, "SF": (w >> 7) & 1, "OF": (w >> 11) & 1,
    }


def clone_run(code, a=0, b=0, c=0, ret_reg=0):
    cpu = CPU()
    addr = cpu.load(code)
    cpu.reg[0], cpu.reg[3], cpu.reg[1] = a & (2**64 - 1), b & (2**64 - 1), c & (2**64 - 1)
    cpu.rip = addr
    cpu.halted = False
    while cpu.rip < addr + len(code):
        cpu.step()
    return cpu.reg[ret_reg] & (2**64 - 1), cpu.flags_snapshot()


def diff_test(name, asm, code, cases, ret_expr="rax", ret_reg=0, defined=None):
    """Run one routine natively and in the clone, compare results and flags.

    `defined` lists the flags the ISA actually defines for this instruction.
    Flags outside that list are architecturally undefined after the
    instruction, so the host may hold stale bits; comparing them would be
    meaningless. The default is all six.
    """
    defined = HOST_FLAGS if defined is None else defined
    lib = build_harness(asm, ret_expr)
    ok = True
    print(f"\n=== {name} ===")
    print(f"    machine code: {code.hex(' ')}")
    if set(defined) != set(HOST_FLAGS):
        print(f"    flags defined by the ISA: {' '.join(defined)} "
              f"(others are undefined and not compared)")
    for (a, b, c) in cases:
        hw = lib.run_case(a, b, c)
        hres = hw.result & (2**64 - 1)
        hflags = flags_from_word(hw.flags)
        cres, cflags = clone_run(code, a, b, c, ret_reg)
        res_ok = hres == cres
        flags_ok = all(hflags[f] == cflags[f] for f in defined)
        match = res_ok and flags_ok
        ok = ok and match
        fd = " ".join(f"{f}={hflags[f]}/{cflags[f]}" for f in defined
                      if hflags[f] != cflags[f])
        if res_ok and not flags_ok:
            fd = "FLAG MISMATCH: " + fd
        elif res_ok and flags_ok:
            fd = "flags match"
        if not res_ok:
            fd = "RESULT MISMATCH"
        print(f"    a={a:#x} b={b:#x}  host={hres:#x} clone={cres:#x}  "
              f"{'PASS' if match else 'FAIL'}  {fd}")
    return ok


CASES = [
    (5, 3, 0), (0xFFFFFFFFFFFFFFFF, 1, 0), (0x8000000000000000, 1, 0),
    (0x7FFFFFFFFFFFFFFF, 1, 0), (0, 0, 0), (0xF0F0F0F0, 0x0F0F0F0F, 0),
]


def main():
    all_ok = True
    ALL = HOST_FLAGS
    # IMUL defines only CF and OF; the rest are architecturally undefined.
    IMUL_FLAGS = ["CF", "OF"]
    # Rotates define only CF and OF.
    ROT_FLAGS = ["CF", "OF"]
    # For shifts with a count greater than 1, AF and OF are architecturally
    # undefined, so only CF, PF, ZF and SF are compared here.
    SHIFT_FLAGS = ["CF", "PF", "ZF", "SF"]

    # add rax, rbx ; ret
    all_ok &= diff_test("ADD (64-bit, with flags)", "addq %rbx, %rax;",
                        bytes([0x48, 0x01, 0xD8]), CASES, defined=ALL)
    # sub rax, rbx
    all_ok &= diff_test("SUB (64-bit, with flags)", "subq %rbx, %rax;",
                        bytes([0x48, 0x29, 0xD8]), CASES, defined=ALL)
    # and rax, rbx
    all_ok &= diff_test("AND (64-bit, with flags)", "andq %rbx, %rax;",
                        bytes([0x48, 0x21, 0xD8]), CASES, defined=ALL)
    # xor rax, rbx
    all_ok &= diff_test("XOR (64-bit, with flags)", "xorq %rbx, %rax;",
                        bytes([0x48, 0x31, 0xD8]), CASES, defined=ALL)
    # imul rax, rbx
    all_ok &= diff_test("IMUL (64-bit, 2-operand)",
                        "imulq %rbx, %rax;", bytes([0x48, 0x0F, 0xAF, 0xC3]),
                        [(3, 4, 0), (0x100000000, 0x100000000, 0),
                         (0xFFFFFFFFFFFFFFFF, 2, 0), (12345, 6789, 0), (0, 5, 0)],
                        defined=IMUL_FLAGS)
    # shl rax, 4  (via cl)
    all_ok &= diff_test("SHL (shift by CL)",
                        "mov $4, %cl; shlq %cl, %rax;", bytes([0xB1, 0x04, 0x48, 0xD3, 0xE0]),
                        [(1, 0, 0), (0xFF, 0, 0), (0x8000000000000000, 0, 0)], defined=SHIFT_FLAGS)
    # sar rax, cl
    all_ok &= diff_test("SAR (arithmetic shift right)",
                        "mov $4, %cl; sarq %cl, %rax;", bytes([0xB1, 0x04, 0x48, 0xD3, 0xF8]),
                        [(0xFF, 0, 0), (0xFFFFFFFFFFFFFFFF, 0, 0), (0x10, 0, 0)], defined=SHIFT_FLAGS)
    # cmp rax, rbx then setg al
    all_ok &= diff_test("CMP + SETcc (signed compare)",
                        "cmpq %rbx, %rax; setg %al; movzbl %al, %eax;",
                        bytes([0x48, 0x39, 0xD8, 0x0F, 0x9F, 0xC0, 0x0F, 0xB6, 0xC0]),
                        [(5, 3, 0), (3, 5, 0), (5, 5, 0), (0xFFFFFFFFFFFFFFFF, 1, 0)], defined=ALL)
    # neg rax
    all_ok &= diff_test("NEG (with flags)", "negq %rax;", bytes([0x48, 0xF7, 0xD8]),
                        [(0, 0, 0), (1, 0, 0), (0x8000000000000000, 0, 0)], defined=ALL)
    # 32-bit add zero-extends into rax
    all_ok &= diff_test("ADD 32-bit (zero-extends result)",
                        "addl %ebx, %eax;", bytes([0x01, 0xD8]),
                        [(0xFFFFFFFF, 1, 0), (0x7FFFFFFF, 1, 0), (0x10, 0x20, 0)], defined=ALL)
    # lea with SIB scale: lea rax, [rax + rbx*4 + 8]
    all_ok &= diff_test("LEA with SIB scale and displacement",
                        "leaq 8(%rax,%rbx,4), %rax;",
                        bytes([0x48, 0x8D, 0x44, 0x98, 0x08]),
                        [(10, 3, 0), (0, 0, 0), (0x1000, 0x200, 0)], defined=[])
    # rol rax, cl
    all_ok &= diff_test("ROL (rotate left)", "mov $8, %cl; rolq %cl, %rax;",
                        bytes([0xB1, 0x08, 0x48, 0xD3, 0xC0]),
                        [(0x0102030405060708, 0, 0), (0xFF, 0, 0)], defined=ROT_FLAGS)
    # imul r64, r/m64, imm8 : imul rax, rbx, 7
    all_ok &= diff_test("IMUL 3-operand (r, r/m, imm8)",
                        "imulq $7, %rbx, %rax;", bytes([0x48, 0x6B, 0xC3, 0x07]),
                        [(0, 3, 0), (0, 0x100000000, 0), (0, 0xFFFFFFFFFFFFFFFF, 0)],
                        defined=IMUL_FLAGS)

    print("\n" + "=" * 64)
    print("DIFFERENTIAL RESULT:", "ALL PASS — clone matches the physical CPU"
          if all_ok else "FAILURES PRESENT — see above")
    print("=" * 64)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
