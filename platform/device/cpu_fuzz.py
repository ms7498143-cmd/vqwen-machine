#!/usr/bin/env python3
"""
Randomized differential verification (fuzzing) of the CPU clone.

For each instruction form, thousands of random operand pairs are generated and
the same machine code is executed natively and in the clone. Any mismatch in
the result or in an architecturally defined flag is reported with the exact
bytes and operands, so it can be reproduced.

This is stronger than the fixed cases in cpu_diff.py because it explores the
carry, overflow and borrow boundaries far more thoroughly than hand-picked
inputs. It is also the loop the self-optimization engine uses: any change to
cpu.py must keep this at zero mismatches.
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpu_diff import build_harness, flags_from_word, clone_run, HOST_FLAGS  # noqa: E402
from asm import assemble  # noqa: E402

# name, asm, bytes, defined flags, generator of operand pairs
TESTS = [
    # (name, asm, defined flags). Bytes come from the real assembler, so the
    # native run and the clone always execute the same instruction.
    ("ADD r64,r64", "addq %rbx, %rax", HOST_FLAGS),
    ("SUB r64,r64", "subq %rbx, %rax", HOST_FLAGS),
    ("AND r64,r64", "andq %rbx, %rax", HOST_FLAGS),
    ("OR  r64,r64", "orq %rbx, %rax", HOST_FLAGS),
    ("XOR r64,r64", "xorq %rbx, %rax", HOST_FLAGS),
    ("ADC r64,r64", "addq $0, %rax; adcq %rbx, %rax", HOST_FLAGS),
    ("SBB r64,r64", "subq $0, %rax; sbbq %rbx, %rax", HOST_FLAGS),
    ("ADD r32,r32", "addl %ebx, %eax", HOST_FLAGS),
    ("SUB r32,r32", "subl %ebx, %eax", HOST_FLAGS),
    ("IMUL r64,r64", "imulq %rbx, %rax", ["CF", "OF"]),
    ("NEG r64", "negq %rax", HOST_FLAGS),
    ("NOT r64", "notq %rax", []),
    ("INC r64", "incq %rax", ["PF", "AF", "ZF", "SF", "OF"]),
    ("DEC r64", "decq %rax", ["PF", "AF", "ZF", "SF", "OF"]),
    ("SHL r64,cl", "mov $13, %cl; shlq %cl, %rax", ["CF", "PF", "ZF", "SF"]),
    ("SHR r64,cl", "mov $13, %cl; shrq %cl, %rax", ["CF", "PF", "ZF", "SF"]),
    ("SAR r64,cl", "mov $13, %cl; sarq %cl, %rax", ["CF", "PF", "ZF", "SF"]),
    ("ROL r64,1", "mov $1, %cl; rolq %cl, %rax", ["CF", "OF"]),
    ("ROR r64,1", "mov $1, %cl; rorq %cl, %rax", ["CF", "OF"]),
    ("ROL r64,13", "mov $13, %cl; rolq %cl, %rax", ["CF"]),
    ("ROR r64,13", "mov $13, %cl; rorq %cl, %rax", ["CF"]),
    ("CMP r64,r64", "cmpq %rbx, %rax", HOST_FLAGS),
    ("MUL r64", "mulq %rbx", ["CF", "OF"]),
    ("LEA base+idx*8+16", "leaq 16(%rax,%rbx,8), %rax", []),
    ("LEA base+idx*1", "leaq (%rax,%rbx,1), %rax", []),
    ("MOVZX r64,r8", "movzbq %bl, %rax", []),
    ("MOVSX r64,r8", "movsbq %bl, %rax", []),
    ("MOVSXD r64,r32", "movslq %ebx, %rax", []),
    ("BSWAP r64", "bswap %rax", []),
]


def gen_operand(rng):
    """Weight the generator toward boundary values, where bugs hide."""
    choice = rng.random()
    if choice < 0.35:
        return rng.choice([0, 1, 2, 0x7F, 0x80, 0xFF,
                           0x7FFF, 0x8000, 0xFFFF,
                           0x7FFFFFFF, 0x80000000, 0xFFFFFFFF,
                           0x7FFFFFFFFFFFFFFF, 0x8000000000000000,
                           0xFFFFFFFFFFFFFFFF])
    if choice < 0.6:
        return rng.getrandbits(rng.choice([1, 4, 8, 12, 16, 24, 32, 48, 63]))
    return rng.getrandbits(64)


def main(rounds_per_test=2000, seed=1234):
    rng = random.Random(seed)
    total = 0
    failures = 0
    print(f"Randomized differential verification — {rounds_per_test} rounds per instruction\n")
    for name, asm, defined in TESTS:
        try:
            code = assemble(asm)
        except Exception as e:
            print(f"  SKIP  {name:<26} assembler rejected: {e}")
            continue
        lib = build_harness(asm)
        bad = 0
        first = None
        for _ in range(rounds_per_test):
            a, b = gen_operand(rng), gen_operand(rng)
            hw = lib.run_case(a, b, 0)
            hres = hw.result & (2**64 - 1)
            hflags = flags_from_word(hw.flags)
            try:
                cres, cflags = clone_run(code, a, b, 0)
            except Exception as e:
                bad += 1
                first = first or (a, b, "exception: %s" % e)
                continue
            total += 1
            if hres != cres or any(hflags[f] != cflags[f] for f in defined):
                bad += 1
                if first is None:
                    mism = [f for f in defined if hflags[f] != cflags[f]]
                    first = (a, b, f"host={hres:#x} clone={cres:#x} flags={mism}")
        failures += bad
        status = "PASS" if bad == 0 else "FAIL"
        detail = "" if bad == 0 else f"  first mismatch: a={first[0]:#x} b={first[1]:#x} {first[2]}"
        print(f"  {status}  {name:<26} {rounds_per_test} rounds{detail}")

    print("\n" + "=" * 64)
    print(f"cases run: {total}   mismatches: {failures}")
    print("RANDOMIZED RESULT:", "ALL PASS" if failures == 0 else "FAILURES PRESENT")
    print("=" * 64)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    sys.exit(main(rounds))
