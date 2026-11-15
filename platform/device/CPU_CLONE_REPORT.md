# CPU Clone — Functional Replication of the AMD EPYC 9B14

This document reports what was built, what was measured, and what is honestly
not achieved. Every number here was produced by running the code in this
directory. Nothing is estimated, projected, or scripted.

## 1. Target CPU (identified, not assumed)

Read directly from `/proc/cpuinfo` on the host that runs this project:

| Property | Value |
|---|---|
| Manufacturer | AuthenticAMD |
| Model | AMD EPYC 9B14 (Zen 4) |
| Family | 25 |
| ISA | x86-64 (AMD64) |
| Cores / threads | 2 cores, 2 threads per core, 4 vCPUs |
| Extensions present | AVX-512 (F/DQ/CD/BW/VL), AVX2, FMA, BMI1/BMI2, AES, SHA-NI, AVX512-VNNI, GFNI, VAES |
| Virtualization | KVM (the host is itself a guest) |

## 2. What the clone is

`device/cpu.py` is an independent x86-64 execution engine: it decodes real
machine-code bytes and executes them against its own register file, flags
register and memory. It does not call the host CPU to perform the arithmetic,
and it is not a specification viewer or a scripted demo.

Implemented and verified:

- integer ALU: `add adc sub sbb and or xor cmp test inc dec neg not imul mul div idiv`
- shifts and rotates: `shl shr sar rol ror`
- data movement: `mov movzx movsx movsxd lea xchg push pop cmovcc bswap`
- control flow: `jmp`, all 16 `jcc`, `call ret`, `setcc`
- RFLAGS: CF, PF, AF, ZF, SF, OF with real x86 carry/overflow/borrow semantics
- addressing: ModRM, SIB with scale, RIP-relative, disp8/disp32
- string ops: `rep movsb`, `rep stosb`

Not implemented (stated plainly): x87 floating point, MMX/SSE/AVX/AVX-512,
segmentation, paging, privileged instructions, hardware interrupts.

## 3. Verification — how we know it is correct

Two independent methods, both comparing against the physical CPU running the
*same machine code*.

**Fixed differential tests** (`cpu_diff.py`): each routine is run natively
through a compiled C shim and in the clone, then the result and the six
architecturally defined flags are compared bit-for-bit.

```
54 cases, 0 mismatches
```

**Randomized differential tests** (`cpu_fuzz.py`): thousands of random operand
pairs per instruction, weighted toward carry/overflow/borrow boundaries. Bytes
come from the host assembler, so both sides always run the identical
instruction.

```
29 instructions x 500 rounds = 14,500 cases, 0 mismatches
```

**Real-program tests** (`cpu_programs.py`): whole routines with loops, memory
traffic, a stack and function calls, checked against an independent Python
reference.

```
loop sum, array sum, store+checksum, nested call, fib(0..40) — all PASS
```

The differential testing earned its keep: it caught three genuine engine bugs
that would otherwise have shipped silently.

- two-operand `IMUL` set CF/OF with unsigned arithmetic, so overflow was
  mis-detected for negative products;
- three-operand `IMUL` had the same fault;
- `ROL`/`ROR` set OF for every count, but x86 defines OF only for a count of 1.

## 4. Performance — measured, not claimed

`cpu_bench.py` runs the same machine code natively and in the clone, best of
several runs, on this host.

| Workload | Native (ops/s) | Clone (ops/s) | Slowdown |
|---|---:|---:|---:|
| integer add | 2,542,506,902 | 315,257 | 8,065x |
| integer multiply | 1,134,980,765 | 341,405 | 3,324x |
| shift left | 2,256,831,485 | 231,770 | 9,737x |
| rotate | 2,276,358,582 | 266,725 | 8,534x |
| compare + setcc | 1,134,485,032 | 116,757 | 9,717x |
| lea | 1,697,595,153 | 416,845 | 4,072x |
| bitwise xor | 2,540,114,117 | 336,139 | 7,557x |
| memory store/load | 2,260,670,960 | 189,452 | 11,933x |

**Median slowdown: ~8,300x.**

This is the honest headline, and it does not flatter the project. The clone
reproduces the *architecture*. It does not reproduce the *performance*, and it
cannot: a software clone executes on the very processor it imitates, so it can
only ever consume that processor's cycles, never add to them. Cloning a CPU in
software does not manufacture compute capacity.

## 5. Self-optimization — the loop, and its honest result

`cpu_optimize.py` implements the measure → optimize → verify → keep-or-discard
loop, with correctness as a hard gate.

The optimization under test is a basic-block translator (`jit.py`) that
compiles each block of machine code to a Python function once, removing the
per-instruction decode and dispatch cost.

| Stage | Time (loop, 20k iterations x 5) | Speedup | Decision |
|---|---:|---:|---|
| interpreter | 1.593 s | 1.00x | baseline |
| JIT, partial coverage | 1.897 s | 0.84x | **discarded** (slower) |
| JIT, extended coverage | 0.328 s | **4.86x** | **kept** (correct and faster) |

The first JIT attempt was *slower* than the interpreter, and the engine said
so instead of pretending otherwise. Extending instruction coverage so a whole
loop body becomes one translated block produced a verified 4.86x — still about
1,700x slower than native, but a real, measured improvement.

`cpu_jit_verify.py` proves the JIT agrees with both the interpreter and the
physical CPU on every case before it is allowed to run.

## 6. Continuous operation (`cpu_supervisor.py`)

| Mechanism | Test | Result |
|---|---|---|
| health monitoring | self-test program must return 42 | PASS |
| watchdog | infinite loop killed after 1,000 instructions | PASS |
| fault isolation | null dereference caught, supervisor survives | PASS |
| checkpoint / restore | full state round-trip is byte-identical | PASS |
| integrity check | corrupted checkpoint is rejected | PASS |
| recovery | healthy after faults and watchdog kills | PASS |

No claim of infinite operation. The claim is that a crash is detected,
recorded, and recoverable from the last checkpoint.

## 7. The neural network that runs on the clone

`ai_model.py`, `clone_net.py` and `clone_net_build.py` implement a small
character-level network whose forward pass is x86-64 machine code, executed by
the clone. This is the part that answers "does the compute come from the clone
or the host".

**How it works.** The model is trained in floating point (numpy, training
only). Its weights are then quantized to integers, and the entire forward pass
— every multiply and add of both layers, the ReLU, and the argmax — is emitted
as x86-64 assembly by `ai_model.emit_asm`. That assembly is assembled to
machine code at build time and stored as `clone_net_artifacts/code.bin`. At run
time, `clone_net.py` loads the weights and the machine code into the clone's
memory and calls it. The clone decodes and executes each instruction itself.

**No math library performs the inference.** numpy appears in exactly two
places, neither of them inference: training the weights, and an independent
reference implementation of the same integer arithmetic used to verify the
clone. The serving path (`clone_net.py`) imports only the clone.

**Verification.**

| Check | Result |
|---|---|
| per-token: clone vs numpy integer reference, every corpus context | 263 contexts, 0 mismatches |
| end-to-end: generated text, clone vs reference | identical |
| forward pass size | 12,044 bytes of x86-64, 2,227 assembly lines |
| instructions per forward pass (executed by the clone) | 2,153 |

Generated text from the clone: `'the fox runs. the fox runs. the fox runs. …'`

**Measured speed** (honest): about **133 forward passes/s** (7.5 ms each) on
the development host, and about **88 passes/s** (11.3 ms each) on the Vercel
serverless runtime in `iad1`. This is far slower than a native BLAS, which is
the expected and honest consequence of interpreting every instruction in
software. The point of the exercise is provenance — the arithmetic is executed
by the clone — not speed.

## 8. Live deployment

| Endpoint | What it does |
|---|---|
| `GET  /api/cpu` | capability report, target-CPU spec, and the AI provenance block |
| `POST /api/cpu {"example": "sum_1_to_100"}` | run a pre-assembled program on the clone |
| `POST /api/cpu {"code": "<hex>"}` | run arbitrary machine code on the clone |
| `POST /api/cpu {"generate": "the ", "n": 80}` | run the neural network on the clone |

Live at `https://vqwen-platform.vercel.app/api/cpu`, page at `/cpu/`.
Verified live: `sum_1_to_100` returns 5050 in 303 instructions; `factorial_10`
returns 3628800; AI generation returns text with `math_library_used: false` and
128,842 instructions executed for 61 forward passes.

The serverless runtime has no assembler, so programs are sent either as
pre-assembled machine code (`code`) or by name (`example`). The examples are
assembled at build time by `device/asm.py`. The `program` form still works
wherever gcc is present.

## 9. Files

| File | Role |
|---|---|
| `cpu.py` | the x86-64 execution engine |
| `cpu_diff.py` | fixed differential tests vs the physical CPU |
| `cpu_fuzz.py` | randomized differential tests |
| `cpu_programs.py` | real-program execution tests |
| `asm.py` | assembles test instructions with the host assembler |
| `cpu_bench.py` | native-vs-clone benchmark, writes `cpu_bench.json` |
| `jit.py` | basic-block translator |
| `cpu_jit_verify.py` | JIT correctness gate |
| `cpu_optimize.py` | self-optimization loop, writes `cpu_opt_history.json` |
| `cpu_supervisor.py` | 24/7 mechanisms |
| `ai_model.py` | trains the network, quantizes it, emits the forward pass as x86-64 |
| `clone_net.py` | loads weights and machine code, runs inference on the clone |
| `clone_net_build.py` | builds and verifies the deployable artifacts |
| `ai_on_clone.py` | verifies clone inference against the numpy integer reference |
| `examples.json` | pre-assembled example programs for the serverless runtime |
| `../api/cpu.py` | live Vercel endpoint: programs and inference on the clone |
| `../public/cpu/index.html` | the live page at /cpu/ |

## 10. What is not achieved

- Not a cycle-accurate model. There is no pipeline, cache, branch predictor or
  out-of-order execution; timing is not modelled at all.
- No floating point or vector units. Only the integer subset above is
  implemented.
- No operating-system compatibility. There is no paging, no privilege levels,
  no syscall interface, so no unmodified ELF binary will run.
- No performance equivalence, and none is possible in software. The measured
  gap is ~8,300x against the host.
- No additional physical compute capacity is created. The clone consumes host
  cycles; it does not add any.
