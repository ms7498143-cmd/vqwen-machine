#!/usr/bin/env python3
"""
A neural network whose forward pass runs as x86-64 machine code on the CPU clone.

The point of this file is that inference does not use numpy, ONNX or any host
math library. It trains a small character-level model, quantizes it to integer
weights, then emits the entire forward pass as x86-64 assembly. That assembly
is assembled to machine code and executed by device/cpu.py, which decodes and
executes every multiply and add itself.

numpy appears here in exactly two roles, neither of which is inference:
  1. training the float model, and
  2. a reference implementation of the same integer arithmetic, used to verify
     the clone produces identical results.

Layout in the clone's memory (all int16 unless noted):
  W1[H][K*V]  hidden weights, scaled by SCALE
  B1[H]       hidden bias,    scaled by SCALE
  W2[V][H]    output weights, scaled by SCALE
  B2[V]       output bias,    scaled by SCALE
  OUT[V]      int64 output logits (scratch)
  SCRATCH[H]  int64 hidden activations (scratch)
"""

import struct

import numpy as np

SCALE = 256

CORPUS = (
    "the quick brown fox jumps over the lazy dog. "
    "the dog sleeps. the fox runs. the quick fox jumps. "
    "a lazy dog sleeps and a quick fox runs. "
    "the brown fox jumps over the dog. "
    "the quick brown dog jumps over the lazy fox. "
    "the fox and the dog run. the dog and the fox sleep. "
)


def train(corpus=CORPUS, K=4, H=16, epochs=6000, lr=0.4, seed=0):
    """Train a small MLP to predict the next character from K previous ones."""
    vocab = sorted(set(corpus))
    V = len(vocab)
    idx = {c: i for i, c in enumerate(vocab)}
    data = [idx[c] for c in corpus]

    X, Y = [], []
    for i in range(K, len(data)):
        x = np.zeros(K * V, dtype=np.float32)
        for k, v in enumerate(data[i - K:i]):
            x[k * V + v] = 1.0
        X.append(x)
        Y.append(data[i])
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y)
    n = len(X)

    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 0.3, (K * V, H)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = rng.normal(0, 0.3, (H, V)).astype(np.float32)
    b2 = np.zeros(V, np.float32)

    for _ in range(epochs):
        h = np.maximum(0, X @ W1 + b1)
        logits = h @ W2 + b2
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        dlog = p
        dlog[np.arange(n), Y] -= 1.0
        dlog /= n
        dW2 = h.T @ dlog
        db2 = dlog.sum(0)
        dh = dlog @ W2.T
        dh[h <= 0] = 0
        dW1 = X.T @ dh
        db1 = dh.sum(0)
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

    acc = float((np.argmax(np.maximum(0, X @ W1 + b1) @ W2 + b2, axis=1) == Y).mean())
    return {"vocab": vocab, "V": V, "K": K, "H": H,
            "W1": W1, "b1": b1, "W2": W2, "b2": b2, "float_acc": acc}


def quantize(model):
    """Convert the trained float model to integer weights."""
    K, H, V = model["K"], model["H"], model["V"]
    W1q = np.round(model["W1"].T * SCALE).astype(np.int64)   # (H, K*V)
    b1q = np.round(model["b1"] * SCALE).astype(np.int64)     # (H,)
    W2q = np.round(model["W2"].T * SCALE).astype(np.int64)   # (V, H)
    b2q = np.round(model["b2"] * SCALE).astype(np.int64)     # (V,)
    return {"vocab": model["vocab"], "K": K, "H": H, "V": V,
            "W1q": W1q, "b1q": b1q, "W2q": W2q, "b2q": b2q,
            "float_acc": model["float_acc"]}


# ---- reference (numpy) implementation of the same integer arithmetic --------
def ref_forward(q, ctx):
    """Return the predicted next character using integer arithmetic only."""
    K, V = q["K"], q["V"]
    h = q["b1q"].copy()
    for k in range(K):
        h += q["W1q"][:, k * V + ctx[k]]
    h = np.maximum(h, 0) >> 8           # relu, then rescale to ~unit magnitude
    logits = q["b2q"] + q["W2q"] @ h
    return int(np.argmax(logits))


# ---- memory layout ----------------------------------------------------------
def layout(q):
    H, V, K = q["H"], q["V"], q["K"]
    off = {}
    off["W1"] = 0
    off["B1"] = off["W1"] + H * K * V * 2
    off["W2"] = off["B1"] + H * 2
    off["B2"] = off["W2"] + V * H * 2
    off["OUT"] = off["B2"] + V * 2
    off["SCRATCH"] = off["OUT"] + V * 8
    off["END"] = off["SCRATCH"] + H * 8
    return off


def data_bytes(q):
    """Serialize the quantized weights and scratch space for the clone's memory."""
    H, V, K = q["H"], q["V"], q["K"]
    parts = [
        struct.pack("<%dh" % (H * K * V), *[int(x) for x in q["W1q"].reshape(-1)]),
        struct.pack("<%dh" % H, *[int(x) for x in q["b1q"]]),
        struct.pack("<%dh" % (V * H), *[int(x) for x in q["W2q"].reshape(-1)]),
        struct.pack("<%dh" % V, *[int(x) for x in q["b2q"]]),
        b"\x00" * (V * 8),      # OUT
        b"\x00" * (H * 8),      # SCRATCH
    ]
    return b"".join(parts)


# ---- emit the forward pass as x86-64 assembly -------------------------------
def emit_asm(q, weights_base):
    """Emit the whole forward pass as AT&T x86-64 assembly.

    All weights are read from memory at weights_base. The four context
    characters arrive in rdi, rsi, rdx, rcx. The predicted index is returned
    in rax.
    """
    H, V, K = q["H"], q["V"], q["K"]
    o = layout(q)
    L = []
    L.append("\t.text")
    L.append("\t.globl fwd")
    L.append("fwd:")
    L.append(f"\tmovabsq ${weights_base}, %r11")   # r11 = weights base
    L.append("\txorl %r8d, %r8d")                     # r8 = 0 (for relu)

    # hidden layer: one weight per context char, because the input is one-hot
    ctx_regs = ["%rdi", "%rsi", "%rdx", "%rcx"]
    for j in range(H):
        L.append(f"\tmovswq {o['B1'] + j * 2}(%r11), %rax")
        for k in range(K):
            disp = o["W1"] + 2 * (j * K * V + k * V)
            L.append(f"\tmovswq {disp}(%r11,{ctx_regs[k]},2), %rbx")
            L.append("\taddq %rbx, %rax")
        L.append("\ttestq %rax, %rax")
        L.append("\tcmovs %r8, %rax")                 # relu
        L.append("\tsarq $8, %rax")                   # rescale
        L.append(f"\tmovq %rax, {o['SCRATCH'] + j * 8}(%r11)")

    # output layer
    for v in range(V):
        L.append(f"\tmovswq {o['B2'] + v * 2}(%r11), %rax")
        for j in range(H):
            L.append(f"\tmovq {o['SCRATCH'] + j * 8}(%r11), %r9")
            L.append(f"\tmovswq {o['W2'] + 2 * (v * H + j)}(%r11), %rbx")
            L.append("\timulq %r9, %rbx")
            L.append("\taddq %rbx, %rax")
        L.append(f"\tmovq %rax, {o['OUT'] + v * 8}(%r11)")

    # argmax over the output logits
    L.append(f"\tmovq {o['OUT']}(%r11), %r8")
    L.append("\txorl %r10d, %r10d")
    for v in range(1, V):
        L.append(f"\tmovq {o['OUT'] + v * 8}(%r11), %rax")
        L.append("\tcmpq %r8, %rax")
        L.append(f"\tjle .Lskip{v}")
        L.append("\tmovq %rax, %r8")
        L.append(f"\tmovl ${v}, %r10d")
        L.append(f".Lskip{v}:")
    L.append("\tmovq %r10, %rax")
    L.append("\tret")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    m = train()
    q = quantize(m)
    print("float accuracy: %.3f" % m["float_acc"])
    print("vocab size:", q["V"], "hidden:", q["H"], "context:", q["K"])
    print("data bytes:", len(data_bytes(q)))
    asm = emit_asm(q, 0x600000)
    print("assembly lines:", asm.count("\n"))
