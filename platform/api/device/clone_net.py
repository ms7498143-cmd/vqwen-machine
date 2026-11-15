#!/usr/bin/env python3
"""
Runtime for the neural network that executes on the CPU clone.

The trained weights and the assembled forward-pass machine code are produced
offline by clone_net_build.py and stored as small binary files. At run time
this module only loads those files and executes them with device/cpu.py, so no
compiler or math library is needed in the serving environment.

Nothing in the inference path uses numpy. numpy is used only to verify, in
clone_net_build.py, that the clone's arithmetic matches an independent
reference.
"""

import json
import os
import struct
import sys

# The engine lives in this same directory as cpu.py. It is loaded explicitly by
# path under a private module name, because a serving environment may also have
# an unrelated module named `cpu` on the import path.
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_engine():
    spec = importlib.util.spec_from_file_location(
        "device_cpu_engine", os.path.join(_HERE, "cpu.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ENGINE = _load_engine()
CPU = _ENGINE.CPU

HERE = _HERE
ARTIFACTS = os.path.join(HERE, "clone_net_artifacts")

WEIGHTS_BASE = 0x600000
CODE_BASE = 0x400000


class CloneNet:
    """The network, executed by the clone. Weights and code come from disk."""

    def __init__(self, artifacts=ARTIFACTS):
        with open(os.path.join(artifacts, "meta.json")) as f:
            self.meta = json.load(f)
        with open(os.path.join(artifacts, "code.bin"), "rb") as f:
            self.code = f.read()
        with open(os.path.join(artifacts, "data.bin"), "rb") as f:
            self.data = f.read()
        self.vocab = self.meta["vocab"]
        self.K = self.meta["K"]
        self.idx = {c: i for i, c in enumerate(self.vocab)}
        self._new_cpu()

    def _new_cpu(self):
        self.cpu = CPU()
        self.cpu.load(self.data, WEIGHTS_BASE)
        self.fwd_addr = self.cpu.load(self.code, CODE_BASE)

    def predict(self, ctx):
        """One forward pass on the clone; returns the predicted character index."""
        c = self.cpu
        c.halted = False
        return c.call(self.fwd_addr, *ctx)

    def generate(self, prompt, n=80, max_ctx_unknown="_"):
        """Generate n characters, one clone forward pass per character."""
        chars = list(prompt)
        for ch in chars:
            if ch not in self.idx:
                chars = [c for c in chars if c in self.idx]
                break
        out = [self.idx[c] for c in chars]
        if len(out) < self.K:
            return prompt
        for _ in range(n):
            nxt = self.predict(out[-self.K:])
            out.append(nxt)
        return "".join(self.vocab[i] for i in out)

    def instructions_per_pass(self, ctx=None):
        """Measure the exact number of instructions the clone executes per pass."""
        if ctx is None:
            ctx = [0] * self.K
        before = self.cpu.insns
        self.predict(ctx)
        return self.cpu.insns - before


if __name__ == "__main__":
    net = CloneNet()
    print("vocab:", "".join(net.vocab))
    print("prompt 'the ' ->", repr(net.generate("the ", 60)))
    print("instructions per forward pass:", net.instructions_per_pass())
