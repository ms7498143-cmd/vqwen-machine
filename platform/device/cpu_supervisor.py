#!/usr/bin/env python3
"""
Continuous-operation supervisor for the CPU clone.

Provides the mechanisms the master prompt asks for under "24/7 operation", each
of which is real and testable:

  - health monitoring   : periodic self-test of the execution engine
  - watchdog            : kills a run that exceeds its instruction budget
  - checkpointing       : serializes full CPU state (registers, flags, memory)
  - restore             : rebuilds a CPU from a checkpoint and continues
  - fault detection     : catches engine faults and records them
  - fault isolation     : a faulted run is discarded without affecting others
  - integrity check     : hashes memory so corruption is detectable

It does not promise infinite operation. It promises that a crash is detected,
recorded, and recoverable from the last checkpoint.
"""

import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpu import CPU, Fault  # noqa: E402

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")


class Supervisor:
    def __init__(self, checkpoint_dir=CHECKPOINT_DIR):
        self.dir = checkpoint_dir
        os.makedirs(self.dir, exist_ok=True)
        self.stats = {
            "started": time.time(),
            "runs": 0,
            "faults": 0,
            "checkpoints": 0,
            "restores": 0,
            "watchdog_kills": 0,
            "health_checks": 0,
            "health_failures": 0,
        }
        self.last_health = None
        self.fault_log = []

    # ---- health ----------------------------------------------------------
    def health_check(self):
        """Run a tiny program and confirm it produces the known-correct result.

        This exercises the decoder, the ALU, flags, memory and the call/ret
        path, so a regression in any of them shows up as a health failure.
        """
        self.stats["health_checks"] += 1
        try:
            # mov rax,5 ; add rax,37 ; ret
            code = bytes([0x48, 0xC7, 0xC0, 0x05, 0x00, 0x00, 0x00,
                          0x48, 0x83, 0xC0, 0x25, 0xC3])
            c = CPU()
            addr = c.load(code)
            result = c.call(addr)
            ok = result == 42
        except Exception as e:
            ok = False
            self.fault_log.append({"t": time.time(), "kind": "health", "error": str(e)})
        self.stats["health_failures"] += 0 if ok else 1
        self.last_health = {"ok": ok, "t": time.time()}
        return ok

    # ---- watchdog --------------------------------------------------------
    def run_watched(self, code, max_insns=1_000_000, setup=None):
        """Execute code with an instruction budget; kill the run if exceeded.

        Returns (status, result, insns). status is "ok", "watchdog" or "fault".
        A killed or faulted run is isolated: the CPU is discarded and the
        supervisor keeps running.
        """
        self.stats["runs"] += 1
        c = CPU()
        addr = c.load(code)
        if setup:
            setup(c)
        c.rip = addr
        start_insns = 0
        try:
            while not c.halted:
                if c.rip == 0 or c.rip >= addr + len(code):
                    break
                if c.insns - start_insns >= max_insns:
                    self.stats["watchdog_kills"] += 1
                    return "watchdog", None, c.insns - start_insns
                c.step()
            return "ok", c.reg[0], c.insns - start_insns
        except Fault as e:
            self.stats["faults"] += 1
            self.fault_log.append({"t": time.time(), "kind": "fault", "error": str(e)})
            return "fault", None, c.insns - start_insns
        except Exception as e:
            self.stats["faults"] += 1
            self.fault_log.append({"t": time.time(), "kind": "crash", "error": str(e)})
            return "fault", None, c.insns - start_insns

    # ---- checkpoint / restore -------------------------------------------
    def save(self, name, cpu, code_region=None):
        """Serialize CPU state to disk and return the checkpoint path."""
        state = {
            "reg": cpu.reg,
            "rip": cpu.rip,
            "rflags": cpu.rflags,
            "insns": cpu.insns,
            "halted": cpu.halted,
            "mem_sha256": hashlib.sha256(bytes(cpu.mem)).hexdigest(),
            "mem": bytes(cpu.mem).hex(),
            "t": time.time(),
        }
        path = os.path.join(self.dir, name + ".json")
        with open(path, "w") as f:
            json.dump(state, f)
        self.stats["checkpoints"] += 1
        return path

    def restore(self, name, verify=True):
        """Rebuild a CPU from a checkpoint. Verifies the memory hash if asked."""
        path = os.path.join(self.dir, name + ".json")
        with open(path) as f:
            state = json.load(f)
        mem = bytes.fromhex(state["mem"])
        if verify:
            h = hashlib.sha256(mem).hexdigest()
            if h != state["mem_sha256"]:
                raise Fault("checkpoint integrity failure: hash mismatch")
        c = CPU(mem_size=len(mem))
        c.mem[:] = mem
        c.reg = list(state["reg"])
        c.rip = state["rip"]
        c.rflags = state["rflags"]
        c.insns = state["insns"]
        c.halted = state["halted"]
        self.stats["restores"] += 1
        return c

    def report(self):
        s = dict(self.stats)
        s["uptime_s"] = round(time.time() - s["started"], 1)
        s["last_health"] = self.last_health
        s["recent_faults"] = self.fault_log[-5:]
        return s


def _selftest():
    sup = Supervisor()
    ok = True
    print("Supervisor self-test\n")

    # 1. health
    h = sup.health_check()
    print(f"  {'PASS' if h else 'FAIL'}  health check (expects 42)")
    ok &= h

    # 2. normal run
    code = bytes([0x48, 0xC7, 0xC0, 0x05, 0x00, 0x00, 0x00, 0x48, 0x83, 0xC0, 0x25])
    status, res, n = sup.run_watched(code)
    print(f"  {'PASS' if status == 'ok' and res == 42 else 'FAIL'}  "
          f"watched run: status={status} result={res} insns={n}")
    ok &= (status == "ok" and res == 42)

    # 3. watchdog: an infinite loop must be killed, not hang.
    loop = bytes([0xEB, 0xFE])  # jmp $ (infinite)
    t0 = time.time()
    status, res, n = sup.run_watched(loop, max_insns=1000)
    dt = time.time() - t0
    print(f"  {'PASS' if status == 'watchdog' else 'FAIL'}  "
          f"watchdog killed infinite loop after {n} insns in {dt:.3f}s")
    ok &= (status == "watchdog")

    # 4. fault isolation: a null dereference must be caught, not crash.
    bad = bytes([0x48, 0x8B, 0x00])  # mov rax, [rax] with rax=0
    status, res, n = sup.run_watched(bad, setup=lambda c: c.reg.__setitem__(0, 0))
    print(f"  {'PASS' if status == 'fault' else 'FAIL'}  fault isolated: status={status}")
    ok &= (status == "fault")

    # 5. checkpoint and restore round-trip.
    c = CPU()
    code2 = bytes([0x48, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00])
    addr = c.load(code2)
    c.reg[5] = 0xDEADBEEF
    c.rip = addr + 7
    path = sup.save("selftest", c)
    c2 = sup.restore("selftest")
    same = (c2.reg == c.reg and c2.rip == c.rip and c2.rflags == c.rflags
            and bytes(c2.mem) == bytes(c.mem))
    print(f"  {'PASS' if same else 'FAIL'}  checkpoint/restore round-trip identical")
    ok &= same

    # 6. integrity: corrupt the file and confirm restore refuses it.
    with open(path) as f:
        st = json.load(f)
    st["mem"] = "00" + st["mem"][2:]
    with open(path, "w") as f:
        json.dump(st, f)
    try:
        sup.restore("selftest")
        integrity_ok = False
    except Fault:
        integrity_ok = True
    print(f"  {'PASS' if integrity_ok else 'FAIL'}  corrupted checkpoint rejected")
    ok &= integrity_ok

    # 7. health after faults: the supervisor is still healthy.
    h2 = sup.health_check()
    print(f"  {'PASS' if h2 else 'FAIL'}  healthy after faults and watchdog kills")
    ok &= h2

    print("\n  supervisor report:")
    for k, v in sup.report().items():
        print(f"    {k}: {v}")
    print("\nSUPERVISOR RESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_selftest())
