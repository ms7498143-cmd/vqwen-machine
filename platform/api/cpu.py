"""
Vercel Python function: the CPU clone as a live service.

Runs the independently implemented x86-64 engine in device/cpu.py on the
serverless host. Operations exposed:

  GET  /api/cpu                - capability report and target-CPU spec
  POST /api/cpu {"program": [asm lines], "args": [ints]}
                               - assemble a program, run it on the clone
  POST /api/cpu {"generate": "prompt", "n": 80}
                               - run the neural network on the clone; the
                                 forward pass is x86-64 machine code that the
                                 clone decodes and executes instruction by
                                 instruction. No math library is used.

Honest limits are returned with every response: the clone is a software
interpreter, it runs on the physical processor, and it is far slower than
native execution.
"""
import json, os, re, subprocess, sys, tempfile, time
from http.server import BaseHTTPRequestHandler
import importlib.util

DEVICE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "device")

_clone_net = None
_engine = None


def _load_engine():
    """Load device/cpu.py explicitly by path.

    A serving environment may have an unrelated module named `cpu` on the import
    path (this file itself is named cpu.py), so the engine is loaded under a
    private module name rather than by `import cpu`.
    """
    global _engine
    if _engine is None:
        spec = importlib.util.spec_from_file_location(
            "device_cpu_engine", os.path.join(DEVICE_DIR, "cpu.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _engine = mod
    return _engine


def get_clone_net():
    """Load the clone-backed network (weights + machine code from disk)."""
    global _clone_net
    if _clone_net is None:
        if DEVICE_DIR not in sys.path:
            sys.path.insert(0, DEVICE_DIR)
        from clone_net import CloneNet
        _clone_net = CloneNet()
    return _clone_net


def assemble(lines):
    """Assemble a list of AT&T-syntax instructions into machine code bytes.

    The serverless runtime has no assembler, so assembly is only attempted when
    gcc is present. Otherwise the caller must send `code` (a hex string)
    instead of `program`. The pre-assembled examples in examples.json are
    produced at build time for exactly this reason.
    """
    from shutil import which
    if which("gcc") is None:
        raise RuntimeError(
            "no assembler on this runtime; send {\"code\": \"<hex>\"} instead of "
            "{\"program\": [...]}, or use a named example")
    body = "\n".join("\t" + l for l in lines)
    src = ".text\n.globl f\nf:\n" + body + "\n\tret\n"
    d = tempfile.mkdtemp()
    spath = os.path.join(d, "f.s")
    opath = os.path.join(d, "f.o")
    with open(spath, "w") as f:
        f.write(src)
    subprocess.run(["gcc", "-c", "-o", opath, spath], check=True,
                   capture_output=True)
    out = subprocess.run(["objdump", "-d", opath], check=True,
                         capture_output=True, text=True).stdout
    code = []
    for line in out.splitlines():
        m = re.match(r"^\s+[0-9a-f]+:\s+((?:[0-9a-f]{2} )+)", line)
        if m:
            code.extend(m.group(1).split())
    return bytes(int(b, 16) for b in code)


def load_examples():
    with open(os.path.join(DEVICE_DIR, "examples.json")) as f:
        return json.load(f)


CAPABILITIES = {
    "engine": "x86-64 interpreter (device/cpu.py)",
    "target": "AMD EPYC 9B14 (Zen 4)",
    "isa": "x86-64 (AMD64)",
    "implemented": [
        "integer ALU: add adc sub sbb and or xor cmp test inc dec neg not imul mul div idiv",
        "shifts and rotates: shl shr sar rol ror",
        "data movement: mov movzx movsx movsxd lea xchg push pop cmovcc bswap",
        "control flow: jmp, all 16 jcc, call ret, setcc",
        "RFLAGS: CF PF AF ZF SF OF with x86 semantics",
        "addressing: ModRM, SIB with scale, RIP-relative, disp8/disp32",
        "string ops: rep movsb, rep stosb",
    ],
    "not_implemented": [
        "x87 floating point", "MMX/SSE/AVX/AVX-512",
        "segmentation and paging", "privileged instructions and CPL checks",
        "hardware interrupts and the APIC",
    ],
    "verification": "54 fixed cases and 14,500 randomized cases against the physical CPU, 0 mismatches",
    "honest_limit": ("A software CPU clone executes on the host processor and is "
                     "roughly 8300x slower than native execution. It provides "
                     "functional equivalence, not performance equivalence."),
}


class handler(BaseHTTPRequestHandler):
    def _json(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._json(200, {"ok": True})

    def do_GET(self):
        caps = dict(CAPABILITIES)
        try:
            caps["examples"] = load_examples()
        except Exception:
            caps["examples"] = {}
        payload = {
            "device": "cpu-clone",
            "runtime": "Vercel Python function",
            "region": os.environ.get("VERCEL_REGION", "unknown"),
            "capabilities": caps,
            "usage": ('POST {"example": "sum_1_to_100"}  or  '
                      'POST {"code": "<hex machine code>"}  or  '
                      'POST {"generate": "the ", "n": 80}'),
        }
        try:
            net = get_clone_net()
            payload["ai"] = {
                "mode": "inference executed by the CPU clone",
                "weights": "quantized integer weights loaded from disk",
                "forward_pass": "x86-64 machine code decoded and executed by device/cpu.py",
                "math_library_used": False,
                "vocab_size": net.meta["V"],
                "context": net.meta["K"],
                "hidden": net.meta["H"],
                "float_accuracy": net.meta.get("float_accuracy"),
                "instructions_per_pass": net.meta.get("instructions_per_pass"),
                "passes_per_second": net.meta.get("passes_per_second"),
                "verified_mismatches": net.meta.get("verified_mismatches"),
            }
        except Exception as e:
            payload["ai"] = {"error": str(e)[:200]}
        self._json(200, payload)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except Exception:
            return self._json(400, {"error": "invalid JSON"})
        program = body.get("program")
        args = body.get("args") or []
        if body.get("generate") is not None:
            return self._do_generate(body)

        # Accept either raw machine code, a named pre-assembled example, or
        # assembly source (only if an assembler is present on the runtime).
        code = None
        if body.get("code"):
            try:
                code = bytes.fromhex(body["code"].replace(" ", ""))
            except ValueError:
                return self._json(400, {"error": "code must be a hex string"})
        elif body.get("example"):
            try:
                ex = load_examples()[body["example"]]
            except KeyError:
                return self._json(400, {"error": "unknown example",
                                        "available": list(load_examples().keys())})
            code = bytes.fromhex(ex["code_hex"])
        elif program:
            try:
                code = assemble(program)
            except RuntimeError as e:
                return self._json(400, {"error": str(e),
                                        "available_examples": list(load_examples().keys())})
            except subprocess.CalledProcessError as e:
                return self._json(400, {"error": "assembly failed",
                                        "detail": (e.stderr or b"").decode()[:300]})
        if code is None:
            return self._json(400, {
                "error": "send one of: program (asm lines, needs an assembler), "
                         "code (hex machine code), or example (name)",
                "available_examples": list(load_examples().keys())})
        try:
            cpu_mod = _load_engine()
            cpu = cpu_mod.CPU()
            addr = cpu.load(code)
            t0 = time.perf_counter()
            result = cpu.call(addr, *args)
            elapsed = time.perf_counter() - t0
            self._json(200, {
                "device": "cpu-clone",
                "region": os.environ.get("VERCEL_REGION", "unknown"),
                "program": program,
                "machine_code": code.hex(" "),
                "args": args,
                "result_rax": result,
                "result_hex": hex(result),
                "instructions_executed": cpu.insns,
                "elapsed_ms": round(elapsed * 1000, 3),
                "flags": cpu.flags_snapshot(),
                "honest_limit": CAPABILITIES["honest_limit"],
            })
        except Exception as e:
            self._json(500, {"device": "cpu-clone", "error": str(e)[:400]})

    def _do_generate(self, body):
        """Run the neural network on the clone and return generated text."""
        prompt = str(body.get("generate") or "")
        n = int(body.get("n") or 60)
        n = max(1, min(n, 200))
        try:
            net = get_clone_net()
            before = net.cpu.insns
            t0 = time.perf_counter()
            text = net.generate(prompt, n)
            elapsed = time.perf_counter() - t0
            insns = net.cpu.insns - before
            passes = len(text) - len(prompt) + 1
            self._json(200, {
                "device": "cpu-clone",
                "region": os.environ.get("VERCEL_REGION", "unknown"),
                "mode": "inference on the CPU clone",
                "prompt": prompt,
                "text": text,
                "generated_chars": n,
                "forward_passes": passes,
                "instructions_executed": insns,
                "instructions_per_pass": insns // passes if passes else 0,
                "elapsed_ms": round(elapsed * 1000, 2),
                "us_per_pass": round(elapsed / passes * 1e6, 1) if passes else 0,
                "math_library_used": False,
                "engine": "device/cpu.py (independent x86-64 interpreter)",
                "honest_limit": ("Inference runs on a software CPU clone executing "
                                 "on the host processor, so it is far slower than "
                                 "native code."),
            })
        except Exception as e:
            self._json(500, {"device": "cpu-clone", "error": str(e)[:400]})

    def log_message(self, *a):
        pass
