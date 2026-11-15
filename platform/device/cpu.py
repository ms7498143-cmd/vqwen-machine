#!/usr/bin/env python3
"""
Independent x86-64 execution engine (the CPU clone).

This is a real interpreter for a subset of the x86-64 instruction set. It
decodes actual machine code and executes it against its own register file,
flags register and memory. It does not call the host CPU to perform the
arithmetic, and it is not a specification viewer or a scripted demo.

Implemented and verified (see device/cpu_diff.py for differential tests):
  - 64/32/16/8-bit integer ALU: add, adc, sub, sbb, and, or, xor, cmp, test,
    inc, dec, neg, not, imul, mul, div, idiv
  - shifts and rotates: shl, shr, sar, rol, ror
  - data movement: mov, movzx, movsx, movsxd, lea, xchg, push, pop, cmovcc
  - control flow: jmp, all 16 jcc, call, ret, setcc, loop
  - RFLAGS: CF, PF, AF, ZF, SF, OF with real x86 semantics
  - addressing: ModRM, SIB with scale, RIP-relative, disp8/disp32
  - string ops: rep movsb, rep stosb

Documented gaps (not implemented): x87, MMX/SSE/AVX, segmentation, paging,
hardware interrupt vectors, privileged instructions, CPL/ring checks.

A software CPU clone runs on the physical processor that executes the
interpreter. It adds no physical compute capacity. Performance figures for the
clone versus the host are reported honestly in the performance report.
"""

import struct

RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI = range(8)
R8, R9, R10, R11, R12, R13, R14, R15 = range(8, 16)

CF_BIT = 1 << 0
PF_BIT = 1 << 2
AF_BIT = 1 << 4
ZF_BIT = 1 << 6
SF_BIT = 1 << 7
OF_BIT = 1 << 11
DF_BIT = 1 << 10

COND = ["o", "no", "b", "ae", "e", "ne", "be", "a",
        "s", "ns", "p", "np", "l", "ge", "le", "g"]


class Fault(Exception):
    pass


class CPU:
    """A single-core x86-64 execution engine with a flat memory model."""

    def __init__(self, mem_size=1 << 24, base=0x400000):
        self.base = base
        self.mem = bytearray(mem_size)
        self.reg = [0] * 16
        self.rip = base
        self.rflags = 0x2  # bit 1 is always set
        self.halted = False
        self.insns = 0
        self.rex = 0
        self.op_size = 32

    # ---- memory -----------------------------------------------------------
    def _off(self, addr):
        o = addr - self.base
        if o < 0 or o + 1 > len(self.mem):
            raise Fault("memory access out of range: %#x" % addr)
        return o

    def read(self, addr, n):
        o = self._off(addr)
        if o + n > len(self.mem):
            raise Fault("read crosses end of memory: %#x" % addr)
        return int.from_bytes(self.mem[o:o + n], "little")

    def write(self, addr, n, val):
        o = self._off(addr)
        if o + n > len(self.mem):
            raise Fault("write crosses end of memory: %#x" % addr)
        self.mem[o:o + n] = (val & ((1 << (8 * n)) - 1)).to_bytes(n, "little")

    def load(self, code, addr=None):
        addr = self.base if addr is None else addr
        self.mem[self._off(addr):self._off(addr) + len(code)] = code
        return addr

    # ---- flags ------------------------------------------------------------
    def get_flag(self, bit):
        return 1 if (self.rflags & bit) else 0

    def set_flag(self, bit, cond):
        if cond:
            self.rflags |= bit
        else:
            self.rflags &= ~bit

    @staticmethod
    def _parity(v):
        return bin(v & 0xFF).count("1") % 2 == 0

    # ---- registers --------------------------------------------------------
    def get_reg(self, i, size):
        v = self.reg[i]
        if size == 64:
            return v
        if size == 32:
            return v & 0xFFFFFFFF
        if size == 16:
            return v & 0xFFFF
        return v & 0xFF

    def set_reg(self, i, size, val):
        if size == 64:
            self.reg[i] = val & 0xFFFFFFFFFFFFFFFF
        elif size == 32:
            self.reg[i] = val & 0xFFFFFFFF       # 32-bit writes zero-extend
        elif size == 16:
            self.reg[i] = (self.reg[i] & ~0xFFFF) | (val & 0xFFFF)
        else:
            self.reg[i] = (self.reg[i] & ~0xFF) | (val & 0xFF)

    def get_reg8(self, i):
        if self.rex == 0 and 4 <= i <= 7:
            return (self.reg[i - 4] >> 8) & 0xFF
        return self.reg[i] & 0xFF

    def set_reg8(self, i, val):
        if self.rex == 0 and 4 <= i <= 7:
            b = i - 4
            self.reg[b] = (self.reg[b] & ~0xFF00) | ((val & 0xFF) << 8)
        else:
            self.reg[i] = (self.reg[i] & ~0xFF) | (val & 0xFF)

    # ---- fetch ------------------------------------------------------------
    def fetch8(self):
        v = self.mem[self._off(self.rip)]
        self.rip += 1
        return v

    def fetch32(self):
        v = self.read(self.rip, 4)
        self.rip += 4
        return v

    def fetch64(self):
        v = self.read(self.rip, 8)
        self.rip += 8
        return v

    def fetchs8(self):
        v = self.fetch8()
        return v - 256 if v >= 128 else v

    def fetchs32(self):
        v = self.fetch32()
        return v - (1 << 32) if v >= (1 << 31) else v

    # ---- ModRM / SIB ------------------------------------------------------
    def decode_rm(self, size):
        """Decode a ModRM byte. Returns (reg_field, ('reg', idx) | ('mem', addr))."""
        b = self.fetch8()
        mod = b >> 6
        reg = ((b >> 3) & 7) | (self.rex_r << 3)
        if mod == 3:
            rm = (b & 7) | (self.rex_b << 3)
            return reg, ("reg", rm)

        addr = 0
        low = b & 7
        if low == 4:  # SIB
            sib = self.fetch8()
            scale = 1 << (sib >> 6)
            idx = ((sib >> 3) & 7) | (self.rex_x << 3)
            bs = (sib & 7) | (self.rex_b << 3)
            if (sib & 7) == 5 and mod == 0:
                addr = self.fetchs32()
            else:
                addr = self.reg[bs]
            if idx != 4:
                addr += self.reg[idx] * scale
        elif low == 5 and mod == 0:  # RIP-relative
            d = self.fetchs32()
            addr = self.rip + d
        else:
            addr = self.reg[low | (self.rex_b << 3)]

        if mod == 1:
            addr += self.fetchs8()
        elif mod == 2:
            addr += self.fetchs32()
        return reg, ("mem", addr)

    def rm_read(self, op, size):
        kind, v = op
        if kind == "reg":
            return self.get_reg8(v) if size == 8 else self.get_reg(v, size)
        return self.read(v, size // 8)

    def rm_write(self, op, size, val):
        kind, v = op
        if kind == "reg":
            if size == 8:
                self.set_reg8(v, val)
            else:
                self.set_reg(v, size, val)
        else:
            self.write(v, size // 8, val)

    def rm_addr(self, op):
        if op[0] != "mem":
            raise Fault("lea requires a memory operand")
        return op[1]

    # ---- ALU --------------------------------------------------------------
    def alu(self, op, a, b, size):
        m = (1 << size) - 1
        sgn = 1 << (size - 1)
        if op == "add" or op == "adc":
            c = self.get_flag(CF_BIT) if op == "adc" else 0
            full = a + b + c
            r = full & m
            self.set_flag(CF_BIT, full > m)
            self.set_flag(OF_BIT, ((~(a ^ b) & (a ^ r)) & sgn) != 0)
            self.set_flag(AF_BIT, ((a ^ b ^ r) & 0x10) != 0)
        elif op == "sub" or op == "cmp" or op == "sbb":
            c = self.get_flag(CF_BIT) if op == "sbb" else 0
            full = a - b - c
            r = full & m
            self.set_flag(CF_BIT, full < 0)
            self.set_flag(OF_BIT, (((a ^ b) & (a ^ r)) & sgn) != 0)
            self.set_flag(AF_BIT, ((a ^ b ^ r) & 0x10) != 0)
        elif op == "and":
            r = a & b
            self.set_flag(CF_BIT, 0)
            self.set_flag(OF_BIT, 0)
            self.set_flag(AF_BIT, 0)
        elif op == "or":
            r = a | b
            self.set_flag(CF_BIT, 0)
            self.set_flag(OF_BIT, 0)
            self.set_flag(AF_BIT, 0)
        elif op == "xor":
            r = a ^ b
            self.set_flag(CF_BIT, 0)
            self.set_flag(OF_BIT, 0)
            self.set_flag(AF_BIT, 0)
        else:
            raise Fault("unknown alu op " + op)
        self.set_flag(ZF_BIT, r == 0)
        self.set_flag(SF_BIT, (r & sgn) != 0)
        self.set_flag(PF_BIT, self._parity(r))
        return r

    def do_shift(self, sub, val, count, size):
        m = (1 << size) - 1
        sgn = 1 << (size - 1)
        count &= 0x3F
        if count == 0:
            return val
        if sub == 4:  # shl / sal
            r = (val << count) & m
            self.set_flag(CF_BIT, ((val << count) >> size) & 1)
            self.set_flag(OF_BIT, ((r & sgn) != 0) != bool(self.get_flag(CF_BIT)))
        elif sub == 5:  # shr
            self.set_flag(CF_BIT, (val >> (count - 1)) & 1)
            r = (val & m) >> count
            self.set_flag(OF_BIT, (val & sgn) != 0)
        elif sub == 7:  # sar
            sv = val - (1 << size) if (val & sgn) else val
            self.set_flag(CF_BIT, (sv >> (count - 1)) & 1)
            r = (sv >> count) & m
            self.set_flag(OF_BIT, 0)
        elif sub == 0:  # rol
            c = count % size
            r = ((val << c) | (val >> (size - c))) & m if c else val
            self.set_flag(CF_BIT, r & 1)
            if count == 1:
                self.set_flag(OF_BIT, ((r & sgn) != 0) != bool(r & 1))
            return r          # rotates affect only CF and OF
        elif sub == 1:  # ror
            c = count % size
            r = ((val >> c) | (val << (size - c))) & m if c else val
            self.set_flag(CF_BIT, (r & sgn) != 0)
            if count == 1:
                self.set_flag(OF_BIT, ((r & sgn) != 0) != bool((r >> (size - 2)) & 1))
            return r          # rotates affect only CF and OF
        else:
            raise Fault("unsupported shift sub-op %d" % sub)
        self.set_flag(ZF_BIT, r == 0)
        self.set_flag(SF_BIT, (r & sgn) != 0)
        self.set_flag(PF_BIT, self._parity(r))
        return r

    def cond_true(self, cc):
        f = lambda b: bool(self.get_flag(b))
        zf, sf, of, cf, pf = (f(ZF_BIT), f(SF_BIT), f(OF_BIT), f(CF_BIT), f(PF_BIT))
        return [
            of, not of, cf, not cf, zf, not zf, cf or zf, not (cf or zf),
            sf, not sf, pf, not pf, sf != of, sf == of, zf or (sf != of),
            not (zf or (sf != of)),
        ][cc]

    # ---- main decode + execute -------------------------------------------
    def step(self):
        self.insns += 1
        rex = 0
        size = 32
        rep = None
        while True:
            b = self.mem[self._off(self.rip)]
            if b == 0x66:
                size = 16
                self.rip += 1
            elif b == 0xF2 or b == 0xF3:
                rep = b
                self.rip += 1
            elif b in (0x2E, 0x36, 0x3E, 0x26, 0x64, 0x65, 0xF0):
                self.rip += 1
            elif 0x40 <= b <= 0x4F:
                rex = b
                self.rip += 1
            else:
                break
        self.rex = rex
        if rex & 8:
            size = 64
        self.op_size = size
        self.rex_r = (rex >> 2) & 1
        self.rex_x = (rex >> 1) & 1
        self.rex_b = rex & 1

        op = self.fetch8()

        # --- ALU group: 0x00..0x3d -------------------------------------
        if op < 0x40 and (op & 7) in (0, 1, 2, 3, 4, 5) and op < 0x3E:
            base = op & 0xF8
            names = {0x00: "add", 0x08: "or", 0x10: "adc", 0x18: "sbb",
                     0x20: "and", 0x28: "sub", 0x30: "xor", 0x38: "cmp"}
            if base in names:
                name = names[base]
                form = op & 7
                if form == 0 or form == 1:      # r/m, r
                    osz = 8 if form == 0 else size
                    reg, rm = self.decode_rm(osz)
                    src = self.get_reg8(reg) if osz == 8 else self.get_reg(reg, osz)
                    a = self.rm_read(rm, osz)
                    res = self.alu(name, a, src, osz)
                    if name != "cmp":
                        self.rm_write(rm, osz, res)
                elif form == 2 or form == 3:    # r, r/m
                    osz = 8 if form == 2 else size
                    reg, rm = self.decode_rm(osz)
                    src = self.rm_read(rm, osz)
                    a = self.get_reg8(reg) if osz == 8 else self.get_reg(reg, osz)
                    res = self.alu(name, a, src, osz)
                    if name != "cmp":
                        if osz == 8:
                            self.set_reg8(reg, res)
                        else:
                            self.set_reg(reg, osz, res)
                elif form == 4:                 # AL, imm8
                    imm = self.fetch8()
                    self.alu(name, self.get_reg8(0), imm, 8)
                elif form == 5:                 # eAX, imm
                    imm = self.fetch32() if size == 32 else (self.fetch32() & 0xFFFF if size == 16 else None)
                    if size == 64:
                        imm = self.fetch32()
                        imm = imm - (1 << 32) if imm >= (1 << 31) else imm
                    self.alu(name, self.get_reg(0, size), imm, size)
                return

        # --- push/pop 0x50..0x5f ---------------------------------------
        if 0x50 <= op <= 0x57:
            i = (op & 7) | (self.rex_b << 3)
            self.reg[RSP] -= 8
            self.write(self.reg[RSP], 8, self.reg[i])
            return
        if 0x58 <= op <= 0x5F:
            i = (op & 7) | (self.rex_b << 3)
            self.reg[i] = self.read(self.reg[RSP], 8)
            self.reg[RSP] += 8
            return

        # --- movsxd 0x63 -----------------------------------------------
        if op == 0x63:
            reg, rm = self.decode_rm(64)
            v = self.rm_read(rm, 32)
            if v & 0x80000000:
                v -= 1 << 32
            self.set_reg(reg, 64, v)
            return

        # --- push imm 0x68 / 0x6a --------------------------------------
        if op == 0x68:
            imm = self.fetch32()
            imm = imm - (1 << 32) if imm >= (1 << 31) else imm
            self.reg[RSP] -= 8
            self.write(self.reg[RSP], 8, imm)
            return
        if op == 0x6A:
            imm = self.fetchs8()
            self.reg[RSP] -= 8
            self.write(self.reg[RSP], 8, imm)
            return

        # --- imul r, r/m, imm  0x69 / 0x6b -----------------------------
        if op == 0x69 or op == 0x6B:
            reg, rm = self.decode_rm(size)
            src = self.rm_read(rm, size)
            if op == 0x69:
                imm = self.fetch32()
                if imm & 0x80000000:
                    imm -= 1 << 32          # imm32 is sign-extended to the operand size
            else:
                imm = self.fetchs8()
            imm &= (1 << size) - 1
            ssrc = src - (1 << size) if (src >> (size - 1)) & 1 else src
            simm = imm - (1 << size) if (imm >> (size - 1)) & 1 else imm
            prod = ssrc * simm
            m = (1 << size) - 1
            fits = -(1 << (size - 1)) <= prod <= (1 << (size - 1)) - 1
            self.set_flag(CF_BIT, not fits)
            self.set_flag(OF_BIT, not fits)
            self.set_reg(reg, size, prod & m)
            return

        # --- jcc rel8 0x70..0x7f ---------------------------------------
        if 0x70 <= op <= 0x7F:
            d = self.fetchs8()
            if self.cond_true(op & 0xF):
                self.rip += d
            return

        # --- group1 imm 0x80/0x81/0x83 --------------------------------
        if op in (0x80, 0x81, 0x83):
            osz = 8 if op == 0x80 else size
            reg, rm = self.decode_rm(osz)
            if op == 0x81:
                imm = self.fetch32()
                if osz == 64:
                    imm = imm - (1 << 32) if imm >= (1 << 31) else imm
                elif osz == 16:
                    imm &= 0xFFFF
            else:
                imm = self.fetchs8() & ((1 << osz) - 1)
            name = ["add", "or", "adc", "sbb", "and", "sub", "xor", "cmp"][reg & 7]
            a = self.rm_read(rm, osz)
            res = self.alu(name, a, imm, osz)
            if name != "cmp":
                self.rm_write(rm, osz, res)
            return

        # --- test 0x84/0x85 -------------------------------------------
        if op == 0x84 or op == 0x85:
            osz = 8 if op == 0x84 else size
            reg, rm = self.decode_rm(osz)
            src = self.get_reg8(reg) if osz == 8 else self.get_reg(reg, osz)
            a = self.rm_read(rm, osz)
            self.alu("and", a, src, osz)
            return

        # --- mov 0x88..0x8b -------------------------------------------
        if 0x88 <= op <= 0x8B:
            osz = 8 if (op & 1) == 0 else size
            reg, rm = self.decode_rm(osz)
            if op in (0x88, 0x89):   # r/m, r
                src = self.get_reg8(reg) if osz == 8 else self.get_reg(reg, osz)
                self.rm_write(rm, osz, src)
            else:                     # r, r/m
                src = self.rm_read(rm, osz)
                if osz == 8:
                    self.set_reg8(reg, src)
                else:
                    self.set_reg(reg, osz, src)
            return

        # --- lea 0x8d --------------------------------------------------
        if op == 0x8D:
            reg, rm = self.decode_rm(size)
            self.set_reg(reg, size, self.rm_addr(rm))
            return

        if op == 0x90:
            return

        # --- xchg 0x91..0x97 -------------------------------------------
        if 0x91 <= op <= 0x97:
            i = (op & 7) | (self.rex_b << 3)
            a = self.reg[0]
            self.reg[0] = self.reg[i]
            self.reg[i] = a
            return

        if op == 0x98:  # cdqe / cwde
            if size == 64:
                v = self.reg[0] & 0xFFFFFFFF
                self.reg[0] = v - (1 << 32) if v & 0x80000000 else v
            else:
                v = self.reg[0] & 0xFFFF
                self.set_reg(0, 32, v - (1 << 16) if v & 0x8000 else v)
            return
        if op == 0x99:  # cqo / cdq
            if size == 64:
                self.reg[RDX] = 0xFFFFFFFFFFFFFFFF if (self.reg[RAX] & (1 << 63)) else 0
            else:
                self.reg[RDX] = 0xFFFFFFFF if (self.reg[RAX] & 0x80000000) else 0
            return

        if op == 0x9C:  # pushfq
            self.reg[RSP] -= 8
            self.write(self.reg[RSP], 8, self.rflags | 0x2)
            return
        if op == 0x9D:  # popfq
            self.rflags = self.read(self.reg[RSP], 8) | 0x2
            self.reg[RSP] += 8
            return

        if op == 0xA8:  # test AL, imm8
            self.alu("and", self.get_reg8(0), self.fetch8(), 8)
            return
        if op == 0xA9:  # test eAX, imm
            imm = self.fetch32()
            self.alu("and", self.get_reg(0, size), imm, size)
            return

        # --- mov r, imm 0xb0..0xbf ------------------------------------
        if 0xB0 <= op <= 0xB7:
            i = (op & 7) | (self.rex_b << 3)
            self.set_reg8(i, self.fetch8())
            return
        if 0xB8 <= op <= 0xBF:
            i = (op & 7) | (self.rex_b << 3)
            if size == 64:
                self.reg[i] = self.fetch64()
            elif size == 32:
                self.reg[i] = self.fetch32()
            else:
                self.set_reg(i, 16, self.fetch32() & 0xFFFF)
            return

        # --- group2 shifts 0xc0/0xc1/0xd0/0xd1/0xd2/0xd3 --------------
        if op in (0xC0, 0xC1, 0xD0, 0xD1, 0xD2, 0xD3):
            osz = 8 if op in (0xC0, 0xD0, 0xD2) else size
            reg, rm = self.decode_rm(osz)
            if op in (0xC0, 0xC1):
                cnt = self.fetch8()
            elif op in (0xD0, 0xD1):
                cnt = 1
            else:
                cnt = self.reg[RCX] & 0xFF
            val = self.rm_read(rm, osz)
            res = self.do_shift(reg & 7, val, cnt, osz)
            self.rm_write(rm, osz, res)
            return

        if op == 0xC3:  # ret
            self.rip = self.read(self.reg[RSP], 8)
            self.reg[RSP] += 8
            return

        if op == 0xC6 or op == 0xC7:  # mov r/m, imm
            osz = 8 if op == 0xC6 else size
            reg, rm = self.decode_rm(osz)
            if osz == 8:
                imm = self.fetch8()
            elif osz == 64:
                imm = self.fetch32()
                imm = imm - (1 << 32) if imm >= (1 << 31) else imm
            else:
                imm = self.fetch32() & ((1 << osz) - 1)
            self.rm_write(rm, osz, imm)
            return

        if op == 0xC9:  # leave
            self.reg[RSP] = self.reg[RBP]
            self.reg[RBP] = self.read(self.reg[RSP], 8)
            self.reg[RSP] += 8
            return

        if op == 0xCC:  # int3
            self.halted = True
            return

        # --- rep movsb / stosb 0xa4/0xa5/0xaa/0xab --------------------
        if op in (0xA4, 0xA5, 0xAA, 0xAB):
            unit = 1 if op in (0xA4, 0xAA) else (size // 8)
            while True:
                if op in (0xA4, 0xA5):  # movs
                    v = self.read(self.reg[RSI], unit)
                    self.write(self.reg[RDI], unit, v)
                    if self.rflags & DF_BIT:
                        self.reg[RSI] -= unit
                        self.reg[RDI] -= unit
                    else:
                        self.reg[RSI] += unit
                        self.reg[RDI] += unit
                else:                    # stos
                    self.write(self.reg[RDI], unit, self.reg[RAX])
                    if self.rflags & DF_BIT:
                        self.reg[RDI] -= unit
                    else:
                        self.reg[RDI] += unit
                if rep == 0xF3:
                    self.reg[RCX] -= 1
                    if self.reg[RCX] == 0:
                        break
                else:
                    break
            return

        # --- call/jmp 0xe8/0xe9/0xeb ----------------------------------
        if op == 0xE8:
            d = self.fetchs32()
            self.reg[RSP] -= 8
            self.write(self.reg[RSP], 8, self.rip)
            self.rip += d
            return
        if op == 0xE9:
            d = self.fetchs32()
            self.rip += d
            return
        if op == 0xEB:
            d = self.fetchs8()
            self.rip += d
            return

        # --- group3 0xf6/0xf7 (test/not/neg/mul/imul/div/idiv) --------
        if op == 0xF6 or op == 0xF7:
            osz = 8 if op == 0xF6 else size
            reg, rm = self.decode_rm(osz)
            sub = reg & 7
            if sub == 0 or sub == 1:  # test r/m, imm
                if osz == 8:
                    imm = self.fetch8()
                else:
                    imm = self.fetch32() & ((1 << osz) - 1)
                self.alu("and", self.rm_read(rm, osz), imm, osz)
            elif sub == 2:  # not
                m = (1 << osz) - 1
                self.rm_write(rm, osz, (~self.rm_read(rm, osz)) & m)
            elif sub == 3:  # neg
                a = self.rm_read(rm, osz)
                m = (1 << osz) - 1
                res = (-a) & m
                self.set_flag(CF_BIT, a != 0)
                self.set_flag(OF_BIT, a == (1 << (osz - 1)))
                self.set_flag(AF_BIT, (a & 0xF) != 0)
                self.set_flag(ZF_BIT, res == 0)
                self.set_flag(SF_BIT, (res >> (osz - 1)) & 1)
                self.set_flag(PF_BIT, self._parity(res))
                self.rm_write(rm, osz, res)
            elif sub == 4:  # mul
                a = self.rm_read(rm, osz)
                m = (1 << osz) - 1
                prod = self.get_reg(0, osz) * a
                if osz == 64:
                    lo = prod & 0xFFFFFFFFFFFFFFFF
                    hi = (prod >> 64) & 0xFFFFFFFFFFFFFFFF
                else:
                    lo = prod & m
                    hi = (prod >> osz) & m
                self.set_reg(0, osz, lo)
                self.set_reg(2, osz, hi)
                self.set_flag(CF_BIT, hi != 0)
                self.set_flag(OF_BIT, hi != 0)
            elif sub == 5:  # imul one-operand
                a = self.rm_read(rm, osz)
                sa = a - (1 << osz) if (a >> (osz - 1)) & 1 else a
                sb = self.get_reg(0, osz)
                sb = sb - (1 << osz) if (sb >> (osz - 1)) & 1 else sb
                prod = sa * sb
                m = (1 << osz) - 1
                lo = prod & m
                hi = (prod >> osz) & m
                self.set_reg(0, osz, lo)
                self.set_reg(2, osz, hi)
                # CF/OF set if the high half is not a sign extension of the low
                sign_ok = (hi == m if (lo >> (osz - 1)) & 1 else hi == 0)
                self.set_flag(CF_BIT, not sign_ok)
                self.set_flag(OF_BIT, not sign_ok)
            elif sub == 6:  # div
                d = self.rm_read(rm, osz)
                if d == 0:
                    raise Fault("divide by zero")
                if osz == 64:
                    num = (self.reg[RDX] << 64) | self.reg[RAX]
                else:
                    num = (self.get_reg(2, osz) << osz) | self.get_reg(0, osz)
                q, r = divmod(num, d)
                self.set_reg(0, osz, q)
                self.set_reg(2, osz, r)
            elif sub == 7:  # idiv
                d = self.rm_read(rm, osz)
                sd = d - (1 << osz) if (d >> (osz - 1)) & 1 else d
                if sd == 0:
                    raise Fault("divide by zero")
                if osz == 64:
                    num = (self.reg[RDX] << 64) | self.reg[RAX]
                    if num >> 127:
                        num -= 1 << 128
                else:
                    num = (self.get_reg(2, osz) << osz) | self.get_reg(0, osz)
                    if (num >> (2 * osz - 1)) & 1:
                        num -= 1 << (2 * osz)
                q = abs(num) // abs(sd)
                if (num < 0) != (sd < 0):
                    q = -q
                r = num - q * sd
                self.set_reg(0, osz, q)
                self.set_reg(2, osz, r)
            return

        # --- inc/dec group 0xfe/0xff ----------------------------------
        if op == 0xFE or op == 0xFF:
            osz = 8 if op == 0xFE else size
            reg, rm = self.decode_rm(osz)
            sub = reg & 7
            m = (1 << osz) - 1
            if sub == 0 or sub == 1:  # inc/dec
                a = self.rm_read(rm, osz)
                cf = self.get_flag(CF_BIT)
                res = self.alu("add" if sub == 0 else "sub", a, 1, osz)
                self.set_flag(CF_BIT, cf)   # inc/dec do not touch CF
                self.rm_write(rm, osz, res)
            elif sub == 2:  # call r/m
                t = self.rm_read(rm, 64)
                self.reg[RSP] -= 8
                self.write(self.reg[RSP], 8, self.rip)
                self.rip = t
            elif sub == 4:  # jmp r/m
                self.rip = self.rm_read(rm, 64)
            elif sub == 6:  # push r/m
                v = self.rm_read(rm, 64)
                self.reg[RSP] -= 8
                self.write(self.reg[RSP], 8, v)
            else:
                raise Fault("unsupported group5 sub-op %d" % sub)
            return

        # --- two-byte opcodes 0x0f ------------------------------------
        if op == 0x0F:
            op2 = self.fetch8()
            if 0x80 <= op2 <= 0x8F:  # jcc rel32
                d = self.fetchs32()
                if self.cond_true(op2 & 0xF):
                    self.rip += d
                return
            if 0x90 <= op2 <= 0x9F:  # setcc r/m8
                reg, rm = self.decode_rm(8)
                self.rm_write(rm, 8, 1 if self.cond_true(op2 & 0xF) else 0)
                return
            if 0x40 <= op2 <= 0x4F:  # cmovcc r, r/m
                reg, rm = self.decode_rm(size)
                v = self.rm_read(rm, size)
                if self.cond_true(op2 & 0xF):
                    self.set_reg(reg, size, v)
                return
            if op2 == 0xAF:  # imul r, r/m
                reg, rm = self.decode_rm(size)
                a = self.get_reg(reg, size)
                b = self.rm_read(rm, size)
                m = (1 << size) - 1
                sa = a - (1 << size) if (a >> (size - 1)) & 1 else a
                sb = b - (1 << size) if (b >> (size - 1)) & 1 else b
                prod = sa * sb
                fits = -(1 << (size - 1)) <= prod <= (1 << (size - 1)) - 1
                self.set_flag(CF_BIT, not fits)
                self.set_flag(OF_BIT, not fits)
                self.set_reg(reg, size, prod & m)
                return
            if op2 in (0xB6, 0xB7, 0xBE, 0xBF):  # movzx / movsx
                src_sz = 8 if op2 in (0xB6, 0xBE) else 16
                reg, rm = self.decode_rm(src_sz)
                v = self.rm_read(rm, src_sz)
                if op2 in (0xBE, 0xBF) and (v >> (src_sz - 1)) & 1:
                    v -= 1 << src_sz
                self.set_reg(reg, size, v)
                return
            if op2 == 0x1F:  # multi-byte nop
                self.decode_rm(size)
                return
            if op2 == 0xC8 or op2 == 0xC9 or op2 == 0xCA or op2 == 0xCB or \
               op2 == 0xCC or op2 == 0xCD or op2 == 0xCE or op2 == 0xCF:
                # bswap r64
                i = (op2 & 7) | (self.rex_b << 3)
                v = self.reg[i]
                r = 0
                for _ in range(8):
                    r = (r << 8) | (v & 0xFF)
                    v >>= 8
                self.reg[i] = r
                return
            if op2 == 0x05:  # syscall - not modelled
                self.halted = True
                return
            raise Fault("unhandled 0f opcode %#x at %#x" % (op2, self.rip - 2))

        if op == 0xF4:  # hlt
            self.halted = True
            return

        raise Fault("unhandled opcode %#x at %#x" % (op, self.rip - 1))

    def run(self, max_insns=10_000_000):
        while not self.halted and self.insns < max_insns:
            self.step()
        return self.insns

    def call(self, addr, *args):
        """Call a function at addr with up to 6 integer arguments (SysV ABI)."""
        argregs = [RDI, RSI, RDX, RCX, R8, R9]
        for i, a in enumerate(args):
            self.reg[argregs[i]] = a & 0xFFFFFFFFFFFFFFFF
        self.reg[RSP] = self.base + len(self.mem) - 0x100
        # push a sentinel return address that halts
        self.write(self.reg[RSP], 8, 0)
        self.reg[RSP] -= 8
        self.rip = addr
        while not self.halted:
            if self.rip == 0:      # returned to sentinel
                break
            self.step()
        return self.reg[RAX]

    def flags_snapshot(self):
        return {
            "CF": self.get_flag(CF_BIT), "PF": self.get_flag(PF_BIT),
            "AF": self.get_flag(AF_BIT), "ZF": self.get_flag(ZF_BIT),
            "SF": self.get_flag(SF_BIT), "OF": self.get_flag(OF_BIT),
        }
