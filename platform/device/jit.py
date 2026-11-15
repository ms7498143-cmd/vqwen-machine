#!/usr/bin/env python3
"""
Basic-block JIT for the CPU clone.

The interpreter in cpu.py decodes and dispatches every instruction, which
costs far more than the arithmetic itself. This module translates each basic
block of machine code into a Python function once, then runs that function.
Decoding and dispatch happen at translation time, not at run time.

Correctness is not traded for speed: the generated code performs the same
flag updates as the interpreter, and cpu_jit_verify.py checks that the JIT and
the interpreter agree instruction for instruction and flag for flag. Any block
containing an instruction the translator does not handle falls back to the
interpreter, so unsupported code still runs correctly.

This is a translation layer, not a hardware JIT. It still executes on the host
CPU and remains far slower than native execution.
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpu import CPU, Fault, CF_BIT, PF_BIT, AF_BIT, ZF_BIT, SF_BIT, OF_BIT  # noqa: E402

MASK64 = (1 << 64) - 1

REG_NAME = {
    "rax": 0, "rcx": 1, "rdx": 2, "rbx": 3, "rsp": 4, "rbp": 5, "rsi": 6, "rdi": 7,
    "r8": 8, "r9": 9, "r10": 10, "r11": 11, "r12": 12, "r13": 13, "r14": 14, "r15": 15,
    "eax": 0, "ecx": 1, "edx": 2, "ebx": 3, "esp": 4, "ebp": 5, "esi": 6, "edi": 7,
    "r8d": 8, "r9d": 9, "r10d": 10, "r11d": 11, "r12d": 12, "r13d": 13, "r14d": 14, "r15d": 15,
}


def _sizes(reg):
    if reg.endswith("d"):
        return 32
    if reg.startswith("e"):
        return 32
    if reg.startswith("r") and reg[1:].isdigit():
        return 64
    return 64


class JIT:
    """Translates basic blocks to Python and runs them with interpreter fallback."""

    def __init__(self, cpu=None, code_end=None):
        self.cpu = cpu or CPU()
        self.blocks = {}
        self.translated = 0
        self.fallback_blocks = 0
        # When set, execution stops once the program counter reaches this
        # address (the end of the loaded code).
        self.code_end = code_end

    # ---- block discovery --------------------------------------------------
    def find_block(self, addr):
        """Return (list_of_(offset, size, text), end_addr) for the block at addr."""
        c = self.cpu
        insns = []
        cur = addr
        while True:
            if self.code_end is not None and cur >= self.code_end:
                break
            c.rip = cur
            start = cur
            try:
                c.step()
            except Fault:
                break
            size = c.rip - start
            insns.append((start, size))
            cur = c.rip
            # A block ends at a control-transfer instruction or after a jump.
            last = c.mem[start:cur]
            if self._is_block_end(last):
                break
            if len(insns) > 64:
                break
        return insns, cur

    @staticmethod
    def _is_block_end(code):
        # ret, jmp, jcc (rel8/rel32), call
        if not code:
            return False
        if code[0] in (0xC3, 0xE9, 0xEB):
            return True
        if code[0] == 0xE8:
            return True
        if 0x70 <= code[0] <= 0x7F:
            return True
        if code[0] == 0xFF and len(code) >= 2 and (code[1] & 0x38) in (0x10, 0x20):
            return True
        if code[0] == 0x0F and len(code) >= 2 and 0x80 <= code[1] <= 0x8F:
            return True
        return False

    # ---- translation ------------------------------------------------------
    def translate(self, addr):
        """Build a Python function for the block at addr, or None to fall back.

        Instructions are decoded (not executed) one at a time by _emit, which
        advances the probe RIP to the next instruction. Nothing is executed
        here, so branches in the block cannot move the probe.
        """
        if addr in self.blocks:
            return self.blocks[addr]
        c = self.cpu
        lines = []
        start = addr
        end = addr
        try:
            while True:
                if self.code_end is not None and start >= self.code_end:
                    break
                c.rip = start
                lines.append(self._emit(start))
                end = c.rip
                raw = bytes(c.mem[c._off(start):c._off(end)])
                start = end
                if self._is_block_end(raw):
                    break
                if len(lines) > 64:
                    break
        except NotImplementedError:
            self.fallback_blocks += 1
            self.blocks[addr] = None
            return None

        if not lines:
            self.blocks[addr] = None
            return None

        src = "def blk(reg, mem, flags, read, write):\n"
        src += "    M=(1<<64)-1\n"
        for l in lines:
            src += "    " + l + "\n"
        src += f"    return {end}, None\n"
        ns = {}
        try:
            exec(src, {"__builtins__": {"bin": bin, "_cond": _cond}}, ns)
        except Exception:
            self.fallback_blocks += 1
            self.blocks[addr] = None
            return None
        self.translated += 1
        self.blocks[addr] = ns["blk"]
        return ns["blk"]

    def _emit(self, start):
        """Decode and translate one instruction to a Python source line."""
        c = self.cpu
        c.rip = start
        rex = 0
        opsize = 32
        while True:
            b = c.mem[c._off(c.rip)]
            if b == 0x66:
                opsize = 16
                c.rip += 1
            elif 0x40 <= b <= 0x4F:
                rex = b
                c.rip += 1
            elif b in (0xF2, 0xF3, 0x2E, 0x36, 0x3E, 0x26, 0x64, 0x65, 0xF0):
                c.rip += 1
            else:
                break
        if rex & 8:
            opsize = 64
        c.rex = rex
        c.rex_r = (rex >> 2) & 1
        c.rex_x = (rex >> 1) & 1
        c.rex_b = rex & 1
        op = c.fetch8()

        def reg_rm(osz):
            return c.decode_rm(osz)

        # mov r/m64, r64  (0x89)
        if op == 0x89 and opsize == 64:
            reg, rm = reg_rm(64)
            if rm[0] == "reg":
                return f"reg[{rm[1]}]=reg[{reg}]&M"
            return f"write({rm[1]},8,reg[{reg}])"
        # mov r64, r/m64 (0x8b)
        if op == 0x8B and opsize == 64:
            reg, rm = reg_rm(64)
            if rm[0] == "reg":
                return f"reg[{reg}]=reg[{rm[1]}]&M"
            return f"reg[{reg}]=read({rm[1]},8)&M"
        # mov r64, imm64 (0xB8+r)
        if 0xB8 <= op <= 0xBF and opsize == 64:
            i = (op & 7) | (c.rex_b << 3)
            imm = c.fetch64()
            return f"reg[{i}]={imm}"
        # add/sub/and/or/xor r/m64, r64  (destination is the r/m operand)
        if op in (0x01, 0x29, 0x21, 0x09, 0x31) and opsize == 64:
            reg, rm = reg_rm(64)
            if rm[0] != "reg":
                raise NotImplementedError
            d = rm[1]      # r/m is the destination
            s = reg        # reg field is the source
            if op == 0x01:
                return self._emit_add(d, s, dst_is_rm=True)
            if op == 0x29:
                return self._emit_sub(d, s, dst_is_rm=True)
            if op == 0x21:
                return self._emit_logic("&", d, s, dst_is_rm=True)
            if op == 0x09:
                return self._emit_logic("|", d, s, dst_is_rm=True)
            if op == 0x31:
                return self._emit_logic("^", d, s, dst_is_rm=True)
        # lea r64, m (0x8D)
        if op == 0x8D and opsize == 64:
            reg, rm = reg_rm(64)
            if rm[0] != "mem":
                raise NotImplementedError
            return f"reg[{reg}]={rm[1]}&M"
        # inc/dec r64 (0xFF /0 /1)
        if op == 0xFF and opsize == 64:
            reg, rm = reg_rm(64)
            if rm[0] != "reg":
                raise NotImplementedError
            i = rm[1]
            sub = reg & 7
            if sub == 0:
                return (f"f=reg[{i}]; reg[{i}]=(f+1)&M; "
                        f"flags[1]=(bin(reg[{i}]&0xff).count('1')%2==0); "
                        f"flags[2]=(reg[{i}]==0); flags[3]=(reg[{i}]>>63)&1; "
                        f"flags[4]=((f^1^reg[{i}])>>4)&1; flags[5]=((f==0x7fffffffffffffff)&1)")
            if sub == 1:
                return (f"f=reg[{i}]; reg[{i}]=(f-1)&M; "
                        f"flags[1]=(bin(reg[{i}]&0xff).count('1')%2==0); "
                        f"flags[2]=(reg[{i}]==0); flags[3]=(reg[{i}]>>63)&1; "
                        f"flags[4]=((f^1^reg[{i}])>>4)&1; flags[5]=((f==0x8000000000000000)&1)")
        # cmp r/m64, r64 (0x39)
        if op == 0x39 and opsize == 64:
            reg, rm = reg_rm(64)
            if rm[0] != "reg":
                raise NotImplementedError
            return self._emit_cmp(rm[1], reg)
        # test r/m64, r64 (0x85)
        if op == 0x85 and opsize == 64:
            reg, rm = reg_rm(64)
            if rm[0] != "reg":
                raise NotImplementedError
            return self._emit_logic("&", rm[1], reg, dst_is_rm=True, write_back=False)
        # imul r64, r/m64 (0x0F 0xAF)
        if op == 0x0F:
            op2 = c.fetch8()
            if op2 == 0xAF and opsize == 64:
                reg, rm = reg_rm(64)
                if rm[0] != "reg":
                    raise NotImplementedError
                return self._emit_imul(reg, rm[1])
            if 0x80 <= op2 <= 0x8F:
                d = c.fetchs32()
                target = c.rip + d
                return f"return ({target}, None) if _cond(flags,{op2 & 0xF}) else ({c.rip}, None)"
            raise NotImplementedError("0f opcode %#x" % op2)
        # jcc rel8 / rel32
        if 0x70 <= op <= 0x7F:
            d = c.fetchs8()
            target = c.rip + d
            return f"return ({target}, None) if _cond(flags,{op & 0xF}) else ({c.rip}, None)"
        raise NotImplementedError("opcode %#x" % op)

    def _emit_cmp(self, d, s):
        return (f"a=reg[{d}]; b=reg[{s}]; t=a-b; "
                f"flags[0]=1 if t<0 else 0; "
                f"flags[2]=((t&M)==0); flags[3]=((t&M)>>63)&1; "
                f"flags[4]=((a^b^(t&M))>>4)&1; "
                f"flags[5]=(((a^b)&(a^(t&M))&0x8000000000000000)!=0); "
                f"flags[1]=(bin(t&M&0xff).count('1')%2==0)")

    def _emit_imul(self, d, s):
        # Signed 64-bit multiply; CF and OF are set when the true product does
        # not fit in 64 bits.
        return (f"a=reg[{d}]; b=reg[{s}]; "
                f"sa=a-(1<<64) if a>>63 else a; sb=b-(1<<64) if b>>63 else b; "
                f"p=sa*sb; reg[{d}]=p&M; "
                f"fits=(-(1<<63)<=p<=(1<<63)-1); "
                f"flags[0]=0 if fits else 1; flags[5]=0 if fits else 1")

    def _emit_add(self, d, s, dst_is_rm):
        # dst = dst + src, with CF/OF/AF/ZF/SF/PF
        return (f"a=reg[{d}]; b=reg[{s}]; t=a+b; "
                f"flags[0]=1 if t>M else 0; "
                f"flags[2]=(t&M==0); flags[3]=(t>>63)&1; "
                f"flags[4]=((a^b^(t&M))>>4)&1; "
                f"flags[5]=((~(a^b)&(a^(t&M))&0x8000000000000000)!=0); "
                f"flags[1]=(bin(t&M&0xff).count('1')%2==0); reg[{d}]=t&M")

    def _emit_sub(self, d, s, dst_is_rm):
        return (f"a=reg[{d}]; b=reg[{s}]; t=a-b; "
                f"flags[0]=1 if t<0 else 0; "
                f"flags[2]=((t&M)==0); flags[3]=((t&M)>>63)&1; "
                f"flags[4]=((a^b^(t&M))>>4)&1; "
                f"flags[5]=(((a^b)&(a^(t&M))&0x8000000000000000)!=0); "
                f"flags[1]=(bin(t&M&0xff).count('1')%2==0); reg[{d}]=t&M")

    def _emit_logic(self, sym, d, s, dst_is_rm, write_back=True):
        wb = f"reg[{d}]=r&M; " if write_back else ""
        return (f"r=reg[{d}]{sym}reg[{s}]; " + wb +
                f"flags[0]=0; flags[5]=0; flags[4]=0; flags[2]=(r&M==0); "
                f"flags[3]=(r>>63)&1; flags[1]=(bin(r&M&0xff).count('1')%2==0)")

    # ---- execution --------------------------------------------------------
    def run(self, max_blocks=100000):
        c = self.cpu
        # Flag array: 0=CF 1=PF 2=ZF 3=SF 4=AF 5=OF. Keeping the flags in this
        # list across blocks avoids six get_flag/set_flag calls per block, which
        # otherwise cost as much as the translation saves.
        flags = [c.get_flag(CF_BIT), c.get_flag(PF_BIT), c.get_flag(ZF_BIT),
                 c.get_flag(SF_BIT), c.get_flag(AF_BIT), c.get_flag(OF_BIT)]
        c.rflags |= 0x2
        reg, mem, read, write = c.reg, c.mem, c.read, c.write
        blocks = self.blocks
        cache = {}
        nblocks = 0
        try:
            while not c.halted and c.rip != 0 and nblocks < max_blocks:
                if self.code_end is not None and c.rip >= self.code_end:
                    break
                nblocks += 1
                entry = c.rip
                fn = cache.get(entry)
                if fn is None:
                    fn = self.translate(entry)
                    cache[entry] = fn if fn is not None else False
                if fn is False or fn is None:
                    c.rip = entry
                    # The flags list is authoritative while the JIT runs, so
                    # push it into c.rflags before the interpreter executes an
                    # instruction, then pull the result back afterwards.
                    self._sync_out(c, flags)
                    c.step()
                    self._sync_in(c, flags)
                    continue
                nxt, _ = fn(reg, mem, flags, read, write)
                c.rip = nxt
        finally:
            self._sync_out(c, flags)
        return nblocks

    @staticmethod
    def _sync_out(c, flags):
        c.set_flag(CF_BIT, flags[0]); c.set_flag(PF_BIT, flags[1])
        c.set_flag(ZF_BIT, flags[2]); c.set_flag(SF_BIT, flags[3])
        c.set_flag(AF_BIT, flags[4]); c.set_flag(OF_BIT, flags[5])

    @staticmethod
    def _sync_in(c, flags):
        flags[0] = c.get_flag(CF_BIT); flags[1] = c.get_flag(PF_BIT)
        flags[2] = c.get_flag(ZF_BIT); flags[3] = c.get_flag(SF_BIT)
        flags[4] = c.get_flag(AF_BIT); flags[5] = c.get_flag(OF_BIT)


def _cond(flags, cc):
    cf, pf, zf, sf, af, of = (flags[0], flags[1], flags[2], flags[3], flags[4], flags[5])
    return [
        bool(of), not of, bool(cf), not cf, bool(zf), not zf,
        bool(cf) or bool(zf), not (cf or zf), bool(sf), not sf,
        bool(pf), not pf, bool(sf) != bool(of), bool(sf) == bool(of),
        bool(zf) or (bool(sf) != bool(of)),
        not (bool(zf) or (bool(sf) != bool(of))),
    ][cc]
