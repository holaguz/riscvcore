#!/usr/bin/python3
from base64 import decode
import struct
from enum import Enum
import glob
import binascii
from elftools.elf.elffile import ELFFile

# reminder cuz dumb:
# char:     1 byte
# int32:    4 bytes
# byte max value: 0xFF

# 0x80000000 is 2147483648 == 2**31

mem = regfile = None
memsize = 1024 * 8 # 8 kB
PC = 32

class Op(Enum):
    JAL = 0b1101111,
    ADDI = 0b0010011,
    SLTI = 0b0010011
class Type(Enum):
    R = 0,
    I = 1,
    S = 2,
    B = 3,
    U = 4, 
    J = 5

def TypeOf(opcode: Op):
    if opcode in [Op.JAL]: # J Type
        return Type.J
    else if opcode in [OP.LDI]


def decode_ins(x, type: Type):
    assert x <= 0x80000000 and x >= 0

    if type == Type.J:
        opcode  = get_bits(6, 0, x)
        rd      = get_bits(11, 7, x)
        imm = get_bits(31, 31, x) << 19 | get_bits(30, 21, x) << 1 | get_bits(20, 20, x) << 10 | get_bits(19, 12, x) << 11

        return opcode, rd, imm
    else:
        assert False


def reset():
    global mem, regfile
    mem = b'\x00' * (memsize)
    regfile = {f'x{i}': b'\x00'*4 for i in range(0, 32)}
    regfile[PC] = 0x80000000

def r32(addr):
    global mem
    addr -= 0x80000000
    if addr % 4 != 0:
        raise Exception("no unaligned access")
    assert addr >= 0 and addr <= len(mem) - 4, "out of bounds"
    return struct.unpack("<I", mem[addr:addr+4])[0]

def w32(addr, dat):
    global mem
    if addr >= 0x80000000:
        addr -= 0x80000000
    if(addr < 0 or addr >= len(mem)):
        raise Exception(f"out of bound exception: can't access {addr}")

    if not isinstance(dat, bytes):
        assert False, 'error?'
        dat = struct.pack("<I", dat)

    d, l = len(dat), len(mem)
    mem = mem[0:addr] + dat + mem[addr + d:]
    assert(len(mem) == l)

def dump_reg():
    rname = []
    rname.append('PC')
    rname = [f'x{i}' for i in range(0, 32)]

    print(f'PC: {regfile[PC]:04x}', end = '')
    h = regfile[PC]
    if len(mem) < regfile[PC]:
        print(f" (which is {r32(h):08x})")
    else:
        print()

    for i in range(0, 32):
        regname = f'x{i}'
        val = struct.unpack("<I", regfile[f'x{i}'])[0]
        if i % 4 == 0: print()
        print(f"{regname}: \t{val:08x}", end = '\t')


def dump_mem():
    for i in range(len(mem) // 4):
        word = struct.unpack("<I", mem[i:i+4])[0]
        if 4*i % 0x80 == 0: print(f'\n0x{4*i:04x}:')
        print(f'{word:08x} ', end = '\n' if i%8 == 7 else '')

def sign_extend(x, l):
  if x >> (l-1) == 1:
    return -((1 << l) - x)
  else:
    return x

def get_bits(r:int, l: int, x: int):
    assert x < 0x80000000
    assert l <= r and l >= 0 and r <= 31
    x = x >> l
    x = x & (2**(r - l + 1) - 1)
    return x

u = 0b00110111
assert get_bits(0, 0, u) == 0b1
assert get_bits(1, 0, u) == 0b11
assert get_bits(2, 0, u) == 0b111

assert get_bits(7, 0, u) == u
assert get_bits(7, 4, u) == 0b0011
assert get_bits(6, 3, u) == 0b110


def step() -> bool:
    ins = 0
    opcode = 0
    try:
        ins = r32(regfile[PC])
        opcode = get_bits(6, 0, ins)
        opcode = Op(opcode)
        if TypeOf(opcode) == Type.J:
            _, rd, imm = decode_ins(ins, Type.J)
            imm = sign_extend(imm, 32)
            print(f'{ins:08x}, {imm:08x}, {rd}, {opcode}')
            if opcode == Op.JAL:
                if rd == 0:
                    regfile[PC] += imm
                    print(f'jumping to {regfile[PC]:08x}')
            else:
                return False
        else:
            print(f'optype not implemented')
            return False
    except Exception:
        print(f"instruction {opcode:07b} not implemented")
        dump_reg()
        return False
    return True

if __name__ == "__main__":
    for x in glob.glob('../riscv-tests/isa/rv32ui-p-add'):
        if x.endswith('dump'):
            continue
        print(x)
        with open(x, 'rb') as f:
            reset()
            e = ELFFile(f)
            for s in e.iter_segments():
                w32(s.header.p_paddr, s.data())

            inscount = 0
            while step():
                inscount += 1
            print("  end after %d instructions" % inscount)

        break
