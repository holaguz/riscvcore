#!/usr/bin/python3
from base64 import decode
import struct
from enum import Enum
import glob
from elftools.elf.elffile import ELFFile

from instruction import Instruction

# reminder cuz dumb:
# char:     1 byte
# int32:    4 bytes
# byte max value: 0xFF

# 0x80000000 is 2147483648 == 2**31

mem = regfile = None
memsize = 1024 * 8 # 8 kB
PC = 32
XLEN = 32 # width of the registers

class Op(Enum):
    JAL         = 0b1101111
    IMMEDIATE   = 0b0010011
    ECALL       = 0b1110011
    BRANCH      = 0b1100011
    LUI         = 0b0110111 # load upper immediate
    AUIPC       = 0b0010111 # add upper immediate to PC

class Funct3(Enum):
    # immediate
    ADDI    = 0b000
    SLTI    = 0b010
    SLTIU   = 0b011
    XORI    = 0b100
    ORI     = 0b110
    ANDI    = 0b111

    SLLI    = 0b001
    SRLI    = 0b101
    # SRAI    = 0b001

    # branch
    BNE     = 0b001

class Funct7(Enum):

    SLLI    = 0b0000000
    SRLI    = 0b0000000
    SRAI    = 0b0100000

    ADD     = 0b0000000

class Type(Enum):
    R = 0
    I = 1
    S = 2
    B = 3
    U = 4
    J = 5

def TypeOf(opcode: Op):
    if opcode in [Op.JAL]:
        return Type.J

    if opcode in [Op.IMMEDIATE]:
        return Type.I

    if opcode in [Op.BRANCH]:
        return Type.B

    if opcode in [Op.AUIPC, Op.LUI]:
        print(f'{opcode} is type U')
        return Type.U

    return NotImplementedError

def decode_ins(x, type: Type):
    assert x >= 0 and x <= 2 ** (XLEN) - 1

    opcode  = gb(6, 0, x)

    if type == Type.J:
        rd      = gb(11, 7, x)
        imm     = gb(31, 31, x) << 19 | gb(30, 21, x) << 1 | gb(20, 20, x) << 10 | gb(19, 12, x) << 11
        return opcode, rd, imm
    if type == Type.I:
        rd      = gb(11, 7, x)
        funct3  = gb(14, 12, x)
        rs1     = gb(19, 15, x)
        imm     = gb(31, 20, x)
        return opcode, rd, funct3, rs1, imm
    if type == Type.B:
        funct3  = gb(14, 12, x)
        rs1     = gb(19, 15, x)
        rs2     = gb(24, 20, x)
        imm     = gb(31, 31, x) << 12 | gb(30, 25, x) << 10 | gb(11, 8, x) << 4 | gb(7, 7, x) << 11
        return opcode, funct3, rs1, rs2, imm
    if type == Type.U:
        rd      = gb(11, 7, x)
        imm     = gb(31, 12, x)
        return opcode, rd, imm
    if type == Type.R:
        rd      = gb(11, 7, x)
        funct3  = gb(12, 14, x)
        rs1     = gb(19, 15, x)
        rs2     = gb(24, 20, x)
        funct7  = gb(31, 25, x)
        return opcode, rd, funct3, rs1, rs2, funct7
    print(f"can't decode {type} -- not implemented")
    assert False

def reset():
    global mem, regfile
    mem = b'\x00' * (memsize)
    # regfile = {i: b'\x00'*4 for i in range(0, 32)}
    regfile = {i: 0 for i in range(0, 32)}
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
    print(f'\nPC: {regfile[PC]:04x}', end = '')
    if len(mem) < regfile[PC]:
        h = regfile[PC]
        print(f" (which is {r32(h):08x})")
    else:
        print()

    for i in range(0, 32):
        if i % 4 == 0: print()
        regname, val = f'x{i}', regfile[i]
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

def gb(r:int, l: int, x: int):
    # assert x < 0x80000000
    # assert l <= r and l >= 0 and r <= 31
    x = x >> l
    x = x & (2**(r - l + 1) - 1)
    return x


u = 0b00110111
assert gb(0, 0, u) == 0b1
assert gb(1, 0, u) == 0b11
assert gb(2, 0, u) == 0b111

assert gb(7, 0, u) == u
assert gb(7, 4, u) == 0b0011
assert gb(6, 3, u) == 0b110

def step() -> bool:
    try:
        # instruction fetch
        ins = r32(regfile[PC])
        opcode = gb(6, 0, ins)
        print(f'Instruction is {ins:08x}, opcode is {opcode:07b}')
        opcode = Op(opcode)
        type = TypeOf(opcode)

        if opcode == Op.ECALL:
            print("ECALL not implemented, continuing")
            regfile[PC] += 0x4
            return True

        if type == Type.J:
            _, rd, imm = decode_ins(ins, Type.J)
            # imm = sign_extend(imm, 32)
            if opcode == Op.JAL:
                if rd == 0:
                    regfile[PC] += imm
            else:
                raise Exception('dont know {opcode:07b}')

        elif type == Type.I:
            _, rd, funct3, rs1, imm = decode_ins(ins, Type.I)
            funct3 = Funct3(funct3)
            print(opcode, rd, funct3, rs1, imm)
            if opcode == Op.IMMEDIATE:
                if funct3 == Funct3.ADDI:
                    regfile[rs1] = regfile[rs1] + sign_extend(imm, XLEN)
                else:
                    raise Exception(f"dont know funct3 {funct3}")
            else:
                raise Exception('dont know {opcode:07b}')

        elif type == Type.B:
            _, funct3, rs1, rs2, imm = decode_ins(ins, Type.B)
            print(opcode, funct3, rs1, rs2, imm)
            funct3 = Funct3(funct3)
            if funct3 == Funct3.BNE:
                if regfile[rs1] == regfile[rs2]:
                    regfile[PC] += imm
            else:
                raise Exception('dont know {opcode:07b}')

        elif type == Type.U:
            _, rd, imm = decode_ins(ins, Type.U)

            if opcode == Op.AUIPC:
                regfile[rd] = imm << 12 + regfile[PC]

            elif opcode == Op.LUI:
                val = imm << 12 & 0xFFFFFFFF
                print('val')
                regfile[rd] 

            else:
                raise Exception('dont know {opcode:07b}')

        else:
            raise Exception("type not implemented")

    except Exception as e:
        print(e)
        dump_reg()
        return False

    regfile[PC] += 0x4
    return True

if __name__ == "__main__":
    for x in glob.glob('../isa/rv32ui-p-add'):
        print(x)
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
            print(f"{x} end after {inscount} instructions")
        break