#!/usr/bin/python3
import glob
import struct
import os
from enum import Enum
from elftools.elf.elffile import ELFFile

from instruction import Ins, Instruction, Opcode, sign_extend

import ipdb

# reminder cuz dumb:
# char:     1 byte
# int32:    4 bytes
# byte max value: 0xFF

# 0x80000000 is 2147483648 == 2**31

mem = regfile = None
memsize = 1024 * 16  # 8 kB
PC = 32
XLEN = 32  # width of the registers

class Regfile():
    def __init__(self):
        self.registers = [0] * 33

    def __getitem__(self, key):
        return self.registers[key]

    def __setitem__(self, key, value):
        if key == 0: return # a0 register, always is zero]
        self.registers[key] = value & 0xFFFFFFFF

def reset():
    global mem, regfile
    mem = b'\x00' * (memsize)
    # regfile = {i: b'\x00'*4 for i in range(0, 32)}
    regfile = Regfile()
    regfile[PC] = 0x80000000

def r32(addr) -> int:
    global mem
    addr -= 0x80000000
    if addr % 4 != 0:
        raise Exception("no unaligned access")
    assert addr >= 0 and addr <= len(mem) - 4, "out of bounds"
    return struct.unpack("<I", mem[addr:addr+4])[0]

def w32(addr, dat):
    global mem

    if(addr):
        addr -= 0x80000000

    assert addr >= 0 and addr <= len(mem) - 0x4
    if(addr < 0 or addr >= len(mem)):
        raise Exception(f"out of bound exception: can't access {addr}")

    if not isinstance(dat, bytes):
        assert False, 'error?'
        dat = struct.pack("<I", dat)

    d, l = len(dat), len(mem)
    mem = mem[0:addr] + dat + mem[addr + d:]
    assert len(mem) == l

def dump_reg():
    regnames = ['zero', 'ra', 'sp', 'gp', 'tp', 't0', 't1', 't2', 's0', 's1'] + \
        [f'a{i}' for i in range(0, 8)] + \
        [f's{i}' for i in range(2, 12)] + \
        [f't{i}' for i in range(3, 7)]

    print(f'\nPC: {regfile[PC]:08x}', end='')

    h = regfile[PC]
    print(f" (which is {r32(h):08x})")

    for i in range(0, 32):
        if i % 4 == 0:
            print()
        name = regnames[i]
        val = regfile[i]
        print(f"{name}: \t{val:08x}", end='\t')
    print()

def dump_mem():
    for i in range(len(mem) // 4):
        word = struct.unpack("<I", mem[i:i+4])[0]
        if 4*i % 0x80 == 0:
            print(f'\n0x{4*i:04x}:')
        print(f'{word:08x} ', end='\n' if i % 8 == 7 else '')

STOP_AT = os.environ.get('D')

def step() -> bool:
    # instruction fetch
    istr = r32(regfile[PC])
    ins = Instruction.decode(istr)
    imm, rd, rs1, rs2 = ins.imm, ins.rd, ins.rs1, ins.rs2
    funct3, funct7 = ins.funct3, ins.funct7

    if ins is None:
        raise Exception('dont know this instruction', ins)

    #instruction name
    try:
        name = [x for x in Ins if ins == x][0].name
        # print(f'{regfile[PC]:08x}, {name}, {ins}')
    except:
        # print(f'{regfile[PC]:08x}, unknown, {ins}')
        pass

    if str(hex(regfile[PC])) == STOP_AT:
        ipdb.set_trace()

    #### not implemented instructions ####
    if ins.opcode == Opcode.SYSTEM:
        if ins == Ins.ECALL:
            print(f'ECALL: {regfile[PC]:08x}, {regfile[3]}')
            if regfile[3] > 1:
                raise Exception("test fail")
            elif regfile[3] == 1:
                print('pass\t', end = '')
                return False    # test pass
            else:
                pass
        else:
            pass
            # print(f'{name} not implemented')

    elif ins == Ins.FENCE:
        pass

    #### branch instructions ####
    elif ins == Ins.JAL:
            regfile[rd] = regfile[PC] + 4
            regfile[PC] += imm
            # print(f'new pc is {regfile[PC]:08x}')
            # return True
    elif ins == Ins.LUI:
        regfile[rd] = (imm << 12)
        regfile[rd] &= 0xFFFFF000
    elif ins == Ins.AUIPC:
        # regfile[rd] = regfile[PC] + (imm << 12) & 0xFFFFF000
        regfile[PC] = regfile[PC] + (imm << 12)
    elif ins.opcode == Opcode.BRANCH:
        b = False
        if ins == Ins.BEQ:
            b = regfile[rs1] == regfile[rs2]
        elif ins == Ins.BNE:
            b = regfile[rs1] != regfile[rs2]
        elif ins == Ins.BLT:
            b = sign_extend(regfile[rs1],32) < sign_extend(regfile[rs2],32)
        elif ins == Ins.BGE:
            b = sign_extend(regfile[rs1],32) >= sign_extend(regfile[rs2],32)
        elif ins == Ins.BLTU:
            b = regfile[rs1] < regfile[rs2]
        elif ins == Ins.BGEU:
            b = regfile[rs1] >= regfile[rs2]
        else:
            raise Exception('branch not implemented')
        if b:
            regfile[PC] += imm
            return True

    #### immediate instructions ####

    elif ins.opcode == Opcode.IMMEDIATE:
        shamt = imm & 0x1F
        if ins == Ins.ADDI:
            regfile[rd] = imm + regfile[rs1]
        elif ins == Ins.ORI:
            regfile[rd] = imm | regfile[rs1]
        elif ins == Ins.ANDI:
            regfile[rd] = imm & regfile[rs1]
        elif ins == Ins.XORI:
            regfile[rd] = imm ^ regfile[rs1]
        elif ins == Ins.SLLI:
            regfile[rd] = (regfile[rs1] << shamt)
        elif ins == Ins.SRLI:
            regfile[rd] = (regfile[rs1] >> shamt)
        elif ins == Ins.SRAI:
            sign = regfile[rs1] >> 31
            out = regfile[rs1] >> shamt
            out |= (0xFFFFFFFF * sign)<<(32-shamt)
            regfile[rd] = out
        else:
            raise Exception(f"funct3 {funct3:3b} not implemented")

    #### arithmetic instructions ####
    elif ins.opcode == Opcode.ARITH:
        a, b = regfile[rs1], regfile[rs2]
        ret = 0
        if ins == Ins.ADD:
            ret = a + b
        elif ins == Ins.AND:
            ret = a & b
        elif ins == Ins.OR:
            ret = a | b
        elif ins == Ins.XOR:
            ret = a ^ b
        else:
            raise Exception('not implemented!')
        regfile[rd] = ret

    else:
        raise Exception(f"{ins.opcode:7b} not implemented")

    regfile[PC] += 0x4
    return True


if __name__ == "__main__":
    for x in glob.glob('../isa/rv32ui-p-*'):
        if x.endswith(('dump', 'fence_i')):
            continue
        with open(x, 'rb') as f:
            reset()
            e = ELFFile(f)
            for s in e.iter_segments():
                w32(s.header.p_paddr, s.data())

            inscount = 0
            try:
                while step():
                    inscount += 1
            except Exception as e:
                # print('--- EXCEPTION ---')
                print(x, e)
                dump_reg()
                exit(0)
            print(f"{x} end after {inscount} instructions")