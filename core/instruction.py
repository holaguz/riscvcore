#!/usr/bin/env python3
from enum import Enum
class Instruction():
    class Type(Enum):
        R = 0
        I = 1
        S = 2
        B = 3
        U = 4
        J = 5

    def gb(r: int, l: int, x: int):
        # assert x < 0x80000000
        # assert l <= r and l >= 0 and r <= 31
        x = x >> l
        x = x & int((2 ** (r - l + 1) - 1))
        return x


    def __init__(self, opcode, type: Type = None, funct3 = None, funct7 = None, rd = None, rs1 = None, rs2 = None, imm = None):
        self.type   = type
        self.opcode = opcode
        self.funct3 = funct3
        self.funct7 = funct7
        self.imm    = imm
        self.rs1 = rs1
        self.rs2 = rs2
        self.rd = rd

    def __repr__(self) -> str:
        return str(self.__dict__)

    def decode(istring: int) -> 'Instruction':
        gb = Instruction.gb

        opcode = gb(6, 0, istring)
        ins = [x for x in ins_lup if x.opcode == opcode]
        print(ins)
        if len(ins) == 0:
            raise Exception("not found")
        ins = ins[0]

        ret     = None
        rd      = gb(11, 7, istring)
        rs1     = gb(19, 15, istring)
        rs2     = gb(24, 20, istring)
        funct3  = gb(12, 14, istring)
        funct7  = gb(31, 25, istring)

        if ins.type == Instruction.Type.J:
            imm = gb(31, 31, istring) << 19 | gb(30, 21, istring) << 1 | gb(20, 20, istring) << 10 | gb(19, 12, istring) << 11
            ret = Instruction(opcode = opcode, imm = imm, rd = rd)
        elif ins.type == Instruction.Type.U:
            imm = gb(31, 12, istring) << 31
            ret = Instruction(opcode = opcode, imm = imm, rd = rd)
        elif ins.type == Instruction.Type.B:
            imm = gb(7, 7, istring) << 11 | gb(11, 8, istring) << 1 | gb(25, 30, istring) << 5 | gb(31, 31, istring) << 12
            ret = Instruction(opcode=opcode, imm=imm, rs1=rs1, rs2=rs2, funct3=funct3)
        elif ins.type == Instruction.Type.S:
            imm = gb(4, 0, istring) | gb(31, 25) << 5
            ret = Instruction(opcode=opcode, imm=imm, rs1=rs1, rs2=rs2, funct3=funct3)
        else:
            print("Not Implemented")
            return None

        return ret

LUI     = Instruction(opcode = 0b0110111,   type = Instruction.Type.U)
AUIPC   = Instruction(opcode = 0b0010111,   type = Instruction.Type.U)
JAL     = Instruction(opcode = 0b1101111,   type = Instruction.Type.J)
JALR    = Instruction(opcode = 0b1100111,   type = Instruction.Type.I, funct3 = 0b000)

BEQ     = Instruction(opcode = 0b1100011,   type = Instruction.Type.B, funct3 = 0b000)
BNE     = Instruction(opcode = 0b1100011,   type = Instruction.Type.B, funct3 = 0b001)
BLT     = Instruction(opcode = 0b1100011,   type = Instruction.Type.B, funct3 = 0b100)
BGE     = Instruction(opcode = 0b1100011,   type = Instruction.Type.B, funct3 = 0b101)
BLTU    = Instruction(opcode = 0b1100011,   type = Instruction.Type.B, funct3 = 0b110)
BGEU    = Instruction(opcode = 0b1100011,   type = Instruction.Type.B, funct3 = 0b111)

ins_lup = [LUI, AUIPC, JAL]


if __name__ == '__main__':
    instructions = {
        0x800002b7: LUI,
        0x00000297: AUIPC
    }

    for k, v in instructions.items():
        if Instruction.decode(k).opcode == v.opcode:
            print(f"{v} ok")
        else:
            print(f"{v} fail")
            break
