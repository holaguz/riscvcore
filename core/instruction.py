#!/usr/bin/env python3
from enum import Enum, IntEnum

def sign_extend(x, l):
    """
    x is the bits to sign extend\n
    l is the bit that contains the sign
    """
    if x >> (l-1) == 1:
        return -((1 << l) - x)
    else:
        return x

class Instruction():
    class Type(IntEnum):
        R = 1
        I = 2
        S = 3
        B=4
        U=5
        J=6
        WEIRDI=7

    def gb(l: int, r: int, x: int):
        assert l >= r and r >= 0 and l <= 31, f'assertion failed in gb: l:{l} r:{r}'
        x = x >> r
        x = x & int((2 ** (l - r + 1) - 1))
        return x


    def __init__(self, opcode, type: Type = None, funct3 = None, funct7 = None, rd = None, rs1 = None, rs2 = None, imm = None):
        self.type   = type
        self.opcode = opcode
        self.funct7 = funct7
        self.funct3 = funct3
        self.imm    = imm
        self.rs1    = rs1
        self.rs2    = rs2
        self.rd     = rd

        assert funct3 == None               if type in [self.Type.U, self.Type.J] else True, '123123'
        assert rs1 == None and rs2 == None  if type in [self.Type.U, self.Type.J] else True, '123123'

    def __repr__(self) -> str:
        return str(
            [f'{k}: {v%(1<<32):07b}' for k, v in self.__dict__.items() if v is not None]
        )

    def __eq__(self, other):
        if not isinstance(other, Instruction):
            other = other.value
            assert type(other) is Instruction
        return self.opcode == other.opcode and self.funct3 == other.funct3 and self.funct7 == other.funct7

    def decode(istring: int) -> 'Instruction':
        assert istring <= (1<<32) - 1, 'decode out of bounds'
        gb = Instruction.gb

        ret     = None
        opcode  = gb(6, 0, istring)
        rd      = gb(11, 7, istring)
        rs1     = gb(19, 15, istring)
        rs2     = gb(24, 20, istring)
        funct3  = gb(14, 12, istring)
        funct7  = gb(31, 25, istring)

        ins = None
        for i in Ins:
            name, istr = i.name, i.value
            if opcode == istr.opcode.value:
                if opcode != Opcode.IMMEDIATE:
                    ins = i.value
                    break
                elif opcode == Opcode.IMMEDIATE:
                    # match funct3
                    if funct3 == istr.funct3:
                        if funct3 == any([0b101, 0b001]):
                            if funct7 == istr.funct7:
                                ins = i.value
                                break
                        else:
                            ins = i.value
                            break

        if ins is None:
            raise Exception(f"opcode {opcode:07b} not found\n instruction was {istring:b}")

        if ins.type == Instruction.Type.J:
            imm = gb(31, 31, istring) << 20 | gb(30, 21, istring) << 1 | gb(20, 20, istring) << 11 | gb(19, 12, istring) << 12
            imm = sign_extend(imm, 20)
            ret = Instruction(opcode = opcode, imm = imm, rd = rd)

        elif ins.type == Instruction.Type.U:
            imm = gb(31, 12, istring)# << 12
            imm = sign_extend(imm, 20)
            ret = Instruction(opcode = opcode, imm = imm, rd = rd)

        elif ins.type == Instruction.Type.B:
            imm = gb(31, 31, istring) << 12 | gb(30, 25, istring) << 5 | gb(11,8,istring)<<1 | gb(7,7,istring)<<11
            imm = sign_extend(imm, 13)
            ret = Instruction(opcode=opcode, imm=imm, rs1=rs1, rs2=rs2, funct3=funct3)

        elif ins.type == Instruction.Type.S:
            imm = sign_extend(gb(11, 7, istring) | gb(31, 25, istring) << 5, 12)
            ret = Instruction(opcode=opcode, imm=imm, rs1=rs1, rs2=rs2, funct3=funct3)

        elif ins.type == Instruction.Type.I:
            imm = sign_extend(gb(31, 20, istring), 12)
            ret = Instruction(opcode=opcode, rd=rd, funct3=funct3, rs1=rs1, imm=imm)

        elif ins.type == Instruction.Type.WEIRDI:
            imm = sign_extend(gb(31, 20, istring), 12)
            ret = Instruction(opcode=opcode, rd=rd, funct3=funct3, funct7=funct7, rs1=rs1, imm=imm)

        elif ins.type == Instruction.Type.R:
            ret = Instruction(opcode=opcode, rd=rd, funct3=funct3, rs1=rs1, rs2=rs2, funct7=funct7)

        else:
            print(f"Type.{ins.type.name} not implemented")
            return None

        return ret

class Opcode(IntEnum):
    JAL         = 0b1101111
    JALR        = 0b1100111
    IMMEDIATE   = 0b0010011
    BRANCH      = 0b1100011
    LUI         = 0b0110111 # load upper immediate
    AUIPC       = 0b0010111 # add upper immediate to PC

    LOAD        = 0b0000011
    STORE       = 0b0100011

    ARITH       = 0b0110011
    SYSTEM      = 0b1110011 # control status register
    FENCE       = 0b0001111

class Ins(Enum):

    # U type
    LUI     = Instruction(opcode = Opcode.LUI,      type = Instruction.Type.U)
    AUIPC   = Instruction(opcode = Opcode.AUIPC,    type = Instruction.Type.U)

    # J type
    JAL     = Instruction(opcode = Opcode.JAL,      type = Instruction.Type.J)
    JALR    = Instruction(opcode = Opcode.JALR,     type = Instruction.Type.I, funct3 = 0b000)

    BEQ     = Instruction(opcode = Opcode.BRANCH,   type = Instruction.Type.B, funct3 = 0b000)
    BNE     = Instruction(opcode = Opcode.BRANCH,   type = Instruction.Type.B, funct3 = 0b001)
    BLT     = Instruction(opcode = Opcode.BRANCH,   type = Instruction.Type.B, funct3 = 0b100)
    BGE     = Instruction(opcode = Opcode.BRANCH,   type = Instruction.Type.B, funct3 = 0b101)
    BLTU    = Instruction(opcode = Opcode.BRANCH,   type = Instruction.Type.B, funct3 = 0b110)
    BGEU    = Instruction(opcode = Opcode.BRANCH,   type = Instruction.Type.B, funct3 = 0b111)

    # I type
    ADDI    = Instruction(opcode = Opcode.IMMEDIATE, type = Instruction.Type.I, funct3 = 0b000)
    SLTI    = Instruction(opcode = Opcode.IMMEDIATE, type = Instruction.Type.I, funct3 = 0b010)
    SLTIU   = Instruction(opcode = Opcode.IMMEDIATE, type = Instruction.Type.I, funct3 = 0b011)
    XORI    = Instruction(opcode = Opcode.IMMEDIATE, type = Instruction.Type.I, funct3 = 0b100)
    ORI     = Instruction(opcode = Opcode.IMMEDIATE, type = Instruction.Type.I, funct3 = 0b110)
    ANDI    = Instruction(opcode = Opcode.IMMEDIATE, type = Instruction.Type.I, funct3 = 0b111)

    LB      = Instruction(opcode = Opcode.LOAD, type = Instruction.Type.I, funct3 = 0b000)
    LH      = Instruction(opcode = Opcode.LOAD, type = Instruction.Type.I, funct3 = 0b001)
    LW      = Instruction(opcode = Opcode.LOAD, type = Instruction.Type.I, funct3 = 0b010)
    LBU     = Instruction(opcode = Opcode.LOAD, type = Instruction.Type.I, funct3 = 0b100)
    LHU     = Instruction(opcode = Opcode.LOAD, type = Instruction.Type.I, funct3 = 0b101)

    # Weird I
    SLLI    = Instruction(opcode = Opcode.IMMEDIATE, type = Instruction.Type.WEIRDI, funct3 = 0b001, funct7=0b0000000)
    SRLI    = Instruction(opcode = Opcode.IMMEDIATE, type = Instruction.Type.WEIRDI, funct3 = 0b101, funct7=0b0000000)
    SRAI    = Instruction(opcode = Opcode.IMMEDIATE, type = Instruction.Type.WEIRDI, funct3 = 0b101, funct7=0b0100000)

    # system, control status registers
    FENCE   = Instruction(opcode = Opcode.FENCE,  type=Instruction.Type.I, funct3 = 0b000)
    ECALL   = Instruction(opcode = Opcode.SYSTEM, type=Instruction.Type.I, funct3 = 0b000)
    CSRRW   = Instruction(opcode = Opcode.SYSTEM, type=Instruction.Type.I, funct3 = 0b001)
    CSRRS   = Instruction(opcode = Opcode.SYSTEM, type=Instruction.Type.I, funct3 = 0b010)
    CSRRC   = Instruction(opcode = Opcode.SYSTEM, type=Instruction.Type.I, funct3 = 0b011)
    CSRRWI  = Instruction(opcode = Opcode.SYSTEM, type=Instruction.Type.I, funct3 = 0b101)
    CSRRSI  = Instruction(opcode = Opcode.SYSTEM, type=Instruction.Type.I, funct3 = 0b110)
    CSRRCI  = Instruction(opcode = Opcode.SYSTEM, type=Instruction.Type.I, funct3 = 0b111)

    # S Type
    SB = Instruction(opcode=Opcode.STORE, type=Instruction.Type.S, funct3=0b000)
    SH = Instruction(opcode=Opcode.STORE, type=Instruction.Type.S, funct3=0b001)
    SW = Instruction(opcode=Opcode.STORE, type=Instruction.Type.S, funct3=0b010)

    # R Type
    ADD     = Instruction(opcode = Opcode.ARITH, type=Instruction.Type.R, funct3 = 0b000, funct7=0b0000000)
    SUB     = Instruction(opcode = Opcode.ARITH, type=Instruction.Type.R, funct3 = 0b000, funct7=0b0100000)
    SLL     = Instruction(opcode = Opcode.ARITH, type=Instruction.Type.R, funct3 = 0b001, funct7=0b0000000)
    SLT     = Instruction(opcode = Opcode.ARITH, type=Instruction.Type.R, funct3 = 0b010, funct7=0b0000000)
    SLTU    = Instruction(opcode = Opcode.ARITH, type=Instruction.Type.R, funct3 = 0b011, funct7=0b0000000)
    XOR     = Instruction(opcode = Opcode.ARITH, type=Instruction.Type.R, funct3 = 0b100, funct7=0b0000000)
    SRL     = Instruction(opcode = Opcode.ARITH, type=Instruction.Type.R, funct3 = 0b101, funct7=0b0000000)
    SRA     = Instruction(opcode = Opcode.ARITH, type=Instruction.Type.R, funct3 = 0b101, funct7=0b0100000)
    OR      = Instruction(opcode = Opcode.ARITH, type=Instruction.Type.R, funct3 = 0b110, funct7=0b0000000)
    AND     = Instruction(opcode = Opcode.ARITH, type=Instruction.Type.R, funct3 = 0b111, funct7=0b0000000)
