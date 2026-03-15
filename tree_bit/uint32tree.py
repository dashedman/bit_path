from obj import BitList, int32_to_list, list_to_int
from tree_bit.base import TreeBitAtom, TreeBit, registry


class UInt32Tree:
    def __init__(self, bits: list[TreeBitAtom]):
        self.bits = bits

    def __invert__(self) -> 'UInt32Tree':
        return UInt32Tree([
            ~bit for bit in self.bits
        ])

    def __xor__(self, other: 'UInt32Tree') -> 'UInt32Tree':
        return UInt32Tree([
            bit1 ^ bit2 for bit1, bit2 in zip(self.bits, other.bits)
        ])

    def __and__(self, other: 'UInt32Tree') -> 'UInt32Tree':
        return UInt32Tree([
            bit1 & bit2 for bit1, bit2 in zip(self.bits, other.bits)
        ])

    def __or__(self, other: 'UInt32Tree') -> 'UInt32Tree':
        return UInt32Tree([
            bit1 | bit2 for bit1, bit2 in zip(self.bits, other.bits)
        ])

    def __add__(self, other: 'UInt32Tree') -> 'UInt32Tree':
        result_bits: list[TreeBitAtom] = []
        overflow_bit = ZERO_BIT
        for bit, other_bit in zip(reversed(self.bits), reversed(other.bits)):
            half_sum = bit ^ other_bit
            result_bits.append(half_sum ^ overflow_bit)
            overflow_bit = bit & other_bit | half_sum & overflow_bit
        result_bits.reverse()
        return UInt32Tree(result_bits)

    def __str__(self):
        return ''.join(
            str(int(bit.value)) if bit.resolved else '?'
            for bit in self.bits
        )

    @classmethod
    def from_bitlist(cls, bit_list: BitList, name: str):
        return cls(TreeBit.from_bitlist(bit_list[-32:], name))

    @classmethod
    def from_int(cls, num: int, name: str):
        return cls.from_bitlist(int32_to_list(num), name)

    def to_int(self):
        if not all(bit.resolved for bit in self.bits):
            raise Exception('Cannot convert to int: unresolved bits')
        return list_to_int([bit.value for bit in self.bits])

    def rol(self, steps: int):
        """ roll right """
        return UInt32Tree(self.bits[steps:] + self.bits[:steps])

    def rev_rol(self, steps: int):
        """ roll left """
        return UInt32Tree(self.bits[-steps:] + self.bits[:-steps])

    def set_exits(self, exits: 'UInt32Tree'):
        for self_bit, exit_bit in zip(self.bits, exits.bits):
            self_bit: TreeBit
            self_bit.set_exit(exit_bit)


ONE_BIT: TreeBit
ZERO_BIT: TreeBit
UINT_ZERO: UInt32Tree


def init_registry():
    global ONE_BIT, ZERO_BIT, UINT_ZERO
    registry.clear()

    ONE_BIT = TreeBit(True, 'ONE')
    ZERO_BIT = TreeBit(False, 'ZERO')
    UINT_ZERO = UInt32Tree([ZERO_BIT] * 32)

def get_one_bit():
    return ONE_BIT


def get_zero_bit():
    return ZERO_BIT


def get_uint_zero():
    return UINT_ZERO