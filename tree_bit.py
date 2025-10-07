from abc import ABC
from dataclasses import dataclass, field
from functools import cached_property

from obj import BitList, int32_to_list, list_to_int


TreeBitKey = str | tuple | frozenset


@dataclass
class RegistryItem:
    bit: 'TreeBitAtom'
    usages: set[TreeBitKey] = field(default_factory=set)


registry: dict[TreeBitKey, RegistryItem] = {}
# registry: dict[TreeBitKey, 'TreeBitAtom'] = {}


class TreeBitAtom:
    value: bool | float = None
    name: str = NotImplemented

    def __init__(self, *args, value: float | bool | None = None):
        if value is not None:
            self.value = value

        if self.key in registry:
            raise Exception(f'Registry is not empty for key {self.key}')
        # registry[self.key] = self
        registry[self.key] = RegistryItem(self)

    def __hash__(self):
        return hash(self.key)

    def __eq__(self, other: 'TreeBitAtom'):
        return self.key == other.key

    def __xor__(self, other):
        return TreeBitXOR.with_resolve(self, other)

    def __and__(self, other):
        return TreeBitAND.with_resolve(self, other)

    def __or__(self, other):
        return TreeBitOR.with_resolve(self, other)

    def __invert__(self):
        return TreeBitNOT.with_resolve(self)

    @property
    def resolved(self):
        return isinstance(self.value, bool)

    @classmethod
    def with_resolve(cls, *operands):
        raise NotImplementedError

    # @classmethod
    # def with_registry(cls, *operands):
    #     return registry.get(cls.get_key(*operands)) or cls(*operands)

    @classmethod
    def with_registry(cls, *operands: 'TreeBitAtom', value: float | bool | None = None):
        from_registry = registry.get(cls.get_key(*operands))
        if from_registry:
            result_bit = from_registry.bit
        else:
            result_bit = cls(*operands, value=value)

        for operand in operands:
            registry[operand.key].usages.add(result_bit.key)

        return result_bit

    @classmethod
    def from_bit_value(cls, value: float | bool, name: str):
        raise NotImplementedError

    @cached_property
    def key(self):
        raise NotImplementedError

    @classmethod
    def get_key(cls, *args):
        raise NotImplementedError

    @property
    def label(self):
        raise NotImplementedError

    # @property
    # def usages(self):
    #     return registry[self.key].usages

    def determinancy(self):
        return abs(self.value - 0.5)


class TreeBit(TreeBitAtom):
    def __init__(self, value: bool | float, name: str):
        self.name = name
        super().__init__(value=value)

    @property
    def key(self):
        return self.name

    @property
    def label(self):
        return f'{self.name} {self.value:.02f}'

    @classmethod
    def from_bitlist(cls, bit_list: BitList, name_prefix: str = ''):
        if name_prefix != '':
            name_prefix += '_'

        return [
            TreeBit.from_bit_value(bit, name_prefix + str(name))
            for name, bit in enumerate(bit_list)
        ]

    @classmethod
    def from_bit_value(cls, value: float | bool, name: str):
        if value == 1.0:
            return ONE_BIT
        if value == 0.0:
            return ZERO_BIT
        return TreeBit(value, name)


class TreeBitNOT(TreeBitAtom):
    cls_name = 'not'

    def __init__(self, bit: TreeBitAtom, *, value: float | bool):
        self.bit = bit
        super().__init__(value=value)

    @cached_property
    def key(self):
        return self.get_key(self.bit)

    @classmethod
    def get_key(cls, bit: TreeBitAtom):
        return bit.key, cls.cls_name

    @property
    def label(self):
        return f'{self.cls_name} {self.value:.02f}'

    @classmethod
    def with_resolve(cls, a: TreeBitAtom):
        if a.resolved:
            return ZERO_BIT if a.value else ONE_BIT
        if isinstance(a, cls):
            return a.bit
        # Nothing is resolved
        return cls.with_registry(a, value=1.0 - a.value)


class TreeBitOperator(TreeBitAtom, ABC):
    cls_name: str = NotImplemented

    def __init__(self, a: TreeBitAtom, b: TreeBitAtom, *, value: float | bool):
        self.a = a
        self.b = b
        # self.name = '(' + str(a.name) + self.cls_name + str(b.name) + ')'
        # print(self.name)
        super().__init__(value=value)

    @cached_property
    def key(self):
        return self.get_key(self.a, self.b)

    @property
    def label(self):
        return f'{self.cls_name} {self.value:.02f}'

    @classmethod
    def get_key(cls, a: TreeBitAtom, b: TreeBitAtom):
        return frozenset((a.key, b.key, cls.cls_name))


class TreeBitXOR(TreeBitOperator):
    cls_name = '^'

    @classmethod
    def with_resolve(cls, a: TreeBitAtom, b: TreeBitAtom):
        if a.resolved:
            if b.resolved:
                return ONE_BIT if a.value ^ b.value else ZERO_BIT
            else:
                return TreeBitNOT.with_resolve(b) if a.value else b

        # A not resolved
        if b.resolved:
            return TreeBitNOT.with_resolve(a) if b.value else a

        if a.key == b.key:
            return ZERO_BIT

        # Nothing is resolved
        return cls.with_registry(
            a, b,
            value=(1.0 - a.value) * b.value + a.value * (1.0 - b.value)
        )


class TreeBitAND(TreeBitOperator):
    cls_name = '&'

    @classmethod
    def with_resolve(cls, a: TreeBitAtom, b: TreeBitAtom):
        if a.resolved:
            if not a.value:
                return ZERO_BIT

            if b.resolved:
                return ONE_BIT if b.value else ZERO_BIT
            else:
                return b

        # A not resolved
        if b.resolved:
            if not b.value:
                return ZERO_BIT
            return a

        if a.key == b.key:
            return a

        # Nothing is resolved
        return cls.with_registry(a, b, value=a.value * b.value)


class TreeBitOR(TreeBitOperator):
    cls_name = '|'

    @classmethod
    def with_resolve(cls, a: TreeBitAtom, b: TreeBitAtom):
        if a.resolved:
            if a.value:
                return ONE_BIT

            if b.resolved:
                return ONE_BIT if b.value else ZERO_BIT
            else:
                return b

        # A not resolved
        if b.resolved:
            if b.value:
                return ONE_BIT
            return a

        if a.key == b.key:
            return a
        # Nothing is resolved
        return cls.with_registry(a, b, value=a.value + b.value - a.value * b.value)


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
