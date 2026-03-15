from abc import ABC
from collections import Counter
from dataclasses import dataclass, field
from functools import cached_property
from typing import Iterable

from obj import BitList, int32_to_list, list_to_int

TreeBitKey = str | tuple | frozenset


@dataclass
class RegistryItem:
    bit: 'TreeBitAtom'
    usages: set[TreeBitKey] = field(default_factory=set)


registry: dict[TreeBitKey, RegistryItem] = {}
registry_hit_counter = Counter()
# registry: dict[TreeBitKey, 'TreeBitAtom'] = {}


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
        return cls.cls_name, bit

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
        # return cls(a, value=1.0 - a.value)
        return cls.with_registry(a, value=1.0 - a.value)

    @property
    def parents(self):
        return (self.bit,)


class TreeBitOperator(TreeBitAtom, ABC):
    cls_name: str = NotImplemented

    def __init__(self, args: tuple[TreeBitAtom, TreeBitAtom], *, value: float | bool):
        self.a, self.b = args
        # self.name = '(' + str(a.name) + self.cls_name + str(b.name) + ')'
        # print(self.name)
        super().__init__(value=value)

    @cached_property
    def key(self):
        return self.get_key(self.parents)

    @property
    def label(self):
        return f'{self.cls_name} {self.value:.02f}'

    @classmethod
    def get_key(cls, operands):
        return frozenset((cls.cls_name, *operands))

    @property
    def parents(self):
        return self.a, self.b


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
            (a, b),
            value=(1.0 - a.value) * b.value + a.value * (1.0 - b.value)
        )


class TreeBitEq(TreeBitOperator):
    cls_name = '='

    @classmethod
    def with_resolve(cls, a: TreeBitAtom, b: TreeBitAtom):
        if a.resolved:
            if b.resolved:
                return ONE_BIT if a.value == b.value else ZERO_BIT
            else:
                return b if a.value else TreeBitNOT.with_resolve(b)

        # A not resolved
        if b.resolved:
            return a if b.value else TreeBitNOT.with_resolve(a)

        if a.key == b.key:
            return ONE_BIT

        # Nothing is resolved
        return cls.with_registry(
            (a, b),
            value=(1.0 - a.value) * (1.0 - b.value) + a.value * b.value
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
        return cls.with_registry((a, b), value=a.value * b.value)


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
        return cls.with_registry((a, b), value=a.value + b.value - a.value * b.value)


class TreeBitMultiOperator(TreeBitAtom, ABC):
    cls_name: str = NotImplemented

    def __init__(self, args_set: frozenset[TreeBitAtom], value: float | bool):
        # if args_set is None:
        #     args_set = set()
        # if args:
        #     args_set.update(args)
        self.args = args_set
        super().__init__(value=value)

    def __len__(self):
        return len(self.args)

    @cached_property
    def key(self):
        return self.get_key(self.args)

    @property
    def label(self):
        return f'{self.cls_name} {self.value:.02f}'

    @classmethod
    def get_key(cls, args: frozenset[TreeBitAtom]):
        return cls.cls_name, args

    @property
    def parents(self):
        return self.args


class TreeBitMultiAnd(TreeBitMultiOperator):
    cls_name = '&'

    @classmethod
    def with_resolve(cls, *operands: TreeBitAtom):
        args = set()
        not_args = set()

        for operand in operands:
            operand_args: Iterable[TreeBitAtom]

            if isinstance(operand, TreeBitMultiAnd):
                operand_args = operand.args
            elif isinstance(operand, TreeBitNOT) and isinstance(operand.bit, TreeBitMultiOr):
                operand_args = [
                    TreeBitNOT.with_resolve(sub_operand)
                    for sub_operand in operand.bit.args
                ]
            else:
                operand_args = (operand,)

            for op_arg in operand_args:
                if op_arg.resolved:
                    if op_arg.value:
                        # A & 1 = A
                        continue
                    else:
                        # A & 0 = 0
                        return ZERO_BIT

                if isinstance(op_arg, TreeBitNOT):
                    if op_arg.bit in args:
                        # A & ~A = 0
                        return ZERO_BIT
                    else:
                        not_args.add(op_arg.bit)
                elif op_arg in not_args:
                    # A & ~A = 0
                    return ZERO_BIT

                args.add(op_arg)

        if len(args) == 0:
            return ONE_BIT
        elif len(args) == 1:
            return args.pop()

        true_probability = 1
        for arg in args:
            true_probability *= arg.value
        return cls.with_registry(
            frozenset(args),
            value=true_probability,
            # args_set=args,
        )



class TreeBitMultiOr(TreeBitMultiOperator):
    cls_name = '|'

    @classmethod
    def with_resolve(cls, *operands: TreeBitAtom):
        args = set()
        not_args = set()

        for operand in operands:
            operand_args: Iterable[TreeBitAtom]

            if isinstance(operand, TreeBitMultiOr):
                operand_args = operand.args
            elif isinstance(operand, TreeBitNOT) and isinstance(operand.bit, TreeBitMultiAnd):
                operand_args = [
                    TreeBitNOT.with_resolve(sub_operand)
                    for sub_operand in operand.bit.args
                ]
            else:
                operand_args = (operand,)

            for op_arg in operand_args:
                if op_arg.resolved:
                    if op_arg.value:
                        # A | 1 = 1
                        return ONE_BIT
                    else:
                        # A | 0 = A
                        continue

                if isinstance(op_arg, TreeBitNOT):
                    if op_arg.bit in args:
                        # A | ~A = 1
                        return ONE_BIT
                    else:
                        not_args.add(op_arg.bit)
                elif op_arg in not_args:
                    # A | ~A = 1
                    return ONE_BIT

                args.add(op_arg)

        if len(args) == 0:
            return ZERO_BIT
        elif len(args) == 1:
            return args.pop()

        false_probability = 1
        for arg in args:
            false_probability *= (1 - arg.value)
        return cls.with_registry(
            frozenset(args),
            value=1 - false_probability,
            # args_set=args,
        )


class TreeBitMultiXor(TreeBitMultiOperator):
    cls_name = '^'

    @classmethod
    def with_resolve(cls, *operands: TreeBitAtom):
        args = set()
        not_counter = 0

        for operand in operands:
            operand_args: Iterable[TreeBitAtom]

            if isinstance(operand, TreeBitNOT):
                operand = operand.bit
                not_counter += 1

            if isinstance(operand, TreeBitMultiXor):
                operand_args = operand.args
            elif isinstance(operand, TreeBitMultiEq):
                if len(operand) % 2 == 1:
                    # odd equality same as xor
                    pass
                else:
                    # even equality same as ~xor
                    not_counter += 1
                operand_args = operand.args

            else:
                operand_args = (operand,)

            for op_arg in operand_args:
                if op_arg.resolved:
                    if op_arg.value:
                        # A ^ 1 = ~A
                        not_counter += 1
                        continue
                    else:
                        # A ^ 0 = A
                        continue

                if isinstance(op_arg, TreeBitNOT):
                    # unpack not
                    # Z ^ ~A = ~(Z ^ A)
                    op_arg = op_arg.bit
                    # move arg's not over xor
                    not_counter += 1

                if op_arg in args:
                    # Z ^ A ^ A = Z ^ 0 = Z
                    # just remove from args
                    args.remove(op_arg)
                else:
                    # Z ^ A
                    args.add(op_arg)

        is_negative = not_counter % 2 == 1
        if len(args) == 0:
            return ONE_BIT if is_negative else ZERO_BIT
        elif len(args) == 1:
            last_arg = args.pop()
            return TreeBitNOT.with_resolve(last_arg) if is_negative else last_arg

        # TODO:
        # false_probability = 1
        # for arg in args:
        #     false_probability *= (1 - arg.value)
        xor_instance = cls.with_registry(
            frozenset(args),
            value=0.5,
            # args_set=args,
        )
        if is_negative:
            return TreeBitNOT.with_resolve(xor_instance)
        else:
            return xor_instance


class TreeBitMultiEq(TreeBitMultiOperator):
    """
    eq(n) = {
        ~xor(n) if n = 2m;
        xor(n) if n = 2m + 1;
    }
    """
    cls_name = '='

    @classmethod
    def with_resolve(cls, *operands: TreeBitAtom):
        xor = TreeBitMultiXor.with_resolve(*operands)
        if len(operands) % 2 == 0:
            # even args same as ~xor
            return TreeBitNOT.with_resolve(xor)
        else:
            return xor
