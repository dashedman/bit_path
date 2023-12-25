import base64
from abc import ABC
from collections import deque
from dataclasses import dataclass
from functools import cached_property

import networkx
import matplotlib.pyplot as plt

from obj import H0_START, H1_START, H2_START, H3_START, H4_START, BitList, int32_to_list, str_bits_to_list, chunks, list_to_int


OperationsCounter = dict[tuple, int]
registry: dict[str | tuple | frozenset] = {}


class TreeBitAtom:
    value: bool | None = None
    name: str = NotImplemented

    def __init__(self, *args):
        if self.key in registry:
            raise Exception(f'Registry is not empty for key {self.key}')
        registry[self.key] = self

    def __hash__(self):
        return hash(self.key)

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

    @classmethod
    def with_registry(cls, *operands):
        return registry.get(cls.get_key(*operands)) or cls(*operands)

    @cached_property
    def key(self):
        raise NotImplementedError

    @classmethod
    def get_key(cls, *args):
        raise NotImplementedError

    @property
    def alias_name(self):
        if self.name is NotImplemented:
            return base64.b64encode(str(hash(self.key)).encode('ascii'))
        return self.name


class TreeBit(TreeBitAtom):
    def __init__(self, bit: bool | None, name: str):
        self.name = name
        self.value = bit
        super().__init__()

    @property
    def key(self):
        return self.name

    @classmethod
    def from_bitlist(cls, bit_list: BitList, name_prefix: str = ''):
        if name_prefix != '':
            name_prefix += '_'

        return [
            TreeBit(bit, name_prefix + str(name))
            for name, bit in enumerate(bit_list)
        ]


class TreeBitNOT(TreeBitAtom):
    cls_name = 'not'

    def __init__(self, bit: TreeBitAtom):
        self.bit = bit
        super().__init__()

    @cached_property
    def key(self):
        return self.get_key(self.bit)

    @classmethod
    def get_key(cls, bit: TreeBitAtom):
        return bit.key, cls.cls_name

    @classmethod
    def with_resolve(cls, a: TreeBit):
        if a.resolved:
            return ZERO_BIT if a.value else ONE_BIT
        if isinstance(a, cls):
            return a.bit
        # Nothing is resolved
        return cls.with_registry(a)


class TreeBitOperator(TreeBitAtom, ABC):
    cls_name: str = NotImplemented

    def __init__(self, a: TreeBitAtom, b: TreeBitAtom):
        self.a = a
        self.b = b
        # self.name = '(' + str(a.name) + self.cls_name + str(b.name) + ')'
        # print(self.name)
        super().__init__()

    @cached_property
    def key(self):
        return self.get_key(self.a, self.b)

    @classmethod
    def get_key(cls, a: TreeBitAtom, b: TreeBitAtom):
        return frozenset((a.key, b.key, cls.cls_name))


class TreeBitXOR(TreeBitOperator):
    cls_name = '^'

    def set_exit(self, exit_bit: TreeBit):
        if not exit_bit.resolved:
            raise Exception('Exit value is not resolved!')

        if self.value is not None:
            raise Exception('Exit value already setted!')

        if self.a.value is None:
            if self.b.resolved:
                self.a.set_exit(exit_bit.value ^ self.b.value)
            elif self.b.value is not None:
                self.a.set_exit(exit_bit.value ^ self.b.value)

        if self.b.value is None:
            if self.a.resolved:
                self.b.set_exit(exit_bit.value ^ self.a.value)
            elif self.a.value is not None:
                self.b.set_exit(exit_bit.value ^ self.a.value)

    @classmethod
    def with_resolve(cls, a: TreeBit, b: TreeBit):
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
        return cls.with_registry(a, b)


class TreeBitAND(TreeBitOperator):
    cls_name = '&'

    @classmethod
    def with_resolve(cls, a: TreeBit, b: TreeBit):
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
        return cls.with_registry(a, b)


class TreeBitOR(TreeBitOperator):
    cls_name = '|'

    @classmethod
    def with_resolve(cls, a: TreeBit, b: TreeBit):
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
        return cls.with_registry(a, b)


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
        if any(bit.value is None for bit in self.bits):
            raise Exception('Cannot convert to int: unresolved bits')
        return list_to_int([bit.value for bit in self.bits])

    def rol(self, steps: int):
        return UInt32Tree(self.bits[steps:] + self.bits[:steps])

    def rev_rol(self, steps: int):
        return UInt32Tree(self.bits[-steps:] + self.bits[:-steps])

    def set_exits(self, exits: 'UInt32Tree'):
        for self_bit, exit_bit in zip(self.bits, exits.bits):
            self_bit.set_exit(exit_bit)


@dataclass
class RouteItem:
    bit: TreeBitAtom
    bit_value: bool
    configurations: tuple[tuple[bool, ...], ...] = ()


@dataclass
class RollbackItem:
    bit: TreeBitAtom
    route_erase: int
    configurations: tuple[tuple[bool, ...], ...] = ()


ONE_BIT: TreeBit
ZERO_BIT: TreeBit
UINT_ZERO: UInt32Tree
h0_start_tree: UInt32Tree
h1_start_tree: UInt32Tree
h2_start_tree: UInt32Tree
h3_start_tree: UInt32Tree
h4_start_tree: UInt32Tree


def init_registry():
    global ONE_BIT, ZERO_BIT, UINT_ZERO, h0_start_tree, h1_start_tree, h2_start_tree, h3_start_tree, h4_start_tree
    registry.clear()

    ONE_BIT = TreeBit(True, 'ONE')
    ZERO_BIT = TreeBit(False, 'ZERO')
    UINT_ZERO = UInt32Tree([ZERO_BIT] * 32)

    h0_start_tree = UInt32Tree.from_int(H0_START, 'H0')
    h1_start_tree = UInt32Tree.from_int(H1_START, 'H1')
    h2_start_tree = UInt32Tree.from_int(H2_START, 'H2')
    h3_start_tree = UInt32Tree.from_int(H3_START, 'H3')
    h4_start_tree = UInt32Tree.from_int(H4_START, 'H4')


def sha1(data: str):
    init_registry()

    bytes_: BitList = []
    for char in data:
        bytes_ += str_bits_to_list(f'{ord(char):08b}')
    h: tuple[UInt32Tree] = algo(bytes_)[1:]

    # sha_bits = [bit for hx in h for bit in hx.bits]
    # return scan_bits(sha_bits)
    h0 = h[0].to_int()
    h1 = h[1].to_int()
    h2 = h[2].to_int()
    h3 = h[3].to_int()
    h4 = h[4].to_int()
    return '%08x%08x%08x%08x%08x' % (h0, h1, h2, h3, h4)


def sha1_rev(sha: str, limit_length: int):
    init_registry()
    w, *predicted_h = forward_move(limit_length)
    backward_move(sha, predicted_h, w)
    return None


def forward_move(limit_length: int):
    # prepare from limit
    bits: BitList = [None] * (limit_length * 8)
    return algo(bits)


def algo(bits: BitList):
    bits += [True]
    p_bits = bits.copy()
    # pad until length equals 448 mod 512
    if len(p_bits) % 512 > 448:
        p_bits += [False] * (512 - len(p_bits) % 512)
    if len(p_bits) % 512 < 448:
        p_bits += [False] * (448 - len(p_bits) % 512)
    # append the original length
    p_bits += str_bits_to_list(f'{len(bits) - 1:064b}')

    h0 = h0_start_tree
    h1 = h1_start_tree
    h2 = h2_start_tree
    h3 = h3_start_tree
    h4 = h4_start_tree

    # NOTE: only 1 chunk_512
    words: list[BitList] = chunks(p_bits, 32)
    w = [UINT_ZERO] * 80
    for n in range(0, 16):
        w[n] = UInt32Tree([
            TreeBit(bit, str(n * 32 + index))
            for index, bit in enumerate(words[n])
        ])
    for i in range(16, 80):
        w[i] = (
            w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]
        ).rol(1)

    a = h0
    b = h1
    c = h2
    d = h3
    e = h4

    k1 = UInt32Tree.from_int(0x5A827999, 'K1')
    k2 = UInt32Tree.from_int(0x6ED9EBA1, 'K2')
    k3 = UInt32Tree.from_int(0x8F1BBCDC, 'K3')
    k4 = UInt32Tree.from_int(0xCA62C1D6, 'K4')

    # Main loop
    for i in range(0, 80):
        if 0 <= i <= 19:
            f = (b & c) | ((~b) & d)
            k = k1
        elif 20 <= i <= 39:
            f = b ^ c ^ d
            k = k2
        elif 40 <= i <= 59:
            f = (b & c) | (b & d) | (c & d)
            k = k3
        else:
            f = b ^ c ^ d
            k = k4

        temp = a.rol(5) + f + e + k + w[i]
        e = d
        d = c
        c = b.rol(30)
        b = a
        a = temp

    h0 = h0 + a
    h1 = h1 + b
    h2 = h2 + c
    h3 = h3 + d
    h4 = h4 + e

    return w, h0, h1, h2, h3, h4


def backward_move(
        sha: str,
        predicted_h: tuple[UInt32Tree, ...],
        w: list[UInt32Tree],
):
    # prepare from sha
    h_end = tuple(
        UInt32Tree.from_int(int(sha[offset:offset + 8], 16), f'h{index}')
        for index, offset in enumerate(range(0, 40, 8))
    )
    result_bits = [bit for word in w[:16] for bit in word.bits]
    # scan_lines(predicted_h)
    # draw_graph(predicted_h)
    push_line_predict(predicted_h, h_end, result_bits)


def scan_lines(predicted_h: tuple[UInt32Tree, ...]):
    next_line = []
    curr_line = [bit for h in predicted_h for bit in h.bits]
    used_bits = set(curr_line)
    counter = 0
    while curr_line:
        used_prev = 0
        for bit in curr_line:
            if isinstance(bit, TreeBit):
                continue
            if isinstance(bit, TreeBitNOT):
                if bit.bit not in used_bits:
                    next_line.append(bit.bit)
                    used_bits.add(bit.bit)
                else:
                    if not bit.bit.resolved:
                        used_prev += 1
                continue
            if isinstance(bit, TreeBitOperator):
                if bit.a not in used_bits:
                    next_line.append(bit.a)
                    used_bits.add(bit.a)
                else:
                    if not bit.a.resolved:
                        used_prev += 1
                if bit.b not in used_bits:
                    next_line.append(bit.b)
                    used_bits.add(bit.b)
                else:
                    if not bit.b.resolved:
                        used_prev += 1
                continue
            raise Exception(f'Undefined TreeBit {type(bit)}')

        print(f'line {counter}: {len(curr_line)}, used on prev lines: {used_prev}')
        curr_line, next_line = next_line, []
        counter += 1

    print('registry size:', len(registry), 'used bits:', len(used_bits))


def draw_graph(predicted_h: tuple[UInt32Tree, ...]):
    graph = networkx.DiGraph()

    next_line = []
    curr_line = []
    counter = 1

    for h in predicted_h:
        for bit in h.bits:
            curr_line.append(bit)
            graph.add_node(bit.alias_name, subset=counter)

    used_bits = set(curr_line)
    while curr_line and counter:
        counter += 1

        used_prev = 0
        for bit in curr_line:
            if isinstance(bit, TreeBit):
                continue
            if isinstance(bit, TreeBitNOT):
                if bit.bit not in used_bits:
                    next_line.append(bit.bit)
                    used_bits.add(bit.bit)
                    graph.add_node(bit.bit.alias_name, subset=counter)
                else:
                    if not bit.bit.resolved:
                        used_prev += 1
                graph.add_edge(bit.alias_name, bit.bit.alias_name)
                continue
            if isinstance(bit, TreeBitOperator):
                if bit.a not in used_bits:
                    next_line.append(bit.a)
                    used_bits.add(bit.a)
                    graph.add_node(bit.a.alias_name, subset=counter)
                else:
                    if not bit.a.resolved:
                        used_prev += 1
                if bit.b not in used_bits:
                    next_line.append(bit.b)
                    used_bits.add(bit.b)
                    graph.add_node(bit.b.alias_name, subset=counter)
                else:
                    if not bit.b.resolved:
                        used_prev += 1

                graph.add_edge(bit.alias_name, bit.a.alias_name)
                graph.add_edge(bit.alias_name, bit.b.alias_name)
                continue
            raise Exception(f'Undefined TreeBit {type(bit)}')

        print(f'line {counter}: {len(curr_line)}, used on prev lines: {used_prev}')
        curr_line, next_line = next_line, []

    print('Starting drawning')
    pos = networkx.multipartite_layout(graph, align='horizontal')
    plt.figure(figsize=(25, 25))
    networkx.draw_networkx(
        graph,
        pos=pos,
        with_labels=False,
        node_size=2,
        width=0.2,
        arrowsize=2,
        node_shape='.',
    )
    # text = networkx.draw_networkx_labels(graph, pos=pos, font_size=2)
    # for t in text.values():
    #     t.set_rotation(45)
    # plt.show()
    print('Saving to file')
    plt.savefig("bits_tree.png", dpi=500, bbox_inches='tight')


def push_line_predict(
        predicted_h: tuple[UInt32Tree, ...],
        h_end: tuple[UInt32Tree, ...],
        result_bits: list[TreeBitAtom],
):
    queue_to_scan: deque[RouteItem] = deque()
    scan_queue_contains: set[TreeBitAtom] = set()

    # Initialisation
    for ph, eh in zip(predicted_h, h_end):
        for p_bit, e_bit in zip(ph.bits, eh.bits):
            if p_bit.resolved:
                # skip
                continue

            if isinstance(e_bit.value, bool):
                queue_to_scan.append(
                    RouteItem(bit=p_bit, bit_value=e_bit.value)
                )
                scan_queue_contains.add(p_bit)
            else:
                raise Exception('Exit bit is not bool!!!')

    while queue_to_scan:
        next_route = queue_to_scan.popleft()
        next_bit = next_route.bit
        data_bit_value = next_route.bit_value
        scan_queue_contains.remove(next_bit)

        if check_conflicts(next_bit, data_bit_value):
            print(f'Conflict!')
            continue

        if check_already_resolved(next_bit):
            print('Already!')
            continue

        resolved, next_route = check_and_try_resolve(next_bit, data_bit_value)
        if resolved:
            print(f'Resolve!')
            queue_to_scan.appendleft(next_route)
            scan_queue_contains.add(next_route.bit)
            continue

        branch()


def check_already_resolved(next_bit: TreeBitAtom):
    # extend queue
    if isinstance(next_bit, TreeBitNOT):
        if next_bit.bit.resolved:
            return True

    if isinstance(next_bit, TreeBitOR):
        if next_bit.a.resolved:
            if next_bit.b.resolved:
                return True

    if isinstance(next_bit, TreeBitAND):
        if next_bit.a.resolved:
            if next_bit.b.resolved:
                return True

    if isinstance(next_bit, TreeBitXOR):
        if next_bit.a.resolved:
            if next_bit.b.resolved:
                return True


def check_conflicts(next_bit: TreeBitAtom, data_bit_value: bool):
    # set predicted value
    if next_bit.resolved:
        if next_bit.value != data_bit_value:
            # Next bit value already resolved and has conflict! Need rollback!
            return True

    # extend queue
    if isinstance(next_bit, TreeBitNOT):
        if next_bit.bit.resolved:
            if not next_bit.bit.value != data_bit_value:
                # NOT Conflict! Rollback!
                return True

    if isinstance(next_bit, TreeBitOR):
        if next_bit.a.resolved:
            if next_bit.b.resolved:
                if next_bit.a.value or next_bit.b.value != data_bit_value:
                    # OR Conflict! Rollback!
                    return True
            else:
                # B not resolved
                if next_bit.a.value:
                    if not data_bit_value:
                        # OR Conflict! Rollback!
                        return True
        else:
            # A not resolved
            if next_bit.b.resolved:
                if next_bit.b.value:
                    if not data_bit_value:
                        # OR Conflict! Rollback!
                        return True

    if isinstance(next_bit, TreeBitAND):
        if next_bit.a.resolved:
            if next_bit.b.resolved:
                if next_bit.a.value and next_bit.b.value != data_bit_value:
                    # AND Conflict! Rollback!
                    return True
            else:
                # B not resolved
                if not next_bit.a.value:
                    if data_bit_value:
                        # AND Conflict! Rollback!
                        return True
        else:
            # A not resolved
            if next_bit.b.resolved:
                if not next_bit.b.value:
                    if data_bit_value:
                        # AND Conflict! Rollback!
                        return True

    if isinstance(next_bit, TreeBitXOR):
        if next_bit.a.resolved:
            if next_bit.b.resolved:
                if next_bit.a.value ^ next_bit.b.value != data_bit_value:
                    # XOR Conflict! Rollback!
                    return True
    return False


def check_and_try_resolve(next_bit: TreeBitAtom, data_bit_value: bool):
    # extend queue
    if isinstance(next_bit, TreeBitNOT):
        return True, RouteItem(next_bit.bit, not data_bit_value)

    if isinstance(next_bit, TreeBitOR):
        if next_bit.a.resolved:
            # B not resolved
            if next_bit.a.value:
                # BRANCH B, A is True, X is True
                return False, None
            else:
                return True, RouteItem(next_bit.b, data_bit_value)
        else:
            # A not resolved
            if next_bit.b.resolved:
                if next_bit.b.value:
                    # BRANCH A, B is True, X is True
                    return False, None
                else:
                    return True, RouteItem(next_bit.a, data_bit_value)
            else:
                # A and B not resolved
                # BRANCH A B
                return False, None

    if isinstance(next_bit, TreeBitAND):
        if next_bit.a.resolved:
            # B not resolved
            if not next_bit.a.value:
                # BRANCH B, A is False, X is False
                return False, None
            else:
                return True, RouteItem(next_bit.b, data_bit_value)
        else:
            # A not resolved
            if next_bit.b.resolved:
                if not next_bit.b.value:
                    # BRANCH A, B is False, X is False
                    return False, None
                else:
                    return True, RouteItem(next_bit.a, data_bit_value)
            else:
                # A and B not resolved
                # BRANCH A B
                return False, None

    if isinstance(next_bit, TreeBitXOR):
        if next_bit.a.resolved:
            # B not resolved
            return True, RouteItem(next_bit.b, next_bit.a.value ^ data_bit_value)
        else:
            # A not resolved
            if next_bit.b.resolved:
                return True, RouteItem(next_bit.a, next_bit.b.value ^ data_bit_value)
            else:
                # BRANCH A B
                return False, None


def branch():
    pass


def push_predict(
        predicted_h: tuple[UInt32Tree, ...],
        h_end: tuple[UInt32Tree, ...],
        result_bits: list[TreeBitAtom],
):

    # False - rollback phase
    # True - push phase
    push_state = True
    routing_queue: deque[RouteItem] = deque()
    rollback_stack: deque[RollbackItem] = deque()

    # Initialisation
    for ph, eh in zip(predicted_h, h_end):
        for p_bit, e_bit in zip(ph.bits, eh.bits):
            if p_bit.resolved:
                # skip
                continue

            if isinstance(e_bit.value, bool):
                routing_queue.append(RouteItem(p_bit, e_bit.value))
            else:
                raise Exception('Exit bit is not bool!!!')

    while True:
        if push_state:
            # PUSH
            if routing_queue:
                next_route = routing_queue.popleft()
                next_bit = next_route.bit
                data_bit_value = next_route.bit_value
                configuration = next_route.configurations[0] if next_route.configurations else None
            else:
                print(scan_bits(result_bits))
                push_state = False
                continue

            # set predicted value
            if next_bit.resolved:
                if next_bit.value != data_bit_value:
                    # Next bit value already resolved and has conflict! Need rollback!
                    push_state = False
                continue
            next_bit.value = data_bit_value

            # extend queue
            if isinstance(next_bit, TreeBitNOT):
                if next_bit.bit.resolved:
                    if not next_bit.bit.value != data_bit_value:
                        # NOT Conflict! Rollback!
                        push_state = False
                        rollback_stack.append(RollbackItem(next_bit, 0))
                    continue
                routing_queue.append(RouteItem(next_bit.bit, not data_bit_value))
                rollback_stack.append(RollbackItem(next_bit, 1))

            if isinstance(next_bit, TreeBitOR):
                if next_bit.a.resolved:
                    if next_bit.b.resolved:
                        if next_bit.a.value or next_bit.b.value != data_bit_value:
                            # OR Conflict! Rollback!
                            push_state = False
                            rollback_stack.append(RollbackItem(next_bit, 0))
                        continue
                    else:
                        # B not resolved
                        if next_bit.a.value:
                            if not data_bit_value:
                                # OR Conflict! Rollback!
                                push_state = False
                                rollback_stack.append(RollbackItem(next_bit, 0))
                                continue
                            # BRANCH B, A is True, X is True
                            if configuration:
                                # set configuration from rollback
                                conf_b = configuration[0]
                                routing_queue.append(RouteItem(next_bit.b, conf_b))
                                rollback_stack.append(RollbackItem(next_bit, 1))
                            else:
                                # free configuration - create new rollback item
                                routing_queue.append(RouteItem(next_bit.b, False))
                                rollback_stack.append(
                                    RollbackItem(next_bit, 1, ((True,),))
                                )
                        else:
                            routing_queue.append(RouteItem(next_bit.b, data_bit_value))
                            rollback_stack.append(RollbackItem(next_bit, 1))
                else:
                    # A not resolved
                    if next_bit.b.resolved:
                        if next_bit.b.value:
                            if not data_bit_value:
                                # OR Conflict! Rollback!
                                push_state = False
                                rollback_stack.append(RollbackItem(next_bit, 0))
                                continue
                            # BRANCH A, B is True, X is True
                            if configuration:
                                # set configuration from rollback
                                conf_a = configuration[0]
                                routing_queue.append(RouteItem(next_bit.a, conf_a))
                                rollback_stack.append(RollbackItem(next_bit, 1))
                            else:
                                # free configuration - create new rollback item
                                routing_queue.append(RouteItem(next_bit.a, False))
                                rollback_stack.append(
                                    RollbackItem(next_bit, 1, ((True,),))
                                )
                        else:
                            routing_queue.append(RouteItem(next_bit.a, data_bit_value))
                            rollback_stack.append(RollbackItem(next_bit, 1))
                    else:
                        # A and B not resolved
                        # BRANCH A B
                        if configuration:
                            # set configuration from rollback
                            conf_a, conf_b = configuration
                            routing_queue.append(RouteItem(next_bit.a, conf_a))
                            routing_queue.append(RouteItem(next_bit.b, conf_b))
                            rollback_stack.append(RollbackItem(
                                next_bit, 2,
                                next_route.configurations[1:]
                            ))
                        else:
                            # free configuration - create new rollback item
                            if not data_bit_value:
                                routing_queue.append(RouteItem(next_bit.a, False))
                                routing_queue.append(RouteItem(next_bit.b, False))
                                rollback_stack.append(RollbackItem(next_bit, 2))
                            else:
                                routing_queue.append(RouteItem(next_bit.a, True))
                                routing_queue.append(RouteItem(next_bit.b, True))
                                rollback_stack.append(
                                    RollbackItem(
                                        next_bit, 2,
                                        ((False, True), (True, False))
                                    )
                                )

            if isinstance(next_bit, TreeBitAND):
                if next_bit.a.resolved:
                    if next_bit.b.resolved:
                        if next_bit.a.value and next_bit.b.value != data_bit_value:
                            # AND Conflict! Rollback!
                            push_state = False
                            rollback_stack.append(RollbackItem(next_bit, 0))
                        continue
                    else:
                        # B not resolved
                        if not next_bit.a.value:
                            if data_bit_value:
                                # AND Conflict! Rollback!
                                push_state = False
                                rollback_stack.append(RollbackItem(next_bit, 0))
                                continue

                            # BRANCH B, A is False, X is False
                            if configuration:
                                # set configuration from rollback
                                conf_b = configuration[0]
                                routing_queue.append(RouteItem(next_bit.b, conf_b))
                                rollback_stack.append(RollbackItem(next_bit, 1))
                            else:
                                # free configuration - create new rollback item
                                routing_queue.append(RouteItem(next_bit.b, False))
                                rollback_stack.append(
                                    RollbackItem(next_bit, 1, ((True,),))
                                )
                        else:
                            routing_queue.append(RouteItem(next_bit.b, data_bit_value))
                            rollback_stack.append(RollbackItem(next_bit, 1))
                else:
                    # A not resolved
                    if next_bit.b.resolved:
                        if not next_bit.b.value:
                            if data_bit_value:
                                # AND Conflict! Rollback!
                                push_state = False
                                rollback_stack.append(RollbackItem(next_bit, 0))
                                continue

                            # BRANCH A, B is False, X is False
                            if configuration:
                                # set configuration from rollback
                                conf_a = configuration[0]
                                routing_queue.append(RouteItem(next_bit.a, conf_a))
                                rollback_stack.append(RollbackItem(next_bit, 1))
                            else:
                                # free configuration - create new rollback item
                                routing_queue.append(RouteItem(next_bit.a, False))
                                rollback_stack.append(
                                    RollbackItem(next_bit, 1, ((True,),))
                                )
                        else:
                            routing_queue.append(RouteItem(next_bit.a, data_bit_value))
                            rollback_stack.append(RollbackItem(next_bit, 1))
                    else:
                        # A and B not resolved
                        # BRANCH A B
                        if configuration:
                            # set configuration from rollback
                            conf_a, conf_b = configuration
                            routing_queue.append(RouteItem(next_bit.a, conf_a))
                            routing_queue.append(RouteItem(next_bit.b, conf_b))
                            rollback_stack.append(RollbackItem(
                                next_bit, 2,
                                next_route.configurations[1:]
                            ))
                        else:
                            # free configuration - create new rollback item
                            if data_bit_value:
                                routing_queue.append(RouteItem(next_bit.a, True))
                                routing_queue.append(RouteItem(next_bit.b, True))
                                rollback_stack.append(RollbackItem(next_bit, 2))
                            else:
                                routing_queue.append(RouteItem(next_bit.a, False))
                                routing_queue.append(RouteItem(next_bit.b, False))
                                rollback_stack.append(
                                    RollbackItem(
                                        next_bit, 2,
                                        ((False, True), (True, False))
                                    )
                                )

            if isinstance(next_bit, TreeBitXOR):
                if next_bit.a.resolved:
                    if next_bit.b.resolved:
                        if next_bit.a.value ^ next_bit.b.value != data_bit_value:
                            # XOR Conflict! Rollback!
                            push_state = False
                            rollback_stack.append(RollbackItem(next_bit, 0))
                        continue
                    else:
                        # B not resolved
                        routing_queue.append(RouteItem(next_bit.b, next_bit.a.value ^ data_bit_value))
                        rollback_stack.append(RollbackItem(next_bit, 1))
                else:
                    # A not resolved
                    if next_bit.b.resolved:
                        routing_queue.append(RouteItem(next_bit.a, next_bit.b.value ^ data_bit_value))
                        rollback_stack.append(RollbackItem(next_bit, 1))
                    else:
                        # BRANCH A B
                        if configuration:
                            # set configuration from rollback
                            conf_a, conf_b = configuration
                            routing_queue.append(RouteItem(next_bit.a, conf_a))
                            routing_queue.append(RouteItem(next_bit.b, conf_b))
                            rollback_stack.append(RollbackItem(
                                next_bit, 2,
                                next_route.configurations[1:]
                            ))
                        else:
                            # free configuration - create new rollback item
                            routing_queue.append(RouteItem(next_bit.a, False))
                            routing_queue.append(RouteItem(next_bit.b, False ^ data_bit_value))
                            rollback_stack.append(
                                RollbackItem(
                                    next_bit, 2,
                                    ((True, True ^ data_bit_value),)
                                )
                            )

        else:
            # ROLLBACK
            if rollback_stack:
                rollback_item = rollback_stack.pop()
                for _ in range(rollback_item.route_erase):
                    routing_queue.pop()

                if isinstance(rollback_item.bit.value, bool):
                    if rollback_item.configurations:
                        # end of rollback, turn forward
                        print(sum(bool(ri.configurations) for ri in rollback_stack), len(rollback_stack), len(routing_queue))
                        push_state = True

                    routing_queue.appendleft(RouteItem(
                        rollback_item.bit,
                        rollback_item.bit.value,
                        rollback_item.configurations
                    ))

                    rollback_item.bit.value = None
                else:
                    raise Exception('Rollback value is not bool')
            else:
                # Rollback stack is empty! Algo is done
                break


def scan_bits(result_bits: list[TreeBitAtom]):
    raw_bits: BitList = []
    for bit in result_bits:
        if not bit.resolved:
            raise Exception('Result bit nor resolved')
        raw_bits.append(bit.value)

    scanned_len = list_to_int(raw_bits[-64:])
    raw_bits = raw_bits[:-64]

    result_bytes = b''
    for i in range(0, scanned_len * 8, 8):
        raw_byte = list_to_int(raw_bits[i:i+8])
        result_bytes += raw_byte.to_bytes(1, 'big')

    return result_bytes


if __name__ == '__main__':
    word = '1'
    print(f'Word: "{word}", len: {len(word)}')
    word_sha1 = sha1(word)
    print(f'Word SHA1: {word_sha1}')

    word_reverse = sha1_rev(word_sha1, len(word))
    print(f'Reverse: {word_reverse}')
