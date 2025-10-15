import itertools
from ast import Index
from collections import defaultdict
from typing import Counter, Self

from tqdm import tqdm

import tree_bit
from tree_bit.base import TreeBit, TreeBitAtom
from tree_bit.tools import extract_base_bits, solve_bits, get_ancestors_gen, get_solving_order

DNF_TERM = TreeBit
DNF_LITERAL = tuple[DNF_TERM, bool]
CONJUNCT = frozenset[DNF_LITERAL]
DNF = set[CONJUNCT]

LogicalIdentity = 0
LogicalNot = 1
LogicalAnd = 2
LogicalOr = 3


def str_conj(conj: CONJUNCT):
    conj_str = ''
    for unit, is_negate in sorted(conj, key=lambda tn: ord(tn[0].name) * 2 + tn[1]):
        if is_negate:
            unit_str = f'~{unit.name}'
        else:
            unit_str = f'{unit.name}'

        conj_str += unit_str + ' '
    return conj_str.rstrip(' *')


class Dnf:
    def __init__(self, dnf: DNF | None = None):
        self.conjuncts: DNF = dnf or set()
        self._terms: list[DNF_TERM] | None = None
        self._full_terms = None
        self.solved = None if self.conjuncts else True

    def __len__(self):
        return len(self.conjuncts)

    def __iter__(self):
        return iter(self.conjuncts)

    def __str__(self):
        dnf_str = f'Dnf ({len(self)}, {max(len(conj) for conj in self) if self else 0}): '
        for conj in self:
            dnf_str += '(' + str_conj(conj) + ') + '

        dnf_str = dnf_str.rstrip(' +')
        return dnf_str

    @property
    def terms(self):
        if self._terms is None:
            self._terms = sorted(
                {term for conj in self for term, _ in conj},
                key=lambda tn: int(tn.name)
            )
        return self._terms

    @property
    def full_terms(self):
        if self._full_terms is None:
            self._full_terms = self.terms
        return self._full_terms

    def drop_terms(self):
        self._terms = None

    def rank(self):
        return sum(map(len, self))

    @classmethod
    def get_sdnf_for_bit_old(cls, bit: TreeBitAtom, control_value: bool):
        base_bits = sorted(extract_base_bits(bit), key=lambda tn: int(tn.name))
        disjunctions_result_map: dict[tuple[bool, ...], bool] = {}
        dnf_mask = 0

        necessary_bits = set(get_ancestors_gen(bit))
        solving_order = get_solving_order(base_bits, bit, necessary_bits)
        for disjunction_values in tqdm(itertools.product(
                *([(False, True)] * len(base_bits)),
        )):
            solve_map = dict(zip(base_bits, disjunction_values))
            solving = solve_bits(solve_map, solving_order)
            # solving = solve_bits_dfs(solve_map.copy(), bit)
            # print(f'Disjunction {solving}={disjunction_values}')
            disjunctions_result_map[tuple(disjunction_values)] = solving
            dnf_mask = (dnf_mask << 1) + solving

        dnf = set()

        for disjunction_values, solving in disjunctions_result_map.items():
            if control_value == solving:
                disjunction_unit = set()
                for base_bit, value in zip(base_bits, disjunction_values):
                    disjunction_unit.add((base_bit, not value))
                dnf.add(frozenset(disjunction_unit))
        dnf_instance = cls(dnf)
        return dnf_instance

    @classmethod
    def get_sdnf_for_bit(cls, exit_bit: TreeBitAtom):
        dnf_for_bit = dict[TreeBitAtom, Dnf]()
        counter = 0
        depth_counter = 0

        def dnf_by_bit_dfs(
            bit: TreeBitAtom,
        ) -> Dnf:
            nonlocal depth_counter
            if (dnf := dnf_for_bit.get(bit)) is not None:
                # memorisation cache
                return dnf

            depth_counter += 1
            # 19 it/sec
            match type(bit):
                case tree_bit.base.TreeBitNOT:
                    dnf = ~dnf_by_bit_dfs(bit.bit)
                case tree_bit.base.TreeBitXOR:
                    dnf = dnf_by_bit_dfs(bit.a) ^ dnf_by_bit_dfs(bit.b)
                case tree_bit.base.TreeBitOR:
                    dnf = dnf_by_bit_dfs(bit.a) | dnf_by_bit_dfs(bit.b)
                case tree_bit.base.TreeBitAND:
                    dnf = dnf_by_bit_dfs(bit.a) & dnf_by_bit_dfs(bit.b)
                case tree_bit.base.TreeBit:
                    dnf = cls.from_explicit_bit(bit)
                case _:
                    raise Exception('unreachable')

            if not isinstance(dnf, Dnf):
                raise Exception()
            dnf_for_bit[bit] = dnf
            print(counter, depth_counter, len(dnf_for_bit), len(dnf), len(dnf.terms))
            depth_counter -= 1
            return dnf

        return dnf_by_bit_dfs(exit_bit)

    @classmethod
    def from_explicit_bit(cls, bit: TreeBit):
        return cls({frozenset(((bit, False),))})

        # sub_heap_for_bit = {}
        # def get_logical_heap(bit: TreeBitAtom):
        #     if bit in sub_heap_for_bit:
        #         return sub_heap_for_bit[bit]
        #
        #     match type(bit):
        #         case tree_bit.base.TreeBit:
        #             heap = (LogicalIdentity, bit)
        #         case tree_bit.base.TreeBitNOT:
        #             heap = (LogicalNot, get_logical_heap(bit.bit))
        #         case tree_bit.base.TreeBitAND:
        #             heap = (LogicalAnd, get_logical_heap(bit.a), get_logical_heap(bit.b))
        #         case tree_bit.base.TreeBitOR:
        #             heap = (LogicalOr, get_logical_heap(bit.a), get_logical_heap(bit.b))
        #         case tree_bit.base.TreeBitXOR:
        #             heap = (
        #                 LogicalOr,
        #                 (
        #                     LogicalAnd,
        #                     (LogicalNot, get_logical_heap(bit.a)),
        #                     get_logical_heap(bit.b)
        #                 ),
        #                 (
        #                     LogicalAnd,
        #                     get_logical_heap(bit.a),
        #                     (LogicalNot, get_logical_heap(bit.b))
        #                 )
        #             )
        #     sub_heap_for_bit[bit] = heap
        #     return heap
        #
        # logical_heap = get_logical_heap(exit_bit)
        # del sub_heap_for_bit

        # dnf_for_heap = {}
        # def logical_heap_to_dnf(heap_head) -> DNF:
        #     if heap_head in dnf_for_heap:
        #         return dnf_for_heap[heap_head]
        #     heap_op = heap_head[0]
        #     match heap_op:
        #         case 0: # Identity
        #             # simple DNF
        #             dnf_heap = {frozenset({(heap_head[1], False)})}
        #         case 1: # Not
        #             next_heap = heap_head[1]
        #             next_command = next_heap[0]
        #             next_args_len = len(next_heap) - 1
        #             if next_args_len == 1:
        #                 next_arg = next_heap[1]
        #                 if next_command == 0:   # not one
        #                     dnf_heap = {frozenset({(heap_head[1], True)})}
        #                 elif next_command == 1:   # not not
        #                     dnf_heap = logical_heap_to_dnf(next_arg)
        #                 else:
        #                     raise Exception('Unreachble')
        #             else:
        #                 if next_command == 2:
        #                     new_command = 3
        #                 else:
        #                     new_command = 2
        #                 next_args = next_heap[1:]
        #                 dnf_heap = logical_heap_to_dnf((
        #                     new_command,
        #                     *((LogicalNot, arg) for arg in next_args)
        #                 ))
        #         case 2: # and
        #             heap_args = heap_head[1:]
        #             dnf_from_args = []
        #             for heap_arg in heap_args:
        #                 dnf_from_args.append(logical_heap_to_dnf(heap_arg))
        #
        #             merged_dnf = dnf_from_args.pop()
        #             for dnf in dnf_from_args:
        #                 merged_dnf = merge_2_dnf(merged_dnf, dnf)
        #             dnf_heap = merged_dnf
        #         case 3: # or
        #             heap_args = heap_head[1:]
        #             dnf_from_args = []
        #             for heap_arg in heap_args:
        #                 dnf_from_args.append(logical_heap_to_dnf(heap_arg))
        #
        #             merged_dnf = dnf_from_args.pop()
        #             for dnf in dnf_from_args:
        #                 merged_dnf |= dnf
        #             dnf_heap = merged_dnf
        #         case _:
        #             raise Exception('unreach')
        #
        #     dnf_for_heap[heap_head] = dnf_heap
        #     return dnf_heap
        #
        # dnf = logical_heap_to_dnf(logical_heap)
        # dnf_instance = cls(dnf)
        # return dnf_instance

    def minimize_dnf(self):
        if self.solved:
            assert len(self.conjuncts) == 0
            return self
        used_literals = Counter[DNF_LITERAL](itertools.chain.from_iterable(self))
        # assert len(used_literals.most_common()[0]) < len(self.terms)

        while True:
            length = len(self)
            rank = self.rank()
            terms_count = len(self.terms)
            mask = self.probe_dnf_mask()

            self.sticky_minimize()
            self.swallow_minimize()

            new_length = len(self)
            if new_length < 1 and self.solved is None:
                raise Exception()
            new_rank = self.rank()
            new_terms_count = len(self.terms)
            new_mask = self.probe_dnf_mask()

            assert new_rank <= rank
            if new_rank == rank:
                break
            # print(
            #     f'Minimization step '
            #     f'({terms_count}, {length}, {rank}) -> ({new_terms_count}, {new_length}, {new_rank}) '
            #     f'mask_same: {mask == new_mask}, '
            #     f'mask: {new_mask}'
            # )
            assert mask == new_mask

        return self

    def sticky_minimize(self):
        for term in self.terms:
            pos_literal = (term, False)
            neg_literal = (term, True)

            pos_conjs = set()
            neg_conjs = set()

            for conj in self:
                if pos_literal in conj:
                    pos_conjs.add(conj - {pos_literal})

                if neg_literal in conj:
                    neg_conjs.add(conj - {neg_literal})

                # if pos_conjs & neg_conjs:
                #     break

            minimized_conjs = pos_conjs & neg_conjs
            if minimized_conjs:
                if frozenset() in minimized_conjs:
                    self.conjuncts = set()
                    self.solved = True
                    self.drop_terms()
                    return

                pos_diff = {conj | {pos_literal} for conj in minimized_conjs}
                neg_diff = {conj | {neg_literal} for conj in minimized_conjs}

                self.conjuncts -= pos_diff
                self.conjuncts -= neg_diff
                self.conjuncts |= minimized_conjs
                self.drop_terms()

    def swallow_minimize(self):
        to_swallow = set()

        for conj in self:
            for conj2 in self:
                if len(conj2) > len(conj) and conj2.issuperset(conj):
                    to_swallow.add(conj2)

        if to_swallow:
            self.conjuncts -= to_swallow
            self.drop_terms()
            # print(f'swallowed conj ({len(self)}): {len(to_swallow)}!!!')
            # print(self.probe_dnf_mask(), self)

    def probe_dnf_mask(self):
        mask = 0
        for dnf_args in itertools.product(
                *([(False, True)] * len(self.full_terms)),
        ):
            bound_args = dict(zip(self.full_terms, dnf_args))
            result = self.apply_dnf_args(bound_args)
            mask = (mask << 1) + result
        return mask

    def calculate_dnf_true_table(self) -> 'DnfTable':
        terms = self.terms
        indexes_map = {term: index for index, term in enumerate(terms)}
        table = DnfTable(indexes_map)

        for dnf_args in itertools.product(
                *([(False, True)] * len(terms)),
        ):
            bound_args = dict(zip(terms, dnf_args))
            result = self.apply_dnf_args(bound_args)
            if result:
                table.add_record(dnf_args)
        return table

    def apply_dnf_args(self, bound_args: dict[DNF_TERM, bool]) -> bool:
        assert len(bound_args) == len(self.full_terms)

        if self.solved is not None:
            return self.solved

        for conj in self:
            conj_result = all(
                bound_args[term] ^ is_negative
                for term, is_negative in conj
            )
            if conj_result:
                return True
        return False

    def __and__(self, other: 'Dnf'):
        if self.solved is not None:
            if self.solved:
                return other
            else:
                return self
        if other.solved is not None:
            if other.solved:
                return self
            else:
                return other

        def merge_2_cnj(conj1: CONJUNCT, conj2: CONJUNCT) -> CONJUNCT | None:
            for term, is_neg in conj1:
                if (term, not is_neg) in conj2:
                    return None
            return conj1 | conj2

        merged_dnf = set()
        for conj1 in self:
            for conj2 in other:
                merged_conj = merge_2_cnj(conj1, conj2)
                if merged_conj:
                    merged_dnf.add(merged_conj)
        return Dnf(merged_dnf).minimize_dnf()

    def __or__(self, other: 'Dnf'):
        if self.solved is not None:
            if self.solved:
                return self
            else:
                return other
        if other.solved is not None:
            if other.solved:
                return other
            else:
                return self

        merged_dnf = self.conjuncts | other.conjuncts
        return Dnf(merged_dnf).minimize_dnf()

    def __xor__(self, other: 'Dnf'):
        if self.solved is not None:
            if self.solved:
                return ~other
            else:
                return other
        if other.solved is not None:
            if other.solved:
                return ~self
            else:
                return self

        xor_result = (~self & other) | (self & ~other)
        return xor_result.minimize_dnf()

    def __invert__(self):
        if self.solved is not None:
            self.solved = not self.solved
            return self

        if not len(self):
            raise Exception()

        neg_conjs = list[list[DNF_LITERAL]]()
        for conj in self:
            neg_conj = set()
            for term, is_neg in conj:
                neg_conj.add((term, not is_neg))
            neg_conjs.append(list(neg_conj))

        # iter_indexes = [0] * len(neg_conjs)
        conjs = {frozenset((literal,)) for literal in neg_conjs.pop()}
        for neg_conj in neg_conjs:
            new_conjs = set()
            for conj in conjs:
                for term, is_neg in neg_conj:
                    if (term, not is_neg) in conj:
                        continue
                    new_conjs.add(conj | {(term, is_neg)})
            conjs = new_conjs

        # current_index = 0
        # while current_index >= 0:
        #     if current_index < len(neg_conjs) - 1:
        #         current_index = len(neg_conjs) - 1
        #
        #     # construct conj
        #     conj = set()
        #     for conj_index, iter_index in enumerate(iter_indexes):
        #         try:
        #             term, is_neg = neg_conjs[conj_index][iter_index]
        #         except IndexError:
        #             print()
        #         if (term, not is_neg) in conj:
        #             # drop this conj
        #             break
        #         conj.add((term, is_neg))
        #     else:
        #         conjes.add(frozenset(conj))
        #
        #     while current_index >= 0:
        #         iter_indexes[current_index] += 1
        #         if iter_indexes[current_index] == len(neg_conjs[current_index]):
        #             iter_indexes[current_index] = 0
        #             current_index -= 1
        #         else:
        #             break
        #
        #     print(iter_indexes)

        return Dnf(conjs).minimize_dnf()


DnfTableRecord = tuple[bool, ...]


class DnfTable:
    def __init__(self, indexes_map: dict[TreeBit, int]):
        self.indexes_map = indexes_map
        self.data = list[DnfTableRecord]()

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def add_record(self, record: DnfTableRecord):
        assert len(self.indexes_map) == len(record)
        self.data.append(record)

    def merge_tables(self, other: 'DnfTable') -> Self:
        all_terms = sorted(
            set(itertools.chain(self.indexes_map, other.indexes_map)),
            key=lambda tn: int(tn.name)
        )
        new_index_map = {term: index for index, term in enumerate(all_terms)}
        new_table = DnfTable(new_index_map)

        intersect_terms = list(set(self.indexes_map) & set(other.indexes_map))
        print('New index map', len(new_index_map))
        print('Intersected terms', len(intersect_terms))

        sub_key_indexes_map = {term: index for index, term in enumerate(intersect_terms)}

        # extract sub-tables
        def extract_sub_table(table: DnfTable) -> dict[DnfTableRecord, list[dict[TreeBit, bool]]]:
            sub_table = defaultdict(list)
            free_terms = list(
                set(table.indexes_map) - set(intersect_terms)
            )
            key_indexes_map = {term: table.indexes_map[term] for term in intersect_terms}
            free_indexes_map = {term: table.indexes_map[term] for term in free_terms}

            for record in table:
                key_record = tuple(record[index] for term, index in key_indexes_map.items())
                free_record = {term: record[index] for term, index in free_indexes_map.items()}
                sub_table[key_record].append(free_record)
            return sub_table

        self_sub_table = extract_sub_table(self)
        other_sub_table = extract_sub_table(other)

        intersect_records = set(self_sub_table) & set(other_sub_table)
        for inter_record in intersect_records:
            self_free_records = self_sub_table[inter_record]
            other_free_records = other_sub_table[inter_record]

            # write intersected info
            merged_record_template: list[bool | None] = [None] * len(new_index_map)
            for term in intersect_terms:
                merged_record_template[new_index_map[term]] = inter_record[sub_key_indexes_map[term]]

            for self_free_record in self_free_records:
                # write self free info
                for term, data in self_free_record.items():
                    merged_record_template[new_index_map[term]] = data

                for other_free_record in other_free_records:
                    # write other free info
                    for term, data in other_free_record.items():
                        merged_record_template[new_index_map[term]] = data

                    assert all(data is not None for data in merged_record_template)
                    merged_record = tuple(merged_record_template)
                    new_table.add_record(merged_record)

        return new_table