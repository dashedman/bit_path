import itertools
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
        self._dnf_table = None

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

        def merge_2_cnj(conj1: CONJUNCT, conj2: CONJUNCT) -> CONJUNCT | None:
            for term, is_neg in conj1:
                if (term, not is_neg) in conj2:
                    return None
            return conj1 | conj2

        def merge_2_dnf(dnf1: DNF, dnf2: DNF):
            merged_dnf = set()
            for conj1 in dnf1:
                for conj2 in dnf2:
                    merged_conj = merge_2_cnj(conj1, conj2)
                    if merged_conj:
                        merged_dnf.add(merged_conj)
            return merged_dnf

        sub_heap_for_bit = {}
        def get_logical_heap(bit: TreeBitAtom):
            if bit in sub_heap_for_bit:
                return sub_heap_for_bit[bit]

            match type(bit):
                case tree_bit.base.TreeBit:
                    heap = (LogicalIdentity, bit)
                case tree_bit.base.TreeBitNOT:
                    heap = (LogicalNot, get_logical_heap(bit.bit))
                case tree_bit.base.TreeBitAND:
                    heap = (LogicalAnd, get_logical_heap(bit.a), get_logical_heap(bit.b))
                case tree_bit.base.TreeBitOR:
                    heap = (LogicalOr, get_logical_heap(bit.a), get_logical_heap(bit.b))
                case tree_bit.base.TreeBitXOR:
                    heap = (
                        LogicalOr,
                        (
                            LogicalAnd,
                            (LogicalNot, get_logical_heap(bit.a)),
                            get_logical_heap(bit.b)
                        ),
                        (
                            LogicalAnd,
                            get_logical_heap(bit.a),
                            (LogicalNot, get_logical_heap(bit.b))
                        )
                    )
            sub_heap_for_bit[bit] = heap
            return heap

        logical_heap = get_logical_heap(exit_bit)
        del sub_heap_for_bit

        dnf_for_heap = {}
        def logical_heap_to_dnf(heap_head) -> DNF:
            heap_op = heap_head[0]
            match heap_op:
                case 0: # Identity
                    # simple DNF
                    return {frozenset({(heap_head[1], False)})}
                case 1: # Not
                    next_heap = heap_head[1]
                    next_command = next_heap[0]
                    next_args_len = len(next_heap) - 1
                    if next_args_len == 1:
                        next_arg = next_heap[1]
                        if next_command == 0:   # not one
                            return {frozenset({(heap_head[1], True)})}
                        elif next_command == 1:   # not not
                            return logical_heap_to_dnf(next_arg)
                        else:
                            raise Exception('Unreachble')
                    else:
                        if next_command == 2:
                            new_command = 3
                        else:
                            new_command = 2
                        next_args = next_heap[1:]
                        return logical_heap_to_dnf((
                            new_command,
                            *((LogicalNot, arg) for arg in next_args)
                        ))
                case 2: # and
                    heap_args = heap_head[1:]
                    dnf_from_args = []
                    for heap_arg in heap_args:
                        dnf_from_args.append(logical_heap_to_dnf(heap_arg))

                    merged_dnf = dnf_from_args.pop()
                    for dnf in dnf_from_args:
                        merged_dnf = merge_2_dnf(merged_dnf, dnf)
                    return merged_dnf
                case 3: # or
                    heap_args = heap_head[1:]
                    dnf_from_args = []
                    for heap_arg in heap_args:
                        dnf_from_args.append(logical_heap_to_dnf(heap_arg))

                    merged_dnf = dnf_from_args.pop()
                    for dnf in dnf_from_args:
                        merged_dnf |= dnf
                    return merged_dnf

        dnf = logical_heap_to_dnf(logical_heap)
        dnf_instance = cls(dnf)
        return dnf_instance

    def minimize_dnf(self):
        used_literals = Counter[DNF_LITERAL](itertools.chain.from_iterable(self))
        assert len(used_literals.most_common()[0]) < len(self.terms)

        while True:
            length = len(self)
            rank = self.rank()
            terms_count = len(self.terms)
            mask = self.probe_dnf_mask()

            self.sticky_minimize()
            self.swallow_minimize()

            new_length = len(self)
            new_rank = self.rank()
            new_terms_count = len(self.terms)
            new_mask = self.probe_dnf_mask()

            assert new_rank <= rank
            if new_rank == rank:
                break
            print(
                f'Minimization step '
                f'({terms_count}, {length}, {rank}) -> ({new_terms_count}, {new_length}, {new_rank}) '
                f'mask_same: {mask == new_mask}, '
                f'mask: {new_mask}'
            )
            assert mask == new_mask

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

        for conj in self:
            conj_result = all(
                bound_args[term] ^ is_negative
                for term, is_negative in conj
            )
            if conj_result:
                return True
        return False


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