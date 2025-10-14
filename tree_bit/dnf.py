import itertools
from typing import Counter

from tqdm import tqdm

from tree_bit.base import TreeBit, TreeBitAtom
from tree_bit.tools import extract_base_bits, solve_bits, get_ancestors_gen, get_solving_order

DNF_TERM = TreeBit
DNF_LITERAL = tuple[DNF_TERM, bool]
CONJUNCT = frozenset[DNF_LITERAL]
DNF = set[CONJUNCT]


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
    def get_sdnf_for_bit(cls, bit: TreeBitAtom, control_value: bool):
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

        print('DNF MASK:', dnf_mask)
        dnf_instance = cls(dnf)
        print('PROBE DNF MASK:', dnf_instance.probe_dnf_mask())

        return dnf_instance


    def minimize_dnf(self):
        used_literals = Counter[DNF_LITERAL](itertools.chain.from_iterable(self))
        assert len(used_literals.most_common()[0]) < len(self.terms)

        while True:
            rank = self.rank()
            self.sticky_minimize()
            self.swallow_minimize()
            new_rank = self.rank()
            assert new_rank <= rank
            if new_rank == rank:
                break
            print('minimization step', rank, new_rank, len(self), self.probe_dnf_mask())


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
