import itertools
from typing import Counter

from tree_bit.base import TreeBit, TreeBitAtom
from tree_bit.tools import extract_base_bits, solve_bits

DNF_TERM = TreeBit
DNF_LITERAL = tuple[DNF_TERM, True]
CONJUNCT = frozenset[DNF_LITERAL]
DNF = set[CONJUNCT]


def str_dnf(dnf: DNF):
    dnf_str = f'Dnf ({len(dnf)}, {max(len(dsj) for dsj in dnf) if dnf else 0}): '
    for dsjn in dnf:
        dsjn_str = ''
        for unit, is_negate in dsjn:
            if is_negate:
                unit_str = f'~{unit.name}'
            else:
                unit_str = f'{unit.name}'

            dsjn_str += unit_str + ' '
        dnf_str += '(' + dsjn_str.rstrip(' *') + ') + '

    dnf_str = dnf_str.rstrip(' +')
    return dnf_str


def get_sdnf_for_bit(bit: TreeBitAtom, control_value: bool) -> DNF:
    base_bits = extract_base_bits(bit)
    disjunctions_result_map: dict[tuple[bool, ...], bool] = {}

    for disjunction_values in itertools.product(
            *([(False, True)] * len(base_bits)),
    ):
        solve_map = dict(zip(base_bits, disjunction_values))
        solving = solve_bits(solve_map, bit)
        print(f'Disjunction {solving}={disjunction_values}')
        disjunctions_result_map[tuple(disjunction_values)] = solving

    dnf = set()

    for disjunction_values, solving in disjunctions_result_map.items():
        if control_value == solving:
            disjunction_unit = set()
            for base_bit, value in zip(base_bits, disjunction_values):
                disjunction_unit.add((base_bit, not value))
            dnf.add(frozenset(disjunction_unit))

    return dnf


def minimize_dnf(dnf: DNF):
    dnf = dnf.copy()

    terms = { term for conj in dnf for term, _ in conj }

    used_literals = Counter[DNF_LITERAL](itertools.chain.from_iterable(dnf))
    assert len(used_literals.most_common()[0]) < len(terms)

    sticky_minimize(terms, dnf)
    swallow_minimize(dnf)

    return dnf


def sticky_minimize(terms: set[DNF_TERM], dnf_mut: DNF):
    for term in terms:
        pos_literal = (term, False)
        neg_literal = (term, True)

        pos_conjs = set()
        neg_conjs = set()

        for conj in dnf_mut:
            if pos_literal in conj:
                pos_conjs.add(conj - {pos_literal})

            if neg_literal in conj:
                neg_conjs.add(conj - {neg_literal})

        minimized_conjs = pos_conjs & neg_conjs
        if minimized_conjs:

            dnf_mut.update(minimized_conjs)
            dnf_mut.difference_update(conj | {pos_literal} for conj in (pos_conjs - minimized_conjs))
            dnf_mut.difference_update(conj | {neg_literal} for conj in (neg_conjs - minimized_conjs))
            print(f'minimized ({len(dnf_mut)}) !!!', len(minimized_conjs), str_dnf(minimized_conjs))
            print('pos_conjs', str_dnf(pos_conjs - minimized_conjs))
            print('neg_conjs', str_dnf(neg_conjs - minimized_conjs))
            print(str_dnf(dnf_mut))


def swallow_minimize(dnf_mut: DNF):
    to_swallow = set()

    for conj in dnf_mut:
        for conj2 in dnf_mut:
            if len(conj2) > len(conj) and conj2.issuperset(conj):
                to_swallow.add(conj2)

    if to_swallow:
        dnf_mut -= to_swallow
        print(f'swallowed conj ({len(dnf_mut)}): {len(to_swallow)}!!!')
        print(str_dnf(dnf_mut))


