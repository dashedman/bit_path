from collections import deque
from typing import Iterable

from tree_bit.base import TreeBitAtom, TreeBit, registry, TreeBitNOT, TreeBitXOR, TreeBitAND, TreeBitOR, TreeBitOperator


def extract_base_bits(final_bit: TreeBitAtom) -> list[TreeBit]:
    # BFS
    bits_to_check = deque()
    bits_to_check.appendleft(final_bit)
    result_bits = []
    visited = set()
    while bits_to_check:
        bit_check = bits_to_check.pop()

        if bit_check in visited:
            continue

        visited.add(bit_check)


        if isinstance(bit_check, TreeBit):
            result_bits.append(bit_check)
        else:
            bits_to_check.extend(bit_check.parents)
    return result_bits


def get_solving_order(
        base_bits_inputs: Iterable[TreeBit],
        exit_bit: TreeBitAtom,
        necessary_bits: set[TreeBitAtom],
) -> list[TreeBitAtom]:
    solved: set[TreeBitAtom] = set(base_bits_inputs)
    solving_order = []

    solved_bits_queue = deque[TreeBitAtom]()
    solved_bits_queue.extend(base_bits_inputs)

    while solved_bits_queue:
        solved_bit = solved_bits_queue.popleft()
        bit_registry = registry[solved_bit.key]
        assert bit_registry.bit == solved_bit

        usages_bits = [
            registry[usage].bit for usage in bit_registry.usages
        ]
        for usages_bit in usages_bits:
            if usages_bit not in necessary_bits:
                continue  # skip

            if usages_bit in solved:
                # print('solve_bits: already solved!')
                continue

            if isinstance(usages_bit, TreeBitNOT):
                solved.add(usages_bit)
                solving_order.append(usages_bit)
                solved_bits_queue.append(usages_bit)
            elif isinstance(usages_bit, TreeBitOperator):
                if usages_bit.a not in solved:
                    continue
                if usages_bit.b not in solved:
                    continue

                solved.add(usages_bit)
                solving_order.append(usages_bit)
                solved_bits_queue.append(usages_bit)

            else:
                raise Exception('unreachable')

        if exit_bit in solved:
            return solving_order
    raise Exception('unsolvable')


def solve_bits(
        base_bits_inputs: dict[TreeBit, bool],
        solving_order: list[TreeBitAtom],
) -> bool:
    secondary_solves: dict[TreeBitAtom, bool] = base_bits_inputs.copy()

    for bit_to_solve in solving_order:
        if isinstance(bit_to_solve, TreeBitNOT):
            secondary_solves[bit_to_solve] = not secondary_solves[bit_to_solve.bit]
        elif isinstance(bit_to_solve, TreeBitOperator):
            if bit_to_solve.a not in secondary_solves:
                continue
            if bit_to_solve.b not in secondary_solves:
                continue

            if isinstance(bit_to_solve, TreeBitXOR):
                solve = secondary_solves[bit_to_solve.a] ^ secondary_solves[bit_to_solve.b]
            elif isinstance(bit_to_solve, TreeBitOR):
                solve = secondary_solves[bit_to_solve.a] or secondary_solves[bit_to_solve.b]
            elif isinstance(bit_to_solve, TreeBitAND):
                solve = secondary_solves[bit_to_solve.a] and secondary_solves[bit_to_solve.b]
            else:
                raise Exception('unreachable')

            secondary_solves[bit_to_solve] = solve

        else:
            raise Exception('unreachable')

    return secondary_solves[solving_order[-1]]   # return result from last order


def get_ancestors_gen(bit: TreeBitAtom):
    # BFS
    bits_to_lookup = deque[TreeBitAtom]()
    bits_to_lookup.append(bit)

    visited = set()

    while bits_to_lookup:
        check_bit = bits_to_lookup.popleft()
        if check_bit in visited:
            continue
        visited.add(check_bit)
        bits_to_lookup.extend(check_bit.parents)
        yield check_bit