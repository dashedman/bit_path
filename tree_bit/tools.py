from collections import deque

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


def solve_bits(base_bits_inputs: dict[TreeBit, bool], exit_bit: TreeBitAtom) -> bool:
    secondary_solves: dict[TreeBitAtom, bool] = base_bits_inputs.copy()

    necessary_bits = set(get_ancestors_gen(exit_bit))

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
                continue    # skip

            if usages_bit in secondary_solves:
                # print('solve_bits: already solved!')
                continue

            if isinstance(usages_bit, TreeBitNOT):
                secondary_solves[usages_bit] = not solved_bit.value
                solved_bits_queue.append(usages_bit)
            elif isinstance(usages_bit, TreeBitOperator):
                if usages_bit.a not in secondary_solves:
                    continue
                if usages_bit.b not in secondary_solves:
                    continue

                if isinstance(usages_bit, TreeBitXOR):
                    solve = secondary_solves[usages_bit.a] ^ secondary_solves[usages_bit.b]
                elif isinstance(usages_bit, TreeBitOR):
                    solve = secondary_solves[usages_bit.a] or secondary_solves[usages_bit.b]
                elif isinstance(usages_bit, TreeBitAND):
                    solve = secondary_solves[usages_bit.a] and secondary_solves[usages_bit.b]
                else:
                    raise Exception('unreachable')

                secondary_solves[usages_bit] = solve
                solved_bits_queue.append(usages_bit)

            else:
                raise Exception('unreachable')

        if exit_bit in secondary_solves:
            return secondary_solves[exit_bit]
    raise Exception('unsolvable')


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