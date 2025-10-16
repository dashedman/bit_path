from collections import deque, Counter
from typing import Iterable

import tree_bit
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

        # 20-21 it/sec
        match type(bit_to_solve):
            case tree_bit.base.TreeBitNOT:
                solve = not secondary_solves[bit_to_solve.bit]
            case tree_bit.base.TreeBitXOR:
                solve = secondary_solves[bit_to_solve.a] ^ secondary_solves[bit_to_solve.b]
            case tree_bit.base.TreeBitOR:
                solve = secondary_solves[bit_to_solve.a] or secondary_solves[bit_to_solve.b]
            case tree_bit.base.TreeBitAND:
                solve = secondary_solves[bit_to_solve.a] and secondary_solves[bit_to_solve.b]
            case _:
                raise Exception('unreachable')

        secondary_solves[bit_to_solve] = solve

        # 13 it/sec
        # if isinstance(bit_to_solve, TreeBitNOT):
        #     secondary_solves[bit_to_solve] = not secondary_solves[bit_to_solve.bit]
        # elif isinstance(bit_to_solve, TreeBitOperator):
        #     if isinstance(bit_to_solve, TreeBitXOR):
        #         solve = secondary_solves[bit_to_solve.a] ^ secondary_solves[bit_to_solve.b]
        #     elif isinstance(bit_to_solve, TreeBitOR):
        #         solve = secondary_solves[bit_to_solve.a] or secondary_solves[bit_to_solve.b]
        #     elif isinstance(bit_to_solve, TreeBitAND):
        #         solve = secondary_solves[bit_to_solve.a] and secondary_solves[bit_to_solve.b]
        #     else:
        #         raise Exception('unreachable')
        #
        #     secondary_solves[bit_to_solve] = solve
        #
        # else:
        #     raise Exception('unreachable')

    return secondary_solves[solving_order[-1]]   # return result from last order


def solve_bits_dfs(
        secondary_solves: dict[TreeBitAtom, bool],
        bit: TreeBitAtom,
):

    if solve := secondary_solves.get(bit) is not None:
        # memorisation cache
        return solve

    # 2 it/s
    # if isinstance(bit, TreeBitOperator):
    #     if isinstance(bit, TreeBitXOR):
    #         solve = dfs_func(bit.a) ^ dfs_func(bit.b)
    #     elif isinstance(bit, TreeBitOR):
    #         solve = dfs_func(bit.a) or dfs_func(bit.b)
    #     elif isinstance(bit, TreeBitAND):
    #         solve = dfs_func(bit.a) and dfs_func(bit.b)
    #     else:
    #         raise Exception('unreachable')
    #
    # elif isinstance(bit, TreeBitNOT):
    #     solve = not dfs_func(bit.bit)
    # else:
    #     raise Exception('unreachable')

    # 19 it/sec
    match type(bit):
        case tree_bit.base.TreeBitNOT:
            solve = not solve_bits_dfs(secondary_solves, bit.bit)
        case tree_bit.base.TreeBitXOR:
            solve = solve_bits_dfs(secondary_solves, bit.a) ^ solve_bits_dfs(secondary_solves, bit.b)
        case tree_bit.base.TreeBitOR:
            solve = solve_bits_dfs(secondary_solves, bit.a) or solve_bits_dfs(secondary_solves, bit.b)
        case tree_bit.base.TreeBitAND:
            solve = solve_bits_dfs(secondary_solves, bit.a) and solve_bits_dfs(secondary_solves, bit.b)
        case _:
            raise Exception('unreachable')

    # 13 it/sec
    # match type(bit).__name__:
    #     case tree_bit.base.TreeBitNOT.__name__:
    #         solve = not dfs_func(bit.bit)
    #     case tree_bit.base.TreeBitXOR.__name__:
    #         solve = dfs_func(bit.a) ^ dfs_func(bit.b)
    #     case tree_bit.base.TreeBitOR.__name__:
    #         solve = dfs_func(bit.a) or dfs_func(bit.b)
    #     case tree_bit.base.TreeBitAND.__name__:
    #         solve = dfs_func(bit.a) and dfs_func(bit.b)
    #     case _:
    #         raise Exception('unreachable')

    secondary_solves[bit] = solve
    return solve

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


def search_for_configurations(
        root_bit: TreeBitAtom,
        configurations_counter: Counter,
        clusters_registry: dict[type[TreeBitAtom], list[int]],
        visited: set,
):

    def searching_same_op_dfs(bit: TreeBitAtom):
        if bit in visited:
            return 0
        visited.add(bit)

        same_counter = 1

        for parent in bit.parents:
            if type(bit) is type(parent):
                same_counter += searching_same_op_dfs(parent)
            else:
                parent_class_counter = searching_same_op_dfs(parent)
                if parent_class_counter > 1:
                    clusters_registry[type(parent)].append(parent_class_counter)
        return same_counter

    root_cluster_count = searching_same_op_dfs(root_bit)
    if root_cluster_count > 1:
        clusters_registry[type(root_bit)].append(root_cluster_count)
    return



    def searching_dfs(bit: TreeBitAtom):
        if bit in visited:
            return
        visited.add(bit)

        match type(bit):
            case tree_bit.base.TreeBitNOT:
                if type(bit.bit) == TreeBitXOR:
                    configurations_counter['not_xor'] += 1
                if type(bit.bit) == TreeBitNOT:
                    configurations_counter['not_not'] += 1
                searching_dfs(bit.bit)
            case tree_bit.base.TreeBitXOR:
                searching_dfs(bit.a)
                searching_dfs(bit.b)
            case tree_bit.base.TreeBitOR:
                searching_dfs(bit.a)
                searching_dfs(bit.b)
            case tree_bit.base.TreeBitAND:
                searching_dfs(bit.a)
                searching_dfs(bit.b)
            case tree_bit.base.TreeBit:
                pass
            case _:
                raise Exception('unreachable')

    searching_dfs(root_bit)
    return configurations_counter

def count_dfs_depth(bit: TreeBitAtom, visited: dict[TreeBitAtom, int], depth: int):
    if bit in visited:
        return 0
    visited[bit] = depth

    nodes_counter = 1
    for parent in bit.parents:
        nodes_counter += count_dfs_depth(parent, visited, depth + 1)
    return nodes_counter
