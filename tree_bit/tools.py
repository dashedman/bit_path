from collections import deque, Counter, defaultdict
from typing import Iterable

import tree_bit
from tree_bit.base import TreeBitAtom, TreeBit, registry, TreeBitNOT, TreeBitXOR, TreeBitAND, TreeBitOR, TreeBitOperator, \
    TreeBitMultiXor, TreeBitMultiAnd, TreeBitMultiOr, TreeBitMultiOperator


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

def get_ancestors_gen(bit: TreeBitAtom, visited: set | None = None):
    # BFS
    bits_to_lookup = deque[TreeBitAtom]()
    bits_to_lookup.append(bit)

    if visited is None:
        visited = set()

    while bits_to_lookup:
        check_bit = bits_to_lookup.popleft()
        if check_bit in visited:
            continue
        visited.add(check_bit)
        bits_to_lookup.extend(check_bit.parents)
        yield check_bit


def get_all_used_bits_gen(bits_to_check: Iterable[TreeBitAtom]):
    visited = set()
    for bit2check in bits_to_check:
        for ancestor in get_ancestors_gen(bit2check, visited=visited):
            yield ancestor


def search_for_configurations(
        root_bit: TreeBitAtom,
        configurations_counter: Counter,
        clusters_registry: dict[type[TreeBitAtom], list[int]],
        visited: set,
):

    # def searching_same_op_dfs(bit: TreeBitAtom):
    #     if bit in visited:
    #         return 0
    #     visited.add(bit)
    #
    #     same_counter = 1
    #
    #     for parent in bit.parents:
    #         if type(bit) is type(parent):
    #             same_counter += searching_same_op_dfs(parent)
    #         else:
    #             parent_class_counter = searching_same_op_dfs(parent)
    #             if parent_class_counter > 1:
    #                 clusters_registry[type(parent)].append(parent_class_counter)
    #     return same_counter
    #
    # root_cluster_count = searching_same_op_dfs(root_bit)
    # if root_cluster_count > 1:
    #     clusters_registry[type(root_bit)].append(root_cluster_count)
    # return



    # def searching_dfs(bit: TreeBitAtom):
    #     if bit in visited:
    #         return
    #     visited.add(bit)
    #
    #     match type(bit):
    #         case tree_bit.base.TreeBitNOT:
    #             if type(bit.bit) == TreeBitMultiXor:
    #                 configurations_counter['not_xor'] += 1
    #             if type(bit.bit) == TreeBitNOT:
    #                 configurations_counter['not_not'] += 1
    #             searching_dfs(bit.bit)
    #         case tree_bit.base.TreeBitXOR:
    #             searching_dfs(bit.a)
    #             searching_dfs(bit.b)
    #         case tree_bit.base.TreeBitOR:
    #             searching_dfs(bit.a)
    #             searching_dfs(bit.b)
    #         case tree_bit.base.TreeBitAND:
    #             searching_dfs(bit.a)
    #             searching_dfs(bit.b)
    #         case tree_bit.base.TreeBit:
    #             pass
    #         case _:
    #             raise Exception('unreachable')

    def count_operators_dfs(bit):
        if bit in visited:
            return
        visited.add(bit)

        conf_name: str
        match type(bit):
            case tree_bit.base.TreeBit | tree_bit.base.TreeBitNOT:
                conf_name = bit.__class__.__name__
            case _:
                conf_name = bit.__class__.__name__ + '_' + str(len(bit))
        configurations_counter[conf_name] += 1

        for parent in bit.parents:
            count_operators_dfs(parent)

    count_operators_dfs(root_bit)

def count_dfs_depth(bit: TreeBitAtom, visited: dict[TreeBitAtom, int], depth: int):
    if bit in visited:
        return 0
    visited[bit] = depth

    nodes_counter = 1
    for parent in bit.parents:
        nodes_counter += count_dfs_depth(parent, visited, depth + 1)
    return nodes_counter

def usages_map_factory(root_bits: list[TreeBitAtom], _debug_exclude=None) -> dict[TreeBitAtom, set[TreeBitAtom]]:
    usages_map = defaultdict[TreeBitAtom, set[TreeBitAtom]](set)
    visited = set()

    def usages_counter_dfs(bit: TreeBitAtom):
        if bit in visited:
            return
        visited.add(bit)

        for parent in bit.parents:
            # if _debug_exclude and parent in _debug_exclude:
            #     raise Exception('unreachable')
            usages_map[parent].add(bit)
            usages_counter_dfs(parent)
    for root_bit in root_bits:
        usages_counter_dfs(root_bit)

    return dict(usages_map)



def multi_operator_replacement(root_bits: list[TreeBitAtom]):
    root_bits_set = set(root_bits)
    usages = usages_map_factory(root_bits)
    visited = set()
    check_bit_queue = deque(root_bits)

    def replacement_dfs(bit: TreeBitAtom):
        if bit in visited:
            return None
        visited.add(bit)
        bit_type = type(bit)

        cluster = set[TreeBitAtom]()
        cluster.add(bit)

        for parent in bit.parents:
            if type(parent) == bit_type and len(usages[parent]) < 2:
                sub_cluster = replacement_dfs(parent)
                if sub_cluster:
                    cluster.update(sub_cluster)
                # else: empty cluster
            else:
                # type diff or several usages
                check_bit_queue.append(parent)
        return cluster

    def mock_cluster_to_operator(cluster_root: TreeBitAtom, cluster: set[TreeBitAtom]):
        cluster_base_type = type(cluster_root)
        if cluster_base_type is TreeBitNOT:
            return
        if cluster_base_type is TreeBit:
            return

        assert cluster_root in cluster
        # should be one type

        assert len({type(b) for b in cluster}) == 1

        operators_type_map: dict[type[TreeBitOperator], type[TreeBitMultiOperator]] = {
            TreeBitXOR: TreeBitMultiXor,
            TreeBitAND: TreeBitMultiAnd,
            TreeBitOR: TreeBitMultiOr,
        }
        if cluster_base_type not in operators_type_map:
            raise Exception('unreachable')
        multi_type = operators_type_map[cluster_base_type]

        all_inputs = set()
        for cluster_bit in cluster:
            for parent in cluster_bit.parents:
                if parent not in cluster:
                    all_inputs.add(parent)
        multi_bit = multi_type(frozenset(all_inputs), value=0.5)

        # update usages_map
        for cluster_bit in cluster:
            if cluster_bit is not cluster_root and cluster_bit in usages:
                if len(usages[cluster_bit]) >= 2:
                    raise Exception('unreachable', cluster_bit)
                del usages[cluster_bit]

        for inp in all_inputs:
            input_usages = usages[inp]
            input_usages -= cluster
            input_usages.add(multi_bit)

        if cluster_root in usages:
            root_usages = usages.pop(cluster_root)
            usages[multi_bit] = root_usages
            for usage_bit in root_usages:
                if isinstance(usage_bit, TreeBitNOT):
                    assert usage_bit.bit is cluster_root
                    usage_bit.bit = multi_bit
                elif isinstance(usage_bit, TreeBitOperator):
                    if usage_bit.a is cluster_root:
                        usage_bit.a = multi_bit
                    elif usage_bit.b is cluster_root:
                        usage_bit.b = multi_bit
                    else:
                        raise Exception('unreachable')
                elif isinstance(usage_bit, TreeBitMultiOperator):
                    args_set = set(list(usage_bit.args))
                    if cluster_root not in args_set:
                        raise Exception('unreachable')
                    args_set.remove(cluster_root)
                    args_set.add(multi_bit)
                    usage_bit.args = frozenset(args_set)
                else:
                    raise Exception('unreachable')
                # delete cached key to recalc
                if hasattr(usage_bit, 'key'):
                    del usage_bit.key
        return multi_bit

    while check_bit_queue:
        print('v', len(visited))
        check_bit = check_bit_queue.popleft()
        replacement_cluster = replacement_dfs(check_bit)
        if replacement_cluster:
            # check_bit - cluster root
            new_cluster_root = mock_cluster_to_operator(check_bit, replacement_cluster)
            if new_cluster_root is not None and check_bit in root_bits_set:
                root_bit_index = root_bits.index(check_bit)
                root_bits[root_bit_index] = new_cluster_root
                root_bits_set.remove(check_bit)
                root_bits_set.add(new_cluster_root)

            # new_usages = usages_map_factory(root_bits, _debug_exclude=replacement_cluster)
            # assert new_usages == usages

