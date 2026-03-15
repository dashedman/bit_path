import itertools
import random
import time
from collections import defaultdict, Counter, deque
from pprint import pprint
from typing import Iterable, Self

from tqdm import tqdm

import tree_bit
from tree_bit.base import TreeBit, TreeBitAtom
from tree_bit.tools import count_all_operators

# bit mask where bit on 'n's place is existing of 'n's bit from hash in conjuctioon
ANF_BIN_CONJUNCT = int
ANF_BIN = set[ANF_BIN_CONJUNCT]

build_time = 0
sort_time = 0
delete_time = 0
skip_counter = 0
round_time = []
round_inside_time = []
choose_time = []
cover_time = []
scan_time = []
max_stat = []


class BinaryAnf:
    bits_map: list[TreeBit | tuple[Self, Self]] = []
    """
    Zhegalkin polynomials
    """
    def __init__(self, anf: ANF_BIN, inversed_by_one: bool = False):
        self.conjuncts: ANF_BIN = anf
        self.inversed_by_one = inversed_by_one

    def __len__(self):
        return len(self.conjuncts) + self.inversed_by_one

    def __iter__(self):
        return iter(self.conjuncts)

    def __str__(self):
        solved = self.solved
        if solved is not None:
            anf_str = f'BAnf ({solved})'
        else:
            anf_str = f'BAnf ({len(self)}, {max(conj.bit_count() for conj in self) if self else 0})'
        return anf_str

    def full_str(self):
        full_conjes_iter = ('(' + self._str_conj(conj) + ')' for conj in self)
        if self.inversed_by_one:
            full_conjes_iter = itertools.chain(full_conjes_iter, ('ONE',))
        return str(self) + '{' + ' ⊕ '.join(sorted(full_conjes_iter)) + '}'

    def _str_conj(self, conj: ANF_BIN_CONJUNCT):
        return ' & '.join(
            sorted(
                bit.name if isinstance(bit, TreeBitAtom) else str(conj.bit_length() - 1)
                for bit in self._conj_bit_iter(conj)
            )
        )

    def _conj_idx_iter(self, conj: ANF_BIN_CONJUNCT):
        idx = 0
        while conj:
            bit = conj & 1

            if bit:
                # bit in conjuction
                yield idx

            idx += 1
            conj >>= 1

    def _conj_bit_iter(self, conj: ANF_BIN_CONJUNCT):
        for idx in self._conj_idx_iter(conj):
            yield self.bits_map[idx]

    @property
    def solved(self):
        if self.conjuncts:
            return None
        return self.inversed_by_one

    def rank(self):
        return sum(map(lambda x: x.bit_count(), self))

    @classmethod
    def get_anf_for_bit(
            cls,
            exit_bit: TreeBitAtom,
            anf_for_bit: dict[TreeBitAtom, 'BinaryAnf'] | None = None
    ):
        if anf_for_bit is None:
            anf_for_bit = {}

        reversed_bits_map = {bit: idx for idx, bit in enumerate(cls.bits_map)}

        counter = 0
        depth_counter = 0
        max_depth = 0
        estimated = count_all_operators(exit_bit)

        cache_hit_counter = 0
        cache_miss_counter = 0

        calc_stats = defaultdict(list)

        pbar = tqdm(total=estimated, initial=len(anf_for_bit))

        def check_cache(bit):
            return anf_for_bit.get(bit)

        def anf_by_bit_dfs(
            bit: TreeBitAtom,
        ) -> BinaryAnf:
            nonlocal depth_counter
            nonlocal counter
            nonlocal cache_hit_counter
            nonlocal cache_miss_counter
            nonlocal max_depth

            time_start = time.perf_counter()
            if (anf := check_cache(bit)) is not None:
                # memorisation cache
                cache_hit_counter += 1
                time_end = time.perf_counter()
                calc_stats['cache'].append(time_end - time_start)
                return anf.copy()
            cache_miss_counter += 1

            depth_counter += 1
            max_depth = max(depth_counter, max_depth)
            # ??? it/sec
            match type(bit):
                case tree_bit.base.TreeBitNOT:
                    bit: tree_bit.base.TreeBitNOT
                    anf_prev = anf_by_bit_dfs(bit.bit)
                    n = bit.cls_name
                    time_start = time.perf_counter()
                    anf = ~anf_prev
                case tree_bit.base.TreeBitXOR:
                    bit: tree_bit.base.TreeBitXOR
                    anf_a = anf_by_bit_dfs(bit.a)
                    anf_b = anf_by_bit_dfs(bit.b)
                    n = bit.cls_name
                    time_start = time.perf_counter()
                    anf = anf_a ^ anf_b
                case tree_bit.base.TreeBitOR:
                    bit: tree_bit.base.TreeBitOR
                    anf_a = anf_by_bit_dfs(bit.a)
                    anf_b = anf_by_bit_dfs(bit.b)
                    n = bit.cls_name
                    time_start = time.perf_counter()
                    anf = anf_a | anf_b
                case tree_bit.base.TreeBitAND:
                    bit: tree_bit.base.TreeBitAND
                    anf_a = anf_by_bit_dfs(bit.a)
                    anf_b = anf_by_bit_dfs(bit.b)
                    n = bit.cls_name
                    time_start = time.perf_counter()
                    anf = anf_a & anf_b
                case tree_bit.base.TreeBit:
                    bit: TreeBit
                    n = 'bit'
                    time_start = time.perf_counter()

                    idx = reversed_bits_map[bit]
                    int_bit = 1 << idx
                    anf = cls.from_explicit_bit(int_bit)
                case _:
                    raise Exception('unreachable')
            time_end = time.perf_counter()
            elapsed = time_end - time_start
            calc_stats[n].append(elapsed)

            counter += 1
            pbar.update(1)

            anf_for_bit[bit] = anf.copy()

            # if counter == 85 :
            #     print()
            #
            # if counter > 0:
            #     sanf = str(anf)
            #     print(counter, sanf[sanf.find('{'):])
            # if counter > 500:
            #     depth_counter = 0
            #     raise Exception('=)')

            # print(str(anf))
            depth_counter -= 1
            return anf

        def print_state():
            started_at = time.perf_counter()
            time.sleep(0.01)
            prev = 0
            prev_time = started_at
            to_sleep = 5
            while depth_counter > 0:
                time.sleep(to_sleep)
                curr = time.perf_counter()
                speed = (counter - prev) / (curr - prev_time)
                elapsed = int(curr - started_at)
                if speed == 0:
                    estimated_time = 10000000000
                else:
                    estimated_time = int((estimated - counter) / speed)
                prev = counter
                prev_time = curr
                print(
                    f'{counter}/{estimated}, elapsed: {elapsed // 60}:{elapsed % 60:.2f}, {speed:.2f} it/s, '
                    f'estimated: {estimated_time // 60}:{estimated_time % 60:.2f}, '
                    f'Cache (hit/miss): {cache_hit_counter}/{cache_miss_counter}, maxd: {max_depth}'
                )
                pprint(sorted(((g, max(i), sum(i), len(i), sum(i) / len(i)) for g, i in calc_stats.items()), key=lambda t: t[2], reverse=True))
                to_sleep = min(estimated_time, to_sleep)


        # stat = Thread(
        #     target=print_state,
        # )
        # stat.start()
        result = anf_by_bit_dfs(exit_bit)
        # stat.join()

        pbar.close()

        return result

    @classmethod
    def from_explicit_bit(cls, bit: int, inversed_by_one: bool = False):
        assert bit.bit_count() == 1
        return cls({bit}, inversed_by_one=inversed_by_one)

    def count_terms_usage(self):
        counter = Counter()
        for conj in self:
            for literal in self._conj_bit_iter(conj):
                counter[literal] += 1
        return counter

    def count_terms_idx_usage(self):
        counter = Counter()
        for conj in self:
            for idx in self._conj_idx_iter(conj):
                counter[idx] += 1
        return counter

    def all_terms_idx_usage(self):
        conj_mask = 0
        for conj in self:
            conj_mask |= conj
        return list(self._conj_idx_iter(conj_mask))

    def __and__(self, other: 'BinaryAnf'):
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

        new_var = len(self.bits_map)
        self.bits_map.append((self, other))
        new_var_mask = 1 << new_var
        anf = {new_var_mask}
        return BinaryAnf(anf)

        # counter = Counter()
        #
        # common_anf = self.conjuncts & other.conjuncts
        # least_self = self.conjuncts - common_anf
        # least_other = other.conjuncts - common_anf
        #
        # merged_inversed_by_one = self.inversed_by_one and other.inversed_by_one
        # if self.inversed_by_one ^ other.inversed_by_one:
        #     merged_anf = set()
        # else:
        #     merged_anf = common_anf.copy()
        #
        # if self.inversed_by_one:
        #     merged_anf.symmetric_difference_update(least_other)
        # if other.inversed_by_one:
        #     merged_anf.symmetric_difference_update(least_self)
        #
        # # counter_check = Counter()
        # # # experimental
        # # t5 = time.perf_counter()
        # counter.update(self._branch_xor_and(common_anf, least_self))
        # # t6 = time.perf_counter()
        # # counter_check.update(conj_m | conj1 for conj1 in least_self for conj_m in common_anf)
        # # t7 = time.perf_counter()
        # # plain = t7 - t6
        # # branch = t6 - t5
        # # if plain > branch > 0.0001:
        # #     print(f'{plain} {branch} {len(least_self)} {len(common_anf)} {len(least_self) + len(common_anf)} {len(least_self) * len(common_anf)}')
        # #
        # # t5 = time.perf_counter()
        # counter.update(self._branch_xor_and(common_anf, least_other))
        # # t6 = time.perf_counter()
        # # counter_check.update(conj_m | conj2 for conj2 in least_other for conj_m in common_anf)
        # # t7 = time.perf_counter()
        # # plain = t7 - t6
        # # branch = t6 - t5
        # # if plain > branch > 0.0001:
        # #     print(f'{plain} {branch} {len(common_anf)} {len(least_other)} {len(common_anf) + len(least_other)} {len(common_anf) * len(least_other)}')
        #
        # counter.update(self._branch_xor_and(least_self, least_other))
        # # counter_check.update(conj1 | conj2 for conj2 in least_other for conj1 in least_self)
        # # plain = t7 - t6
        # # branch = t6 - t5
        # # if plain > branch > 0.0001:
        # #     print(f'{plain} {branch} {len(least_self)} {len(least_other)} {len(least_self) + len(least_other)} {len(least_self) * len(least_other)}')
        #
        #
        #
        # # common
        # # max_stat.append(max(
        # #     len(common_anf) * len(least_self),
        # #     len(common_anf) * len(least_other),
        # #     len(least_self) * len(least_other),
        # # ))
        # # for conj_m in common_anf:
        # #     counter.update(conj_m | conj1 for conj1 in least_self)
        # #     counter.update(conj_m | conj2 for conj2 in least_other)
        # # t5 = time.perf_counter()
        # # counter.update(conj1 | conj2 for conj2 in least_other for conj1 in least_self)
        # #
        # # if counter_check != counter:
        # #     print()
        #
        # merged_anf.symmetric_difference_update(
        #     item for item, count in counter.items() if count & 1
        # )
        #
        # # te = time.perf_counter()
        # # el = te - t0
        #
        # # if el > 10:
        # #     print(f'{el:.3f}: {t1 - t0:.3f} {t2 - t1:.3f} {t3 - t2:.3f} {t4 - t3:.3f} {t5 - t4:.3f} {te - t5:.3f}')
        # #     print()
        # self.conjuncts = merged_anf
        # self.inversed_by_one = merged_inversed_by_one
        #
        # return self

    @staticmethod
    def lob(sub_anf_a: ANF_BIN, sub_anf_b: ANF_BIN):
        products_counter = Counter()
        if not sub_anf_a or not sub_anf_b:
            return products_counter

        for conj_a in sub_anf_a:
            for conj_b in sub_anf_b:
                production = conj_a | conj_b
                products_counter[production] += 1
        return products_counter

    @classmethod
    def _tt_xor_and(cls, sub_anf_a: ANF_BIN, sub_anf_b: ANF_BIN):
        all_conjes = sub_anf_a | sub_anf_b
        max_arg_index = max(conj.bit_length() for conj in all_conjes)
        frequency_counter = [0] * max_arg_index

        # count args usages and mask
        bits_mask_a = 0
        for conj_a in sub_anf_a:
            bits_mask_a |= conj_a
            idx = 0
            while conj_a:
                if conj_a & 1:
                    frequency_counter[idx] += 1
                idx += 1
                conj_a >>= 1
        # count args usages
        for conj_b in sub_anf_b:
            idx = 0
            while conj_b:
                if conj_b & 1:
                    frequency_counter[idx] += 1
                idx += 1
                conj_b >>= 1

        frequency_map = []
        for idx, frequency in enumerate(frequency_counter):
            bit_mask = 1 << idx
            is_in_a = bit_mask & bits_mask_a
            frequency_map.append((idx, bit_mask, frequency, is_in_a))
        frequency_stack = [
            (idx, bit_mask)
            for idx, bit_mask, _, _ in sorted(frequency_map, key=lambda x: x[3] * 10000000 + x[2])
        ]

        return cls._tt_xor_and_frequency(sub_anf_a, sub_anf_b, frequency_stack)

    @staticmethod
    def _tt_xor_and_plain(sub_anf_a: ANF_BIN, sub_anf_b: ANF_BIN):
        raise NotImplementedError
        args_mask_a = 0
        for conj_a in sub_anf_a:
            args_mask_a |= conj_a6

        # compress sub_anf_a with mapping
        # gather bit indexes
        args_mask_a_copy = args_mask_a
        index = 0
        index_map = []
        while args_mask_a_copy:
            if args_mask_a_copy & 1:
                index_map.append(index)
            index += 1
            args_mask_a_copy >>= 1
        # do compress

    @staticmethod
    def _tt_xor_and_frequency(sub_anf_a: ANF_BIN, sub_anf_b: ANF_BIN, frequency_stack: list[tuple[int, int]]) -> ANF_BIN:
        solve: list[bool | None] = [None] * len(frequency_stack)
        one_state_a = False
        one_state_b = False
        true_table: list[tuple[bool | None, ...]] = []

        def solve_as_zero(anf_to_solve: ANF_BIN, bit_to_solve: int) -> ANF_BIN:
            subanf = anf_to_solve.copy()
            conjest_to_remove = list()
            for conj in subanf:
                if bit_to_solve & conj:
                    conjest_to_remove.append(conj)
            subanf.difference_update(conjest_to_remove)
            return subanf

        def solve_as_one(anf_to_solve: ANF_BIN, bit_to_solve: int) -> tuple[ANF_BIN, bool]:
            subanf = anf_to_solve.copy()
            conjest_to_remove = list()
            short_conjes_counter = Counter()
            for conj in subanf:
                if bit_to_solve & conj:
                    conjest_to_remove.append(conj)
                    short_conjes_counter[conj ^ bit_to_solve] += 1

            subanf.difference_update(conjest_to_remove)

            if short_conjes_counter[0] % 2 == 1:
                # check conjes with zero lenght (empty conjes has ONE)
                one_flag = True
                del short_conjes_counter[0]
            else:
                one_flag = False

            subanf.symmetric_difference_update(
                map(
                    lambda sc: sc[0],
                    filter(lambda sc: sc[1] % 2 == 1, short_conjes_counter.items())
                )
            )
            return subanf, one_flag

        def prepare_b_anf():
            prepared_anf_b = sub_anf_b.copy()

            # remove all conjes with solved zeros
            zero_solved_bit_mask = 0
            for idx, bit_solve in enumerate(solve):
                if bit_solve is False:
                    zero_solved_bit_mask |= 1 << idx
            conjest_to_remove = []
            for conj_b in prepared_anf_b:
                if zero_solved_bit_mask & conj_b:
                    conjest_to_remove.append(conj_b)
            prepared_anf_b.difference_update(conjest_to_remove)

            # simplify all conjes with solved ones
            one_solved_bit_mask = 0
            for idx, bit_solve in enumerate(solve):
                if bit_solve is True:
                    one_solved_bit_mask |= 1 << idx
            conjest_to_remove = []
            short_conjes_counter = Counter()
            for conj in prepared_anf_b:
                # has at least one interception
                if interception_mask := one_solved_bit_mask & conj:
                    # mark to remove old conj
                    conjest_to_remove.append(conj)
                    # erase to zero all args that has interception
                    short_conjes_counter[conj ^ interception_mask] += 1
            prepared_anf_b.difference_update(conjest_to_remove)

            if short_conjes_counter[0] % 2 == 1:
                # check conjes with zero lenght (empty conjes has ONE)
                one_flag = True
                del short_conjes_counter[0]
            else:
                one_flag = False

            prepared_anf_b.symmetric_difference_update(
                map(
                    lambda sc: sc[0],
                    filter(lambda sc: sc[1] % 2 == 1, short_conjes_counter.items())
                )
            )
            return prepared_anf_b, one_flag

        def dfs_solve_a(prev_anf: ANF_BIN):
            nonlocal one_state_a
            nonlocal one_state_b
            if not prev_anf:
                # not has any conjuctions with args
                # check one state
                if one_state_a:
                    # solved as one
                    # time to solve b
                    # construct anf b with solved bits
                    prepared_anf_b, one_flag = prepare_b_anf()
                    one_state_b = one_flag
                    dfs_solve_b(prepared_anf_b)
                return

            most_common_bit_idx, most_common_bit_mask = frequency_stack.pop()

            # solve as zero
            solve[most_common_bit_idx] = False
            subanf = solve_as_zero(prev_anf, most_common_bit_mask)
            dfs_solve_a(subanf)

            # solve as one
            solve[most_common_bit_idx] = True
            subanf, one_flag = solve_as_one(prev_anf, most_common_bit_mask)
            if one_flag:
                # revert status of 1 conjunction
                one_state_a = not one_state_a
            dfs_solve_a(subanf)

            # reset state
            solve[most_common_bit_idx] = None
            if one_flag:
                # return status of 1 conjunction
                one_state_a = not one_state_a
            frequency_stack.append((most_common_bit_idx, most_common_bit_mask))

        def dfs_solve_b(prev_anf: ANF_BIN):
            nonlocal one_state_b
            if not prev_anf:
                # not has any conjuctions with args
                # check one state
                if one_state_b:
                    # solved as one
                    # push to true table
                    true_table.append(tuple(solve))
                return

            most_common_bit_idx, most_common_bit_mask = frequency_stack.pop()

            # solve as zero
            solve[most_common_bit_idx] = False
            subanf = solve_as_zero(prev_anf, most_common_bit_mask)
            dfs_solve_b(subanf)

            # solve as one
            solve[most_common_bit_idx] = True
            subanf, one_flag = solve_as_one(prev_anf, most_common_bit_mask)
            if one_flag:
                # revert status of 1 conjunction
                one_state_b = not one_state_b
            dfs_solve_b(subanf)

            # reset state
            solve[most_common_bit_idx] = None
            if one_flag:
                # return status of 1 conjunction
                one_state_b = not one_state_b
            frequency_stack.append((most_common_bit_idx, most_common_bit_mask))

        # solve true table
        dfs_solve_a(sub_anf_a)

        # construct anf by true table
        solved_table_indexes = set()
        def def_tt_index_builder(pre_tt_index, bit_idx, tt_row):
            tt_cell = tt_row[bit_idx]
            if tt_cell is None:
                tt_index_zero = pre_tt_index
                tt_index_one = pre_tt_index | (1 << bit_idx)

                next_bit_idx = bit_idx + 1
                if next_bit_idx >= len(tt_row):
                    solved_table_indexes.add(tt_index_zero)
                    solved_table_indexes.add(tt_index_one)
                else:
                    def_tt_index_builder(tt_index_zero, bit_idx + 1, tt_row)
                    def_tt_index_builder(tt_index_one, bit_idx + 1, tt_row)
            else:
                if tt_cell:
                    tt_index = pre_tt_index | (1 << bit_idx)
                else:
                    tt_index = pre_tt_index

                next_bit_idx = bit_idx + 1
                if next_bit_idx >= len(tt_row):
                    solved_table_indexes.add(tt_index)
                else:
                    def_tt_index_builder(tt_index, bit_idx + 1, tt_row)

        for tt_row in true_table:
            def_tt_index_builder(0, 0, tt_row)

        for level_num in range(len(solve)):
            level_offset = 1 << level_num
            offset_block = {tti + level_offset for tti in solved_table_indexes if not tti & level_offset}
            solved_table_indexes.symmetric_difference_update(offset_block)
        return solved_table_indexes

    @staticmethod
    def _branch_xor_and(sub_anf_a: ANF_BIN, sub_anf_b: ANF_BIN):
        products_counter = Counter()
        if not sub_anf_a or not sub_anf_b:
            return products_counter

        sorted_sub_anf_a = sorted(sub_anf_a, key=lambda x: x.bit_count(), reverse=True)
        sorted_sub_anf_b = sorted(sub_anf_b, key=lambda x: x.bit_count(), reverse=True)

        tree_a, value_map_over_items_a = XorAndTreeNode.build(sorted_sub_anf_a)
        tree_b, value_map_over_items_b = XorAndTreeNode.build(sorted_sub_anf_b)

        # get first groups
        iter_a = iter(sorted_sub_anf_a)
        iter_b = iter(sorted_sub_anf_b)

        item_a = next(iter_a)
        item_a_rank = item_a.bit_count()
        item_b = next(iter_b)
        item_b_rank = item_b.bit_count()
        global skip_counter

        while item_a_rank > 0 and item_b_rank > 0:
            if item_a_rank >= item_b_rank:
                item = item_a
                tree_node = tree_b

                to_remove = value_map_over_items_a[item_a]
                next_item = to_remove.self_remove()
                del value_map_over_items_a[item_a]

                if tree_a and tree_a.value == item_a:
                    # update tree first node
                    tree_a = next_item

                try:
                    item_a = next(iter_a)
                except StopIteration:
                    # items from a is done
                    # skip a
                    item_a_rank = -1
                else:
                    item_a_rank = item_a.bit_count()
            else:
                item = item_b
                tree_node = tree_a

                to_remove = value_map_over_items_b[item_b]
                next_item = to_remove.self_remove()
                del value_map_over_items_b[item_b]

                if tree_b and tree_b.value == item_b:
                    # update tree first node
                    tree_b = next_item

                try:
                    item_b = next(iter_b)
                except StopIteration:
                    # items from a is done
                    # skip a
                    item_b_rank = -1
                else:
                    item_b_rank = item_b.bit_count()

            tr0 = time.perf_counter()
            parents_next_stack = []
            while tree_node is not None:
                tri0 = time.perf_counter()
                tnv = tree_node.value
                product = tnv | item
                if item & tnv ^ tnv:
                    tc0 = time.perf_counter()
                    # item not cover node, count them self
                    product_multiplier = 1
                    if tree_node.first_children:
                        # scan children if has
                        # if has next tree save it to return stack
                        if tree_node.next:
                            parents_next_stack.append(tree_node.next)
                        # start from first child
                        tree_node = tree_node.first_children
                    else:
                        # DRY violation: see same code above
                        if tree_node.next:
                            tree_node = tree_node.next
                        else:
                            if parents_next_stack:
                                tree_node = parents_next_stack.pop()
                            else:
                                tree_node = None
                    tc1 = time.perf_counter()
                    scan_time.append(tc1 - tc0)
                else:
                    tc0 = time.perf_counter()
                    # item cover tree node - skip children
                    product_multiplier = tree_node.children_counter

                    # get next tree if can
                    if tree_node.next:
                        tree_node = tree_node.next
                    else:
                        # return to parents
                        if parents_next_stack:
                            # seek for first parent with neighbor next tree
                            tree_node = parents_next_stack.pop()
                        else:
                            # if no parents with next tree
                            # cycle done
                            tree_node = None
                    tc1 = time.perf_counter()
                    skip_counter += product_multiplier
                    cover_time.append(tc1 - tc0)
                products_counter[product] += product_multiplier
                tri1 = time.perf_counter()
                round_inside_time.append(tri1 - tri0)

            tr1 = time.perf_counter()
            round_time.append(tr1 - tr0)
        return products_counter

    def __or__(self, other: 'BinaryAnf'):
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

        self_copy = self.copy()
        other_copy = other.copy()

        result = self ^ other ^ (self_copy & other_copy)
        return result

    def __xor__(self, other: 'BinaryAnf'):
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

        self.conjuncts.symmetric_difference_update(other.conjuncts)
        self.inversed_by_one = self.inversed_by_one ^ other.inversed_by_one
        return self

    def __invert__(self):
        self._invert_from_plus_one()
        return self

    def copy(self):
        return BinaryAnf(self.conjuncts.copy(), inversed_by_one=self.inversed_by_one)

    def _invert_from_plus_one(self):
        self.inversed_by_one = not self.inversed_by_one

    def solve(self):
        return AnfSolver(self.bits_map, self)


class AnfSolver:
    resolve_order_stack:  list[tuple[int, int]]

    def __init__(
            self,
            bits_map: list[TreeBit | tuple[BinaryAnf, BinaryAnf]],
            root_anf: BinaryAnf,
    ):
        self.bits_map = bits_map
        self.solve_container: list[bool | None] = [None] * len(self.bits_map)

        added = set()
        anfs_list = [root_anf]
        for bit in reversed(self.bits_map):
            if isinstance(bit, tuple):
                for anf in bit:
                    if anf in added:
                        continue
                    added.add(anf)
                    anfs_list.append(anf)

        self.anfs_list = anfs_list
        self.usages_by_operand = defaultdict(set)
        for anf in self.anfs_list:
            children = self.anf_to_pair(anf)
            if children:
                for child in children:
                    self.usages_by_operand[child].add(anf)
        self.solve_for_anf = {}

    def __iter__(self):
        return self.solve_system()

    @staticmethod
    def anf_to_term(parent_anf: BinaryAnf) -> int | None:
        if len(parent_anf) != 1:
            return None
        parent_conj = next(iter(parent_anf))
        if parent_conj.bit_count() != 1:
            return None
        term_idx = parent_conj.bit_length() - 1
        return term_idx

    def term_to_pair(self, term_idx: int) -> tuple[BinaryAnf, BinaryAnf] | None:
        bit = self.bits_map[term_idx]
        if isinstance(bit, tuple):
            bit: tuple[BinaryAnf, BinaryAnf]
            return bit
        return None

    def anf_to_pair(self, parent_anf: BinaryAnf) -> tuple[BinaryAnf, BinaryAnf] | None:
        term = self.anf_to_term(parent_anf)
        if term:
            return self.term_to_pair(term)

    def solve_system(self):
        root_gen = self.solve_anf_branches(self.anfs_list[0], (True,))
        in_solve_gens_stack = [root_gen]
        next(root_gen)
        # setup first solver gen
        anf_index = 1
        while anf_index > 0:
            if anf_index >= len(self.anfs_list):
                yield self.solve_container.copy()
                anf_index -= 1
                return
            elif len(in_solve_gens_stack) > anf_index:
                solves_gen = in_solve_gens_stack[anf_index]
            else:
                print(anf_index)
                # create generator
                anf_to_solve = self.anfs_list[anf_index]
                anf_idx = self.anf_to_term(anf_to_solve)
                if anf_idx is not None:
                    already_solved = self.solve_container[anf_idx]
                    assert already_solved is None

                # search solve val
                usages = self.usages_by_operand[anf_to_solve]

                usages_and_siblings_solves = []
                for usage in usages:
                    usage_idx = self.anf_to_term(usage)
                    usage_solve = self.solve_container[usage_idx]
                    assert usage_solve is not None

                    sibling_a, sibling_b = self.term_to_pair(usage_idx)
                    sibling = sibling_a if sibling_b is anf_to_solve else sibling_b
                    sibling_idx = self.anf_to_term(sibling)
                    if sibling_idx is not None:
                        sibling_solve = self.solve_container[sibling_idx]
                    else:
                        # it's simple anf
                        sibling_solve = self.solve_for_anf.get(sibling)

                    usages_and_siblings_solves.append((usage_solve, sibling_solve))

                one_in_usages = any(u for u, _ in usages_and_siblings_solves)
                if one_in_usages:
                    solve_vals = (True,)
                    # check bounds
                    for u, s in usages_and_siblings_solves:
                        if s is not None:
                            if u:
                                assert s
                            else:
                                assert not s
                else:
                    bound_solves = [
                        (u, s) for u, s in usages_and_siblings_solves if s is not None
                    ]
                    should_be_zero = False
                    for u, s in bound_solves:
                        assert not u
                        if s:
                            # sibling is one
                            # usage is zero
                            # should be zero
                            should_be_zero = True
                        else:
                            # sibling is zero
                            # can be any
                            pass

                    if should_be_zero:
                        solve_vals = (False,)
                        # check bounds
                        for u, s in usages_and_siblings_solves:
                            assert not u
                    else:
                        solve_vals = (False, True)

                # solve
                prepared_anf = self.prepare_anf(anf_to_solve)
                solves_gen = self.solve_anf_branches(prepared_anf, solve_vals, original_anf=anf_to_solve)
                in_solve_gens_stack.append(solves_gen)
            try:
                next(solves_gen)
            except StopIteration:
                # unsolvable
                in_solve_gens_stack.pop()
                anf_index -= 1
            else:
                # solve found
                anf_index += 1


    def set_solved(self, idx: int, value: bool | None):
        self.solve_container[idx] = value

    def solve_anf_branches(self, anf: BinaryAnf, solve_vals: tuple[bool, ...], original_anf: BinaryAnf | None = None):
        if original_anf is None:
            original_anf = anf

        least_common = []
        for idx in reversed(anf.all_terms_idx_usage()):
            idx: int
            int_bit = 1 << idx
            least_common.append((idx, int_bit))

        def dfs_solve(prev_anf: BinaryAnf):
            if prev_anf.solved is not None:
                if prev_anf.solved in solve_vals:
                    assert original_anf not in self.solve_for_anf
                    self.solve_for_anf[original_anf] = prev_anf.solved
                    yield
                    del self.solve_for_anf[original_anf]
                return

            bit_idx, bit_mask = least_common.pop()

            # solve as zero
            self.set_solved(bit_idx, False)
            subanf = self.solve_anf_as_zero(prev_anf, bit_mask)
            yield from dfs_solve(subanf)

            # solve as one
            self.set_solved(bit_idx, True)
            subanf = self.solve_anf_as_one(prev_anf, bit_mask)
            yield from dfs_solve(subanf)

            # reset as unsolved
            self.set_solved(bit_idx, None)
            least_common.append((bit_idx, bit_mask))

        # prepare anf
        print('d', len(least_common))
        yield from dfs_solve(anf)

    def prepare_anf(self, anf: BinaryAnf):
        prepared_anf = anf.copy()

        # remove all conjes with solved zeros
        zero_solved_bit_mask = 0

        # simplify all conjes with solved ones
        one_solved_bit_mask = 0
        for idx, bit_solve in enumerate(self.solve_container):
            if bit_solve is False:
                zero_solved_bit_mask |= 1 << idx
            if bit_solve is True:
                one_solved_bit_mask |= 1 << idx

        conjest_to_remove = []
        short_conjes_counter = Counter()
        for conj in prepared_anf:
            if zero_solved_bit_mask & conj:
                conjest_to_remove.append(conj)
                continue

            # has at least one interception
            if interception_mask := one_solved_bit_mask & conj:
                # mark to remove old conj
                conjest_to_remove.append(conj)
                # erase to zero all args that has interception
                short_conjes_counter[conj ^ interception_mask] += 1
                continue

        prepared_anf.conjuncts.difference_update(conjest_to_remove)

        if short_conjes_counter[0] & 1:
            # check conjes with zero lenght (empty conjes has ONE)
            prepared_anf = ~prepared_anf
            del short_conjes_counter[0]

        prepared_anf.conjuncts.symmetric_difference_update(
            conj for conj, count in short_conjes_counter.items() if count & 1
        )
        return prepared_anf

    @staticmethod
    def solve_anf_as_zero(anf_to_solve: BinaryAnf, bit_to_solve: int) -> BinaryAnf:
        subanf = anf_to_solve.copy()
        conjest_to_remove = list()
        for conj in subanf:
            if bit_to_solve & conj:
                conjest_to_remove.append(conj)
        subanf.conjuncts.difference_update(conjest_to_remove)
        return subanf

    @staticmethod
    def solve_anf_as_one(anf_to_solve: BinaryAnf, bit_to_solve: int) -> BinaryAnf:
        subanf = anf_to_solve.copy()
        conjest_to_remove = list()
        short_conjes_counter = Counter()
        for conj in subanf:
            if bit_to_solve & conj:
                conjest_to_remove.append(conj)
                short_conjes_counter[conj ^ bit_to_solve] += 1

        subanf.conjuncts.difference_update(conjest_to_remove)

        if short_conjes_counter[0] % 2 == 1:
            # check conjes with zero lenght (empty conjes has ONE)
            subanf = ~subanf
            del short_conjes_counter[0]

        subanf.conjuncts.symmetric_difference_update(
            map(
                lambda sc: sc[0],
                filter(lambda sc: sc[1] % 2 == 1, short_conjes_counter.items())
            )
        )
        return subanf


class XorAndTreeNode:
    __slots__ = ('first_children', 'last_children', 'next', 'prev', 'children_counter', 'value')

    def __init__(self, value: ANF_BIN_CONJUNCT):
        self.children_counter = 1
        self.next: Self | None = None
        self.prev: Self | None = None
        self.value = value
        self.first_children: Self | None = None
        self.last_children: Self | None = None

    def __hash__(self):
        return hash(self.value)

    def __contains__(self, item: Self):
        return item.value & self.value == item.value

    def __str__(self):
        return self._str_with_level().rstrip('\n')

    def _str_with_level(self, level = 0):
        prefix = ' ' * level
        node_to_print = self
        base = ''
        while node_to_print:
            base += f'{prefix}<{node_to_print.value}>\n'
            if node_to_print.first_children:
                base += node_to_print.first_children._str_with_level(level + 1)
            node_to_print = node_to_print.next
        return base

    def add_child(self, child: Self):
        # erase next
        if not self.first_children:
            self.first_children = self.last_children = child
        else:
            # >= 2 childs
            child.prev = self.last_children
            self.last_children.next = child
            self.last_children = child

        self.children_counter += child.children_counter

    def self_remove(self):
        td0 = time.perf_counter()
        # relink prev
        new_next_to_prev = self.first_children or self.next
        if self.prev:
            # link first child or next
            self.prev.next = new_next_to_prev
        if new_next_to_prev:
            new_next_to_prev.prev = self.prev

        # relink next
        new_prev_to_next = self.last_children or self.prev
        if self.next:
            # link last child or prev
            self.next.prev = new_prev_to_next
        if new_prev_to_next:
            new_prev_to_next.next = self.next

        global delete_time
        delete_time += time.perf_counter() - td0
        # return new `next node` for prev
        return new_next_to_prev

    @classmethod
    def build_old(cls, sorted_conjes: Iterable[ANF_BIN_CONJUNCT]):
        orphans = set()
        value_map_over_items = {}

        prev_conj_level = 1
        # create nodes and trees
        for conj in sorted_conjes:
            conj_level = conj.bit_count()
            assert conj_level >= prev_conj_level
            prev_conj_level = conj_level

            new_node = cls(conj)
            value_map_over_items[conj] = new_node
            if conj_level == 1:
                # first level just add
                orphans.add(new_node)
            else:
                realised_orphans = set()
                for orphan in orphans:
                    if orphan in new_node:
                        new_node.add_child(orphan)
                        realised_orphans.add(orphan)

                orphans.symmetric_difference_update(realised_orphans)
                orphans.add(new_node)

        # link tree's

        # temporary root to link trees
        zero_root = cls((1 << (prev_conj_level + 1)) - 1)
        for next_tree_root in orphans:
            zero_root.add_child(next_tree_root)

        return zero_root.first_children, value_map_over_items

    @classmethod
    def build_old_2(cls, sorted_conjes: Iterable[ANF_BIN_CONJUNCT]):
        max_conj = max(sorted_conjes)
        conjes_iter = iter(sorted_conjes)
        first_conj = next(conjes_iter)

        prev_conj_level = first_conj.bit_count()
        full_conj = (1 << max_conj.bit_length()) - 1


        # if first_conj == full_conj:
        #     # group_tree_root = BinBitmapGroupTreeNode(first_conj, True)
        # else:
        #     # group_tree_root = BinBitmapGroupTreeNode(full_conj, False)
        group_tree_root = MBitmapGroupTreeNode(full_conj)
        group_tree_root.add(first_conj)

        # create nodes and trees
        for conj in conjes_iter:
            conj_level = conj.bit_count()
            assert conj_level <= prev_conj_level
            prev_conj_level = conj_level

            group_tree_root.add(conj)
        # link tree's

        # temporary root to link trees
        zero_root, value_map_over_items = group_tree_root.convert_to_xor_and_forest()
        # print({c: bin(c) for c in sorted_conjes})
        # print(sorted_conjes)
        # print(group_tree_root.full_str())
        # print(zero_root)
        return zero_root.first_children, value_map_over_items

    @classmethod
    def build(cls, sorted_conjes: list[ANF_BIN_CONJUNCT]):
        value_map_over_items = {}
        parents_map = defaultdict(list)
        max_level = sorted_conjes[0].bit_count()
        quote_manager = QuoteManager(max_level)
        orphans = set()

        prev_conj_level = max_level
        # desc
        for conj in sorted_conjes:
            conj_level = conj.bit_count()
            assert conj_level <= prev_conj_level
            prev_conj_level = conj_level

            parent = quote_manager.check(conj, conj_level)
            if parent:
                parents_map[parent].append(conj)
            quote_manager.add(conj, conj_level)

        # ask
        for conj in reversed(sorted_conjes):
            # create nodes and trees
            new_node = cls(conj)
            orphans.add(new_node)
            value_map_over_items[conj] = new_node
            conj_level = conj.bit_count()
            if conj_level == 1:
                # first level just add
                continue
            else:
                for child in parents_map[conj]:
                    child_node = value_map_over_items[child]
                    new_node.add_child(child_node)
                    orphans.remove(child_node)
        # temporary root to link trees
        zero_root = cls((1 << (prev_conj_level + 1)) - 1)
        for orphan in orphans:
            zero_root.add_child(orphan)

        return zero_root.first_children, value_map_over_items


class QuoteManager:
    def __init__(self, max_level):
        self.quota_by_level = [{} for _ in range(max_level + 1)]

    def check(self, value: int , level: int):
        quota = self.quota_by_level[level]
        if value in quota:
            parent = quota[value]
            return parent
        return None

    def add(self, value: int, level: int):
        quota = self.quota_by_level[level - 1]
        length = value.bit_length()
        mask = 1
        while length:
            if value & mask:
                child_value = value ^ mask
                if child_value not in quota:
                    quota[child_value] = value
            mask <<= 1
            length -= 1


class BinBitmapGroupTreeNode:
    __slots__ = ('value', 'left', 'right', 'is_real')

    def __init__(self, value: int, is_real: bool):
        self.value = value
        self.is_real = is_real
        self.left: Self | None = None
        self.right: Self | None = None

    def add(self, value: int) -> Self:
        # return self or new merged node
        assert value != self.value
        assert value & self.value == value

        if self.left:
            # check value in node
            if self.left.value & value == value:
                self.left.add(value)
                return
            # not in left - check right
        else:
            self.left = BinBitmapGroupTreeNode(value, True)
            return

        if self.right:
            # check value in node
            if self.right.value & value == value:
                self.right.add(value)
                return
            # not in right - new branch - need to merdge some another
        else:
            self.right = BinBitmapGroupTreeNode(value, True)
            return

        if self.left.value ^ value <= self.right.value ^ value:
            to_merge = self.left
        else:
            to_merge = self.right

        new_node = BinBitmapGroupTreeNode(value, True)
        merged_node = BinBitmapGroupTreeNode(value | to_merge.value, False)
        merged_node.left = to_merge
        merged_node.right = new_node

        if self.left is to_merge:
            self.left = merged_node
        else:
            self.right = merged_node
        return

    def convert_to_xor_and_forest(self) -> tuple[XorAndTreeNode, dict[int, XorAndTreeNode]]:
        value_map_over_items = {}
        zero_root_node = XorAndTreeNode(self.value)
        self._dfs_xor_and_tree_apply(zero_root_node, value_map_over_items)
        return zero_root_node, value_map_over_items

    def _dfs_xor_and_tree_apply(self, xor_parent: XorAndTreeNode, value_map_over_items: dict[int, XorAndTreeNode]):
        if self.is_real:
            self_xor_node = XorAndTreeNode(self.value)
            value_map_over_items[self.value] = self_xor_node

            if self.left:
                self.left._dfs_xor_and_tree_apply(self_xor_node, value_map_over_items)
            if self.right:
                self.right._dfs_xor_and_tree_apply(self_xor_node, value_map_over_items)

            xor_parent.add_child(self_xor_node)

        else:
            if self.left:
                self.left._dfs_xor_and_tree_apply(xor_parent, value_map_over_items)
            if self.right:
                self.right._dfs_xor_and_tree_apply(xor_parent, value_map_over_items)

    def full_str(self):
        return self._str_with_level(0)

    def _str_with_level(self, level = 0):
        prefix = ' ' * level
        base = f'{prefix}<{self.value}{"" if self.is_real else ":m"}>\n'
        if self.left:
            base += self.left._str_with_level(level + 1)
        if self.right:
            base += self.right._str_with_level(level + 1)
        return base


class MBitmapGroupTreeNode:
    __slots__ = ('value', 'children')

    def __init__(self, value: int):
        self.value = value
        self.children: list[Self] = []

    def add(self, value: int) -> Self:
        # return self or new merged node
        # assert value != self.value
        assert value & self.value == value

        for child in self.children:
            if child.value & value == value:
                child.add(value)
                return

        new_node = MBitmapGroupTreeNode(value)
        self.children.append(new_node)
        return

    def convert_to_xor_and_forest(self) -> tuple[XorAndTreeNode, dict[int, XorAndTreeNode]]:
        value_map_over_items = {}
        zero_root_node = XorAndTreeNode(self.value)
        for child in self.children:
            child._dfs_xor_and_tree_apply(zero_root_node, value_map_over_items)
        return zero_root_node, value_map_over_items

    def _dfs_xor_and_tree_apply(self, xor_parent: XorAndTreeNode, value_map_over_items: dict[int, XorAndTreeNode]):
        self_xor_node = XorAndTreeNode(self.value)
        value_map_over_items[self.value] = self_xor_node

        for child in self.children:
            child._dfs_xor_and_tree_apply(self_xor_node, value_map_over_items)

        xor_parent.add_child(self_xor_node)

    def full_str(self):
        return self._str_with_level(0)

    def _str_with_level(self, level = 0):
        prefix = ' ' * level
        base = f'{prefix}<{self.value}>\n'

        for child in self.children:
            base += child._str_with_level(level + 1)
        return base


def counter2conjes(counter):
    return {c for c, f in counter.items() if f & 1}


if __name__ == '__main__':
    # for ratio_int in range(50, 1, -1):
    #     ratio = ratio_int / 100
    #     need_conjes = 8
    #     conjes = set()
    #     while True:
    #         # generate
    #         while len(conjes) < need_conjes:
    #             conjes.add(random.randint(1, 1<<64))
    #         print('try', need_conjes)
    #
    #         conjes_a = conjes.copy()
    #         conjes_b = set()
    #         for _ in range(int(len(conjes_a) * ratio)):
    #             conjes_b.add(conjes_a.pop())
    #
    #         counter_plain = Counter()
    #
    #         t0 = time.perf_counter()
    #         cp = BinaryAnf.lob(conjes_a, conjes_b)
    #         t1 = time.perf_counter()
    #         t2 = time.perf_counter()
    #         counter_plain.update(cp)
    #
    #         plain = len(conjes_a) * len(conjes_b)
    #         eco = len([c for c, cc in cp.items() if cc % 2 == 1])
    #
    #         args_a_mask = 0
    #         args_b_mask = 0
    #         for c in conjes_a:
    #             args_a_mask |= c
    #         for c in conjes_b:
    #             args_b_mask |= c
    #         args_mask = args_a_mask & args_b_mask
    #         args = args_mask.bit_count()
    #
    #         tt = 2 ** args
    #
    #         print(
    #             f'N:{len(conjes_a)}x{len(conjes_b)}, '
    #             f'plain: {plain:.5f}, args: {args}, true table: {tt}, {plain * 100 / tt :.2f}%'
    #             f''
    #         )
    #         need_conjes *= 1.5

    stat_p = defaultdict(dict)
    stat_t = defaultdict(dict)
    stat_d = defaultdict(dict)

    for max_bits in range(4, 17):
        need_conjes = 8
        conjes = set()

        max_num = 1 << max_bits
        while need_conjes < max_num:
            # generate
            while len(conjes) < need_conjes:
                conjes.add(random.randint(1, max_num))

            conjes_a = conjes.copy()
            conjes_b = set()
            for _ in range(int(len(conjes_a) * 0.5)):
                conjes_b.add(conjes_a.pop())

            prod = len(conjes_a) * len(conjes_b)

            counter_plain = Counter()
            counter_branch = Counter()

            t0 = time.perf_counter()
            cp = BinaryAnf.lob(conjes_a, conjes_b)
            t1 = time.perf_counter()
            ct = BinaryAnf._tt_xor_and(conjes_a, conjes_b)
            t2 = time.perf_counter()
            counter_plain.update(cp)

            assert counter2conjes(counter_plain) == ct
            plain = t1 - t0
            tt = t2 - t1

            print(
                f'B: {max_bits} N:{len(conjes)} P:{prod}, '
                f'plain: {plain:.5f}, '
                f'tt: {tt:.5f}, '
                f'diff: {plain - tt:.5f}'
            )
            args_mask = 0
            for c in conjes:
                args_mask |= c
            bnum = args_mask.bit_count()
            stat_p[bnum][prod] = plain
            stat_t[bnum][prod] = tt
            stat_d[bnum][prod] = plain - tt
            need_conjes *= 2

    # print bit stat

    min_table = []
    max_table = []
    for b, d in sorted(stat_t.items(), key=lambda x: x[0]):
        _, mind = min(d.items(), key=lambda x: x[0])
        _, maxd = max(d.items(), key=lambda x: x[0])
        min_table.append((b, mind))
        max_table.append((b, maxd))
    print('min t')
    for r in min_table:
        print(*r, sep='\t')
    print('max t')
    for r in max_table:
        print(*r, sep='\t')

    # print plain stat
    agg = defaultdict(list)
    for _, d in stat_p.items():
        for n, t in d.items():
            agg[n].append(t)

    print('min p')
    for n, tt in sorted( agg.items(), key=lambda x: x[0]):
        print(n, min(tt), sep='\t')
    print('max p')
    for n, tt in sorted( agg.items(), key=lambda x: x[0]):
        print(n, max(tt), sep='\t')
    # print random

    pg = []
    tg = []
    for b, d in stat_d.items():
        for n, t in d.items():
            if t > 0:
                tg.append((b, n))
            else:
                pg.append((b, n))
    print('to tg')
    for tgi in tg:
        print(*tgi, sep='\t')
    print('to pg')
    for pgi in pg:
        print(*pgi, sep='\t')

    # a = {15}
    # b = {13, 14, 25, 9, 24, 8}
    # t0 = time.perf_counter()
    # print(BinaryAnf._sub_xor_and(a, b))
    # t2 = time.perf_counter()

    # print(BinaryAnf._tt_xor_and({1}, {2, 4}))




