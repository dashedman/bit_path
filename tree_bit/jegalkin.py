import itertools
import time
from collections import defaultdict, Counter
from pprint import pprint
from threading import Thread
from typing import Self

import tree_bit
from tree_bit.base import TreeBit, TreeBitAtom, ONE_BIT
from tree_bit.tools import count_all_operators, HashableSet

ANF_LITERAL = TreeBit
ANF_CONJUNCT = frozenset[ANF_LITERAL]
# ANF_CONJUNCT = HashableSet[ANF_LITERAL]
ANF = set[ANF_CONJUNCT]


def str_conj(conj: ANF_CONJUNCT):
    return ' & '.join(unit.name for unit in sorted(conj, key=lambda tn: tn.name))


class Anf:
    """
    Zhegalkin polynomials
    """
    def __init__(self, anf: ANF):
        self.conjuncts: ANF = anf
        self._terms: list[ANF_LITERAL] | None = None
        self._full_terms = None
        self.recalc_solved()

    def __len__(self):
        return len(self.conjuncts)

    def __iter__(self):
        return iter(self.conjuncts)

    def __str__(self):
        solved = self.solved
        if solved is not None:
            anf_str = f'Anf ({solved}): '
        else:
            anf_str = f'Anf ({len(self)}, {max(len(conj) for conj in self) if self else 0})'

        cs = ' ⊕ '.join(sorted('(' + str_conj(conj) + ')' for conj in self)).replace('(ONE)', 'ONE')
        if 'ONE ⊕' in cs:
            cs = cs.replace('ONE ⊕', '') + '⊕ ONE'

        anf_str += '{' + cs + '}'
        return anf_str

    def recalc_solved(self):
        if len(self.conjuncts) == 0:
            self.solved = False
            return
        if len(self.conjuncts) == 1:
            single_conj = next(iter(self.conjuncts))
            assert len(single_conj) > 0
            if len(single_conj) == 1:
                single_bit = next(iter(single_conj))
                if single_bit.resolved:
                    assert single_bit.value
                    self.solved = True
                    return
        self.solved = None

    @property
    def terms(self):
        if self._terms is None:
            self._terms = sorted(
                {term for conj in self for term in conj},
                key=lambda tn: tn.name
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
    def get_anf_for_bit(cls, exit_bit: TreeBitAtom, anf_for_bit: dict[TreeBitAtom, 'Anf'] | None = None):
        if anf_for_bit is None:
            anf_for_bit = {}
        counter = 0
        depth_counter = 0
        max_depth = 0
        estimated = count_all_operators(exit_bit)

        cache_hit_counter = 0
        cache_miss_counter = 0

        calc_stats = defaultdict(list)

        def check_cache(bit):
            return anf_for_bit.get(bit)

        def anf_by_bit_dfs(
            bit: TreeBitAtom,
        ) -> Anf:
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
                    anf_prev = anf_by_bit_dfs(bit.bit)
                    n = bit.cls_name
                    time_start = time.perf_counter()
                    anf = ~anf_prev
                case tree_bit.base.TreeBitXOR:
                    anf_a = anf_by_bit_dfs(bit.a)
                    anf_b = anf_by_bit_dfs(bit.b)
                    n = bit.cls_name
                    time_start = time.perf_counter()
                    anf = anf_a ^ anf_b
                case tree_bit.base.TreeBitOR:
                    anf_a = anf_by_bit_dfs(bit.a)
                    anf_b = anf_by_bit_dfs(bit.b)
                    n = bit.cls_name
                    time_start = time.perf_counter()
                    anf = anf_a | anf_b
                case tree_bit.base.TreeBitAND:
                    anf_a = anf_by_bit_dfs(bit.a)
                    anf_b = anf_by_bit_dfs(bit.b)
                    n = bit.cls_name
                    time_start = time.perf_counter()
                    anf = anf_a & anf_b
                case tree_bit.base.TreeBit:
                    n = 'bit'
                    time_start = time.perf_counter()
                    anf = cls.from_explicit_bit(bit)
                case _:
                    raise Exception('unreachable')
            time_end = time.perf_counter()
            elapsed = time_end - time_start
            calc_stats[n].append(elapsed)

            counter += 1

            if not isinstance(anf, Anf):
                raise Exception()
            anf_for_bit[bit] = anf.copy()

            if counter == 85:
                print()

            if counter > 0:
                sanf = str(anf)
                print(counter, sanf[sanf.find('{'):])
            if counter > 500:
                depth_counter = 0
                raise Exception('=)')

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


        stat = Thread(
            target=print_state,
        )
        stat.start()
        result = anf_by_bit_dfs(exit_bit)
        stat.join()

        return result

    @classmethod
    def from_explicit_bit(cls, bit: TreeBit):
        return cls({frozenset((bit,))})
        # return cls({HashableSet((bit,))})

    def probe_mask(self):
        mask = 0
        for anf_args in itertools.product(
                *([(False, True)] * len(self.full_terms)),
        ):
            bound_args = dict(zip(self.full_terms, anf_args))
            result = self.apply_anf_args(bound_args)
            mask = (mask << 1) + result
        return mask

    def apply_anf_args(self, bound_args: dict[ANF_LITERAL, bool]) -> bool:
        assert len(bound_args) == len(self.full_terms)

        if self.solved is not None:
            return self.solved

        count_of_true_conjs = sum(all(bound_args[term] for term in conj) for conj in self)
        return count_of_true_conjs % 2 == 1

    def count_terms_usage(self):
        counter = Counter()
        for conj in self:
            for literal in conj:
                counter[literal] += 1
        return counter

    def __and__(self, other: 'Anf'):
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

        # l1 = len(self)
        # l2 = len(other)
        t0 = time.perf_counter()
        common_anf = self.conjuncts & other.conjuncts
        t1 = time.perf_counter()
        least_self = self.conjuncts - common_anf
        t2 = time.perf_counter()
        least_other = other.conjuncts - common_anf
        t3 = time.perf_counter()
        merged_anf = common_anf.copy()
        t4 = time.perf_counter()

        for conj_m in common_anf:
            if conj_m == ONE_BIT_AS_CONJ:
                # just add all conjes
                merged_anf.symmetric_difference_update(least_self)
                merged_anf.symmetric_difference_update(least_other)
                continue

            for conj1 in least_self:
                if conj1 == ONE_BIT_AS_CONJ:
                    # just add
                    merged_conj = conj_m
                else:
                    merged_conj = conj_m | conj1
                self._apply_conjunct(merged_anf, merged_conj)

            for conj2 in least_other:
                if conj2 == ONE_BIT_AS_CONJ:
                    merged_conj = conj_m
                else:
                    merged_conj = conj_m | conj2
                self._apply_conjunct(merged_anf, merged_conj)

        t5 = time.perf_counter()
        for conj1 in least_self:
            if conj1 == ONE_BIT_AS_CONJ:
                merged_anf.symmetric_difference_update(least_other)
                continue

            for conj2 in least_other:
                if conj2 == ONE_BIT_AS_CONJ:
                    merged_conj = conj1
                else:
                    merged_conj = conj1 | conj2
                self._apply_conjunct(merged_anf, merged_conj)

        te = time.perf_counter()
        el = te - t0

        if el > 10:
            print(f'{el:.3f}: {t1 - t0:.3f} {t2 - t1:.3f} {t3 - t2:.3f} {t4 - t3:.3f} {t5 - t4:.3f} {te - t5:.3f}')
            print()
        self.conjuncts = merged_anf

        # if el1 > 1:
        #     print(f'long {l1} x {l2} ({l1 * l2}) -> {len(m1r)} {len(m2r)}'
        #           # f' {len(m3r)}'
        #           f' | {el1:.2f} {el2:.2f}'
        #           # f' {el3:.2f}'
        #     )

        self.recalc_solved()
        return self

    def __or__(self, other: 'Anf'):
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
        self.recalc_solved()
        return result

    def __xor__(self, other: 'Anf'):
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
        self.recalc_solved()
        return self

    def __invert__(self):
        self._invert_from_plus_one()
        self.recalc_solved()
        return self

    def copy(self):
        return Anf(self.conjuncts.copy())

    def _invert_from_plus_one(self):
        self._apply_conjunct(self.conjuncts, ONE_BIT_AS_CONJ)

    @staticmethod
    def _apply_conjunct(anf_mut: ANF, conj: ANF_CONJUNCT):
        if conj in anf_mut:
            anf_mut.remove(conj)
        else:
            anf_mut.add(conj)

ONE_BIT_AS_CONJ = frozenset((ONE_BIT,))
# ONE_BIT_AS_CONJ = HashableSet((ONE_BIT,))

def solve_anf_branches(anf: Anf):
    def solve_as_zero(anf_to_solve: Anf, bit_to_solve):
        subanf = anf_to_solve.copy()
        conjest_to_remove = list()
        for conj in subanf:
            if bit_to_solve in conj:
                conjest_to_remove.append(conj)
        subanf.conjuncts.difference_update(conjest_to_remove)
        subanf.recalc_solved()
        return subanf

    def solve_as_one(anf_to_solve: Anf, bit_to_solve):
        subanf = anf_to_solve.copy()
        conjest_to_replace = list()
        for conj in subanf:
            if bit_to_solve in conj:
                shorter_conj = conj - frozenset((bit_to_solve,))
                if len(shorter_conj) == 0:
                    conjest_to_replace.append((conj, None))
                else:
                    conjest_to_replace.append((conj, shorter_conj))
        for conj, shorter in conjest_to_replace:
            subanf.conjuncts.remove(conj)
            if shorter is None:
                subanf = ~subanf
            else:
                if shorter in subanf.conjuncts:
                    subanf.conjuncts.remove(shorter)
                else:
                    subanf.conjuncts.add(shorter)
        subanf.recalc_solved()
        return subanf

    current_bits = {}
    least_common = []
    prepared_anf = anf
    for bit, _ in anf.count_terms_usage().most_common():
        bit: TreeBit
        if bit is ONE_BIT:
            continue
        if bit.resolved:
            current_bits[bit] = bit.value
            if bit.value:
                prepared_anf = solve_as_one(prepared_anf, bit)
            else:
                prepared_anf = solve_as_zero(prepared_anf, bit)
        else:
            least_common.append(bit)


    unsolved_terms_len = len(least_common)

    def scan_solve():
        assert len(current_bits) == unsolved_terms_len
        assert None not in current_bits.values()
        solve = current_bits.copy()
        return solve

    def dfs_solve(prev_anf: Anf):
        if prev_anf.solved is not None:
            if prev_anf.solved:
                yield scan_solve()
            return

        most_common_bit = least_common.pop()

        # solve as zero
        current_bits[most_common_bit] = False
        subanf = solve_as_zero(prev_anf, most_common_bit)
        yield from dfs_solve(subanf)

        # solve as one
        current_bits[most_common_bit] = True
        subanf = solve_as_one(prev_anf, most_common_bit)
        yield from dfs_solve(subanf)

        current_bits[most_common_bit] = None
        least_common.append(most_common_bit)

    # prepare anf
    yield from dfs_solve(prepared_anf)
