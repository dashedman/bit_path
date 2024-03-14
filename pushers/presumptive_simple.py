from collections import deque, Counter
from typing import Iterable

from sortedcontainers import SortedList

from pushers.base import RouteItem, RollbackItem, BranchingItem
from reverse import print_tree, real_bits_scan
from tree_bit import TreeBitAtom, TreeBit, TreeBitNOT, TreeBitOR, TreeBitAND, TreeBitXOR


class PredictsPusher:
    def __init__(self):
        self.best_rbs = 0
        self.global_abp_balance = 0
        self.tries = set()

        self.rollback_queue: deque[RollbackItem] = deque()
        self.branching_options: SortedList[BranchingItem] = SortedList(key=BranchingItem.determinancy)
        self.already_resolved_counter: Counter[TreeBitAtom] = Counter()

    def push_presumptive_predict(
            self,
            predicted_h_bits: Iterable[TreeBitAtom],
            h_end_bits: Iterable[TreeBitAtom],
            # predicted_h: Iterable[UInt32Tree],
            # h_end: Iterable[UInt32Tree],
            result_bits: list[TreeBitAtom],
    ):

        # Initialisation
        for p_bit, e_bit in zip(predicted_h_bits, h_end_bits):
            if True:
        # for ph, eh in zip(predicted_h, h_end):
        #     for p_bit, e_bit in zip(ph.bits, eh.bits):
                if p_bit.resolved:
                    # skip
                    continue

                if isinstance(e_bit.value, bool):
                    self.branching_options.add(BranchingItem(bit=p_bit, bit_value=e_bit.value))
                    print(f'[{hash(p_bit)}] BO({len(self.branching_options)}) < [INIT]')
                else:
                    raise Exception('Exit bit is not bool!!!')

        counter = 0
        need_rollback = False
        while True:
            while self.propagate_queue or self.branching_options or need_rollback:
                print_tree(list(predicted_h_bits))
                counter += 1

                if need_rollback:
                    pass
                elif self.propagate_queue:
                    curr_route = self.propagate_queue.popleft()
                    print(f'[{hash(curr_route.bit)}] < PQ({len(self.propagate_queue)}) [POP]')
                    if curr_route.bit in self.branching_options:
                        self.branching_options.remove(curr_route.bit)
                        print(f'[{hash(curr_route.bit)}] < BO({len(self.branching_options)}) [PQ POP]')
                else:
                    # Prediction phase
                    branching: TreeBitAtom = self.branching_options.pop()
                    if branching.resolved:
                        raise Exception('Branch option already resolved!')
                    curr_route = RouteItem(
                        bit=branching,
                        bit_value=branching.value > 0.5,
                        branched_item=True
                    )
                    print(f'[{hash(curr_route.bit)}] < BO({len(self.branching_options)}) [POP]')

                if not need_rollback:
                    curr_bit = curr_route.bit
                    data_bit_value = curr_route.bit_value

                # setting propagate value
                if need_rollback or self.check_conflicts(curr_bit, data_bit_value):
                    if not need_rollback:
                        print(f'[{hash(curr_bit)}] Conflict!')
                    rollback_mode = True
                    if need_rollback:
                        pass
                    elif curr_route.branched_item:
                        print(f'[{hash(curr_bit)}] BO Conflict! Reverse prediction!')
                        data_bit_value = not data_bit_value
                        if self.check_conflicts(curr_bit, data_bit_value):
                            self.branching_options.add(curr_bit)
                            print(f'[{hash(curr_bit)}] > BO({len(self.branching_options)}) [CONFLICT ADD]')
                        else:
                            print(f'[{hash(curr_bit)}] BO Conflict resolved!')
                            rollback_mode = False
                    else:
                        self.propagate_queue.appendleft(curr_route)
                        print(f'[{hash(curr_route.bit)}] > PQ({len(self.propagate_queue)}) [CONFLICT PUSH]')

                    if rollback_mode:
                        while self.rollback_queue:
                            rollback_item = self.rollback_queue.pop()
                            print(f'[{hash(rollback_item.bit)}] RQ({len(self.rollback_queue)}) >')

                            self.global_pq_balance -= len(rollback_item.remove_from_predicted)
                            self.global_abp_balance -= len(rollback_item.remove_from_branching_options)

                            for rrb in rollback_item.remove_from_predicted:
                                if self.already_resolved_counter[rrb]:
                                    self.already_resolved_counter[rrb] -= 1
                                    print(f'[{hash(rrb)}] ARC-- -> {self.already_resolved_counter[rrb]}')
                                else:
                                    rri = self.propagate_queue.pop()
                                    print(
                                        f'[{hash(rri.bit)}] PQ({len(self.propagate_queue)}) > [ROLL]  must[{hash(rrb)}]')
                                    assert rri.bit == rrb
                            for rbo in rollback_item.remove_from_branching_options:
                                self.branching_options.discard(rbo)
                                print(f'[{hash(rbo)}] < BO({len(self.branching_options)}) [ROLL]')

                            if rollback_item.branching is None:
                                # simple rollback
                                self.propagate_queue.appendleft(RouteItem(
                                    rollback_item.bit, rollback_item.bit.value
                                ))
                                print(f'[{hash(rollback_item.bit)}] > PQ({len(self.propagate_queue)}) [ROLL]')
                                rollback_item.bit.value = rollback_item.rollback_value
                            else:
                                if (rollback_item.rollback_value > 0.5) == rollback_item.branching:
                                    # its first branching rollback
                                    self.propagate_queue.appendleft(RouteItem(
                                        rollback_item.bit,
                                        not rollback_item.branching,
                                        branched_item=True
                                    ))
                                    print(f'[{hash(rollback_item.bit)}] > PQ({len(self.propagate_queue)}) [ROLL] BO')
                                    # print(f'[{hash(rollback_item.bit)}] | BO({len(branching_options)}) [ROLL]')
                                    rollback_item.bit.value = rollback_item.rollback_value
                                    break
                                else:
                                    # its second branching rollback - remove to branching options him
                                    rollback_item.bit.value = rollback_item.rollback_value
                                    self.branching_options.add(rollback_item.bit)
                                    print(f'[{hash(rollback_item.bit)}] > BO({len(self.branching_options)}) [ROLL]')
                        else:
                            print('TRIES:')
                            print(*self.tries, sep='\n')
                            raise Exception('Rollback empty!')
                        # end of rollback mode
                        need_rollback = False
                        continue

                if self.check_already_resolved(curr_bit):
                    if curr_route.branched_item:
                        self.rollback_queue.append(
                            RollbackItem(
                                curr_bit,
                                curr_bit.value,
                                branching=data_bit_value,
                                story_id=counter,
                            )
                        )
                        print(f'[{hash(curr_bit)}] > RQ({len(self.rollback_queue)}) [AR] BO')
                    else:
                        self.already_resolved_counter[curr_bit] += 1
                        print(f'[{hash(curr_bit)}] ARC++ -> {self.already_resolved_counter[curr_bit]}')
                        self.rollback_queue.append(
                            RollbackItem(
                                curr_bit,
                                curr_bit.value,
                                story_id=counter
                            )
                        )
                        print(f'[{hash(curr_bit)}] > RQ({len(self.rollback_queue)}) [AR]')
                    curr_bit.value = data_bit_value
                    continue

                added_predicted_bits: tuple[TreeBitAtom] = tuple()
                added_branching_options: tuple[TreeBitAtom] = tuple()
                if isinstance(curr_bit, TreeBit):
                    print(counter, f'Touch TreeBit! Scanning result...')
                    scanned_bits = ''.join(str(int(bit.value)) if bit.resolved else '?' for bit in result_bits)
                    self.tries.add(scanned_bits[:9])
                    # assert len(scanned_bits) == len(real_bits_scan)

                    # curr_rbs = sum(s == r for s, r in zip(scanned_bits, real_bits_scan))
                    # best_rbs = max(best_rbs, curr_rbs)
                    # rollback_branch_points_counter = sum(r.branching is not None for r in rollback_queue)
                    # print(f'RBS Similarity: {curr_rbs}/{len(scanned_bits)} '
                    #       f'(best: {best_rbs}) RBPC: {rollback_branch_points_counter}')
                    # print(real_bits_scan[:9])
                    # print(scanned_bits[:9])
                else:

                    resolved, *routes = self.check_and_try_resolve(curr_bit, data_bit_value)
                    if resolved:
                        next_route: RouteItem = routes[0]
                        # print(counter, f'Resolved!')

                        self.propagate_queue.append(next_route)
                        print(f'[{hash(next_route.bit)}] PQ({len(self.propagate_queue)}) < [PUSH]')
                        added_predicted_bits = (next_route.bit,) + added_predicted_bits
                    else:
                        # print(counter, f'Branching {len(routes)}!')
                        for branching in routes:
                            branching: BranchingItem
                            if branching.bit in self.branching_options:
                                # print('Branching option already in list!')
                                pass
                            else:
                                self.branching_options.add(branching.bit)
                                print(f'[{hash(branching.bit)}] > BO({len(self.branching_options)}) [ADD]')
                                added_branching_options = added_branching_options + (branching.bit,)
                self.global_pq_balance += len(added_predicted_bits)
                self.global_abp_balance += len(added_branching_options)
                self.rollback_queue.append(
                    RollbackItem(
                        curr_bit,
                        curr_bit.value,
                        branching=data_bit_value if curr_route.branched_item else None,
                        remove_from_predicted=added_predicted_bits,
                        remove_from_branching_options=added_branching_options,
                        story_id=counter
                    )
                )
                print(f'[{hash(curr_bit)}] > RQ({len(self.rollback_queue)})' + (
                    ' bo' if curr_route.branched_item else ''))
                self.branching_options.discard(curr_bit)
                print(f'[{hash(curr_bit)}] < BO({len(self.branching_options)}) [RESOLVE]')
                curr_bit.value = data_bit_value
            scanned_bits = ''.join(str(int(bit.value)) if bit.resolved else '?' for bit in result_bits)
            print(f'!!!RESULT!!!: {scanned_bits}')
            print(f'____real____: {real_bits_scan}')
            # PROVOKING ROLLBACK
            need_rollback = True
        raise Exception('End of cycle!')

    def check_already_resolved(self, next_bit: TreeBitAtom):
        # extend queue
        if isinstance(next_bit, TreeBitNOT):
            if next_bit.bit.resolved:
                return True

        if isinstance(next_bit, TreeBitOR):
            if next_bit.a.resolved:
                if next_bit.b.resolved:
                    return True

        if isinstance(next_bit, TreeBitAND):
            if next_bit.a.resolved:
                if next_bit.b.resolved:
                    return True

        if isinstance(next_bit, TreeBitXOR):
            if next_bit.a.resolved:
                if next_bit.b.resolved:
                    return True

    def check_conflicts(self, next_bit: TreeBitAtom, data_bit_value: bool):
        # set predicted value
        if next_bit.resolved:
            if next_bit.value != data_bit_value:
                # Next bit value already resolved and has conflict! Need rollback!
                return True

        # extend queue
        if isinstance(next_bit, TreeBitNOT):
            if next_bit.bit.resolved:
                if not next_bit.bit.value != data_bit_value:
                    # NOT Conflict! Rollback!
                    return True

        if isinstance(next_bit, TreeBitOR):
            if next_bit.a.resolved:
                if next_bit.b.resolved:
                    if next_bit.a.value or next_bit.b.value != data_bit_value:
                        # OR Conflict! Rollback!
                        return True
                else:
                    # B not resolved
                    if next_bit.a.value:
                        if not data_bit_value:
                            # OR Conflict! Rollback!
                            return True
            else:
                # A not resolved
                if next_bit.b.resolved:
                    if next_bit.b.value:
                        if not data_bit_value:
                            # OR Conflict! Rollback!
                            return True

        if isinstance(next_bit, TreeBitAND):
            if next_bit.a.resolved:
                if next_bit.b.resolved:
                    if next_bit.a.value and next_bit.b.value != data_bit_value:
                        # AND Conflict! Rollback!
                        return True
                else:
                    # B not resolved
                    if not next_bit.a.value:
                        if data_bit_value:
                            # AND Conflict! Rollback!
                            return True
            else:
                # A not resolved
                if next_bit.b.resolved:
                    if not next_bit.b.value:
                        if data_bit_value:
                            # AND Conflict! Rollback!
                            return True

        if isinstance(next_bit, TreeBitXOR):
            if next_bit.a.resolved:
                if next_bit.b.resolved:
                    if next_bit.a.value ^ next_bit.b.value != data_bit_value:
                        # XOR Conflict! Rollback!
                        return True
        return False

    def check_and_try_resolve(self, next_bit: TreeBitAtom, data_bit_value: bool) -> tuple[
        bool, RouteItem | BranchingItem]:
        # extend queue
        if isinstance(next_bit, TreeBitNOT):
            return True, RouteItem(next_bit.bit, not data_bit_value)

        if isinstance(next_bit, TreeBitOR):
            if next_bit.a.resolved:
                # B not resolved
                if next_bit.a.value:
                    # BRANCH B, A is True, X is True
                    return False, BranchingItem(next_bit.b, next_bit)
                else:
                    return True, RouteItem(next_bit.b, data_bit_value)
            else:
                # A not resolved
                if next_bit.b.resolved:
                    if next_bit.b.value:
                        # BRANCH A, B is True, X is True
                        return False, BranchingItem(next_bit.a, next_bit)
                    else:
                        return True, RouteItem(next_bit.a, data_bit_value)
                else:
                    # A and B not resolved
                    # BRANCH A B
                    return False, BranchingItem(next_bit.a, next_bit), BranchingItem(next_bit.b, next_bit)

        if isinstance(next_bit, TreeBitAND):
            if next_bit.a.resolved:
                # B not resolved
                if not next_bit.a.value:
                    # BRANCH B, A is False, X is False
                    return False, BranchingItem(next_bit.b, next_bit)
                else:
                    return True, RouteItem(next_bit.b, data_bit_value)
            else:
                # A not resolved
                if next_bit.b.resolved:
                    if not next_bit.b.value:
                        # BRANCH A, B is False, X is False
                        return False, BranchingItem(next_bit.a, next_bit)
                    else:
                        return True, RouteItem(next_bit.a, data_bit_value)
                else:
                    # A and B not resolved
                    # BRANCH A B
                    return False, BranchingItem(next_bit.a, next_bit), BranchingItem(next_bit.b, next_bit)

        if isinstance(next_bit, TreeBitXOR):
            if next_bit.a.resolved:
                # B not resolved
                return True, RouteItem(next_bit.b, next_bit.a.value ^ data_bit_value)
            else:
                # A not resolved
                if next_bit.b.resolved:
                    return True, RouteItem(next_bit.a, next_bit.b.value ^ data_bit_value)
                else:
                    # BRANCH A B
                    return False, BranchingItem(next_bit.a, next_bit), BranchingItem(next_bit.b, next_bit)
