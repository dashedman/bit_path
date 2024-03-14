from dataclasses import dataclass

from tree_bit import TreeBitAtom


@dataclass
class RouteItem:
    bit: TreeBitAtom
    bit_value: bool
    branched_item: bool = False


@dataclass(frozen=True)
class BranchingItem:
    bit: TreeBitAtom
    bit_value: bool | None

    def __eq__(self, other: 'BranchingItem'):
        return self.bit == other.bit

    def determinancy(self):
        return 1 if isinstance(self.bit_value, bool) else self.bit.determinancy()


@dataclass
class RollbackItem:
    bit: TreeBitAtom
    rollback_value: bool | float
    branching: bool | None = None
    remove_from_branching_options: list[TreeBitAtom] = ()
    story_id: int = -1
