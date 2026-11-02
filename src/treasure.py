"""题目四：两两 A* 预计算 + 四回合零和博弈。"""
from __future__ import annotations
from dataclasses import dataclass
from .astar import astar_search, admissible_grid_heuristic, treasure_enter_cost
from .adversarial import GameSearchResult, search_game
from .maps import TREASURE_GRID, TREASURE_POINTS, TREASURE_VALUES, TREASURE_ORDER


@dataclass(frozen=True)
class GameState:
    pos_a: str
    pos_b: str
    remaining_treasures: tuple[str, ...]
    score_a: int
    score_b: int
    current_player: str

    @property
    def utility(self) -> int:
        return self.score_a - self.score_b


def compute_pairwise_distances() -> dict[str, dict[str, int]]:
    """以真正的 A* 分别计算 36 个起点/终点组合，而非手写矩阵。"""
    matrix: dict[str, dict[str, int]] = {}
    for source in TREASURE_ORDER:
        matrix[source] = {}
        for target in TREASURE_ORDER:
            h = admissible_grid_heuristic(TREASURE_GRID, TREASURE_POINTS[target], treasure_enter_cost)
            result = astar_search(
                TREASURE_GRID, TREASURE_POINTS[source], TREASURE_POINTS[target],
                cost_fn=treasure_enter_cost, heuristic=h,
            )
            if not result.found:
                raise ValueError(f"关键地点之间不可达：{source} -> {target}")
            matrix[source][target] = int(result.cost)
    return matrix


class TreasureGame:
    def __init__(self, distances: dict[str, dict[str, int]] | None = None,
                 values: dict[str, int] | None = None):
        self.distances = compute_pairwise_distances() if distances is None else distances
        self.values = dict(TREASURE_VALUES if values is None else values)
        self.root = GameState("A", "B", tuple(self.values), 0, 0, "A")

    @staticmethod
    def available(state: GameState) -> tuple[str, ...]:
        return state.remaining_treasures

    def gain(self, state: GameState, treasure: str) -> int:
        pos = state.pos_a if state.current_player == "A" else state.pos_b
        return self.values[treasure] - self.distances[pos][treasure]

    def move(self, state: GameState, treasure: str) -> GameState:
        if treasure not in state.remaining_treasures:
            raise ValueError("不能领取已经领取的宝藏")
        gain = self.gain(state, treasure)
        return GameState(
            pos_a=treasure if state.current_player == "A" else state.pos_a,
            pos_b=treasure if state.current_player == "B" else state.pos_b,
            remaining_treasures=tuple(x for x in state.remaining_treasures if x != treasure),
            score_a=state.score_a + gain if state.current_player == "A" else state.score_a,
            score_b=state.score_b + gain if state.current_player == "B" else state.score_b,
            current_player="B" if state.current_player == "A" else "A",
        )

    def solve(self, depth: int, *, algorithm: str = "alpha_beta",
              order: str = "forward", lambda_weight: float = 1.0,
              trace=None) -> GameSearchResult:
        """order=forward/reverse/priority/opponent；opponent 排序不改变评估函数。

        对当前行动者，使用 own_gain - λ × opponent_best_next_gain 降序。
        无论 A(MAX) 或 B(MIN)，排序方向都是其自身机会优先；真实博弈
        仍由 MAX/MIN 的递归求解，绝不将排序分数当成叶子评价。
        """
        from .move_ordering import opponent_aware_priority
        if order == "opponent":
            priority = lambda state, target: opponent_aware_priority(self, state, target, lambda_weight)
            internal_order = "priority"
        else:
            priority = self.gain
            internal_order = order
        return search_game(
            self.root,
            depth,
            actions=self.available,
            transition=self.move,
            evaluate=lambda state: state.utility,
            maximizing=lambda state: state.current_player == "A",
            algorithm=algorithm,
            order=internal_order,
            priority=priority,
            trace=trace,
        )


def solve_treasure(depth: int, algorithm: str = "alpha_beta") -> dict[str, int | str]:
    result = TreasureGame().solve(depth=depth, algorithm=algorithm)
    return {"best_target": result.best_action, "best_value": result.best_value}
