"""可复用 A*：四邻域、可变进入代价、路径还原和统计。

约定：沿边 (u,v) 移动的代价由进入 v 的单元格决定；起点不收费。
无路径时 SearchResult.path=[]、SearchResult.cost=-1。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import heapq
from itertools import count
from math import inf
from time import perf_counter
from typing import Callable, Sequence, TypeAlias

Coord: TypeAlias = tuple[int, int]
Grid: TypeAlias = Sequence[Sequence[object]]
CostFn: TypeAlias = Callable[[object], int | float]
HeuristicFn: TypeAlias = Callable[[Coord, Coord], float]


@dataclass
class AStarStats:
    expanded_nodes: int = 0  # 实际取出且展开的状态；跳过过时堆条目
    generated_nodes: int = 0  # 入堆次数，含起点与改进 g 后再次入堆
    relaxed_edges: int = 0    # 成功更新最优 g 的边数
    stale_pops: int = 0
    reopened_nodes: int = 0
    max_frontier_size: int = 0
    elapsed_ms: float = 0.0


@dataclass
class AStarResult:
    path: list[Coord] = field(default_factory=list)
    cost: int | float = -1
    stats: AStarStats = field(default_factory=AStarStats)

    @property
    def found(self) -> bool:
        return bool(self.path)


def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def zero_heuristic(_a: Coord, _b: Coord) -> float:
    return 0.0


def _parse_numeric(cell: object, *, endpoint_cost: int) -> int:
    if isinstance(cell, (int, float)):
        value = cell
    elif isinstance(cell, str) and cell.isdecimal():
        value = int(cell)
    elif cell == "E":
        value = endpoint_cost
    else:
        value = 1  # S、A/B、宝藏以及普通空格
    if not (0 <= value < inf):
        raise ValueError(f"单元格进入代价必须为有限非负数: {cell!r}")
    return int(value) if int(value) == value else float(value)


def enter_cost(cell: object) -> int | float:
    """加权地形：题目二的 E=0；其余标记=1。"""
    return _parse_numeric(cell, endpoint_cost=0)


def treasure_enter_cost(cell: object) -> int | float:
    """综合题：A、B 和所有 T 格=1。"""
    return _parse_numeric(cell, endpoint_cost=1)


def unit_cost(_cell: object) -> int:
    return 1


def admissible_grid_heuristic(grid: Grid, goal: Coord, cost_fn: CostFn) -> HeuristicFn:
    """h(n) = (d1(n,goal)-1)*min_cost_non_goal + goal_cost。

    仅对 n != goal 计算上式；无负权时可采纳且一致：每条路径
    必须在终点之前至少经过 d1-1 个普通格，再进入终点。
    即使终点 E 的进入代价为 0 仍然有效。
    """
    if not grid or not grid[0]:
        return zero_heuristic
    candidates = [
        cost_fn(cell)
        for r, row in enumerate(grid)
        for c, cell in enumerate(row)
        if cell != "X" and (r, c) != goal
    ]
    min_non_goal = min(candidates, default=0)
    gr, gc = goal
    terminal_cost = cost_fn(grid[gr][gc])

    def heuristic(point: Coord, target: Coord) -> float:
        if target != goal:
            raise ValueError("此启发函数固定绑定到构造时指定的终点")
        d = manhattan(point, goal)
        return 0.0 if d == 0 else float((d - 1) * min_non_goal + terminal_cost)

    return heuristic


def astar_search(
    grid: Grid,
    start: Coord,
    goal: Coord,
    *,
    cost_fn: CostFn = unit_cost,
    heuristic: HeuristicFn = manhattan,
) -> AStarResult:
    """返回最小代价路径（若 h 可采纳且一致）。平局优先更大 g，再按生成次序。"""
    t0 = perf_counter()
    stats = AStarStats()

    def finish(path: list[Coord], cost: int | float) -> AStarResult:
        stats.elapsed_ms = (perf_counter() - t0) * 1000
        return AStarResult(path, cost, stats)

    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise ValueError("地图必须为非空矩形二维数组")
    rows, cols = len(grid), len(grid[0])
    if (not 0 <= start[0] < rows or not 0 <= start[1] < cols or
            not 0 <= goal[0] < rows or not 0 <= goal[1] < cols):
        return finish([], -1)
    if grid[start[0]][start[1]] == "X" or grid[goal[0]][goal[1]] == "X":
        return finish([], -1)
    if start == goal:
        stats.generated_nodes = 1
        stats.expanded_nodes = 1
        stats.max_frontier_size = 1
        return finish([start], 0)

    # (f, -g, seq, position, g)；反向 g tie-break 减少同 f 候选的无效扩展。
    order = count()
    frontier: list[tuple[float, float, int, Coord, float]] = [
        (float(heuristic(start, goal)), 0, next(order), start, 0)
    ]
    g_score: dict[Coord, float] = {start: 0}
    predecessor: dict[Coord, Coord] = {}
    expanded_best: dict[Coord, float] = {}
    stats.generated_nodes = 1
    stats.max_frontier_size = 1
    directions = ((-1, 0), (1, 0), (0, -1), (0, 1))

    while frontier:
        _f, _neg_g, _seq, node, current_g = heapq.heappop(frontier)
        if current_g != g_score[node]:
            stats.stale_pops += 1
            continue
        if node in expanded_best:
            if current_g >= expanded_best[node]:
                stats.stale_pops += 1
                continue
            stats.reopened_nodes += 1
        expanded_best[node] = current_g
        stats.expanded_nodes += 1
        if node == goal:
            path = [node]
            while path[-1] != start:
                path.append(predecessor[path[-1]])
            path.reverse()
            out_cost: int | float = int(current_g) if current_g.is_integer() else current_g
            return finish(path, out_cost)
        for dr, dc in directions:
            nxt = (node[0] + dr, node[1] + dc)
            r, c = nxt
            if not (0 <= r < rows and 0 <= c < cols) or grid[r][c] == "X":
                continue
            step = float(cost_fn(grid[r][c]))
            if not (0 <= step < inf):
                raise ValueError("A* 只允许有限非负边权")
            tentative = current_g + step
            if tentative < g_score.get(nxt, inf):
                predecessor[nxt] = node
                g_score[nxt] = tentative
                f = tentative + float(heuristic(nxt, goal))
                heapq.heappush(frontier, (f, -tentative, next(order), nxt, tentative))
                stats.generated_nodes += 1
                stats.relaxed_edges += 1
                stats.max_frontier_size = max(stats.max_frontier_size, len(frontier))
    return finish([], -1)
