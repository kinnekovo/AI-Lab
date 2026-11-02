"""扩展实验 B：适用于有向进入代价网格的 ALT / Landmark A*。

注意：若边权 w(u,v)=cost(v)，那么 d(u,v) 不一定等于 d(v,u)。
每个 Landmark 必须分别求 L->n (正向) 和 n->L (反向图) 距离。
启发函数是 max(0, d(L,t)-d(L,n), d(n,L)-d(t,L)) 的逐地标最大值。
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from math import inf
from time import perf_counter
from typing import Callable

from .astar import Coord, Grid, CostFn, HeuristicFn, admissible_grid_heuristic


DIRECTIONS = ((0, 1), (0, -1), (1, 0), (-1, 0))


def dijkstra_distances(grid: Grid, start: Coord, cost_fn: CostFn, *,
                       reverse: bool = False) -> dict[Coord, float]:
    """在反向图上，当前位置 u->相邻 v 的费用是原图 v->u 的 cost(u)。"""
    rows, cols = len(grid), len(grid[0])
    if grid[start[0]][start[1]] == "X":
        raise ValueError("Landmark 不得位于障碍物")
    dist = {start: 0.0}
    q = [(0.0, start)]
    while q:
        g, u = heapq.heappop(q)
        if g != dist[u]:
            continue
        for dr, dc in DIRECTIONS:
            v = (u[0] + dr, u[1] + dc)
            if not (0 <= v[0] < rows and 0 <= v[1] < cols):
                continue
            if grid[v[0]][v[1]] == "X":
                continue
            cost_cell = grid[u[0]][u[1]] if reverse else grid[v[0]][v[1]]
            nd = g + cost_fn(cost_cell)
            if nd < dist.get(v, inf):
                dist[v] = nd
                heapq.heappush(q, (nd, v))
    return dist


def select_landmarks(grid: Grid, count: int) -> tuple[Coord, ...]:
    """确定性地标：先近角点选取，再用曼哈顿最远点采样（不依赖查询终点）。"""
    candidates = [(r,c) for r,row in enumerate(grid) for c,x in enumerate(row) if x != "X"]
    if count < 1 or count > len(candidates):
        raise ValueError("地标数必须位于 1..可通行格数")
    rows, cols = len(grid), len(grid[0])
    corners = [(0,0),(rows-1,cols-1),(0,cols-1),(rows-1,0)]
    selected: list[Coord] = []
    for target in corners:
        if len(selected) >= count:
            break
        p = min((p for p in candidates if p not in selected),
                key=lambda p:(abs(p[0]-target[0])+abs(p[1]-target[1]),p))
        selected.append(p)
    while len(selected) < count:
        p = max((p for p in candidates if p not in selected),
                key=lambda p:(min(abs(p[0]-q[0])+abs(p[1]-q[1]) for q in selected), -p[0],-p[1]))
        selected.append(p)
    return tuple(selected)


@dataclass
class LandmarkIndex:
    landmarks: tuple[Coord, ...]
    from_landmark: tuple[dict[Coord,float], ...]
    to_landmark: tuple[dict[Coord,float], ...]
    precompute_ms: float

    @classmethod
    def build(cls, grid: Grid, cost_fn: CostFn, count: int) -> "LandmarkIndex":
        started = perf_counter()
        chosen = select_landmarks(grid, count)
        fwd = tuple(dijkstra_distances(grid,L,cost_fn) for L in chosen)
        rev = tuple(dijkstra_distances(grid,L,cost_fn,reverse=True) for L in chosen)
        return cls(chosen,fwd,rev,(perf_counter()-started)*1000)

    def heuristic(self, goal: Coord, *, base: HeuristicFn | None = None) -> HeuristicFn:
        """如果端点不可达，忽略对应的无穷距离；每项仅在两边均有限时使用。"""
        def h(node: Coord, target: Coord) -> float:
            if target != goal:
                raise ValueError("Landmark heuristic 与预先指定目标绑定")
            if node == goal:
                return 0.0
            best = float(base(node,goal)) if base is not None else 0.0
            for forward,backward in zip(self.from_landmark,self.to_landmark):
                for a,b in ((forward.get(goal),forward.get(node)),
                            (backward.get(node),backward.get(goal))):
                    if a is not None and b is not None:
                        best = max(best,a-b)
            return max(best,0.0)
        return h
