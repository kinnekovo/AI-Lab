"""对外接口：对应 PDF 四题的函数签名（也可以从模块单独导入）。"""
from __future__ import annotations
from .astar import astar_search, enter_cost, admissible_grid_heuristic, manhattan
from .tree import TreeNode, alpha_beta_best_score
from .treasure import solve_treasure


def find_shortest_path(start, end, grid):
    """题目一：成功返回 [(row,col),...]；无路径返回 []。"""
    return astar_search(grid, start, end, heuristic=manhattan).path


def minimum_travel_cost(start, end, grid):
    """题目二：E 的进入代价为 0；无法抵达返回 -1。"""
    heuristic = admissible_grid_heuristic(grid, end, enter_cost)
    return astar_search(grid, start, end, cost_fn=enter_cost, heuristic=heuristic).cost


__all__ = ["find_shortest_path", "minimum_travel_cost", "TreeNode", "alpha_beta_best_score", "solve_treasure"]
