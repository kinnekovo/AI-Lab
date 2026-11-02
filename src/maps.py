"""PDF 第 3--15 页提供的四道题的官方测试输入。"""
from __future__ import annotations

MAZE_GRID = [
    ["S", ".", ".", "X", "."],
    [".", "X", ".", ".", "."],
    [".", "X", ".", "X", "."],
    [".", ".", ".", "X", "E"],
    ["X", "X", ".", ".", "."],
]
MAZE_START = (0, 0)
MAZE_END = (3, 4)
MAZE_EXPECTED = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4)]

WEIGHTED_GRID = [
    ["S", 1, 1, 8, 1],
    [1, 8, 3, 1, 1],
    [1, 8, 1, 8, 1],
    [1, 1, 1, 8, 1],
    [8, 8, 1, 1, "E"],
]
WEIGHTED_START = (0, 0)
WEIGHTED_END = (4, 4)
WEIGHTED_EXPECTED_COST = 7

# X 障碍；进入 A/B/T1..T4 的代价均为 1（不同于题目二的 E=0）。
TREASURE_GRID = [
    ["A", 1, 1, "X", 2, 1, "T1"],
    [1, "X", 6, "X", 1, "X", 1],
    [1, 1, "T2", 1, 1, 1, 1],
    ["X", 1, "X", 8, "X", 1, "X"],
    [1, 1, 1, 1, "T3", 1, 1],
    [1, "X", 9, "X", 1, "X", 1],
    ["T4", 1, 1, "X", 1, 1, "B"],
]
TREASURE_POINTS = {
    "A": (0, 0),
    "B": (6, 6),
    "T1": (0, 6),
    "T2": (2, 2),
    "T3": (4, 4),
    "T4": (6, 0),
}
TREASURE_VALUES = {"T1": 9, "T2": 6, "T3": 7, "T4": 7}
TREASURE_ORDER = tuple(TREASURE_POINTS)
TREASURE_EXPECTED = {
    2: {"best_target": "T2", "best_value": -1},
    4: {"best_target": "T3", "best_value": 3},
}
