"""通用 Minimax 与 Alpha-Beta，不绑定特定题目或状态类型。"""
from __future__ import annotations
from dataclasses import dataclass, field
from math import inf
from time import perf_counter
from typing import Callable, Generic, Sequence, TypeVar, Any

S = TypeVar("S")
Action = TypeVar("Action")


@dataclass
class GameSearchStats:
    visited_nodes: int = 0   # 递归调用次数，含根及叶子
    evaluated_leaves: int = 0
    cutoffs: int = 0         # 真正跳过至少一条分支的剪枝位置数量
    pruned_branches: int = 0 # 直接未访问的子分支数
    max_reached_depth: int = 0
    elapsed_ms: float = 0.0


@dataclass
class GameSearchResult:
    best_action: Any
    best_value: int | float
    stats: GameSearchStats = field(default_factory=GameSearchStats)


def search_game(
    initial_state: S,
    depth: int,
    *,
    actions: Callable[[S], Sequence[Action]],
    transition: Callable[[S, Action], S],
    evaluate: Callable[[S], int | float],
    maximizing: Callable[[S], bool],
    algorithm: str = "alpha_beta",
    order: str = "forward",
    priority: Callable[[S, Action], float] | None = None,
    trace: Callable[[dict], None] | None = None,
) -> GameSearchResult:
    """depth 为剩余行动层数；终局或 depth=0 使用当前状态的评价值。

    order=forward/reverse/priority：priority 模式需显式提供排序函数；
    priority 值越大越优先（适合双方按即时宝藏收益贪心排序）。
    """
    if depth < 0:
        raise ValueError("depth 不能为负数")
    if algorithm not in {"minimax", "alpha_beta"}:
        raise ValueError(f"无效算法：{algorithm}")
    if order not in {"forward", "reverse", "priority"}:
        raise ValueError(f"无效节点顺序：{order}")
    if order == "priority" and priority is None:
        raise ValueError("priority 顺序需要提供 priority 回调")

    begin = perf_counter()
    stats = GameSearchStats()

    def ordered_moves(state: S) -> list[Action]:
        moves = list(actions(state))
        if order == "reverse":
            moves.reverse()
        elif order == "priority":
            moves.sort(key=lambda a: priority(state, a), reverse=True)  # type: ignore[misc]
        return moves

    def event(kind: str, *, path: tuple, level: int, **fields):
        if trace is not None:
            trace({"event": kind, "path": "/".join(map(str, path)), "depth": level, **fields})

    def rec(state: S, remaining: int, alpha: float, beta: float, level: int, path: tuple):
        stats.visited_nodes += 1
        stats.max_reached_depth = max(stats.max_reached_depth, level)
        if remaining == 0:
            stats.evaluated_leaves += 1
            val = evaluate(state)
            if trace is not None:
                event("leaf", path=path, level=level, value=val, alpha=alpha, beta=beta)
            return val, None
        choices = ordered_moves(state)
        if not choices:
            stats.evaluated_leaves += 1
            val = evaluate(state)
            if trace is not None:
                event("leaf", path=path, level=level, value=val, alpha=alpha, beta=beta)
            return val, None
        is_max = maximizing(state)
        if trace is not None:
            event("expand", path=path, level=level, choices="|".join(map(str, choices)),
                  player="MAX" if is_max else "MIN", alpha=alpha, beta=beta)
        best = -inf if is_max else inf
        best_move = None
        for i, move in enumerate(choices):
            score, _ = rec(transition(state, move), remaining - 1, alpha, beta, level + 1, path + (move,))
            if (is_max and score > best) or (not is_max and score < best):
                best, best_move = score, move
            if trace is not None:
                event("child_result", path=path + (move,), level=level + 1,
                      score=score, best=best, alpha=alpha, beta=beta)
            if algorithm == "alpha_beta":
                if is_max:
                    alpha = max(alpha, best)
                else:
                    beta = min(beta, best)
                if alpha >= beta and i + 1 < len(choices):
                    stats.cutoffs += 1
                    stats.pruned_branches += len(choices) - i - 1
                    if trace is not None:
                        event("cutoff", path=path, level=level, alpha=alpha, beta=beta,
                              skipped="|".join(map(str, choices[i+1:])), best=best)
                    break
        return best, best_move

    value, move = rec(initial_state, depth, -inf, inf, 0, ())
    stats.elapsed_ms = (perf_counter() - begin) * 1000
    val = int(value) if float(value).is_integer() else value
    return GameSearchResult(move, val, stats)
