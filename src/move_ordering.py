"""扩展实验 A：对手感知的 Alpha-Beta 节点排序。只重排，不修改最终收益评价。"""
from __future__ import annotations
from math import isfinite


def opponent_aware_priority(game, state, treasure: str, lambda_weight: float = 1.0) -> float:
    if lambda_weight < 0 or not isfinite(lambda_weight):
        raise ValueError("lambda_weight 必须为有限非负数")
    own_gain = game.gain(state, treasure)
    if lambda_weight == 0:
        return float(own_gain)
    next_state = game.move(state, treasure)
    remaining = next_state.remaining_treasures
    if not remaining:
        return float(own_gain)
    opponent_best_gain = max(game.gain(next_state, t) for t in remaining)
    return float(own_gain - lambda_weight * opponent_best_gain)
