"""题目三：以通用博弈搜索框架求极大极小树的根节点最佳分数。"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from .adversarial import GameSearchResult, search_game


@dataclass
class TreeNode:
    value: int | float | None = None
    children: list["TreeNode"] = field(default_factory=list)


def sample_tree() -> TreeNode:
    """PDF 第 12 页：max2 在左右分支中共享（DAG），搜索时按树展开。"""
    leaf1, leaf2, leaf3, leaf4, leaf5, leaf6 = [TreeNode(value=x) for x in (3, 5, 6, 9, 1, 2)]
    max1 = TreeNode(children=[leaf1, leaf2])
    max2 = TreeNode(children=[leaf3, leaf4])
    max3 = TreeNode(children=[leaf5, leaf6])
    return TreeNode(children=[TreeNode(children=[max1, max2]), TreeNode(children=[max2, max3])])


@dataclass(frozen=True)
class PositionedNode:
    node: TreeNode
    is_max: bool = True


def evaluate_tree(root: TreeNode, *, algorithm: str = "alpha_beta", order: str = "forward") -> GameSearchResult:
    """叶节点按 value 评价；根据树深度设置上限，根为 MAX。"""
    def height(node: TreeNode) -> int:
        return 0 if not node.children else 1 + max(height(child) for child in node.children)

    return search_game(
        PositionedNode(root),
        height(root),
        actions=lambda state: list(range(len(state.node.children))),
        transition=lambda state, i: PositionedNode(state.node.children[i], not state.is_max),
        evaluate=lambda state: _leaf_value(state.node),
        maximizing=lambda state: state.is_max,
        algorithm=algorithm,
        order=order,
    )


def _leaf_value(node: TreeNode) -> int | float:
    if node.value is None:
        raise ValueError("叶节点必须有 value；不建议使用不等深度的树")
    return node.value


def alpha_beta_best_score(root: TreeNode) -> int | float:
    return evaluate_tree(root, algorithm="alpha_beta").best_value
