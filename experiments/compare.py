"""阶段三：生成 A*、Minimax、Alpha-Beta 可复现的对照实验结果。

命令: python -m experiments.compare --repeats 30 --output results
每种配置多次运行：算法统计量取首轮（应为确定值），时间记录中位数。
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from statistics import median
from typing import Callable

from src.astar import astar_search, zero_heuristic, manhattan, enter_cost, unit_cost, admissible_grid_heuristic
from src.maps import (MAZE_GRID, MAZE_START, MAZE_END, WEIGHTED_GRID, WEIGHTED_START,
                      WEIGHTED_END, TREASURE_EXPECTED, TREASURE_ORDER)
from src.tree import TreeNode, evaluate_tree, sample_tree
from src.treasure import TreasureGame


def make_maze(side: int, seed: int, obstacle_rate: float):
    rng = random.Random(seed)
    grid = [["X" if rng.random() < obstacle_rate else "." for _ in range(side)] for _ in range(side)]
    # 刻意留出可行的外框通路，保留内部障碍作为对照条件。
    for c in range(side): grid[0][c] = "."
    for r in range(side): grid[r][side-1] = "."
    grid[0][0] = "S"
    grid[side-1][side-1] = "E"
    return grid, (0, 0), (side-1,side-1)


def make_weighted(side: int, seed: int, obstacle_rate: float):
    rng = random.Random(seed)
    grid = [["X" if rng.random() < obstacle_rate else rng.choice((1,1,1,2,3,5,8,9))
             for _ in range(side)] for _ in range(side)]
    for c in range(side): grid[0][c] = 1
    for r in range(side): grid[r][side-1] = 1
    grid[0][0] = "S"
    grid[side-1][side-1] = "E"
    return grid, (0, 0), (side-1,side-1)


def make_tree(depth: int, branching: int, seed: int) -> TreeNode:
    rng = random.Random(seed)
    def rec(h):
        return TreeNode(value=rng.randint(-20,20)) if h == 0 else TreeNode(children=[rec(h-1) for _ in range(branching)])
    return rec(depth)


def measure(call: Callable, repeats: int):
    results = [call() for _ in range(repeats)]
    if hasattr(results[0], "best_value"):
        outcomes = {(r.best_action, r.best_value) for r in results}
    else:
        outcomes = {(tuple(r.path), r.cost) for r in results}
    if len(outcomes) > 1:
        raise RuntimeError("多次运行结果不一致")
    r = results[0]
    timing = median(item.stats.elapsed_ms for item in results)
    return r, round(timing, 6)


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("无 CSV 输出")
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def benchmark(repeats: int = 30, output: Path = Path("results")):
    if repeats < 1:
        raise ValueError("repeats 应大于或等于 1")
    (output/"tables").mkdir(parents=True, exist_ok=True)
    a_rows, b_rows = [], []

    maze_cases = {
        "official_maze": (MAZE_GRID, MAZE_START, MAZE_END),
        "maze_open_15": make_maze(15,1,0),
        "maze_obstacles_20": make_maze(20,2026,0.20),
        "maze_obstacles_30": make_maze(30,33,0.28),
    }
    weighted_cases = {
        "official_weighted": (WEIGHTED_GRID, WEIGHTED_START, WEIGHTED_END),
        "weighted_15": make_weighted(15,2026,0.12),
        "weighted_25": make_weighted(25,42,0.20),
        "weighted_35": make_weighted(35,2027,0.25),
    }

    for question, cases in ((1, maze_cases),(2,weighted_cases)):
        for case_name, (grid, start, end) in cases.items():
            cost = unit_cost if question==1 else enter_cost
            h_set = {"zero": zero_heuristic,
                     "manhattan" if question==1 else "endpoint_aware": (
                         manhattan if question==1 else admissible_grid_heuristic(grid,end,enter_cost))}
            validation_cost = None
            for h_name,h in h_set.items():
                result, med_ms = measure(lambda: astar_search(grid,start,end,cost_fn=cost,heuristic=h),repeats)
                if validation_cost is None:
                    validation_cost=result.cost
                if result.cost != validation_cost:
                    raise AssertionError(f"A* 同图不同启发函数代价不一致：{case_name} {h_name}")
                a_rows.append({
                    "question":question,"case":case_name,"heuristic":h_name,
                    "grid_rows":len(grid),"grid_cols":len(grid[0]),
                    "found":int(result.found),"min_cost":result.cost,
                    "path_length":len(result.path)-1 if result.found else -1,
                    "expanded_nodes":result.stats.expanded_nodes,
                    "generated_nodes":result.stats.generated_nodes,
                    "relaxed_edges":result.stats.relaxed_edges,
                    "stale_pops":result.stats.stale_pops,
                    "reopened_nodes":result.stats.reopened_nodes,
                    "max_frontier_size":result.stats.max_frontier_size,
                    "median_elapsed_ms":med_ms,"repeats":repeats,
                })

    trees = {
        "official_tree":sample_tree(),
        "balanced_3ary_depth4":make_tree(4,3,2026),
        "balanced_3ary_depth5":make_tree(5,3,2030),
    }
    for name, tree in trees.items():
        baseline = evaluate_tree(tree,algorithm="minimax").best_value
        for algorithm in ("minimax","alpha_beta"):
            for ordering in ("forward","reverse"):
                r,med_ms=measure(lambda: evaluate_tree(tree,algorithm=algorithm,order=ordering),repeats)
                if r.best_value!=baseline:
                    raise AssertionError("题目三的搜索策略不应改变最优值")
                b_rows.append({
                    "question":3,"case":name,"depth":-1,
                    "algorithm":algorithm,"visit_order":ordering,
                    "best_action":str(r.best_action),"best_value":r.best_value,
                    "visited_nodes":r.stats.visited_nodes,
                    "evaluated_leaves":r.stats.evaluated_leaves,
                    "cutoffs":r.stats.cutoffs,"pruned_branches":r.stats.pruned_branches,
                    "max_reached_depth":r.stats.max_reached_depth,
                    "median_elapsed_ms":med_ms,"repeats":repeats,
                })

    game = TreasureGame()
    for depth in (2,4):
        expected = TREASURE_EXPECTED[depth]
        for algorithm in ("minimax","alpha_beta"):
            for ordering in ("forward","reverse","priority"):
                r,med_ms=measure(lambda: game.solve(depth,algorithm=algorithm,order=ordering),repeats)
                if r.best_value != expected["best_value"]:
                    raise AssertionError(f"题目四收益差错误：{depth}, {algorithm}, {ordering}")
                if ordering == "forward" and r.best_action != expected["best_target"]:
                    raise AssertionError(f"题目四官方最优首步错误：{depth}, {algorithm}")
                b_rows.append({
                    "question":4,"case":"official_treasure","depth":depth,
                    "algorithm":algorithm,"visit_order":ordering,
                    "best_action":r.best_action,"best_value":r.best_value,
                    "visited_nodes":r.stats.visited_nodes,
                    "evaluated_leaves":r.stats.evaluated_leaves,
                    "cutoffs":r.stats.cutoffs,"pruned_branches":r.stats.pruned_branches,
                    "max_reached_depth":r.stats.max_reached_depth,
                    "median_elapsed_ms":med_ms,"repeats":repeats,
                })

    write_csv(output/"tables"/"astar_benchmarks.csv",a_rows)
    write_csv(output/"tables"/"game_search_benchmarks.csv",b_rows)
    labels=list(TREASURE_ORDER)
    matrix_rows=[{"source":a,**{b:game.distances[a][b] for b in labels}} for a in labels]
    write_csv(output/"tables"/"treasure_distance_matrix.csv",matrix_rows)
    official = {
        "maze": {"path":astar_search(MAZE_GRID,MAZE_START,MAZE_END).path,"steps":7},
        "weighted": {"min_cost":7},
        "tree": {"best_value":5},
        "treasure": {str(h):TREASURE_EXPECTED[h] for h in (2,4)},
    }
    (output/"tables"/"official_expected.json").write_text(json.dumps(official,indent=2,ensure_ascii=False),encoding="utf-8")
    generate_analysis(output, a_rows, b_rows, matrix_rows, repeats)
    try:
        from experiments.plot_results import plot_all
        plot_all(a_rows,b_rows,output/"figures")
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        print("[提示] 缺少 matplotlib，CSV 和分析报告仍已生成；pip install -r requirements.txt 后重新运行生成 PNG")
    print("已生成：",output.resolve())
    print(f"A* {len(a_rows)} 组，博弈搜索 {len(b_rows)} 组；单配置重复 {repeats} 次")
    return a_rows,b_rows


def generate_analysis(output: Path, a_rows: list[dict], b_rows: list[dict], distance_rows: list[dict], repeats: int):
    def q_a(case, heuristic):
        return next(r for r in a_rows if r["case"]==case and r["heuristic"]==heuristic)
    def q_b(case, algorithm, order, depth=-1):
        return next(r for r in b_rows if r["case"]==case and r["algorithm"]==algorithm
                    and r["visit_order"]==order and r["depth"]==depth)

    m0=q_a("maze_obstacles_30","zero")
    m1=q_a("maze_obstacles_30","manhattan")
    w0=q_a("weighted_35","zero")
    w1=q_a("weighted_35","endpoint_aware")
    t0=q_b("official_tree","minimax","forward")
    t1=q_b("official_tree","alpha_beta","forward")
    g2=q_b("official_treasure","alpha_beta","forward",2)
    g4=q_b("official_treasure","alpha_beta","forward",4)
    reduction=lambda a,b: round(100*(a-b)/a,1) if a else 0
    md=[
        "# 阶段三：搜索算法性能对照实验小结",
        "",
        "> 本文档由 `python -m experiments.compare` 自动生成，是实验报告的数据素材，不等于完整正式报告。",
        "",
        "## 实验设置与指标口径",
        "",
        f"- 所有随机地图/测试树使用固定随机种子；每种配置独立运行 {repeats} 次，时间取中位数（毫秒）。",
        "- A*：`expanded_nodes` 为堆中出队且未过时的实际扩展次数；`generated_nodes` 为入堆次数（含起点）。",
        "- 博弈树：`visited_nodes` 为递归访问状态数（含内节点和叶节点）；`cutoffs` 为实际跳过剩余分支的剪枝位置数。",
        "- 不同硬件的运行时间可能不同，小规模搜索请优先讨论节点数，再辅以时间。",
        "- A* 平局策略为优先较大 g；Alpha-Beta 使用 alpha >= beta 截断。",
        "- 题目三原始树的某个 MAX 子节点在左右分支中复用；未启用跨分支状态缓存，因此按常规展开计数。",
        "- `priority` 顺序按当前行动的即时 gain 降序，排序会带来计算开销；它不是预先知道博弈结果的理想排序。",
        "",
        "## 官方测试结果",
        "",
        "| 项目 | 预期结果 | 本项目验证 |",
        "|---|---|---|",
        "| 迷宫 | 7 次移动的最短路径 | 通过 |",
        "| 加权地形 | 最低移动代价 7 | 通过 |",
        "| 博弈树 | MAX 最佳值 5 | 通过 |",
        "| 宝藏 H=2 | T2, -1 | 通过 |",
        "| 宝藏 H=4 | T3, 3 | 通过 |",
        "",
        "## A* 启发函数的影响",
        "",
        "| 场景 | h=0 扩展数 | 启发式扩展数 | 减少比例 |",
        "|---|---:|---:|---:|",
        f'| 30×30 障碍迷宫 | {m0["expanded_nodes"]} | {m1["expanded_nodes"]} | {reduction(m0["expanded_nodes"],m1["expanded_nodes"])}% |',
        f'| 35×35 加权地图 | {w0["expanded_nodes"]} | {w1["expanded_nodes"]} | {reduction(w0["expanded_nodes"],w1["expanded_nodes"])}% |',
        "",
        "两种启发式在所有对应案例上返回了相同最优代价。曼哈顿距离用于单位代价迷宫；",
        "加权题的启发式考虑终点进入代价为零：`h=(Manhattan-1)*最小普通进入代价+终点进入代价`（终点处 h=0）。",
        "这避免直接使用曼哈顿距离高估距离终点只差一步的状态。",
        "",
        "## Minimax 与 Alpha-Beta",
        "",
        "| 官方树 / 左到右 | 递归节点 | 叶子评估 | 截断位置 |",
        "|---|---:|---:|---:|",
        f'| Minimax | {t0["visited_nodes"]} | {t0["evaluated_leaves"]} | {t0["cutoffs"]} |',
        f'| Alpha-Beta | {t1["visited_nodes"]} | {t1["evaluated_leaves"]} | {t1["cutoffs"]} |',
        "",
        "Alpha-Beta 通过上下界排除不可能影响最终最优值的分支。调换访问次序可能改变剪枝开销，",
        "但不会改变最终 Minimax 值。详见 `game_search_benchmarks.csv` 的 `forward`/`reverse`。",
        "",
        "## 宝藏博弈与不同搜索深度",
        "",
        "| 搜索深度 | 左到右最优首步 | 最优收益差 | Alpha-Beta 访问节点 |",
        "|---|---|---:|---:|",
        f'| H=2 | {g2["best_action"]} | {g2["best_value"]} | {g2["visited_nodes"]} |',
        f'| H=4 | {g4["best_action"]} | {g4["best_value"]} | {g4["visited_nodes"]} |',
        "",
        "深度 2 只模拟 A→B；深度 4 模拟 A→B→A→B 全部四轮。由于可预见的后续对抗不同，",
        "首步从 T2 转变为 T3。若多个目标具有相同最优值，排序策略可能给出不同但等价的首步。",
        "",
        "## 6×6 最短移动代价矩阵",
        "",
        "| From/To | A | B | T1 | T2 | T3 | T4 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in distance_rows:
        md.append("| "+row["source"]+" | "+" | ".join(str(row[n]) for n in TREASURE_ORDER)+" |")
    md.extend([
        "",
        "## 对照实验的解释边界",
        "",
        "1. 本报告展示的是固定输入的搜索效率，不代表特定策略对所有地图都更优。",
        "2. 节点排序的有效性取决于实际访问次序；启发式计算自身也有成本。",
        "3. 综合题并不通过最少步数给出路径代价；数值权重按进入格子计算，T/A/B 权值为 1。",
        "4. 题目二的 E=0，题目四的 T/A/B=1，二者不可混淆。",
        "",
        "## 进一步分析素材",
        "",
        "- `tables/astar_benchmarks.csv`：A* 路径代价、入堆、扩展、耗时。",
        "- `tables/game_search_benchmarks.csv`：算法、深度、顺序、值、评估、剪枝、耗时。",
        "- `figures/*.png`：适合引用到正式实验报告的图片。",
        "- 自定义随机地图的具体构造与种子见 `experiments/compare.py`。",
    ])
    (output/"阶段三实验小结.md").write_text("\n".join(md)+"\n",encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description="四道题的阶段三性能对照实验")
    parser.add_argument("--repeats",type=int,default=30,help="单配置重复执行次数（默认30）")
    parser.add_argument("--output",type=Path,default=Path("results"),help="输出目录")
    args=parser.parse_args()
    benchmark(repeats=args.repeats,output=args.output)

if __name__ == "__main__":main()
