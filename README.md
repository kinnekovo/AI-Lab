# 当代人工智能实验二：搜索算法

基于实验二的四道题目，完成**基线对照实验**和**三个创新拓展实验 A/B/C**。项目包含 Python 源码、测试、可复现实验脚本、原始 CSV、实测图表、具体剪枝轨迹和中文分析文档。

## 核心能力

| 模块 | 能力 | 关键设计 |
|---|---|---|
| **Grid Search** | A*、Dijkstra、最短路径重建 | 四邻域网格、可配置进入代价与启发函数、搜索统计 |
| **Landmark Heuristics** | 基于地标的 ALT 启发式 | 正向 / 反向最短距离预计算，支持非对称移动代价 |
| **Adversarial Search** | Minimax、Alpha-Beta | 通用状态转移接口、深度限制、节点访问顺序与剪枝事件追踪 |
| **Treasure Game** | 路径代价驱动的多轮零和博弈 | A* 预计算距离、不可变博弈状态、可配置宝藏和收益 |
| **Benchmark Suite** | 基线、消融、规模扩展 | 固定场景与随机种子，统计访问节点、叶子评估、耗时等指标 |
| **Reproducible Outputs** | CSV、图表、JSONL 轨迹 | 结果与算法实现解耦，支持独立检查及重新生成 |

## 项目结构

```text
AI-Experiment-2/
├── src/                              # 核心算法与问题建模
│   ├── astar.py                      # 通用 A*、Dijkstra 基线、路径重建与统计
│   ├── landmark.py                   # Landmark 选择、双向距离预计算、ALT 启发式
│   ├── adversarial.py                # 通用 Minimax / Alpha-Beta 搜索引擎
│   ├── move_ordering.py              # 对手感知的候选动作排序
│   ├── treasure.py                   # TreasureGame、状态转移及距离矩阵
│   ├── tree.py                       # 树节点建模与博弈树搜索适配
│   ├── maps.py                       # 内置网格、宝藏位置与参考场景
│   └── problems.py                   # 面向常见搜索任务的简洁调用接口
│
├── experiments/                      # 基准测试与分析流水线
│   ├── compare.py                    # 网格搜索 / 博弈搜索基础性能对照
│   ├── extension_ordering.py         # 节点排序消融、权重扫描和独立验证
│   ├── extension_landmark.py         # 地标数量、查询规模与预处理成本
│   ├── extension_scenarios.py        # 可复现的随机地图和宝藏场景
│   ├── extension_scalability.py      # 问题规模扩展及有效 / 失效案例
│   ├── extension_report.py           # 汇总对照实验结果
│   ├── plot_results.py               # 基础实验图表
│   ├── plot_extensions.py            # 扩展实验图表
│   ├── run_all.py                    # 测试、参考案例与基础基准
│   ├── run_extensions.py             # 启发式 / 排序 / 规模扩展实验
│   └── run_everything.py             # 完整复现入口
│
├── tests/                            # 算法正确性与回归测试
│   ├── test_astar.py                 # 路径、总代价、边界情形、Dijkstra 真值对照
│   ├── test_adversarial.py           # 博弈搜索与剪枝一致性
│   ├── test_treasure.py              # 距离矩阵、状态更新和最优行动
│   └── test_extension.py             # ALT 可采纳性、排序正确性及场景复现
│
├── results/                          # 已生成的可复查实验产物
│   ├── tables/                       # 基础基准 CSV、距离矩阵、预期结果
│   ├── figures/                      # 基础基准图表
│   ├── logs/                         # 测试日志、运行环境与参考输出
│   ├── 阶段三实验小结.md
│   └── extension/
│       ├── tables/                   # 参数扫描、查询明细、规模对照等
│       ├── figures/                  # 扩展实验统计图
│       ├── traces/                   # 剪枝路径追踪 JSONL
│       └── 阶段四创新拓展实验小结.md
│
├── requirements.txt
└── README.md
```

### 模块间关系

```text
参考地图 / 随机场景
       │
       ├──▶ src/astar.py ──────▶ 最短路径与代价
       │          ▲
       │          └── src/landmark.py（可选启发式）
       │
       └──▶ src/treasure.py ───▶ 博弈状态及合法动作
                    │
                    ▼
             src/adversarial.py ◀── src/move_ordering.py（可选排序）
                    │
                    ▼
        experiments/（对照、消融、规模扩展）
                    │
                    ▼
        results/（CSV、PNG、JSONL、分析摘要）

        tests/ ──▶ 对核心算法与策略进行独立正确性校验
```

`src/` 不依赖绘图框架，负责算法与状态模型；`experiments/` 只负责构造场景、运行比较、收集指标和生成报告；`tests/` 为算法正确性提供回归保护。这样的分层使得新增启发函数或排序策略时，不必修改整套实验流水线。

## 快速开始

**环境：** Python 3.10+。核心算法及测试仅依赖 Python 标准库；绘图需要 `numpy` 和 `matplotlib`。

在项目根目录运行（Windows PowerShell、Linux/macOS 终端均可）：

```bash
python -m pip install -r requirements.txt

# 仅验证算法正确性
python -m unittest discover -s tests -v

# 参考场景 + 基础性能基准
python -m experiments.run_all --repeats 30

# 节点排序、Landmark、规模扩展
python -m experiments.run_extensions --repeats 5 --query-repeats 2

# 完整流水线：测试 + 全部基准 + 数据和图表
python -m experiments.run_everything --repeats 30 --extension-repeats 5 --query-repeats 2
```

以上命令需从包含 `src/` 和 `experiments/` 的项目根目录执行。完整流水线包含较多重复查询和博弈树搜索，运行时间与机器性能有关；重跑实验会更新 `results/` 中相应的生成文件。

## Python 调用示例

### 路径搜索

```python
from src.astar import astar_search, manhattan
from src.maps import MAZE_GRID, MAZE_START, MAZE_END

result = astar_search(MAZE_GRID, MAZE_START, MAZE_END, heuristic=manhattan)
print(result.path)                 # 最短路径坐标序列
print(result.cost)                 # 7
print(result.stats.expanded_nodes) # 实际扩展的节点数
```

对于加权地图，可将 `cost_fn` 替换为 `enter_cost`，并由 `admissible_grid_heuristic(...)` 构造与实际代价规则一致的启发式。`astar_search` 在无法到达目标时返回空路径与代价 `-1`。

### 对抗搜索与节点排序

```python
from src.treasure import TreasureGame

game = TreasureGame()

baseline = game.solve(depth=4, algorithm="minimax")
pruned = game.solve(depth=4, algorithm="alpha_beta", order="priority")
aware = game.solve(depth=4, order="opponent", lambda_weight=0.5)

print(baseline.best_value, pruned.best_value, aware.best_value)  # 3 3 3
print(pruned.best_action)                  # T3
print(aware.stats.visited_nodes)          # 搜索调用节点数
```

`TreasureGame.solve` 支持 `forward`、`reverse`、`priority` 和 `opponent` 等访问策略。对手感知排序只调整候选动作的搜索顺序，**不修改 Minimax 的终局评价值**。对于具有并列最优行动的场景，不同排序可能返回不同但同样最优的首步。

## 基准实验

| 实验 | 研究问题 | 主要变量 | 主要产物 |
|---|---|---|---|
| 启发式搜索对照 | 代价估计能否减少路径搜索开销？ | h=0、曼哈顿、加权可采纳启发式 | `results/tables/astar_benchmarks.csv` |
| 博弈剪枝对照 | 访问顺序如何影响 Alpha-Beta？ | Minimax / Alpha-Beta、正序 / 逆序 / 收益优先 | `results/tables/game_search_benchmarks.csv` |
| 对手感知排序 | 对手收益信息是否值得计算？ | λ 参数、开发集 / 独立验证集 | `results/extension/tables/opponent_aware_benchmarks.csv` |
| Landmark ALT | 更精确启发式的预处理何时回本？ | 地标数、地图规模、查询量 Q | `results/extension/tables/landmark_benchmarks.csv` |
| 可扩展性与案例 | 优化在更大博弈树中是否稳定？ | 4/6/8 个宝藏、搜索深度、排序策略 | `results/extension/tables/scalability_benchmarks.csv` |

生成的图片位于 `results/figures/` 和 `results/extension/figures/`。如需排查具体搜索行为，可查看 `results/extension/tables/trace_examples.json` 和 `results/extension/traces/*.jsonl`：后者记录节点扩展、子分支返回值、α/β 边界和剪枝事件。

### 指标与比较口径

- **A*：** `expanded_nodes` 是真正弹出并扩展的节点数；`generated_nodes` 统计入队次数（包含重复入队）。二者不应混用。
- **博弈搜索：** `visited_nodes` 是递归搜索访问的节点数，`evaluated_leaves` 是叶子或深度截断处的评价次数。
- **剪枝统计：** `cutoffs` 统计实际跳过至少一个后续分支的剪枝位置，`pruned_branches` 统计直接未访问的兄弟分支数；两者均不等于整棵被跳过子树的节点数量。
- **耗时：** 主要用于同一环境、同一输入配置下的相对比较；微小耗时差异可能受到运行环境影响。
- **Landmark 成本：** 总耗时包含地标索引预计算和在线路径查询，不能只看扩展节点减少比例。

全部场景保持确定性种子与可复查的实验输出；算法正确性由内置参考输入、随机场景及独立 Dijkstra/Minimax 对照共同验证。示例 CSV 与图表是仓库已有的运行结果，重新运行会受本机计时环境影响。

## 已观察到的权衡

- **对手信息并非越多越好。** 开发集选出的 λ=0.5 在独立验证任务上仅带来较小的平均节点数改善，同时引入明显额外计算开销。
- **Landmark 能缩小搜索空间，但不保证降低总耗时。** 地标预处理成本需要由足够多的查询摊销；地标数也并非越多越好。
- **剪枝排序依赖局面结构。** 即时收益排序在较大宝藏场景中能显著减少搜索，但对手感知策略也存在实际退化案例。完整轨迹被保留下来用于解释失败原因。

详细数据与实验限制请参考 `results/阶段三实验小结.md`、`results/extension/阶段四创新拓展实验小结.md`。

