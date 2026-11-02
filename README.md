# 当代人工智能实验二：搜索算法（阶段一至四完整工程）

基于《实验二要求_new.pdf》完成四道官方题目，附阶段三的基线对照和阶段四的**三个创新拓展实验 A/B/C**。项目包含 Python 源码、测试、可复现实验脚本、原始 CSV、实测图表、具体剪枝轨迹和中文分析文档。

> 注意：优化并非在所有场景都提升。报告保留了减速和节点增加的结果，不能将“搜索节点下降”直接等同于“实际耗时下降”。不包含独立排版的正式期末提交报告。

## 1. 环境和一键运行

- Python >= 3.10；核心实现只使用标准库；绘图依赖 `numpy` 和 `matplotlib`。
- 本次实测环境：Python 3.13.5、numpy 2.3.5、matplotlib 3.10.8（具体运行时间受机器影响）。
- 在解压后的 `AI-Experiment-2` 根目录打开 VS Code/PowerShell 终端：

```powershell
python -m pip install -r requirements.txt

# 单元测试（目前 29 项）
python -m unittest discover -s tests -v

# 完整复现：阶段三基线 + 阶段四创新 + CSV + 图表 + 中文小结
python -m experiments.run_everything --repeats 30 --extension-repeats 5 --query-repeats 2
```

如果只需重跑阶段四：

```powershell
python -m experiments.run_extensions --repeats 5 --query-repeats 2
```

如果只需重跑阶段一至三：

```powershell
python -m experiments.run_all --repeats 30
```

以上命令从项目根目录执行。可使用 `py -m ...` 代替 `python -m ...`。阶段四涉及多次 A* 与多宝藏全树搜索，比原来阶段三明显更耗时。

## 2. 官方题目和验收结果

| 题目 | 预期与实测结果 | 入口 |
|---|---|---|
| 1. 迷宫最短路 | 7 步，返回 8 个坐标的最短路径 | `src/problems.py::find_shortest_path` |
| 2. 加权地形 | 最低代价 7；进入 E 代价 0 | `src/problems.py::minimum_travel_cost` |
| 3. 博弈树 | 最佳 MAX 得分 5 | `src/problems.py::alpha_beta_best_score` |
| 4. 宝藏分配 | H=2 → T2/-1；H=4 → T3/3 | `src/problems.py::solve_treasure` |

官方地图在 `src/maps.py`；官方距离矩阵在 `results/tables/treasure_distance_matrix.csv`，测试不以硬编码结果替代算法。

## 3. 阶段四创新 A：对手感知排序与消融

新增 `src/move_ordering.py`，并扩展 `TreasureGame.solve(order='opponent',lambda_weight=...)`：

`priority = 当前玩家即时收益 − λ × 对手下一步最大可能即时收益`。

这只是 **Alpha-Beta 的子节点排序规则**，不是叶子评价函数；仍使用 `Score_A − Score_B`，所有结果与未剪枝 Minimax 交叉校验。

- 开发集：6 个固定随机种子的 6 宝藏实例，H=4/6，扫描 λ=0/0.25/0.5/1/2/4/8。
- 选参：最小化开发集“相对即时收益基线的平均访问节点比例”，仅用开发集选参。
- 独立验证：8 个未参与调参的随机种子，6/8 宝藏，H=4/6，共 32 个任务实例；官方 4 宝藏 H=2/4 另外核验。
- 消融：原始顺序、只用自身即时收益、额外加入对手收益；记录访问节点、叶子评估、有效剪枝位置、跳过分支、耗时中位数。

本次实测 λ=0.5；独立验证 **18 胜/9 平/5 负**（以访问节点数减少为胜），平均减少约 1.11%，但中位运行时间约为即时收益排序的 3.11 倍。参见 `results/extension/阶段四创新拓展实验小结.md`。

## 4. 阶段四创新 B：Landmark ALT 启发式

新增 `src/landmark.py`。由于进入格子计费导致路径代价有向非对称，必须分别预计算 **d(L,n)** 和 **d(n,L)**，组合：

```text
h_ALT(n,t) = max_L(0, d(L,t)-d(L,n), d(n,L)-d(t,L))
h_final(n,t) = max(h_ALT(n,t), h_endpoint_aware(n,t))
```

- 8 幅固定地图：20×20/40×40 × 两档障碍概率 × 两个种子。
- 每幅地图有 100 个固定查询；对照地标数 0/2/4/8。
- 查询量为 Q=1/10/50/100；必须将**预处理耗时 + 在线搜索耗时**作为端到端总代价。
- 每次查询重复 2 次取耗时中位数；Q 批次使用相同查询序列前缀，保证公平。
- 强化最优性测试：官方 E=0 以及多张随机加权地图均与 Dijkstra 真值核对；同一地图的所有启发函数必须得到相同最优值。

本次 Q=100 平均扩展节点：Endpoint-aware **216.97**、ALT-2 **106.07**、ALT-4 **68.85**、ALT-8 **53.35**。但是 40×40、Q=100 下的平均含预处理总耗时，ALT-8 比原方案更长，属于应保留的负面结果。

## 5. 阶段四创新 C：宝藏规模适应性与失败分析

新增 `experiments/extension_scenarios.py`，在 12×12 随机地形上构造可达的 4/6/8 宝藏场景，所有两两距离仍由 A* 实际计算。

- 10 个固定随机种子，分别搜索 H=2/4/6（H 不大于宝藏数）。
- 比较 Minimax、Alpha-Beta 原始顺序、即时收益优先、开发集选出的对手感知排序。
- 保存每个场景的原始结果，而不仅是跨种子均值。
- 自动挑选对手感知排序最有效和最弱（可能退化）的实例，记录实际搜索轨迹，包括扩展次序、每条子分支返回值、α/β 边界与实际剪枝事件。
- 可用 `results/extension/traces/*.jsonl` 追踪案例；每个策略最多保存前 1400 个事件，截断只影响展示，不影响节点总数。

当前实测 8 宝藏 H=6：Minimax **28,961**、原始 Alpha-Beta **3,364**、即时收益排序 **806**、对手感知 **809**（均为跨 10 种子节点数中位数）。对手感知不是所有规模下的最优策略。

## 6. 文件结构

```text
AI-Experiment-2/
├── src/
│   ├── astar.py                 # 原有 A* / 可采纳启发函数 / 路径还原
│   ├── adversarial.py           # 通用 Minimax、Alpha-Beta、可选路径事件追踪
│   ├── treasure.py              # 通用 TreasureGame（支持自定义宝藏价值/距离）
│   ├── move_ordering.py         # [A] 对手感知排序启发式
│   ├── landmark.py              # [B] 有向 Landmark 预计算与启发式
│   ├── tree.py, maps.py, problems.py
│   └── __init__.py
├── tests/
│   ├── test_astar.py, test_adversarial.py, test_treasure.py
│   └── test_extension.py       # [新增] 最优性、随机状态、轨迹测试
├── experiments/
│   ├── compare.py, plot_results.py, run_all.py     # 原阶段三
│   ├── extension_ordering.py                       # [A] 消融和独立验证
│   ├── extension_landmark.py                       # [B] 多查询成本分析
│   ├── extension_scenarios.py                      # [C] 可复现场景
│   ├── extension_scalability.py                    # [C] 规模与失效案例
│   ├── extension_report.py, plot_extensions.py     # [新增] 报告/图表
│   ├── run_extensions.py, run_everything.py        # [新增] 一键运行
│   └── __init__.py
├── results/
│   ├── tables/, figures/, logs/, 阶段三实验小结.md
│   └── extension/
│       ├── tables/               # 6 个创新实验 CSV/JSON
│       ├── figures/              # 5 张创新实验图表
│       ├── traces/               # 4 个成败案例剪枝追踪文件
│       └── 阶段四创新拓展实验小结.md
├── requirements.txt
└── README.md
```

## 7. 数据口径、局限与报告建议

- `visited_nodes` 为递归调用节点数，`evaluated_leaves` 为叶子/截断评价数；`cutoffs` 是有效跳过兄弟分支的剪枝位置数，不能把它当作跳过的整个子树节点数。另记录 `pruned_branches`。
- A/C 各排序配置重复 5 次；C 的 Minimax 为避免过大开销仅重复 2 次；阶段三基线 30 次。不同重复数下不宜直接将微小的耗时差异当作可靠结论。
- B 的 `total_ms = 单次预处理 + 查询序列中各查询耗时中位数求和`，不是多轮完整系统重启耗时的统计估计。重复运行会有浮动。
- 成败案例由实测节点数自动选出，**不属于随机样本上的普遍规律**。算法排序最优值一致，但最优首步在有多个并列最优解时可能不同。
- 数值表和图可用于正式报告；讲解时建议遵循“观察现象 → 假设 → 消融验证 → 解释负面结果/成本 → 适用边界”的顺序。
- 实验一讲评特别强调复现条件、边界参数与有意义的失败案例。本工程保留了中间参数与负面案例，避免只汇报最优配置。
