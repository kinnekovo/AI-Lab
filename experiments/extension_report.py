"""使用实测数据形成可复现的阶段四分析草稿，避免事先假设优化必然有效。"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from statistics import mean,median


def generate_report(output:Path,order_rows,selected,scores,landmark_rows,scaling,examples,repeats,qrepeats):
    development=[r for r in order_rows if r['split']=='development']
    hold=[r for r in order_rows if r['split']=='holdout']
    grouped=defaultdict(dict)
    for r in hold:
        if r['split']=='holdout':grouped[r['case'],r['depth']][r['method']]=r
    diffs=[];times=[]
    for g in grouped.values():
        c=g['priority'];n=g['opponent']
        diffs.append((c['visited_nodes']-n['visited_nodes'])/c['visited_nodes']*100)
        times.append((c['median_elapsed_ms']-n['median_elapsed_ms'])/c['median_elapsed_ms']*100)
    win=sum(x>0 for x in diffs);tie=sum(x==0 for x in diffs);loss=sum(x<0 for x in diffs)
    time_factors=[g['opponent']['median_elapsed_ms']/g['priority']['median_elapsed_ms'] for g in grouped.values()]
    def fm(x):return f'{x:.2f}'
    lines=['# 阶段四：创新与拓展实验——实测结果及分析',
        '', '> 自动根据 CSV 统计生成的报告素材；这是实测结果，不是保证各策略提升的预期。',
        '', '## 实验条件、正确性与指标',
        '', f'- A、C 中 Alpha-Beta 各配置重复 {repeats} 次，记录耗时中位数；C 中 Minimax 为控制开销仅重复 {min(repeats,2)} 次（不可视为与其他配置同次数的时间实验）。',
        f'- B 中各查询重复 {qrepeats} 次取耗时中位数，预计算执行一次，前 Q 次的中位数耗时求和；合计时间为预处理 + 在线查询，具体机器上会有波动。',
        '- A：开发集和独立验证集随机种子互斥；选参只使用开发集；官方样例单独验证。',
        '- 评价函数始终为 Score_A − Score_B；排序仅影响访问顺序；所有求解结果均以不剪枝的 Minimax 最优收益差做断言。',
        '- B：Landmark 对有向进入代价同时预计算正向/反向距离；每组查询所有启发函数得到相同最优路径代价。',
        '- 生成的随机地图在保证连通性的前提下随机设置障碍和权值；属于自设拓展场景，不替代官方测试。',
        '', '## A. 对手感知节点排序：消融与参数选择',
        '', '| λ | 开发集平均节点数 / 即时收益基线 | 是否选中 |',
        '|---:|---:|:---:|',
    ]
    for lam,score in sorted(scores.items()):lines.append(f'| {lam} | {score:.4f} | {"是" if lam==selected else ""} |')
    lines.extend(['',f'依据开发集各场景的平均相对节点数，选择 **λ={selected}**，固定用于独立验证；搜索范围有限，不能称为全局最优。',
        f'独立验证集共 **{len(diffs)}** 个任务实例；对手感知相对收益优先排序：**{win}** 个节点数减少，**{tie}** 个持平，**{loss}** 个增加。',
        f'平均节点数相对减少 {fm(mean(diffs))}%，中位减少 {fm(median(diffs))}%；对手感知耗时 / 贪心耗时的中位倍率 **{fm(median(time_factors))}×**、平均倍率 **{fm(mean(time_factors))}×**。',
        '注意：节点数减少不等于耗时一定更短；对手收益估计也需要额外计算。',
        '', '## B. Landmark 启发式：节点扩展与预处理成本',
        '', '| 启发函数 | Q=100 时跨地图平均扩展节点/查询 | 40×40 跨地图 Q=100 的平均总耗时(ms) |',
        '|---|---:|---:|'])
    configs=[('endpoint_aware',0,'Endpoint-aware')]+[('ALT_plus_endpoint',i,f'ALT-{i}+Endpoint') for i in (2,4,8)]
    for method,n,name in configs:
        sub=[r for r in landmark_rows if r['num_queries']==100 and r['heuristic']==method and r['landmarks']==n]
        high=[r['total_ms'] for r in sub if r['side']==40]
        lines.append(f'| {name} | {mean(x["mean_expanded_nodes"] for x in sub):.2f} | {mean(high):.2f} |')
    # 查询次数的盈亏分析，不对偶然达成的时间结果做过度承诺。
    lines.extend(['','**分地图总耗时低于原启发式的数量（包含预处理）：**','',
                  '| 地标数量 | Q=1 | Q=10 | Q=50 | Q=100 |','|---:|---:|---:|---:|---:|'])
    for n in (2,4,8):
        vals=[]
        for q in (1,10,50,100):
            baseline={r['map']:r for r in landmark_rows if r['num_queries']==q and r['landmarks']==0}
            sub=[r for r in landmark_rows if r['num_queries']==q and r['landmarks']==n]
            wins=sum(r['total_ms']<baseline[r['map']]['total_ms'] for r in sub)
            vals.append(f'{wins}/{len(sub)}')
        lines.append(f'| {n} | '+ ' | '.join(vals)+' |')
    lines.extend(['','解读：Landmark 的启发函数理论上提供更强的下界，但 Python 中预处理和每节点计算多个地标距离会产生额外成本；应区分扩展节点、在线查询与端到端总耗时。',
                  '由于存在有向路径代价，反向距离不能简单照搬正向表；若只预计算 d(L,n)，就无法使用完整双向 ALT 下界。',
                  '', '## C. 多宝藏规模扩展与搜索行为',
                  '', '| 宝藏数 | H | Minimax 中位节点 | AB 原始中位 | AB 即时收益中位 | AB 对手感知中位 |',
                  '|---:|---:|---:|---:|---:|---:|'])
    for count in (4,6,8):
        for h in (2,4,6):
            if h>count:continue
            def md(algorithm,ordering):
                sub=[r['visited_nodes'] for r in scaling if r['treasures']==count and r['depth']==h and r['algorithm']==algorithm and r['ordering']==ordering]
                return int(median(sub))
            lines.append(f'| {count} | {h} | {md("minimax","forward")} | {md("alpha_beta","forward")} | {md("alpha_beta","priority")} | {md("alpha_beta","opponent")} |')
    lines.extend(['','**搜索轨迹个案（选取阶段 C 中对手感知相比贪心的最好和最弱表现）：**',''])
    scale_lookup={(r['case'],r['ordering']):r for r in scaling if r['algorithm']=='alpha_beta'}
    for x in examples:
        direction='减少' if x['relative_node_reduction']>0 else '增加' if x['relative_node_reduction']<0 else '不变'
        old=scale_lookup[(x['case'],'priority')];new=scale_lookup[(x['case'],'opponent')]
        lines.extend([f'- `{x["role"]}`：{x["case"]}，节点数由 {x["greedy_nodes"]} → {x["opponent_nodes"]}（相对{direction} {abs(x["relative_node_reduction"])*100:.2f}%）。',
                      f'  - 贪心根节点访问顺序：`{x["greedy_root_order"]}`；对手感知：`{x["opponent_root_order"]}`。',
                      f'  - 叶子评估 {old["evaluated_leaves"]} → {new["evaluated_leaves"]}，剪枝位置数 {x["greedy_cutoffs"]} → {x["opponent_cutoffs"]}，直接跳过分支数 {old["pruned_branches"]} → {new["pruned_branches"]}。',
                      f'  - 注意剪枝位置更多不代表访问节点更少：不同剪枝位置覆盖的子树大小不同。详细 `expand`、`child_result` 与 `cutoff` 事件在 `traces/{x["role"]}_*.jsonl`（每策略最多 1400 条，截断仅影响展示，不影响实际统计）。'])
    lines.extend(['','**讨论提示：**排序主要影响 alpha/beta 上下界的更新时机，较早遇到强分支可能触发更多有效剪枝；但是单纯比较 cutoff 数不充分，还应同时比较每次截断跳过的子分支数量、访问叶子数与实际路径。',
                 '', '## 局限与结论',
                 '', '- 两种改进均为成熟思想的任务化应用，不声称提出全新通用搜索算法。',
                 '- 开发集调参后的提升不保证在验证集保持；验证集上的负面结果必须保留。',
                 '- 随机生成场景不代表所有真实博弈任务；较大宝藏数和更深搜索会造成阶乘增长。',
                 '- 跨设备运行时间不宜直接比较，节点统计更可复现；Landmark 的端到端成本仍应一起报告。',
                 '', '## 文件索引',
                 '', '- `tables/opponent_aware_benchmarks.csv`、`tables/lambda_sensitivity.csv`',
                 '- `tables/landmark_benchmarks.csv`、`tables/landmark_query_detail.csv`',
                 '- `tables/scalability_benchmarks.csv`、`tables/trace_examples.json`',
                 '- `traces/*` 和 `figures/01..05_*.png`',
                 ''])
    (output/'阶段四创新拓展实验小结.md').write_text('\n'.join(lines),encoding='utf-8')
