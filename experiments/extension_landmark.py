"""实验 B：Landmark ALT 启发式、扩展节点及预处理/多查询收支平衡。"""
from __future__ import annotations
from pathlib import Path
from statistics import median
import random
from src.astar import astar_search, enter_cost, admissible_grid_heuristic
from src.landmark import LandmarkIndex
from .compare import make_weighted, write_csv
from .extension_scenarios import _largest_component

QUERIES=(1,10,50,100)


def run_landmark(output: Path, query_repeats: int = 2):
    rows,details=[],[]
    # 2种规模 × 2种障碍密度 × 2个随机种子 = 8幅固定地图。
    for side in (20,40):
        for rate in (0.10,0.27):
            for seed in (52,87):
                case=f'grid{side}_obstacles{rate:.2f}_seed{seed}'
                grid,_,_=make_weighted(side,seed,rate)
                grid[0][0]=1;grid[-1][-1]=1  # E=0 不适用于任意目标的多查询地图
                component=_largest_component(grid)
                rng=random.Random(seed+3000)
                pairs=[];seen=set()
                while len(pairs)<max(QUERIES):
                    a,b=rng.sample(component,2)
                    if (a,b) not in seen:
                        seen.add((a,b));pairs.append((a,b))
                configs=[('endpoint_aware',0,None,0.0)]
                for count in (2,4,8):
                    index=LandmarkIndex.build(grid,enter_cost,count)
                    configs.append(('ALT_plus_endpoint',count,index,index.precompute_ms))
                # 缓存同一输入的最优代价，确保每个策略/地标数解相同。
                oracle={}
                for method,count,index,pre_ms in configs:
                    query_records=[]
                    for qid,(start,goal) in enumerate(pairs):
                        base=admissible_grid_heuristic(grid,goal,enter_cost)
                        h=base if index is None else index.heuristic(goal,base=base)
                        calls=[astar_search(grid,start,goal,cost_fn=enter_cost,heuristic=h)
                               for _ in range(query_repeats)]
                        r=calls[0]
                        assert all(x.cost==r.cost for x in calls),'重复查询不一致'
                        if qid in oracle:
                            assert oracle[qid]==r.cost,(case,qid,method,count,oracle[qid],r.cost)
                        else:oracle[qid]=r.cost
                        timing=median(x.stats.elapsed_ms for x in calls)
                        query_records.append({'expanded':r.stats.expanded_nodes,'elapsed_ms':timing,'cost':r.cost})
                        details.append({'map':case,'side':side,'obstacle_rate':rate,'seed':seed,
                                        'heuristic':method,'landmarks':count,'query_id':qid,
                                        'start':str(start),'goal':str(goal),
                                        'optimal_cost':r.cost,'expanded_nodes':r.stats.expanded_nodes,
                                        'median_query_ms':round(timing,6),'query_repeats':query_repeats})
                    for Q in QUERIES:
                        prefix=query_records[:Q]
                        online_ms=sum(x['elapsed_ms'] for x in prefix)
                        rows.append({'map':case,'side':side,'obstacle_rate':rate,'seed':seed,
                                     'heuristic':method,'landmarks':count,'num_queries':Q,
                                     'precompute_ms':round(pre_ms,6),'online_ms':round(online_ms,6),
                                     'total_ms':round(pre_ms+online_ms,6),
                                     'mean_expanded_nodes':round(sum(x['expanded'] for x in prefix)/Q,3),
                                     'found_queries':sum(x['cost']>=0 for x in prefix),
                                     'query_repeats':query_repeats})
    table=output/'tables'
    write_csv(table/'landmark_benchmarks.csv',rows)
    write_csv(table/'landmark_query_detail.csv',details)
    return rows
