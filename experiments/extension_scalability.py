"""实验 C：4/6/8 宝藏 × 2/4/6 深度；独立测试与真实剪枝路径示例。"""
from __future__ import annotations
import json
from pathlib import Path
from statistics import median
from .compare import write_csv
from .extension_ordering import eval_search
from .extension_scenarios import scenario_game, scenario_description


def run_scalability(output: Path, selected_lambda: float, repeats: int = 5):
    rows=[]
    for seed in (301,302,303,304,305,306,307,308,309,310):
        for count in (4,6,8):
            game=scenario_game(seed,count)
            for depth in (2,4,6):
                if depth>count:continue
                # 最优值由完全枚举的 Minimax 确定；深度 6 时它的耗时
                # 较大，仅重复 2 次，避免与剪枝策略对比时被误称等重复。
                m_results=[game.solve(depth,algorithm='minimax') for _ in range(min(repeats,2))]
                reference=m_results[0]
                assert all(r.best_value==reference.best_value for r in m_results)
                elapsed=median(r.stats.elapsed_ms for r in m_results)
                case=f'seed{seed}_n{count}_h{depth}'
                rows.append({'case':case,'seed':seed,'treasures':count,'depth':depth,
                             'algorithm':'minimax','ordering':'forward','lambda':'',
                             'best_action':reference.best_action,'optimal_utility':reference.best_value,
                             'visited_nodes':reference.stats.visited_nodes,
                             'evaluated_leaves':reference.stats.evaluated_leaves,
                             'cutoffs':reference.stats.cutoffs,
                             'pruned_branches':reference.stats.pruned_branches,
                             'median_elapsed_ms':round(elapsed,6),'repeats':len(m_results)})
                for order,lam in [('forward',0),('priority',0),('opponent',selected_lambda)]:
                    results=[game.solve(depth,order=order,lambda_weight=lam) for _ in range(repeats)]
                    r=results[0]
                    if r.best_value!=reference.best_value:
                        raise AssertionError(f'{case}:{order} 得分与 Minimax 不同')
                    rows.append({'case':case,'seed':seed,'treasures':count,'depth':depth,
                                 'algorithm':'alpha_beta','ordering':order,
                                 'lambda':lam if order=='opponent' else '',
                                 'best_action':r.best_action,'optimal_utility':r.best_value,
                                 'visited_nodes':r.stats.visited_nodes,
                                 'evaluated_leaves':r.stats.evaluated_leaves,
                                 'cutoffs':r.stats.cutoffs,
                                 'pruned_branches':r.stats.pruned_branches,
                                 'median_elapsed_ms':round(median(x.stats.elapsed_ms for x in results),6),
                                 'repeats':repeats})
    write_csv(output/'tables'/'scalability_benchmarks.csv',rows)
    examples=write_trace_examples(output, rows, selected_lambda)
    return rows,examples


def write_trace_examples(output: Path, rows: list[dict], lam: float):
    # 只从阶段 C 独立场景中找效益最好与最差的两例；没有变差时如实写“最弱改进”。
    groups={}
    for r in rows:
        if r['algorithm']!='alpha_beta':continue
        groups.setdefault(r['case'],{})[r['ordering']]=r
    candidates=[]
    for name,g in groups.items():
        if 'priority' not in g or 'opponent' not in g:continue
        greedy=g['priority'];new=g['opponent']
        difference=greedy['visited_nodes']-new['visited_nodes']
        candidates.append((difference/greedy['visited_nodes'],name,g))
    positive=max(candidates,key=lambda x:x[0]); negative=min(candidates,key=lambda x:x[0])
    examples=[]
    (output/'traces').mkdir(parents=True,exist_ok=True)
    for role,(fraction,name,g) in [('most_effective',positive),('least_effective',negative)]:
        r=g['opponent'];game=scenario_game(r['seed'],r['treasures'])
        traces={}
        for method in ('priority','opponent'):
            trace=[]
            game.solve(r['depth'],order=method,lambda_weight=lam,
                       trace=lambda event:trace.append(event) if len(trace)<1400 else None)
            traces[method]=trace
            with (output/'traces'/f'{role}_{method}.jsonl').open('w',encoding='utf-8') as f:
                for ev in trace:f.write(json.dumps(ev,ensure_ascii=False,allow_nan=True)+'\n')
        examples.append({'role':role,'case':name,'seed':r['seed'],'treasures':r['treasures'],
                         'depth':r['depth'],'relative_node_reduction':round(fraction,6),
                         'greedy_nodes':g['priority']['visited_nodes'],
                         'opponent_nodes':r['visited_nodes'],
                         'greedy_cutoffs':g['priority']['cutoffs'],
                         'opponent_cutoffs':r['cutoffs'],
                         'greedy_root_order':next((x['choices'] for x in traces['priority'] if x['event']=='expand' and x['depth']==0),''),
                         'opponent_root_order':next((x['choices'] for x in traces['opponent'] if x['event']=='expand' and x['depth']==0),''),
                         'trace_events_captured':{k:len(v) for k,v in traces.items()}})
    (output/'tables'/'trace_examples.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2),encoding='utf-8')
    return examples
