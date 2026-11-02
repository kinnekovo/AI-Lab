"""实验 A：收益排序 / 对手感知排序的消融、参数敏感性、独立验证。"""
from __future__ import annotations
from statistics import median
from pathlib import Path
from src.treasure import TreasureGame
from .compare import write_csv
from .extension_scenarios import scenario_game

LAMBDAS=(0,0.25,0.5,1,2,4,8)


def eval_search(game, depth, order, lam, repeats):
    results=[game.solve(depth,order=order,lambda_weight=lam) for _ in range(repeats)]
    vals={r.best_value for r in results}
    assert len(vals)==1,'重复实验结果不一致'
    r=results[0]
    return r,round(median(v.stats.elapsed_ms for v in results),6)


def row_for(case, split, treasure_count, depth, method, lam, game, repeats, oracle):
    r,t=eval_search(game,depth,method,lam,repeats)
    if r.best_value!=oracle.best_value:
        raise AssertionError(f'排序改变了最优值：{case},{depth},{method},{lam}')
    return {'case':case,'split':split,'treasures':treasure_count,'depth':depth,
            'method':method,'lambda':lam if method=='opponent' else '',
            'best_action':r.best_action,'optimal_utility':r.best_value,
            'visited_nodes':r.stats.visited_nodes,'evaluated_leaves':r.stats.evaluated_leaves,
            'cutoffs':r.stats.cutoffs,'pruned_branches':r.stats.pruned_branches,
            'median_elapsed_ms':t,'repeats':repeats}


def run_ordering(output: Path, repeats: int = 5):
    out=output/'tables';out.mkdir(parents=True,exist_ok=True)
    development=[]
    for seed in (101,102,103,104,105,106):
        game=scenario_game(seed,6)
        for depth in (4,6):
            case=f'custom_seed{seed}_n6_h{depth}'
            oracle=game.solve(depth,algorithm='minimax')
            for method,lam in [('forward',0),('priority',0)]+[('opponent',x) for x in LAMBDAS]:
                development.append(row_for(case,'development',6,depth,method,lam,game,repeats,oracle))
    # 在开发集依据各场景中相比即时收益排序的节点数平均相对值选参。
    # 不使用最终验证集作选参；相同目标值情况下取更小 λ。
    grouping={(r['case'],r['depth']):r['visited_nodes'] for r in development if r['method']=='priority'}
    scores={lam:sum(r['visited_nodes']/grouping[r['case'],r['depth']]
                    for r in development if r['method']=='opponent' and r['lambda']==lam)
                / len(grouping) for lam in LAMBDAS}
    selected=min(LAMBDAS,key=lambda lam:(scores[lam],lam))
    holdout=[]
    game=TreasureGame()
    for depth in (2,4):
        oracle=game.solve(depth,algorithm='minimax')
        for method,lam in [('forward',0),('priority',0),('opponent',selected)]:
            holdout.append(row_for('official','official',4,depth,method,lam,game,repeats,oracle))
    for seed in (201,202,203,204,205,206,207,208):
        for count in (6,8):
            game=scenario_game(seed,count)
            for depth in (4,6):
                oracle=game.solve(depth,algorithm='minimax')
                case=f'holdout_seed{seed}_n{count}_h{depth}'
                for method,lam in [('forward',0),('priority',0),('opponent',selected)]:
                    holdout.append(row_for(case,'holdout',count,depth,method,lam,game,repeats,oracle))
    write_csv(out/'opponent_aware_benchmarks.csv',development+holdout)
    write_csv(out/'lambda_sensitivity.csv',[
        {'lambda':lam,'mean_relative_nodes_vs_greedy':round(scores[lam],6),
         'selected':int(lam==selected),'scenarios':len(grouping)} for lam in LAMBDAS])
    return development+holdout,selected,scores
