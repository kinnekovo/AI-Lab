"""绘制创新拓展实验统计图，英文标签兼容未安装中文字库的环境。"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_extensions(output: Path, ordering, landmark, scaling, scores, selected):
    path=output/'figures';path.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'figure.dpi':135,'savefig.dpi':170,'font.size':10})

    # 1 参数敏感性。1.0 代表与收益贪心的平均访问节点一致。
    x=sorted(scores); y=[scores[a] for a in x]
    fig,ax=plt.subplots(figsize=(7.5,4.4))
    ax.plot(range(len(x)),y,marker='o',linewidth=1.8)
    ax.axhline(1,linestyle='--',linewidth=1,label='Greedy gain baseline')
    ax.scatter([x.index(selected)],[scores[selected]],s=100,marker='*',label=f'Selected: {selected}')
    ax.set_xticks(range(len(x)),[str(v) for v in x]);ax.set_xlabel('Opponent weight (lambda)')
    ax.set_ylabel('Mean visited nodes / greedy nodes')
    ax.set_title('A: Parameter sensitivity (development set)');ax.legend(frameon=False)
    fig.tight_layout();fig.savefig(path/'01_lambda_sensitivity.png');plt.close(fig)

    # 2 独立评估：各实例相对 greedy 的节点数。
    grouped=defaultdict(dict)
    for r in ordering:
        if r['split']=='holdout':grouped[r['case'],r['depth']][r['method']]=r
    ratios=defaultdict(list)
    for key,group in grouped.items():
        if 'priority' not in group or 'opponent' not in group:continue
        c=group['priority']['visited_nodes'];a=group['opponent']['visited_nodes']
        label='8 treasures' if group['priority']['treasures']==8 else '6 treasures'
        ratios[label].append((c-a)/c*100)
    fig,ax=plt.subplots(figsize=(7.5,4.4))
    labels=list(ratios)
    data=[ratios[l] for l in labels]
    if data:
        ax.boxplot(data,tick_labels=labels,showmeans=True)
    ax.axhline(0,linestyle='--',linewidth=1)
    ax.set_ylabel('Node reduction vs greedy (%)')
    ax.set_title('A: Generalization to held-out treasure maps')
    fig.tight_layout();fig.savefig(path/'02_opponent_holdout.png');plt.close(fig)

    # 3 Landmark: 按地图平均在线节点数量，Q=100。
    g=defaultdict(list)
    for r in landmark:
        if r['num_queries']==100:
            g[(r['heuristic'],r['landmarks'])].append(r['mean_expanded_nodes'])
    order=[('endpoint_aware',0)]+[('ALT_plus_endpoint',k) for k in (2,4,8)]
    names=['Endpoint','ALT-2','ALT-4','ALT-8'];vals=[mean(g[k]) for k in order]
    fig,ax=plt.subplots(figsize=(7.5,4.4))
    bars=ax.bar(names,vals)
    ax.bar_label(bars,fmt='%.1f',padding=3)
    ax.set_ylabel('Mean nodes expanded per query')
    ax.set_title('B: Heuristic strength across eight maps (100 queries)')
    fig.tight_layout();fig.savefig(path/'03_landmark_nodes.png');plt.close(fig)

    # 4 Landmark: 以固定 40x40 地图检查时间收支（每幅地图每种参数同 Q）。
    groups=defaultdict(list)
    for r in landmark:
        if r['side']==40:
            groups[(r['heuristic'],r['landmarks'],r['num_queries'])].append(r['total_ms'])
    fig,ax=plt.subplots(figsize=(7.5,4.4))
    for name,key in zip(names,order):
        ax.plot([1,10,50,100],[median(groups[(key[0],key[1],Q)]) for Q in (1,10,50,100)],marker='o',label=name)
    ax.set_xlabel('Number of queries on the same map')
    ax.set_ylabel('Total time including preprocessing (ms)')
    ax.set_title('B: Landmark preprocessing vs query savings (40x40)')
    ax.legend(frameon=False);fig.tight_layout();fig.savefig(path/'04_landmark_break_even.png');plt.close(fig)

    # 5 规模效应：h=4 和 h=6 分面独立，不混合不同深度。
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.6))
    for ax,depth in zip(axes,(4,6)):
        for algorithm,order,label in [('minimax','forward','Minimax'),('alpha_beta','forward','AB original'),
                                      ('alpha_beta','priority','AB greedy'),('alpha_beta','opponent','AB opponent')]:
            coords=[]
            for n in (4,6,8):
                if depth>n:continue
                sub=[r['visited_nodes'] for r in scaling if r['treasures']==n and r['depth']==depth
                     and r['algorithm']==algorithm and r['ordering']==order]
                if sub:coords.append((n,median(sub)))
            if coords:ax.plot([x for x,_ in coords],[y for _,y in coords],marker='o',label=label)
        ax.set_yscale('log');ax.set_xticks([4,6,8]);ax.set_xlabel('Treasures')
        ax.set_ylabel('Median visited nodes (log)')
        ax.set_title(f'C: Depth H={depth}')
    axes[1].legend(frameon=False,fontsize=8,loc='upper left')
    fig.tight_layout();fig.savefig(path/'05_scalability.png');plt.close(fig)
