"""绘制四种结果图。可从 compare.py 自动调用或手动调用。"""
from __future__ import annotations
import csv
from pathlib import Path


def plot_all(astar_rows, game_rows, output_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi":150,"savefig.dpi":180,"font.size":10})

    # 无权网格：Dijkstra vs Manhattan
    group=[r for r in astar_rows if r["question"]==1]
    cases=list(dict.fromkeys(r["case"] for r in group))
    zero=[next(r for r in group if r["case"]==c and r["heuristic"]=="zero")["expanded_nodes"] for c in cases]
    mh=[next(r for r in group if r["case"]==c and r["heuristic"]=="manhattan")["expanded_nodes"] for c in cases]
    _grouped_bars(plt,np,cases,zero,mh,"Zero heuristic (Dijkstra)","Manhattan heuristic",
                  "Maze A*: expanded nodes",output_dir/"01_maze_heuristics.png")

    group=[r for r in astar_rows if r["question"]==2]
    cases=list(dict.fromkeys(r["case"] for r in group))
    zero=[next(r for r in group if r["case"]==c and r["heuristic"]=="zero")["expanded_nodes"] for c in cases]
    custom=[next(r for r in group if r["case"]==c and r["heuristic"]=="endpoint_aware")["expanded_nodes"] for c in cases]
    _grouped_bars(plt,np,cases,zero,custom,"Zero heuristic (Dijkstra)","Endpoint-aware lower bound",
                  "Weighted A*: expanded nodes",output_dir/"02_weighted_heuristics.png")

    group=[r for r in game_rows if r["question"]==3]
    cases=list(dict.fromkeys(r["case"] for r in group))
    base=[next(r for r in group if r["case"]==c and r["algorithm"]=="minimax" and r["visit_order"]=="forward")["visited_nodes"] for c in cases]
    pruned=[next(r for r in group if r["case"]==c and r["algorithm"]=="alpha_beta" and r["visit_order"]=="forward")["visited_nodes"] for c in cases]
    _grouped_bars(plt,np,cases,base,pruned,"Minimax","Alpha-Beta",
                  "Game tree: nodes visited (left-to-right)",output_dir/"03_tree_pruning.png")

    # 综合题：不同顺序和深度的节点访问量
    group=[r for r in game_rows if r["question"]==4 and r["algorithm"]=="alpha_beta"]
    labels=[f"H={h}" for h in (2,4)]
    orders=("forward","reverse","priority")
    arrays=[[next(r for r in group if r["depth"]==h and r["visit_order"]==order)["visited_nodes"] for h in (2,4)] for order in orders]
    fig,ax=plt.subplots(figsize=(7.2,4.2))
    x=np.arange(2)
    w=0.22
    for i,order in enumerate(orders):
        bars=ax.bar(x+(i-1)*w,arrays[i],width=w,label=order.replace("priority","greedy gain"))
        ax.bar_label(bars,padding=2,fontsize=8)
    ax.set_xticks(x,labels)
    ax.set_ylabel("Visited nodes")
    ax.set_title("Treasure Alpha-Beta: depth and move ordering")
    ax.legend(frameon=False)
    ax.set_ylim(top=max(max(v) for v in arrays)*1.2)
    fig.tight_layout()
    fig.savefig(output_dir/"04_treasure_depth_and_order.png")
    plt.close(fig)


def _grouped_bars(plt,np,cases,first,second,first_name,second_name,title,path):
    x=np.arange(len(cases))
    fig,ax=plt.subplots(figsize=(9.2,4.5))
    for dx,series,label in ((-0.19,first,first_name),(0.19,second,second_name)):
        bars=ax.bar(x+dx,series,0.36,label=label)
        ax.bar_label(bars,padding=2,fontsize=8)
    ax.set_xticks(x,cases,rotation=12,ha="right")
    ax.set_ylabel("Expanded / visited nodes")
    ax.set_ylim(top=max(first+second)*1.19)
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--output",default="results")
    args=p.parse_args()
    output=Path(args.output)
    with (output/"tables"/"astar_benchmarks.csv").open(encoding="utf-8-sig",newline="") as f:
        a=list(csv.DictReader(f))
    with (output/"tables"/"game_search_benchmarks.csv").open(encoding="utf-8-sig",newline="") as f:
        b=list(csv.DictReader(f))
    for row in a:
        for k in ("question","expanded_nodes"):
            row[k]=int(row[k])
    for row in b:
        for k in ("question","depth","visited_nodes"):
            row[k]=int(row[k])
    plot_all(a,b,output/"figures")
