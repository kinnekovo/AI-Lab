"""完整阶段四拓展实验；复现：python -m experiments.run_extensions --repeats 5 --query-repeats 2。"""
from __future__ import annotations
import argparse
from pathlib import Path
from .extension_ordering import run_ordering
from .extension_landmark import run_landmark
from .extension_scalability import run_scalability
from .extension_report import generate_report
from .plot_extensions import plot_extensions


def run_all_extensions(output:Path,repeats:int=5,query_repeats:int=2):
    if repeats<1 or query_repeats<1:raise ValueError('重复次数必须 >=1')
    output.mkdir(parents=True,exist_ok=True)
    print('[A] 对手感知排序：开发集调参、官方实例及独立验证集',flush=True)
    ordering,selected,scores=run_ordering(output,repeats)
    print(f'    开发集选择 lambda={selected}；记录 {len(ordering)} 行',flush=True)
    print('[B] Landmark：有向距离预计算、8 幅随机地图与每图 100 个查询',flush=True)
    landmark=run_landmark(output,query_repeats)
    print(f'    完成 {len(landmark)} 行查询规模汇总',flush=True)
    print('[C] 规模扩展：10 个随机种子、4/6/8 宝藏和不同深度，记录真实剪枝轨迹',flush=True)
    scaling,examples=run_scalability(output,selected,repeats)
    print(f'    完成 {len(scaling)} 行搜索对照',flush=True)
    generate_report(output,ordering,selected,scores,landmark,scaling,examples,repeats,query_repeats)
    plot_extensions(output,ordering,landmark,scaling,scores,selected)
    print('阶段四完成；数据、五张图与实验小结位于',output.resolve(),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('results/extension'))
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--query-repeats',type=int,default=2)
    args=parser.parse_args()
    run_all_extensions(args.output,args.repeats,args.query_repeats)
