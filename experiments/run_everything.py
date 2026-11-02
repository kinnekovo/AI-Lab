"""一条命令完成：测试、官方样例、阶段三对照、阶段四拓展。"""
from __future__ import annotations
import argparse
from pathlib import Path
from .run_all import main as stage3_main
from .run_extensions import run_all_extensions

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--repeats',type=int,default=30,help='阶段三单配置重复次数')
    p.add_argument('--extension-repeats',type=int,default=5,help='阶段四 A/C 单配置重复次数')
    p.add_argument('--query-repeats',type=int,default=2,help='阶段四 B 单次查询重复次数')
    args=p.parse_args()
    # 阶段三主程序接收 sys.argv，直接执行子进程便于可靠地保留日志。
    import subprocess,sys
    subprocess.run([sys.executable,'-m','experiments.run_all','--repeats',str(args.repeats)],check=True)
    run_all_extensions(Path('results/extension'),args.extension_repeats,args.query_repeats)
