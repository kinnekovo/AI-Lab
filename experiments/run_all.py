"""一键运行单元测试、官方样例与阶段三性能对照，保存日志。"""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys
from .compare import benchmark
from src.problems import find_shortest_path, minimum_travel_cost, alpha_beta_best_score, solve_treasure
from src.maps import MAZE_GRID,MAZE_START,MAZE_END,WEIGHTED_GRID,WEIGHTED_START,WEIGHTED_END
from src.tree import sample_tree


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--repeats",type=int,default=30)
    p.add_argument("--output",type=Path,default=Path("results"))
    args=p.parse_args()
    logs=args.output/"logs"
    logs.mkdir(parents=True,exist_ok=True)
    test=subprocess.run([sys.executable,"-m","unittest","discover","-s","tests","-v"],
                        text=True,capture_output=True)
    log=test.stdout+test.stderr
    (logs/"test_output.txt").write_text(log,encoding="utf-8")
    print(log)
    if test.returncode:
        raise SystemExit("单元测试失败，已停止生成性能报告")

    official = "\n".join([
        f"题目一（迷宫）: {find_shortest_path(MAZE_START,MAZE_END,MAZE_GRID)}",
        f"题目二（加权）: {minimum_travel_cost(WEIGHTED_START,WEIGHTED_END,WEIGHTED_GRID)}",
        f"题目三（博弈树）: {alpha_beta_best_score(sample_tree())}",
        f"题目四（H=2）: {solve_treasure(2)}",
        f"题目四（H=4）: {solve_treasure(4)}",
    ])
    print(official)
    (logs/"official_outputs.txt").write_text(official+"\n",encoding="utf-8")
    benchmark(repeats=args.repeats,output=args.output)

if __name__=="__main__": main()
