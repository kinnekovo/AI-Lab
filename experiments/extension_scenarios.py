"""固定随机种子下构造可达、可复现的 4/6/8 宝藏场景。"""
from __future__ import annotations
from collections import deque
from functools import lru_cache
import random
from src.astar import astar_search, admissible_grid_heuristic, treasure_enter_cost
from src.treasure import TreasureGame


def _largest_component(grid):
    rows,cols=len(grid),len(grid[0])
    seen=set()
    groups=[]
    for r in range(rows):
        for c in range(cols):
            if (r,c) in seen or grid[r][c]=='X':continue
            q=deque([(r,c)]); seen.add((r,c)); comp=[]
            while q:
                p=q.popleft();comp.append(p)
                for dr,dc in ((0,1),(0,-1),(1,0),(-1,0)):
                    nx=(p[0]+dr,p[1]+dc)
                    if 0<=nx[0]<rows and 0<=nx[1]<cols and nx not in seen and grid[nx[0]][nx[1]]!='X':
                        seen.add(nx);q.append(nx)
            groups.append(comp)
    return max(groups,key=len)


@lru_cache(maxsize=None)
def _full_scenario(seed: int):
    rng=random.Random(seed)
    side=12
    grid=[]
    for r in range(side):
        row=[]
        for c in range(side):
            if rng.random()<0.14:row.append('X')
            else:row.append(rng.choice([1,1,1,1,1,1,2,3,5,8]))
        grid.append(row)
    # 保证至少一个大的连通分量，同时保留内部随机障碍。
    for c in range(side):grid[0][c]=1
    for r in range(side):grid[r][side-1]=1
    component=_largest_component(grid)
    options=[pos for pos in component if grid[pos[0]][pos[1]]==1]
    if len(options)<10:raise RuntimeError('可选位置不足')
    # A/B 分置两侧，防止双方总是从同一位置出发。
    a=min(options,key=lambda p:(p[0]+p[1],p))
    b=max((p for p in options if p!=a),key=lambda p:(p[0]+p[1],p))
    options=[p for p in options if p not in (a,b)]
    treasure_positions=rng.sample(options,8)
    points={'A':a,'B':b,**{f'T{i+1}':p for i,p in enumerate(treasure_positions)}}
    values={f'T{i+1}':rng.randint(6,15) for i in range(8)}
    for k,p in points.items():grid[p[0]][p[1]]=k
    distances={}
    for source,pos in points.items():
        distances[source]={}
        for target,tpos in points.items():
            h=admissible_grid_heuristic(grid,tpos,treasure_enter_cost)
            result=astar_search(grid,pos,tpos,cost_fn=treasure_enter_cost,heuristic=h)
            if result.cost<0:raise RuntimeError('连通性断言失败')
            distances[source][target]=int(result.cost)
    return grid,points,values,distances


def scenario_game(seed: int, count: int) -> TreasureGame:
    if count not in (4,6,8):raise ValueError('只支持 4/6/8 宝藏')
    _,_,all_values,all_distances=_full_scenario(seed)
    values={key:all_values[key] for key in list(all_values)[:count]}
    keep=['A','B']+list(values)
    distances={a:{b:all_distances[a][b] for b in keep} for a in keep}
    return TreasureGame(distances,values)


def scenario_description(seed: int) -> dict:
    grid,points,values,_=_full_scenario(seed)
    return {'seed':seed,'grid':grid,'points':points,'treasure_values':values,
            'grid_side':len(grid),'obstacle_count':sum(cell=='X' for row in grid for cell in row)}
