"""拓展实验：最优性、定向 Landmark、排序不得修改得分、场景可复现。"""
from __future__ import annotations
import random
import unittest
from src.astar import astar_search,enter_cost,admissible_grid_heuristic
from src.landmark import LandmarkIndex,dijkstra_distances,select_landmarks
from src.maps import WEIGHTED_GRID,WEIGHTED_END,TREASURE_EXPECTED
from src.treasure import TreasureGame
from src.move_ordering import opponent_aware_priority
from experiments.extension_scenarios import scenario_game,scenario_description


class TestLandmark(unittest.TestCase):
    def test_asymmetric_reverse_distances(self):
        grid=[[1,9,1],[1,1,1]]
        src=(0,0); target=(0,1)
        self.assertEqual(dijkstra_distances(grid,src,enter_cost)[target],9)
        self.assertEqual(dijkstra_distances(grid,src,enter_cost,reverse=True)[target],1)
        self.assertEqual(dijkstra_distances(grid,target,enter_cost)[src],1)

    def test_goal_cost_zero_and_direction(self):
        index=LandmarkIndex.build(WEIGHTED_GRID,enter_cost,4)
        for r,row in enumerate(WEIGHTED_GRID):
            for c,x in enumerate(row):
                if x=='X':continue
                goal=WEIGHTED_END
                h=index.heuristic(goal,base=admissible_grid_heuristic(WEIGHTED_GRID,goal,enter_cost))
                got=astar_search(WEIGHTED_GRID,(r,c),goal,cost_fn=enter_cost,heuristic=h)
                true=dijkstra_distances(WEIGHTED_GRID,(r,c),enter_cost).get(goal,float('inf'))
                self.assertLessEqual(h((r,c),goal),true+1e-8)
                self.assertEqual(got.cost,true)

    def test_random_maps_oracle_and_consistency(self):
        rng=random.Random(995)
        for seed in range(6):
            grid=[['X' if rng.random()<0.16 else rng.randint(1,9) for _ in range(8)] for _ in range(8)]
            grid[0][0]='S';grid[-1][-1]='E'
            index=LandmarkIndex.build(grid,enter_cost,4)
            valid=[(r,c) for r,row in enumerate(grid) for c,x in enumerate(row) if x!='X']
            for goal in rng.sample(valid,3):
                h=index.heuristic(goal,base=admissible_grid_heuristic(grid,goal,enter_cost))
                # 所有可通行点 h <= 真正最短距离；包括进入 E=0 的地图。
                for start in valid:
                    oracle=dijkstra_distances(grid,start,enter_cost).get(goal,float('inf'))
                    self.assertLessEqual(h(start,goal),oracle+1e-8)
                for start in rng.sample(valid,8):
                    r=astar_search(grid,start,goal,cost_fn=enter_cost,heuristic=h)
                    expected=dijkstra_distances(grid,start,enter_cost).get(goal,-1)
                    self.assertEqual(r.cost,expected,(seed,start,goal))

    def test_count_and_validity(self):
        grid=[[1,'X',1],[1,1,1]]
        self.assertEqual(len(select_landmarks(grid,4)),4)
        self.assertEqual(len(set(select_landmarks(grid,4))),4)
        with self.assertRaises(ValueError):select_landmarks(grid,6)


class TestMoveOrdering(unittest.TestCase):
    def test_lambda_zero_is_greedy(self):
        game=TreasureGame();root=game.root
        for target in root.remaining_treasures:
            self.assertEqual(opponent_aware_priority(game,root,target,0),game.gain(root,target))

    def test_negative_lambda(self):
        with self.assertRaises(ValueError):
            TreasureGame().solve(4,order='opponent',lambda_weight=-1)

    def test_official_best_value_unchanged(self):
        game=TreasureGame()
        for depth,expected in TREASURE_EXPECTED.items():
            for lam in (0,.25,.5,1,2,4,8):
                r=game.solve(depth,order='opponent',lambda_weight=lam)
                self.assertEqual(r.best_value,expected['best_value'])

    def test_scenarios_reproducible_and_correct(self):
        for seed in (101,201,301):
            self.assertEqual(scenario_description(seed),scenario_description(seed))
            for count in (4,6,8):
                game=scenario_game(seed,count)
                self.assertEqual(len(game.root.remaining_treasures),count)
                for h in (2,4):
                    baseline=game.solve(h,algorithm='minimax').best_value
                    for order,lam in [('forward',0),('priority',0),('opponent',.5),('opponent',2)]:
                        self.assertEqual(game.solve(h,order=order,lambda_weight=lam).best_value,baseline)

    def test_trace_does_not_change_counts(self):
        game=TreasureGame();events=[]
        a=game.solve(4,order='opponent',trace=events.append)
        b=game.solve(4,order='opponent')
        self.assertEqual((a.best_value,a.stats.visited_nodes),(b.best_value,b.stats.visited_nodes))
        self.assertTrue(any(x['event']=='expand' for x in events))
        self.assertTrue(any(x['event']=='cutoff' for x in events))

if __name__=='__main__':unittest.main()
