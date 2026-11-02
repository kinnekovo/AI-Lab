from __future__ import annotations
import heapq
import random
import unittest
from src.astar import astar_search, manhattan, zero_heuristic, enter_cost, admissible_grid_heuristic, treasure_enter_cost
from src.maps import MAZE_GRID, MAZE_START, MAZE_END, MAZE_EXPECTED, WEIGHTED_GRID, WEIGHTED_START, WEIGHTED_END
from src.problems import find_shortest_path, minimum_travel_cost


def verify_path(grid, path, start, end, *, cost_fn):
    assert path[0] == start and path[-1] == end
    total = 0
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        assert abs(r1-r2)+abs(c1-c2) == 1
        assert grid[r2][c2] != "X"
        total += cost_fn(grid[r2][c2])
    return total


def dijkstra_oracle(grid, start, goal, cost_fn):
    pq = [(0, start)]
    best = {start: 0}
    while pq:
        distance, point = heapq.heappop(pq)
        if distance != best[point]:
            continue
        if point == goal:
            return distance
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nr, nc = point[0]+dr, point[1]+dc
            if not (0<=nr<len(grid) and 0<=nc<len(grid[0])) or grid[nr][nc] == "X":
                continue
            nb = (nr,nc)
            new = distance+cost_fn(grid[nr][nc])
            if new < best.get(nb, float("inf")):
                best[nb]=new
                heapq.heappush(pq,(new,nb))
    return -1


class TestAStar(unittest.TestCase):
    def test_official_maze(self):
        path = find_shortest_path(MAZE_START, MAZE_END, MAZE_GRID)
        self.assertEqual(len(path), len(MAZE_EXPECTED))
        self.assertEqual(verify_path(MAZE_GRID, path, MAZE_START, MAZE_END, cost_fn=lambda _: 1), 7)

    def test_official_weighted(self):
        self.assertEqual(minimum_travel_cost(WEIGHTED_START, WEIGHTED_END, WEIGHTED_GRID), 7)
        result = astar_search(WEIGHTED_GRID, WEIGHTED_START, WEIGHTED_END, cost_fn=enter_cost,
                              heuristic=admissible_grid_heuristic(WEIGHTED_GRID, WEIGHTED_END, enter_cost))
        self.assertEqual(verify_path(WEIGHTED_GRID,result.path,WEIGHTED_START,WEIGHTED_END,cost_fn=enter_cost),7)

    def test_no_path_and_unreachable_weighted(self):
        grid = [["S","X"],["X","E"]]
        self.assertEqual(find_shortest_path((0,0),(1,1),grid),[])
        self.assertEqual(minimum_travel_cost((0,0),(1,1),grid),-1)

    def test_start_equals_goal(self):
        self.assertEqual(find_shortest_path((0,0),(0,0),[["S"]]),[(0,0)])
        self.assertEqual(minimum_travel_cost((0,0),(0,0),[["S"]]),0)

    def test_goal_enter_cost_zero_and_no_overestimate(self):
        grid=[["S","E"]]
        self.assertEqual(minimum_travel_cost((0,0),(0,1),grid),0)
        h=admissible_grid_heuristic(grid,(0,1),enter_cost)
        self.assertEqual(h((0,0),(0,1)),0)

    def test_mixed_numeric_and_string_cells(self):
        grid=[["S","3","E"],[1,1,1]]
        self.assertEqual(minimum_travel_cost((0,0),(0,2),grid),3)

    def test_invalid_shape(self):
        with self.assertRaises(ValueError):
            astar_search([["S"],[".","E"]],(0,0),(1,1))

    def test_random_weighted_against_independent_dijkstra(self):
        for seed in range(20):
            rng=random.Random(seed)
            grid=[["X" if rng.random()<0.18 else rng.randint(1,9) for _ in range(9)] for _ in range(9)]
            grid[0][0]="S"
            grid[-1][-1]="E"
            goal=(8,8)
            h=admissible_grid_heuristic(grid,goal,enter_cost)
            got=astar_search(grid,(0,0),goal,cost_fn=enter_cost,heuristic=h)
            expected=dijkstra_oracle(grid,(0,0),goal,enter_cost)
            self.assertEqual(got.cost,expected,f"seed={seed}")
            if got.found:
                self.assertEqual(verify_path(grid,got.path,(0,0),goal,cost_fn=enter_cost),expected)

    def test_treasure_cost_special_cells_equal_one(self):
        self.assertEqual(treasure_enter_cost("T1"),1)
        self.assertEqual(treasure_enter_cost("A"),1)
        self.assertEqual(treasure_enter_cost("B"),1)


if __name__ == "__main__": unittest.main()
