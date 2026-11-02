import unittest
from src.treasure import TreasureGame, compute_pairwise_distances
from src.maps import TREASURE_EXPECTED, TREASURE_ORDER, TREASURE_VALUES

EXPECTED_MATRIX = [
    [0,12,10,4,8,8],
    [12,0,8,8,4,10],
    [10,8,0,6,6,12],
    [4,8,6,0,6,6],
    [8,4,6,6,0,6],
    [8,10,12,6,6,0],
]

class TestTreasure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.distances=compute_pairwise_distances()
        cls.game=TreasureGame(cls.distances)

    def test_pairwise_matrix(self):
        self.assertEqual([[self.distances[a][b] for b in TREASURE_ORDER] for a in TREASURE_ORDER],EXPECTED_MATRIX)

    def test_official_h2_h4(self):
        for h, expected in TREASURE_EXPECTED.items():
            for algorithm in ("minimax", "alpha_beta"):
                r=self.game.solve(h,algorithm=algorithm)
                self.assertEqual(r.best_action,expected["best_target"])
                self.assertEqual(r.best_value,expected["best_value"])

    def test_all_orderings_utility_consistency(self):
        for depth in (2,4):
            reference=self.game.solve(depth,algorithm="minimax").best_value
            for order in ("forward","reverse","priority"):
                for algorithm in ("minimax","alpha_beta"):
                    self.assertEqual(self.game.solve(depth,algorithm=algorithm,order=order).best_value,reference)

    def test_state_updates_and_no_auto_pickup(self):
        state=self.game.root
        state=self.game.move(state,"T3")
        self.assertEqual(state.pos_a,"T3")
        self.assertEqual(state.score_a,TREASURE_VALUES["T3"]-self.distances["A"]["T3"])
        self.assertEqual(state.remaining_treasures,("T1","T2","T4"))
        self.assertEqual(state.current_player,"B")
        nxt=self.game.move(state,"T2")
        self.assertEqual(nxt.pos_b,"T2")
        self.assertEqual(nxt.pos_a,"T3")
        self.assertEqual(nxt.remaining_treasures,("T1","T4"))
        with self.assertRaises(ValueError): self.game.move(nxt,"T3")

    def test_no_other_player_blocking_and_full_depth(self):
        self.assertEqual(len(self.distances),6)
        self.assertEqual(self.game.solve(4,algorithm="minimax").stats.max_reached_depth,4)

if __name__ == "__main__": unittest.main()
