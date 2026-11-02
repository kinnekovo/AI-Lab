import unittest
from src.tree import TreeNode, sample_tree, evaluate_tree
from src.adversarial import search_game

class TestGameTrees(unittest.TestCase):
    def test_official_tree(self):
        for algorithm in ("minimax", "alpha_beta"):
            self.assertEqual(evaluate_tree(sample_tree(), algorithm=algorithm).best_value,5)

    def test_reverse_visitation_still_same_value(self):
        for order in ("forward","reverse"):
            for algorithm in ("minimax","alpha_beta"):
                self.assertEqual(evaluate_tree(sample_tree(),order=order,algorithm=algorithm).best_value,5)

    def test_alpha_beta_not_more_visits(self):
        for order in ("forward","reverse"):
            a=evaluate_tree(sample_tree(), algorithm="minimax",order=order)
            b=evaluate_tree(sample_tree(), algorithm="alpha_beta",order=order)
            self.assertLessEqual(b.stats.visited_nodes,a.stats.visited_nodes)

    def test_single_leaf_and_negative(self):
        leaf=TreeNode(value=-7)
        self.assertEqual(evaluate_tree(leaf).best_value,-7)
        tree=TreeNode(children=[TreeNode(children=[TreeNode(value=-7),TreeNode(value=4)]),
                                TreeNode(children=[TreeNode(value=2),TreeNode(value=1)])])
        self.assertEqual(evaluate_tree(tree).best_value,1)

    def test_empty_leaf_is_invalid(self):
        with self.assertRaises(ValueError):
            evaluate_tree(TreeNode())

    def test_wide_tree_crosscheck(self):
        vals=list(range(-12,13))
        for seed in (3,7,21):
            import random
            rng=random.Random(seed)
            def make_tree(depth):
                if depth==0:return TreeNode(value=rng.choice(vals))
                return TreeNode(children=[make_tree(depth-1) for _ in range(3)])
            root=make_tree(4)
            base=evaluate_tree(root,algorithm="minimax").best_value
            self.assertEqual(evaluate_tree(root,algorithm="alpha_beta").best_value,base)

if __name__ == "__main__": unittest.main()
