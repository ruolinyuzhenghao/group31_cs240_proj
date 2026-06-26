import math
import unittest

from tsp_algorithms import (
    build_distance_matrix,
    compare_algorithms,
    dfs_exact,
    double_tree,
    held_karp,
    nearest_neighbor,
    prim_mst,
    route_length,
)


class TSPAlgorithmsTest(unittest.TestCase):
    def test_square_optimum(self):
        points = [
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
        ]

        solution = held_karp(points)

        self.assertEqual(solution.route[0], 0)
        self.assertEqual(solution.route[-1], 0)
        self.assertEqual(set(solution.route[1:-1]), {1, 2, 3})
        self.assertTrue(math.isclose(solution.length, 4.0))

    def test_dfs_exact_matches_held_karp(self):
        points = [
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 2.0),
            (0.0, 2.0),
            (1.0, 1.0),
            (3.0, 1.0),
        ]

        std = dfs_exact(points)
        dynamic_programming = held_karp(points)

        self.assertTrue(math.isclose(std.length, dynamic_programming.length))
        self.assertEqual(set(std.route[1:-1]), set(range(1, len(points))))

    def test_heuristics_return_valid_tours(self):
        points = [
            (0.0, 0.0),
            (3.0, 0.0),
            (2.0, 2.0),
            (0.0, 3.0),
            (1.0, 1.0),
        ]
        distances = build_distance_matrix(points)

        for algorithm in (nearest_neighbor, double_tree):
            solution = algorithm(points)
            self.assertEqual(solution.route[0], 0)
            self.assertEqual(solution.route[-1], 0)
            self.assertEqual(set(solution.route[1:-1]), {1, 2, 3, 4})
            self.assertEqual(len(solution.route), len(points) + 1)
            self.assertTrue(
                math.isclose(solution.length, route_length(solution.route, distances))
            )

    def test_double_tree_respects_two_approximation_on_small_instance(self):
        points = [
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 1.0),
            (0.0, 2.0),
            (1.0, 1.0),
        ]

        approximate = double_tree(points)
        optimum = held_karp(points)

        self.assertLessEqual(approximate.length, 2 * optimum.length)

    def test_mst_has_n_minus_one_edges(self):
        points = [
            (0.0, 0.0),
            (1.0, 0.0),
            (2.0, 0.0),
            (3.0, 0.0),
        ]
        tree = prim_mst(build_distance_matrix(points))
        edge_count = sum(len(neighbors) for neighbors in tree) // 2

        self.assertEqual(edge_count, len(points) - 1)

    def test_compare_algorithms_can_include_dfs_std(self):
        points = [
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 2.0),
            (0.0, 2.0),
            (1.0, 1.0),
        ]

        runs = compare_algorithms(points, include_dfs_std=True)

        self.assertEqual(
            [run.profile.name for run in runs],
            ["Nearest neighbor", "MST double-tree", "Held-Karp", "DFS exact std"],
        )
        self.assertTrue(math.isclose(runs[2].solution.length, runs[3].solution.length))
        self.assertTrue(math.isclose(runs[3].approximation_ratio, 1.0))


if __name__ == "__main__":
    unittest.main()
