"""Algorithms for the metric TSP delivery-route model.

The input is a list of 2D points.  Point 0 is treated as the depot, and the
returned route always starts at 0, visits every other point once, and ends at 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import hypot
from time import perf_counter
from typing import Callable, Iterable, List, Sequence, Tuple


Point = Tuple[float, float]
DistanceMatrix = List[List[float]]


@dataclass(frozen=True)
class TSPSolution:
    """A TSP route and its total length."""

    route: List[int]
    length: float


@dataclass(frozen=True)
class AlgorithmProfile:
    """Theoretical properties summarized from the algorithm design note."""

    name: str
    time_complexity: str
    space_complexity: str
    quality: str
    best_for: str


@dataclass(frozen=True)
class AlgorithmRun:
    """One algorithm's measured result on a concrete input instance."""

    profile: AlgorithmProfile
    solution: TSPSolution
    elapsed_seconds: float
    approximation_ratio: float | None


ALGORITHM_PROFILES = {
    "nearest_neighbor": AlgorithmProfile(
        name="Nearest neighbor",
        time_complexity="O(n^2)",
        space_complexity="O(n)",
        quality="No constant approximation guarantee",
        best_for="Fast baseline for larger instances",
    ),
    "double_tree": AlgorithmProfile(
        name="MST double-tree",
        time_complexity="O(n^2 log n)",
        space_complexity="O(n^2)",
        quality="2-approximation for metric TSP",
        best_for="Polynomial-time route with a provable bound",
    ),
    "held_karp": AlgorithmProfile(
        name="Held-Karp",
        time_complexity="O(n^2 * 2^n)",
        space_complexity="O(n * 2^n)",
        quality="Exact optimum",
        best_for="Small instances and benchmarking",
    ),
    "dfs_exact": AlgorithmProfile(
        name="DFS exact std",
        time_complexity="O(n!)",
        space_complexity="O(n)",
        quality="Exact optimum by exhaustive search",
        best_for="Small-instance correctness checking",
    ),
}


def euclidean_distance(a: Point, b: Point) -> float:
    """Return the Euclidean distance between two plane points."""

    return hypot(a[0] - b[0], a[1] - b[1])


def build_distance_matrix(points: Sequence[Point]) -> DistanceMatrix:
    """Build the complete metric distance matrix for the given points."""

    n = len(points)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            distance = euclidean_distance(points[i], points[j])
            matrix[i][j] = distance
            matrix[j][i] = distance
    return matrix


def route_length(route: Sequence[int], distances: DistanceMatrix) -> float:
    """Return the total length of a route represented by point indices."""

    return sum(distances[route[i]][route[i + 1]] for i in range(len(route) - 1))


def _validate_points(points: Sequence[Point]) -> None:
    if not points:
        raise ValueError("points must contain at least the depot at index 0")


def nearest_neighbor(points: Sequence[Point]) -> TSPSolution:
    """Nearest-neighbor greedy heuristic.

    At each step this chooses the closest unvisited point from the current
    location.  Ties are broken by the smaller point index for reproducibility.
    """

    _validate_points(points)
    distances = build_distance_matrix(points)
    n = len(points)

    current = 0
    route = [0]
    unvisited = set(range(1, n))

    while unvisited:
        next_point = min(unvisited, key=lambda point: (distances[current][point], point))
        route.append(next_point)
        unvisited.remove(next_point)
        current = next_point

    route.append(0)
    return TSPSolution(route=route, length=route_length(route, distances))


def prim_mst(distances: DistanceMatrix) -> List[List[int]]:
    """Compute a minimum spanning tree of a complete weighted graph.

    The graph is supplied as a distance matrix.  The returned adjacency list
    contains the tree edges.
    """

    n = len(distances)
    if n == 0:
        raise ValueError("distances must not be empty")

    in_tree = [False] * n
    min_weight = [float("inf")] * n
    parent = [-1] * n
    min_weight[0] = 0.0

    for _ in range(n):
        u = -1
        for candidate in range(n):
            if not in_tree[candidate] and (
                u == -1 or min_weight[candidate] < min_weight[u]
            ):
                u = candidate

        if u == -1:
            raise ValueError("graph is disconnected")

        in_tree[u] = True
        for v, weight in enumerate(distances[u]):
            if not in_tree[v] and weight < min_weight[v]:
                min_weight[v] = weight
                parent[v] = u

    tree = [[] for _ in range(n)]
    for v in range(1, n):
        u = parent[v]
        tree[u].append(v)
        tree[v].append(u)

    for neighbors in tree:
        neighbors.sort()
    return tree


def double_tree(points: Sequence[Point]) -> TSPSolution:
    """MST double-tree 2-approximation for metric TSP.

    Doubling MST edges, taking an Euler tour, and shortcutting repeated
    vertices is equivalent to visiting the MST in preorder.  This implementation
    uses that compact preorder formulation.
    """

    _validate_points(points)
    distances = build_distance_matrix(points)
    tree = prim_mst(distances)

    route = []
    stack = [(0, -1)]
    while stack:
        node, parent = stack.pop()
        route.append(node)
        for neighbor in reversed(tree[node]):
            if neighbor != parent:
                stack.append((neighbor, node))

    route.append(0)
    return TSPSolution(route=route, length=route_length(route, distances))


def held_karp(points: Sequence[Point]) -> TSPSolution:
    """Exact Held-Karp dynamic programming algorithm.

    This is exponential in the number of delivery points, so it is intended for
    small instances and benchmarking.
    """

    _validate_points(points)
    distances = build_distance_matrix(points)
    vertex_count = len(points)
    delivery_count = vertex_count - 1

    if delivery_count == 0:
        return TSPSolution(route=[0, 0], length=0.0)

    dp: dict[tuple[int, int], float] = {}
    parent: dict[tuple[int, int], int] = {}

    for vertex in range(1, vertex_count):
        mask = 1 << (vertex - 1)
        dp[(mask, vertex)] = distances[0][vertex]

    for subset_size in range(2, delivery_count + 1):
        for subset in combinations(range(1, vertex_count), subset_size):
            mask = _vertices_to_mask(subset)
            for endpoint in subset:
                previous_mask = mask ^ (1 << (endpoint - 1))
                best_length = float("inf")
                best_previous = -1

                for previous in subset:
                    if previous == endpoint:
                        continue

                    candidate = (
                        dp[(previous_mask, previous)] + distances[previous][endpoint]
                    )
                    if candidate < best_length:
                        best_length = candidate
                        best_previous = previous

                dp[(mask, endpoint)] = best_length
                parent[(mask, endpoint)] = best_previous

    full_mask = (1 << delivery_count) - 1
    best_length = float("inf")
    best_endpoint = -1

    for endpoint in range(1, vertex_count):
        candidate = dp[(full_mask, endpoint)] + distances[endpoint][0]
        if candidate < best_length:
            best_length = candidate
            best_endpoint = endpoint

    route_without_depot = _reconstruct_held_karp_route(
        parent=parent,
        full_mask=full_mask,
        endpoint=best_endpoint,
    )
    route = [0, *route_without_depot, 0]
    return TSPSolution(route=route, length=best_length)


def dfs_exact(
    points: Sequence[Point],
    *,
    max_delivery_points: int = 9,
) -> TSPSolution:
    """Exhaustive DFS exact solver used as the standard answer on small cases.

    This recursively enumerates every possible visiting order, so it is much
    slower than Held-Karp.  It is useful because the logic is simple enough to
    serve as an independent correctness check for small inputs.
    """

    _validate_points(points)
    distances = build_distance_matrix(points)
    vertex_count = len(points)
    delivery_count = vertex_count - 1

    if delivery_count > max_delivery_points:
        raise ValueError(
            "dfs_exact is factorial-time; use it only for small instances or "
            "raise max_delivery_points explicitly"
        )

    if delivery_count == 0:
        return TSPSolution(route=[0, 0], length=0.0)

    best_route: List[int] | None = None
    best_length = float("inf")
    route = [0]
    visited = [False] * vertex_count
    visited[0] = True

    def search(current: int, current_length: float) -> None:
        nonlocal best_route, best_length

        if len(route) == vertex_count:
            total_length = current_length + distances[current][0]
            if total_length < best_length:
                best_length = total_length
                best_route = [*route, 0]
            return

        if current_length >= best_length:
            return

        for next_vertex in range(1, vertex_count):
            if visited[next_vertex]:
                continue

            visited[next_vertex] = True
            route.append(next_vertex)
            search(
                current=next_vertex,
                current_length=current_length + distances[current][next_vertex],
            )
            route.pop()
            visited[next_vertex] = False

    search(current=0, current_length=0.0)
    if best_route is None:
        raise RuntimeError("DFS did not find a route")

    return TSPSolution(route=best_route, length=best_length)


def compare_algorithms(
    points: Sequence[Point],
    *,
    exact_limit: int = 12,
    include_dfs_std: bool = False,
    dfs_limit: int = 9,
) -> List[AlgorithmRun]:
    """Run the three methods and expose their practical/theoretical tradeoffs.

    Held-Karp is only run when the number of delivery points is at most
    ``exact_limit``.  When the exact optimum is available, heuristic runs also
    report ``approximation_ratio = heuristic_length / optimum_length``.  Set
    ``include_dfs_std`` to include the slower DFS standard answer for small
    correctness checks.
    """

    _validate_points(points)
    delivery_count = len(points) - 1
    algorithms: list[tuple[str, Callable[[Sequence[Point]], TSPSolution]]] = [
        ("nearest_neighbor", nearest_neighbor),
        ("double_tree", double_tree),
    ]
    if delivery_count <= exact_limit:
        algorithms.append(("held_karp", held_karp))
    if include_dfs_std and delivery_count <= dfs_limit:
        algorithms.append(("dfs_exact", lambda input_points: dfs_exact(input_points)))

    runs = []
    optimum_length = None

    for key, algorithm in algorithms:
        started_at = perf_counter()
        solution = algorithm(points)
        elapsed_seconds = perf_counter() - started_at

        if key in {"held_karp", "dfs_exact"} and optimum_length is None:
            optimum_length = solution.length

        runs.append(
            AlgorithmRun(
                profile=ALGORITHM_PROFILES[key],
                solution=solution,
                elapsed_seconds=elapsed_seconds,
                approximation_ratio=None,
            )
        )

    if optimum_length is None:
        return runs

    return [
        AlgorithmRun(
            profile=run.profile,
            solution=run.solution,
            elapsed_seconds=run.elapsed_seconds,
            approximation_ratio=run.solution.length / optimum_length
            if optimum_length > 0
            else 1.0,
        )
        for run in runs
    ]


def _vertices_to_mask(vertices: Iterable[int]) -> int:
    mask = 0
    for vertex in vertices:
        mask |= 1 << (vertex - 1)
    return mask


def _reconstruct_held_karp_route(
    parent: dict[tuple[int, int], int],
    full_mask: int,
    endpoint: int,
) -> List[int]:
    route_reversed = []
    mask = full_mask
    current = endpoint

    while current != -1:
        route_reversed.append(current)
        previous = parent.get((mask, current), -1)
        mask ^= 1 << (current - 1)
        current = previous

    route_reversed.reverse()
    return route_reversed


if __name__ == "__main__":
    sample_points = [
        (0.0, 0.0),
        (2.0, 3.0),
        (5.0, 4.0),
        (1.0, 7.0),
        (6.0, 8.0),
        (8.0, 1.0),
        (3.0, 9.0),
        (9.0, 6.0),
        (4.0, 1.0),
    ]

    for run in compare_algorithms(sample_points, include_dfs_std=True):
        ratio = (
            f"{run.approximation_ratio:.3f}"
            if run.approximation_ratio is not None
            else "n/a"
        )
        print(
            f"{run.profile.name:16s} "
            f"length={run.solution.length:7.3f} "
            f"ratio={ratio:>5s} "
            f"time={run.elapsed_seconds:.6f}s "
            f"route={run.solution.route}"
        )
        print(
            f"{'':16s} "
            f"time={run.profile.time_complexity}, "
            f"space={run.profile.space_complexity}, "
            f"quality={run.profile.quality}"
        )
