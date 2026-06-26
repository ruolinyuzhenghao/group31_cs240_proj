# Running Instructions

## 1. Environment Setup

Python 3.10 or later is recommended. This project uses only the Python
standard library.

## 2. Run the Algorithm Example

Run the following command in the project directory:

```bash
python tsp_algorithms.py
```

The program uses the built-in delivery-point example and runs:

- Nearest-neighbor greedy algorithm
- MST double-tree 2-approximation algorithm
- Held-Karp exact dynamic programming algorithm
- DFS exact standard-answer search

The output includes the route length, approximation ratio relative to the
optimal solution, running time, and visiting order.

`DFS exact std` is a brute-force correctness check for small inputs. It
enumerates all possible visiting orders and returns the shortest route, so it
should match the Held-Karp optimum on small examples.


## 3. Use in Code

```python
from tsp_algorithms import compare_algorithms

points = [
    (0.0, 0.0),  # depot
    (2.0, 3.0),
    (5.0, 4.0),
    (1.0, 7.0),
]

for run in compare_algorithms(points):
    print(run.profile.name, run.solution.route, run.solution.length)
```
