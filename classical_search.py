"""
Classical Search Algorithms and Baselines for Autonomous Rover Navigation.
Implements:
  - Uninformed Search: DFS (Depth-First Search), BFS (Breadth-First Search)
  - Informed Search: A* (Manhattan, Euclidean, Weighted A*, Energy-Aware)
  - Baselines: Greedy Best-First, Uniform Random Walk
Tracks search efficiency: nodes expanded, path length, path cost, runtime.
"""

import time
import heapq
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Set, Optional, Any
from environment import MarsRoverGridEnv


class SearchResult:
    def __init__(
        self,
        algorithm_name: str,
        success: bool,
        path: List[Tuple[int, int]],
        actions: List[int],
        nodes_expanded: int,
        path_cost: float,
        path_length: int,
        runtime_ms: float,
        visited_nodes: Set[Tuple[int, int]],
    ):
        self.algorithm_name = algorithm_name
        self.success = success
        self.path = path
        self.actions = actions
        self.nodes_expanded = nodes_expanded
        self.path_cost = path_cost
        self.path_length = path_length
        self.runtime_ms = runtime_ms
        self.visited_nodes = visited_nodes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm_name,
            "success": self.success,
            "nodes_expanded": self.nodes_expanded,
            "path_length": self.path_length,
            "path_cost": round(self.path_cost, 2),
            "runtime_ms": round(self.runtime_ms, 3),
        }


def depth_first_search(env: MarsRoverGridEnv) -> SearchResult:
    """Depth-First Search (Uninformed Graph Search with LIFO stack)."""
    t0 = time.perf_counter()
    start = env.start_pos
    goal = env.goal_pos

    stack = [(start, [start], [], 0.0)]
    visited = set()
    nodes_expanded = 0

    while stack:
        current, path, actions, cost = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        nodes_expanded += 1

        if current == goal:
            runtime_ms = (time.perf_counter() - t0) * 1000
            return SearchResult(
                "DFS", True, path, actions, nodes_expanded, cost, len(path) - 1, runtime_ms, visited
            )

        for neighbor, action_idx, step_cost in env.get_neighbors(current):
            if neighbor not in visited and not env.hazards[neighbor]:
                stack.append((neighbor, path + [neighbor], actions + [action_idx], cost + step_cost))

    runtime_ms = (time.perf_counter() - t0) * 1000
    return SearchResult("DFS", False, [], [], nodes_expanded, 0.0, 0, runtime_ms, visited)


def breadth_first_search(env: MarsRoverGridEnv) -> SearchResult:
    """Breadth-First Search (Uninformed Graph Search with FIFO queue - guarantees minimal hop count)."""
    t0 = time.perf_counter()
    start = env.start_pos
    goal = env.goal_pos

    queue = deque([(start, [start], [], 0.0)])
    visited = {start}
    nodes_expanded = 0

    while queue:
        current, path, actions, cost = queue.popleft()
        nodes_expanded += 1

        if current == goal:
            runtime_ms = (time.perf_counter() - t0) * 1000
            return SearchResult(
                "BFS", True, path, actions, nodes_expanded, cost, len(path) - 1, runtime_ms, visited
            )

        for neighbor, action_idx, step_cost in env.get_neighbors(current):
            if neighbor not in visited and not env.hazards[neighbor]:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor], actions + [action_idx], cost + step_cost))

    runtime_ms = (time.perf_counter() - t0) * 1000
    return SearchResult("BFS", False, [], [], nodes_expanded, 0.0, 0, runtime_ms, visited)


def compute_heuristic(
    pos: Tuple[int, int],
    goal: Tuple[int, int],
    heuristic_type: str,
    env: Optional[MarsRoverGridEnv] = None,
    weight: float = 1.0,
) -> float:
    """
    Computes heuristic estimate h(s) to goal.
    Supports: 'manhattan', 'euclidean', 'weighted_manhattan', 'energy_aware'
    """
    r, c = pos
    gr, gc = goal
    dr, dc = abs(r - gr), abs(c - gc)

    if heuristic_type == "manhattan":
        h = float(dr + dc)
    elif heuristic_type == "euclidean":
        h = float(np.sqrt(dr ** 2 + dc ** 2))
    elif heuristic_type == "weighted":
        h = weight * float(dr + dc)
    elif heuristic_type == "energy_aware":
        # Considers Manhattan distance plus minimum required uphill elevation delta
        dist_h = float(dr + dc)
        if env is not None:
            elev_diff = max(0.0, float(env.elevation[goal]) - float(env.elevation[pos]))
            h = dist_h + 1.5 * elev_diff
        else:
            h = dist_h
    else:
        raise ValueError(f"Unknown heuristic type: {heuristic_type}")
    return h


def a_star_search(
    env: MarsRoverGridEnv,
    heuristic_type: str = "manhattan",
    weight: float = 1.0,
) -> SearchResult:
    """
    A* Graph Search (Informed Search with Priority Queue).
    Evaluation function: f(n) = g(n) + h(n).
    Guarantees optimal cost if h(n) is admissible and consistent.
    """
    t0 = time.perf_counter()
    start = env.start_pos
    goal = env.goal_pos

    # Priority queue stores tuples: (f_score, counter, current_node, g_score, path, actions)
    counter = 0
    start_h = compute_heuristic(start, goal, heuristic_type, env, weight)
    open_set = [(start_h, counter, start, 0.0, [start], [])]

    g_scores = {start: 0.0}
    visited = set()
    nodes_expanded = 0

    algo_label = f"A* ({heuristic_type.capitalize()})"
    if heuristic_type == "weighted":
        algo_label = f"A* (Weighted w={weight})"

    while open_set:
        f_score, _, current, g_score, path, actions = heapq.heappop(open_set)

        if current in visited:
            continue
        visited.add(current)
        nodes_expanded += 1

        if current == goal:
            runtime_ms = (time.perf_counter() - t0) * 1000
            return SearchResult(
                algo_label, True, path, actions, nodes_expanded, g_score, len(path) - 1, runtime_ms, visited
            )

        for neighbor, action_idx, step_cost in env.get_neighbors(current):
            if env.hazards[neighbor]:
                continue

            tentative_g = g_score + step_cost
            if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                g_scores[neighbor] = tentative_g
                h_val = compute_heuristic(neighbor, goal, heuristic_type, env, weight)
                f_val = tentative_g + h_val
                counter += 1
                heapq.heappush(
                    open_set,
                    (f_val, counter, neighbor, tentative_g, path + [neighbor], actions + [action_idx]),
                )

    runtime_ms = (time.perf_counter() - t0) * 1000
    return SearchResult(algo_label, False, [], [], nodes_expanded, 0.0, 0, runtime_ms, visited)


def greedy_best_first_search(env: MarsRoverGridEnv) -> SearchResult:
    """Greedy Best-First Search: Evaluates nodes strictly by heuristic h(n). Fast but suboptimal."""
    t0 = time.perf_counter()
    start = env.start_pos
    goal = env.goal_pos

    counter = 0
    start_h = compute_heuristic(start, goal, "manhattan", env)
    open_set = [(start_h, counter, start, 0.0, [start], [])]
    visited = set()
    nodes_expanded = 0

    while open_set:
        h_score, _, current, cost, path, actions = heapq.heappop(open_set)

        if current in visited:
            continue
        visited.add(current)
        nodes_expanded += 1

        if current == goal:
            runtime_ms = (time.perf_counter() - t0) * 1000
            return SearchResult(
                "Greedy Baseline", True, path, actions, nodes_expanded, cost, len(path) - 1, runtime_ms, visited
            )

        for neighbor, action_idx, step_cost in env.get_neighbors(current):
            if neighbor not in visited and not env.hazards[neighbor]:
                h_val = compute_heuristic(neighbor, goal, "manhattan", env)
                counter += 1
                heapq.heappush(
                    open_set,
                    (h_val, counter, neighbor, cost + step_cost, path + [neighbor], actions + [action_idx]),
                )

    runtime_ms = (time.perf_counter() - t0) * 1000
    return SearchResult("Greedy Baseline", False, [], [], nodes_expanded, 0.0, 0, runtime_ms, visited)


def random_walk_baseline(env: MarsRoverGridEnv, max_steps: int = 500, seed: int = 42) -> SearchResult:
    """Uniform Random Policy Baseline: Takes random valid actions without planning."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    current = env.start_pos
    path = [current]
    actions = []
    total_cost = 0.0
    visited = {current}

    for step in range(max_steps):
        if current == env.goal_pos:
            runtime_ms = (time.perf_counter() - t0) * 1000
            return SearchResult(
                "Random Baseline", True, path, actions, step + 1, total_cost, len(path) - 1, runtime_ms, visited
            )

        neighbors = env.get_neighbors(current)
        valid_moves = [n for n in neighbors if not env.hazards[n[0]]]
        if not valid_moves:
            break

        chosen = valid_moves[rng.integers(0, len(valid_moves))]
        next_pos, action_idx, step_cost = chosen
        current = next_pos
        path.append(current)
        actions.append(action_idx)
        total_cost += step_cost
        visited.add(current)

    runtime_ms = (time.perf_counter() - t0) * 1000
    return SearchResult(
        "Random Baseline", False, path, actions, len(path), total_cost, len(path) - 1, runtime_ms, visited
    )
