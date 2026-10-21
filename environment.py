"""
MarsRoverGridEnv: Autonomous Planetary Rover Environment
Designed for Machine Learning (23MEE305) Assignment.
Supports Classical Search (DFS, BFS, A*) and Reinforcement Learning (Q-Learning, PPO).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional, List


class MarsRoverGridEnv(gym.Env):
    """
    Planetary Rover Grid Navigation Environment with:
      - 2D Elevation gradient (uphill moves cost extra battery energy)
      - Crater / Sandtrap hazard zones with severe damage penalties
      - Stochastic wind slip / wheel slippage probability
      - Configurable reward shaping: 'sparse', 'naive', 'pbrs' (Potential-Based Reward Shaping)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    # Actions: 0: Up, 1: Right, 2: Down, 3: Left
    ACTION_DELTAS = [
        (-1, 0),  # 0: Up (decreases row y)
        (0, 1),   # 1: Right (increases col x)
        (1, 0),   # 2: Down (increases row y)
        (0, -1),  # 3: Left (decreases col x)
    ]
    ACTION_NAMES = ["Up", "Right", "Down", "Left"]

    def __init__(
        self,
        grid_size: int = 10,
        start_pos: Tuple[int, int] = (0, 0),
        goal_pos: Optional[Tuple[int, int]] = None,
        crater_hazard_rate: float = 0.15,
        slip_probability: float = 0.0,
        wind_direction: Optional[int] = 1,  # e.g., 1: Right drift
        reward_mode: str = "pbrs",           # 'sparse', 'naive', 'pbrs'
        max_steps: int = 150,
        elevation_scale: float = 2.0,
        seed: Optional[int] = 42,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.start_pos = start_pos
        self.goal_pos = goal_pos if goal_pos is not None else (grid_size - 1, grid_size - 1)
        self.crater_hazard_rate = crater_hazard_rate
        self.slip_probability = slip_probability
        self.wind_direction = wind_direction
        self.reward_mode = reward_mode
        self.max_steps = max_steps
        self.elevation_scale = elevation_scale
        self.rng = np.random.default_rng(seed)

        # Action space: 4 discrete directions
        self.action_space = spaces.Discrete(4)

        # Continuous Observation Space for PPO / Neural Networks:
        # [norm_r, norm_c, elevation, dist_to_goal, hazard_flag]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 5.0, np.sqrt(2.0), 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Generate Terrain (Elevation and Hazards)
        self._init_terrain(seed=seed)

        # Runtime state
        self.current_pos = self.start_pos
        self.current_step = 0

    def _init_terrain(self, seed: Optional[int] = 42):
        gen_rng = np.random.default_rng(seed)
        N = self.grid_size

        # 1. Elevation Map: Synthetic Martian ridge using 2D sinusoidal + slope
        y_coords, x_coords = np.mgrid[0:N, 0:N]
        slope = (y_coords + x_coords) / (2.0 * N)
        ridge = 0.5 * np.sin(np.pi * x_coords / (N / 2)) * np.cos(np.pi * y_coords / (N / 2))
        self.elevation = (slope + ridge - np.min(slope + ridge)) * self.elevation_scale

        # 2. Hazard Map: Crater/Sandtrap locations (lethal or high damage)
        self.hazards = np.zeros((N, N), dtype=bool)
        num_hazards = int((N * N) * self.crater_hazard_rate)

        # Candidate locations (exclude start and goal)
        candidates = [
            (r, c)
            for r in range(N)
            for c in range(N)
            if (r, c) != self.start_pos and (r, c) != self.goal_pos
        ]
        chosen_indices = gen_rng.choice(len(candidates), size=num_hazards, replace=False)
        for idx in chosen_indices:
            r, c = candidates[idx]
            # Ensure path exists by not blocking start/goal immediately adjacent cells completely
            if (r, c) not in [(0, 1), (1, 0), (N - 2, N - 1), (N - 1, N - 2)]:
                self.hazards[r, c] = True

    def state_to_discrete_index(self, pos: Tuple[int, int]) -> int:
        """Helper for tabular Q-learning: converts (r, c) to unique integer state."""
        return pos[0] * self.grid_size + pos[1]

    def discrete_index_to_pos(self, state_idx: int) -> Tuple[int, int]:
        """Helper to convert discrete integer index back to (r, c)."""
        return (state_idx // self.grid_size, state_idx % self.grid_size)

    def _get_obs(self) -> np.ndarray:
        r, c = self.current_pos
        N = self.grid_size
        norm_r = r / (N - 1)
        norm_c = c / (N - 1)
        elev = float(self.elevation[r, c])
        gr, gc = self.goal_pos
        dist_to_goal = np.sqrt((r - gr) ** 2 + (c - gc) ** 2) / (np.sqrt(2) * (N - 1))
        hazard = 1.0 if self.hazards[r, c] else 0.0
        return np.array([norm_r, norm_c, elev, dist_to_goal, hazard], dtype=np.float32)

    def _get_potential(self, pos: Tuple[int, int]) -> float:
        """
        Potential function Phi(s) for Potential-Based Reward Shaping (PBRS - Ng et al., 1999).
        Guarantees policy invariance while accelerating convergence.
        """
        r, c = pos
        gr, gc = self.goal_pos
        # Negative Manhattan distance to goal (higher potential closer to goal)
        dist = abs(r - gr) + abs(c - gc)
        base_potential = -float(dist) * 2.0
        # Elevation penalty in potential
        elev_cost = -float(self.elevation[r, c]) * 0.5
        # Hazard proximity repulsion
        hazard_penalty = -10.0 if self.hazards[r, c] else 0.0
        return base_potential + elev_cost + hazard_penalty

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_pos = self.start_pos
        self.current_step = 0

        info = {
            "current_pos": self.current_pos,
            "goal_pos": self.goal_pos,
            "elevation": float(self.elevation[self.current_pos]),
            "in_hazard": bool(self.hazards[self.current_pos]),
        }
        return self._get_obs(), info

    def get_step_cost(self, s1: Tuple[int, int], s2: Tuple[int, int]) -> float:
        """
        Calculates realistic physical transit cost:
        Base cost (1.0) + Uphill elevation climb penalty (positive Delta h).
        """
        base_cost = 1.0
        delta_h = float(self.elevation[s2]) - float(self.elevation[s1])
        climb_cost = max(0.0, delta_h * 1.5)
        return base_cost + climb_cost

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1
        r, c = self.current_pos

        # Stochastic Transition: apply wheel slip / wind gust if triggered
        actual_action = action
        if self.slip_probability > 0.0 and self.rng.random() < self.slip_probability:
            if self.wind_direction is not None and self.rng.random() < 0.7:
                actual_action = self.wind_direction  # Prevailing wind drift
            else:
                actual_action = self.rng.integers(0, 4)  # Random slip

        dr, dc = self.ACTION_DELTAS[actual_action]
        next_r = int(np.clip(r + dr, 0, self.grid_size - 1))
        next_c = int(np.clip(c + dc, 0, self.grid_size - 1))
        next_pos = (next_r, next_c)

        # Transit energy cost
        energy_cost = self.get_step_cost(self.current_pos, next_pos)
        in_hazard = bool(self.hazards[next_pos])
        at_goal = (next_pos == self.goal_pos)
        timeout = (self.current_step >= self.max_steps)

        # Reward Calculation based on active regime
        reward = 0.0
        if self.reward_mode == "sparse":
            # Sparse: Only terminal rewards and base step penalty
            if at_goal:
                reward = 100.0
            elif in_hazard:
                reward = -50.0
            else:
                reward = -energy_cost

        elif self.reward_mode == "naive":
            # Naive: Step penalty + direct negative Euclidean distance + hazard penalty
            gr, gc = self.goal_pos
            dist = np.sqrt((next_r - gr) ** 2 + (next_c - gc) ** 2)
            reward = -energy_cost - 0.5 * dist
            if in_hazard:
                reward -= 20.0
            if at_goal:
                reward += 100.0

        elif self.reward_mode == "pbrs":
            # Potential-Based Reward Shaping: F(s, a, s') = gamma * Phi(s') - Phi(s)
            gamma = 0.99
            phi_s = self._get_potential(self.current_pos)
            phi_next = self._get_potential(next_pos)
            shaping_f = gamma * phi_next - phi_s

            base_r = -energy_cost
            if in_hazard:
                base_r -= 15.0
            if at_goal:
                base_r += 100.0

            reward = base_r + shaping_f
        else:
            raise ValueError(f"Unknown reward_mode: {self.reward_mode}")

        self.current_pos = next_pos
        terminated = at_goal or in_hazard
        truncated = timeout and not terminated

        info = {
            "current_pos": self.current_pos,
            "reached_goal": at_goal,
            "fell_in_hazard": in_hazard,
            "energy_cost": energy_cost,
            "action_taken": actual_action,
            "intended_action": action,
            "slipped": (actual_action != action),
            "step": self.current_step,
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[Tuple[int, int], int, float]]:
        """
        Deterministic graph neighbor generator for Classical Search algorithms (DFS, BFS, A*).
        Returns list of (neighbor_pos, action, transition_cost).
        """
        r, c = pos
        neighbors = []
        for a_idx, (dr, dc) in enumerate(self.ACTION_DELTAS):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                next_p = (nr, nc)
                # Hazards have heavy avoidance penalty or are blocked in search
                hazard_mult = 20.0 if self.hazards[next_p] else 1.0
                cost = self.get_step_cost(pos, next_p) * hazard_mult
                neighbors.append((next_p, a_idx, cost))
        return neighbors
