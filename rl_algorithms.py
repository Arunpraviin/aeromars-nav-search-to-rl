"""
Reinforcement Learning Algorithms for Autonomous Rover Navigation.
Implements:
  1. Tabular Q-Learning (Value-based TD Control)
  2. Proximal Policy Optimization (PPO - Policy-gradient Actor-Critic via Stable-Baselines3)
Provides policy extraction, value landscape estimation, and evaluation rollouts.
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from environment import MarsRoverGridEnv
import stable_baselines3 as sb3
from stable_baselines3.common.callbacks import BaseCallback


class QLearningAgent:
    """Tabular Q-Learning Agent with Epsilon-Greedy Exploration and Bellman Updates."""

    def __init__(
        self,
        grid_size: int = 10,
        num_actions: int = 4,
        learning_rate: float = 0.15,
        discount_factor: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int = 42,
    ):
        self.grid_size = grid_size
        self.num_states = grid_size * grid_size
        self.num_actions = num_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)

        # Initialize Q-table to zeros
        self.q_table = np.zeros((self.num_states, self.num_actions), dtype=np.float32)

    def select_action(self, state_idx: int, training: bool = True) -> int:
        """Epsilon-greedy action selection during training; greedy at inference."""
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.num_actions))
        # Break ties randomly among best actions
        q_vals = self.q_table[state_idx]
        max_val = np.max(q_vals)
        best_actions = np.where(q_vals == max_val)[0]
        return int(self.rng.choice(best_actions))

    def update(self, s: int, a: int, r: float, s_next: int, done: bool) -> float:
        """TD-error Bellman update: Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]"""
        target = r if done else r + self.gamma * np.max(self.q_table[s_next])
        td_error = target - self.q_table[s, a]
        self.q_table[s, a] += self.lr * td_error
        return float(td_error)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(
        self,
        env: MarsRoverGridEnv,
        num_episodes: int = 1000,
    ) -> Dict[str, Any]:
        """Runs tabular Q-learning training loop and logs learning curves."""
        t0 = time.perf_counter()
        episode_rewards = []
        episode_lengths = []
        success_history = []
        td_errors = []

        for ep in range(num_episodes):
            _, info = env.reset()
            s_idx = env.state_to_discrete_index(info["current_pos"])
            total_r = 0.0
            steps = 0
            ep_td_error = 0.0

            while True:
                action = self.select_action(s_idx, training=True)
                _, reward, terminated, truncated, step_info = env.step(action)
                next_s_idx = env.state_to_discrete_index(step_info["current_pos"])
                done = terminated or truncated

                err = self.update(s_idx, action, reward, next_s_idx, done)
                ep_td_error += abs(err)
                total_r += reward
                steps += 1
                s_idx = next_s_idx

                if done:
                    reached = step_info.get("reached_goal", False)
                    success_history.append(1 if reached else 0)
                    break

            self.decay_epsilon()
            episode_rewards.append(total_r)
            episode_lengths.append(steps)
            td_errors.append(ep_td_error / max(1, steps))

        runtime_sec = time.perf_counter() - t0
        return {
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "success_history": success_history,
            "td_errors": td_errors,
            "train_time_sec": runtime_sec,
        }

    def evaluate(
        self,
        env: MarsRoverGridEnv,
        num_eval_episodes: int = 50,
    ) -> Dict[str, Any]:
        """Evaluates frozen greedy policy on environment."""
        returns = []
        steps_list = []
        successes = 0

        for _ in range(num_eval_episodes):
            _, info = env.reset()
            s_idx = env.state_to_discrete_index(info["current_pos"])
            ep_r = 0.0
            st = 0
            while True:
                a = self.select_action(s_idx, training=False)
                _, r, term, trunc, s_info = env.step(a)
                s_idx = env.state_to_discrete_index(s_info["current_pos"])
                ep_r += r
                st += 1
                if term or trunc:
                    if s_info.get("reached_goal", False):
                        successes += 1
                    break
            returns.append(ep_r)
            steps_list.append(st)

        return {
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "success_rate": successes / num_eval_episodes,
            "mean_steps": float(np.mean(steps_list)),
        }

    def get_policy_and_values(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns 2D optimal action grid and 2D value landscape."""
        V = np.max(self.q_table, axis=1).reshape((self.grid_size, self.grid_size))
        pi = np.argmax(self.q_table, axis=1).reshape((self.grid_size, self.grid_size))
        return pi, V


class RewardLoggingCallback(BaseCallback):
    """Callback for tracking training rewards and episode lengths in Stable-Baselines3."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self._current_reward = 0.0
        self._current_length = 0

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [0.0])
        dones = self.locals.get("dones", [False])

        self._current_reward += rewards[0]
        self._current_length += 1

        if dones[0]:
            self.episode_rewards.append(self._current_reward)
            self.episode_lengths.append(self._current_length)
            self._current_reward = 0.0
            self._current_length = 0

        return True


class PPOAgentWrapper:
    """Proximal Policy Optimization (PPO) Actor-Critic Agent wrapper using Stable-Baselines3."""

    def __init__(
        self,
        env: MarsRoverGridEnv,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        clip_range: float = 0.2,
        seed: int = 42,
    ):
        self.env = env
        self.seed = seed
        self.callback = RewardLoggingCallback()

        self.model = sb3.PPO(
            "MlpPolicy",
            self.env,
            learning_rate=learning_rate,
            gamma=gamma,
            clip_range=clip_range,
            n_steps=256,
            batch_size=64,
            n_epochs=10,
            ent_coef=0.01,
            seed=seed,
            verbose=0,
        )

    def train(self, total_timesteps: int = 25000) -> Dict[str, Any]:
        """Trains PPO agent and logs runtime and metrics."""
        t0 = time.perf_counter()
        self.model.learn(total_timesteps=total_timesteps, callback=self.callback)
        runtime_sec = time.perf_counter() - t0

        return {
            "episode_rewards": self.callback.episode_rewards,
            "episode_lengths": self.callback.episode_lengths,
            "train_time_sec": runtime_sec,
            "timesteps": total_timesteps,
        }

    def evaluate(self, env: MarsRoverGridEnv, num_eval_episodes: int = 50) -> Dict[str, Any]:
        """Evaluates trained PPO model across test episodes."""
        returns = []
        steps_list = []
        successes = 0

        for _ in range(num_eval_episodes):
            obs, info = env.reset()
            ep_r = 0.0
            st = 0
            while True:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, r, term, trunc, s_info = env.step(int(action))
                ep_r += r
                st += 1
                if term or trunc:
                    if s_info.get("reached_goal", False):
                        successes += 1
                    break
            returns.append(ep_r)
            steps_list.append(st)

        return {
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "success_rate": successes / num_eval_episodes,
            "mean_steps": float(np.mean(steps_list)),
        }

    def get_policy_and_values(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts deterministic policy and value function over all grid positions."""
        import torch

        N = self.env.grid_size
        pi = np.zeros((N, N), dtype=int)
        V = np.zeros((N, N), dtype=np.float32)

        for r in range(N):
            for c in range(N):
                self.env.current_pos = (r, c)
                obs = self.env._get_obs()
                # Action prediction
                action, _ = self.model.predict(obs, deterministic=True)
                pi[r, c] = int(action)

                # Value prediction from critic network
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    val = self.model.policy.predict_values(obs_tensor)
                    V[r, c] = float(val.item())

        return pi, V
