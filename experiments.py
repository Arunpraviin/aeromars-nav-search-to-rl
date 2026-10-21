"""
Experimental Benchmark and Analysis Suite.
Executes:
  1. Classical Search Comparison (DFS, BFS, A*-variants, Greedy, Random)
  2. Reinforcement Learning Training & Curves (Q-Learning vs PPO)
  3. Reward Shaping Ablation (Sparse vs Naive vs PBRS)
  4. Policy & Value Landscape Visualizations (Heatmaps & Quiver Plots)
  5. Robustness to Stochastic Slip / Wind Disturbance (Noise Robustness Sweep)
Saves publication-quality figures to 'figures/' directory.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple

from environment import MarsRoverGridEnv
from classical_search import (
    depth_first_search,
    breadth_first_search,
    a_star_search,
    greedy_best_first_search,
    random_walk_baseline,
)
from rl_algorithms import QLearningAgent, PPOAgentWrapper

# Set plotting aesthetic
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def run_classical_search_benchmarks(grid_size: int = 10, seed: int = 42) -> List[Dict[str, Any]]:
    """Runs all classical search methods and baselines on the same environment."""
    print("\n" + "=" * 60)
    print(" [1/5] RUNNING CLASSICAL SEARCH BENCHMARK SUITE")
    print("=" * 60)

    env = MarsRoverGridEnv(grid_size=grid_size, seed=seed)

    results = []
    # 1. DFS
    res_dfs = depth_first_search(env)
    results.append(res_dfs.to_dict())

    # 2. BFS
    res_bfs = breadth_first_search(env)
    results.append(res_bfs.to_dict())

    # 3. A* Manhattan
    res_a_man = a_star_search(env, heuristic_type="manhattan")
    results.append(res_a_man.to_dict())

    # 4. A* Euclidean
    res_a_euc = a_star_search(env, heuristic_type="euclidean")
    results.append(res_a_euc.to_dict())

    # 5. A* Weighted
    res_a_w = a_star_search(env, heuristic_type="weighted", weight=1.5)
    results.append(res_a_w.to_dict())

    # 6. A* Energy-Aware
    res_a_eng = a_star_search(env, heuristic_type="energy_aware")
    results.append(res_a_eng.to_dict())

    # 7. Greedy Baseline
    res_greedy = greedy_best_first_search(env)
    results.append(res_greedy.to_dict())

    # 8. Random Baseline
    res_rnd = random_walk_baseline(env, max_steps=400, seed=seed)
    results.append(res_rnd.to_dict())

    # Print summary table
    print(f"{'Algorithm':<24} | {'Success':<8} | {'Nodes Expanded':<14} | {'Path Cost':<10} | {'Time (ms)':<10}")
    print("-" * 75)
    for r in results:
        succ_str = "YES" if r["success"] else "NO"
        print(f"{r['algorithm']:<24} | {succ_str:<8} | {r['nodes_expanded']:<14} | {r['path_cost']:<10.2f} | {r['runtime_ms']:<10.3f}")

    # Plot Figure 1: Classical Search Efficiency
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    algos = [r["algorithm"] for r in results if r["algorithm"] != "Random Baseline"]
    nodes = [r["nodes_expanded"] for r in results if r["algorithm"] != "Random Baseline"]
    costs = [r["path_cost"] for r in results if r["algorithm"] != "Random Baseline"]
    times = [r["runtime_ms"] for r in results if r["algorithm"] != "Random Baseline"]

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD", "#E377C2"]

    # (a) Nodes Expanded
    axes[0].barh(algos, nodes, color=colors[:len(algos)], edgecolor="black", alpha=0.85)
    axes[0].set_xlabel("Nodes Expanded (Lower is more efficient)")
    axes[0].set_title("Search Efficiency: Node Expansions", fontweight="bold")
    for i, v in enumerate(nodes):
        axes[0].text(v + 1, i, str(v), va="center", fontsize=9)

    # (b) Path Cost
    axes[1].barh(algos, costs, color=colors[:len(algos)], edgecolor="black", alpha=0.85)
    axes[1].set_xlabel("Path Transit Energy Cost (Lower is better)")
    axes[1].set_title("Solution Optimality: Path Cost", fontweight="bold")
    for i, v in enumerate(costs):
        axes[1].text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=9)

    # (c) Runtime in ms
    axes[2].barh(algos, times, color=colors[:len(algos)], edgecolor="black", alpha=0.85)
    axes[2].set_xlabel("Execution Time in ms (Log Scale)")
    axes[2].set_xscale("log")
    axes[2].set_title("Computational Latency (ms)", fontweight="bold")

    plt.tight_layout()
    fig1_path = os.path.join(FIG_DIR, "fig1_search_efficiency.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"-> Saved Search Efficiency Benchmark to: {fig1_path}")

    return results


def run_reward_shaping_ablation(grid_size: int = 10, episodes: int = 900, seed: int = 42) -> Dict[str, Any]:
    """
    Reward Shaping Ablation Study:
    Compares Tabular Q-Learning under:
      - Sparse Reward (terminal only)
      - Naive Distance-based Reward
      - Potential-Based Reward Shaping (PBRS - Ng et al., 1999)
    """
    print("\n" + "=" * 60)
    print(" [2/5] RUNNING REWARD SHAPING ABLATION STUDY")
    print("=" * 60)

    regimes = ["sparse", "naive", "pbrs"]
    history = {}

    for mode in regimes:
        print(f"Training Q-Learning with '{mode.upper()}' reward regime...")
        env = MarsRoverGridEnv(grid_size=grid_size, reward_mode=mode, seed=seed)
        agent = QLearningAgent(grid_size=grid_size, seed=seed, learning_rate=0.15, epsilon_decay=0.994)
        logs = agent.train(env, num_episodes=episodes)
        eval_metrics = agent.evaluate(env, num_eval_episodes=50)
        history[mode] = {
            "rewards": logs["episode_rewards"],
            "lengths": logs["episode_lengths"],
            "success_rate": eval_metrics["success_rate"],
            "mean_steps": eval_metrics["mean_steps"],
            "time_sec": logs["train_time_sec"],
        }
        print(f"  Result: Eval Success Rate = {eval_metrics['success_rate']*100:.1f}%, Mean Steps = {eval_metrics['mean_steps']:.1f}")

    # Plot Figure 3: Reward Shaping Ablation
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    window = 30

    colors = {"sparse": "#E74C3C", "naive": "#F39C12", "pbrs": "#2ECC71"}
    labels = {
        "sparse": "Sparse Terminal Reward",
        "naive": "Naive Distance Reward",
        "pbrs": "Potential-Based Shaping (PBRS)",
    }

    # (a) Episode Lengths (Steps to Goal)
    for mode in regimes:
        raw_lens = np.array(history[mode]["lengths"])
        rolling = np.convolve(raw_lens, np.ones(window) / window, mode="valid")
        axes[0].plot(rolling, label=labels[mode], color=colors[mode], linewidth=2.2)

    axes[0].set_title("Convergence Speed: Episode Length", fontweight="bold")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Steps to Reach Goal (Rolling Avg)")
    axes[0].legend()

    # (b) Rolling Return
    for mode in regimes:
        raw_r = np.array(history[mode]["rewards"])
        rolling = np.convolve(raw_r, np.ones(window) / window, mode="valid")
        axes[1].plot(rolling, label=labels[mode], color=colors[mode], linewidth=2.2)

    axes[1].set_title("Learning Dynamics: Rolling Return", fontweight="bold")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Episode Return (Rolling Avg)")
    axes[1].legend()

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "fig3_reward_shaping_ablation.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"-> Saved Reward Shaping Ablation to: {fig_path}")

    return history


def run_rl_comparison(grid_size: int = 10, seed: int = 42) -> Dict[str, Any]:
    """Trains both Q-Learning and PPO on the PBRS environment and compares their curves."""
    print("\n" + "=" * 60)
    print(" [3/5] TRAINING & COMPARING RL PARADIGMS (Q-LEARNING VS PPO)")
    print("=" * 60)

    # 1. Q-Learning
    print("Training Tabular Q-Learning (Value-based TD Control)...")
    env_q = MarsRoverGridEnv(grid_size=grid_size, reward_mode="pbrs", seed=seed)
    q_agent = QLearningAgent(grid_size=grid_size, seed=seed, learning_rate=0.15, epsilon_decay=0.995)
    q_logs = q_agent.train(env_q, num_episodes=1000)
    q_eval = q_agent.evaluate(env_q, num_eval_episodes=50)

    # 2. PPO
    print("Training PPO (Policy-Gradient Actor-Critic via Stable-Baselines3)...")
    env_ppo = MarsRoverGridEnv(grid_size=grid_size, reward_mode="pbrs", seed=seed)
    ppo_agent = PPOAgentWrapper(env_ppo, learning_rate=3e-4, seed=seed)
    ppo_logs = ppo_agent.train(total_timesteps=30000)
    ppo_eval = ppo_agent.evaluate(env_ppo, num_eval_episodes=50)

    print("\n--- RL EVALUATION BENCHMARK ---")
    print(f"Q-Learning: Success = {q_eval['success_rate']*100:.1f}%, Mean Return = {q_eval['mean_return']:.2f}, Time = {q_logs['train_time_sec']:.2f}s")
    print(f"PPO:        Success = {ppo_eval['success_rate']*100:.1f}%, Mean Return = {ppo_eval['mean_return']:.2f}, Time = {ppo_logs['train_time_sec']:.2f}s")

    # Plot Figure 2: Learning Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    window = 25

    # Q-learning rolling return
    q_r = np.array(q_logs["episode_rewards"])
    q_rolling = np.convolve(q_r, np.ones(window) / window, mode="valid")
    axes[0].plot(q_rolling, color="#2980B9", linewidth=2.0, label="Q-Learning (Value-Based)")
    axes[0].set_title("Q-Learning: Training Return Curve", fontweight="bold")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Rolling Episode Return")
    axes[0].legend()

    # PPO rolling return
    if len(ppo_logs["episode_rewards"]) > window:
        ppo_r = np.array(ppo_logs["episode_rewards"])
        ppo_rolling = np.convolve(ppo_r, np.ones(window) / window, mode="valid")
        axes[1].plot(ppo_rolling, color="#8E44AD", linewidth=2.0, label="PPO (Actor-Critic)")
    else:
        axes[1].plot(ppo_logs["episode_rewards"], color="#8E44AD", linewidth=2.0, label="PPO (Actor-Critic)")
    axes[1].set_title("PPO: Training Return Curve", fontweight="bold")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Rolling Episode Return")
    axes[1].legend()

    plt.tight_layout()
    fig2_path = os.path.join(FIG_DIR, "fig2_rl_learning_curves.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"-> Saved RL Learning Curves to: {fig2_path}")

    # Plot Figure 4: Policy & Value Landscapes (Heatmaps + Quivers)
    plot_policy_and_value_landscapes(env_q, q_agent, ppo_agent)

    return {
        "q_eval": q_eval,
        "ppo_eval": ppo_eval,
        "q_agent": q_agent,
        "ppo_agent": ppo_agent,
    }


def plot_policy_and_value_landscapes(
    env: MarsRoverGridEnv,
    q_agent: QLearningAgent,
    ppo_agent: PPOAgentWrapper,
):
    """
    Visualizes Value Landscapes and Optimal Policy Quiver Vector Fields.
    Overlays terrain elevation and hazard craters.
    """
    print("\n" + "=" * 60)
    print(" [4/5] GENERATING POLICY & VALUE LANDSCAPE VISUALIZATIONS")
    print("=" * 60)

    q_pi, q_V = q_agent.get_policy_and_values()
    ppo_pi, ppo_V = ppo_agent.get_policy_and_values()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    N = env.grid_size

    # Direction vectors for actions: 0: Up, 1: Right, 2: Down, 3: Left
    action_uv = [
        (0, -0.6),  # Up (points negative y in imshow coords)
        (0.6, 0),   # Right
        (0, 0.6),   # Down
        (-0.6, 0),  # Left
    ]

    # Panel 1: Terrain Elevation & Hazards
    im0 = axes[0].imshow(env.elevation, cmap="terrain", origin="upper")
    for r in range(N):
        for c in range(N):
            if env.hazards[r, c]:
                axes[0].scatter(c, r, color="crimson", marker="x", s=130, linewidths=2.5)
    axes[0].scatter(env.start_pos[1], env.start_pos[0], color="lime", marker="o", s=160, edgecolor="black", label="Start (0,0)")
    axes[0].scatter(env.goal_pos[1], env.goal_pos[0], color="gold", marker="*", s=260, edgecolor="black", label="Goal (N-1,N-1)")
    axes[0].set_title("Planetary Elevation Map & Craters (X)", fontweight="bold")
    axes[0].legend(loc="upper left", framealpha=0.9)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Elevation (m)")

    # Panel 2: Q-Learning Value Heatmap & Policy Arrows
    im1 = axes[1].imshow(q_V, cmap="viridis", origin="upper")
    for r in range(N):
        for c in range(N):
            if env.hazards[r, c]:
                axes[1].scatter(c, r, color="crimson", marker="x", s=90, linewidths=2)
            elif (r, c) != env.goal_pos:
                act = q_pi[r, c]
                u, v = action_uv[act]
                axes[1].quiver(c, r, u, v, color="white", scale=1, scale_units="xy", angles="xy", width=0.015, headwidth=4)
    axes[1].set_title("Q-Learning: Value Field V*(s) & Policy Quiver", fontweight="bold")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="State Value V*(s)")

    # Panel 3: PPO Value Heatmap & Policy Arrows
    im2 = axes[2].imshow(ppo_V, cmap="plasma", origin="upper")
    for r in range(N):
        for c in range(N):
            if env.hazards[r, c]:
                axes[2].scatter(c, r, color="crimson", marker="x", s=90, linewidths=2)
            elif (r, c) != env.goal_pos:
                act = ppo_pi[r, c]
                u, v = action_uv[act]
                axes[2].quiver(c, r, u, v, color="white", scale=1, scale_units="xy", angles="xy", width=0.015, headwidth=4)
    axes[2].set_title("PPO Critic: Value Field V(s) & Policy Quiver", fontweight="bold")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="PPO Critic V(s)")

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "fig4_value_policy_heatmaps.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"-> Saved Value Landscape & Policy Heatmaps to: {fig_path}")


def run_noise_robustness_sweep(
    grid_size: int = 10,
    seed: int = 42,
    q_agent: Optional[QLearningAgent] = None,
    ppo_agent: Optional[PPOAgentWrapper] = None,
) -> Dict[str, Any]:
    """
    Bonus (+5 Marks): Transition Noise / Wind Slip Robustness Sweep.
    Evaluates:
      - Open-loop Classical A* (deterministic plan executed in stochastic world)
      - Closed-loop Q-Learning (reactive feedback policy)
      - Closed-loop PPO (reactive actor network)
    Across slip probabilities: [0.0, 0.05, 0.10, 0.20, 0.30, 0.40].
    """
    print("\n" + "=" * 60)
    print(" [5/5] RUNNING TRANSITION NOISE & WIND SLIP ROBUSTNESS SWEEP")
    print("=" * 60)

    slip_rates = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]
    num_trials = 60

    # Base environment for A* plan
    base_env = MarsRoverGridEnv(grid_size=grid_size, slip_probability=0.0, seed=seed)
    a_star_res = a_star_search(base_env, heuristic_type="manhattan")
    precomputed_actions = a_star_res.actions

    # Train agents if not passed
    if q_agent is None:
        q_agent = QLearningAgent(grid_size=grid_size, seed=seed)
        q_agent.train(base_env, num_episodes=900)

    if ppo_agent is None:
        ppo_agent = PPOAgentWrapper(base_env, seed=seed)
        ppo_agent.train(total_timesteps=25000)

    a_star_successes = []
    q_successes = []
    ppo_successes = []

    for slip in slip_rates:
        print(f"Testing under Slip Probability = {slip*100:.0f}%...")
        test_env = MarsRoverGridEnv(grid_size=grid_size, slip_probability=slip, seed=seed + int(slip * 100))

        # 1. Test Open-loop A* plan
        a_succ = 0
        for trial in range(num_trials):
            test_env.reset(seed=seed + trial)
            for a in precomputed_actions:
                _, _, term, trunc, info = test_env.step(a)
                if term or trunc:
                    if info.get("reached_goal", False):
                        a_succ += 1
                    break
        a_star_successes.append(a_succ / num_trials)

        # 2. Test Closed-loop Q-Learning
        q_eval = q_agent.evaluate(test_env, num_eval_episodes=num_trials)
        q_successes.append(q_eval["success_rate"])

        # 3. Test Closed-loop PPO
        ppo_eval = ppo_agent.evaluate(test_env, num_eval_episodes=num_trials)
        ppo_successes.append(ppo_eval["success_rate"])

    print("\n--- ROBUSTNESS RESULTS ---")
    for i, slip in enumerate(slip_rates):
        print(f"Slip {slip*100:4.1f}% | A* Open-Loop: {a_star_successes[i]*100:5.1f}% | Q-Learning: {q_successes[i]*100:5.1f}% | PPO: {ppo_successes[i]*100:5.1f}%")

    # Plot Figure 5: Noise Robustness Curve
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x_axis = np.array(slip_rates) * 100

    ax.plot(x_axis, np.array(a_star_successes) * 100, marker="o", linewidth=2.5, color="#E74C3C", label="Open-Loop A* (Deterministic Plan)")
    ax.plot(x_axis, np.array(q_successes) * 100, marker="s", linewidth=2.5, color="#2980B9", label="Closed-Loop Q-Learning (Tabular Policy)")
    ax.plot(x_axis, np.array(ppo_successes) * 100, marker="^", linewidth=2.5, color="#27AE60", label="Closed-Loop PPO (Actor-Critic Policy)")

    ax.set_title("Robustness under Stochastic Wind & Wheel Slip Disturbances", fontweight="bold", fontsize=13)
    ax.set_xlabel("Transition Slip Probability (%)", fontsize=11)
    ax.set_ylabel("Goal Arrival Success Rate (%)", fontsize=11)
    ax.set_ylim(-5, 105)
    ax.legend(framealpha=0.95, loc="lower left")

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "fig5_noise_robustness.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"-> Saved Stochastic Noise Robustness Curve to: {fig_path}")

    return {
        "slip_rates": slip_rates,
        "a_star_successes": a_star_successes,
        "q_successes": q_successes,
        "ppo_successes": ppo_successes,
    }


def run_all_experiments():
    """Runs all 5 experiment phases sequentially."""
    print("STARTING FULL EXPERIMENTAL EVALUATION SUITE FOR 23MEE305...")
    search_results = run_classical_search_benchmarks()
    ablation_results = run_reward_shaping_ablation()
    rl_results = run_rl_comparison()
    robustness_results = run_noise_robustness_sweep(
        q_agent=rl_results["q_agent"],
        ppo_agent=rl_results["ppo_agent"],
    )
    print("\n" + "=" * 60)
    print(" ALL EXPERIMENTS COMPLETE! ALL FIGURES SAVED IN: " + FIG_DIR)
    print("=" * 60)


if __name__ == "__main__":
    run_all_experiments()
