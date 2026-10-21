"""
Main Entry Point for 23MEE305 Machine Learning Assignment:
"Understanding the evolution of search algorithms from DFS, BFS, To Reinforcement Learning – PPO, SAC, Q-Learning"

Usage:
    python main.py --all         # Runs full experiment suite + demo GIF
    python main.py --search      # Runs Classical Search benchmarks
    python main.py --rl          # Runs Q-Learning vs PPO training & curves
    python main.py --ablation    # Runs Reward Shaping ablation study
    python main.py --robustness  # Runs Stochastic Noise / Wind sweep
    python main.py --gif         # Generates animated GIF demo
"""

import sys
import argparse
from experiments import (
    run_classical_search_benchmarks,
    run_reward_shaping_ablation,
    run_rl_comparison,
    run_noise_robustness_sweep,
)
from generate_demo_gif import generate_animated_demo


def print_banner():
    banner = r"""
=============================================================================
  EVOLUTION OF SEARCH AND LEARNING ALGORITHMS (23MEE305)
  From Uninformed/Heuristic Search (DFS, BFS, A*) to Modern RL (Q-Learning, PPO)
  Environment: AeroMars Planetary Rover Navigation (Hazards, Slopes, Wind Slip)
=============================================================================
"""
    print(banner)


def main():
    print_banner()

    parser = argparse.ArgumentParser(
        description="Benchmark and evaluate Search to RL evolution on MarsRoverGridEnv."
    )
    parser.add_argument("--all", action="store_true", help="Run entire experiment suite and generate GIF")
    parser.add_argument("--search", action="store_true", help="Run Classical Search benchmark")
    parser.add_argument("--rl", action="store_true", help="Run Q-Learning vs PPO benchmark")
    parser.add_argument("--ablation", action="store_true", help="Run Reward Shaping ablation study")
    parser.add_argument("--robustness", action="store_true", help="Run transition noise / slip robustness sweep")
    parser.add_argument("--gif", action="store_true", help="Generate animated demo GIF")

    args = parser.parse_args()

    # If no argument is passed, default to running all
    if not any([args.all, args.search, args.rl, args.ablation, args.robustness, args.gif]):
        print("No specific flag provided; executing complete pipeline (--all)...\n")
        args.all = True

    if args.all or args.search:
        run_classical_search_benchmarks()

    if args.all or args.ablation:
        run_reward_shaping_ablation()

    rl_res = None
    if args.all or args.rl:
        rl_res = run_rl_comparison()

    if args.all or args.robustness:
        q_agent = rl_res["q_agent"] if rl_res else None
        ppo_agent = rl_res["ppo_agent"] if rl_res else None
        run_noise_robustness_sweep(q_agent=q_agent, ppo_agent=ppo_agent)

    if args.all or args.gif:
        generate_animated_demo()

    print("\n[SUCCESS] Pipeline execution finished. All figures and assets are ready!")


if __name__ == "__main__":
    main()
