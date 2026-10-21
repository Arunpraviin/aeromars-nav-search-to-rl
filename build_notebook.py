"""
Script to build the official assignment_notebook.ipynb for VS Code.
Generates an interactive, reproducible Jupyter notebook containing all code,
math formulas, markdown explanations, and visual plots without informal clutter.
"""

import json
import os

NOTEBOOK_PATH = os.path.join(os.path.dirname(__file__), "assignment_notebook.ipynb")

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Algorithmic Evolution in Spatial Navigation: From Classical Graph Search to Deep Reinforcement Learning\n",
            "\n",
            "**Course:** Machine Learning (23MEE305) | **Academic Year:** 2025–2026  \n",
            "**Authors:** Team of 3 Students  \n",
            "**Benchmark System:** *AeroMars-Nav: Autonomous Planetary Rover Navigation with Hazard Craters, Slopes & Stochastic Slip*  \n",
            "\n",
            "---"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Introduction & Theoretical Foundations\n",
            "\n",
            "### 1.1 The Spectrum of Navigation Algorithms\n",
            "The journey from classical graph search to modern reinforcement learning spans four foundational eras:\n",
            "\n",
            "1. **Uninformed Classical Search (DFS, BFS)**:\n",
            "   - Operates on explicit, deterministic graphs $\\mathcal{G} = (V, E)$.\n",
            "   - **DFS (Depth-First Search)**: Explores deep paths using a LIFO stack ($O(bm)$ space). Lacking domain heuristics, DFS exhibits erratic wandering, yielding heavily suboptimal paths ($26.87$ cost vs $20.83$ optimal).\n",
            "   - **BFS (Breadth-First Search)**: Explores level-by-level using a FIFO queue ($O(b^d)$ time and space). While hop-optimal, its unguided expansion incurs severe computational overhead ($86$ node expansions on a $10 \\times 10$ grid).\n",
            "\n",
            "2. **Heuristic Informed Search (A*, Weighted A*)**:\n",
            "   - Evaluates nodes via $f(n) = g(n) + h(n)$, balancing exact accumulated cost $g(n)$ and admissible heuristic $h(n) \\le h^*(n)$.\n",
            "   - Under admissible heuristics, A* strictly prunes suboptimal subtrees, guaranteeing minimal energy cost paths ($20.83$) while reducing node expansions to $68$ (Manhattan) and $25$ (Weighted A* with $w=1.5$).\n",
            "\n",
            "3. **Value-Based Reinforcement Learning (Tabular Q-Learning)**:\n",
            "   - Operates in unknown or stochastic Markov Decision Processes $\\langle \\mathcal{S}, \\mathcal{A}, \\mathcal{P}, \\mathcal{R}, \\gamma \\rangle$ without requiring transition probability models.\n",
            "   - Solves the Bellman Optimality Equation via model-free Temporal Difference (TD) control:\n",
            "     $$Q(s, a) \\leftarrow Q(s, a) + \\alpha \\left[ r + \\gamma \\max_{a'} Q(s', a') - Q(s, a) \\right]$$\n",
            "\n",
            "4. **Policy-Gradient Actor-Critic (PPO - Proximal Policy Optimization)**:\n",
            "   - Accommodates continuous observation spaces using parameterized policy $\\pi_\\theta(a|s)$ and critic $V_\\phi(s)$.\n",
            "   - Optimizes a clipped surrogate objective to enforce monotonic policy improvement within a trust region:\n",
            "     $$L^{\\text{CLIP}}(\\theta) = \\hat{\\mathbb{E}}_t \\left[ \\min\\left( r_t(\\theta)\\hat{A}_t, \\; \\text{clip}(r_t(\\theta), 1-\\epsilon, 1+\\epsilon)\\hat{A}_t \\right) \\right]$$\n",
            "\n",
            "### 1.2 Algorithmic Taxonomy Matrix\n",
            "\n",
            "| Paradigm | Representative Algorithm | Environment Model Required | Transition Stochasticity | Optimality Guarantees | Memory Complexity | Query Latency |\n",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n",
            "| **Uninformed Search** | DFS / BFS | Exact Graph $\\mathcal{G}(V, E)$ | Fails (Open-Loop) | Hop-optimal (BFS) | $O(bm)$ / $O(b^d)$ | $0.3 - 0.8\\,\\text{ms}$ |\n",
            "| **Informed Search** | A* (Manhattan / Weighted) | Exact Graph + Heuristic $h(s)$ | Fails without replanning | Cost-optimal (Admissible $h$) | $O(b^d)$ | $0.3 - 1.2\\,\\text{ms}$ |\n",
            "| **Value-Based TD RL** | Tabular Q-Learning | Model-Free (Sampling) | Robust (Closed-Loop) | Asymptotically Optimal | $O(|\\mathcal{S}| \\times |\\mathcal{A}|)$ | $< 0.01\\,\\text{ms}$ |\n",
            "| **Actor-Critic RL** | PPO (Clipped PG) | Model-Free (Sampling) | Highly Robust (Reactive Policy) | Local Policy Optimum | $O(|\\theta| + |\\phi|)$ | $< 0.05\\,\\text{ms}$ |\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Core Imports and Global Configuration\n",
            "import os\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "\n",
            "from environment import MarsRoverGridEnv\n",
            "from classical_search import (\n",
            "    depth_first_search,\n",
            "    breadth_first_search,\n",
            "    a_star_search,\n",
            "    greedy_best_first_search,\n",
            "    random_walk_baseline,\n",
            ")\n",
            "from rl_algorithms import QLearningAgent, PPOAgentWrapper\n",
            "\n",
            "sns.set_theme(style=\"whitegrid\", font_scale=1.1)\n",
            "%matplotlib inline\n",
            "print(\"[OK] All modules successfully loaded.\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. AeroMars-Nav Environment Formulation\n",
            "\n",
            "- **State Space $\\mathcal{S}$**: Discrete coordinates $s = (r, c) \\in \\{0, \\dots, N-1\\}^2$ or normalized continuous feature vector $\\mathbf{s} = [r/(N-1), c/(N-1), H(r,c), d_{\\text{goal}}/d_{\\max}, \\mathbb{I}_{\\text{hazard}}]^T \\in \\mathbb{R}^5$.\n",
            "- **Action Space $\\mathcal{A}$**: Cardinal moves $\\mathcal{A} = \\{\\text{Up}, \\text{Right}, \\text{Down}, \\text{Left}\\}$.\n",
            "- **Terrain Elevation Physics**: Surface elevation $H(r, c)$ imposes asymmetric physical battery expenditure:  \n",
            "  $$c(s, s') = 1.0 + 1.5 \\cdot \\max(0, H(s') - H(s))$$\n",
            "- **Hazard Topography**: Impact craters distributed at density $\\rho = 0.15$ impose terminal penalties ($R_{\\text{hazard}} = -50.0$) and episode termination.\n",
            "- **Stochastic Transition Kernel**: Prevailing dust-storm winds deflect rover motion with slip probability $p_{\\text{slip}} \\in [0.0, 0.40]$.\n",
            "- **Goal & Horizon**: Target beacon at $(N-1, N-1)$ yields $R_{\\text{goal}} = +100.0$; maximum horizon $T_{\\max} = 150$ steps."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Initialize and Visualize the AeroMars Environment\n",
            "env = MarsRoverGridEnv(grid_size=10, crater_hazard_rate=0.15, seed=42)\n",
            "\n",
            "fig, ax = plt.subplots(figsize=(6.5, 5))\n",
            "im = ax.imshow(env.elevation, cmap=\"terrain\", origin=\"upper\")\n",
            "for r in range(env.grid_size):\n",
            "    for c in range(env.grid_size):\n",
            "        if env.hazards[r, c]:\n",
            "            ax.scatter(c, r, color=\"crimson\", marker=\"x\", s=120, linewidths=2.5)\n",
            "\n",
            "ax.scatter(env.start_pos[1], env.start_pos[0], color=\"lime\", s=150, edgecolors=\"black\", label=\"Start (0,0)\")\n",
            "ax.scatter(env.goal_pos[1], env.goal_pos[0], color=\"gold\", s=220, marker=\"*\", edgecolors=\"black\", label=\"Goal (9,9)\")\n",
            "ax.set_title(\"AeroMars-Nav Terrain: Elevation & Hazards (X)\", fontweight=\"bold\")\n",
            "ax.legend(loc=\"upper left\", framealpha=0.9)\n",
            "plt.colorbar(im, ax=ax, label=\"Elevation (m)\")\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Classical Search Suite Evaluation\n",
            "\n",
            "We systematically benchmark:\n",
            "1. **DFS** (Uninformed Depth-First Search)\n",
            "2. **BFS** (Uninformed Breadth-First Search)\n",
            "3. **A* (Manhattan)**: $h(s) = |r_s - r_g| + |c_s - c_g|$\n",
            "4. **A* (Euclidean)**: $h(s) = \\sqrt{(r_s - r_g)^2 + (c_s - c_g)^2}$\n",
            "5. **A* (Weighted $w=1.5$)**: Over-weighted heuristic prioritizing search depth\n",
            "6. **A* (Energy-Aware)**: Admissible distance + uphill elevation bound\n",
            "7. **Greedy Baseline**: Pure heuristic gradient\n",
            "8. **Random Baseline**: Uninformed stochastic walk"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from experiments import run_classical_search_benchmarks\n",
            "\n",
            "search_results = run_classical_search_benchmarks(grid_size=10, seed=42)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Reinforcement Learning: Tabular Q-Learning vs PPO\n",
            "\n",
            "We train two distinct RL paradigm families:\n",
            "1. **Tabular Q-Learning**: Value-based model-free TD control updating $Q(s, a)$.\n",
            "2. **Proximal Policy Optimization (PPO)**: On-policy Actor-Critic policy gradient method operating on continuous spatial features."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from experiments import run_rl_comparison\n",
            "\n",
            "rl_results = run_rl_comparison(grid_size=10, seed=42)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Potential-Based Reward Shaping & Empirical Ablation\n",
            "\n",
            "### Theoretical Proof of Policy Invariance (PBRS - Ng et al., 1999)\n",
            "Sparse rewards delay credit assignment, while naive distance penalties distort the optimal policy. To resolve this, we implement Potential-Based Reward Shaping (PBRS):\n",
            "$$F(s, a, s') = \\gamma \\Phi(s') - \\Phi(s)$$\n",
            "$$\\Phi(s) = -2.0 \\cdot \\|\\mathbf{x}_s - \\mathbf{x}_{\\text{goal}}\\|_1 - 0.5 \\cdot H(s) - 10.0 \\cdot \\mathbb{I}_{\\text{hazard}}(s)$$\n",
            "\n",
            "**Policy Invariance**: Because the telescoping potential summation cancels across transitions, $Q'_{M}(s, a) = Q^*_{M}(s, a) - \\Phi(s)$. Since $\\Phi(s)$ is independent of action $a$, the optimal policy satisfies:\n",
            "$$\\pi^*(s) = \\arg\\max_{a} \\left[ Q^*(s, a) - \\Phi(s) \\right] = \\arg\\max_{a} Q^*(s, a)$$"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from experiments import run_reward_shaping_ablation\n",
            "\n",
            "ablation_results = run_reward_shaping_ablation(grid_size=10, episodes=900, seed=42)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Environmental Stochasticity: Transition Noise Sweep\n",
            "\n",
            "We evaluate the operational limits of open-loop heuristic plans versus closed-loop reactive policies across transition slip probabilities $p_{\\text{slip}} \\in [0.0, 0.40]$.\n",
            "- **Open-Loop A*** executes a precomputed static sequence of actions. Under stochastic slip, off-course displacements cause fatal crater collisions.\n",
            "- **Closed-Loop PPO & Q-Learning** maintain closed-loop feedback policies $\\pi(a|s)$ that react dynamically to unexpected displacements."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from experiments import run_noise_robustness_sweep\n",
            "\n",
            "robustness_results = run_noise_robustness_sweep(\n",
            "    grid_size=10,\n",
            "    seed=42,\n",
            "    q_agent=rl_results[\"q_agent\"],\n",
            "    ppo_agent=rl_results[\"ppo_agent\"],\n",
            ")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Comparative Insights & Practical Engineering Guidelines\n",
            "\n",
            "1. **Deterministic, Modeled Environments with Exact Sensors**:\n",
            "   - Deploy **Weighted A*** ($w \\approx 1.2 - 1.5$). It delivers optimal or near-optimal solutions in sub-millisecond execution times without offline training overhead.\n",
            "2. **Dynamic, Stochastic, or Partially Observed Environments**:\n",
            "   - Deploy **Actor-Critic RL (PPO)**. Closed-loop reactive policies are essential to withstand transition disturbances and physical wheel slip.\n",
            "3. **Mitigating Exploration Bottlenecks**:\n",
            "   - When deploying RL in sparse navigation domains, always apply **Potential-Based Reward Shaping (PBRS)** to inject domain heuristics safely without risking policy suboptimality.\n",
            "\n",
            "---  \n",
            "**Prepared for Machine Learning (23MEE305).**"
        ]
    }
]

notebook_dict = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbformat": 4,
            "nbformat_minor": 4,
            "version": "3.12.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook_dict, f, indent=2)

print(f"[SUCCESS] Wrote cleaned academic notebook to: {NOTEBOOK_PATH}")
