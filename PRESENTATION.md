# Presentation Slides Deck: Understanding the Evolution of Search to Reinforcement Learning

**Course:** Machine Learning (23MEE305)  
**Project Title:** Evolution of Search and Learning Paradigms: From DFS, BFS, A* to Q-Learning and PPO  
**Team Members:** Team of 3 Students  
**Environment:** *AeroMars-Nav (Martian Elevation, Hazard Craters & Stochastic Dust Storms)*  

---

### Slide 1: Title Slide
- **Title:** From Uninformed Search to Deep Reinforcement Learning
- **Subtitle:** Analyzing Search Efficiency, Reward Shaping, and Policy Robustness in Planetary Navigation
- **Course:** Machine Learning (23MEE305)
- **Team Members:** [Student 1, Student 2, Student 3]
- **Visual:** Display `figures/demo_agents.gif` in background / side-by-side.

---

### Slide 2: Motivation & Algorithmic Evolution
- **The Core Question:** How do autonomous agents navigate from Start to Goal when the environment shifts from deterministic, known graphs to stochastic, unknown dynamics?
- **Timeline of Paradigms:**
  1. *1950s–1970s:* Uninformed Search (DFS, BFS) — Brute-force graph traversal.
  2. *1968:* Heuristic Search (A\*) — Incorporating domain heuristics $h(n)$ for optimal pruning.
  3. *1992:* Value-Based RL (Q-Learning) — Model-free Temporal Difference learning over unknown MDPs.
  4. *2017:* Policy-Gradient Actor-Critic (PPO) — Clipped surrogate objective for continuous feature spaces and reactive policies.

---

### Slide 3: AeroMars-Nav: A Novel Experimental Environment
- **Why Beyond Standard Toy Benchmarks?**
  - Standard FrozenLake / CliffWalking lack physical elevation physics, continuous gradients, and realistic wind slip.
- **Environment Architecture:**
  - **Grid Size:** $10 \times 10$ terrain with undulating 2D Martian elevation field $H(r, c)$.
  - **Transit Cost:** $c(s, s') = 1.0 + 1.5 \cdot \max(0, \Delta H)$ (uphill climbing consumes battery energy).
  - **Hazards:** Lethal crater obstacles requiring avoidance ($-50$ penalty).
  - **Transition Noise:** Stochastic slip probability $p_{\text{slip}}$ modeling dust-storm wind gusts.
- **Visual:** Show terrain elevation map from `figures/fig4_value_policy_heatmaps.png` (Panel 1).

---

### Slide 4: Classical Search Suite Benchmarking
- **Algorithms Evaluated:** DFS, BFS, A\* (Manhattan), A\* (Euclidean), A\* (Weighted $w=1.5$), A\* (Energy-Aware), Greedy baseline, Random walk.
- **Key Quantitative Findings:**
  - **BFS** explores blindly in circles $\rightarrow 86$ nodes expanded.
  - **A\* (Manhattan)** prunes search directed toward goal $\rightarrow 68$ nodes expanded, global optimal cost ($20.83$).
  - **A\* (Weighted $w=1.5$)** drops expansions to **$25$ nodes** (63% reduction) with only $3.3\%$ cost suboptimality.
  - **Random walk** suffers $0\%$ success rate ($>400$ steps).
- **Visual:** Display `figures/fig1_search_efficiency.png` (Horizontal bar charts of expansions, cost, runtime).

---

### Slide 5: Reinforcement Learning: Tabular Q-Learning vs PPO
- **Two Distinct Paradigm Families:**
  - **Q-Learning (Value-Based):**
    - Directly estimates state-action value $Q(s, a)$ via Bellman backups.
    - Super fast on discrete grids ($0.72\,\text{s}$ training time).
  - **PPO (Policy-Gradient Actor-Critic):**
    - Optimizes clipped surrogate objective $L^{\text{CLIP}}(\theta)$ using neural network policy and critic.
    - Handles continuous state vector $[r, c, \text{elevation}, \text{dist\_goal}, \text{hazard}]$.
- **Both achieve $100\%$ task success** and converge to the minimal $18$-step trajectory!
- **Visual:** Display `figures/fig2_rl_learning_curves.png` (Training curves).

---

### Slide 6: Reward Shaping & Theoretical Invariance (Part 3)
- **The Credit Assignment Problem:**
  - *Sparse Rewards:* Delayed feedback causes slow, erratic learning.
  - *Naive Distance Rewards:* Greedy bias traps agent near crater edges or creates circular loops.
- **Solution — Potential-Based Reward Shaping (PBRS - Ng et al., 1999):**
  $$F(s, a, s') = \gamma \Phi(s') - \Phi(s)$$
  $$\Phi(s) = -2.0 \cdot \|\mathbf{x}_s - \mathbf{x}_g\|_1 - 0.5 \cdot H(s) - 10.0 \cdot \mathbb{I}_{\text{hazard}}(s)$$
- **Mathematical Guarantee:** Policy Invariance Theorem guarantees $\arg\max_a \hat{Q}^*(s, a) = \arg\max_a Q^*(s, a)$.
- **Empirical Validation:**
  - PBRS converges within **$150$ episodes** with zero reward distortion!
- **Visual:** Display `figures/fig3_reward_shaping_ablation.png`.

---

### Slide 7: Policy & Value Landscape Visualization
- **Value Heatmaps & Policy Quivers:**
  - **State-Value Field $V^*(s)$:** Forms a smooth monotonic potential gradient funneling toward Goal $(9, 9)$.
  - **Policy Quiver Arrows $\pi^*(s)$:** Steer around crimson hazard craters and follow gentle valleys rather than steep uphill peaks.
- **Visual:** Display `figures/fig4_value_policy_heatmaps.png` (Side-by-side Elevation, Q-Learning Value Quiver, PPO Critic Value Quiver).

---

### Slide 8: Bonus (+5 Marks): Transition Noise & Slip Robustness
- **The Critical Test:** What happens when environmental noise strikes ($p_{\text{slip}} \in [0\%, 40\%]$)?
- **Open-Loop Classical A\* Collapses:**
  - Drops from $100\%$ success to **$0\%$ success** immediately at $p_{\text{slip}} = 5\%$.
  - Because A\* pre-plans a fixed path, a single slip pushes the rover into craters.
- **Closed-Loop RL Prevails:**
  - PPO maintains **$90\%$ success at $10\%$ slip** and **$61.7\%$ success at $30\%$ slip**!
  - Neural Actor reacts continuously to the rover's true updated position at every timestep.
- **Visual:** Display `figures/fig5_noise_robustness.png`.

---

### Slide 9: Multi-Agent Dynamic Simulation (Demo GIF)
- **Visual Slide:** Embed / Play `figures/demo_agents.gif` (24 frames, $\approx 8$ seconds):
  1. *Panel 1 (DFS):* Demonstrates deep branch wandering and backtracking.
  2. *Panel 2 (BFS):* Demonstrates uniform wavefront expansion.
  3. *Panel 3 (A\*):* Demonstrates focused heuristic beam avoiding hazards.
  4. *Panel 4 (Q-Learning):* Demonstrates optimal greedy value-path execution.
  5. *Panel 5 (PPO):* Demonstrates reactive Actor-Critic trajectory avoiding elevation ridges.

---

### Slide 10: Conclusion & Engineering Guidelines
1. **Model Available & Deterministic?**
   $\rightarrow$ Use **Weighted A\*** or A\*. Near-zero runtime ($<1\,\text{ms}$), provable optimality bounds.
2. **Dynamic, Stochastic, or Unknown Environment?**
   $\rightarrow$ Use **Reinforcement Learning (PPO / Actor-Critic)**. Provides closed-loop feedback policy resilient to slip and disturbance.
3. **Training RL in Sparse Domains?**
   $\rightarrow$ Always apply **Potential-Based Reward Shaping (PBRS)** to inject heuristic domain knowledge safely without corrupting optimal behavior.

---

### Slide 11: Q&A and Team Contributions
- **Team Responsibilities:**
  - Member 1: Environment formulation, elevation physics, Classical Search suite.
  - Member 2: Q-Learning & PPO implementation, hyperparameter tuning, RL evaluation.
  - Member 3: Reward shaping ablation, stochastic noise robustness study, visual reports.
- **Thank you! Questions?**
