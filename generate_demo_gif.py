"""
Realistic & Cartoon-Themed Animated GIF Demonstration Generator.
Brings the Martian Rover Simulation to life with:
  - 3D Hillshaded Topographic Martian Surface (NASA MOLA styled)
  - Cartoon Character Rover Sprites & Emojis (Detective Bot, Radar Bot, Pathfinder, Cyber Bot, Aero Drone)
  - Animated Glowing Tire Tracks & Neon Trajectory Trails
  - Realistic Crater Hazards with Impact Decals & Warning Symbols
  - Extraction Base Camp with Pulsing Beacon
  - Real-time Mission Control HUD (Battery Bars, Step Counter, Radar Alert, Telemetry)
Output: figures/demo_agents.gif (<= 30 seconds)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, LinearSegmentedColormap
from PIL import Image, ImageDraw, ImageFont

from environment import MarsRoverGridEnv
from classical_search import depth_first_search, breadth_first_search, a_star_search
from rl_algorithms import QLearningAgent, PPOAgentWrapper

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Custom Martian Colormap: Deep canyon rust -> ochre plains -> golden sunlit ridges
MARTIAN_COLORS = [
    (0.18, 0.05, 0.02),  # Deep trench / chasm
    (0.45, 0.14, 0.08),  # Dark crater basin
    (0.72, 0.28, 0.12),  # Martian red clay
    (0.85, 0.45, 0.20),  # Ochre dust plains
    (0.95, 0.70, 0.40),  # Sunlit sand dunes
    (1.00, 0.90, 0.75),  # High mountain peak ridge
]
MARTIAN_CMAP = LinearSegmentedColormap.from_list("martian_relief", MARTIAN_COLORS, N=256)


def get_emoji_font(size: int = 24):
    """Safely loads Segoe UI Emoji font on Windows with fallback."""
    try:
        return ImageFont.truetype("seguiemj.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype("seguiui.ttf", size)
        except Exception:
            return ImageFont.load_default()


def get_regular_font(size: int = 14, bold: bool = False):
    """Loads clean sans-serif UI font."""
    font_name = "segoeuib.ttf" if bold else "segoeui.ttf"
    try:
        return ImageFont.truetype(font_name, size)
    except Exception:
        try:
            return ImageFont.truetype("arial.ttf", size)
        except Exception:
            return ImageFont.load_default()


def generate_realistic_cartoon_demo(grid_size: int = 10, seed: int = 42, fps: int = 3):
    print("\n" + "=" * 70)
    print(" GENERATING REALISTIC & CARTOON-ENHANCED ANIMATED DEMO GIF")
    print("=" * 70)

    env = MarsRoverGridEnv(grid_size=grid_size, reward_mode="pbrs", seed=seed)

    # 1. Classical Paths
    print("Computing Classical Search trajectories...")
    dfs_res = depth_first_search(env)
    bfs_res = breadth_first_search(env)
    astar_res = a_star_search(env, heuristic_type="manhattan")

    # 2. Train Q-Learning agent
    print("Pre-training Q-Learning agent...")
    q_agent = QLearningAgent(grid_size=grid_size, seed=seed)
    q_agent.train(env, num_episodes=800)

    env.reset(seed=seed)
    q_path = [env.start_pos]
    s_idx = env.state_to_discrete_index(env.start_pos)
    for _ in range(50):
        a = q_agent.select_action(s_idx, training=False)
        _, _, term, trunc, info = env.step(a)
        s_idx = env.state_to_discrete_index(info["current_pos"])
        q_path.append(info["current_pos"])
        if term or trunc:
            break

    # 3. Train PPO agent
    print("Pre-training PPO agent...")
    ppo_agent = PPOAgentWrapper(env, seed=seed)
    ppo_agent.train(total_timesteps=20000)

    obs, info = env.reset(seed=seed)
    ppo_path = [env.start_pos]
    for _ in range(50):
        a, _ = ppo_agent.model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(int(a))
        ppo_path.append(info["current_pos"])
        if term or trunc:
            break

    # 4. Agent Configurations (Cartoon Persona, Emojis, Colors)
    agents_meta = [
        {
            "name": "DFS: Detective Bot",
            "emoji": "🤖🔍",
            "sub": "Uninformed LIFO Traversal",
            "path": dfs_res.path,
            "color": "#FF4757",       # Crimson Neon
            "trail_color": "#FFA502", # Amber glow
            "personality": "Wandering branches...",
        },
        {
            "name": "BFS: Radar Bot",
            "emoji": "🐙📡",
            "sub": "Wavefront Hop-Optimal",
            "path": bfs_res.path,
            "color": "#1E90FF",       # Electric Blue
            "trail_color": "#70A1FF", # Cyan glow
            "personality": "Expanding perimeter...",
        },
        {
            "name": "A*: Pathfinder Pro",
            "emoji": "🧭🎯",
            "sub": "Heuristic Optimal Beam",
            "path": astar_res.path,
            "color": "#2ED573",       # Emerald Green
            "trail_color": "#7BED9F", # Mint glow
            "personality": "Target locked (h=optimal)",
        },
        {
            "name": "Q-Learning: Cyber Bot",
            "emoji": "👾🧠",
            "sub": "Bellman TD Value Policy",
            "path": q_path,
            "color": "#9B59B6",       # Amethyst Purple
            "trail_color": "#E056FD", # Neon violet
            "personality": "Executing max Q(s,a)...",
        },
        {
            "name": "PPO: Aero Drone",
            "emoji": "🛸⚡",
            "sub": "Actor-Critic Neural Policy",
            "path": ppo_path,
            "color": "#E67E22",       # Solar Flare Orange
            "trail_color": "#F1C40F", # Golden pulse
            "personality": "Smooth neural flight...",
        },
    ]

    # Pre-render 3D Hillshaded Martian Relief Map
    ls = LightSource(azdeg=315, altdeg=45)
    shaded_terrain = ls.shade(
        env.elevation,
        cmap=MARTIAN_CMAP,
        vert_exag=2.2,
        blend_mode="overlay",
        fraction=1.2,
    )

    max_steps = max(len(meta["path"]) for meta in agents_meta)
    frames = []

    print(f"Rendering {max_steps + 2} ultra-realistic cartoon animation frames...")

    emoji_font_large = get_emoji_font(28)
    emoji_font_mid = get_emoji_font(20)
    hud_font_title = get_regular_font(15, bold=True)
    hud_font_body = get_regular_font(12, bold=False)
    hud_font_bold = get_regular_font(12, bold=True)

    for step in range(max_steps + 2):
        # Create ultra-wide multi-panel figure with dark sci-fi background
        fig, axes = plt.subplots(1, 5, figsize=(22, 5.2), facecolor="#0F111A")

        for idx, meta in enumerate(agents_meta):
            ax = axes[idx]
            ax.set_facecolor("#0F111A")

            # 1. Render realistic 3D Martian terrain
            ax.imshow(shaded_terrain, origin="upper", extent=[-0.5, grid_size - 0.5, grid_size - 0.5, -0.5])

            # 2. Render contour lines for topographical realism
            ax.contour(
                env.elevation,
                levels=6,
                colors="#FFFFFF",
                alpha=0.18,
                linewidths=0.8,
                extent=[-0.5, grid_size - 0.5, -0.5, grid_size - 0.5],
            )

            # 3. Render Crater Hazard Zones with realistic impact craters
            for r in range(grid_size):
                for c in range(grid_size):
                    if env.hazards[r, c]:
                        # Dark crater floor
                        crater_circle = plt.Circle((c, r), 0.38, color="#120404", alpha=0.85, zorder=3)
                        ax.add_patch(crater_circle)
                        # Red warning shockwave ring
                        rim = plt.Circle((c, r), 0.44, fill=False, edgecolor="#E74C3C", linewidth=1.5, linestyle="--", alpha=0.9, zorder=3)
                        ax.add_patch(rim)

            # 4. Render Base Camp (Start) and Extraction Beacon (Goal)
            # Start: Base Hab ring
            start_ring = plt.Circle((env.start_pos[1], env.start_pos[0]), 0.42, color="#00D2D3", alpha=0.3, zorder=4)
            ax.add_patch(start_ring)
            # Goal: Target beacon pulsing circle
            pulse_ring1 = plt.Circle((env.goal_pos[1], env.goal_pos[0]), 0.45, fill=False, edgecolor="#FEE140", linewidth=2.0, alpha=0.9, zorder=4)
            pulse_ring2 = plt.Circle((env.goal_pos[1], env.goal_pos[0]), 0.30, color="#FF9F43", alpha=0.5, zorder=4)
            ax.add_patch(pulse_ring1)
            ax.add_patch(pulse_ring2)

            # 5. Render Trajectory Trail with Glowing Neon Path
            path = meta["path"]
            curr_idx = min(step, len(path) - 1)
            curr_slice = path[: curr_idx + 1]

            if len(curr_slice) > 1:
                ys = [p[0] for p in curr_slice]
                xs = [p[1] for p in curr_slice]
                # Outer glow
                ax.plot(xs, ys, color=meta["trail_color"], linewidth=6.0, alpha=0.4, zorder=5)
                # Core laser track
                ax.plot(xs, ys, color=meta["color"], linewidth=2.6, linestyle="-", marker="o", markersize=4, zorder=6)

            # 6. Render Rover Current Position
            curr_pos = curr_slice[-1]
            # Glowing aura underneath rover
            aura = plt.Circle((curr_pos[1], curr_pos[0]), 0.45, color=meta["color"], alpha=0.35, zorder=7)
            core = plt.Circle((curr_pos[1], curr_pos[0]), 0.28, color="#FFFFFF", edgecolor=meta["color"], linewidth=2.2, zorder=8)
            ax.add_patch(aura)
            ax.add_patch(core)

            # Clean borders and grid markers
            ax.set_xlim(-0.5, grid_size - 0.5)
            ax.set_ylim(grid_size - 0.5, -0.5)
            ax.set_xticks(range(0, grid_size, 2))
            ax.set_yticks(range(0, grid_size, 2))
            ax.tick_params(colors="#718096", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#2D3748")
                spine.set_linewidth(1.2)

            # Card Header Title
            ax.set_title(
                f"{meta['name']}",
                color="#FFFFFF",
                fontsize=11,
                fontweight="bold",
                pad=8,
            )

        fig.suptitle(
            "MARS EXPEDITION MISSION CONTROL: AUTONOMOUS ROVER NAVIGATION BENCHMARK",
            fontsize=13,
            fontweight="bold",
            color="#00D2D3",
            y=0.98,
        )
        plt.subplots_adjust(top=0.88, bottom=0.08, left=0.03, right=0.97, wspace=0.18)

        # Render matplotlib canvas to PIL image
        fig.canvas.draw()
        rgba_canvas = np.asarray(fig.canvas.buffer_rgba())
        pil_frame = Image.fromarray(rgba_canvas).convert("RGBA")
        plt.close(fig)

        # -------------------------------------------------------------
        # PIL COMPOSITING LAYER: High-Res Cartoon Characters, Emojis & HUD
        # -------------------------------------------------------------
        draw = ImageDraw.Draw(pil_frame)
        W, H = pil_frame.size
        panel_w = W / 5.0

        for idx, meta in enumerate(agents_meta):
            panel_left = idx * panel_w
            path = meta["path"]
            curr_idx = min(step, len(path) - 1)
            curr_pos = path[curr_idx]
            is_arrived = (curr_pos == env.goal_pos)

            # Convert grid coordinate to approximate pixel position on this panel
            # Matplotlib subplot area approximation:
            margin_x = panel_w * 0.12
            margin_y_top = H * 0.15
            sub_w = panel_w * 0.76
            sub_h = H * 0.70

            def grid_to_px(r, c):
                px_x = panel_left + margin_x + (c / (grid_size - 1)) * sub_w
                px_y = margin_y_top + (r / (grid_size - 1)) * sub_h
                return int(px_x), int(px_y)

            # Draw Base and Goal Emojis
            s_px = grid_to_px(env.start_pos[0], env.start_pos[1])
            g_px = grid_to_px(env.goal_pos[0], env.goal_pos[1])
            draw.text((s_px[0] - 12, s_px[1] - 22), "🚀", font=emoji_font_mid, embedded_color=True)
            draw.text((g_px[0] - 12, g_px[1] - 22), "💎", font=emoji_font_mid, embedded_color=True)

            # Draw Crater Hazard Emojis
            for r in range(grid_size):
                for c in range(grid_size):
                    if env.hazards[r, c]:
                        c_px = grid_to_px(r, c)
                        draw.text((c_px[0] - 10, c_px[1] - 12), "🕳️", font=ImageFont.truetype("seguiemj.ttf", 15) if os.path.exists("C:/Windows/Fonts/seguiemj.ttf") else emoji_font_mid, embedded_color=True)

            # Draw Cartoon Rover Avatar directly above current rover position
            r_px = grid_to_px(curr_pos[0], curr_pos[1])
            rover_emoji = meta["emoji"][:2]  # First emoji symbol
            draw.text((r_px[0] - 14, r_px[1] - 30), rover_emoji, font=emoji_font_large, embedded_color=True)

            # Draw Status Telemetry HUD Bubble above or below panel
            status_text = "🎉 GOAL SECURED!" if is_arrived else f"Step {curr_idx}/{len(path)-1}"
            battery_pct = max(10, 100 - int((curr_idx / max(1, len(path))) * 50))
            badge_color = (46, 213, 115, 230) if is_arrived else (30, 39, 46, 220)

            # HUD Box at bottom of each panel
            hud_box = [panel_left + 15, H - 48, panel_left + panel_w - 15, H - 8]
            draw.rounded_rectangle(hud_box, radius=6, fill=badge_color, outline=meta["color"], width=1)

            # Battery & Status text
            draw.text((panel_left + 22, H - 44), f"🔋 {battery_pct}%", font=hud_font_bold, fill="#FFFFFF")
            draw.text((panel_left + 78, H - 44), f"⏱️ {status_text}", font=hud_font_body, fill="#F1F2F6")
            draw.text((panel_left + 22, H - 26), f"{meta['personality']}", font=hud_font_body, fill=meta["trail_color"])

        frames.append(pil_frame)

    # Save animated GIF (<= 30s)
    gif_path = os.path.join(FIG_DIR, "demo_agents.gif")
    duration_ms = int(1000 / fps)
    print(f"Compiling {len(frames)} frames into animated GIF ({duration_ms}ms per frame)...")
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    print(f"-> Successfully generated Realistic Cartoon Demo GIF: {gif_path}")
    return gif_path


if __name__ == "__main__":
    generate_realistic_cartoon_demo()
