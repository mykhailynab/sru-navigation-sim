"""Render a top-down view of all terrain types at two difficulty levels.

Generates a 2x4 grid image:
- Rows: difficulty ≈ 0.5 (top), difficulty ≈ 1.0 (bottom)
- Columns: maze, random pillars, stairs, pits

Usage:
    ./isaaclab.sh -p source/isaaclab_nav_task/scripts/render_terrains.py
    ./isaaclab.sh -p source/isaaclab_nav_task/scripts/render_terrains.py --output my_terrains.png
"""
from __future__ import annotations

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Render top-down terrain visualization.")
parser.add_argument("--output", type=str, default="terrain_grid.png", help="Output image path.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# Force headless + cameras
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Imports after AppLauncher (Isaac Sim requirement) ---
import gymnasium as gym
import torch
from PIL import Image

import isaaclab_nav_task  # noqa: F401 — registers gym tasks
from isaaclab.envs import ManagerBasedRLEnvCfg


def main():
    task = "Isaac-Nav-MDPO-B2W-Viz-v0"
    env_cfg: ManagerBasedRLEnvCfg = gym.spec(task).kwargs["env_cfg_entry_point"]()
    env = gym.make(task, cfg=env_cfg, render_mode="rgb_array")

    # Reset and warm up the renderer
    env.reset()
    min_steps = 50
    # The viewport annotator needs many render() calls to produce non-black output.
    # Step the sim and call render explicitly to warm up.
    for i in range(100):
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        env.step(actions)
        frame = env.unwrapped.render()
        if i < min_steps:
            continue
        if frame is not None and frame.max() > 0:
            print(f"[INFO] Renderer warmed up after {i + 1} steps")
            break
    else:
        # Extra render passes if still black
        for i in range(100):
            env.unwrapped.sim.render()
            frame = env.unwrapped.render()
            if frame is not None and frame.max() > 0:
                print(f"[INFO] Renderer warmed up after {i + 1} extra renders")
                break

    if frame is not None:
        img = Image.fromarray(frame)
        img.save(args_cli.output)
        print(f"[INFO] Saved terrain visualization to: {args_cli.output}")
    else:
        print("[ERROR] render() returned None — check that cameras are enabled")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
