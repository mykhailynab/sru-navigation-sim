#!/usr/bin/env python3
# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Record a top-down video of BallTarget navigation on a 2x4 terrain grid.

Spawns robots on the BallTarget-Viz terrain (2 difficulty rows x 4 terrain types),
loads a trained checkpoint, and records a top-down mp4 video.

Usage:
    ./isaaclab.sh -p source/isaaclab_nav_task/scripts/record_balltarget_video.py \
        --checkpoint logs/rsl_rl/b2w_navigation_mdpo_ball/.../model_15000.pt

    ./isaaclab.sh -p source/isaaclab_nav_task/scripts/record_balltarget_video.py \
        --checkpoint logs/rsl_rl/b2w_navigation_mdpo_ball/.../model_15000.pt \
        --num_envs 8 --num_steps 500 --output balltarget_nav.mp4
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record top-down BallTarget navigation video.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt file).")
parser.add_argument("--num_envs", type=int, default=8, help="Number of robots to spawn.")
parser.add_argument("--num_steps", type=int, default=500, help="Number of simulation steps to record.")
parser.add_argument("--output", type=str, default="balltarget_navigation.mp4", help="Output video file path.")
parser.add_argument("--task", type=str, default="Isaac-Nav-MDPO-B2W-BallTarget-Viz-v0", help="Task ID.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# Force headless + cameras
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Imports after AppLauncher ---
import gymnasium as gym
import numpy as np
import os
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import isaaclab_nav_task  # noqa: F401

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper


def load_checkpoint(runner: OnPolicyRunner, checkpoint_path: str):
    """Load checkpoint into runner."""
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    loaded_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if runner.is_mdpo:
        runner.alg.actor_critic_1.load_state_dict(loaded_dict["model_state_dict"], strict=True)
        runner.alg.actor_critic_2.load_state_dict(loaded_dict["model_state_dict"], strict=True)
    else:
        runner.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"], strict=True)

    if runner.empirical_normalization:
        runner.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
        runner.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])

    print(f"[INFO] Loaded checkpoint from iteration {loaded_dict['iter']}")


def main():
    # Load configs from registry
    spec = gym.spec(args_cli.task)
    env_cfg: ManagerBasedRLEnvCfg = spec.kwargs["env_cfg_entry_point"]()
    agent_cfg: RslRlOnPolicyRunnerCfg = spec.kwargs["rsl_rl_cfg_entry_point"]()

    # Override num_envs for video (more robots = more interesting)
    env_cfg.scene.num_envs = args_cli.num_envs

    # Use a reasonable video resolution (landscape 1920x1080)
    env_cfg.viewer.resolution = (1920, 1080)

    # Create environment with rgb_array for frame capture
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(env)

    # Create runner and load checkpoint
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    load_checkpoint(runner, args_cli.checkpoint)

    # Get inference policy
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Reset and get initial observations
    obs_dict = env.get_observations()
    obs = obs_dict["policy"]

    # Warm up renderer (zero actions to keep agents stationary)
    print("[INFO] Warming up renderer...")
    zero_actions = torch.zeros(args_cli.num_envs, env.num_actions, device=env.unwrapped.device)
    min_warmup = 50
    for i in range(100):
        obs_dict, _, _, _ = env.step(zero_actions)
        obs = obs_dict["policy"]
        frame = env.unwrapped.render()
        if i < min_warmup:
            continue
        if frame is not None and frame.max() > 0:
            print(f"[INFO] Renderer warmed up after {i + 1} steps")
            break

    # Record frames
    print(f"[INFO] Recording {args_cli.num_steps} steps...")
    frames = []
    for step in range(args_cli.num_steps):
        with torch.inference_mode():
            actions = policy(obs)
        obs_dict, _, _, _ = env.step(actions)
        obs = obs_dict["policy"]

        frame = env.unwrapped.render()
        if frame is not None and frame.max() > 0:
            frames.append(frame)

        if (step + 1) % 100 == 0:
            print(f"  Step {step + 1}/{args_cli.num_steps} ({len(frames)} frames captured)")

    env.close()

    if not frames:
        print("[ERROR] No frames captured — check that cameras are enabled")
        return

    # Write mp4 using imageio
    import imageio.v3 as iio

    fps = int(1.0 / env.unwrapped.step_dt)
    output_path = os.path.abspath(args_cli.output)
    print(f"[INFO] Writing {len(frames)} frames at {fps} FPS to: {output_path}")
    iio.imwrite(output_path, np.stack(frames), fps=fps, codec="h264")
    print(f"[INFO] Done! Video saved to: {output_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
