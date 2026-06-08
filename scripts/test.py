#!/usr/bin/env python3
"""Evaluate a trained navigation policy and report per-terrain success rates.

Reproduces the Table 4 evaluation protocol: run episodes across 120 environments
(4 terrain types x 30 variations each) and track success rate, reward, and
episode length per terrain type.

Usage:
    ./isaaclab.sh -p source/isaaclab_nav_task/scripts/test.py --task Isaac-Nav-MDPO-B2W-Test-v0 --headless
    ./isaaclab.sh -p source/isaaclab_nav_task/scripts/test.py --task Isaac-Nav-MDPO-B2W-Test-v0 --checkpoint path/to/model.pt --headless
    ./isaaclab.sh -p source/isaaclab_nav_task/scripts/test.py --task Isaac-Nav-PPO-B2W-Test-v0 --num_episodes 120 --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a trained navigation policy.")
parser.add_argument("--task", type=str, default="Isaac-Nav-MDPO-B2W-Test-v0", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--num_episodes", type=int, default=4800, help="Total episodes to collect.")
parser.add_argument("--output", type=str, default="test_results.json", help="Output JSON file path.")

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Imports after AppLauncher (Isaac Sim requirement) ---
import json
import os
import re
import time

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import isaaclab_nav_task  # noqa: F401

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper


# ---------------------------------------------------------------------------
# Checkpoint helpers (same as play.py)
# ---------------------------------------------------------------------------

def find_latest_checkpoint(log_path: str, checkpoint_pattern: str = "model_.*.pt") -> str:
    if not os.path.exists(log_path):
        raise ValueError(f"Log path does not exist: {log_path}")
    run_dirs = []
    for entry in os.scandir(log_path):
        if entry.is_dir() and re.match(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", entry.name):
            run_dirs.append(entry.name)
    if not run_dirs:
        raise ValueError(f"No run directories found in: {log_path}")
    run_dirs.sort()
    run_path = os.path.join(log_path, run_dirs[-1])
    checkpoint_files = [f for f in os.listdir(run_path) if re.match(checkpoint_pattern, f)]
    if not checkpoint_files:
        raise ValueError(f"No checkpoint files matching '{checkpoint_pattern}' found in: {run_path}")
    checkpoint_files.sort(key=lambda m: f"{m:0>15}")
    return os.path.join(run_path, checkpoint_files[-1])


def load_checkpoint(runner: OnPolicyRunner, checkpoint_path: str):
    print(f"[INFO] Loading checkpoint from: {checkpoint_path}")
    loaded_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if runner.is_mdpo:
        runner.alg.actor_critic_1.load_state_dict(loaded_dict["model_state_dict"], strict=True)
        runner.alg.actor_critic_2.load_state_dict(loaded_dict["model_state_dict"], strict=True)
    else:
        runner.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"], strict=True)
    if runner.empirical_normalization:
        runner.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
        runner.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])
    runner.current_learning_iteration = loaded_dict["iter"]
    print(f"[INFO] Loaded checkpoint from iteration {loaded_dict['iter']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Config ---
    spec = gym.spec(args_cli.task)
    env_cfg: ManagerBasedRLEnvCfg = spec.kwargs["env_cfg_entry_point"]()
    agent_cfg: RslRlOnPolicyRunnerCfg = spec.kwargs["rsl_rl_cfg_entry_point"]()

    num_episodes = args_cli.num_episodes

    # --- Environment ---
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    num_envs = env.num_envs
    device = env.unwrapped.device

    # --- Checkpoint ---
    if args_cli.checkpoint:
        checkpoint_path = args_cli.checkpoint
    else:
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        checkpoint_path = find_latest_checkpoint(log_root_path)

    # --- Runner + policy ---
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    load_checkpoint(runner, checkpoint_path)
    policy = runner.get_inference_policy(device=device)

    # --- Terrain type mapping (static, read once) ---
    terrain = env.unwrapped.scene.terrain
    terrain_types_per_env = terrain.terrain_types.cpu()  # [num_envs]
    type_names = list(env_cfg.scene.terrain.terrain_generator.sub_terrains.keys())

    # --- Success tracker reference ---
    goal_cmd = env.unwrapped.command_manager._terms["robot_goal"]
    success_tracker = goal_cmd.success_tracker

    # --- Per-env accumulators ---
    cur_reward_sum = torch.zeros(num_envs, device=device)
    cur_episode_length = torch.zeros(num_envs, dtype=torch.long, device=device)

    # --- Collection ---
    episodes = []
    ep_infos = []
    obs, _ = env.get_observations()

    print(f"[INFO] Starting evaluation: {num_episodes} episodes across {num_envs} environments")
    print(f"[INFO] Terrain types: {type_names}")
    t_start = time.time()
    last_report = 0

    while len(episodes) < num_episodes:
        prev_write_idx = success_tracker.write_index.clone()

        with torch.inference_mode():
            actions = policy(obs)
        obs, rewards, dones, extras = env.step(actions)

        cur_reward_sum += rewards
        cur_episode_length += 1

        # Collect extras["log"] (same keys as TensorBoard metrics)
        if "log" in extras:
            ep_infos.append(extras["log"])

        # Process terminated environments
        done_mask = dones > 0
        if done_mask.any():
            done_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)
            for eid in done_ids.cpu().tolist():
                if len(episodes) >= num_episodes:
                    break
                # Determine success from tracker buffer
                wi = success_tracker.write_index[eid].item()
                if wi > prev_write_idx[eid].item():
                    result = success_tracker.buffer[eid, (wi - 1) % success_tracker.buffer_size].item()
                    success = result > 0.5
                else:
                    success = False

                tt = terrain_types_per_env[eid].item()
                episodes.append({
                    "env_id": eid,
                    "terrain_type": tt,
                    "terrain_type_name": type_names[tt],
                    "success": success,
                    "episode_reward": cur_reward_sum[eid].item(),
                    "episode_length": cur_episode_length[eid].item(),
                })

            cur_reward_sum[done_ids] = 0.0
            cur_episode_length[done_ids] = 0

        # Progress report every 500 episodes
        n = len(episodes)
        if n >= last_report + 500:
            sr = sum(e["success"] for e in episodes) / n * 100
            elapsed = time.time() - t_start
            print(f"[INFO] Episodes: {n}/{num_episodes} ({n / num_episodes * 100:.1f}%) | SR: {sr:.1f}% | {elapsed:.0f}s")
            last_report = (n // 500) * 500

    elapsed = time.time() - t_start
    print(f"[INFO] Evaluation complete: {len(episodes)} episodes in {elapsed:.1f}s")

    # --- Aggregate metrics ---
    # Overall
    total = len(episodes)
    overall_sr = sum(e["success"] for e in episodes) / total
    overall_reward = sum(e["episode_reward"] for e in episodes) / total
    overall_length = sum(e["episode_length"] for e in episodes) / total

    # Per terrain type
    per_type = {}
    for tname in type_names:
        type_eps = [e for e in episodes if e["terrain_type_name"] == tname]
        count = len(type_eps)
        if count > 0:
            per_type[tname] = {
                "success_rate": sum(e["success"] for e in type_eps) / count,
                "num_episodes": count,
                "mean_reward": sum(e["episode_reward"] for e in type_eps) / count,
                "mean_length": sum(e["episode_length"] for e in type_eps) / count,
            }

    # Aggregate ep_infos (same as on_policy_runner.py log method)
    log_metrics = {}
    if ep_infos:
        all_keys = set()
        for info in ep_infos:
            all_keys.update(info.keys())
        for key in sorted(all_keys):
            vals = []
            for info in ep_infos:
                if key in info:
                    v = info[key]
                    if isinstance(v, torch.Tensor):
                        vals.append(v.item() if v.numel() == 1 else v.mean().item())
                    else:
                        vals.append(float(v))
            if vals:
                log_metrics[key] = sum(vals) / len(vals)

    # --- Console summary ---
    print()
    print("=" * 80)
    print("Navigation Policy Evaluation Results")
    print("=" * 80)
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Task:       {args_cli.task}")
    print(f"  Episodes:   {total}")
    print("-" * 80)
    print(f"  {'Terrain Type':<16} | {'Episodes':>8} | {'SR':>8} | {'Mean Reward':>12} | {'Mean Length':>11}")
    print(f"  {'-'*16}-+-{'-'*8}-+-{'-'*8}-+-{'-'*12}-+-{'-'*11}")
    for tname in type_names:
        if tname in per_type:
            pt = per_type[tname]
            print(f"  {tname:<16} | {pt['num_episodes']:>8} | {pt['success_rate']*100:>7.1f}% | {pt['mean_reward']:>12.4f} | {pt['mean_length']:>11.1f}")
    print(f"  {'-'*16}-+-{'-'*8}-+-{'-'*8}-+-{'-'*12}-+-{'-'*11}")
    print(f"  {'Overall':<16} | {total:>8} | {overall_sr*100:>7.1f}% | {overall_reward:>12.4f} | {overall_length:>11.1f}")
    print("=" * 80)

    # --- Save JSON ---
    results = {
        "config": {
            "task": args_cli.task,
            "checkpoint": checkpoint_path,
            "num_envs": num_envs,
            "num_episodes": total,
            "terrain_type_names": type_names,
        },
        "summary": {
            "overall_success_rate": overall_sr,
            "mean_episode_reward": overall_reward,
            "mean_episode_length": overall_length,
            "per_terrain_type": per_type,
        },
        "log_metrics": log_metrics,
        "episodes": episodes,
    }

    with open(args_cli.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] Results saved to: {args_cli.output}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
