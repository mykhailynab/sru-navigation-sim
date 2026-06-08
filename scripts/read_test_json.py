#!/usr/bin/env python3
"""Read and present test evaluation results from JSON.

Usage:
    python source/isaaclab_nav_task/scripts/read_test_json.py logs/rsl_rl/b2w_navigation_mdpo/2026-05-22_11-55-51/test_results.json
    python source/isaaclab_nav_task/scripts/read_test_json.py test_results.json --verbose
"""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="Read and present test evaluation results.")
    parser.add_argument("json_path", type=str, help="Path to test_results.json")
    parser.add_argument("--verbose", action="store_true", help="Show log metrics and per-env stats.")
    args = parser.parse_args()

    with open(args.json_path) as f:
        data = json.load(f)

    config = data["config"]
    summary = data["summary"]
    per_type = summary["per_terrain_type"]
    log_metrics = data.get("log_metrics", {})
    episodes = data.get("episodes", [])

    # --- Header ---
    print()
    print("=" * 80)
    print("Navigation Policy Evaluation Results")
    print("=" * 80)
    print(f"  Task:       {config['task']}")
    print(f"  Checkpoint: {config['checkpoint']}")
    print(f"  Envs:       {config['num_envs']}")
    print(f"  Episodes:   {config['num_episodes']}")

    # --- Per-terrain table ---
    print()
    print("-" * 80)
    print(f"  {'Terrain Type':<16} | {'Episodes':>8} | {'SR':>8} | {'Mean Reward':>12} | {'Mean Length':>11}")
    print(f"  {'-'*16}-+-{'-'*8}-+-{'-'*8}-+-{'-'*12}-+-{'-'*11}")
    for tname in config.get("terrain_type_names", per_type.keys()):
        if tname in per_type:
            pt = per_type[tname]
            sr = pt["success_rate"]
            print(f"  {tname:<16} | {pt['num_episodes']:>8} | {sr*100:>7.1f}% | {pt['mean_reward']:>12.2f} | {pt['mean_length']:>11.1f}")
    print(f"  {'-'*16}-+-{'-'*8}-+-{'-'*8}-+-{'-'*12}-+-{'-'*11}")
    print(f"  {'Overall':<16} | {config['num_episodes']:>8} | {summary['overall_success_rate']*100:>7.1f}% | {summary['mean_episode_reward']:>12.2f} | {summary['mean_episode_length']:>11.1f}")
    print("=" * 80)

    # --- Log metrics ---
    if args.verbose and log_metrics:
        print()
        print("Log Metrics (averaged over all steps):")
        print("-" * 60)
        for key in sorted(log_metrics.keys()):
            print(f"  {key:<45} {log_metrics[key]:>10.4f}")
        print()

    # --- Per-env breakdown (verbose) ---
    if args.verbose and episodes:
        print()
        print("Per-Environment Stats:")
        print("-" * 80)
        env_stats = {}
        for ep in episodes:
            eid = ep["env_id"]
            if eid not in env_stats:
                env_stats[eid] = {"terrain": ep["terrain_type_name"], "episodes": 0, "successes": 0, "total_reward": 0.0, "total_length": 0}
            env_stats[eid]["episodes"] += 1
            env_stats[eid]["successes"] += int(ep["success"])
            env_stats[eid]["total_reward"] += ep["episode_reward"]
            env_stats[eid]["total_length"] += ep["episode_length"]

        print(f"  {'Env':>4} | {'Terrain':<16} | {'Episodes':>8} | {'SR':>8} | {'Mean Reward':>12} | {'Mean Length':>11}")
        print(f"  {'-'*4}-+-{'-'*16}-+-{'-'*8}-+-{'-'*8}-+-{'-'*12}-+-{'-'*11}")
        for eid in sorted(env_stats.keys()):
            s = env_stats[eid]
            n = s["episodes"]
            sr = s["successes"] / n if n > 0 else 0
            mr = s["total_reward"] / n if n > 0 else 0
            ml = s["total_length"] / n if n > 0 else 0
            print(f"  {eid:>4} | {s['terrain']:<16} | {n:>8} | {sr*100:>7.1f}% | {mr:>12.2f} | {ml:>11.1f}")
        print()


if __name__ == "__main__":
    main()
