"""Test script to probe RslRlVecEnvWrapper API in IsaacLab v2.3.2.

Run on the server after installation:
    ./isaaclab.sh -p source/isaaclab_nav_task/scripts/test_wrapper_api.py --task Isaac-Nav-MDPO-B2W-Dev-v0 --num_envs 4 --headless
"""

import argparse
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)

# --- After AppLauncher ---
import isaaclab_tasks  # noqa: F401
import isaaclab_nav_task  # noqa: F401

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


def main():
    # Create environment
    env_cfg: ManagerBasedRLEnvCfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args.num_envs

    import gymnasium as gym
    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    print("\n" + "=" * 60)
    print("WRAPPER API TEST")
    print("=" * 60)

    # --- Test get_observations() ---
    print("\n--- get_observations() ---")
    result = env.get_observations()
    print(f"  type: {type(result)}")
    print(f"  repr: {repr(result)[:200]}")

    if isinstance(result, tuple):
        print(f"  len(tuple): {len(result)}")
        for i, v in enumerate(result):
            print(f"  [{i}] type={type(v)}, ", end="")
            if hasattr(v, "shape"):
                print(f"shape={v.shape}")
            elif isinstance(v, dict):
                print(f"keys={list(v.keys())}")
            else:
                print(f"repr={repr(v)[:100]}")
    else:
        if hasattr(result, "shape"):
            print(f"  shape: {result.shape}")
        if hasattr(result, "keys"):
            print(f"  keys: {list(result.keys())}")
            for k in result.keys():
                v = result[k]
                print(f"    [{k}] type={type(v)}, ", end="")
                if hasattr(v, "shape"):
                    print(f"shape={v.shape}")
                else:
                    print(f"repr={repr(v)[:100]}")

    # --- Test reset() ---
    print("\n--- reset() ---")
    result = env.reset()
    print(f"  type: {type(result)}")
    if isinstance(result, tuple):
        print(f"  len(tuple): {len(result)}")
        for i, v in enumerate(result):
            print(f"  [{i}] type={type(v)}, ", end="")
            if hasattr(v, "shape"):
                print(f"shape={v.shape}")
            elif isinstance(v, dict):
                print(f"keys={list(v.keys())}")
                for k2 in list(v.keys())[:5]:
                    v2 = v[k2]
                    print(f"       [{k2}] type={type(v2)}, ", end="")
                    if hasattr(v2, "shape"):
                        print(f"shape={v2.shape}")
                    elif isinstance(v2, dict):
                        print(f"keys={list(v2.keys())}")
                    else:
                        print(f"repr={repr(v2)[:80]}")
            else:
                print(f"repr={repr(v)[:100]}")

    # --- Test step() ---
    print("\n--- step() ---")
    actions = torch.zeros(args.num_envs, env.num_actions, device=env.device)
    result = env.step(actions)
    print(f"  type: {type(result)}")
    if isinstance(result, tuple):
        print(f"  len(tuple): {len(result)}")
        for i, v in enumerate(result):
            print(f"  [{i}] type={type(v)}, ", end="")
            if hasattr(v, "shape"):
                print(f"shape={v.shape}")
            elif isinstance(v, dict):
                print(f"keys={list(v.keys())}")
                # Show observations sub-dict if present
                if "observations" in v:
                    obs_v = v["observations"]
                    print(f"       [observations] type={type(obs_v)}, ", end="")
                    if isinstance(obs_v, dict):
                        print(f"keys={list(obs_v.keys())}")
                        for k3 in list(obs_v.keys())[:5]:
                            v3 = obs_v[k3]
                            print(f"           [{k3}] type={type(v3)}, ", end="")
                            if hasattr(v3, "shape"):
                                print(f"shape={v3.shape}")
                            else:
                                print(f"repr={repr(v3)[:80]}")
                    elif hasattr(obs_v, "shape"):
                        print(f"shape={obs_v.shape}")
                    elif hasattr(obs_v, "keys"):
                        print(f"keys={list(obs_v.keys())}")
            else:
                print(f"repr={repr(v)[:100]}")

    # --- Test TensorDict compatibility ---
    print("\n--- Compatibility checks ---")
    obs_result = env.get_observations()

    # Check if obs["policy"] works (dict-like access)
    if isinstance(obs_result, tuple):
        obs = obs_result[0]
    else:
        obs = obs_result

    print(f"  obs type: {type(obs)}")

    # Try dict-like access
    try:
        policy_obs = obs["policy"]
        print(f"  obs['policy'] works: type={type(policy_obs)}, shape={policy_obs.shape}")
    except Exception as e:
        print(f"  obs['policy'] FAILED: {e}")

    # Try .shape[1]
    try:
        dim = obs.shape[1]
        print(f"  obs.shape[1] works: {dim}")
    except Exception as e:
        print(f"  obs.shape[1] FAILED: {e}")

    # Try .to(device)
    try:
        obs_moved = obs.to(env.device)
        print(f"  obs.to(device) works: type={type(obs_moved)}")
    except Exception as e:
        print(f"  obs.to(device) FAILED: {e}")

    # --- Test InteractiveScene containment ---
    print("\n--- Scene containment check ---")
    scene = env.unwrapped.scene

    try:
        result = "goal_sphere" in scene
        print(f"  'goal_sphere' in scene: {result}")
    except Exception as e:
        print(f"  'goal_sphere' in scene FAILED: {type(e).__name__}: {e}")

    try:
        sphere = scene["goal_sphere"]
        print(f"  scene['goal_sphere'] works: {type(sphere)}")
    except KeyError:
        print(f"  scene['goal_sphere'] raises KeyError (expected for non-BallTarget tasks)")
    except Exception as e:
        print(f"  scene['goal_sphere'] FAILED: {type(e).__name__}: {e}")

    # Check hasattr on scene cfg
    has_sphere = hasattr(scene.cfg, "goal_sphere")
    print(f"  hasattr(scene.cfg, 'goal_sphere'): {has_sphere}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
    app_launcher.app.close()
