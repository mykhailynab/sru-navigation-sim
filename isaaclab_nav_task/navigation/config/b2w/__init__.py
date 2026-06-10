# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

import gymnasium as gym

from . import agents, navigation_env_cfg

##
# Register Gym environments.
##

##############################################################################################################
# MDPO

gym.register(
    id="Isaac-Nav-MDPO-B2W-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.B2WNavigationEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.B2WNavMDPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Nav-MDPO-B2W-AblateLSTM-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.B2WNavigationEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.B2WNavMDPOLSTMRunnerCfg,
    },
)

gym.register(
    id="Isaac-Nav-MDPO-B2W-BallTarget-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.B2WNavigationEnvCfgBallTarget,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.B2WNavMDPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Nav-MDPO-B2W-AblatePriop-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.B2WNavigationEnvCfgAblatePrioperceptive,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.B2WNavMDPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Nav-MDPO-B2W-Play-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.B2WNavigationEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.B2WNavMDPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Nav-MDPO-B2W-Dev-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.B2WNavigationEnvCfg_DEV,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.B2WNavMDPORunnerDevCfg,
    },
)

gym.register(
    id="Isaac-Nav-MDPO-B2W-Viz-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.B2WNavigationEnvCfg_VIZ,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.B2WNavMDPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Nav-MDPO-B2W-BallTarget-Viz-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.B2WNavigationEnvCfgBallTarget_VIZ,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.B2WNavMDPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Nav-MDPO-B2W-Test-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.B2WNavigationEnvCfg_TEST,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.B2WNavMDPORunnerCfg,
    },
)

######################################################################################
# PPO

gym.register(
    id="Isaac-Nav-PPO-B2W-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.B2WNavigationEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.B2WNavPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Nav-PPO-B2W-Play-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.B2WNavigationEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.B2WNavPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Nav-PPO-B2W-Dev-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.B2WNavigationEnvCfg_DEV,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.B2WNavPPORunnerDevCfg,
    },
)

gym.register(
    id="Isaac-Nav-PPO-B2W-Test-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.B2WNavigationEnvCfg_TEST,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.B2WNavPPORunnerCfg,
    },
)
