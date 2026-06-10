# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""B2W specific configuration for navigation environment."""

import os

from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

from isaaclab_nav_task.navigation.navigation_env_cfg import (
    NavigationEnvCfg,
    ObservationsCfgAblatePrioperceptive,
    ObservationsCfgBallTarget,
    RewardsCfgBallTarget,
    TerminationsCfgBallTarget,
)
import isaaclab_nav_task.navigation.mdp as mdp

from isaaclab_nav_task.navigation.assets import B2W_CFG, ISAACLAB_NAV_TASKS_ASSETS_DIR  # isort: skip


LEG_JOINT_NAMES = [".*hip_joint", ".*thigh_joint", ".*calf_joint"]
WHEEL_JOINT_NAMES = [".*foot_joint"]

@configclass
class B2WNavigationEnvCfg(NavigationEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        from isaaclab_nav_task.navigation.mdp.observations import initialize_depth_noise_generator
        from isaaclab_nav_task.navigation.mdp.depth_utils.camera_config import get_camera_config

        initialize_depth_noise_generator(robot_name="b2w", use_jit_precompiled=False)

        camera_config = get_camera_config("b2w")
        CAMERA_RESOLUTION = camera_config.resolution

        self.scene.robot = B2W_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.raycast_camera.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
        self.scene.raycast_camera.offset.pos = (0.387, 0.0, 0.28)
        self.scene.height_scanner_critic.prim_path = "{ENV_REGEX_NS}/Robot/base_link"

        self.terminations.base_contact.params = {"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link", ".*hip", ".*thigh"]), "threshold": 1.0}

        self.actions.velocity_command.low_level_position_action = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*hip_joint", ".*thigh_joint", ".*calf_joint"], scale=0.5, use_default_offset=True)
        self.actions.velocity_command.low_level_velocity_action = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=[".*foot_joint"], scale=5.0, use_default_offset=True)
        self.actions.velocity_command.low_level_policy_file = os.path.join(ISAACLAB_NAV_TASKS_ASSETS_DIR, "Policies", "locomotion", "b2w", "policy_b2w_new_2.pt")

        self.rewards.joint_acc_l2_joint.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES+WHEEL_JOINT_NAMES)}

        self.terminations.base_contact.params = {"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link", ".*hip", ".*thigh"]), "threshold": 1.0}

        self.events.randomize_low_pass_filter_alpha.params = {
            "alpha_range": (0.1, 0.6),
            "action_term": "velocity_command",
            "per_dimension": True,
            "alpha_range_vx": (0.1, 0.6),
            "alpha_range_vy": (0.1, 0.6),
            "alpha_range_omega": (0.1, 0.6),
        }

        self.scene.terrain.max_init_terrain_level = 10
        self.scene.terrain.terrain_generator.difficulty_range = [0.5, 1.0]
        self.scene.terrain.terrain_generator.curriculum = False

@configclass
class B2WNavigationEnvCfgAblatePrioperceptive(B2WNavigationEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.observations = ObservationsCfgAblatePrioperceptive()

@configclass
class B2WNavigationEnvCfgBallTarget(B2WNavigationEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Swap observations, rewards, terminations for visual target ablation
        self.observations = ObservationsCfgBallTarget()
        self.rewards = RewardsCfgBallTarget()
        self.terminations = TerminationsCfgBallTarget()

        # Override B2W-specific reward+termination params
        self.rewards.joint_acc_l2_joint.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES+WHEEL_JOINT_NAMES)}
        self.terminations.base_contact.params = {"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link", ".*hip", ".*thigh"]), "threshold": 1.0}

        # Add goal sphere to scene (kinematic, collidable, 1m diameter)
        self.scene.goal_sphere = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/goal_sphere",
            spawn=sim_utils.SphereCfg(
                radius=0.5,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0)),
        )

        # Make depth camera see the sphere
        self.scene.raycast_camera.mesh_prim_paths = ["/World/ground", "{ENV_REGEX_NS}/goal_sphere"]


@configclass
class B2WNavigationEnvCfg_DEV(B2WNavigationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 30
        self.scene.terrain.max_init_terrain_level = 10
        self.scene.terrain.terrain_generator.difficulty_range = [0.5, 1.0]
        self.scene.terrain.terrain_generator.curriculum = False

@configclass
class B2WNavigationEnvCfg_PLAY(B2WNavigationEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 2
            self.scene.terrain.terrain_generator.num_cols = 2

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class B2WNavigationEnvCfg_VIZ(B2WNavigationEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Minimal env count — we only need the terrain mesh
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5

        # 2x4 grid: 2 difficulty rows × 4 terrain type columns
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 4
        self.scene.terrain.terrain_generator.curriculum = True
        # self.scene.terrain.terrain_generator.difficulty_range = [0.25, 1.25]
        self.scene.terrain.terrain_generator.difficulty_range = [1.0, 1.0]
        # self.scene.terrain.terrain_generator.border_width = 5.0
        self.scene.terrain.max_init_terrain_level = None

        # Equal proportions so each column gets exactly one terrain type
        for sub_cfg in self.scene.terrain.terrain_generator.sub_terrains.values():
            sub_cfg.proportion = 0.25

        # Top-down camera — actual position is set dynamically by the render script
        # after terrain generation (the monkey-patched TerrainGenerator does not center
        # the mesh at world origin). Portrait resolution: rotated 90° CW by the script
        # so that 4 terrain-type columns become the horizontal axis.
        self.viewer.eye = (30.0, 60.0, 100.0)
        self.viewer.lookat = (30.0, 60.0, 0.0)
        self.viewer.origin_type = "world"
        self.viewer.resolution = (3840, 7680)

        # Disable corruption/randomization for clean renders
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class B2WNavigationEnvCfg_TEST(B2WNavigationEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 120
        # 30 rows × 4 cols = 120 tiles. curriculum=True → deterministic col-to-type mapping:
        # col 0=maze, 1=non_maze (pillars), 2=stairs, 3=pits
        self.scene.terrain.terrain_generator.num_rows = 30
        self.scene.terrain.terrain_generator.num_cols = 4
        self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.terrain_generator.difficulty_range = [1.0, 1.0]
        self.scene.terrain.max_init_terrain_level = None

        # Equal proportions so each column gets exactly one terrain type
        for sub_cfg in self.scene.terrain.terrain_generator.sub_terrains.values():
            sub_cfg.proportion = 0.25
