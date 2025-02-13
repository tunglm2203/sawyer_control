import copy
from collections import OrderedDict
import numpy as np
import gym
gym.logger.set_level(40)

from sawyer_control.envs.sawyer_env_base import SawyerEnvBase

class SawyerPickPlaceXYZEnv(SawyerEnvBase):
    def __init__(self, max_episode_steps=50):

        control_type = 'ik_pos'
        reset_free = False
        move_speed = 0.1
        rotation_speed = 1.0
        max_speed = 0.3

        super().__init__(
            control_type=control_type, reset_free=reset_free,
            move_speed=move_speed, rotation_speed=rotation_speed,
            max_speed=max_speed, use_visual_ob=True
        )

        self.set_max_episode_steps(max_episode_steps)

        self._set_observation_space()
        self._set_action_space()
        self.global_step = 0

    def _set_observation_space(self):
        low = np.hstack([self.config.EE_POS_LOWER, -np.inf * np.ones(3),])
        high = np.hstack([self.config.EE_POS_UPPER, np.inf * np.ones(3)])
        self.observation_space = gym.spaces.Box(
            low=low, high=high, shape=(3 + 3, ),    # eef_pos (xyz), eef_vel_pos (xyz)
            dtype=np.float32,
        )

    def _set_action_space(self):
        assert self._control_type == "ik_pos", f"SawyerReacher only supports ik_pos, but {self._control_type} provided."
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(3,),     # move (xyz)
            dtype=np.float32
        )

    def _get_observation(self):
        return None

    def step(self, action):
        done = False
        self.global_step += 1
        assert action.shape == self.action_space.shape
        action = np.clip(action, self.action_space.low, self.action_space.high) # clip to [-1, 1]

        sending_action = np.concatenate([action, np.array([-1.0, 0.0])])    # Always close gripper
        raw_obs, _, _, _ = super().step(sending_action)
        obs = self.extract_observation(raw_obs)

        reward = 0.0
        infos = {}

        if self.global_step == self._max_episode_steps:
            done = True

        return obs, reward, done, infos

    def reset(self):
        self.global_step = 0
        raw_obs = super().reset()
        obs = self.extract_observation(raw_obs)
        return obs

    def compute_rewards(self, observation, action):
        reward = 0
        return reward

    def extract_observation(self, raw_obs):
        gripper_pos = raw_obs['robot_ob'][:1]
        pos_xyz = raw_obs['robot_ob'][1:4]
        vel_xyz = raw_obs['robot_ob'][8:11]
        ee_state = np.concatenate([pos_xyz, vel_xyz, gripper_pos])
        obs = {
            "proprioceptive": ee_state,
            "rgb_image": copy.deepcopy(raw_obs["camera_ob"])
        }
        return obs


class SawyerPickPlaceXYZYawEnv(SawyerEnvBase):
    def __init__(self, max_episode_steps=50):

        control_type = 'ik'
        reset_free = False
        move_speed = 0.1
        rotation_speed = 15.0   # max ~ 6-6.5 degree
        max_speed = 0.3

        super().__init__(
            control_type=control_type, reset_free=reset_free,
            move_speed=move_speed, rotation_speed=rotation_speed,
            max_speed=max_speed, use_visual_ob=True, use_allinone_observation=True,
            yaw_only=True
        )

        self.set_max_episode_steps(max_episode_steps)

        self._set_observation_space()
        self._set_action_space()
        self.global_step = 0

    def _set_observation_space(self):
        low = np.hstack([self.config.EE_POS_LOWER, -np.inf * np.ones(3), np.zeros(1)])
        high = np.hstack([self.config.EE_POS_UPPER, np.inf * np.ones(3), np.ones(1)])
        self.observation_space = gym.spaces.Box(
            low=low, high=high, shape=(3 + 3 + 1, ),    # eef_pos (xyz), eef_vel_pos (xyz), gripper state (open/close)
            dtype=np.float32,
        )

    def _set_action_space(self):
        assert self._control_type == "ik", f"SawyerReacher only supports ik, but {self._control_type} provided."
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(5,),     # move delta(xyz), delta(yaw), gripper
            dtype=np.float32
        )

    def _get_observation(self):
        return None

    def step(self, action):
        done = False
        self.global_step += 1
        assert action.shape == self.action_space.shape
        action = np.clip(action, self.action_space.low, self.action_space.high) # clip to [-1, 1]

        UNUSED_CONNECTOR = 0.0
        delta_position = action[0:3]
        delta_orientation = np.array([0.0, 0.0, action[3]])
        gripper = np.array([-1.0]) if action[4] == 0 else np.array([1.0])
        sending_action = np.concatenate([delta_position, delta_orientation, gripper, np.array([UNUSED_CONNECTOR])])
        raw_obs, _, _, _ = super().step(sending_action)
        obs = self.extract_observation(raw_obs)

        reward = 0.0
        infos = {}

        if self.global_step == self._max_episode_steps:
            done = True

        return obs, reward, done, infos

    def reset(self):
        self.global_step = 0
        raw_obs = super().reset()
        obs = self.extract_observation(raw_obs)
        return obs

    def compute_rewards(self, observation, action):
        reward = 0
        return reward

    def extract_observation(self, raw_obs):
        gripper_pos = raw_obs['robot_ob'][:1]
        pos_xyz = raw_obs['robot_ob'][1:4]
        vel_xyz = raw_obs['robot_ob'][8:11]
        ee_state = np.concatenate([pos_xyz, vel_xyz, gripper_pos])
        obs = {
            "proprioceptive": ee_state,
            "rgb_image": copy.deepcopy(raw_obs["camera_ob"])
        }
        return obs
