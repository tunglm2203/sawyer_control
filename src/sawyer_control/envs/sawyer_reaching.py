from collections import OrderedDict
import numpy as np
import gym
gym.logger.set_level(40)

from sawyer_control.envs.sawyer_env_base import SawyerEnvBase

class SawyerReachXYZEnv(SawyerEnvBase):
    def __init__(self, max_episode_steps=20, noise_target=0.05):
        self.original_goal = np.array([0.92, 0.28, 0.078])
        self.target_pos = np.array([0.92, 0.28, 0.078])
        if noise_target > 0:
            self.target_pos += np.random.rand(self.target_pos.shape[0]) * noise_target
        self.target_tolerance = 0.05

        control_type = 'ik_pos'
        reset_free = False
        move_speed = 0.1
        rotation_speed = 1.0
        max_speed = 0.3

        super().__init__(
            control_type=control_type, reset_free=reset_free,
            move_speed=move_speed, rotation_speed=rotation_speed,
            max_speed=max_speed
        )

        self.set_max_episode_steps(max_episode_steps)

        self._set_observation_space()
        self._set_action_space()
        self.current_step = 0

    def _set_observation_space(self):
        low = np.hstack([self.config.EE_POS_LOWER, -np.inf * np.ones(3),])
        high = np.hstack([self.config.EE_POS_UPPER, np.inf * np.ones(3)])
        self.observation_space = gym.spaces.Box(
            low=low, high=high, shape=(3 + 3, ),    # eef_pos (xyz), eef_vel_pos (xyz)
            dtype=np.float32,
        )

    def _set_action_space(self):
        assert self._control_type == f"ik_pos", "SawyerReacher only supports ik_pos, but {self._control_type} provided."
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(3,),     # move (xyz)
            dtype=np.float32
        )

    def _get_observation(self):
        return None

    def step(self, action):
        done = False
        self.current_step += 1
        assert action.shape == self.action_space.shape
        action = np.clip(action, self.action_space.low, self.action_space.high) # clip to [-1, 1]

        sending_action = np.concatenate([action, np.array([-1.0, 0.0])])    # Always close gripper
        raw_obs, _, _, _ = super().step(sending_action)
        obs = self.convert_raw_obs_dict_to_array(raw_obs)

        _reward = self.compute_rewards(obs, action)
        infos = {"dense_reward": _reward}

        reward = 1.0  if _reward < self.target_tolerance else 0.0

        if reward == 1.0 or self.current_step == self._max_episode_steps:
            done = True

        return obs, reward, done, infos

    def reset(self, goal=None):
        self.current_step = 0
        raw_obs = super().reset()
        obs = self.convert_raw_obs_dict_to_array(raw_obs)
        if goal is not None:
            self.target_pos = goal
        return obs

    def compute_rewards(self, observation, action):
        cur_pos_xyz = observation[:3]
        target = self.target_pos

        reward = np.linalg.norm(target - cur_pos_xyz)
        return reward

    def convert_raw_obs_dict_to_array(self, raw_obs):
        pos_xyz = raw_obs['robot_ob'][1:4]
        vel_xyz = raw_obs['robot_ob'][8:11]
        obs = np.concatenate([pos_xyz, vel_xyz])
        return obs
