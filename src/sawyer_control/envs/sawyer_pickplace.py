import copy
import math
import time
from collections import OrderedDict
import numpy as np
import gym
gym.logger.set_level(40)

from sawyer_control.envs.sawyer_env_base import SawyerEnvBase
from sawyer_control.envs.utils import quaternion_to_euler

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
    def __init__(self, task_name='sawyer-pickup-banana-v0'):

        """
        Do not change these following parameters
        """
        control_type = 'ik'
        reset_free = False
        move_speed = 0.06   # 0.05, 0.1
        rotation_speed = 15.0   # max ~ 6-6.5 degree
        self.task_name = task_name

        super().__init__(
            control_type=control_type, reset_free=reset_free,
            move_speed=move_speed, rotation_speed=rotation_speed,
            use_visual_ob=True, use_allinone_observation=True,
            yaw_only=True
        )

        self._set_observation_space()
        self._set_action_space()
        self.initialize_param_for_task()
        self.global_step = 0

    def initialize_param_for_task(self):
        task_params = {
            'sawyer-pickup-banana-v0': {
                'max_episode_steps': 40,
                'initial_joint': self.config.INITIAL_JOINT_ANGLES.copy(),
            },
            'sawyer-pickup-banana-v1': {
                'max_episode_steps': 90,
                'initial_joint': self.config.INITIAL_JOINT_ANGLES.copy(),
            },
            'sawyer-drawer-open-v0': {
                'max_episode_steps': 60,
                'initial_joint': np.array([-0.1462168, -0.76034473, -0.16979297, 1.90736523, 0.26566504, 0.45724512, -1.90486621]),
            },
        }

        self._max_episode_steps = task_params[self.task_name]['max_episode_steps']
        self.initial_joint = task_params[self.task_name]['initial_joint']

    def _set_observation_space(self):
        low = np.hstack([self.config.EE_POS_LOWER, -np.inf * np.ones(3), np.array([-np.pi]), np.zeros(1)])
        high = np.hstack([self.config.EE_POS_UPPER, np.inf * np.ones(3), np.array([np.pi]), np.ones(1)])
        self.observation_space = gym.spaces.Box(
            low=low, high=high, shape=(3 + 3 + 1 + 1, ),  # ee_pos (xyz), ee_pos_vel (xyz), yaw (radian), gripper (open/close)
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

        if self.global_step >= self._max_episode_steps:
            done = True

        return obs, reward, done, infos

    def reset(self):
        self.global_step = 0
        raw_obs = super().reset()
        obs = self.extract_observation(raw_obs)
        return obs

    def _reset_robot(self):
        """
        Resets robot: initial position, internal variables.
        """

        if self.reset_free:
            pass
        else:
            # Move to neutral pose
            time_start = time.time()
            finished, n_trials = False, 1
            self.send_gripper_action(self.config.GRIPPER_OPEN_POSITION)
            # self.send_gripper_action(self.config.GRIPPER_CLOSE_POSITION)
            cur_ee_pos = self.eef_pose[:3]

            # Move to safe position
            if cur_ee_pos[0] > 0.8 and cur_ee_pos[1] < -0.12:
                # Gradually move EE in safe position before reset
                safe_joint = np.array([-0.066, -0.1034, -0.474, 0.774, 0.5743, 0.9909, -1.7708])  # middle of tray
                for _ in range(self.config.NUM_TRIALS_AT_RESET):
                    finished = self.move_joint_to_position(safe_joint, speed=0.1)
                    if finished:
                        break
            elif cur_ee_pos[2] < -0.04:
                # Gradually move up before reset
                target_ee_pos = cur_ee_pos.copy()
                target_ee_pos[2] = 0.02
                action = np.concatenate([target_ee_pos[:3] - cur_ee_pos[:3], np.array([0, 0, 0, 1, 0])])
                for _ in range(4):
                    self._do_ik_step(action)

            for _ in range(self.config.NUM_TRIALS_AT_RESET):
                finished = self.move_joint_to_position(self.initial_joint, speed=0.1)
                self.send_gripper_action(self.config.GRIPPER_OPEN_POSITION)
                # self.send_gripper_action(self.config.GRIPPER_CLOSE_POSITION)
                n_trials += 1
                if finished:
                    print(f"[ENV] Reset finished in {(time.time() - time_start):.4f} (s).")
                    break
            time.sleep(self._time_sleep)
            if not finished:
                print(f"[ENV] Reset is not finished after {n_trials} trials.")

    def compute_rewards(self, observation, action):
        reward = 0
        return reward

    def extract_observation(self, raw_obs):
        # TODO: state: EE's xyz, EE's yaw (radian), Gripper's state
        gripper_pos = raw_obs['robot_ob'][:1]
        ee_pos_xyz = raw_obs['robot_ob'][1:4]       # xyz
        ee_rot_quat = raw_obs['robot_ob'][4:8]      # xyzw
        ee_rot_x, ee_rot_y, ee_rot_z = quaternion_to_euler(ee_rot_quat[0], ee_rot_quat[1], ee_rot_quat[2], ee_rot_quat[3])
        ee_rot_z = np.array([math.radians(ee_rot_z)])
        ee_pos_vel_xyz = raw_obs['robot_ob'][8:11]
        ee_state = np.concatenate([ee_pos_xyz, ee_pos_vel_xyz, ee_rot_z, gripper_pos])
        obs = {
            "ee_state": ee_state,
            "rgb_image": copy.deepcopy(raw_obs["camera_ob"])
        }
        return obs
