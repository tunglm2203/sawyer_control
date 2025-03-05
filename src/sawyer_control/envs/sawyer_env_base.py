import numpy as np
import rospy

import os
import abc
import cv2
import copy
import time

from collections import OrderedDict
import gym
gym.logger.set_level(40)

from sawyer_control.envs.utils import *
from sawyer_control.ros.ros_utils import *

from sawyer_control.controllers.joint_angle_pd_controller import AnglePDController
from sawyer_control import PREFIX
from sawyer_control.core.multitask_env import MultitaskEnv
from sawyer_control.config import default_config

# Import message types
from sawyer_control.msg import (
    msg_arm_joint_torque_action, msg_arm_joint_velocity_action, msg_arm_joint_position_action,
    msg_gripper_action,
)


# TODO: Need to explain action modes
ACTION_MODE_SUPPORT = ["torque", "impedance", "ik_pos", "ik", "ik_quaternion"]
NEW_ACTION_MODE = [
    "position",
    "position_orientation",
    "joint_impedance",
    "joint_torque",
    "joint_velocity",
]

CAMERA_WIDTH = 480
CAMERA_HEIGHT = 480


class SawyerEnvBase(gym.Env, metaclass=abc.ABCMeta):
    def __init__(
            self,
            control_type="torque",
            use_safety_box=True,
            move_speed=0.01,
            rotation_speed=0.01,   # 22.5 in degree
            max_speed = 0.05,
            reset_free=False,
            img_start_col=350, #can range from  0-999
            img_start_row=200, #can range from  0-999
            img_col_delta=300, #can range from  0-999
            img_row_delta=600, #can range from  0-999
            seed=1,
            time_sleep=0.2,
            use_visual_ob=False,
            use_allinone_observation=False,
            yaw_only=False,
    ):
        self.config = default_config

        assert control_type in ACTION_MODE_SUPPORT, f"Action mode: {control_type} does not support."

        self._control_freq = self.config.CONTROL_FREQ
        self._control_type = control_type
        self._max_speed = max_speed     # Max speed to move joints (used to compute duration)
        self._move_speed = move_speed   # step size of move actions (scale ik action in position)
        self._rotate_speed = rotation_speed # step size of rotate actions (scale rotation in "ik" control mode)
        self._discrete_grip = True      # make gripper action either -1 or 1
        self._rescale_actions = False    # rescale actions to [-1,1] and normalize to the control range
        self._auto_align = True         # automatically (perfectly) align two parts when connected
        self._arms = ["right"]          # For Sawyer

        self._robot_ob = True           # includes agent state in observation
        self._object_ob = False          # includes object pose in observation
        self._object_ob_all = False      # includes all object pose in observation
        self._visual_ob = use_visual_ob         # includes camera image in observation
        self._subtask_ob = False        # includes subtask (furniture part id) in observation
        self._segmentation_ob = False   # includes object segmentation for camera
        self._depth_ob = False          # includes depth mapping for camera
        self._camera_ids = [0]          # it can be the camera is our system, if we have more than one
        self._img_obs_width = 256       # width of image observation (Note: it might be different to camera resolution)
        self._img_obs_height = 256      # height of image observation
        self._use_gripper = True        # Option to use gripper or not

        self._control_timestep = 1/20.   # From IKEA: Time between 2 actions of policy
        self._model_timestep = 1./self._control_freq    # From Mujoco: Time between 2 forward times of Mujoco sim. (In mujoco is 0.002)

        self._action_repeat = 3 if self._control_type in ["ik_pos", "ik", "ik_quaternion"] else 1

        self._max_episode_steps = 500
        # Set internal params
        self.joint_torques_scale = 1.0
        self.joint_velocities_scale = 1.0
        self._time_sleep = time_sleep
        self.user_sensitivity = 1     # Following furniture simulation
        
        # self._endpoint_name = "right_gripper_tip"  # ["right_hand", "right_gripper_tip"]
        self._endpoint_name = "right_hand"  # ["right_hand", "right_gripper_tip"]

        # Define min/max range for arm action (i.e., first 7 actions) to scale
        if self._control_type in ["torque"]:
            self.min_range_arm_action = np.array(self.config.JOINT_TORQUE_LOWER)
            self.max_range_arm_action = np.array(self.config.JOINT_TORQUE_UPPER)
            self.arm_action_bias = 0.5 * (self.min_range_arm_action + self.max_range_arm_action)
            self.arm_action_scale = 0.5 * (self.max_range_arm_action - self.min_range_arm_action)
        elif self._control_type in ["impedance", "ik_pos", "ik", "ik_quaternion"]:
            self.min_range_arm_action = np.array(self.config.JOINT_VEL_LOWER)
            self.max_range_arm_action = np.array(self.config.JOINT_VEL_UPPER)
            self.arm_action_bias = 0.5 * (self.min_range_arm_action + self.max_range_arm_action)
            self.arm_action_scale = 0.5 * (self.max_range_arm_action - self.min_range_arm_action)

        self.use_connect_action = False  # Use action "connect", that automatically attempts to connect 2 furniture parts

        self.n_objects = None           # Set to number of object in the scenes

        self._action_on = False         # This flag is used to control by keyboard

        self.queue_size = 1
        self.init_rospy()
        print("[ENV] ROS is successfully initialized.")

        self.use_safety_box = use_safety_box
        self.AnglePDController = AnglePDController(config=self.config)

        self._rng = np.random.RandomState(seed)

        # Set observation & action space from the config
        self._set_observation_space()
        self._set_action_space()

        self.pose_jacobian_dict = self.get_latest_pose_jacobian_dict()

        self.in_reset = True

        if self._endpoint_name == "right_gripper_tip":
            self.ee_geom_at_reset = np.array(self.config.RIGHT_GRIPPER_TIP_RESET_POSE, dtype=np.float64)
        elif self._endpoint_name == "right_hand":
            self.ee_geom_at_reset = np.array(self.config.RIGHT_HAND_RESET_POSE, dtype=np.float64)
        else:
            raise NotImplementedError
        self.reset_free = reset_free

        self.img_start_col = img_start_col
        self.img_start_row = img_start_row
        self.img_col_delta = img_col_delta
        self.img_row_delta = img_row_delta

        self.use_allinone_observation = use_allinone_observation
        self.yaw_only = yaw_only

    def set_max_episode_steps(self, max_episode_steps):
        self._max_episode_steps = max_episode_steps

    @property
    def dof(self):
        return self.config.ROBOT_DOF

    @property
    def gripper_dof(self):
        return self.config.GRIPPER_DOF

    def _set_observation_space(self):
        """
        Setup dict of observation space where keys are ob names and values are dimensions.
        """
        ob_space = OrderedDict()
        if self._visual_ob:
            num_cam = len(self._camera_ids)
            ob_space["camera_ob"] = gym.spaces.Box(
                low=0, high=255,
                shape=(num_cam, self._img_obs_width, self._img_obs_height, 3),
                dtype=np.uint8
            )

        if self._object_ob:
            if self._object_ob_all:
                ob_space["object_ob"] = gym.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=((3 + 4) * self.n_objects, ),
                )
            else:
                ob_space["object_ob"] = gym.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=((3 + 4) * 2,),
                )

        if self._subtask_ob:
            ob_space["subtask_ob"] = gym.spaces.Box(
                low=0.0, high=np.inf,
                shape=(2, ),
            )

        if self._robot_ob:
            GRIPPER_MIN_POS = 0.0
            GRIPPER_MAX_POS = 0.041667
            if self._control_type in ["impedance", "torque"]:
                low = np.hstack([
                    self.config.JOINT_POS_LOWER, self.config.JOINT_VEL_LOWER, GRIPPER_MIN_POS,
                    self.config.EE_POS_LOWER, -np.inf * np.ones(4 + 3 + 3),
                ])
                high = np.hstack([
                    self.config.JOINT_POS_UPPER, self.config.JOINT_VEL_UPPER, GRIPPER_MAX_POS,
                    self.config.EE_POS_UPPER, np.inf * np.ones(4 + 3 + 3),
                ])
                ob_space["robot_ob"] = gym.spaces.Box(
                    low=low, high=high,
                    shape=(7 + 7 + 1 + 3 + 4 + 3 + 3,), # qpos, qvel, gripper, eef_pose (xyz), eef_quat (xyzw), vel_pose (xyz), vel_rot (xyz)
                )
            elif self._control_type in ["ik_pos", "ik", "ik_quaternion"]:
                low = np.hstack([
                    GRIPPER_MIN_POS, self.config.EE_POS_LOWER, -np.inf * np.ones(4), -np.inf * np.ones(3), -np.inf * np.ones(3)
                ])
                high = np.hstack([
                    GRIPPER_MAX_POS, self.config.EE_POS_UPPER, np.inf * np.ones(4), np.inf * np.ones(3), np.inf * np.ones(3)
                ])
                ob_space["robot_ob"] = gym.spaces.Box(
                    low=low, high=high,
                    shape=(1 + 3 + 4 + 3 + 3,), # gripper, eef_pos (xyz), eef_quat (xyzw), eef_vel_pos (xyz), eef_vel_rot (xyz)
                    dtype=np.float32,
                )
            else:
                raise NotImplementedError

        self.observation_space = gym.spaces.Dict(ob_space)

    def _set_action_space(self):
        """
        Setup action space depends on _control_type.
        """
        if self._control_type in ["torque", "impedance"]:
            self.action_space = gym.spaces.Box(
                low=-1.0, high=1.0,
                shape=(self.dof + self.gripper_dof + 1, ),  # joints (7), select (1), connect (1)
                dtype=np.float32,
            )
        elif self._control_type in ["ik_pos"]:
            self.action_space = gym.spaces.Box(
                low=-1.0, high=1.0,
                shape=(3 + self.gripper_dof + 1, ),      # move (3), select (1), connect (1)
                dtype=np.float32,
            )
        elif self._control_type in ["ik"]:
            self.action_space = gym.spaces.Box(
                low=-1.0, high=1.0,
                shape=(3 + 3 + self.gripper_dof + 1, ),    # move (3), rotate (3), select (1), connect (1)
                dtype=np.float32
            )
        elif self._control_type in ["ik_quaternion"]:
            self.action_space = gym.spaces.Box(
                low=-1.0, high=1.0,
                shape=(3 + 4 + self.gripper_dof + 1, ),  # move (3), rotate (4-wxyz), select (1), connect (1)
                dtype=np.float32
            )
        else:
            raise NotImplementedError

    def reset(self, *kwargs):
        """
        Resets the environment.
        """
        self.in_reset = True
        self._reset_robot()
        self._reset_environment()
        self._after_reset()
        self.in_reset = False

        if self.use_allinone_observation:
            return self._get_all_obs()
        else:
            return self._get_obs()

    def step(self, action):
        """
        Computes the next environment state given @action.
        Returns observation dict, reward float, done bool, and info dict.
        """
        self._before_step()
        ob, reward, done, info = self._step(action)
        done, info, penalty = self._after_step(reward, done, info)
        reward += penalty
        return ob, reward, done, info

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
            for _ in range(self.config.NUM_TRIALS_AT_RESET):
                finished = self.move_joint_to_position(self.config.INITIAL_JOINT_ANGLES)
                self.send_gripper_action(self.config.GRIPPER_OPEN_POSITION)
                # self.send_gripper_action(self.config.GRIPPER_CLOSE_POSITION)
                n_trials += 1
                if finished:
                    print(f"[ENV] Reset finished in {(time.time() - time_start):.4f} (s).")
                    break
            time.sleep(self._time_sleep)
            if not finished:
                print(f"[ENV] Reset is not finished after {n_trials} trials.")


    def _reset_environment(self):
        """
        Resets other stuffs from environment.
        """
        pass

    def _after_reset(self):
        """
        Reset timekeeping and internal state for episode.
        """
        pass

    def _before_step(self):
        """
        Called before _step. Do everything before stepping into environment.
        """
        pass

    def _step(self, action):
        """
        Internal step function. Moves agent, updates internal variables, and then
        returns ob, reward, done, info tuple
        """
        self._control(action)
        time.sleep(self._time_sleep)
        if self.use_allinone_observation:
            obs = self._get_all_obs()
        else:
            obs = self._get_obs()

        # Process done, info in _after_step(), not here
        done = False
        reward = 0
        info = {}
        return obs, reward, done, info

    def _after_step(self, reward, terminal, info):
        """
        Called after _step, adds additional information and calculate penalty.
        """
        penalty = 0
        return terminal, info, penalty

    def _control(self, action):
        applied_action = action.copy()
        # Make action for gripper
        if self._discrete_grip:
            applied_action[-2] = self.config.GRIPPER_CLOSE_POSITION if action[-2] < 0 else self.config.GRIPPER_OPEN_POSITION

        # Make 'connect' action
        connect = action[-1]

        # Choose corresponding controller to execute action
        if self._control_type in ["ik_pos", "ik", "ik_quaternion"]:
            self._do_ik_step(applied_action)
        elif self._control_type in ["torque"]:
            self._torque_act(applied_action)
        elif self._control_type in ["impedance"]:
            self._velocity_act(applied_action)
        else:
            raise NotImplementedError

    def _do_ik_step(self, action, use_impedance_control=False):
        """
        action[:3]: difference in position of EE (xyz)
        action[3:7] or action[3:6]: difference in orientation of EE in quaternion (wxyz) or euler (xyz)
        """
        gripper_action = action[-2]

        # Flow: Get current position -> compute next position based on delta -> clip if out of range
        ee_geom_current = self._get_endeffector_pose(self._endpoint_name)
        ee_pos_current = ee_geom_current[:3]
        d_pos = action[:3] * self._move_speed
        ee_pos_next = (ee_pos_current + d_pos)
        ee_pos_next = self._bounded_ee_pos(ee_pos_next)
        d_pos = ee_pos_next - ee_pos_current
        d_pos = d_pos * self.user_sensitivity
        ee_pos_next = ee_pos_current + d_pos

        # Flow: Get current orientation -> compute next orientation based on delta
        if self._control_type == "ik":              # action=[d_pos=(xyz), d_rot=(xyz)]
            ee_ori_current = ee_geom_current[3:7]                   # Quat (xyzw)
            ee_ori_current = convert_quat(ee_ori_current, to="wxyz")
            d_angle = action[3:6] * self._rotate_speed              # Euler (xyz)
            ee_ori_next = euler_to_quat_mul(d_angle, ee_ori_current)# Quat (wxyz)
            ee_ori_next = convert_quat(ee_ori_next, to="xyzw")      # ee_geom_next requires xyzw order
            if self.yaw_only:
                ee_ori_next[2:] *= 0.0
        elif self._control_type == "ik_quaternion": # action=[d_pos=(xyz), d_rot=(wxyz)]
            ee_ori_current = ee_geom_current[3:7]                   # Quat (xyzw)
            d_quat = convert_quat(action[3:7] * self._rotate_speed, to="xyzw")   # input is in wxyz, thus need to convert
            ee_ori_next = quat_multiply(ee_ori_current, d_quat)
        elif self._control_type == "ik_pos":
            # ee_ori_next = self.ee_geom_at_reset[3:7]   # already in xyzw
            ee_ori_next = ee_geom_current[3:7]  # already in xyzw
        else:
            raise NotImplementedError

        # Note: required pose and orientation are in is xyz and xyzw order, respectively
        ee_geom_next = np.concatenate((ee_pos_next, ee_ori_next))

        # Compute target joint angles using IK server
        joint_angles_seed = self.config.INITIAL_JOINT_ANGLES    # Using seed_angle is reset angle seems better
        joint_angles_next = request_ik_server(ee_geom_next, joint_angles_seed, self._endpoint_name)

        if use_impedance_control:
            # From (current and next joint) + duration, intermediate waypoints are constructed, then from these waypoints,
            # we compute intermediate velocities, acceleration to send joint_command
            # Reference: see ImpedanceController in src/sawyer_control/controllers/impedance_controller.py
            ee_geom_current = self._get_endeffector_pose(self._endpoint_name)
            ee_pos_current = ee_geom_current[:3]
            if joint_angles_next is not None:
                self.send_angle_action(joint_angles_next, ee_pos_current, ee_pos_next)
                # Perform action for gripper
                self.send_gripper_action(gripper_action)
        else:
            # P controller from target joint positions to velocities
            if joint_angles_next is None:
                print("[ENV] Warning: IK server did not find solution.")
                raise ValueError
            velocities = self.compute_velocity_control(self.joint_angles, joint_angles_next)

            # scale velocity in range
            if self._rescale_actions:
                velocities = self._scale_action(velocities)

            # keep trying to reach the target in a closed-loop
            for i in range(self._action_repeat):
                self.send_joint_velocity_action(velocities)
                if i < self._action_repeat:
                    velocities = self.compute_velocity_control(self.joint_angles)
                    if self._rescale_actions:# scale velocity in range
                        velocities = self._scale_action(velocities)

                self.send_gripper_action(gripper_action)

    def _torque_act(self, action):
        gripper_action = action[-2]
        torques = action[:self.dof].copy()
        if self._rescale_actions:
            torques = self._scale_action(torques)
        print(f"Commanded torques: {torques}")
        self.send_joint_torque_action(torques)
        self.send_gripper_action(gripper_action)

    def _velocity_act(self, action):
        gripper_action = action[-2]
        velocities = action[:self.dof].copy()
        if self._rescale_actions:
            velocities = self._scale_action(velocities)
        velocities = velocities * self.joint_velocities_scale
        print(f"Commanded velocities: {velocities}")
        self.send_joint_velocity_action(velocities)
        self.send_gripper_action(gripper_action)

    def _get_obs(self, include_qpos=False):
        state = OrderedDict()
        if self._visual_ob:
            state["camera_ob"] = self.get_image()

        if self._segmentation_ob:
            state["segmentation_ob"] = None

        if self._object_ob:
            state["object_ob"] = None

        if self._subtask_ob:
            state["subtask_ob"] = None

        if self._robot_ob:
            robot_states = OrderedDict()
            joint_angles, joint_velocities, endpoint_geometry, endpoint_velocity = request_observation_server(self._endpoint_name)
            if self._control_type in ["impedance", "torque"] or include_qpos:
                robot_states["joint_pos"] = joint_angles.astype(np.float32)
                robot_states["joint_vel"] = joint_velocities.astype(np.float32)

            gripper_pos, gripper_vel, _ = request_gripper_server()
            robot_states["gripper_qpos"] = np.array([gripper_pos])  # 1-dim
            robot_states["eef_pos"] = endpoint_geometry[:3]     # Position of gripper (right_gripper_tip) (xyz)
            robot_states["eef_quat"] = endpoint_geometry[3:7]   # Orientation of gripper (right_gripper_tip) (xyzw)
            robot_states["eef_velp"] = endpoint_velocity[:3]    # Linear velocity of EE (right_gripper_tip) (xyz)
            robot_states["eef_velr"] = endpoint_geometry[3:6]   # Angular velocity of EE (right_gripper_tip) (xyz)

            state["robot_ob"] = np.concatenate([x.ravel() for _, x in robot_states.items()])

        return state

    def _get_all_obs(self, include_qpos=False):
        assert self._control_type in ["ik"]
        state = OrderedDict()
        state["camera_ob"] = self.get_image()

        robot_states = OrderedDict()
        (joint_angles, joint_velocities, endpoint_geometry, endpoint_velocity,
         gripper_position, gripper_velocity,
         robot_image) = request_observation_allinone_server(self._endpoint_name)

        image = np.array(robot_image).reshape(CAMERA_WIDTH, CAMERA_HEIGHT, 3)
        image = image.astype(np.uint8)
        state["camera_ob"] = image

        robot_states["gripper_qpos"] = np.array([gripper_position])  # 1-dim
        robot_states["eef_pos"] = endpoint_geometry[:3]     # Position of gripper (right_gripper_tip) (xyz)
        robot_states["eef_quat"] = endpoint_geometry[3:7]   # Orientation of gripper (right_gripper_tip) (xyzw)
        robot_states["eef_velp"] = endpoint_velocity[:3]    # Linear velocity of EE (right_gripper_tip) (xyz)
        robot_states["eef_velr"] = endpoint_geometry[3:6]   # Angular velocity of EE (right_gripper_tip) (xyz)

        state["robot_ob"] = np.concatenate([x.ravel() for _, x in robot_states.items()])

        return state

    def _compute_reward(self, action):
        pass

    def compute_velocity_control(self, current_joint_angles, target_joint_angles=None):
        # P controller from target joint positions (from IK) to velocities
        # Refer: from IKEA benchmark

        if target_joint_angles is not None:
            self.commanded_joint_positions = target_joint_angles
        delta = current_joint_angles - self.commanded_joint_positions
        velocities = -1.0 * delta       # -5.0 * delta
        velocities = np.clip(velocities, -1.0, 1.0)
        self.commanded_joint_velocities = velocities
        return velocities

    def _scale_action(self, action):
        applied_action = action.copy()
        assert (action <= 1.0).all() and (action >= -1.0).all()
        if self._control_type in ["torque"]:
            applied_action[:self.dof] = self.arm_action_bias + self.arm_action_scale * action[:self.dof]
        elif self._control_type in ["ik_pos", "ik", "ik_quaternion", "impedance"]:
            applied_action[:self.dof] = self.arm_action_bias + self.arm_action_scale * action[:self.dof]
        else:
            raise NotImplementedError
        return applied_action

    @property
    def joint_angles(self):
        return self._get_joint_angles()

    @property
    def joint_velocities(self):
        return self._get_joint_velocities()
    @property
    def eef_pose(self):
        return self._get_endeffector_pose(self._endpoint_name)
    @property
    def eef_velp(self):
        return self._get_endeffector_pose(self._endpoint_name)

    def _get_joint_angles(self):
        joint_angles, _, _, _ = request_observation_server(self._endpoint_name)
        return joint_angles

    def _get_joint_velocities(self):
        _, joint_velocities, _, _ = request_observation_server(self._endpoint_name)
        return joint_velocities

    def _get_endeffector_pose(self, tip_name):
        # Return [(x, y, z), (x, y, z, w)]
        _, _, endpoint_geometry, _ = request_observation_server(tip_name)
        return endpoint_geometry

    def _get_endeffector_vel(self, tip_name):
        _, _, _, endpoint_vel = request_observation_server(tip_name)
        return endpoint_vel

    def _get_endeffector_pos(self, tip_name):
        _, _, endpoint_geometry, _ = request_observation_server(tip_name)
        endpoint_pos = endpoint_geometry[:3]
        return endpoint_pos

    def _get_endeffector_orientation(self, tip_name):
        _, _, endpoint_geometry, _ = request_observation_server(tip_name)
        endpoint_orientation = endpoint_geometry[3:7]
        return endpoint_orientation


    def get_latest_pose_jacobian_dict(self):
        pose_jacobian_dict = request_robot_pose_jacobian_server(self.config.LINK_NAMES)
        return pose_jacobian_dict

    # def _get_positions_from_pose_jacobian_dict(self):
    #     poses = []
    #     for joint in self.pose_jacobian_dict.keys():
    #         poses.append(self.pose_jacobian_dict[joint][0])
    #     return np.array(poses)

    def get_pose_jacobian_dict_of_joints_not_in_box(self, safety_box):
        joint_dict = self.pose_jacobian_dict.copy()
        keys_to_remove = []

        for joint in joint_dict.keys():
            if check_pose_in_box(joint_dict[joint][0], safety_box):
                keys_to_remove.append(joint)

        for key in keys_to_remove:
            del joint_dict[key]

        return joint_dict

    def _get_adjustment_forces_per_joint_dict(self, joint_dict, safety_box):
        forces_dict = {}
        for joint in joint_dict:
            force = self._get_adjustment_force_from_pose(joint_dict[joint][0], safety_box)
            forces_dict[joint] = force
        return forces_dict

    def _get_adjustment_force_from_pose(self, pose, safety_box):
        x, y, z = 0, 0, 0

        curr_x = pose[0]
        curr_y = pose[1]
        curr_z = pose[2]

        if curr_x > safety_box.high[0]:
            x = -1 * np.exp(np.abs(curr_x - safety_box.high[0]) * self.config.SAFETY_FORCE_TEMPERATURE) * self.config.SAFETY_FORCE_MAGNITUDE
        elif curr_x < safety_box.low[0]:
            x = np.exp(np.abs(curr_x - safety_box.low[0]) * self.config.SAFETY_FORCE_TEMPERATURE) * self.config.SAFETY_FORCE_MAGNITUDE

        if curr_y > safety_box.high[1]:
            y = -1 * np.exp(np.abs(curr_y - safety_box.high[1]) * self.config.SAFETY_FORCE_TEMPERATURE) * self.config.SAFETY_FORCE_MAGNITUDE
        elif curr_y < safety_box.low[1]:
            y = np.exp(np.abs(curr_y - safety_box.low[1]) * self.config.SAFETY_FORCE_TEMPERATURE) * self.config.SAFETY_FORCE_MAGNITUDE

        if curr_z > safety_box.high[2]:
            z = -1 * np.exp(np.abs(curr_z - safety_box.high[2]) * self.config.SAFETY_FORCE_TEMPERATURE) * self.config.SAFETY_FORCE_MAGNITUDE
        elif curr_z < safety_box.low[2]:
            z = np.exp(np.abs(curr_z - safety_box.high[2]) * self.config.SAFETY_FORCE_TEMPERATURE) * self.config.SAFETY_FORCE_MAGNITUDE
        return np.array([x, y, z])

    def _bounded_ee_pos(self, ee_pos):
        # Clip next EE pose within predefined safe range
        ee_pos = np.clip(ee_pos, self.config.EE_POS_LOWER, self.config.EE_POS_UPPER)
        return ee_pos


    """ 
    ROS Functions 
    """

    def init_rospy(self):
        node_name = PREFIX + 'sawyer_env'
        rospy.init_node(node_name, anonymous=True)
        self.rate = rospy.Rate(self._control_freq)

    def send_angle_action(self, target_joint_angles, ee_pos_current, ee_pos_next):
        # Compute duration based on _max_speed value
        dist = np.linalg.norm(ee_pos_current - ee_pos_next)
        duration = dist / self._max_speed
        request_angle_action_server(target_joint_angles, duration)
        self.rate.sleep()

    def send_joint_torque_action(self, joint_torque_action, repeat=True):
        response = None
        if repeat:
            for _ in range(int(self._control_timestep / self._model_timestep)):
                response = request_arm_joint_set_torque_server(joint_torque_action)
                self.rate.sleep()
        else:
            response = request_arm_joint_set_torque_server(joint_torque_action)
            self.rate.sleep()
        assert response.done

    def send_joint_velocity_action(self, joint_velocity_action):
        response = None
        for _ in range(int(self._control_timestep / self._model_timestep)):
            response = request_arm_joint_set_velocity_server(joint_velocity_action)
            self.rate.sleep()
        assert response.done

    def send_joint_position_action(self, joint_position_action, speed=0.3):
        response = None
        response = request_arm_joint_set_position_server(joint_position_action, speed)
        self.rate.sleep()
        assert response.done

    def move_joint_to_position(self, joint_position, speed=0.3, timeout=15.0):
        response = None
        response = request_arm_joint_move_to_position_server(joint_position, speed, timeout)
        return response.done

    def send_gripper_action(self, action):
        response = None
        response = request_arm_gripper_set_position_server(action)
        self.rate.sleep()
        assert response.done

    def crop_image(self, img):
        endcol = self.img_start_col + self.img_col_delta
        endrow = self.img_start_row + self.img_row_delta
        img = copy.deepcopy(img[self.img_start_row:endrow, self.img_start_col:endcol])
        return img

    def get_image(self, width=84, height=84):
        image_flatten = request_image_observation_server()
        if image_flatten is None:
            raise Exception("Unable to get image from image_observation server.")

        image = np.array(image_flatten).reshape(CAMERA_WIDTH, CAMERA_HEIGHT, 3)
        # image = image[::-1, :, ::-1]
        # image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        return image.astype(np.uint8)


if __name__ == '__main__':
    # Test get observation
    angle, vel, ee_geom, ee_vel = request_observation_server()
