import rospy

import abc
import cv2
import copy


from collections import OrderedDict
import gym
gym.logger.set_level(40)

from sawyer_control.envs.utils import *
from sawyer_control.ros.ros_utils import *

from sawyer_control.pd_controllers.joint_angle_pd_controller import AnglePDController
from sawyer_control import PREFIX
from sawyer_control.core.multitask_env import MultitaskEnv
from sawyer_control.configs.config import config_dict as config

# Import message types

from sawyer_control.msg import (
    msg_arm_joint_torque_action, msg_arm_joint_velocity_action,
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

ROBOT_DOF = 7
GRIPPER_DOF = 1

GRIPPER_CLOSE_POSITION = 0.0
GRIPPER_OPEN_POSITION = 0.041667

CAMERA_WIDTH = 480
CAMERA_HEIGHT = 480



class SawyerEnvBase(gym.Env, metaclass=abc.ABCMeta):
    def __init__(
            self,
            control_type="torque",
            use_safety_box=True,
            torque_action_scale=1,
            move_speed=1. / 100,
            rotation_speed=22.5,   # 22.5 in degree
            config_name = 'base_config',
            fix_goal=False,
            max_speed = 0.05,
            reset_free=False,
            img_start_col=350, #can range from  0-999
            img_start_row=200, #can range from  0-999
            img_col_delta=300, #can range from  0-999
            img_row_delta=600, #can range from  0-999
            seed=1,
    ):
        self.config = config[config_name]

        assert control_type in ACTION_MODE_SUPPORT, f"Action mode: {control_type} does not support."

        self._control_freq = self.config.UPDATE_HZ
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
        self._visual_ob = False         # includes camera image in observation
        self._subtask_ob = False        # includes subtask (furniture part id) in observation
        self._segmentation_ob = False   # includes object segmentation for camera
        self._depth_ob = False          # includes depth mapping for camera
        self._camera_ids = [0]          # it can be the camera is our system, if we have more than one
        self._img_obs_width = 256       # width of image observation (Note: it might be different to camera resolution)
        self._img_obs_height = 256      # height of image observation
        self._use_gripper = True        # Option to use gripper or not

        self._control_timestep = 1   # From IKEA: Time between 2 actions of policy
        self._model_timestep = 1. / self.config.UPDATE_HZ    # From Mujoco: Time between 2 forward times of Mujoco sim. (In mujoco is 0.002)

        self._action_repeat = 3 if self._control_type in ["ik_pos", "ik", "ik_quaternion"] else 1
        
        # self._endpoint_name = "right_gripper_tip"  # ["right_hand", "right_gripper_tip"]
        self._endpoint_name = "right_hand"  # ["right_hand", "right_gripper_tip"]

        self.use_connect_action = True  # Use action "connect", that automatically attempts to connect 2 furniture parts

        self.n_objects = None           # Set to number of object in the scenes

        self._action_on = False    # This flag is used to control by keyboard

        self.init_rospy()
        print("[ENV] ROS is successfully initialized.")

        self.use_safety_box = use_safety_box
        self.AnglePDController = AnglePDController(config=self.config)
        print("[ENV] AnglePDController is successfully initialized.")

        self._rng = np.random.RandomState(seed)

        # Set observation & action space from the config
        self._set_observation_space()
        self._set_action_space()

        self._max_episode_steps = 500

        self.pose_jacobian_dict = self.get_latest_pose_jacobian_dict()

        # Set internal params
        self.torque_action_scale = torque_action_scale
        # self.position_action_scale = position_action_scale
        self.in_reset = True
        self._state_goal = None
        self.fix_goal = fix_goal

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

    def set_max_episode_steps(self, max_episode_steps):
        self._max_episode_steps = max_episode_steps

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
                    self.config.JOINT_ANGLES_LOW, self.config.JOINT_VEL_LOW, GRIPPER_MIN_POS,
                    self.config.END_EFFECTOR_POS_LOW, -np.inf * np.ones(4), -np.inf * np.ones(3), -np.inf * np.ones(3),
                ])
                high = np.hstack([
                    self.config.JOINT_ANGLES_HIGH, self.config.JOINT_VEL_HIGH, GRIPPER_MAX_POS,
                    self.config.END_EFFECTOR_POS_HIGH, np.inf * np.ones(4), np.inf * np.ones(3), np.inf * np.ones(3),
                ])
                ob_space["robot_ob"] = gym.spaces.Box(
                    low=low, high=high,
                    shape=(7 + 7 + 1 + 3 + 4 + 3 + 3,), # qpos, qvel, gripper, eef_pose (xyz), eef_quat (xyzw), vel_pose (xyz), vel_rot (xyz)
                )
            elif self._control_type in ["ik_pos", "ik", "ik_quaternion"]:
                low = np.hstack([
                    GRIPPER_MIN_POS, self.config.END_EFFECTOR_POS_LOW, -np.inf * np.ones(4), -np.inf * np.ones(3), -np.inf * np.ones(3)
                ])
                high = np.hstack([
                    GRIPPER_MAX_POS, self.config.END_EFFECTOR_POS_HIGH, np.inf * np.ones(4), np.inf * np.ones(3), np.inf * np.ones(3)
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
            low = np.hstack([self.config.JOINT_TORQUE_LOW, -1.0, -1.0])
            high = np.hstack([self.config.JOINT_TORQUE_HIGH, 1.0 ,1.0])
            self.action_space = gym.spaces.Box(
                low=low, high=high,
                shape=(ROBOT_DOF + GRIPPER_DOF + 1, ),  # joints (7), select (1), connect (1)
                dtype=np.float32,
            )
        elif self._control_type in ["ik_pos"]:
            low = np.hstack([self.config.POSITION_CONTROL_LOW, -1.0, -1.0])
            high = np.hstack([self.config.POSITION_CONTROL_HIGH, 1.0, 1.0])
            self.action_space = gym.spaces.Box(
                low=low, high=high,
                shape=(3 + GRIPPER_DOF + 1, ),      # move (3), select (1), connect (1)
                dtype=np.float32,
            )
        elif self._control_type in ["ik"]:
            const = 1.0
            low = np.hstack([self.config.POSITION_CONTROL_LOW, -const * np.ones(3), -1.0, -1.0])
            high = np.hstack([self.config.POSITION_CONTROL_HIGH, const * np.ones(3), 1.0, 1.0])
            self.action_space = gym.spaces.Box(
                low=low, high=high,
                shape=(3 + 3 + GRIPPER_DOF + 1, ),    # move (3), rotate (3), select (1), connect (1)
                dtype=np.float32
            )
        elif self._control_type in ["ik_quaternion"]:
            const = 1.0
            low = np.hstack([self.config.POSITION_CONTROL_LOW, -const * np.ones(4), -1.0, -1.0])
            high = np.hstack([self.config.POSITION_CONTROL_HIGH, const * np.ones(4), 1.0, 1.0])
            self.action_space = gym.spaces.Box(
                low=low, high=high,
                shape=(3 + 4 + GRIPPER_DOF + 1, ),  # move (3), rotate (4-wxyz), select (1), connect (1)
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
        def check_ee_pose_close_to_target(current_pose, target_pose):
            diff_pos = np.linalg.norm(target_pose[:3] - current_pose[:3])
            diff_quat = np.linalg.norm(target_pose[3:7] - current_pose[3:7])
            if diff_pos < 0.01 and diff_quat < 0.01:
                return True
            return False

        if self.reset_free:
            pass
        else:
            if self._control_type in ["ik_pos", "ik", "ik_quaternion"]:
                gripper_action = GRIPPER_OPEN_POSITION if self._use_gripper else GRIPPER_CLOSE_POSITION

                joint_angles_seed = self.config.RESET_ANGLES  # Using seed_angle is reset angle seems better
                joint_angles_next = request_ik_server(self.ee_geom_at_reset, joint_angles_seed, self._endpoint_name)

                while True:
                    ee_geom_current = self._get_endeffector_geom(self._endpoint_name)
                    self.send_angle_action(joint_angles_next, ee_geom_current[:3], self.ee_geom_at_reset[:3])
                    self.send_gripper_action(gripper_action)

                    if check_ee_pose_close_to_target(ee_geom_current, self.ee_geom_at_reset):
                        break
            elif self._control_type in ["torque", "impedance"]:
                self._safe_move_to_neutral()
            else:
                    raise NotImplementedError

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
            applied_action[-2] = GRIPPER_CLOSE_POSITION if action[-2] < 0 else GRIPPER_OPEN_POSITION

        # Make 'connect' action
        connect = action[-1]

        # Choose corresponding controller to execute action
        if self._control_type in ["ik_pos", "ik", "ik_quaternion"]:
            self._do_ik_step(applied_action)
        elif self._control_type in ["torque"]:
            self._torque_act(applied_action)
        elif self._control_type in ["impedance"]:
            raise
        elif self._control_type in ["position"]:
            raise
        elif self._control_type in ["position_orientation"]:
            raise
        elif self._control_type in ["joint_impedance"]:
            raise
        elif self._control_type in ["joint_torque"]:
            raise
        elif self._control_type in ["joint_velocity"]:
            raise
        else:
            raise NotImplementedError

    def _do_ik_step(self, action, use_impedance_control=False):
        """
        action[:3]: difference in position of EE (xyz)
        action[3:7] or action[3:6]: difference in orientation of EE in quaternion (wxyz) or euler (xyz)
        """
        gripper_action = action[-2]

        # Flow: Get current position -> compute next position based on delta -> clip if out of range
        ee_geom_current = self._get_endeffector_geom(self._endpoint_name)
        ee_pos_current = ee_geom_current[:3]
        d_pos = action[:3] * self._move_speed
        ee_pos_next = (ee_pos_current + d_pos)
        ee_pos_next = self._bounded_ee_pos(ee_pos_next)

        # Flow: Get current orientation -> compute next orientation based on delta
        if self._control_type == "ik":              # action=[d_pos=(xyz), d_rot=(xyz)]
            ee_ori_current = ee_geom_current[3:7]                   # Quat (xyzw)
            ee_ori_current = convert_quat(ee_ori_current, to="wxyz")
            d_angle = action[3:6] * self._rotate_speed              # Euler (xyz)
            ee_ori_next = euler_to_quat_mul(d_angle, ee_ori_current)# Quat (wxyz)
            ee_ori_next = convert_quat(ee_ori_next, to="xyzw")      # ee_geom_next requires xyzw order
        elif self._control_type == "ik_quaternion": # action=[d_pos=(xyz), d_rot=(wxyz)]
            ee_ori_current = ee_geom_current[3:7]                   # Quat (xyzw)
            d_quat = convert_quat(action[3:7] * self._rotate_speed, to="xyzw")   # input is in wxyz, thus need to convert
            ee_ori_next = quat_multiply(ee_ori_current, d_quat)
        elif self._control_type == "ik_pos":
            ee_ori_next = self.ee_geom_at_reset[3:7]   # already in xyzw
        else:
            raise NotImplementedError

        # Note: required pose and orientation are in is xyz and xyzw order, respectively
        ee_geom_next = np.concatenate((ee_pos_next, ee_ori_next))

        # Compute target joint angles using IK server
        joint_angles_seed = self.config.RESET_ANGLES    # Using seed_angle is reset angle seems better
        joint_angles_next = request_ik_server(ee_geom_next, joint_angles_seed, self._endpoint_name)

        if use_impedance_control:
            # From (current and next joint) + duration, intermediate waypoints are constructed, then from these waypoints,
            # we compute intermediate velocities, acceleration to send joint_command
            # Reference: see ImpedanceController in src/sawyer_control/pd_controllers/impedance_controller.py
            ee_geom_current = self._get_endeffector_geom(self._endpoint_name)
            ee_pos_current = ee_geom_current[:3]
            if joint_angles_next is not None:
                self.send_angle_action(joint_angles_next, ee_pos_current, ee_pos_next)
                # Perform action for gripper
                self.send_gripper_action(gripper_action)
        else:
            # P controller from target joint positions to velocities
            if joint_angles_next is None:
                print("[ENV] Warning: IK server did not find solution.")
                pass
            velocities = self.get_velocity_control(self.joint_angles, joint_angles_next)

            # scale velocity in range
            velocities = self._scale_action(velocities)

            # keep trying to reach the target in a closed-loop
            for i in range(self._action_repeat):
                for _ in range(int(self._control_timestep / self._model_timestep)):
                    self.send_joint_velocity_action(velocities)

                if i < self._action_repeat:
                    velocities = self.get_velocity_control(self.joint_angles)
                    # scale velocity in range
                    velocities = self._scale_action(velocities)

                self.send_gripper_action(gripper_action)

    def _torque_act(self, action):
        gripper_action = action[-2]
        if self.use_safety_box:
            safety_box = self.config.RESET_SAFETY_BOX if self.in_reset else self.config.TORQUE_SAFETY_BOX
            self.pose_jacobian_dict = self.get_latest_pose_jacobian_dict()

            # Adjust the action if its elements out of safety box
            pose_jacobian_dict_of_joints_not_in_box = self.get_pose_jacobian_dict_of_joints_not_in_box(safety_box)
            if len(pose_jacobian_dict_of_joints_not_in_box) > 0:
                forces_dict = self._get_adjustment_forces_per_joint_dict(pose_jacobian_dict_of_joints_not_in_box, safety_box)
                torques = np.zeros(ROBOT_DOF)
                for joint in forces_dict:
                    jacobian = pose_jacobian_dict_of_joints_not_in_box[joint][1]
                    force = forces_dict[joint]
                    torques = torques + np.dot(jacobian.T, force).T
                torques[-1] = 0 # we don't need to move the wrist
                action = torques

        torques = action[:ROBOT_DOF]
        if self.in_reset:
            torques = np.clip(torques, self.config.RESET_TORQUE_LOW, self.config.RESET_TORQUE_HIGH)
        else:
            torques = np.clip(torques, self.config.JOINT_TORQUE_LOW, self.config.JOINT_TORQUE_HIGH)
        self.send_joint_torque_action(torques)
        self.send_gripper_action(gripper_action)

    def _get_obs(self, include_qpos=False):
        state = OrderedDict()
        if self._visual_ob:
            state["camera_ob"] = None

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

    def get_velocity_control(self, current_joint_angles, target_joint_angles=None):
        # P controller from target joint positions (from IK) to velocities
        # Refer: from IKEA benchmark

        if target_joint_angles is not None:
            self.commanded_joint_positions = target_joint_angles
        delta = current_joint_angles - self.commanded_joint_positions
        velocities = -2.0 * delta       # -5.0 * delta
        velocities = np.clip(velocities, -1.0, 1.0)
        self.commanded_joint_velocities = velocities
        return velocities

    def _scale_action(self, action):
        _action = action.copy()
        if self._rescale_actions:
            action = np.clip(_action, -1.0, 1.0)
            # TODO: Need to fill this value
            ctrl_range_min, ctrl_range_max = 0., 0.
            mean = 0.5 * (ctrl_range_min + ctrl_range_max)
            scale = 0.5 * (ctrl_range_max - ctrl_range_min)
            applied_action = mean + scale * _action
        else:
            applied_action = _action
        return applied_action

    @property
    def joint_angles(self):
        """
        Returns a numpy array of joint positions.
        Sawyer robots have 7 joints and positions are in rotation angles.
        Order: 'right_j0', 'right_j1', 'right_j2', 'right_j3', 'right_j4', 'right_j5', 'right_j6'
        """
        return self._get_joint_angles()

    def _get_joint_angles(self):
        joint_angles, _, _, _ = request_observation_server()
        return joint_angles

    def _get_endeffector_geom(self, tip_name):
        # Return [(x, y, z), (x, y, z, w)]
        _, _, endpoint_geometry, _ = request_observation_server(tip_name)
        return endpoint_geometry

    def _get_endeffector_pose(self, tip_name):
        _, _, endpoint_geometry, _ = request_observation_server(tip_name)
        endpoint_pose = endpoint_geometry[:3]
        return endpoint_pose

    def _get_endeffector_orientation(self, tip_name):
        _, _, endpoint_geometry, _ = request_observation_server(tip_name)
        endpoint_orientation = endpoint_geometry[3:7]
        return endpoint_orientation

    # @abc.abstractmethod
    # def compute_rewards(self, actions, obs, goals):
    #     pass
    
    def _get_info(self):
        return dict()

    def _safe_move_to_neutral(self):
        for i in range(self.config.RESET_LENGTH):
            current_joint_angles, current_joint_vels, _, _ = request_observation_server()
            torques = self.AnglePDController._compute_pd_forces(current_joint_angles, current_joint_vels)
            self._torque_act(torques * 0.2)
            # TODO: check here
            if self._check_reset_complete():
                break

    def _check_reset_complete(self):
        # Check whether joint angles are near the predefined reset position
        joint_angle_current = self._get_joint_angles()
        joint_angles_neutral = self.AnglePDController._des_angles
        joint_angles_neutral = np.array([joint_angles_neutral[joint] for joint in self.config.JOINT_NAMES])

        errors = compute_angle_difference(joint_angle_current, joint_angles_neutral)
        within_desired_reset_pose = (errors < self.config.RESET_ERROR_THRESHOLD).all()

        # Check whether the arm stops moving
        _, velocities, _, _ = request_observation_server()
        velocities = np.abs(velocities)

        # TODO: Move this one to the config
        VELOCITY_THRESHOLD = .002 * np.ones(ROBOT_DOF)

        is_pause = (velocities < VELOCITY_THRESHOLD).all()
        reset_completed = within_desired_reset_pose and is_pause
        return reset_completed
    
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
        if self.in_reset:
            ee_pos = np.clip(ee_pos, self.config.POSITION_SAFETY_BOX_LOWS, self.config.POSITION_SAFETY_BOX_HIGHS)
        else:
            ee_pos = np.clip(ee_pos, self.config.POSITION_SAFETY_BOX_LOWS, self.config.POSITION_SAFETY_BOX_HIGHS_ACT)
        return ee_pos

    # def _compute_joint_distance_outside_box(self, pose, safety_box):
    #     curr_x = pose[0]
    #     curr_y = pose[1]
    #     curr_z = pose[2]
    #     if(check_pose_in_box(pose, safety_box)):
    #         x, y, z = 0, 0, 0
    #     else:
    #         x, y, z = 0, 0, 0
    #         if curr_x > safety_box.high[0]:
    #             x = np.abs(curr_x - safety_box.high[0])
    #         elif curr_x < safety_box.low[0]:
    #             x = np.abs(curr_x - safety_box.low[0])
    #         if curr_y > safety_box.high[1]:
    #             y = np.abs(curr_y - safety_box.high[1])
    #         elif curr_y < safety_box.low[1]:
    #             y = np.abs(curr_y - safety_box.low[1])
    #         if curr_z > safety_box.high[2]:
    #             z = np.abs(curr_z - safety_box.high[2])
    #         elif curr_z < safety_box.low[2]:
    #             z = np.abs(curr_z - safety_box.low[2])
    #     return np.linalg.norm([x, y, z])

    # @abc.abstractmethod
    # def get_diagnostics(self, paths, prefix=''):
    #     pass


    """ 
    ROS Functions 
    """

    def init_rospy(self):
        node_name = PREFIX + 'sawyer_env'
        arm_joint_torque_pub_name = PREFIX + 'arm_joint_torque_action_pub'
        arm_joint_velocity_pub_name = PREFIX + 'arm_joint_velocity_action_pub'
        gripper_pub_name = PREFIX + 'gripper_action_pub'
        rospy.init_node(node_name)
        self.arm_joint_torque_publisher = rospy.Publisher(arm_joint_torque_pub_name, msg_arm_joint_torque_action, queue_size=1)
        self.arm_joint_velocity_publisher = rospy.Publisher(arm_joint_velocity_pub_name, msg_arm_joint_velocity_action, queue_size=1)
        self.gripper_action_publisher = rospy.Publisher(gripper_pub_name, msg_gripper_action, queue_size=1)
        self.rate = rospy.Rate(self._control_freq)

    def send_angle_action(self, target_joint_angles, ee_pos_current, ee_pos_next):
        # Compute duration based on _max_speed value
        dist = np.linalg.norm(ee_pos_current - ee_pos_next)
        duration = dist / self._max_speed
        request_angle_action_server(target_joint_angles, duration)
        self.rate.sleep()

    def send_joint_torque_action(self, joint_torque_action):
        self.arm_joint_torque_publisher.publish(joint_torque_action)
        self.rate.sleep()

    def send_joint_velocity_action(self, joint_velocity_action):
        self.arm_joint_velocity_publisher.publish(joint_velocity_action)
        self.rate.sleep()

    def send_gripper_action(self, action):
        self.gripper_action_publisher.publish(action)
        self.rate.sleep()

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
        image = image[::-1, :, ::-1]
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        return image.astype(np.uint8)

    """
    Multitask functions
    """

    @property
    def goal_dim(self):
        raise NotImplementedError()

    def get_goal(self):
        return self._state_goal

    def set_goal(self, goal):
        self._state_goal = goal

    def sample_goals(self, batch_size):
        if self.fix_goal:
            goals = np.repeat(
                self.fixed_goal.copy()[None],
                batch_size,
                0
            )
        else:
            if self.use_gazebo_auto and self.__class__.__name__ == 'SawyerPushXYEnv':
                # This `if` code only use for SawyerPushXYEnv env
                while True:
                    goals = np.random.uniform(
                        self.goal_space.low,
                        self.goal_space.high,
                        size=(batch_size, self.goal_space.low.size),
                    )
                    dis_obj_vs_ee = np.linalg.norm(goals[:, :2] - goals[:, 2:], axis=1)
                    if (dis_obj_vs_ee > self.config.OBJECT_RADIUS).all():
                        break
            else:
                goals = np.random.uniform(
                    self.goal_space.low,
                    self.goal_space.high,
                    size=(batch_size, self.goal_space.low.size),
                )
        return goals

    # @abc.abstractmethod
    # def set_to_goal(self, goal):
    #     pass

    """
    Image Env Functions
    """

    def get_env_state(self):
        return self._get_joint_angles(), self._get_endeffector_pose()

    def set_env_state(self, env_state):
        angles, ee_pos = env_state
        for _ in range(3):
            self.send_angle_action(angles, ee_pos)

    def initialize_camera(self, init_fctn):
        pass


if __name__ == '__main__':
    # Test get observation
    angle, vel, ee_geom, ee_vel = request_observation_server()
