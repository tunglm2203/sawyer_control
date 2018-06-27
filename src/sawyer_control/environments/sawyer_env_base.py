import numpy as np
import rospy
from sawyer_control.joint_angle_pd_controller import AnglePDController
from sawyer_control.eval_util import create_stats_ordered_dict
from sawyer_control.serializable import Serializable
from collections import OrderedDict
from rllab.spaces.box import Box
from gym.spaces import Box #TODO: INTEGRATE THIS IN TO WORK
from sawyer_control.srv import observation
from sawyer_control.msg import actions
from sawyer_control.srv import getRobotPoseAndJacobian
from sawyer_control.srv import ik
from sawyer_control.srv import angle_action
from rllab.envs.base import Env
import gym
import time
#TODO: FIX IMPORTS
from sawyer_control.core.multitask_env import MultitaskEnv


class SawyerEnv(gym.Env, Serializable, MultitaskEnv):
    def __init__(
            self,
            update_hz=20,
            action_mode='torque',
            safety_box=True,
            huber_delta=10,
            safety_force_magnitude=5,
            safety_force_temp=5,
            safe_reset_length=200,
            ee_pd_time_steps=25,
            ee_pd_scale = 25,
            ee_pd_damping_scale=20,
            ee_pd_action_limit=1,
            img_observation = False
    ):
        Serializable.quick_init(self, locals())
        self.init_rospy(update_hz)
        self.reset_safety_box_lows = np.array([-.2, -0.6, 0])
        self.reset_safety_box_highs = np.array([.9, 0.4, 2])
        self.safety_box_lows = self.not_reset_safety_box_lows = [0.1, -0.5, 0]
        self.safety_box_highs = self.not_reset_safety_box_highs = [0.7, 0.5, 0.7]

        self.joint_names = ['right_j0', 'right_j1', 'right_j2', 'right_j3', 'right_j4', 'right_j5', 'right_j6']
        self.link_names = ['right_l2', 'right_l3', 'right_l4', 'right_l5', 'right_l6', '_hand']
        if action_mode == 'position':
            # self.ee_safety_box_low = np.array([0.23, -.302, 0.03])
            # self.ee_safety_box_high = np.array([0.60, .47, 0.409])
            self.ee_safety_box_low = np.array([.2, -.2, .03])
            self.ee_safety_box_high = np.array([.6, .2, .5])
            # original ee safety box
            #self.ee_safety_box_high = np.array([0.73, 0.32, 0.4])
            #self.ee_safety_box_low = np.array([0.52, 0.03, 0.05])

        # image box
        # self.safety_box_lows = self.not_reset_safety_box_lows = [.2, -.332, .00]
        # self.safety_box_highs = self.not_reset_safety_box_highs = [.63, .5, .429]
        # self.safety_box_lows = self.not_reset_safety_box_lows = [.2, -.2, .00]
        # self.safety_box_highs = self.not_reset_safety_box_highs = [.6, .2, .5]

        self.action_mode = action_mode
        self.safety_box = safety_box
        self.safe_reset_length = safe_reset_length
        self.huber_delta = huber_delta
        self.safety_force_magnitude = safety_force_magnitude
        self.safety_force_temp = safety_force_temp
        self.AnglePDController = AnglePDController()

        max_torques = 1/2 * np.array([8, 7, 6, 5, 4, 3, 2])
        self.joint_torque_high = max_torques
        self.joint_torque_low = -1 * max_torques
        self.ee_pd_time_steps = ee_pd_time_steps
        self.ee_pd_scale = ee_pd_scale
        self.ee_pd_damping_scale = ee_pd_damping_scale
        self.ee_pd_action_limit = 1
        self.img_observation = img_observation
        self._set_action_space()
        self._set_observation_space()
        self.get_latest_pose_jacobian_dict()
        self.in_reset = True
        self.amplify = np.ones(7)*self.joint_torque_high

    def _act(self, action):
        if self.action_mode == 'position':
            self._joint_act(action/10)
        else:
            self._torque_act(action)
        return

    def _joint_act(self, action):
        ee_pos = self._end_effector_pose()
        target_ee_pos = (ee_pos[:3] + action)
        target_ee_pos = np.clip(target_ee_pos, self.ee_safety_box_low, self.ee_safety_box_high)
        target_ee_pos = np.concatenate((target_ee_pos, ee_pos[3:]))
        angles = self.request_ik_angles(target_ee_pos, self._joint_angles())
        self.send_angle_action(angles)

    def _torque_act(self, action):
        if self.safety_box:
            if self.in_reset:
                self.safety_box_highs = self.reset_safety_box_highs
                self.safety_box_lows = self.reset_safety_box_lows
            else:
                self.safety_box_lows = self.not_reset_safety_box_lows
                self.safety_box_highs = self.not_reset_safety_box_highs
            self.get_latest_pose_jacobian_dict()
            truncated_dict = self.check_joints_in_box()
            if len(truncated_dict) > 0:
                forces_dict = self._get_adjustment_forces_per_joint_dict(truncated_dict)
                torques = np.zeros(7)
                for joint in forces_dict:
                    jacobian = truncated_dict[joint][1]
                    force = forces_dict[joint]
                    torques = torques + np.dot(jacobian.T, force).T
                torques[-1] = 0
                action = torques
        if self.in_reset:
            np.clip(action, -5, 5, out=action)
        else:
            action = self.amplify * action
            action = np.clip(np.asarray(action), self.joint_torque_low, self.joint_torque_high)
        self.send_action(action)
        self.rate.sleep()
        curr = time.time()
        self.prev = curr

    def _reset_angles_within_threshold(self):
        desired_neutral = self.AnglePDController._des_angles
        desired_neutral = np.array([desired_neutral[joint] for joint in self.joint_names])
        actual_neutral = (self._joint_angles())
        errors = self.compute_angle_difference(desired_neutral, actual_neutral)
        ERROR_THRESHOLD = .15*np.ones(7)
        is_within_threshold = (errors < ERROR_THRESHOLD).all()
        return is_within_threshold

    def _wrap_angles(self, angles):
        return angles % (2*np.pi)

    def _joint_angles(self):
        angles, _, _, _ = self.request_observation()
        angles = np.array(angles)
        return angles

    def _end_effector_pose(self):
        _, _, _, endpoint_pose = self.request_observation()
        return np.array(endpoint_pose)

    def compute_angle_difference(self, angles1, angles2):
        self._wrap_angles(angles1)
        self._wrap_angles(angles2)
        deltas = np.abs(angles1 - angles2)
        differences = np.minimum(2 * np.pi - deltas, deltas)
        return differences

    def step(self, action):
        self._act(action)
        observation = self._get_observation()
        reward = self.compute_rewards(action, observation, self._state_goal)
        done = False
        info = dict(true_state=observation)
        if self.img_observation:
            observation = self._get_image()
        return observation, reward, done, info

    def _get_observation(self):
        angles = self._joint_angles()
        _, velocities, torques, _ = self.request_observation()
        velocities = np.array(velocities)
        torques = np.array(torques)
        endpoint_pose = self._end_effector_pose()

        temp = np.hstack((
            angles,
            velocities,
            torques,
            endpoint_pose,
            self._state_goal
        ))
        return temp

    def _safe_move_to_neutral(self):
        for i in range(self.safe_reset_length):
            cur_pos, cur_vel, _, _ = self.request_observation()
            torques = self.AnglePDController._compute_pd_forces(cur_pos, cur_vel)
            self._torque_act(torques)
            if self._reset_complete():
                break

    def _reset_complete(self):
        close_to_desired_reset_pos = self._reset_angles_within_threshold()
        _, velocities, _, _ = self.request_observation()
        velocities = np.abs(np.array(velocities))
        VELOCITY_THRESHOLD = .002 * np.ones(7)
        no_velocity = (velocities < VELOCITY_THRESHOLD).all()
        return close_to_desired_reset_pos and no_velocity

    def reset(self):
        self.in_reset = True
        self._safe_move_to_neutral()
        self.in_reset = False
        return self._get_observation()

    def get_latest_pose_jacobian_dict(self):
        self.pose_jacobian_dict = self._get_robot_pose_jacobian_client('right')

    def _get_robot_pose_jacobian_client(self, name):
        rospy.wait_for_service('get_robot_pose_jacobian')
        try:
            get_robot_pose_jacobian = rospy.ServiceProxy('get_robot_pose_jacobian', getRobotPoseAndJacobian,
                                                         persistent=True)
            resp = get_robot_pose_jacobian(name)
            pose_jac_dict = self.get_pose_jacobian_dict(resp.poses, resp.jacobians)
            return pose_jac_dict
        except rospy.ServiceException as e:
            print(e)

    def get_pose_jacobian_dict(self, poses, jacobians):
        pose_jacobian_dict = {}
        pose_counter = 0
        jac_counter = 0
        poses = np.array(poses)
        jacobians = np.array(jacobians)
        for link in self.link_names:
            pose = poses[pose_counter:pose_counter + 3]
            jacobian = []
            for i in range(jac_counter, jac_counter+21, 7):
                jacobian.append(jacobians[i:i+7])
            jacobian = np.array(jacobian)
            pose_counter += 3
            jac_counter += 21
            pose_jacobian_dict[link] = [pose, jacobian]
        return pose_jacobian_dict

    def _get_positions_from_pose_jacobian_dict(self):
        poses = []
        for joint in self.pose_jacobian_dict.keys():
            poses.append(self.pose_jacobian_dict[joint][0])
        return np.array(poses)

    def check_joints_in_box(self):
        joint_dict = self.pose_jacobian_dict.copy()
        keys_to_remove = []
        for joint in joint_dict.keys():
            if self._pose_in_box(joint_dict[joint][0]):
                keys_to_remove.append(joint)
        for key in keys_to_remove:
            del joint_dict[key]
        return joint_dict

    def _pose_in_box(self, pose):
        #TODO: DOUBLE CHECK THIS WORKS
        within_box = self.safety_box.contains(pose)
        return within_box

    def _get_adjustment_forces_per_joint_dict(self, joint_dict):
        forces_dict = {}
        for joint in joint_dict:
            force = self._get_adjustment_force_from_pose(joint_dict[joint][0])
            forces_dict[joint] = force
        return forces_dict

    def _get_adjustment_force_from_pose(self, pose):
        x, y, z = 0, 0, 0

        curr_x = pose[0]
        curr_y = pose[1]
        curr_z = pose[2]

        if curr_x > self.safety_box_highs[0]:
            x = -1 * np.exp(np.abs(curr_x - self.safety_box_highs[0]) * self.safety_force_temp) * self.safety_force_magnitude
        elif curr_x < self.safety_box_lows[0]:
            x = np.exp(np.abs(curr_x - self.safety_box_lows[0]) * self.safety_force_temp) * self.safety_force_magnitude

        if curr_y > self.safety_box_highs[1]:
            y = -1 * np.exp(np.abs(curr_y - self.safety_box_highs[1]) * self.safety_force_temp) * self.safety_force_magnitude
        elif curr_y < self.safety_box_lows[1]:
            y = np.exp(np.abs(curr_y - self.safety_box_lows[1]) * self.safety_force_temp) * self.safety_force_magnitude

        if curr_z > self.safety_box_highs[2]:
            z = -1 * np.exp(np.abs(curr_z - self.safety_box_highs[2]) * self.safety_force_temp) * self.safety_force_magnitude
        elif curr_z < self.safety_box_lows[2]:
            z = np.exp(np.abs(curr_z - self.safety_box_highs[2]) * self.safety_force_temp) * self.safety_force_magnitude
        return np.array([x, y, z])

    def _compute_joint_distance_outside_box(self, pose):
        curr_x = pose[0]
        curr_y = pose[1]
        curr_z = pose[2]
        if(self._pose_in_box(pose)):
            x, y, z = 0, 0, 0
        else:
            x, y, z = 0, 0, 0
            if curr_x > self.safety_box_highs[0]:
                x = np.abs(curr_x - self.safety_box_highs[0])
            elif curr_x < self.safety_box_lows[0]:
                x = np.abs(curr_x - self.safety_box_lows[0])
            if curr_y > self.safety_box_highs[1]:
                y = np.abs(curr_y - self.safety_box_highs[1])
            elif curr_y < self.safety_box_lows[1]:
                y = np.abs(curr_y - self.safety_box_lows[1])
            if curr_z > self.safety_box_highs[2]:
                z = np.abs(curr_z - self.safety_box_highs[2])
            elif curr_z < self.safety_box_lows[2]:
                z = np.abs(curr_z - self.safety_box_lows[2])
        return np.linalg.norm([x, y, z])

    def log_diagnostics(self, paths, logger=None):
        '''
        :param paths: dictionary of trajectory information
        :param logger: rllab logger or similar variant should be passed in
        :return: None
        '''
        if logger == None:
            pass
        else:
            statistics = self._get_statistics_from_paths(paths)
            for key, value in statistics.items():
                logger.record_tabular(key, value)

    def _update_statistics_with_observation(self, observation, stat_prefix, log_title):
        statistics = OrderedDict()
        statistics.update(create_stats_ordered_dict(
            '{} {}'.format(stat_prefix, log_title),
            observation,
        ))
        return statistics

    def _get_statistics_from_paths(self, paths):
        raise NotImplementedError()

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space
    
    def _set_action_space(self):
        max_torques = 0.5 * np.array([8, 7, 6, 5, 4, 3, 2])
        self.joint_torque_high = max_torques
        self.joint_torque_low = -1 * max_torques

        if self.action_mode == 'position':
            delta_high = self.ee_pd_action_limit * np.ones(3)
            delta_low = self.ee_pd_action_limit * -1 * np.ones(3)

            self._action_space = Box(
                delta_low,
                delta_high,
            )
        else:
            self._action_space = Box(
                self.joint_torque_low,
                self.joint_torque_high
            )
            
    """ 
    ROS Functions 
    """

    def init_rospy(self, update_hz):
        rospy.init_node('sawyer_env', anonymous=True)
        self.action_publisher = rospy.Publisher('actions_publisher', actions, queue_size=10)
        self.rate = rospy.Rate(update_hz)

    def send_action(self, action):
        self.action_publisher.publish(action)

    def send_angle_action(self, action):
        self.request_angle_action(action)

    def request_observation(self):
        rospy.wait_for_service('observations')
        try:
            request = rospy.ServiceProxy('observations', observation, persistent=True)
            obs = request()
            #TODO: FIX THIS TO RETURN NP ARRAYS ONLY 
            return (
                    obs.angles,
                    obs.velocities,
                    obs.torques,
                    obs.endpoint_pose
            )
        except rospy.ServiceException as e:
            print(e)

    def request_angle_action(self, angles):
        rospy.wait_for_service('angle_action')
        try:
            execute_action = rospy.ServiceProxy('angle_action', angle_action, persistent=True)
            execute_action(angles)
            return None
        except rospy.ServiceException as e:
            print(e)


    def request_ik_angles(self, ee_pos, joint_angles):
        rospy.wait_for_service('ik')
        try:
            get_joint_angles = rospy.ServiceProxy('ik', ik, persistent=True)
            resp = get_joint_angles(ee_pos, joint_angles)

            return (
                resp.joint_angles
            ) #TODO: why is this in a tuple? can this be removed?
        except rospy.ServiceException as e:
            print(e)

    """
       Multitask functions
       """

    @property
    def goal_dim(self):
        return 3

    def get_goal(self):
        return self._state_goal

    def sample_goals(self, batch_size):
        if self.fix_goal:
            goals = np.repeat(
                self.fixed_goal.copy()[None],
                batch_size,
                0
            )
        else:
            goals = np.random.uniform(
                self.goal_space.low,
                self.goal_space.high,
                size=(batch_size, self.goal_space.low.size),
            )
        return goals

    def compute_rewards(self, actions, obs, goals):
        distances = np.linalg.norm(obs - goals, axis=1)
        if self.reward_type == 'hand_distance':
            r = -distances
        elif self.reward_type == 'hand_success':
            r = -(distances < self.indicator_threshold).astype(float)
        else:
            raise NotImplementedError("Invalid/no reward type.")
        return r

    def set_to_goal(self, goal):
        raise NotImplementedError()

    def get_env_state(self):
        raise NotImplementedError()

    def set_env_state(self, state):
        raise NotImplementedError()