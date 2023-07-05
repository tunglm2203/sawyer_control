import rospy

from sawyer_control.srv import (
    type_observation, type_ik, type_arm_pose_and_jacobian,
    type_angle_action, type_image, type_gripper
)
from sawyer_control.envs.utils import unpack_pose_jacobian_dict
from sawyer_control import PREFIX

import numpy as np


EXCEPTION_VERBOSE = False   # Option to print all exception from ROS


"""
# ========================== ROS clients ==========================
"""
def request_observation_server():
    """
    Return:
        joint_angles: ordered dict of joint name Keys to angle (rad) Values
        joint_velocities: ordered dict of joint name Keys to velocity (rad/s) Values
        endpoint_geometry: Cartesian endpoint pose {position (xyz), orientation (xyzw)}
        endpoint_velocity: Cartesian endpoint twist {linear (xyz), angular (xyz)}
    """
    server_name = PREFIX + 'observation'
    rospy.wait_for_service(server_name)
    try:
        request = rospy.ServiceProxy(server_name, type_observation, persistent=True)
        response = request()    # Get observation from observation_server
        joint_angles = np.array(response.joint_angles, dtype=np.float64)
        joint_velocities = np.array(response.joint_velocities, dtype=np.float64)
        endpoint_geometry = np.array(response.endpoint_geometry, dtype=np.float64)
        endpoint_velocity = np.array(response.endpoint_velocity, dtype=np.float64)
        return (joint_angles, joint_velocities, endpoint_geometry, endpoint_velocity)
    except rospy.ServiceException as e:
        if EXCEPTION_VERBOSE:
            print(e)


def request_joint_angles_server(ee_geometry, current_joint_angles, tip_name="right_gripper_tip"):
    server_name = PREFIX + 'ik'
    rospy.wait_for_service(server_name)
    try:
        get_joint_angles = rospy.ServiceProxy(server_name, type_ik, persistent=True)
        response = get_joint_angles(tip_name, ee_geometry, current_joint_angles)
        return (response.target_joint_angles)
    except rospy.ServiceException as e:
        if EXCEPTION_VERBOSE:
            print(e)


def request_angle_action_server(angles, duration):
    server_name = PREFIX + 'angle_action'
    rospy.wait_for_service(server_name)
    try:
        execute_action = rospy.ServiceProxy(server_name, type_angle_action, persistent=True)
        execute_action(angles, duration)
    except rospy.ServiceException as e:
        if EXCEPTION_VERBOSE:
            print(e)


def request_robot_pose_jacobian_server(link_name):
    server_name = PREFIX + 'arm_pose_jacobian'
    rospy.wait_for_service(server_name)
    try:
        get_robot_pose_jacobian = rospy.ServiceProxy(server_name, type_arm_pose_and_jacobian, persistent=True)
        response = get_robot_pose_jacobian('right')
        return unpack_pose_jacobian_dict(link_name, response.poses, response.jacobians)
    except rospy.ServiceException as e:
        if EXCEPTION_VERBOSE:
            print(e)


def request_image_observation_server():
    server_name = PREFIX + 'image_observation'
    rospy.wait_for_service(server_name)
    try:
        request = rospy.ServiceProxy(server_name, type_image, persistent=True)
        response = request()
        return response.image
    except rospy.ServiceException as e:
        if EXCEPTION_VERBOSE:
            print(e)


def request_gripper_server():
    server_name = PREFIX + 'gripper'
    rospy.wait_for_service(server_name)
    try:
        request = rospy.ServiceProxy(server_name, type_gripper, persistent=True)
        response = request()
        return (response.position, response.velocity, response.force)

    except rospy.ServiceException as e:
        if EXCEPTION_VERBOSE:
            print(e)