import rospy

from sawyer_control.srv import (
    type_observation, type_observation_allinone, type_ik, type_arm_pose_and_jacobian,
    type_angle_action, type_image, type_gripper,
    type_arm_joint_torque_action,
    type_arm_joint_velocity_action,
    type_arm_joint_position_action,
    type_gripper_position,
    type_object_pose
)
from sawyer_control.envs.utils import unpack_pose_jacobian_dict
from sawyer_control import PREFIX

import numpy as np


EXCEPTION_VERBOSE = False   # Option to print all exception from ROS


"""
# ========================== ROS clients ==========================
"""
def request_observation_allinone_server(tip_name="right_hand"):
    """
    Return:
        joint_angles: ordered dict of joint name Keys to angle (rad) Values
        joint_velocities: ordered dict of joint name Keys to velocity (rad/s) Values
        endpoint_geometry: Cartesian endpoint pose {position (xyz), orientation (xyzw)}
        endpoint_velocity: Cartesian endpoint twist {linear (xyz), angular (xyz)}
        gripper_
    """
    server_name = PREFIX + 'arm_state_allinone_server'
    rospy.wait_for_service(server_name)
    try:
        request = rospy.ServiceProxy(server_name, type_observation_allinone, persistent=True)
        response = request(tip_name)    # Get observation from observation_server

        joint_angles = np.array(response.joint_angles, dtype=np.float64)
        joint_velocities = np.array(response.joint_velocities, dtype=np.float64)
        endpoint_geometry = np.array(response.endpoint_geometry, dtype=np.float64)
        endpoint_velocity = np.array(response.endpoint_velocity, dtype=np.float64)

        gripper_position = np.array(response.gripper_position)
        gripper_velocity = np.array(response.gripper_velocity)

        image = np.array(response.image)

        return (
            joint_angles, joint_velocities, endpoint_geometry, endpoint_velocity,
            gripper_position, gripper_velocity,
            image
        )
    except rospy.ServiceException as e:
        if EXCEPTION_VERBOSE:
            print(e)


def request_observation_server(tip_name="right_hand"):
    """
    Return:
        joint_angles: ordered dict of joint name Keys to angle (rad) Values
        joint_velocities: ordered dict of joint name Keys to velocity (rad/s) Values
        endpoint_geometry: Cartesian endpoint pose {position (xyz), orientation (xyzw)}
        endpoint_velocity: Cartesian endpoint twist {linear (xyz), angular (xyz)}
    """
    server_name = PREFIX + 'arm_joint_state_server'
    rospy.wait_for_service(server_name)
    try:
        request = rospy.ServiceProxy(server_name, type_observation, persistent=True)
        response = request(tip_name)    # Get observation from observation_server
        joint_angles = np.array(response.joint_angles, dtype=np.float64)
        joint_velocities = np.array(response.joint_velocities, dtype=np.float64)
        endpoint_geometry = np.array(response.endpoint_geometry, dtype=np.float64)
        endpoint_velocity = np.array(response.endpoint_velocity, dtype=np.float64)
        return (joint_angles, joint_velocities, endpoint_geometry, endpoint_velocity)
    except rospy.ServiceException as e:
        if EXCEPTION_VERBOSE:
            print(e)


def request_ik_server(ee_geometry, current_joint_angles, tip_name="right_hand"):
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
    server_name = PREFIX + 'arm_gripper_state_server'
    rospy.wait_for_service(server_name)
    try:
        request = rospy.ServiceProxy(server_name, type_gripper, persistent=True)
        response = request()
        return (response.position, response.velocity, response.force)

    except rospy.ServiceException as e:
        if EXCEPTION_VERBOSE:
            print(e)


def request_arm_joint_set_torque_server(torques):
    """
    Send torque command to each joint
    """
    server_name = PREFIX + 'arm_joint_set_torque_server'
    rospy.wait_for_service(server_name)
    try:
        request = rospy.ServiceProxy(server_name, type_arm_joint_torque_action, persistent=True)
        response = request(torques)  # Structure of message in srv/type_arm_joint_torque_action.srv
        return response
    except rospy.ServiceException as e:
        if EXCEPTION_VERBOSE:
            print(e)

def request_arm_joint_set_velocity_server(velocities):
    """
    Send velocity command to each joint
    """
    server_name = PREFIX + 'arm_joint_set_velocity_server'
    rospy.wait_for_service(server_name)
    try:
        request = rospy.ServiceProxy(server_name, type_arm_joint_velocity_action, persistent=True)
        response = request(velocities)  # Structure of message in srv/type_arm_joint_velocity_action.srv
        return response
    except rospy.ServiceException as e:
        if EXCEPTION_VERBOSE:
            print(e)


def request_arm_joint_set_position_server(positions, speed, timeout=1.0):
    """
    Send position command to each joint
    """
    server_name = PREFIX + 'arm_joint_set_position_server'
    rospy.wait_for_service(server_name)
    try:
        request = rospy.ServiceProxy(server_name, type_arm_joint_position_action, persistent=True)
        response = request(positions, speed, timeout)  # Structure of message in srv/type_arm_joint_position_action.srv
        return response
    except rospy.ServiceException as e:
        if EXCEPTION_VERBOSE:
            print(e)


def request_arm_joint_move_to_position_server(positions, speed, timeout):
    server_name = PREFIX + 'arm_joint_move_to_position_server'
    rospy.wait_for_service(server_name)
    try:
        request = rospy.ServiceProxy(server_name, type_arm_joint_position_action, persistent=True)
        response = request(positions, speed, timeout)
        return response
    except rospy.ServiceException as e:
        if EXCEPTION_VERBOSE:
            print(e)


def request_arm_gripper_set_position_server(position):
    server_name = PREFIX + 'arm_gripper_set_position_server'
    rospy.wait_for_service(server_name)
    try:
        request = rospy.ServiceProxy(server_name, type_gripper_position, persistent=True)
        response = request(position)
        return response
    except rospy.ServiceException as e:
        if EXCEPTION_VERBOSE:
            print(e)

def request_workspace_state_server():
    server_name = PREFIX + 'object_state_server'
    rospy.wait_for_service(server_name)
    try:
        request = rospy.ServiceProxy(server_name, type_object_pose, persistent=True)
        response = request()  # Structure of message in srv/type_object_pose.srv
        return (
            np.array(response.pose_table_corner0),
            np.array(response.pose_table_corner1),
            np.array(response.pose_table_corner2),
            np.array(response.pose_table_corner3),
            np.array(response.pose_leg_black),
            np.array(response.pose_leg_blue),
            np.array(response.pose_leg_white),
            np.array(response.pose_leg_pink),
        )
    except rospy.ServiceException as e:
        if EXCEPTION_VERBOSE:
            print(e)
