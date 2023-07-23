#!/usr/bin/env python
import rospy

from sawyer_control.srv import (
    type_arm_joint_torque_action,
    type_arm_joint_velocity_action,
    type_arm_joint_position_action,
    type_gripper_position
)
from sawyer_control import PREFIX


EXCEPTION_VERBOSE = True

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


if __name__ == "__main__":
    # ======= Example of client: set joint torque =======
    # velocities = [0, 0, 0., 0., 0., 0., 1.0]
    # response = request_arm_joint_set_torque_server(velocities)

    # ======= Example of client: set joint velocity =======
    # velocities = [0, 0, 0., 0., 0., 0., 0.5]
    # response = request_arm_joint_set_velocity_server(velocities)

    # ======= Example of client: set joint position =======
    # positions = [0.1, 0.2, 1., 1., 1., 1., 1.]
    # speed = 0.3
    # response = request_arm_joint_set_position_server(positions, speed)

    # ======= Example of client: move joint to position =======
    # position = [-0.28, -0.60, 0.00, 1.86, 0.00, 0.3, 1.57]
    # # position = [0.0, -1.18, 0.0, 2.18, 0.0, 0.57, 3.3161]  # neutral pose
    # speed = 0.3
    # timeout = 15.0
    # response = request_arm_joint_move_to_position_server(position, speed, timeout)

    # ======= Example of client: set gripper position =======
    position = 0.0
    # position = 0.041667
    response = request_arm_gripper_set_position_server(position)


    print("Done: ", response.done)