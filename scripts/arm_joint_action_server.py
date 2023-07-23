#!/usr/bin/env python
import rospy

from sawyer_control import PREFIX
from sawyer_control.srv import (
    type_arm_joint_torque_action, type_arm_joint_torque_actionResponse,
    type_arm_joint_velocity_action, type_arm_joint_velocity_actionResponse,
    type_arm_joint_position_action, type_arm_joint_position_actionResponse,
    type_gripper_position, type_gripper_positionResponse,
)
from sawyer_control.config.default_config import JOINT_NAMES, DEFAULT_GRIPPER_VELOCITY, CONTROL_FREQ, USE_GRIPPER

from robot.sawyer_robot import SawyerArm


def handle_arm_joint_torque(request):
    torques = request.torques
    joint_torque_values_dict = dict(zip(JOINT_NAMES, torques))
    arm.set_joint_torques(joint_torque_values_dict)
    return type_arm_joint_torque_actionResponse(True)

def handle_arm_joint_velocity(request):
    velocities = request.velocities
    joint_velocity_values_dict = dict(zip(JOINT_NAMES, velocities))
    arm.set_joint_velocities(joint_velocity_values_dict)
    return type_arm_joint_velocity_actionResponse(True)

def handle_arm_joint_position(request):
    positions = request.positions
    speed = request.speed
    joint_position_values_dict = dict(zip(JOINT_NAMES, positions))
    arm.set_joint_position_speed(speed)  # in range of [0.0 - 1.0]
    arm.set_joint_positions(joint_position_values_dict)
    return type_arm_joint_position_actionResponse(True)

def handle_arm_joint_move_to_position(request):
    positions = request.positions
    speed = request.speed
    timeout = request.timeout
    joint_position_values_dict = dict(zip(JOINT_NAMES, positions))
    arm.set_joint_position_speed(speed)
    done = arm.move_to_joint_positions(joint_position_values_dict, timeout)
    response = type_arm_joint_position_actionResponse()
    response.done = done
    return response

def handle_set_gripper_position(request):
    position = request.position
    arm.gripper.set_position(position)
    return type_gripper_positionResponse(True)

def arm_joint_action_server():
    node_name = PREFIX + 'arm_servers'
    arm_joint_torque_server_name = PREFIX + 'arm_joint_set_torque_server'  # Name of topic that publisher publish to subscriber
    arm_joint_velocity_server_name = PREFIX + 'arm_joint_set_velocity_server'  # Name of topic that publisher publish to subscriber
    arm_joint_position_server_name = PREFIX + 'arm_joint_set_position_server'  # Name of topic that publisher publish to subscriber
    arm_joint_move_to_position_server_name = PREFIX + 'arm_joint_move_to_position_server'  # Name of topic that publisher publish to subscriber
    gripper_position_server_name = PREFIX + 'arm_gripper_set_position_server'  # Name of topic that publisher publish to subscriber
    rospy.init_node(node_name)

    global arm
    arm = SawyerArm(
        default_gripper_vel=DEFAULT_GRIPPER_VELOCITY, control_freq=CONTROL_FREQ,
        use_gripper=USE_GRIPPER
    )

    sv1 = rospy.Service(arm_joint_torque_server_name, type_arm_joint_torque_action, handle_arm_joint_torque)
    sv2 = rospy.Service(arm_joint_velocity_server_name, type_arm_joint_velocity_action, handle_arm_joint_velocity)
    sv3 = rospy.Service(arm_joint_position_server_name, type_arm_joint_position_action, handle_arm_joint_position)
    sv4 = rospy.Service(arm_joint_move_to_position_server_name, type_arm_joint_position_action, handle_arm_joint_move_to_position)
    sv5 = rospy.Service(gripper_position_server_name, type_gripper_position, handle_set_gripper_position)

    rospy.spin()


if __name__ == "__main__":
    arm_joint_action_server()