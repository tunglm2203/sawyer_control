#!/usr/bin/env python
import rospy

from sawyer_control.msg import (
    msg_arm_joint_torque_action,
    msg_arm_joint_velocity_action,
    msg_arm_joint_position_action,
    msg_gripper_action
)
from sawyer_control import PREFIX
from sawyer_control.config.default_config import JOINT_NAMES

import intera_interface


def set_joint_torques(action_msg):
    action = action_msg.torques
    joint_torque_values_dict = dict(zip(JOINT_NAMES, action))
    arm.set_joint_torques(joint_torque_values_dict)


def set_joint_velocities(action_msg):
    action = action_msg.velocities
    joint_velocity_values_dict = dict(zip(JOINT_NAMES, action))
    arm.set_joint_velocities(joint_velocity_values_dict)


def set_joint_positions(action_msg):
    speed = action_msg.speed
    action = action_msg.positions
    joint_position_values_dict = dict(zip(JOINT_NAMES, action))
    arm.set_joint_position_speed(speed)     # in range of [0.0 - 1.0]
    arm.set_joint_positions(joint_position_values_dict)


def set_gripper_positions(gripper_action_msg):
    gripper_act = gripper_action_msg.position
    gripper.set_position(gripper_act)


def listener():
    node_name = PREFIX + 'arm_subscribers'
    arm_joint_torque_topic_name = PREFIX + 'arm_joint_torque_topic'     # Name of topic that publisher publish to subscriber
    arm_joint_velocity_topic_name = PREFIX + 'arm_joint_velocity_topic' # Name of topic that publisher publish to subscriber
    arm_joint_position_topic_name = PREFIX + 'arm_joint_position_topic' # Name of topic that publisher publish to subscriber
    gripper_position_topic_name = PREFIX + 'gripper_position_topic'     # Name of topic that publisher publish to subscriber

    rospy.init_node(node_name)

    rospy.Subscriber(arm_joint_torque_topic_name, msg_arm_joint_torque_action, set_joint_torques)
    rospy.Subscriber(arm_joint_velocity_topic_name, msg_arm_joint_velocity_action, set_joint_velocities)
    rospy.Subscriber(arm_joint_position_topic_name, msg_arm_joint_position_action, set_joint_positions)
    rospy.Subscriber(gripper_position_topic_name, msg_gripper_action, set_gripper_positions)

    global arm
    global gripper

    arm = intera_interface.Limb('right')
    gripper = intera_interface.Gripper('right_gripper')
    gripper.set_cmd_velocity(3.0)

    rospy.spin()


if __name__ == '__main__':
    listener()
