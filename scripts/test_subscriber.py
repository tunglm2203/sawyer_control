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
    rospy.loginfo("Send any thing here ...")

def listener():
    node_name = PREFIX + 'arm_subscribers'
    arm_joint_torque_topic_name = PREFIX + 'tung_topic'     # Name of topic that publisher publish to subscriber

    rospy.init_node(node_name)

    rospy.Subscriber(arm_joint_torque_topic_name, msg_arm_joint_torque_action, set_joint_torques)

    global arm

    rospy.spin()


if __name__ == '__main__':
    listener()