#!/usr/bin/env python
import rospy

from sawyer_control.msg import msg_arm_joint_velocity_action
from sawyer_control import PREFIX
from sawyer_control.config.default_config import JOINT_NAMES

import intera_interface


def set_joint_velocity_action(action_msg):
    action = action_msg.velocities
    rs.enable()
    joint_velocity_values_dict = dict(zip(JOINT_NAMES, action))
    arm.set_joint_velocities(joint_velocity_values_dict)


def listener():
    node_name = PREFIX + 'arm_joint_velocity_action_subscriber'
    sub_name = PREFIX + 'arm_joint_velocity_action_pub'  # Name of publisher to subscribe
    rospy.init_node(node_name, anonymous=True)

    rospy.Subscriber(sub_name, msg_arm_joint_velocity_action, set_joint_velocity_action)

    global arm
    global rs

    rs = intera_interface.RobotEnable(intera_interface.CHECK_VERSION)
    arm = intera_interface.Limb('right')

    rospy.spin()


if __name__ == '__main__':
    listener()
