#!/usr/bin/env python
import rospy

from sawyer_control.msg import msg_arm_actions
from sawyer_control import PREFIX

import intera_interface as ii


def execute_action(action_msg):
    action = action_msg.torques
    rs.enable()
    joint_names = arm.joint_names()
    joint_to_values = dict(zip(joint_names, action))
    arm.set_joint_torques(joint_to_values)


def listener():
    node_name = PREFIX + 'arm_actions_subscriber'
    sub_name = PREFIX + 'arm_actions_pub'     # Name of publisher to subscribe
    rospy.init_node(node_name, anonymous=True)
    rospy.Subscriber(sub_name, actions, execute_action)

    global arm
    global rs

    rs = ii.RobotEnable(ii.CHECK_VERSION)
    arm = ii.Limb('right')

    rospy.spin()


if __name__ == '__main__':
    listener()
