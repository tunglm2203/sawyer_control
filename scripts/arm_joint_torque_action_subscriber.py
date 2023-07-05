#!/usr/bin/env python
import rospy

from sawyer_control.msg import msg_arm_joint_torque_action
from sawyer_control import PREFIX

import intera_interface as ii


def set_joint_torque_action(action_msg):
    action = action_msg.torques
    rs.enable()
    joint_names = arm.joint_names()
    joint_torque_values_dict = dict(zip(joint_names, action))
    arm.set_joint_torques(joint_torque_values_dict)


def listener():
    node_name = PREFIX + 'arm_joint_torque_action_subscriber'
    sub_name = PREFIX + 'arm_joint_torque_action_pub' # Name of publisher to subscribe
    rospy.init_node(node_name, anonymous=True)

    rospy.Subscriber(sub_name, msg_arm_joint_torque_action, set_joint_torque_action)

    global arm
    global rs

    rs = ii.RobotEnable(ii.CHECK_VERSION)
    arm = ii.Limb('right')

    rospy.spin()


if __name__ == '__main__':
    listener()
