#!/usr/bin/env python
import rospy

from sawyer_control.msg import msg_gripper_action
from sawyer_control import PREFIX

import intera_interface as ii


def execute_gripper_action(gripper_action_msg):
    gripper_act = gripper_action_msg.position
    gripper.set_position(gripper_act)


def listener():
    node_name = PREFIX + 'gripper_action_subscriber'
    sub_name = PREFIX + 'gripper_action_pub'     # Name of publisher to subscribe
    rospy.init_node(node_name, anonymous=True)
    rospy.Subscriber(sub_name, msg_gripper_action, execute_gripper_action)

    global gripper

    gripper = ii.Gripper('right_gripper')

    rospy.spin()


if __name__ == '__main__':
    listener()
