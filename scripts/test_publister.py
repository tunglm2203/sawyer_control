#!/usr/bin/env python
import rospy

from sawyer_control.msg import msg_arm_joint_torque_action
from sawyer_control import PREFIX


def actions_publisher():
    node_name = PREFIX + 'arm_joint_torque_action_publisher'
    pub_name = PREFIX + 'tung_topic'
    rospy.init_node(node_name, anonymous=True)

    global action_pub
    action_pub = rospy.Publisher(pub_name, msg_arm_joint_torque_action, queue_size=10)


def send_action(action):
    action_pub.publish(action)


if __name__ == '__main__':
    try:
        actions_publisher()
        rate = rospy.Rate(1)
        rate.sleep()
        send_action([0, 0, 0, 0, 0, 0, 1])
    except rospy.ROSInterruptException as e:
        print(e)