#!/usr/bin/env python
import rospy

from sawyer_control.msg import msg_arm_joint_position_action
from sawyer_control import PREFIX


def actions_publisher():
    node_name = PREFIX + 'arm_joint_position_publisher'
    pub_name = PREFIX + 'arm_joint_position_topic'
    rospy.init_node(node_name, anonymous=True)

    global action_pub
    action_pub = rospy.Publisher(pub_name, msg_arm_joint_position_action, queue_size=10)


def send_action(speed, action):
    timeout = 1.0   # unused
    action_pub.publish(speed, action, timeout)


if __name__ == '__main__':
    try:
        actions_publisher()
        rate = rospy.Rate(1)
        rate.sleep()
        position = [0.1, 0.2, 1., 1., 1., 1., 1.]
        speed = 0.3
        send_action(speed, position)
    except rospy.ROSInterruptException as e:
        print(e)
