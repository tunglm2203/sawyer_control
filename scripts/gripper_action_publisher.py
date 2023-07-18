#!/usr/bin/env python
import rospy

from sawyer_control.msg import msg_gripper_action
from sawyer_control import PREFIX


def gripper_action_publisher():
    node_name = PREFIX + 'gripper_action_publisher'
    pub_name = PREFIX + 'gripper_position_topic'
    rospy.init_node(node_name, anonymous=True)

    global gripper_action_pub
    gripper_action_pub = rospy.Publisher(pub_name, msg_gripper_action, queue_size=10)


def send_action(action):
    gripper_action_pub.publish(action)


if __name__ == '__main__':
    try:
        gripper_action_publisher()
        rate = rospy.Rate(1)
        rate.sleep()
        MIN_POSITION = 0.0
        MAX_POSITION = 0.041667
        send_action(MIN_POSITION)
        # send_action(MAX_POSITION)
    except rospy.ROSInterruptException as e:
        print(e)
