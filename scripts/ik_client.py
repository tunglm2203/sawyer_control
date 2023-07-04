#!/usr/bin/env python
import rospy

from sawyer_control.srv import type_ik
from sawyer_control import PREFIX

"""
This file is used for test ik_server, it is not called in launch file.
"""

joint_names = ['right_j0', 'right_j1', 'right_j2', 'right_j3', 'right_j4', 'right_j5', 'right_j6']


def request_joint_angles(tip_name, ee_pos, joint_angles):
    server_name = PREFIX + 'ik'
    rospy.wait_for_service(server_name)
    try:
        request = rospy.ServiceProxy(server_name, type_ik, persistent=True)
        response = request(tip_name, ee_pos, joint_angles)

        return response.target_joint_angles
    except rospy.ServiceException as e:
        print(e)


if __name__ == "__main__":
    ee_pose = [0.603529721243, 0.00140361630124, 0.199834650308,
               -0.00141560520355, 0.999987273934, 0.00140890396491, -0.00463282001645]
    seed_joint_angles = {
        'right_j6': 1.0, 'right_j5': 1.0, 'right_j4': 0.6,
        'right_j3': 1.5, 'right_j2': -0.7, 'right_j1': -0.76, 'right_j0': 0.2
    }
    tip_name = 'right_gripper_tip'
    seed_joint_angles = [seed_joint_angles[joint] for joint in joint_names]
    target_joint_angles = request_joint_angles(tip_name, ee_pose, seed_joint_angles)
    print(target_joint_angles)
