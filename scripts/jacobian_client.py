#!/usr/bin/env python

import rospy

from sawyer_control.srv import type_arm_pose_and_jacobian
from sawyer_control import PREFIX

import numpy as np

"""
This file is used for test robot_pose_jacobian_server, it is not called in launch file.
"""

link_names = ['right_l2', 'right_l3', 'right_l4', 'right_l5', 'right_l6', 'right_hand']

def unpack_pose_jacobian_dict(poses, jacobians):
    pose_jacobian_dict = {}
    pose_counter = jac_counter = 0
    for link in link_names:
        pose = poses[pose_counter:pose_counter + 3]
        jacobian = np.array([
            jacobians[jac_counter + 3:jac_counter + 10],
            jacobians[jac_counter + 10:jac_counter + 17],
            jacobians[jac_counter + 17:jac_counter + 24],
        ])
        pose_counter += 3
        jac_counter += 21

        pose_jacobian_dict[link] = [pose, jacobian]
    return pose_jacobian_dict


def request_robot_pose_jacobian_server(name):
    server_name = PREFIX + 'arm_pose_jacobian'
    rospy.wait_for_service(server_name)
    try:
        request = rospy.ServiceProxy(server_name, type_arm_pose_and_jacobian, persistent=True)
        response = request(name)
        return unpack_pose_jacobian_dict(response.poses, response.jacobians)
    except rospy.ServiceException as e:
        print(e)

if __name__ == "__main__":
    pose_jacobian_dict = request_robot_pose_jacobian_server('right')
    print(pose_jacobian_dict)
