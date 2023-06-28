#!/usr/bin/env python

import rospy

from sawyer_control.srv import robot_pose_and_jacobian
from sawyer_control import PREFIX

import numpy as np

"""
This file is used for test robot_pose_jacobian_server, it is not called in launch file.
"""

joint_names = [
    '_l2', '_l3', '_l4', '_l5', '_l6', '_hand'
]

def extract_pose_jacobian(poses, jacobians):
    pose_jacobian_dict = {}
    idx_joint = pose_counter = jac_counter = 0
    for i in range(len(joint_names)):
        pose = poses[pose_counter:pose_counter+3]
        jacobian = np.array([
            jacobians[jac_counter + 3:jac_counter + 10],
            jacobians[jac_counter + 10:jac_counter + 17],
            jacobians[jac_counter + 17:jac_counter+ 24],
        ])
        pose_counter += 3
        jac_counter += 21

        pose_jacobian_dict['right' + joint_names[idx_joint]] = [pose, jacobian]
        idx_joint += 1
    return pose_jacobian_dict


def request_robot_pose_jacobian_server(name):
    server_name = PREFIX + 'robot_pose_jacobian'
    rospy.wait_for_service(server_name)
    try:
        request = rospy.ServiceProxy(server_name, robot_pose_and_jacobian, persistent=True)
        response = request(name)
        return extract_pose_jacobian(response.poses, response.jacobians)
    except rospy.ServiceException as e:
        print(e)

if __name__ == "__main__":
    pose_jacobian_dict = request_robot_pose_jacobian_server('right')
    print(pose_jacobian_dict)
