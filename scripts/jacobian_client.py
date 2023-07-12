#!/usr/bin/env python

import rospy

from sawyer_control.srv import type_arm_pose_and_jacobian
from sawyer_control import PREFIX

from collections import OrderedDict
import numpy as np

from sawyer_control.config.default_config import LINK_NAMES


"""
This file is used for test robot_pose_jacobian_server, it is not called in launch file.
"""

def unpack_pose_jacobian_dict(poses, jacobians):
    pose_jacobian_dict = OrderedDict()
    pose_counter = jac_counter = 0
    for link in LINK_NAMES:
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


def request_robot_pose_jacobian_server():
    server_name = PREFIX + 'arm_pose_jacobian'
    rospy.wait_for_service(server_name)
    try:
        request = rospy.ServiceProxy(server_name, type_arm_pose_and_jacobian, persistent=True)
        response = request()
        return unpack_pose_jacobian_dict(response.poses, response.jacobians)
    except rospy.ServiceException as e:
        print(e)

if __name__ == "__main__":
    pose_jacobian_dict = request_robot_pose_jacobian_server()

    for k, v in pose_jacobian_dict.items():
        print("{}: {}".format(k, v))
