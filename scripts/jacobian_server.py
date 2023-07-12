#!/usr/bin/env python

import rospy
import intera_interface

import numpy as np
import copy

from sawyer_control.srv import type_arm_pose_and_jacobian, type_arm_pose_and_jacobianResponse
from sawyer_control import PREFIX

from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_kinematics import KDLKinematics

from sawyer_control.config.default_config import LINK_NAMES



def handle_get_robot_pose_jacobian(request):
    poses = []
    jacobians = []

    # Get name of joints: expect ['right_j0', 'right_j1', 'right_j2', 'right_j3', 'right_j4', 'right_j5', 'right_j6']
    joint_names = arm.joint_names()

    # Get joint angle information
    joint_angles_dict = arm.joint_angles()
    joint_angles = [joint_angles_dict[joint] for joint in joint_names]

    for joint in LINK_NAMES:
        # Compute pose for each joint
        pose = kin.forward(joint_angles, joint)
        pose = np.squeeze(np.asarray(pose))
        pose = [pose[0][3], pose[1][3], pose[2][3]]

        # Compute velocity for each joint
        poses.extend(pose)
        jacobian = kin.jacobian(joint_angles, pose).getA()
        for i in range(3):
            jacobians.extend(jacobian[i])
    return type_arm_pose_and_jacobianResponse(poses, jacobians)


def robot_pose_jacobian_server():
    node_name = PREFIX + 'arm_pose_jacobian_server'
    server_name = PREFIX + 'arm_pose_jacobian'
    rospy.init_node(node_name)
    robot = URDF.from_parameter_server(key='robot_description')

    global kin
    global arm

    kin = KDLKinematics(robot, 'base', 'right_gripper_tip')
    arm = intera_interface.Limb('right')

    server = rospy.Service(server_name, type_arm_pose_and_jacobian, handle_get_robot_pose_jacobian)
    rospy.spin()


if __name__ == "__main__":
    robot_pose_jacobian_server()
