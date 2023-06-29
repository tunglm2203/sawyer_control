#!/usr/bin/env python

import rospy

from sawyer_control.srv import robot_pose_and_jacobian, robot_pose_and_jacobianResponse
from sawyer_control import PREFIX

import numpy as np
import intera_interface as ii
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_kinematics import KDLKinematics

link_names = [
    '_l2', '_l3', '_l4', '_l5', '_l6', '_hand'
]


def handle_get_robot_pose_jacobian(request):
    poses = []
    jacobians = []
    # Get joint angle information
    angles_dict = arm.joint_angles()
    angles = [
        angles_dict[request.name + '_j0'], angles_dict[request.name + '_j1'], angles_dict[request.name + '_j2'],
        angles_dict[request.name + '_j3'], angles_dict[request.name + '_j4'], angles_dict[request.name + '_j5'],
        angles_dict[request.name + '_j6']
    ]
    for joint in link_names:
        # Compute pose for each joint
        joint = request.name + joint
        pose = kin.forward(angles, joint)
        pose = np.squeeze(np.asarray(pose))
        pose = [pose[0][3], pose[1][3], pose[2][3]]

        # Compute velocity for each joint
        poses.extend(pose)
        jacobian = kin.jacobian(angles, pose).getA()
        for i in range(3):
            jacobians.extend(jacobian[i])
    return robot_pose_and_jacobianResponse(poses, jacobians)


def robot_pose_jacobian_server():
    node_name = PREFIX + 'robot_pose_jacobian_server'
    server_name = PREFIX + 'robot_pose_jacobian'
    rospy.init_node(node_name, anonymous=True)
    robot = URDF.from_parameter_server(key='robot_description')

    global kin
    global arm

    kin = KDLKinematics(robot, 'base', 'right_hand')
    arm = ii.Limb('right')

    server = rospy.Service(server_name, robot_pose_and_jacobian, handle_get_robot_pose_jacobian)
    rospy.spin()


if __name__ == "__main__":
    robot_pose_jacobian_server()
