#!/usr/bin/env python
import rospy

import intera_interface

from sawyer_control.srv import type_ik, type_ikResponse
from sawyer_control import PREFIX
from sawyer_control.controllers.inverse_kinematics import get_pose_stamped, get_joint_angles, joint_state_from_cmd



def handle_compute_joint_angle(request):
    # Get name of joints: expect ['right_j0', 'right_j1', 'right_j2', 'right_j3', 'right_j4', 'right_j5', 'right_j6']
    joint_names = arm.joint_names()

    tip_name = request.tip_name

    ee_pos_x, ee_pos_y, ee_pos_z = request.geometry[0], request.geometry[1], request.geometry[2]
    ee_ori_x, ee_ori_y, ee_ori_z, ee_ori_w = request.geometry[3], request.geometry[4], request.geometry[5], request.geometry[6]

    pose = get_pose_stamped(ee_pos_x, ee_pos_y, ee_pos_z, ee_ori_x, ee_ori_y, ee_ori_z, ee_ori_w)

    seed_angles = request.current_joint_angles
    seed_angles = dict(zip(joint_names, seed_angles))
    seed_cmd = joint_state_from_cmd(seed_angles)

    ik_angles = get_joint_angles(pose, seed_cmd, True, False, tip_name)
    ik_angles = [ik_angles[joint] for joint in joint_names]
    return type_ikResponse(ik_angles)


def inverse_kinematics_server():
    node_name = PREFIX + 'ik_server'
    server_name = PREFIX + 'ik'
    rospy.init_node(node_name)

    global arm
    arm = intera_interface.Limb('right')

    server = rospy.Service(server_name, type_ik, handle_compute_joint_angle)
    rospy.spin()


if __name__ == "__main__":
    inverse_kinematics_server()
