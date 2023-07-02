#!/usr/bin/env python
import rospy

from sawyer_control.srv import ik, ikResponse
from sawyer_control import PREFIX
from sawyer_control.pd_controllers.inverse_kinematics import get_pose_stamped, get_joint_angles
from sawyer_control.configs import ros_config

import intera_interface as ii


def handle_compute_joint_angle(request):
    # Get name of joints
    joint_names = arm.joint_names()

    ee_pos_x, ee_pos_y, ee_pos_z = request.ee_pos[0], request.ee_pos[1], request.ee_pos[2]
    ee_ori_x, ee_ori_y, ee_ori_z, ee_ori_w = request.ee_pos[3], request.ee_pos[4], request.ee_pos[5], request.ee_pos[6]
    pose = get_pose_stamped(ee_pos_x, ee_pos_y, ee_pos_z, ee_ori_x, ee_ori_y, ee_ori_z, ee_ori_w)

    reset_angles = ros_config.RESET_ANGLES
    reset_angles = dict(zip(joint_names, reset_angles))
    ik_angles = get_joint_angles(pose, reset_angles, True, False)
    ik_angles = [ik_angles[joint] for joint in joint_names]
    return ikResponse(ik_angles)


def inverse_kinematics_server():
    node_name = PREFIX + 'ik_server'
    server_name = PREFIX + 'ik'
    rospy.init_node(node_name, anonymous=True)

    global arm
    arm = ii.Limb('right')

    server = rospy.Service(server_name, ik, handle_compute_joint_angle)
    rospy.spin()


if __name__ == "__main__":
    inverse_kinematics_server()
