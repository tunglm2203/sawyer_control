#!/usr/bin/env python
import rospy

from sawyer_control.srv import type_observation, type_gripper
from sawyer_control import PREFIX

import numpy as np

"""
This file is used for test observation_server, it is not called in launch file.
"""
def request_observation_server(tip_name="right_gripper_tip"):
    server_name = PREFIX + 'arm_joint_state_server'
    rospy.wait_for_service(server_name)
    try:
        request = rospy.ServiceProxy(server_name, type_observation, persistent=True)
        response = request(tip_name)  # Structure of message in srv/observation.srv
        return (
            np.array(response.joint_angles),
            np.array(response.joint_velocities),
            np.array(response.joint_torques),
            np.array(response.endpoint_geometry),
            np.array(response.endpoint_velocity),
        )
    except rospy.ServiceException as e:
        print(e)


def request_gripper_server():
    server_name = PREFIX + 'arm_gripper_state_server'
    rospy.wait_for_service(server_name)
    try:
        request = rospy.ServiceProxy(server_name, type_gripper, persistent=True)
        response = request()
        return (response.position, response.velocity, response.force)
    except rospy.ServiceException as e:
        print(e)


if __name__ == "__main__":
    # ======= Example of client: get state of joints =======
    tip_name = "right_hand"
    # tip_name = "right_gripper_tip"
    # tip_name = "right_hand_camera"
    angle, vel, torque, ee_pose, ee_vel = request_observation_server(tip_name)
    print("Joint angle (dim={}): {}".format(angle.shape, angle))
    print("Joint velocity (dim={}): {}".format(vel.shape, vel))
    print("Joint torque (dim={}): {}".format(torque.shape, torque))
    print("EE pose (dim={}): {}".format(ee_pose.shape, ee_pose))
    print("EE vel (dim={}): {}".format(ee_vel.shape, ee_vel))

    # ======= Example of client: get state of gripper =======
    pos, vel, force = request_gripper_server()
    print("gripper pos: ", pos)
    print("gripper vel: ", vel)
    print("gripper force: ", force)