#!/usr/bin/env python
import rospy

from sawyer_control.srv import type_observation_allinone
from sawyer_control import PREFIX

import numpy as np

"""
This file is used for test observation_server, it is not called in launch file.
"""
def request_allinone_observation_server(tip_name="right_gripper_tip"):
    server_name = PREFIX + 'arm_state_allinone_server'
    rospy.wait_for_service(server_name)
    try:
        request = rospy.ServiceProxy(server_name, type_observation_allinone, persistent=True)
        response = request(tip_name)  # Structure of message in srv/observation.srv
        return (
            np.array(response.joint_angles),
            np.array(response.joint_velocities),
            np.array(response.joint_torques),
            np.array(response.endpoint_geometry),
            np.array(response.endpoint_velocity),

            np.array(response.gripper_position),
            np.array(response.gripper_velocity),
            np.array(response.gripper_force),

            np.array(response.image)
        )
    except rospy.ServiceException as e:
        print(e)


if __name__ == "__main__":
    # ======= Example of client: get state of joints =======
    tip_name = "right_hand"
    # tip_name = "right_gripper_tip"
    # tip_name = "right_hand_camera"
    angle, vel, torque, ee_pose, ee_vel, gripper_pos, gripper_vel, gripper_force, image = (
        request_allinone_observation_server(tip_name))

    print("Joint angle (dim={}): {}".format(angle.shape, angle))
    print("Joint velocity (dim={}): {}".format(vel.shape, vel))
    print("Joint torque (dim={}): {}".format(torque.shape, torque))
    print("EE pose (dim={}): {}".format(ee_pose.shape, ee_pose))
    print("EE vel (dim={}): {}".format(ee_vel.shape, ee_vel))

    print("Gripper pos (dim={}): {}".format(gripper_pos.shape, gripper_pos))
    print("Gripper vel (dim={}): {}".format(gripper_vel.shape, gripper_vel))
    print("Gripper force (dim={}): {}".format(gripper_force.shape, gripper_force))

    print("Image (dim={}): {}".format(image.shape, image))
