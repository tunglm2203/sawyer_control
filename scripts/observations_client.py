#!/usr/bin/env python
import rospy

from sawyer_control.srv import observation
from sawyer_control import PREFIX

import numpy as np

"""
This file is used for test observation_server, it is not called in launch file.
"""
def request_observation_server():
    server_name = PREFIX + 'observation'
    rospy.wait_for_service(server_name)
    try:
        request = rospy.ServiceProxy(server_name, observation, persistent=True)
        response = request()  # Structure of message in srv/observation.srv
        return (
            np.array(response.angles),
            np.array(response.velocities),
            np.array(response.torques),
            np.array(response.endpoint_geom),
            np.array(response.endpoint_vel),
        )
    except rospy.ServiceException as e:
        print(e)


if __name__ == "__main__":
    angle, vel, torque, ee_pose, ee_vel = request_observation_server()
    print("Joint angle (dim={}): {}".format(angle.shape, angle))
    print("Joint velocity (dim={}): {}".format(vel.shape, vel))
    print("Joint torque (dim={}): {}".format(torque.shape, torque))
    print("EE pose (dim={}): {}".format(ee_pose.shape, ee_pose))
    print("EE vel (dim={}): {}".format(ee_vel.shape, ee_vel))
