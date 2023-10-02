#!/usr/bin/env python
import rospy

from sawyer_control.srv import type_object_pose
from sawyer_control import PREFIX

import numpy as np

"""
This file is used for test workspace_state_server, it is not called in launch file.
"""
def request_workspace_observation_server():
    server_name = PREFIX + 'object_state_server'
    rospy.wait_for_service(server_name)
    try:
        request = rospy.ServiceProxy(server_name, type_object_pose, persistent=True)
        response = request()  # Structure of message in srv/type_object_pose.srv
        return (
            np.array(response.pose_table_corner0),
            np.array(response.pose_table_corner1),
            np.array(response.pose_table_corner2),
            np.array(response.pose_table_corner3),
            np.array(response.pose_leg_black),
            np.array(response.pose_leg_blue),
            np.array(response.pose_leg_white),
            np.array(response.pose_leg_pink),
        )
    except rospy.ServiceException as e:
        print(e)



if __name__ == "__main__":
    # ======= Example of client: get state of joints =======
    tip_name = "right_hand"
    # tip_name = "right_gripper_tip"
    # tip_name = "right_hand_camera"
    corner0, corner1, corner2, corner3, leg_black, leg_blue, leg_white, leg_pink = request_workspace_observation_server()
    print("corner 0: pos={}, orientation={}".format(corner0[:3], corner0[3:]))
    print("corner 1: pos={}, orientation={}".format(corner1[:3], corner1[3:]))
    print("corner 2: pos={}, orientation={}".format(corner2[:3], corner2[3:]))
    print("corner 3: pos={}, orientation={}".format(corner3[:3], corner3[3:]))

    print("leg black: pos={}, orientation={}".format(leg_black[:3], leg_black[3:]))
    print("leg blue: pos={}, orientation={}".format(leg_blue[:3], leg_blue[3:]))
    print("leg white: pos={}, orientation={}".format(leg_white[:3], leg_white[3:]))
    print("leg pink: pos={}, orientation={}".format(leg_pink[:3], leg_pink[3:]))
