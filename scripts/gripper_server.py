#!/usr/bin/env python
import rospy

from sawyer_control.srv import type_gripper, type_gripperResponse
from sawyer_control import PREFIX

import intera_interface as ii


"""
This file creates server to get the gripper's status.
Reference: https://support.rethinkrobotics.com/support/solutions/articles/80000980355-gripper-example
"""

def handle_get_gripper_info(request):
    # Get current position value in meters (m)
    gripper_position = sawyer_gripper.get_position()

    # Get the velocity of gripper in meters per second (m/s)
    gripper_velocity = sawyer_gripper.get_cmd_velocity()

    # Get the force sensed by the gripper in estimated Newtons (Current force value in Newton-Meters (N-m))
    gripper_force = sawyer_gripper.get_force()

    return type_gripperResponse(gripper_position, gripper_velocity, gripper_force)


def gripper_server():
    node_name = PREFIX + 'gripper_server'
    server_name = PREFIX + 'gripper'
    rospy.init_node(node_name)

    global sawyer_gripper
    sawyer_gripper = ii.Gripper('right_gripper')

    server = rospy.Service(server_name, type_gripper, handle_get_gripper_info)
    rospy.spin()


if __name__ == "__main__":
    gripper_server()
