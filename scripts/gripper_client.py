#!/usr/bin/env python
import rospy

from sawyer_control.srv import type_gripper
from sawyer_control import PREFIX



"""
This file is used to test gripper_server to get the gripper's status.
"""

def request_gripper_server():
    server_name = PREFIX + 'gripper'
    rospy.wait_for_service(server_name)
    try:
        request = rospy.ServiceProxy(server_name, type_gripper, persistent=True)
        response = request()
        return (response.position, response.velocity, response.force)
    except rospy.ServiceException as e:
        print(e)


if __name__ == "__main__":
    pos, vel, force = request_gripper_server()
    print("pos: ", pos)
    print("vel: ", vel)
    print("force: ", force)
