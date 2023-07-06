#!/usr/bin/env python
import rospy

from sawyer_control.srv import type_angle_action
from sawyer_control import PREFIX



def request_angle_action_server(angles, duration):
    server_name = PREFIX + 'angle_action'
    rospy.wait_for_service(server_name)
    try:
        execute_action = rospy.ServiceProxy(server_name, type_angle_action, persistent=True)
        execute_action(angles, duration)
    except rospy.ServiceException as e:
        print(e)


if __name__ == "__main__":
    # angles = [0.24974225425226723, -0.4705250734087791, -0.6125282007590721, 1.371568814985761,
    #           0.7130258342179211, 0.9049490935382062, 1.1875654704177894]
    angles = [0.20683294539879993, -0.7932586163385188, -0.7563392179080931, 1.5870541415105552, 0.6044525943755639,
              1.0247266578334868, 1.0120021901793528]
    duration = 1.0
    request_angle_action_server(angles, duration)
