#!/usr/bin/env python
import rospy

from sawyer_control.controllers.impedance_controller import ImpedanceController
from sawyer_control.srv import type_angle_action, type_angle_actionResponse
from sawyer_control import PREFIX

import intera_interface as ii


def handle_execute_action(request):
    # Get name of joints
    joint_names = arm.joint_names()

    action = request.angles
    duration = request.duration

    joint_to_values = dict(zip(joint_names, action))

    joint_angles = [joint_to_values[name] for name in arm.joint_names()]
    controller.move_with_impedance([joint_angles], duration=duration)
    return type_angle_actionResponse(True)


def angle_action_server():
    node_name = PREFIX + 'angle_action_server'
    server_name = PREFIX + 'angle_action'
    rospy.init_node(node_name)

    global arm
    global controller
    arm = ii.Limb('right')
    arm.set_joint_position_speed(0.1)

    controller = ImpedanceController(control_rate=1000)

    server = rospy.Service(server_name, type_angle_action, handle_execute_action)
    rospy.spin()


if __name__ == '__main__':
    angle_action_server()
