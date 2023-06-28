#!/usr/bin/env python
import rospy

from sawyer_control.srv import observation, observationResponse
from sawyer_control import PREFIX

import intera_interface as ii


# Note: do not remove input arg "request", it is required for ROS
def handle_get_observation(request):
    # Get name of joints
    joint_names = arm.joint_names()

    # Get joint angle information
    angles_dict = arm.joint_angles()
    angles = [angles_dict[joint] for joint in joint_names]

    # Get joint velocity information
    velocities_dict = arm.joint_velocities()
    velocities = [velocities_dict[joint] for joint in joint_names]

    # Get joint torque information
    torques_dict = arm.joint_efforts()
    torques = [torques_dict[joint] for joint in joint_names]

    # Get end effector information
    end_effector_dict = arm.endpoint_pose()
    pos = end_effector_dict['position']
    orientation = end_effector_dict['orientation']
    endpoint_pose = [
        pos.x, pos.y, pos.z,
        orientation.x, orientation.y, orientation.z, orientation.w
    ]
    return observationResponse(angles, velocities, torques, endpoint_pose)


def observation_server():
    node_name = PREFIX + 'observation_server'
    server_name = PREFIX + 'observation'
    rospy.init_node(node_name, anonymous=True)

    global arm
    arm = ii.Limb('right')

    server = rospy.Service(server_name, observation, handle_get_observation)
    rospy.spin()


if __name__ == "__main__":
    observation_server()
