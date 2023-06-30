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
    angles_dict = arm.joint_angles()            # unordered dict of joint name Keys to angle (rad) Values
    angles = [angles_dict[joint] for joint in joint_names]

    # Get joint velocity information
    velocities_dict = arm.joint_velocities()    # unordered dict of joint name Keys to velocity (rad/s) Values
    velocities = [velocities_dict[joint] for joint in joint_names]

    # Get joint torque information
    torques_dict = arm.joint_efforts()          # unordered dict of joint name Keys to effort (Nm) Values
    torques = [torques_dict[joint] for joint in joint_names]

    # Get end effector geometry
    endeffector_pose_dict = arm.endpoint_pose()     # Cartesian endpoint pose {position, orientation}.
    pos = endeffector_pose_dict['position']             # 'position': Cartesian  (x, y, z)
    orientation = endeffector_pose_dict['orientation']  # 'orientation': Quaternion (x, y, z, w)
    endpoint_geometry = [pos.x, pos.y, pos.z,
                         orientation.x, orientation.y, orientation.z, orientation.w]

    # Get end effector velocity
    endeffector_vel_dict = arm.endpoint_velocity()  # Return Cartesian endpoint twist {linear, angular}
    linear = endeffector_vel_dict['linear']         # Cartesian velocity in x,y,z directions
    angular = endeffector_vel_dict['angular']       # Angular velocity around x,y,z axes
    endpoint_vel = [linear.x, linear.y, linear.z, angular.x, angular.y, angular.z]

    return observationResponse(angles, velocities, torques, endpoint_geometry, endpoint_vel)


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
