#!/usr/bin/env python
import rospy
import intera_interface

import copy

from sawyer_control.srv import type_observation, type_observationResponse
from sawyer_control import PREFIX


# Note: do not remove input arg "request", it is required for ROS
def handle_get_observation(request):
    # Get name of joints: expect ['right_j0', 'right_j1', 'right_j2', 'right_j3', 'right_j4', 'right_j5', 'right_j6']
    joint_names = arm.joint_names()

    # Get joint angles
    joint_angles_dict = arm.joint_angles()  # unordered dict of joint name Keys to angle (rad) Values
    joint_angles = [joint_angles_dict[joint] for joint in joint_names]

    # Get joint velocities
    joint_velocities_dict = arm.joint_velocities()  # unordered dict of joint name Keys to velocity (rad/s) Values
    joint_velocities = [joint_velocities_dict[joint] for joint in joint_names]

    # Get joint torques
    joint_torques_dict = arm.joint_efforts()  # unordered dict of joint name Keys to effort (Nm) Values
    joint_torques = [joint_torques_dict[joint] for joint in joint_names]

    # *** Important ***: There are 3 supported tips: ['right_hand', 'right_gripper_tip', 'right_hand_camera']
    # We only consider pose and velocity of a tip named "right_gripper_tip"
    endpoint_state = arm.tip_state('right_gripper_tip')

    # Get end-effector geometry
    endpoint_pose = copy.copy(endpoint_state.pose)  # Cartesian pose {position (Point msg), orientation (Quaternion msg)}
    endpoint_geometry = [
        endpoint_pose.position.x, endpoint_pose.position.y, endpoint_pose.position.z,
        endpoint_pose.orientation.x, endpoint_pose.orientation.y, endpoint_pose.orientation.z, endpoint_pose.orientation.w]

    # Get end-effector velocity
    endpoint_twist = copy.copy(endpoint_state.twist)  # Cartesian twist {linear (Point msg), angular (Point msg)}
    endpoint_vel = [
        endpoint_twist.linear.x, endpoint_twist.linear.y, endpoint_twist.linear.z,
        endpoint_twist.angular.x, endpoint_twist.angular.y, endpoint_twist.angular.z]

    return type_observationResponse(joint_angles, joint_velocities, joint_torques, endpoint_geometry, endpoint_vel)


def observation_server():
    node_name = PREFIX + 'observation_server'
    server_name = PREFIX + 'observation'
    rospy.init_node(node_name)

    global arm
    arm = intera_interface.Limb('right')

    server = rospy.Service(server_name, type_observation, handle_get_observation)
    rospy.spin()


if __name__ == "__main__":
    observation_server()
