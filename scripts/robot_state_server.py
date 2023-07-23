#!/usr/bin/env python
import rospy
import intera_interface

import copy
import time

from sawyer_control.srv import (
    type_observation, type_observationResponse,
    type_gripper, type_gripperResponse,
)
from sawyer_control import PREFIX
from sawyer_control.config.default_config import DEFAULT_GRIPPER_VELOCITY, CONTROL_FREQ, USE_GRIPPER

from robot.sawyer_robot import SawyerArm


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
    tip_name = request.tip_name
    endpoint_state = arm.tip_state(tip_name)

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


def handle_get_gripper_info(request):
    # Get current position value in meters (m)
    gripper_position = arm.gripper.get_position()

    # Get the velocity of gripper in meters per second (m/s)
    gripper_velocity = arm.gripper.get_cmd_velocity()

    # Get the force sensed by the gripper in estimated Newtons (Current force value in Newton-Meters (N-m))
    gripper_force = arm.gripper.get_force()

    return type_gripperResponse(gripper_position, gripper_velocity, gripper_force)


def observation_server():
    node_name = PREFIX + 'observation_servers'
    arm_joint_state_server_name = PREFIX + 'arm_joint_state_server'
    arm_gripper_state_server = PREFIX + 'arm_gripper_state_server'
    rospy.init_node(node_name)

    global arm
    arm = SawyerArm(
        default_gripper_vel=DEFAULT_GRIPPER_VELOCITY, control_freq=CONTROL_FREQ,
        use_gripper=USE_GRIPPER
    )

    time.sleep(0.1)     # This is important: waiting Limb() to be initialized completely
    # This is used to initialize tips to get their pose in gazebo, not sure why it solved problem that lacks 'right_hand'
    tip_names_to_initialize = ['right_hand', 'right_gripper_tip', 'right_hand_camera']
    for tip_name in tip_names_to_initialize:
        arm.joint_angles_to_cartesian_pose(arm.joint_angles(), end_point=tip_name)

    sv1 = rospy.Service(arm_joint_state_server_name, type_observation, handle_get_observation)
    sv2 = rospy.Service(arm_gripper_state_server, type_gripper, handle_get_gripper_info)

    rospy.spin()


if __name__ == "__main__":
    observation_server()
