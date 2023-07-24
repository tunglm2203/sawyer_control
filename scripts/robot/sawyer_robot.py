#!/usr/bin/env python
from intera_interface import Limb, Gripper

import collections
import warnings

from copy import deepcopy

import rospy

from sensor_msgs.msg import (
    JointState
)
from std_msgs.msg import (
    Float64,
    Header
)
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)

import intera_dataflow

from intera_core_msgs.msg import (
    JointCommand,
    EndpointState,
    EndpointStates,
    CollisionDetectionState,
)
from intera_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
    SolvePositionFK,
    SolvePositionFKRequest
)
from intera_interface import settings
from intera_interface.robot_params import RobotParams

from sawyer_control.config.default_config import JOINT_NAMES


class SawyerArm(Limb):
    def __init__(self, limb="right", synchronous_pub=False, raise_on_error=False,
                 use_gripper=False, default_gripper_vel=3.0, control_freq=20):
        super(SawyerArm, self).__init__(limb, synchronous_pub)
        self.raise_on_error = True

        if use_gripper:
            self.gripper = Gripper('right_gripper')
            self.gripper.set_cmd_velocity(default_gripper_vel)
        else:
            self.gripper = None

        self.rate = rospy.Rate(control_freq)

    def set_joint_position_speed(self, speed=0.3):
        """
        Set ratio of max joint speed to use during joint position moves.

        @type speed: float
        @param speed: ratio of maximum joint speed for execution
                      default= 0.3; range= [0.0-1.0]
        """
        self._pub_speed_ratio.publish(Float64(speed))

    def set_joint_positions(self, positions):
        """
        Commands the joints of this limb to the specified positions.

        @type positions: dict({str:float})
        @param positions: joint_name:angle command
        @type raw: bool
        @param raw: advanced, direct position control mode
        """
        self._command_msg.names = positions.keys()
        self._command_msg.position = positions.values()
        self._command_msg.mode = JointCommand.POSITION_MODE
        self._command_msg.header.stamp = rospy.Time.now()
        self._pub_joint_cmd.publish(self._command_msg)

    def set_joint_velocities(self, velocities):
        """
        Commands the joints of this limb to the specified velocities.

        @type velocities: dict({str:float})
        @param velocities: joint_name:velocity command
        """
        self._command_msg.names = velocities.keys()
        self._command_msg.velocity = velocities.values()
        self._command_msg.mode = JointCommand.VELOCITY_MODE
        self._command_msg.header.stamp = rospy.Time.now()
        self._pub_joint_cmd.publish(self._command_msg)

    def set_joint_torques(self, torques):
        """
        Commands the joints of this limb to the specified torques.

        @type torques: dict({str:float})
        @param torques: joint_name:torque command
        """
        self._command_msg.names = torques.keys()
        self._command_msg.effort = torques.values()
        self._command_msg.mode = JointCommand.TORQUE_MODE
        self._command_msg.header.stamp = rospy.Time.now()
        self._pub_joint_cmd.publish(self._command_msg)

    def move_to_joint_positions(self, positions, timeout=15.0,
                                threshold=settings.JOINT_ANGLE_TOLERANCE,
                                test=None):
        """
        (Blocking) Commands the limb to the provided positions.

        @type positions: dict({str:float})
        @param positions: joint_name:angle command
        @type timeout: float
        @param timeout: seconds to wait for move to finish [15]
        @type threshold: float
        @param threshold: position threshold in radians across each joint when
        move is considered successful [0.008726646]
        @param test: optional function returning True if motion must be aborted
        """

        def genf(joint, angle):
            def joint_diff():
                return abs(angle - self._joint_angle[joint])

            return joint_diff

        diffs = [genf(j, a) for j, a in positions.items() if
                 j in self._joint_angle]
        fail_msg = "{0} limb failed to reach commanded joint positions.".format(
            self.name.capitalize())

        def test_collision():
            if self.has_collided():
                rospy.logerr(' '.join(["Collision detected.", fail_msg]))
                return True
            return False

        self.set_joint_positions(positions)
        done = intera_dataflow.wait_for(
            test=lambda: test_collision() or \
                         (callable(test) and test() == True) or \
                         (all(diff() < threshold for diff in diffs)),
            timeout=timeout,
            timeout_msg=fail_msg,
            rate=100,
            raise_on_error=self.raise_on_error,
            body=lambda: self.set_joint_positions(positions)
        )
        return done


if __name__ == "__main__":
    rospy.init_node("test_arm")
    arm = SawyerArm()
    import pdb; pdb.set_trace()

