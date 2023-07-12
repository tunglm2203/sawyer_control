import rospy
from std_msgs.msg import Empty

import numpy as np

from sawyer_control.config import default_config

class AnglePDController(object):
    """
    PD Controller for Moving to Neutral
    """
    def __init__(self, config=default_config):
        # control parameters
        self._rate = 1000  # Hz
        self._missed_cmds = 20.0  # Missed cycles before triggering timeout

        # initialize parameters
        self._springs = dict()
        self._damping = dict()
        self._des_angles = dict()

        # create cuff disable publisher
        cuff_ns = 'robot/limb/right/suppress_cuff_interaction'
        self._pub_cuff_disable = rospy.Publisher(cuff_ns, Empty, queue_size=1)

        self.joint_names = config.JOINT_NAMES
        self._des_angles = dict(zip(self.joint_names, config.INITIAL_JOINT_ANGLES))

        self.max_stiffness = 20
        self.time_to_maxstiffness = .3
        self.t_release = rospy.get_time()

        self._imp_ctrl_is_active = True

        default_spring = (10.0, 15.0, 5.0, 5.0, 3.0, 2.0, 1.5)
        # default_damping = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
        default_damping = (0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2)

        for i, joint in enumerate(self.joint_names):
            self._springs[joint] = default_spring[i]
            self._damping[joint] = default_damping[i]

    def adjust_springs(self):
        for i, joint in enumerate(list(self._des_angles.keys())):
            t_delta = rospy.get_time() - self.t_release
            print("Joint {}: time= {}".format(joint, t_delta))
            if t_delta > 0:
                if t_delta < self.time_to_maxstiffness:
                    self._springs[joint] = t_delta/self.time_to_maxstiffness * self.max_stiffness
                else:
                    self._springs[joint] = self.max_stiffness
            else:
                print("warning t_delta smaller than zero!")

    def _compute_pd_forces(self, current_joint_angles, current_joint_velocities):
        """
        Computes the required torque to be applied using the sawyer's current joint angles and joint velocities
        """
        # self.adjust_springs()

        # disable cuff interaction
        if self._imp_ctrl_is_active:
            self._pub_cuff_disable.publish()

        # create our command dict
        cmd = dict()

        # calculate current forces
        for idx, joint in enumerate(self.joint_names):
            # spring portion
            cmd[joint] = self._springs[joint] * (self._des_angles[joint] - current_joint_angles[idx])
            # damping portion
            cmd[joint] -= self._damping[joint] * current_joint_velocities[idx]

        cmd = np.array([cmd[joint] for joint in self.joint_names])
        return cmd
