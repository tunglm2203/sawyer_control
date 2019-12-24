from collections import OrderedDict
import numpy as np
from gym.spaces import Box
from sawyer_control.envs.sawyer_env_base import SawyerEnvBase
from sawyer_control.core.serializable import Serializable
from sawyer_control.envs.client_server_utils import ClientProcess
import time


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class SawyerPushXYEnv(SawyerEnvBase):
    ''' Must Wrap with Image Env to use!'''

    def __init__(self,
                 fixed_goal=(0.5, 1, 1, 1),
                 pause_on_reset=True,
                 action_mode='position',
                 z=.06,
                 goal_low=None,
                 goal_high=None,
                 **kwargs
                 ):
        Serializable.quick_init(self, locals())
        SawyerEnvBase.__init__(self, action_mode=action_mode, **kwargs)
        if goal_low is None:
            goal_low = self.config.POSITION_SAFETY_BOX.low[:2]
        if goal_high is None:
            goal_high = self.config.POSITION_SAFETY_BOX.high[:2]
        self.goal_space = Box(np.concatenate((goal_low, goal_low)), np.concatenate((goal_high, goal_high)),
                              dtype=np.float32)
        self._state_goal = None
        self.pause_on_reset = pause_on_reset
        self.fixed_goal = np.array(fixed_goal)

        self.z = z

        self.use_gazebo_auto = True
        if self.use_gazebo_auto:
            self.client = ClientProcess()
        self.pos_object_reset_position = self.config.OBJ_RESET_POS

    @property
    def goal_dim(self):
        return 4  # xy for object position, xy for end effector position

    def set_to_goal(self, goal):
        print('moving arm to desired object goal')
        obj_goal = np.concatenate((goal[:2], [self.z]))
        ee_goal = np.concatenate((goal[2:4], [self.z]))
        self._position_act(obj_goal - self._get_endeffector_pose()[:3])
        if self.use_gazebo_auto:
            print(bcolors.OKGREEN + 'place object at end effector location and press enter' +
                  bcolors.ENDC)
            # TUNG: +- 0.05 to avoid collision since object right below gripper
            args = dict(x=goal[0] + 0.05,
                        y=goal[1] - 0.05,
                        z=self.pos_object_reset_position[2])
            msg = dict(func='set_object_los', args=args)
            self.client.sending(msg, sleep_before=self.config.SLEEP_BEFORE_SENDING_CMD_SOCKET,
                                sleep_after=self.config.SLEEP_BETWEEN_2_CMDS)
            self.client.sending(msg, sleep_before=0,
                                sleep_after=self.config.SLEEP_AFTER_SENDING_CMD_SOCKET)
        else:
            input(bcolors.OKGREEN + 'place object at end effector location and press enter' +
                  bcolors.ENDC)
        self._position_act(ee_goal - self._get_endeffector_pose()[:3])

    def _reset_robot(self):
        self.in_reset = True
        self._safe_move_to_neutral()
        self.in_reset = False
        if self.pause_on_reset:
            if self.use_gazebo_auto:
                print(bcolors.OKBLUE+'move object to reset position and press enter'+bcolors.ENDC)
                args = dict(x=self.pos_object_reset_position[0],
                            y=self.pos_object_reset_position[1],
                            z=self.pos_object_reset_position[2])
                msg = dict(func='set_object_los', args=args)
                self.client.sending(msg, sleep_before=self.config.SLEEP_BEFORE_SENDING_CMD_SOCKET,
                                    sleep_after=self.config.SLEEP_BETWEEN_2_CMDS)
                self.client.sending(msg, sleep_before=0,
                                    sleep_after=self.config.SLEEP_AFTER_SENDING_CMD_SOCKET)
            else:
                input(
                    bcolors.OKBLUE + 'move object to reset position and press enter' + bcolors.ENDC)

    def reset(self):
        self._reset_robot()
        self._state_goal = self.sample_goal()
        return self._get_obs()

    def get_diagnostics(self, paths, prefix=''):
        return OrderedDict()

    def compute_rewards(self, actions, obs, goals):
        raise NotImplementedError('Use Image based reward')
