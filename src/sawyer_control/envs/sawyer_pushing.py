from collections import OrderedDict
import numpy as np
from gym.spaces import Box
from sawyer_control.envs.sawyer_env_base import SawyerEnvBase
from sawyer_control.core.serializable import Serializable
import rospy
import rospkg
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.srv import GetModelState


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
                 puck_goal_low=None,
                 puck_goal_high=None,
                 hand_goal_low=None,
                 hand_goal_high=None,
                 random_init=False,
                 use_gazebo=True,
                 use_gazebo_auto=False,
                 **kwargs
                 ):
        Serializable.quick_init(self, locals())
        SawyerEnvBase.__init__(self, action_mode=action_mode, **kwargs)
        self.use_gazebo_auto = use_gazebo_auto
        if hand_goal_low is None:
            hand_goal_low = self.config.POSITION_SAFETY_BOX.low[:2]
        if hand_goal_high is None:
            hand_goal_high = self.config.POSITION_SAFETY_BOX.high[:2]
        if puck_goal_low is None:
            puck_goal_low = hand_goal_low
        if hand_goal_high is None:
            puck_goal_high = hand_goal_high

        self.goal_space = Box(np.concatenate((puck_goal_low, hand_goal_low)),
                              np.concatenate((puck_goal_high, hand_goal_high)),
                              dtype=np.float64)
        self._state_goal = None
        self.pause_on_reset = pause_on_reset
        self.fixed_goal = np.array(fixed_goal)

        self.pos_object_reset_position = self.config.OBJ_RESET_POS.copy()

        self.random_init = random_init
        self.use_gazebo = use_gazebo
        self.safe_pos_to_move_to_goal = self.config.POSITION_SAFETY_BOX_LOWS.copy()
        self.safe_pos_to_move_to_goal[2] = self.config.POSITION_SAFETY_BOX_HIGHS[2]

    @property
    def goal_dim(self):
        return 4  # xy for object position, xy for end effector position

    def set_to_goal(self, goal):
        print('moving arm to desired object goal')
        ee_reset_high = np.concatenate((self.pos_control_reset_position[:2],
                                        [self.config.POSITION_SAFETY_BOX_HIGHS[2]]))
        obj_goal_high = np.concatenate((goal[:2], [self.config.POSITION_SAFETY_BOX_HIGHS[2]]))
        ee_goal_high = np.concatenate((goal[2:], [self.config.POSITION_SAFETY_BOX_HIGHS[2]]))
        ee_goal = np.concatenate((goal[2:], [self.config.POSITION_SAFETY_BOX_LOWS[2]]))
        self._position_act(ee_reset_high - self._get_endeffector_pose()[:3])
        self._position_act(obj_goal_high - self._get_endeffector_pose()[:3])
        if self.use_gazebo_auto and self.use_gazebo:
            print(bcolors.OKGREEN + 'place object at end effector location and press enter' +
                  bcolors.ENDC)
            obj_pos = [goal[0], goal[1], self.pos_object_reset_position[2]]
            self.set_obj_to_pos_in_gazebo(self.config.OBJECT_NAME, obj_pos)
        else:
            input(bcolors.OKGREEN + 'place object at end effector location and press enter' +
                  bcolors.ENDC)

        # This step to move to safe position before moving to goal. It helps to avoid the situation
        # that EE collide with object when ee's goal pos and object's goal position in same
        # coordinate.
        self._position_act(ee_goal_high - self._get_endeffector_pose()[:3])
        self._position_act(ee_goal - self._get_endeffector_pose()[:3])

    def _reset_robot(self):
        self.in_reset = True
        self._safe_move_to_neutral()
        self.in_reset = False
        if self.pause_on_reset:
            if self.use_gazebo_auto and self.use_gazebo:
                print(
                    bcolors.OKBLUE + 'move object to reset position and press enter' + bcolors.ENDC)
                if self.random_init:
                    while True:
                        obj_pos_rand = np.random.uniform(
                            self.goal_space.low,
                            self.goal_space.high,
                            size=(1, self.goal_space.low.size),
                        )
                        dis_obj_vs_ee = np.linalg.norm(self.pos_control_reset_position[:2] -
                                                       obj_pos_rand[0, :2])
                        if dis_obj_vs_ee > self.config.OBJECT_RADIUS:
                            break
                    self.pos_object_reset_position[:2] = obj_pos_rand[0][:2]

                obj_pos = [self.pos_object_reset_position[0],
                           self.pos_object_reset_position[1],
                           self.pos_object_reset_position[2]]
                self.set_obj_to_pos_in_gazebo(self.config.OBJECT_NAME, obj_pos)
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

    def _get_info(self):
        """
        This function helps to obtain information in gazebo environment such as object's position,
        distance from hand to hand's target, object to object's target.
        Author: Tung
        :return: The dictionary contains hand's, object's, touch's distance, and success flag
        """
        if self.use_gazebo:
            goal = self.get_goal()['state_desired_goal']
            puck_goal_pos, hand_goal_pos = goal[:2], goal[2:]
            hand_distance = np.linalg.norm(
                hand_goal_pos - self._get_endeffector_pose()[:2]
            )
            puck_distance = np.linalg.norm(
                puck_goal_pos - self.get_obj_pos_in_gazebo(object_name=self.config.OBJECT_NAME)[:2])
            touch_distance = np.linalg.norm(
                self._get_endeffector_pose()[:2] -
                self.get_obj_pos_in_gazebo(object_name=self.config.OBJECT_NAME)[:2])
            info = dict(
                hand_distance=hand_distance,
                puck_distance=puck_distance,
                touch_distance=touch_distance,
                success=float(hand_distance + puck_distance < 0.06),
            )
        else:
            info = dict()
        return info

    def get_obj_pos_in_gazebo(self, object_name):
        """
        This function is only used with gazebo to get object's position.
        Author: Tung
        Reference:
            http://gazebosim.org/tutorials/?tut=ros_comm
            https://answers.ros.org/question/261782/how-to-use-getmodelstate-service-from-gazebo-in-python/
        """
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            obj_info = model_coordinates(object_name, '')
            obj_pos = np.array([obj_info.pose.position.x,
                                obj_info.pose.position.y,
                                obj_info.pose.position.z])
            return obj_pos

        except rospy.ServiceException as e:
            print("Get Model State service call failed: %s" % e)
            rospy.loginfo("Get Model State service call failed:  {0}".format(e))

    def set_obj_to_pos_in_gazebo(self, object_name, object_pos):
        """
        This function is only used with gazebo to set object's position.
        Author: Tung
        Reference:
            http://gazebosim.org/tutorials/?tut=ros_comm
            http://answers.gazebosim.org/question/22125/how-to-set-a-models-position-using-gazeboset_model_state-service-in-python/
        """
        state_msg = ModelState()
        state_msg.model_name = object_name
        state_msg.pose.position.x = float(object_pos[0])
        state_msg.pose.position.y = float(object_pos[1])
        state_msg.pose.position.z = float(object_pos[2])
        state_msg.pose.orientation.x = 0.0  # pos[3]
        state_msg.pose.orientation.y = 1.0  # pos[4]
        state_msg.pose.orientation.z = 0.0  # pos[5]
        state_msg.pose.orientation.w = 0.0  # pos[6]
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state(state_msg)
        except rospy.ServiceException as e:
            print("Set Model State service call failed: %s" % e)
            rospy.loginfo("Set Model State service call failed:  {0}".format(e))
