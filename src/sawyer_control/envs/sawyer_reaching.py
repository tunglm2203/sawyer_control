from collections import OrderedDict
import numpy as np
from gym.spaces import Box
from sawyer_control.envs.sawyer_env_base import SawyerEnvBase
from sawyer_control.core.serializable import Serializable
from sawyer_control.core.eval_util import get_stat_in_paths, \
    create_stats_ordered_dict

class SawyerReachXYZEnv(SawyerEnvBase):
    def __init__(self,
                 fixed_goal=(1, 1, 1),
                 indicator_threshold=.05,
                 reward_type='hand_distance',
                 goal_low=None,
                 goal_high=None,
                 **kwargs
                 ):
        Serializable.quick_init(self, locals())
        SawyerEnvBase.__init__(self, **kwargs)
        if self.action_mode == 'torque':
            if goal_low is None:
                goal_low = self.config.TORQUE_SAFETY_BOX.low
            if goal_high is None:
                goal_high = self.config.TORQUE_SAFETY_BOX.high
        else:
            if goal_low is None:
                goal_low = self.config.POSITION_SAFETY_BOX.low
            if goal_high is None:
                goal_high = self.config.POSITION_SAFETY_BOX.high
        self.goal_space = Box(goal_low, goal_high, dtype=np.float32)
        self.indicator_threshold=indicator_threshold
        self.reward_type = reward_type
        self._state_goal = np.array(fixed_goal)

    def compute_rewards(self, actions, obs, goals):
        distances = np.linalg.norm(obs - goals, axis=1)
        if self.reward_type == 'hand_distance':
            r = -distances
        elif self.reward_type == 'hand_success':
            r = -(distances < self.indicator_threshold).astype(float)
        else:
            raise NotImplementedError("Invalid/no reward type.")
        return r

    def _get_info(self):
        hand_distance = np.linalg.norm(self._state_goal - self._get_endeffector_pose())
        return dict(
            hand_distance=hand_distance,
            hand_success=(hand_distance<self.indicator_threshold).astype(float)
        )

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        for stat_name in [
            'hand_distance',
            'hand_success',
        ]:
            stat_name = stat_name
            stat = get_stat_in_paths(paths, 'env_infos', stat_name)
            statistics.update(create_stats_ordered_dict(
                '%s%s' % (prefix, stat_name),
                stat,
                always_show_all_stats=True,
                ))
            statistics.update(create_stats_ordered_dict(
                'Final %s%s' % (prefix, stat_name),
                [s[-1] for s in stat],
                always_show_all_stats=True,
                ))
        return statistics

    """
    Multitask functions
    """

    @property
    def goal_dim(self):
        return 3

    def convert_obs_to_goals(self, obs):
        return obs[:, -3:]

    def set_to_goal(self, goal):
        self._position_act(goal - self._get_endeffector_pose()[:3])

