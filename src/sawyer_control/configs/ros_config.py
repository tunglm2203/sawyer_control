from geometry_msgs.msg import Quaternion
import numpy as np
#JOINT_CONTROLLER_SETTINGS
JOINT_POSITION_SPEED = .1
JOINT_POSITION_TIMEOUT = .2

#JOINT INFO
JOINT_NAMES = ['right_j0',
               'right_j1',
               'right_j2',
               'right_j3',
               'right_j4',
               'right_j5',
               'right_j6'
               ]
LINK_NAMES = ['right_l2', 'right_l3', 'right_l4', 'right_l5', 'right_l6', 'right_hand']
# RESET_ANGLES = np.array(
#     [0.15262207,  0.91435349, -2.02594233,  1.6647979, -2.41721773, 1.14999604, -2.47703505]
# )
RESET_ANGLES = np.array(
    [0.0892115500713, -0.599608568183, -0.450740519289, 2.09783649738, 1.18837727974,
     0.347754006328, 0.559621185782]
)


RESET_DICT = dict(zip(JOINT_NAMES, RESET_ANGLES))
POSITION_CONTROL_EE_ORIENTATION=Quaternion(
    x=-0.00142460053167, y=0.99999999999, z=-0.00177030764765, w=0.00253311793936
)
