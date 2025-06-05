import numpy as np


# =============================== SCOPE FOR ROBOT CONSTANT ===============================
USE_GAZEBO = True

CONTROL_FREQ = 20
ROBOT_DOF = 7
GRIPPER_DOF = 1

USE_GRIPPER = True
GRIPPER_CLOSE_POSITION = 0.0
GRIPPER_OPEN_POSITION = 0.041667
DEFAULT_GRIPPER_VELOCITY = 3.0

CAMERA_WIDTH = 480
CAMERA_HEIGHT = 480

# Joints name convention: https://support.rethinkrobotics.com/support/solutions/articles/80000976455-sawyer-hardware
JOINT_NAMES = ['right_j0', 'right_j1', 'right_j2', 'right_j3', 'right_j4', 'right_j5', 'right_j6']

# Links name used to compute Jacobian
LINK_NAMES = ['right_l2', 'right_l3', 'right_l4', 'right_l5', 'right_l6', 'right_hand', 'right_gripper_tip']




# =============================== SCOPE FOR INITIALIZATION ===============================
# Initial joint positions at reset time for two type of end-effector frame: 'right_hand', 'right_gripper_tip'
RIGHT_HAND_RESET_POSE = [
    0.626337315384, 0.10272965114, 0.4217483153,
    -0.00142460053167, 0.99999999999, -0.00177030764765, 0.00253311793936
]  # [(xyz), (xyzw)]
RIGHT_GRIPPER_TIP_RESET_POSE = [
    0.630835664505, 0.100420400837, 0.28622261066,
    -0.00193828719291, 0.99999999999, -0.00177030764765, 0.00253311793936
]  # [(xyz), (xyzw)]

# This is used to seed the angles for computing Inverse Kinematic
# INITIAL_JOINT_ANGLES = np.array([-0.28, -0.60, 0.00, 1.86, 0.00, 0.3, 1.57])
# INITIAL_JOINT_ANGLES = np.array([-0.28, -0.60, 0.00, 1.86, 0.00, 0.3, -1.57])
INITIAL_JOINT_ANGLES = np.array([-0.28930469, -0.60313184,  0.00958203,  1.85056348, -0.00938867,  0.31810352,
 -1.66642773])

NUM_TRIALS_AT_RESET = 20   # Num of iterations for resetting
TOLERANCE_AT_RESET = 0.15 * np.ones(ROBOT_DOF)  # Threshold to stop loop at reset




# =============================== SCOPE FOR LIMITS ===============================
# This range limit of joint's positions, velocities, torques is obtained
# from sawyer_robot/sawyer_description/urdf/sawyer_base.urdf.xacro
if USE_GAZEBO:
    JOINT_POS_UPPER = np.array([3.0503, 2.2736, 3.0426, 3.0439, 2.9761, 2.9761, 3.14], dtype=np.float32)
    JOINT_POS_LOWER = np.array([-3.0503, -3.8095, -3.0426, -3.0439, -2.9761, -2.9761, -3.14], dtype=np.float32)
else:
    JOINT_POS_UPPER = np.array([3.0503, 2.2736, 3.0426, 3.0439, 2.9761, 2.9761, 4.7124], dtype=np.float32)
    JOINT_POS_LOWER = np.array([-3.0503, -3.8095, -3.0426, -3.0439, -2.9761, -2.9761, -4.7124], dtype=np.float32)

JOINT_VEL_UPPER = np.array([1.74, 1.328, 1.957, 1.957, 3.485, 3.485, 4.545], dtype=np.float32)
JOINT_VEL_LOWER = -1.0 * JOINT_VEL_UPPER

JOINT_TORQUE_UPPER = np.array([80.0, 80.0, 40.0, 40.0, 9.0, 9.0, 9.0], dtype=np.float32)
JOINT_TORQUE_LOWER = -1.0 * JOINT_TORQUE_UPPER

# End-effector limits: user define
EE_POS_LOWER = np.array([0.43, -0.145, -0.065], dtype=np.float64)
EE_POS_UPPER = np.array([0.85, 0.17, 0.15], dtype=np.float64)
# EE_POS_LOWER = np.array([0.43, -0.14, -0.065], dtype=np.float64)
# EE_POS_UPPER = np.array([0.85, 0.16, 0.15], dtype=np.float64)

# SAFETY BOX SETTINGS
SAFETY_FORCE_MAGNITUDE = 5
SAFETY_FORCE_TEMPERATURE = 5

