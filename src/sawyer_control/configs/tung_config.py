import numpy as np
from gym.spaces import Box
from sawyer_control.configs.ros_config import *

# SPACES
JOINT_ANGLES_HIGH = np.array([
    1.70167993,
    1.04700017,
    3.0541791,
    2.61797006,
    3.05900002,
    2.09400001,
    3.05899961
])

JOINT_ANGLES_LOW = np.array([
    -1.70167995,
    -2.14700025,
    -3.0541801,
    -0.04995198,
    -3.05900015,
    -1.5708003,
    -3.05899989
])

JOINT_VEL_HIGH = 2 * np.ones(7)
JOINT_VEL_LOW = -2 * np.ones(7)

MAX_TORQUES = 0.5 * np.array([8, 7, 6, 5, 4, 3, 2])

JOINT_TORQUE_HIGH = MAX_TORQUES
JOINT_TORQUE_LOW = -1 * MAX_TORQUES

JOINT_VALUE_HIGH = {
    'position': JOINT_ANGLES_HIGH,
    'velocity': JOINT_VEL_HIGH,
    'torque': JOINT_TORQUE_HIGH,
}
JOINT_VALUE_LOW = {
    'position': JOINT_ANGLES_LOW,
    'velocity': JOINT_VEL_LOW,
    'torque': JOINT_TORQUE_LOW,
}

END_EFFECTOR_POS_LOW = -1.2 * np.ones(3)
END_EFFECTOR_POS_HIGH = 1.2 * np.ones(3)

END_EFFECTOR_ANGLE_LOW = -1 * np.ones(4)
END_EFFECTOR_ANGLE_HIGH = np.ones(4)

END_EFFECTOR_VALUE_LOW = {
    'position': END_EFFECTOR_POS_LOW,
    'angle': END_EFFECTOR_ANGLE_LOW,
}

END_EFFECTOR_VALUE_HIGH = {
    'position': END_EFFECTOR_POS_HIGH,
    'angle': END_EFFECTOR_ANGLE_HIGH,
}

POSITION_CONTROL_LOW = -1 * np.ones(3)
POSITION_CONTROL_HIGH = np.ones(3)

RESET_TORQUE_LOW = -5
RESET_TORQUE_HIGH = 5

# SAFETY BOX SETTINGS
SAFETY_FORCE_MAGNITUDE = 5
SAFETY_FORCE_TEMPERATURE = 5

# overwrite these for your setup
RESET_SAFETY_BOX_LOWS = np.array([-.2, -0.6, 0.2])
RESET_SAFETY_BOX_HIGHS = np.array([.9, 0.4, 1])
RESET_SAFETY_BOX = Box(RESET_SAFETY_BOX_LOWS, RESET_SAFETY_BOX_HIGHS, dtype=np.float32)

TORQUE_SAFETY_BOX_LOWS = np.array([0.3, -0.4, 0.2])
TORQUE_SAFETY_BOX_HIGHS = np.array([0.7, 0.4, 0.7])
TORQUE_SAFETY_BOX = Box(TORQUE_SAFETY_BOX_LOWS, TORQUE_SAFETY_BOX_HIGHS, dtype=np.float32)

# POSITION_SAFETY_BOX_LOWS = np.array([.2, -.2, .06])
# POSITION_SAFETY_BOX_HIGHS = np.array([.6, .2, 0.5])

# TUNG: Space for moving in XY-plane
POSITION_SAFETY_BOX_LOWS = np.array([0.45, -.2, .065])
POSITION_SAFETY_BOX_HIGHS = np.array([0.85, .2, .07])
POSITION_SAFETY_BOX = Box(POSITION_SAFETY_BOX_LOWS, POSITION_SAFETY_BOX_HIGHS, dtype=np.float32)

TEST_POSITION = np.array([.6, .2, .06])  # (x, y, z)
# POSITION_RESET_POS = np.array([0.73, 0.0, 0.34222245])
POSITION_RESET_POS = np.array([0.50335, 0.046513, 0.06])   # (x, y, z)

# MISCELLANEOUS
RESET_LENGTH = 200
RESET_ERROR_THRESHOLD = .15 * np.ones(7)
UPDATE_HZ = 20  # 20

# JOINT_CONTROLLER_SETTINGS
JOINT_POSITION_SPEED = .1
JOINT_POSITION_TIMEOUT = .5
