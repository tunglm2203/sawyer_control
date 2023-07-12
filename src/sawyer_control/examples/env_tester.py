#!/usr/bin/python3
from sawyer_control.envs.sawyer_reaching import SawyerReachXYZEnv
from sawyer_control.envs.sawyer_env_base import SawyerEnvBase
import time
import numpy as np

# env = SawyerReachXYZEnv(
#     action_mode='position',
#     config_name='austri_config',
#     reset_free=False,
# 	position_action_scale=0.01,
# 	max_speed=0.4,
# )
# env.reset()

print("Constructing environment...")
env = SawyerEnvBase(
    # control_type="ik",
    # control_type="ik_quaternion",
    # control_type="ik_pos",
    control_type="torque",
    reset_free=False,
	move_speed=0.01,
    rotation_speed=0.01,
	max_speed=0.3,
    use_safety_box=False,
)
print("Resetting...")
obs = env.reset()
print("Velocity at reset: ", env.joint_velocities)

print("Stepping...")

import pdb; pdb.set_trace()
for i in range(5):
    # action = env.action_space.sample()
    # action[3:7] = np.array([1.0, 0., 0., 0.])   # Identity rotation
    # if i % 2 == 0:
    #     action[:3] = np.array([-1.0, 0.0, 0])
    # else:
    #     action[:3] = np.array([1.0, 0.0, 0])
    # action = np.array([-0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0 , 0.0])
    action = np.array([0., 0., 0., 0., 0., 0., 0.5, 1.0, 0.0])
    print("Action: ", action)
    obs, reward, done, info = env.step(action)
    print("Velocity: ", env.joint_velocities)
    breakpoint()

print("Finished.")