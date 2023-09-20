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
# control_type="ik"
control_type="ik_quaternion"
# control_type="ik_pos"
# control_type="torque"
# control_type="impedance"
env = SawyerEnvBase(
    control_type=control_type,
    reset_free=False,
	move_speed=0.1,
    rotation_speed=1.0,
	max_speed=0.3,
    use_safety_box=False,
)
# breakpoint()
print(f"N repeats control: {env._control_timestep / env._model_timestep}.")
print("\n==> Resetting...")
obs = env.reset()
# print("Velocity at reset: ", env.joint_velocities)
# print("Joint angle at reset: ", env.joint_angles.tolist())
print(f"EE pose at reset: ", env.eef_pose.tolist())
# print(f"Obs at reset: ", obs['robot_ob'].tolist())
print("\n==> Stepping...")

breakpoint()
if control_type == "impedance":
    actions_test = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, -1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0, -1.0, 0.0],
        [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, -1.0, 0.0],
        [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, -1.0, 0.0],
        [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, -1.0, 0.0],
    ])
    for i in range(6):
        # action = env.action_space.sample()
        # action[3:7] = np.array([1.0, 0., 0., 0.])   # Identity rotation
        # if i % 2 == 0:
        #     action[:3] = np.array([-1.0, 0.0, 0])
        # else:
        #     action[:3] = np.array([1.0, 0.0, 0])
        # action = np.array([-0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0 , 0.0])
        # action = np.array([0., 0., 0., 0., 0., 0., 0.5, 1.0, 0.0])
        idx = int(i//2)
        action = actions_test[idx]
        print("Action: ", action)
        obs, reward, done, info = env.step(action)
        print("Velocity: ", env.joint_velocities.tolist())
        print("Joint pos: ", env.joint_angles)

elif control_type == "ik_quaternion":
    for i in range(10):
        action = np.array([
            0.5, 0, 0,
            1.0, 0., 0., 0.,
            1.0, 0.0
        ])
        print("Action: ", action)

        obs, reward, done, info = env.step(action)

        print("Joint: ", env.joint_angles.tolist())
        print("EE pose: ", env.eef_pose.tolist())
        # print("Velocity: ", env.joint_velocities.tolist())

        breakpoint()

print("Finished.")