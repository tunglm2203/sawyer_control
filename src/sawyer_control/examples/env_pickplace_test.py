import numpy as np
import cv2

from sawyer_control.envs.sawyer_pickplace import SawyerPickPlaceXYZEnv

env = SawyerPickPlaceXYZEnv()

print(f"obs space: {env.observation_space}")
print(f"act space: {env.action_space}")

print("\n==> Resetting...")
obs = env.reset()

for i in range(5):
    action = np.array([0.5, 0, 0])
    # action = env.action_space.sample()
    # print("Action: ", action)

    obs, reward, done, info = env.step(action)
    cv2.imshow("CV Image", obs['rgb_image'])
    cv2.waitKey(1)

    print("EE pose: ", env.eef_pose.tolist())

for i in range(5):
    action = np.array([0, 0.5, 0])
    # print("Action: ", action)

    obs, reward, done, info = env.step(action)
    cv2.imshow("CV Image", obs['rgb_image'])
    cv2.waitKey(1)

    print("EE pose: ", env.eef_pose.tolist())

# for i in range(10):
#     action = np.array([0.5, 0.5, 0.5])
#     # print("Action: ", action)
#
#     obs, reward, done, info = env.step(action)
#     cv2.imshow("CV Image", obs['rgb_image'])
#     cv2.waitKey(1)
