import os
import sys
import re
import cv2
import copy
import pickle
import signal
import time
import numpy as np
from moviepy.editor import ImageSequenceClip

from sawyer_control.envs.sawyer_pickplace import SawyerPickPlaceXYZYawEnv



task_name = 'sawyer-pickup-banana-v1'
# task_name = 'sawyer-drawer-open-v0'

env = SawyerPickPlaceXYZYawEnv(task_name=task_name)
breakpoint()

obs = env.reset()

# action = env.action_space.sample()
# next_obs, reward, done, info = env.step(action)
