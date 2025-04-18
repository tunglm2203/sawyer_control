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


def save_numpy_as_gif(array, filename, fps=20, scale=1.0):
    # ensure that the file has the .gif extension
    fname, _ = os.path.splitext(filename)
    filename = fname + '.gif'

    # copy into the color dimension if the images are black and white
    if array.ndim == 3:
        array = array[..., np.newaxis] * np.ones(3)

    # make the moviepy clip
    clip = ImageSequenceClip(list(array), fps=fps).resize(scale)
    clip.write_gif(filename, fps=fps)
    return clip

class PickleLogger:
    def __init__(self, filename):
        self.filename = filename
        self.data = []
        self.step = 0

    def __call__(self, observation, action, reward, done=0, metadata=None):
        step = copy.deepcopy(
            dict(
                observation=observation,
                action=action,
                reward=reward,
                done=done,
                metadata=metadata,
            )
        )
        self.data.append(step)
        self.step += 1

    def make_new_rollout(self, filename=None):
        if filename is not None:
            self.filename = filename
        self.data = []
        self.step = 0

    def save(self):
        print(f"Saving rollout to: {self.filename}")
        with open(self.filename, "wb") as f:
            pickle.dump(self.data, f)
        print(f"Done saving.")


def print_yellow(x):
    return print("\033[93m {}\033[00m".format(x))


def print_help():
    print_yellow("  Teleop Controls:")

    print_yellow("    w, s : move forward/backward (in your camera view)")
    print_yellow("    a, d : move left/right (in your camera view)")
    print_yellow("    c, z : move up/down (in your camera view)")
    print_yellow("    o, p:  rotate yaw (clockwise/counter-clockwise)")

    print_yellow("    space: toggle gripper")
    print_yellow("    r: reset robot")
    print_yellow("    m: to save demonstration")
    print_yellow("    g: to save gif")
    print_yellow("    h: help")
    print_yellow("    q: quit")


def show_video(image):
    cv2.imshow("Teleoperation Window (Robot image)", image)


def get_new_episode_idx(task_demo_path):
    def extract_episode_idx(filename):
        numbers = re.findall(r'\d+', filename)  # Find all numbers
        return int(numbers[-1]) if numbers else 0  # Return the last one, or 0 if no number

    all_files = os.listdir(task_demo_path)
    if len(all_files) > 0:
        sorted_files = sorted(all_files, key=extract_episode_idx)
        last_ep_idx = extract_episode_idx(sorted_files[-1])
        new_ep_idx = int(last_ep_idx) + 1
    else:
        new_ep_idx = 1

    return new_ep_idx


if __name__ == "__main__":
    def signal_handler(sig, frame):
        print("\nCtrl+C detected. Exiting Teleoperation program.")
        sys.exit(0)  # Exit cleanly

    # Register SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)

    """ Definition for user's hyperparameters and constants """
    _dt = 0.5       # max = 1
    _dr = 0.5       # max = 1
    KEYBOARD_ACTION_MAP = {
        ord("w"): np.array([-_dt, 0, 0, 0, 0]),
        ord("s"): np.array([_dt, 0, 0, 0, 0]),
        ord("a"): np.array([0, -_dt, 0, 0, 0]),
        ord("d"): np.array([0, _dt, 0, 0, 0]),
        ord("z"): np.array([0, 0, -_dt, 0, 0]),
        ord("c"): np.array([0, 0, _dt, 0, 0]),
        ord("o"): np.array([0, 0, 0, _dr, 0]),
        ord("p"): np.array([0, 0, 0, -_dr, 0]),
    }
    GRIPPER_STATE = {0: 'CLOSE', 1: 'OPEN'}


    """ Select tasks """
    # task_name = 'sawyer-pickup-banana-v0'
    # task_name = 'sawyer-pickup-banana-v1'
    task_name = 'sawyer-drawer-open-v0'

    env = SawyerPickPlaceXYZYawEnv(task_name=task_name)


    """ Utilities """
    def _execute_action(env, action):
        obs, reward, done, info = env.step(action)
        image = obs['rgb_image']

        logger(obs, action, 0.0, 0, None)
        print(f"Global step: {env.global_step}")
        return image


    def _execute_reset(env):
        null_action = np.array([0, 0, 0, 0, 1.0])
        obs = env.reset()
        image = obs['rgb_image']

        logger(obs, null_action, 0.0, 0, None)
        print(f"Global step: {env.global_step}")
        return image

    def _get_current_state(env):
        raw_obs = env._get_all_obs()
        image = raw_obs['camera_ob']
        gripper_pos = raw_obs['robot_ob'][:1]
        gripper_state = 1.0 if gripper_pos >= 0.04 else 0.0
        return image, gripper_state


    """ Logger to store rollout data """
    root_demo_path = '/home/tung/workspace/hrl_bench/preference_rl/sawyer_dataset'
    task_demo_path = os.path.join(root_demo_path, task_name)
    if not os.path.exists(task_demo_path):
        os.makedirs(task_demo_path)

    filename_template = "{task_name}_episode_{ep_idx}.pkl"
    new_ep_idx = get_new_episode_idx(task_demo_path)
    filename = os.path.join(task_demo_path, filename_template.format(task_name=task_name, ep_idx=new_ep_idx))
    logger = PickleLogger(filename=filename)


    """ Start Teleoperation """
    image, cur_gripper_state = _get_current_state(env)
    print_help()
    print("Started Teleoperation.")
    print(f"Current log's file: {logger.filename}")

    running = True
    is_open = cur_gripper_state     # The gripper is open at initial time
    gif_images = []
    while running:
        # Check for key press
        key = cv2.waitKey(40) & 0xFF

        # escape key to quit
        if key == ord("q"):
            print("Quitting teleoperation.")
            running = False
            continue

        # space bar to change gripper state
        elif key == ord(" "):
            is_open = 1 - is_open
            image = _execute_action(env, np.array([0, 0, 0, 0, is_open]))
            print(f"Gripper is now: {GRIPPER_STATE[is_open]}")
            gif_images.append(copy.deepcopy(image))

        elif key == ord("r"):
            print("Resetting robot...")
            image = _execute_reset(env)
            new_ep_idx = get_new_episode_idx(task_demo_path)
            new_filename = os.path.join(task_demo_path, filename_template.format(task_name=task_name, ep_idx=new_ep_idx))
            logger.make_new_rollout(filename=new_filename)
            is_open = 1
            print(f"Gripper is now: {GRIPPER_STATE[is_open]}")
            print_help()
            print(f"Current log's file: {logger.filename}")
            gif_images.append(copy.deepcopy(image))

        elif key == ord("h"):
            print_help()

        elif key == ord("m"):
            logger.save()
            new_ep_idx = get_new_episode_idx(task_demo_path)
            new_filename = os.path.join(task_demo_path, filename_template.format(task_name=task_name, ep_idx=new_ep_idx))
            logger.make_new_rollout(filename=new_filename)
            print(f"New log's file: {logger.filename}")

        elif key == ord("g"):
            if len(gif_images) > 0:
                save_numpy_as_gif(np.array(gif_images)[:, :, :, ::-1], 'cur_episode.gif')
            gif_images = []

        if key in KEYBOARD_ACTION_MAP:
            action = KEYBOARD_ACTION_MAP[key]
            action[-1] = is_open
            image = _execute_action(env, action)
            gif_images.append(copy.deepcopy(image))

            # print(f"cur_joint: {env.joint_angles}")
            # print(f"cur_ee_pos: {env.eef_pose[:3]}")

        if image is not None:
            show_video(image)

    cv2.destroyAllWindows()
    print("Teleoperation ended.")