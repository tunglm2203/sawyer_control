import os
import sys
import re
import cv2
import copy
import pickle
import signal
import time
import rospy
import numpy as np
from os.path import join
from geometry_msgs.msg import PoseStamped
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
    def __init__(self, filename, trial_name):
        self.filename = filename
        self.trial_name = trial_name
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
        print(f"Saving rollout to: {join(self.trial_name, self.filename)}")
        with open(join(self.trial_name, self.filename), "wb") as f:
            pickle.dump(self.data, f)
        print(f"Done saving.")

class StatesLogger:
    def __init__(self):
        # 1. Create a dictionary to hold the freshest poses
        self.latest_poses = {}
        
        # 2. Set up a persistent subscriber that runs in the background
        self.pose_sub = rospy.Subscriber(
            "apriltag/3d_pose", 
            PoseStamped, 
            self._pose_callback
        )
        rospy.sleep(1.0) 

    def _pose_callback(self, msg):
        """
        This runs automatically every time a new message hits the topic.
        It simply overwrites the old position with the newest one.
        """
        self.latest_poses[msg.header.frame_id] = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        # print(self.latest_poses)

    def _get_current_gripper_state(self, env):
        raw_obs = env._get_all_obs()
        gripper_pos = raw_obs['robot_ob'][:1]
        gripper_state = 1.0 if gripper_pos >= 0.04 else 0.0
        return gripper_state


    def _get_current_state_space(self, env, target_object):
        """
        Reads instantly from the dictionary without any loops or waiting.
        """
        # Safety check: Ensure the tags have been detected at least once
        if "gripper" not in self.latest_poses or target_object not in self.latest_poses:
            rospy.logerr(f"Cannot find tags! Current known tags: {list(self.latest_poses.keys())}")
            # You might want to pause or return a zero-array here 
            # to prevent the RL agent from crashing if a tag is occluded.
            
        # Instantly grab the latest arrays
        gripper_pos = self.latest_poses["gripper"]
        target_pos = self.latest_poses[target_object]
        
        # Calculate relative distance
        rel_distance = target_pos - gripper_pos
        
        # Get the current gripper state (assuming this is defined elsewhere in your class)
        gripper_state = self._get_current_gripper_state(env)
        
        # Concatenate everything into the 10-D NumPy array
        state_space = np.concatenate([
            gripper_pos,
            target_pos,
            rel_distance,
            [gripper_state]
        ]).astype(np.float32)
        
        return state_space


def print_yellow(x):
    return print("\033[93m {}\033[00m".format(x))


def print_help():
    print_yellow("  Teleop Controls:")

    print_yellow("    w, s : move forward/backward (in your camera view)")
    print_yellow("    a, d : move left/right (in your camera view)")
    print_yellow("    c, z : move up/down (in your camera view)")
    print_yellow("    o, p:  rotate yaw (clockwise/counter-clockwise)")
    print_yellow("    x: do nothing (Hall action)")

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
        ord("x"): np.array([0, 0, 0, 0, 0])
        # ord("j"):np.array([_dt+0.07, -_dt, _dt, 0, 0])
    }
    GRIPPER_STATE = {0: 'CLOSE', 1: 'OPEN'}


    """ Select tasks """
    # task_name = 'sawyer-pickup-banana-v2'
    # task_name = 'sawyer-open-drawer-v0'
    # task_name = 'sawyer-pick-lift-banana-v0'
    task_name = 'sawyer-move-box-v0'

    """ Select trial name """
    trial_name = 'successful_trajectories'

    """ Select target object """
    if task_name == 'sawyer-open-drawer-v0':
        target_object = "upper_drawer"
    elif task_name == 'sawyer-move-box-v0':
        target_object = "red_box"
    elif task_name == 'sawyer-pick-lift-banana-v0':
        target_object = "banana" 

    env = SawyerPickPlaceXYZYawEnv(task_name=task_name)
    states_logger = StatesLogger()


    """ Utilities """
    def _execute_action(env, action):
        cur_tag_state_space = states_logger._get_current_state_space(env, target_object)
        # breakpoint()
        obs, reward, done, info = env.step(action, cur_tag_state_space) # , cur_tag_state_space)
        # print(f"Current EE height: {obs['ee_state'][2]}")
        image = obs['rgb_image']

        logger(cur_tag_state_space, action, reward, done, None)
        # print(f"Global step: {env.global_step}")
        print(f"Reward Value: {reward}")
        return image


    def _execute_reset(env):
        cur_tag_state_space = states_logger._get_current_state_space(env, target_object)
        null_action = np.array([0, 0, 0, 0, 1.0])
        obs = env.reset()
        image = obs['rgb_image']

        logger(cur_tag_state_space, null_action, 0.0, 0, None)
        print(f"Global step: {env.global_step}")
        return image

    def _get_current_state(env):
        raw_obs = env._get_all_obs()
        image = raw_obs['camera_ob']
        gripper_pos = raw_obs['robot_ob'][:1]
        gripper_state = 1.0 if gripper_pos >= 0.04 else 0.0
        return image, gripper_state

    """ Logger to store rollout data """
    root_demo_path = '/home/tung/workspace/rlhf_bench/iql-pytorch-sawyer/datasets'
    task_demo_path = os.path.join(root_demo_path, task_name, trial_name)
    if not os.path.exists(task_demo_path):
        os.makedirs(task_demo_path)

    # # """ Start Fresh """
    # for filename in os.listdir(task_demo_path):
    #     file_path = os.path.join(task_demo_path, filename)
    #     if os.path.isfile(file_path):
    #         os.remove(file_path)

    filename_template = "{task_name}_episode_{ep_idx}.pkl"
    new_ep_idx = get_new_episode_idx(task_demo_path)
    filename = os.path.join(task_demo_path, filename_template.format(task_name=task_name, ep_idx=new_ep_idx))
    logger = PickleLogger(filename=filename, trial_name = trial_name)

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
            print(f"New log's file: {logger.filename}\n")

        elif key == ord("g"):
            if len(gif_images) > 0:
                save_numpy_as_gif(np.array(gif_images)[:, :, :, ::-1], 'cur_episode.gif')
            gif_images = []

        if key in KEYBOARD_ACTION_MAP:
            # print(f"cur_joint: {env.joint_angles}")
            # print(f"cur_ee_pos: {env.eef_pose[:3]}")
            print(f"cur_tag_state_space: {states_logger._get_current_state_space(env, 'red_box')}")
            action = KEYBOARD_ACTION_MAP[key]
            action[-1] = is_open
            print(f"executed_action: {action}")
            image = _execute_action(env, action)
            gif_images.append(copy.deepcopy(image))
            print("="*10)
            
        if image is not None:
            show_video(image)

    cv2.destroyAllWindows()
    print("Teleoperation ended.")