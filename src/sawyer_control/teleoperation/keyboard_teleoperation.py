import numpy as np
import cv2
import time
import rospy

from sawyer_control.envs.sawyer_pickplace import SawyerPickPlaceXYZYawEnv

def print_yellow(x):
    return print("\033[93m {}\033[00m".format(x))


def print_help():
    print_yellow("  Teleop Controls:")

    print_yellow("    w, s : move forward/backward")
    print_yellow("    a, d : move left/right")
    print_yellow("    z, c : move up/down")
    print_yellow("    o, p:  rotate yaw (clockwise/counter-clockwise)")

    print_yellow("    space: toggle gripper")
    print_yellow("    r: reset robot")
    print_yellow("    h: help")
    print_yellow("    q: quit")


def show_video(image):
    """
    This shows the video from the camera for a given duration.
    """
    cv2.imshow("robot img", image)


if __name__ == "__main__":
    _dt = 0.5       # max = 1
    _dr = 0.5       # max = 1
    keyboard_action_map = {
        ord("w"): np.array([_dt, 0, 0, 0, 0]),
        ord("s"): np.array([-_dt, 0, 0, 0, 0]),
        ord("a"): np.array([0, -_dt, 0, 0, 0]),
        ord("d"): np.array([0, _dt, 0, 0, 0]),
        ord("z"): np.array([0, 0, _dt, 0, 0]),
        ord("c"): np.array([0, 0, -_dt, 0, 0]),
        ord("o"): np.array([0, 0, 0, _dr, 0]),
        ord("p"): np.array([0, 0, 0, -_dr, 0]),
    }

    def _execute_action(env, action):
        obs, reward, done, info = env.step(action)
        image = obs['rgb_image']
        return image

    def _execute_reset(env):
        obs = env.reset()
        image = obs['rgb_image']
        return image

    env = SawyerPickPlaceXYZYawEnv()
    image = _execute_reset(env)
    print_help()
    print("Started Teleoperation.")

    running = True
    is_open = 1     # The gripper is open at initial time
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
            print("Gripper is now: ")

        elif key == ord("r"):
            print("Resetting robot...")
            image = _execute_reset(env)
            is_open = 1
            print("Gripper is now: ")
            print_help()

        elif key == ord("h"):
            print_help()

        if key in keyboard_action_map:
            print(f"Key pressed: {key}")
            action = keyboard_action_map[key]
            action[-1] = is_open
            image = _execute_action(env, action)

        show_video(image)

    cv2.destroyAllWindows()
    print("Teleoperation ended.")