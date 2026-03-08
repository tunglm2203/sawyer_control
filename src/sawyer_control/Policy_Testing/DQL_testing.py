import os
import re
import rospy
import numpy as np
import torch
import copy
import cv2
import pickle
import cv2
import numpy as np
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from sawyer_control.envs.sawyer_pickplace import SawyerPickPlaceXYZYawEnv

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import your Diffusion QL classes
# Ensure these files are in your python path or the same directory
from agents.ql_diffusion import Diffusion_QL
from agents.bc_diffusion import Diffusion_BC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. The StatesLogger (Keep as is)
# ==========================================
class StatesLogger:
    def __init__(self):
        self.latest_poses = {}
        self.pose_sub = rospy.Subscriber("apriltag/3d_pose", PoseStamped, self._pose_callback)
        rospy.sleep(1.0) 

    def _pose_callback(self, msg):
        self.latest_poses[msg.header.frame_id] = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])

    def _get_current_gripper_state(self, env):
        raw_obs = env._get_all_obs()
        gripper_pos = raw_obs['robot_ob'][:1]
        return 1.0 if gripper_pos >= 0.04 else 0.0

    def _get_current_state_space(self, env, target_object):
        if "gripper" not in self.latest_poses or target_object not in self.latest_poses:
            rospy.logwarn(f"Waiting for tags... Current: {list(self.latest_poses.keys())}")
            return None
        gripper_pos = self.latest_poses["gripper"]
        target_pos = self.latest_poses[target_object]
        rel_distance = target_pos - gripper_pos
        gripper_state = self._get_current_gripper_state(env)
        return np.concatenate([gripper_pos, target_pos, rel_distance, [gripper_state]]).astype(np.float32)

# ==========================================
# 2. Evaluation Loop
# ==========================================

def create_video_from_frames(frames, output_name="output_video.mp4", fps=10):
    """
    Converts a list/array of RGB image sequences into an MP4 video.
    
    Args:
        frames (list or np.ndarray): List of images as numpy arrays (H, W, 3).
        output_name (str): The filename for the resulting mp4.
        fps (int): Frames per second.
    """
    if not frames:
        print("No frames provided.")
        return

    # Get dimensions from the first frame
    height, width, layers = frames[0].shape
    size = (width, height)

    # Define the codec and create VideoWriter object
    # 'mp4v' is a standard codec for .mp4 files
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_name, fourcc, fps, size)

    for frame in frames:
        # Convert RGB to BGR (OpenCV uses BGR)
        # bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)

    out.release()
    print(f"Video saved successfully as {output_name}")

def test_diffusion_policy():
    pub = rospy.Publisher('/activate_record_video', String, queue_size=10)
    # --- Configuration ---
    task_name = 'sawyer-move-box-v0'
    """ Select target object """
    if task_name == 'sawyer-open-drawer-v0':
        target_object = "upper_drawer"
    elif task_name == 'sawyer-move-box-v0':
        target_object = "red_box"
    elif task_name == 'sawyer-pick-lift-banana-v0':
        target_object = "banana" 

    algorithm = "diffusion-ql"
    max_q_backup = False
    eta = 0.001
    lr = 0.0003
    model_id = "80"
    
    # Path to your Diffusion QL models directory
    checkpoint_name = f"{task_name}|exp_1|{algorithm}|T-100|ms-offline|k-1|0|lr-{lr}|eta-{eta}|max_q_backup-{max_q_backup}|reward_tune-no|gn-5.0|mixed_v2"
    model_dir = f"/home/tung/workspace/rlhf_bench/iql-pytorch-sawyer/checkpoints/{task_name}/{algorithm}/{checkpoint_name}/ckpt"
    output_dir = f"/home/tung/workspace/rlhf_bench/iql-pytorch-sawyer/output_video/{task_name}/{algorithm}/{checkpoint_name}/epoch_{model_id}"
    os.makedirs(output_dir,exist_ok=True)
    
    num_test_episodes = 10
    max_steps = 150
    
    # Task specific dimensions
    STATE_DIM = 10
    ACTION_DIM = 5
    MAX_ACTION = 1.0

    print(f"--- Loading Environment: {task_name} ---")
    env = SawyerPickPlaceXYZYawEnv(task_name=task_name)
    states_logger = StatesLogger()

    # --- Initialize Diffusion QL Agent ---
    # Parameters must match the training script exactly
    if algorithm == "diffusion-ql":
        agent = Diffusion_QL(state_dim=STATE_DIM,
                      action_dim=ACTION_DIM,
                      max_action=MAX_ACTION,
                      device=device,
                      discount=0.99,
                      tau=0.005,
                      max_q_backup=max_q_backup,
                      beta_schedule='vp',
                      n_timesteps=100,
                      eta=eta,
                      lr=lr,
                      lr_decay=False,
                      lr_maxt=1000,
                      grad_norm=5.0)
    else:
        agent = Diffusion_QL(state_dim=STATE_DIM,
                      action_dim=ACTION_DIM,
                      max_action=MAX_ACTION,
                      device=device,
                      discount=0.99,
                      tau=0.005,
                      beta_schedule='vp',
                      n_timesteps=5,
                      lr=lr)
    print(f"--- Loading Diffusion QL Model from {model_dir} (ID: {model_id}) ---")
    agent.load_model(model_dir, id=model_id)
    # Target EMA model is usually used for evaluation in Diffusion QL
    agent.ema_model.eval() 
    agent.critic.eval()

    print("--- Starting Diffusion Evaluation ---")
    
    for ep in range(num_test_episodes):
        obs = env.reset()
        rospy.sleep(1.0)
        pub.publish("0")
        
        print(f"\n--- Episode {ep+1} Start ---")
        pub.publish(os.path.join(output_dir,f"episode_{ep+1}.mp4"))
        
        for step in range(max_steps):
            # 1. Get current state
            state = states_logger._get_current_state_space(env, target_object)
            if state is None:
                continue
                
            # 2. Sample Action using Diffusion Denoising + Q-Selection
            # This calls agent.sample_action which performs the 50-sample Q-filtering
            action_5d = agent.sample_action(state)

            # 3. Apply Safety Constraints
            if state[2] <= -0.285 and action_5d[2] < 0.0:
                action_5d[2] = 0.0 

            # 4. Format for Environment
            # gripper_cmd = 0.0 if abs(action_5d[-1]) < 0.5 else 1.0
            if task_name == 'sawyer-open-drawer-v0':
                gripper_cmd = 1.0
            elif task_name == 'sawyer-move-box-v0':
                gripper_cmd = 0.0
            elif task_name == 'sawyer-pick-lift-banana-v0':
                gripper_cmd = 0.0 if abs(action_5d[-1]) < 0.5 else 1.0
            env_action = np.array([
                action_5d[0],
                action_5d[1],
                action_5d[2],
                0.0, # Yaw locked
                gripper_cmd
            ])
            
            # 5. Execute Action
            obs, reward, done, info = env.step(env_action, state)
            
            cv2.imshow("Sawyer Diffusion Evaluation", obs['rgb_image'][:, :, ::])
            key = cv2.waitKey(1) & 0xFF
            if key == ord('i'):
                print(f" >>> [INTERVENED] Skipping Episode {ep+1} and saving...")
                break 
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return
            
            if done:
                print(f"Episode {ep+1} Success at step {step}!")
                break

        if not done:
            print(f"Episode {ep+1} timed out.")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_diffusion_policy()