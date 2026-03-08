import os
import re
import rospy
import numpy as np
import torch
import copy
import cv2
import pickle
from geometry_msgs.msg import PoseStamped
from sawyer_control.envs.sawyer_pickplace import SawyerPickPlaceXYZYawEnv

# Import your Diffusion QL classes
# Ensure these files are in your python path or the same directory
from agents.diffusion_ql import Diffusion_QL 

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
def test_diffusion_policy():
    # --- Configuration ---
    task_name = 'sawyer-pick-lift-banana-v0'
    target_object = "banana" 
    
    # Path to your Diffusion QL models directory
    model_dir = "/home/tung/workspace/rlhf_bench/iql-pytorch-sawyer/checkpoints/diffusion_ql"
    model_id = "1000" # Use the specific iteration/epoch ID
    
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
    # Parameters must match your training script exactly
    agent = Diffusion_QL(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        max_action=MAX_ACTION,
        device=device,
        discount=0.99,
        tau=0.005,
        n_timesteps=100, # Must match training
        beta_schedule='linear'
    )
    
    print(f"--- Loading Diffusion QL Model from {model_dir} (ID: {model_id}) ---")
    agent.load_model(model_dir, id=model_id)
    # Target EMA model is usually used for evaluation in Diffusion QL
    agent.ema_model.eval() 
    agent.critic.eval()

    print("--- Starting Diffusion Evaluation ---")
    
    for ep in range(num_test_episodes):
        obs = env.reset()
        rospy.sleep(1.0)
        
        print(f"\n--- Episode {ep+1} Start ---")
        
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
            
            cv2.imshow("Sawyer Diffusion Evaluation", obs['rgb_image'][:, :, ::-1])
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
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