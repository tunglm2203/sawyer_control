import os
import re
import rospy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import cv2
import pickle
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from sawyer_control.envs.sawyer_pickplace import SawyerPickPlaceXYZYawEnv

import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. The Actor Network (From your TD3-BC snippet)
# ==========================================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action=1.0):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = torch.FloatTensor(max_action).to(device)

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

# ==========================================
# 2. The StatesLogger
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
# 3. Evaluation Loop
# ==========================================
def test_td3_bc_policy():
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

    # Path to your TD3-BC model
    model_dir = f"/home/tung/workspace/rlhf_bench/iql-pytorch-sawyer/checkpoints/{task_name}/td3-bc/"
    model_load_path = os.path.join(model_dir, "steps_2600.pth_actor")
    output_dir = f"/home/tung/workspace/rlhf_bench/iql-pytorch-sawyer/output_video/{task_name}/td3-bc"
    os.makedirs(output_dir,exist_ok=True)
    
    num_test_episodes = 10
    max_steps = 150
    
    # Task specific dimensions
    STATE_DIM = 10
    ACTION_DIM = 5
    MAX_ACTION = np.ones(ACTION_DIM) # Matches your snippet requirement

    print(f"--- Loading Environment: {task_name} ---")
    env = SawyerPickPlaceXYZYawEnv(task_name=task_name)
    states_logger = StatesLogger()

    # --- Initialize TD3-BC Actor ---
    print(f"--- Loading TD3-BC Actor Model from {model_load_path} ---")
    policy = Actor(state_dim=STATE_DIM, action_dim=ACTION_DIM, max_action=MAX_ACTION).to(device)
    policy.load_state_dict(torch.load(model_load_path))
    policy.eval()

    print("--- Starting Evaluation ---")
    
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
                
            # 2. Predict Action
            state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(device)
            with torch.no_grad():
                action_5d = policy(state_tensor).cpu().data.numpy().flatten()

            # 3. Apply Safety Constraints
            if state[2] <= -0.285 and action_5d[2] < 0.0:
                action_5d[2] = 0.0 

            # 4. Format for Environment
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
            
            cv2.imshow("Sawyer TD3-BC Evaluation", obs['rgb_image'][:, :, ::-1])
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('i'):
                print(f" >>> [INTERVENED] Skipping Episode {ep+1}...")
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
    test_td3_bc_policy()