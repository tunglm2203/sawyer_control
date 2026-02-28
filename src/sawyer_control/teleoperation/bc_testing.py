import os
import re
from os.path import join
import rospy
import numpy as np
import torch
import copy
import random
import torch.nn as nn
import torch.nn.functional as F
import cv2
import pickle
from geometry_msgs.msg import PoseStamped
from sawyer_control.envs.sawyer_pickplace import SawyerPickPlaceXYZYawEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. The Actor Network (Must match exactly)
# ==========================================
class Actor(nn.Module):
    def __init__(self, state_dim=10, action_dim=5, max_action=1.0):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

# ==========================================
# 2. The StatesLogger (From your teleop script)
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


# ==========================================
# 3. Main Evaluation Loop
# ==========================================
def test_policy():
    # --- Configuration ---
    task_name = 'sawyer-pick-lift-banana-v0'
    trial_name = 'suboptimal_trajectories'
    """ Select target object """
    if task_name == 'sawyer-open-drawer-v0':
        target_object = "upper_drawer"
    elif task_name == 'sawyer-move-box-v0':
        target_object = "red_box"
    elif task_name == 'sawyer-pick-lift-banana-v0':
        target_object = "banana" 
    
    # Change this to whichever epoch performed best in your learning curves!
    model_path = f"/home/tung/workspace/rlhf_bench/iql-pytorch-sawyer/checkpoints/{task_name}/behavioral-cloning/bc_actor_epepisode.pth"
    
    num_test_episodes = 45
    max_steps = 150
    
    print(f"--- Loading Environment: {task_name} ---")
    env = SawyerPickPlaceXYZYawEnv(task_name=task_name)
    states_logger = StatesLogger()

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
    
    print("--- Starting Evaluation ---")
    possible_ep = [i for i in range(50,1001,50)]
    
    for ep in range(num_test_episodes):
        print(f"--- Loading Trained Actor Model ---")
        random_ep = random.choice(possible_ep)
        model_path_ = model_path.replace("episode",str(random_ep))
        print(f"Model Path: {model_path_}")
        if task_name == 'sawyer-open-drawer-v0':
            ACTION_DIM = 3
        elif task_name == 'sawyer-move-box-v0':
            ACTION_DIM = 5
        elif task_name == 'sawyer-pick-lift-banana-v0':
            ACTION_DIM = 5
        actor = Actor(state_dim=10, action_dim=ACTION_DIM, max_action=1.0).to(device)
        # Load weights and set to evaluation mode (disables dropout/batchnorm updates)
        actor.load_state_dict(torch.load(model_path_))
        actor.eval()

        obs = env.reset()
        rospy.sleep(1.0) # Let physics and tags settle
        
        print(f"\n--- Episode {ep+1} Start ---")
        
        for step in range(max_steps):
            # 1. Get current state
            state = states_logger._get_current_state_space(env, target_object)
            if state is None:
                continue # Skip step if tags are temporarily occluded
                
            # 2. Predict Action (NO EXPLORATION NOISE!)
            state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(device)
            with torch.no_grad():
                action_3d = actor(state_tensor).cpu().data.numpy().flatten()
            print(action_3d)

            # 3. Apply Safety Constraints
            current_gripper_z = state[2]
            z_lower_limit = -0.285
            if current_gripper_z <= z_lower_limit and action_3d[2] < 0.0:
                action_3d[2] = 0.0 # Prevent moving further down


            # 4. Pad to 5D for the Environment (dx, dy, dz, yaw=0, gripper=open)
            if task_name == 'sawyer-open-drawer-v0':
                env_action = np.array([
                action_3d[0],
                action_3d[1],
                action_3d[2],
                0.0, 
                1.0
            ])
            elif task_name == 'sawyer-move-box-v0':
                env_action = np.array([
                action_3d[0],
                action_3d[1],
                action_3d[2],
                0.0, 
                0.0
            ])
            elif task_name == 'sawyer-pick-lift-banana-v0':
                gripper_cmd = 0.0 if abs(action_3d[-1])<0.5 else 1.0
                env_action = np.array([
                action_3d[0],
                action_3d[1],
                action_3d[2],
                0.0, 
                gripper_cmd
            ])
            
            # 5. Execute Action
            obs, reward, done, info = env.step(env_action, state)
            logger(state, env_action, reward, done, None)
            
            # Optional: Show the camera view so you can watch it live
            cv2.imshow("Sawyer Autonomous Evaluation", obs['rgb_image'][:, :, ::-1])
            key = cv2.waitKey(1) & 0xFF

            if key == ord('i'):
                print(f" >>> [INTERVENED] Skipping Episode {ep+1} and saving...")
                break # Jumps out to logger.save()
            elif key == ord('q'):
                print("Exiting...")
                cv2.destroyAllWindows()
                return
            
            if done:
                print(f"Episode {ep+1} finished successfully at step {step}!")
                break
            
        # Save trajectories
        logger.save()
        new_ep_idx = get_new_episode_idx(task_demo_path)
        new_filename = os.path.join(task_demo_path, filename_template.format(task_name=task_name, ep_idx=new_ep_idx))
        logger.make_new_rollout(filename=new_filename)
        print(f"New log's file: {logger.filename}\n")
                
        if not done:
            print(f"Episode {ep+1} timed out after {max_steps} steps.")

    cv2.destroyAllWindows()
    print("--- Evaluation Complete ---")

if __name__ == "__main__":
    test_policy()