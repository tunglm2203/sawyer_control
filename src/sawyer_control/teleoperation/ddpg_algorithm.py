import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import rospy
import numpy as np
import torch
import torch
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

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        # Pre-allocate memory for performance
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, reward, next_state, done):
        # Insert transition at the current pointer
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done

        # Advance pointer and loop back to 0 if max_size is reached
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        # Randomly sample a batch of indices
        ind = np.random.randint(0, self.size, size=batch_size)

        # Return as PyTorch tensors pushed to the correct device (GPU/CPU)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device)
        )



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim=10, action_dim=4, max_action=1.0):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        # Tanh bounds the output to [-1, 1], then scale by your max velocity
        return self.max_action * torch.tanh(self.l3(a))

class Critic(nn.Module):
    def __init__(self, state_dim=10, action_dim=4):
        super(Critic, self).__init__()
        # The first layer takes the concatenated state and action
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state, action):
        q = F.relu(self.l1(torch.cat([state, action], 1)))
        q = F.relu(self.l2(q))
        return self.l3(q)

class DDPGAgent:
    def __init__(self, state_dim=10, action_dim=4, max_action=1.0):
        # Initialize Main Networks
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        # Initialize Target Networks (cloned from main networks)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.max_action = max_action

    def select_action(self, state):
        """Used during environment interaction to get an action from the Actor"""
        state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state_tensor).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256, discount=0.99, tau=0.005):
        """Samples a batch from the buffer and performs one optimization step"""
        
        # 1. Sample from the Replay Buffer (assumes you are using the buffer class we discussed)
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        # -----------------------------
        # Optimize Critic
        # -----------------------------
        # Get target Q-value: y = r + gamma * Q'(s', \mu'(s'))
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_Q = self.critic_target(next_state, next_action)
            target_Q = reward + (1 - done) * discount * target_Q

        # Get current Q-value estimate
        current_Q = self.critic(state, action)

        # Compute Critic loss (Mean Squared Error)
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Backpropagate Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # -----------------------------
        # Optimize Actor
        # -----------------------------
        # Compute Actor loss: -Q(s, \mu(s))
        # We want to maximize Q, which means minimizing negative Q
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Backpropagate Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # -----------------------------
        # Soft Update Target Networks
        # -----------------------------
        # \theta' <- \tau * \theta + (1 - \tau) * \theta'
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

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

# Assuming these are imported from the files where you defined them previously
# from ddpg_agent import DDPGAgent, ReplayBuffer
# from states_logger import StatesLogger
# from sawyer_control.envs.sawyer_pickplace import SawyerPickPlaceXYZYawEnv

def train_ddpg():
    """ Configuration and Hyperparameters """
    task_name = 'sawyer-open-drawer-v0'
    trial_name = 'failed_trajectories'
    target_object = 'upper_drawer'
    
    max_episodes = 2000
    max_steps_per_episode = 150 # Prevent infinite wandering
    batch_size = 512            # Leveraging high-end GPUs
    exploration_noise = 0.1     # Standard deviation of Gaussian noise
    start_timesteps = 100      # Steps to take random actions before training begins
    
    # Initialize Environment and State Logger
    env = SawyerPickPlaceXYZYawEnv(task_name=task_name)
    states_logger = StatesLogger()
    
    # Initialize Agent and Buffer (10-D state, 5-D action to match your env)
    # Note: Your teleop uses a 5-D action [x, y, z, yaw, gripper]. 
    # If yaw is fixed, keep it 0. DDPG will output 5 values.
    state_dim = 10
    action_dim = 3
    max_action = 1.0 # Assuming velocities/gripper are bounded [-1, 1]
    
    agent = DDPGAgent(state_dim=state_dim, action_dim=action_dim, max_action=max_action)
    replay_buffer = ReplayBuffer(state_dim=state_dim, action_dim=action_dim)
    
    total_timesteps = 0
    episode_reward = 0
    episode_num = 0

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

    print("--- Starting DDPG Training ---")

    for episode in range(max_episodes):
        # 1. Reset the environment at the start of each episode
        env.reset()
        
        # Allow Gazebo physics to settle and Apriltags to update
        rospy.sleep(0.5) 
        
        # 2. Get the initial 10-D State
        state = states_logger._get_current_state_space(env, target_object)
        episode_reward = 0
        
        for step in range(max_steps_per_episode):
            total_timesteps += 1
            
            # 3. Select Action
            if total_timesteps < start_timesteps:
                # Warm-up phase: purely random actions to fill the buffer with diverse data
                action = np.random.uniform(-max_action, max_action, size=action_dim)
            else:
                # Normal phase: Agent predicts action, add Gaussian noise for exploration
                action = agent.select_action(state)
                noise = np.random.normal(0, exploration_noise, size=action_dim)
                action = np.clip(action + noise, -max_action, max_action)

            # Format action for your specific environment mapping
            # (e.g., ensuring the gripper threshold behaves correctly if needed)

            # --- NEW: Z-Axis Safety Constraint ---
            current_gripper_z = state[2]
            z_lower_limit = -0.285

            # If the gripper is at/below the limit AND the action commands it to go up
            if current_gripper_z <= z_lower_limit and action[2] > 0.0:
                action[2] = 0.0  # Nullify the downward velocity command

            # Force the yaw action (index 3) to be strictly 0.0
            yaw_cmd = 0.0

            # Binarize the gripper command from the 4th output (index 3 of the NN action)
            # gripper_cmd = 1.0 if action[3] > 0.0 else 0.0
            gripper_cmd = 1.0

            # Build the 5D array expected by SawyerPickPlaceXYZYawEnv
            env_action = np.array([
                action[0],   # x velocity
                action[1],   # y velocity
                action[2],   # z velocity
                yaw_cmd,     # 0.0 (No rotation)
                gripper_cmd  # Binarized gripper
            ])
            
            # 4. Execute Action in Environment
            # Passing the 10-D state into env.step just like your teleop script does
            obs, reward, done, info = env.step(env_action, state)
            
            # 5. Observe Next State
            next_state = states_logger._get_current_state_space(env, target_object)
            
            # Note: If your environment isn't providing the custom shaped reward we discussed,
            # you would call `reward, done = compute_custom_reward(next_state, action)` here.

            # 6. Store transition in Replay Buffer
            # (Ensure 'done' is converted to a float: 1.0 for terminal state, 0.0 otherwise)
            done_float = 1.0 if done else 0.0
            replay_buffer.add(state, action, reward, next_state, done_float)
            
            state = next_state
            episode_reward += reward

            # 7. Train the Agent
            # Only start backpropagating after the warm-up phase is complete
            if total_timesteps >= start_timesteps:
                agent.train(replay_buffer, batch_size=batch_size)

            # 8. Handle Episode Termination
            if done:
                break

            print(f"--- timestep {total_timesteps} ---")
            print(f"executed_action: {env_action}")
            print(f"reward: {reward}")
        # Logging progress
        print("="*30)
        print(f"Episode: {episode + 1} | Total Steps: {total_timesteps} | Reward: {episode_reward:.2f}")
        print()
        
        # Save checkpoints every 100 episodes
        if (episode + 1) % 100 == 0:
            torch.save(agent.actor.state_dict(), f"actor_checkpoint_ep{episode+1}.pth")
            print(f"--- Saved Model Checkpoint at Episode {episode + 1} ---")

        # Save trajectories
        logger.save()
        new_ep_idx = get_new_episode_idx(task_demo_path)
        new_filename = os.path.join(task_demo_path, filename_template.format(task_name=task_name, ep_idx=new_ep_idx))
        logger.make_new_rollout(filename=new_filename)
        print(f"New log's file: {logger.filename}\n")

if __name__ == "__main__":
    # Ensure ROS node is initialized if it hasn't been already
    train_ddpg()