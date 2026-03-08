import os
import math
import rospy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from geometry_msgs.msg import PoseStamped
from sawyer_control.envs.sawyer_pickplace import SawyerPickPlaceXYZYawEnv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 1. Neural Network Architectures
# ==========================================
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.q1_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

        self.q2_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x), self.q2_model(x)

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device_local = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device_local) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class TwoStepMLP(nn.Module):
    def __init__(self, state_dim, action_dim, device, t_dim=16, r_dim=16):
        super(TwoStepMLP, self).__init__()
        self.device = device
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )
        self.r_mlp = nn.Sequential(
            SinusoidalPosEmb(r_dim),
            nn.Linear(r_dim, r_dim * 2),
            nn.Mish(),
            nn.Linear(r_dim * 2, r_dim),
        )
        input_dim = state_dim + action_dim + t_dim + r_dim
        self.mid_layer = nn.Sequential(nn.Linear(input_dim, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish())
        self.final_layer = nn.Linear(256, action_dim)

    def forward(self, x, time, r_time, state):
        t = self.time_mlp(time)
        r = self.r_mlp(r_time)
        x = torch.cat([x, t, r, state], dim=1)
        x = self.mid_layer(x)
        return self.final_layer(x)

# ==========================================
# 2. Sampling Logic (One-Step Diffusion + Q-Filter)
# ==========================================

def sample_action(model, state_tensor, action_dim, device='cuda', noise_clip=0.5):
    batch_size = state_tensor.shape[0]
    t_time = torch.full((batch_size,), 1, device=device, dtype=torch.long)
    r_time = torch.full((batch_size,), 0, device=device, dtype=torch.long)
    
    # Generate initial noise
    noise = torch.randn((batch_size, action_dim)).to(device)
    
    # CROP/CLAMP the noise to limit the randomness
    # This keeps the values within [-noise_clip, noise_clip]
    noise = torch.clamp(noise, -noise_clip, noise_clip)

    with torch.no_grad():
        # The model predicts the noise component to be subtracted
        action = noise - model(noise, t_time, r_time, state_tensor)
        
    return action

def sample_action_with_q(model, critic, state_tensor, action_dim, num_samples=50,noise_clip=0.5):
    # Duplicate the state 50 times to create a batch
    aug_state_tensor = torch.repeat_interleave(state_tensor, repeats=num_samples, dim=0)
    batch_size = aug_state_tensor.shape[0]

    # Time embeddings for the InstantFlow model
    t_time = torch.full((batch_size,), 1, device=device, dtype=torch.long)
    r_time = torch.full((batch_size,), 0, device=device, dtype=torch.long)
    
    # Generate 50 random noise vectors
    noise = torch.randn((num_samples, action_dim), device=device)

    # CROP/CLAMP the noise to limit the randomness
    # This keeps the values within [-noise_clip, noise_clip]
    noise = torch.clamp(noise, -noise_clip, noise_clip)

    
    with torch.no_grad():
        # Denoise all 50 samples in one step
        actions = noise - model(noise, t_time, r_time, aug_state_tensor)
        
        # Evaluate all 50 actions using the Critic
        q_values = critic.q_min(aug_state_tensor, actions).flatten()
        
        # Pick the action with the highest Q-value probabilistically using Softmax
        idx = torch.multinomial(F.softmax(q_values, dim=0), 1)
        
    return actions[idx]

# ==========================================
# 3. The StatesLogger
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
            return None
        gripper_pos = self.latest_poses["gripper"]
        target_pos = self.latest_poses[target_object]
        rel_distance = target_pos - gripper_pos
        gripper_state = self._get_current_gripper_state(env)
        return np.concatenate([gripper_pos, target_pos, rel_distance, [gripper_state]]).astype(np.float32)

# ==========================================
# 4. Evaluation Loop
# ==========================================
def test_instantflow_policy():
    # --- Configuration ---
    task_name = 'sawyer-move-box-v0'
    
    if task_name == 'sawyer-open-drawer-v0':
        target_object = "upper_drawer"
    elif task_name == 'sawyer-move-box-v0':
        target_object = "red_box"
    elif task_name == 'sawyer-pick-lift-banana-v0':
        target_object = "banana" 

    # Paths to your trained InstantFlow models
    base_model_dir = "/home/tung/workspace/rlhf_bench/iql-pytorch-sawyer/checkpoints/sawyer-pick-lift-banana-v0/InsQL/"
    model_load_path = os.path.join(base_model_dir, "actor.pth")
    critic_load_path = os.path.join(base_model_dir, "critic.pth")
    
    num_test_episodes = 10
    max_steps = 150
    STATE_DIM = 10
    ACTION_DIM = 5

    print(f"--- Loading Environment: {task_name} ---")
    env = SawyerPickPlaceXYZYawEnv(task_name=task_name)
    states_logger = StatesLogger()

    # --- Initialize Models ---
    print(f"--- Loading InstantFlow Models ---")
    model = TwoStepMLP(state_dim=STATE_DIM, action_dim=ACTION_DIM, device=device).to(device)
    critic = Critic(state_dim=STATE_DIM, action_dim=ACTION_DIM).to(device)

    # Load Actor/Model weights
    actor_state_dict = torch.load(model_load_path)
    # Extract only the TwoStepMLP weights (strip the 'model.' prefix)
    model_state_dict = {k.replace('model.', ''): v for k, v in actor_state_dict.items() if k.startswith('model.')}
    model.load_state_dict(model_state_dict)
    
    # Load Critic weights
    critic.load_state_dict(torch.load(critic_load_path))
    
    model.eval()
    critic.eval()

    print("--- Starting InstantFlow Evaluation ---")
    
    for ep in range(num_test_episodes):
        obs = env.reset()
        rospy.sleep(1.0)
        
        print(f"\n--- Episode {ep+1} Start ---")
        
        for step in range(max_steps):
            # 1. Get current state
            state = states_logger._get_current_state_space(env, target_object)
            if state is None:
                continue
                
            # 2. Predict Action using InstantFlow + Q-Filtering
            state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(device)
            # action_5d = sample_action_with_q(model, critic, state_tensor, ACTION_DIM, num_samples=50)

            action_5d = sample_action(model, state_tensor, ACTION_DIM)
            
            action_5d = action_5d.cpu().data.numpy().flatten()

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
            
            cv2.imshow("Sawyer InstantFlow Evaluation", obs['rgb_image'][:, :, ::-1])
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
    test_instantflow_policy()