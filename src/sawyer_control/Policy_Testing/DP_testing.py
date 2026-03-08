import os
import rospy
import numpy as np
import torch
import time
import hydra
import cv2
import dill
import collections
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from sawyer_control.envs.sawyer_pickplace import SawyerPickPlaceXYZYawEnv

# Stanford Diffusion Policy & DyAC imports
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.dynamic_adaptive_chunk import DynamicAdaptiveChunk
from diffusion_policy.model.dynamics.dynamics_model import DynamicsModel
from diffusion_policy.sampler.single import sgac_sampler, ac_sampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. The StatesLogger (Unchanged)
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
def test_stanford_diffusion_policy():
    pub = rospy.Publisher('/activate_record_video', String, queue_size=10)
    # --- Configuration ---
    task_name = 'sawyer-move-box-v0'
    
    if task_name == 'sawyer-open-drawer-v0':
        target_object = "upper_drawer"
    elif task_name == 'sawyer-move-box-v0':
        target_object = "red_box"
    elif task_name == 'sawyer-pick-lift-banana-v0':
        target_object = "banana" 

    checkpoint_name = "2026.03.02_14.51.38"
    policy_ckpt = f"/home/tung/workspace/rlhf_bench/iql-pytorch-sawyer/checkpoints/{task_name}/diffusion-policy/{checkpoint_name}_epoch_0200.ckpt"
    dynamics_ckpt_path = f"/home/tung/workspace/rlhf_bench/iql-pytorch-sawyer/checkpoints/{task_name}/dynamics/epoch_200.pth"
    
    num_test_episodes = 10
    max_steps = 150
    
    # --- DyAC Configuration ---
    sampler = "dp"
    use_dyac = False
    beta = 0.0003 # Tolerance multiplier for replanning

    output_dir = f"/home/tung/workspace/rlhf_bench/iql-pytorch-sawyer/output_video/{task_name}/diffusion-policy/{checkpoint_name}/epoch_200/{sampler}_{use_dyac}_{beta}"    
    os.makedirs(output_dir,exist_ok=True)

    print(f"--- Loading Environment: {task_name} ---")
    env = SawyerPickPlaceXYZYawEnv(task_name=task_name)
    states_logger = StatesLogger()

    # --- Initialize Stanford Diffusion Policy ---
    print(f"--- Loading Model from {policy_ckpt} ---")
    payload = torch.load(policy_ckpt, pickle_module=dill)
    
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir="output")

    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.model
    policy.to(device)
    policy.eval()

    obs_horizon = cfg.n_obs_steps if hasattr(cfg, 'n_obs_steps') else 2
    n_action_steps = cfg.n_action_steps if hasattr(cfg, 'n_action_steps') else 8
    n_action_steps = 1
    print(f"N_Action_Steps = {n_action_steps}")
    

    # --- Load forward dynamics model ---
    fwd_model = None 
    if use_dyac:
        print(f"--- [Notice] DyAC is ENABLED. Loading forward dynamics model from {dynamics_ckpt_path} ---")
        fwd_model = DynamicsModel(
            action_dim=policy.action_dim,
            action_horizon=policy.n_action_steps,
            state_dim=policy.obs_dim,
            state_horizon=1,
            n_head=8,
            n_enc_layers=4,
            pe_learnable=False,
            pdrop=0.1,
            embedding_dim=384,
        )
        dynamics_ckpt = torch.load(dynamics_ckpt_path)
        fwd_model.load_state_dict(dynamics_ckpt['forward_model'])
        fwd_model.to(device)
        fwd_model.eval()

    print(f"--- Starting Evaluation (Obs Horizon: {obs_horizon}, Action Horizon: {n_action_steps}) ---")
    
    for ep in range(num_test_episodes):
        start_time = time.time()
        obs = env.reset()
        rospy.sleep(1.0)
        
        obs_deque = collections.deque(maxlen=obs_horizon)
        pub.publish("0")
        
        # Initialize DyAC for a single environment (n_envs=1)
        dyac = None
        replanning_mask = None
        if use_dyac:
            dyac = DynamicAdaptiveChunk(n_envs=1, ah=n_action_steps, task='robomimic')
        
        print(f"\n--- Episode {ep+1} Start ---")
        pub.publish(os.path.join(output_dir,f"episode_{ep+11}.mp4"))
        step = 0
        
        while step < max_steps:
            state = states_logger._get_current_state_space(env, target_object)
            if state is None:
                continue
                
            obs_deque.append(state)
            while len(obs_deque) < obs_horizon:
                obs_deque.append(state)

            obs_seq = np.stack(list(obs_deque))
            obs_tensor = torch.tensor(obs_seq, dtype=torch.float32, device=device).unsqueeze(0)
            obs_dict = {
                'obs': obs_tensor
            }
            prev_obs_dict = {
                'obs': obs_seq[:,-3:-1].astype(np.float32)
            }
            
            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict, obs_dict)
                action_tensor = action_dict['action']
                action_numpy = action_tensor.cpu().numpy()
            with torch.no_grad():
                if sampler == 'dp':
                    action_dict = policy.predict_action(obs_dict, obs_dict)
                elif sampler == 'sgac':
                    if 'action_prior' not in locals():
                        action_prior = None
                        action_dict = sgac_sampler(policy, action_prior, obs_dict, obs_dict, beta)
                    else:                                                   
                        action_dict = sgac_sampler(policy, action_prior, obs_dict, prev_obs_dict, beta)
                    action_prior = action_dict['action_pred'][:, n_action_steps:]   
                elif sampler == 'ac':
                    if 'action_prior' not in locals():
                        action_prior = None
                        action_dict = ac_sampler(policy, action_prior, obs_dict, obs_dict, beta)
                    else:                                                   
                        action_dict = ac_sampler(policy, action_prior, obs_dict, prev_obs_dict, beta)
                    action_prior = action_dict['action_pred'][:, n_action_steps:]        
                elif sampler == 'sg':
                    action_dict = policy.predict_action(obs_dict, prev_obs_dict)
                else:
                    raise NotImplementedError
            
            pred_states = None
            if use_dyac:
                # Get State Predictions from Forward Model
                with torch.no_grad():
                    cur_s = obs_tensor[:, -1:, :] # Current state
                    with torch.no_grad():
                        pred_s_tensor = fwd_model.get_prediction(
                            cur_s, 
                            action_tensor, 
                            residual=False, 
                            use_model_normalizer=True
                        ) # [B, n_action_steps, Do]
                        pred_states = pred_s_tensor.detach().cpu().numpy()
                
            # Execute action chunk
            if use_dyac:
                # Update DyAC buffers
                if dyac.first_time:
                    dyac.update_action_chunk(action_numpy, pred_states, replanning_mask=None)
                else:
                    dyac.update_action_chunk(action_numpy, pred_states, replanning_mask=replanning_mask)

                while True:
                    # Extract single action for our single env (index 0)
                    action_3d = dyac.get_action()[0, 0]
                
                    # Apply constraints
                    if state[2] <= -0.285 and action_3d[2] < 0.0: action_3d[2] = 0.0 
                
                    gripper_cmd = 1.0 if task_name == 'sawyer-open-drawer-v0' else 0.0
                    if task_name == 'sawyer-pick-lift-banana-v0':
                        gripper_cmd = 0.0 if abs(action_3d[-1]) < 0.5 else 1.0
                    
                    env_action = np.array([action_3d[0], action_3d[1], action_3d[2], 0.0, gripper_cmd])
                
                    # Step environment
                    obs, reward, done, info = env.step(env_action, state)
                    step += 1
                
                    # Get new state to calculate error mask
                    state = states_logger._get_current_state_space(env, target_object)
                    obs_deque.append(state) # Keep history updated during inner loop!
                    
                    # Calculate if we need to replan based on state divergence
                    state_batch = np.expand_dims(state, axis=(0, 1))
                    replanning_mask = dyac.compute_mask_to_replan(state_batch, np.array([reward]), [info], np.array([done]), beta)
                    
                    cv2.imshow("Sawyer DP + DyAC", obs['rgb_image'][:, :, ::-1])
                    key = cv2.waitKey(1) & 0xFF
                    
                    # Break inner loop if replanning is needed, or if task is done/interrupted
                    if replanning_mask.any() or done or key == ord('i') or key == ord('q'):
                        break
            else:
                for a_idx in range(n_action_steps):
                    # Just take the first step of the chunk
                    action_3d = action_numpy[0][a_idx]
                    
                    # Apply constraints
                    if state[2] <= -0.285 and action_3d[2] < 0.0: action_3d[2] = 0.0 
                    
                    gripper_cmd = 1.0 if task_name == 'sawyer-open-drawer-v0' else 0.0
                    if task_name == 'sawyer-pick-lift-banana-v0':
                        gripper_cmd = 0.0 if abs(action_3d[-1]) < 0.5 else 1.0
                        
                    env_action = np.array([action_3d[0], action_3d[1], action_3d[2], 0.0, gripper_cmd])
                    
                    # Step environment
                    obs, reward, done, info = env.step(env_action, state)
                    step += 1
                    
                    cv2.imshow("Sawyer DP Evaluation", obs['rgb_image'][:, :, ::-1])
                    key = cv2.waitKey(1) & 0xFF
            
            # --- Handle Exits & Completions ---
            if key == ord('i'):
                print(f" >>> [INTERVENED] Skipping Episode {ep+1}...")
                break 
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return

            if done:
                print(f"Episode {ep+1} Success at step {step}!")
                break

        print(f"TIME COMPLETION: {time.time()-start_time}\n")
                
        if not done and step >= max_steps:
            print(f"Episode {ep+1} timed out.")
            
        # Print DyAC stats at the end of the episode if used
        if use_dyac and dyac.total_steps_taken > 0:
            replan_ratio = dyac.total_replan_by_error / dyac.total_steps_taken
            print(f"[DyAC Ep {ep+1}] Replan Ratio: {replan_ratio*100:.2f}% | Max Error: {dyac.mse_max:.4f}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_stanford_diffusion_policy()