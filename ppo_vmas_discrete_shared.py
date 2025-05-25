import argparse
import os
import random
import time
from distutils.util import strtobool

# import gym # Not strictly needed for VMAS
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import vmas
from tqdm import tqdm

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str,
        default=os.path.basename(__file__).rstrip(".py") + "_mlp", # Added _mlp to distinguish
        help="the name of this experiment")
    parser.add_argument("--scenario", type=str, default="navigation",
        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=1e-4, # Adjusted from 3e-5 for MLP, can be tuned
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=50_000_000,
        help="total timesteps (frames) for the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)),
        default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)),
        default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)),
        default=False, nargs="?", const=True,
        help="if toggled, track with Weights & Biases")
    parser.add_argument("--wandb-project-name", type=str,
        default="graph-ml-projects", # Consider changing for MLP experiments
        help="Weights & Biases project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="Weights & Biases entity/team")

    # GNN specific arguments (will not be used by MLP agent but kept for arg parser consistency if needed)
    parser.add_argument("--agg-type", type=str, default="gcn",
        help="gcn, mean, maxpool (NOT USED FOR MLP)")
    parser.add_argument("--hidden-dim", type=int, default=64, # MLP has 256 hardcoded
        help="dim of communicated observation (NOT USED FOR MLP)")
    parser.add_argument("--dist", type=float, default=1.0,
        help="maximum comminicable distance (NOT USED FOR MLP)")
    
    # Algorithm specific arguments
    parser.add_argument("--num-agents", type=int, default=4,
        help="number of agents in the environment")
    parser.add_argument("--num-envs", type=int, default=600,
        help="number of parallel envs per worker")
    parser.add_argument("--num-steps", type=int, default=100,
        help="number of steps per env per rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)),
        default=True, nargs="?", const=False, # Consider True if LR needs careful tuning
        help="Toggle learning rate annealing")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)),
        default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="discount factor γ")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="GAE λ")
    parser.add_argument("--num-minibatches", type=int, default=45, # Adjusted from 45, makes minibatch size larger
        help="number of PPO minibatch passes per epoch")
    parser.add_argument("--update-epochs", type=int, default=4, # Adjusted from 3
        help="number of PPO update epochs")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)),
        default=True, nargs="?", const=True,
        help="Toggles advantage normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="PPO clip coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)),
        default=True, nargs="?", const=True,
        help="Toggles clipped value loss")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="entropy bonus coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="value loss coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="max gradient norm")
    parser.add_argument("--target-kl", type=float, default=None,
        help="PPO target KL (early stop)")

    args = parser.parse_args()
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    # fmt: on
    return args

def make_env(scenario, num_envs, continuous_actions, seed, device, n_agents):
    return vmas.make_env(
        scenario=scenario,
        num_envs=num_envs,
        device=device,
        continuous_actions=continuous_actions,
        seed=seed,
        n_agents=n_agents,
        max_steps=args.num_steps, # Using args.num_steps for env max_steps
    )

def add_agent_id(obs_flat_batch, num_envs, num_agents, device, agent_obs_dim):
    # obs_flat_batch: [num_envs * num_agents, agent_obs_dim]
    agent_ids = torch.eye(num_agents, device=device)  # [num_agents, num_agents]
    # Repeat for each env, then flatten correctly
    # Target shape for agent_ids to append: [num_envs * num_agents, num_agents]
    agent_ids_repeated = agent_ids.unsqueeze(0).expand(num_envs, -1, -1).reshape(num_envs * num_agents, num_agents)
    return torch.cat([obs_flat_batch, agent_ids_repeated], dim=-1)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module): # The new MLP Agent
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)), # Added one more layer to match critic depth
            nn.Tanh(),
            layer_init(nn.Linear(256, act_dim), std=0.01),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
    
    def get_value(self, x):
        # x shape: [batch_of_agent_observations, obs_dim]
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        # x shape: [batch_of_agent_observations, obs_dim]
        logits = self.actor(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(x)

    def get_action(self, x, deterministic=True): # Added for completeness, if needed for eval
        # x shape: [batch_of_agent_observations, obs_dim]
        logits = self.actor(x)
        if deterministic:
            return logits.argmax(dim=-1)
        dist = Categorical(logits=logits)
        return dist.sample()
        
if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.scenario}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True, # VMAS is not a gym env in the traditional sense for this
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = make_env(
        scenario=args.scenario,
        num_envs=args.num_envs,
        continuous_actions=False,
        seed=args.seed,
        device=device,
        n_agents=args.num_agents,
    )

    initial_obs_list = envs.reset() 
    obs_dim_agent_native = initial_obs_list[0].shape[-1] # Native obs dim for one agent from env
    agent_input_obs_dim = obs_dim_agent_native + args.num_agents # After appending agent ID
    act_dim_agent = envs.action_space[0].n
    
    # Initialize the new MLP agent
    agent = Agent(obs_dim=agent_input_obs_dim, act_dim=act_dim_agent).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Storage setup
    obs_storage = torch.zeros((args.num_steps, args.num_envs, args.num_agents, agent_input_obs_dim)).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs * args.num_agents)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs * args.num_agents)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs * args.num_agents)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs * args.num_agents)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs * args.num_agents)).to(device)
    
    global_step = 0
    SAVE_INTERVAL = 10_000_000 
    next_save_step = SAVE_INTERVAL 

    current_episode_returns = np.zeros(args.num_envs, dtype=np.float32)
    current_episode_lengths = np.zeros(args.num_envs, dtype=np.int32)

    start_time = time.time()
    
    # Initial reset and observation 
    next_obs_list_from_env = envs.reset(seed=args.seed) 
    next_obs_stacked_agents = torch.stack(next_obs_list_from_env, dim=1).to(device)
    # [num_envs * num_agents, obs_dim_agent_native]
    next_obs_flat_for_id = next_obs_stacked_agents.reshape(-1, obs_dim_agent_native)
    # Add agent IDs: [num_envs * num_agents, agent_input_obs_dim]
    next_obs_with_id_flat = add_agent_id(next_obs_flat_for_id, args.num_envs, args.num_agents, device, obs_dim_agent_native)
    # [num_envs, num_agents, agent_input_obs_dim]
    next_agent_obs_structured = next_obs_with_id_flat.reshape(args.num_envs, args.num_agents, agent_input_obs_dim)
    
    next_done_per_agent = torch.zeros(args.num_envs, args.num_agents, device=device)
    
    agent_steps_per_rollout = args.num_envs * args.num_steps * args.num_agents
    num_updates = args.total_timesteps // agent_steps_per_rollout

    for update in range(1, num_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        completed_episode_returns = []
        completed_episode_lengths = []

        for step in range(0, args.num_steps):
            global_step += args.num_envs * args.num_agents 
            
            obs_storage[step] = next_agent_obs_structured # Store [E, A, D]
            dones[step] = next_done_per_agent.flatten() 

            with torch.no_grad():
                # Agent expects flat input: [batch_size_agents, obs_dim]
                # next_agent_obs_structured is [E, A, D], reshape to [E*A, D]
                current_obs_flat_for_agent = next_agent_obs_structured.reshape(-1, agent_input_obs_dim)
                action, logprob, entropy, value = agent.get_action_and_value(current_obs_flat_for_agent)

            values[step] = value.flatten() 
            actions[step] = action         
            logprobs[step] = logprob       

            action_array = action.reshape(args.num_envs, args.num_agents).cpu().numpy()
            action_list_for_env_step = [action_array[:, i] for i in range(args.num_agents)]
            
            # VMAS step
            next_obs_list_from_env, reward_list_from_env, done_from_env_signal, info = envs.step(action_list_for_env_step)
            
            reward_tensor = torch.stack(reward_list_from_env, dim=1).to(device) 
            rewards[step] = reward_tensor.flatten() 

            env_reward_sum = reward_tensor.sum(dim=1).cpu().numpy()
            current_episode_returns += env_reward_sum
            current_episode_lengths += 1

            # Prepare next observation for the agent
            next_obs_stacked_agents_env = torch.stack(next_obs_list_from_env, dim=1).to(device)
            next_obs_flat_for_id_env = next_obs_stacked_agents_env.reshape(-1, obs_dim_agent_native)
            next_obs_with_id_flat_env = add_agent_id(next_obs_flat_for_id_env, args.num_envs, args.num_agents, device, obs_dim_agent_native)
            next_agent_obs_structured = next_obs_with_id_flat_env.reshape(args.num_envs, args.num_agents, agent_input_obs_dim)
            
            next_done_per_agent = done_from_env_signal.to(device).unsqueeze(1).repeat(1, args.num_agents)

            for i in range(args.num_envs):
                if done_from_env_signal[i]: # Use the per-environment done signal
                    completed_episode_returns.append(current_episode_returns[i])
                    completed_episode_lengths.append(current_episode_lengths[i])
                    current_episode_returns[i] = 0
                    current_episode_lengths[i] = 0
        
        if completed_episode_returns:
            avg_episode_return = np.mean(completed_episode_returns)
            avg_episode_length = np.mean(completed_episode_lengths)
            print(f"Global Step: {global_step}, Avg Episodic Return: {avg_episode_return}, Avg Episodic Length: {avg_episode_length}")
            writer.add_scalar("charts/avg_episodic_return", avg_episode_return, global_step)
            writer.add_scalar("charts/avg_episodic_length", avg_episode_length, global_step)

        with torch.no_grad():
            # Bootstrap value using the next state observations
            next_value_input_flat = next_agent_obs_structured.reshape(-1, agent_input_obs_dim)
            next_value_agents = agent.get_value(next_value_input_flat).flatten()
            
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device) 
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done_per_agent.flatten().float() 
                        nextvalues_gae = next_value_agents
                    else:
                        nextnonterminal = 1.0 - dones[t + 1].float() 
                        nextvalues_gae = values[t + 1] 
                    delta = rewards[t] + args.gamma * nextvalues_gae * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                # Standard advantage (not GAE)
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done_per_agent.flatten().float()
                        next_return_val = next_value_agents
                    else:
                        nextnonterminal = 1.0 - dones[t+1].float() # dones are already flat
                        next_return_val = returns[t+1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return_val
                advantages = returns - values # Calculate advantages after computing returns

        # Flatten batch for PPO
        num_env_steps_in_buffer = args.num_envs * args.num_steps
        
        # obs_storage is [Steps, Envs, Agents, ObsDim]
        # Permute to [Envs, Steps, Agents, ObsDim] -> [EnvSteps, Agents, ObsDim]
        b_agent_obs_structured = obs_storage.permute(1, 0, 2, 3).reshape(num_env_steps_in_buffer, args.num_agents, agent_input_obs_dim)

        b_actions_flat = actions.reshape(-1) 
        b_logprobs_flat = logprobs.reshape(-1)
        b_advantages_flat = advantages.reshape(-1)
        b_returns_flat = returns.reshape(-1)
        b_values_flat = values.reshape(-1) 

        env_step_indices_for_ppo_batch = np.arange(num_env_steps_in_buffer)
        clipfracs = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(env_step_indices_for_ppo_batch)
            for start in range(0, num_env_steps_in_buffer, args.minibatch_size): # minibatch_size is env_steps
                end = start + args.minibatch_size
                mb_env_step_inds = env_step_indices_for_ppo_batch[start:end] 

                current_mb_agent_obs_structured = b_agent_obs_structured[mb_env_step_inds]
                current_mb_obs_flat_for_agent = current_mb_agent_obs_structured.reshape(-1, agent_input_obs_dim)
                
                # Select corresponding actions, logprobs, etc.
                mb_agent_actions = b_actions_flat.view(num_env_steps_in_buffer, args.num_agents)[mb_env_step_inds].reshape(-1).long()
                mb_agent_logprobs_old = b_logprobs_flat.view(num_env_steps_in_buffer, args.num_agents)[mb_env_step_inds].reshape(-1)
                mb_agent_advantages = b_advantages_flat.view(num_env_steps_in_buffer, args.num_agents)[mb_env_step_inds].reshape(-1)
                mb_agent_returns = b_returns_flat.view(num_env_steps_in_buffer, args.num_agents)[mb_env_step_inds].reshape(-1)
                mb_agent_values_old = b_values_flat.view(num_env_steps_in_buffer, args.num_agents)[mb_env_step_inds].reshape(-1)
                
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    current_mb_obs_flat_for_agent, # Pass flattened obs
                    action=mb_agent_actions 
                )
                newvalue = newvalue.flatten()

                logratio = newlogprob - mb_agent_logprobs_old
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    if mb_agent_actions.numel() > 0: # Avoid division by zero if minibatch is empty
                         clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]


                current_mb_advantages = mb_agent_advantages
                if args.norm_adv:
                    current_mb_advantages = (current_mb_advantages - current_mb_advantages.mean()) / (current_mb_advantages.std() + 1e-8)

                pg_loss1 = -current_mb_advantages * ratio
                pg_loss2 = -current_mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - mb_agent_returns) ** 2
                    v_clipped = mb_agent_values_old + torch.clamp(
                        newvalue - mb_agent_values_old,
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - mb_agent_returns) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - mb_agent_returns) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                print(f"Early stopping at epoch {epoch+1} due to reaching target KL {approx_kl.item():.4f} > {args.target_kl:.4f}")
                break
        
        y_pred_old, y_true_old = b_values_flat.cpu().numpy(), b_returns_flat.cpu().numpy()
        var_y_old = np.var(y_true_old)
        explained_var = np.nan if var_y_old == 0 else 1 - np.var(y_true_old - y_pred_old) / var_y_old

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        if clipfracs: # Ensure clipfracs is not empty
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        current_sps = int(global_step / (time.time() - start_time)) if (time.time() - start_time) > 0 else 0
        print(f"SPS: {current_sps}")
        writer.add_scalar("charts/SPS", current_sps, global_step)
        
        if global_step >= next_save_step : 
            torch.save(agent.state_dict(), f"{run_name}_{global_step}.pth")
            print(f"SAVE MODEL in global_step = {global_step}")
            next_save_step += SAVE_INTERVAL

    torch.save(agent.state_dict(), f"{run_name}_final.pth") 
    
    writer.close()
    envs.close() # Close the VMAS environments