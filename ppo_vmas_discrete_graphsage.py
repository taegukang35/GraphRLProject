import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import vmas
from tqdm import tqdm

from gnn_utils import GraphSageLayer, build_edge_lists

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str,
        default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--scenario", type=str, default="navigation",
        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
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
        default="graph-ml-projects",
        help="Weights & Biases project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="Weights & Biases entity/team")

    # GNN specific arguments
    parser.add_argument("--agg-type", type=str, default="gcn",
        help="gcn, mean, maxpool")
    parser.add_argument("--hidden-dim", type=int, default=64,
        help="dim of communicated observation")
    parser.add_argument("--dist", type=float, default=1.0,
        help="maximum comminicable distance")
    
    # Algorithm specific arguments
    parser.add_argument("--num-agents", type=int, default=4,
        help="number of agents in the environment")
    parser.add_argument("--num-envs", type=int, default=600,
        help="number of parallel envs per worker (=> 600*100=60k frames/batch)")
    parser.add_argument("--num-steps", type=int, default=100,
        help="number of steps per env per rollout (100 => 60k frames)")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)),
        default=True, nargs="?", const=False,
        help="Toggle learning rate annealing")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)),
        default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="discount factor γ")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="GAE λ")
    parser.add_argument("--num-minibatches", type=int, default=45,
        help="number of PPO minibatch passes per epoch")
    parser.add_argument("--update-epochs", type=int, default=4,
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
    # total batch size = num_envs * num_steps = 600 * 100 = 60_000
    args.batch_size = args.num_envs * args.num_steps
    # minibatch size per update = batch_size // num_minibatches = 60_000 / 45 ≈ 1333
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
        max_steps=100,
    )

def add_agent_id(obs, num_envs, num_agents, device):
    agent_ids = torch.eye(num_agents, device=device)  # [num_agents, num_agents]
    agent_ids = agent_ids.unsqueeze(0).repeat(num_envs, 1, 1)  # [num_envs, num_agents, num_agents]
    agent_ids = agent_ids.view(-1, num_agents)  # [num_envs * num_agents, num_agents]
    return torch.cat([obs, agent_ids], dim=-1)  # [num_envs * num_agents, obs_dim + num_agents]

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class GraphSageAgent(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim, dist=1.0, agg_type='gcn'):
        super().__init__()
        self.dist = dist
        # GraphSAGE policy
        self.gsage1 = GraphSageLayer(obs_dim, hidden_dim, agg_type=agg_type)
        self.gsage2 = GraphSageLayer(hidden_dim, hidden_dim, agg_type=agg_type)
        self.policy_head = layer_init(nn.Linear(hidden_dim, act_dim), std=0.01)
        # Critic: MLP
        self.critic = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0)
        )

    def get_communicated_obs(self, x: torch.Tensor, positions: torch.Tensor):
        """
        x: [num_envs, num_agents, obs_dim]
        positions: [num_envs, num_agents, pos_dim]  (e.g. pos_dim=2)
        """
        device = x.device
        num_envs, n_agents, feat_dim = x.shape

        x_flat = x.reshape(-1, feat_dim)  # Shape: [num_envs * n_agents, feat_dim]
        batch_idx = torch.arange(num_envs, device=device).repeat_interleave(n_agents * n_agents)
        dist_mat = torch.cdist(positions, positions, p=2) # [num_envs, n_agents, n_agents]
        mask = dist_mat <= self.dist  # [num_envs, n_agents, n_agents]
        
        # Create edge list
        adj = mask.to(torch.int)
        env_idx, src_idx, dst_idx = adj.nonzero(as_tuple=True)
        
        # Offset indices to be unique across the flattened batch
        src_flat = env_idx * n_agents + src_idx
        dst_flat = env_idx * n_agents + dst_idx
        edges = torch.stack([src_flat, dst_flat], dim=1).to(device)
        
        N_total_nodes = num_envs * n_agents
        degree = torch.bincount(dst_flat, minlength=N_total_nodes).to(device)
        
        h = self.gsage1(x_flat, edges, degree)
        h = self.gsage2(h, edges, degree) # Shape: [num_envs * n_agents, hidden_dim]

        return h

    def get_value(self, raw_obs_batch: torch.Tensor, positions_batch: torch.Tensor) -> torch.Tensor:
        # raw_obs_batch: [num_envs_in_batch, num_agents, raw_obs_dim (obs_dim + num_agents)]
        # positions_batch: [num_envs_in_batch, num_agents, pos_dim]
        communicated_obs = self.get_communicated_obs(raw_obs_batch, positions_batch)
        # communicated_obs shape: [num_envs_in_batch * num_agents, hidden_dim]
        return self.critic(communicated_obs) # Output shape: [num_envs_in_batch * num_agents, 1]

    def get_action_and_value(self, raw_obs_batch: torch.Tensor, positions_batch: torch.Tensor, action:torch.Tensor=None):
        # raw_obs_batch: [num_envs_in_batch, num_agents, raw_obs_dim]
        # positions_batch: [num_envs_in_batch, num_agents, pos_dim]
        communicated_obs = self.get_communicated_obs(raw_obs_batch, positions_batch)
        # communicated_obs shape: [num_envs_in_batch * num_agents, hidden_dim]
        
        logits = self.policy_head(communicated_obs) # Shape: [num_envs_in_batch * num_agents, act_dim]
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample() # Shape: [num_envs_in_batch * num_agents]
        
        log_probs = dist.log_prob(action) # Shape: [num_envs_in_batch * num_agents]
        entropy = dist.entropy()         # Shape: [num_envs_in_batch * num_agents]
        value = self.critic(communicated_obs) # Shape: [num_envs_in_batch * num_agents, 1]
        
        return action, log_probs, entropy, value

    def get_action(self, x, deterministic=True):

        pass
        
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
            monitor_gym=True,
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

    # env setup
    envs = make_env(
        scenario=args.scenario,
        num_envs=args.num_envs,
        continuous_actions=False, # only consider discrete action space
        seed=args.seed,
        device=device,
        n_agents=args.num_agents,
    )

    # check dim of env
    # Reset envs to get obs_dim and act_dim
    initial_obs_list = envs.reset() 
    obs_dim = initial_obs_list[0].shape[-1] # obs_dim for a single agent
    act_dim = envs.action_space[0].n
    
    agent = GraphSageAgent(obs_dim=obs_dim + args.num_agents, act_dim=act_dim, hidden_dim=args.hidden_dim, dist=args.dist, agg_type=args.agg_type).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Storage setup
    raw_obs_storage = torch.zeros((args.num_steps, args.num_envs, args.num_agents, obs_dim + args.num_agents)).to(device)
    positions_storage = torch.zeros((args.num_steps, args.num_envs, args.num_agents, 2)).to(device) # Assuming 2D positions
    actions = torch.zeros((args.num_steps, args.num_envs * args.num_agents)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs * args.num_agents)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs * args.num_agents)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs * args.num_agents)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs * args.num_agents)).to(device)
    
    # start the game
    global_step = 0
    SAVE_INTERVAL = 10_000_000 # You can adjust this
    next_save_step = SAVE_INTERVAL # Changed variable name for clarity

    # Episode tracking
    current_episode_returns = np.zeros(args.num_envs, dtype=np.float32)
    current_episode_lengths = np.zeros(args.num_envs, dtype=np.int32)

    start_time = time.time()
    # Initial reset and observation processing
    next_obs_list = envs.reset(seed=args.seed) # List of obs_per_agent, each [num_envs, obs_dim_agent]
    next_obs_stacked = torch.stack(next_obs_list, dim=1).to(device) # [num_envs, num_agents, obs_dim]
    next_obs_flat_for_id = next_obs_stacked.reshape(-1, obs_dim) # [num_envs * num_agents, obs_dim]
    next_obs_with_id_flat = add_agent_id(next_obs_flat_for_id, args.num_envs, args.num_agents, device)
    next_gnn_obs = next_obs_with_id_flat.reshape(args.num_envs, args.num_agents, obs_dim + args.num_agents)

    current_positions = torch.zeros(args.num_envs, args.num_agents, 2, device=device) # Assuming 2D positions
    for e_idx in range(args.num_envs): # Initial positions
        env_agent_pos_list = []
        for agent_idx_in_env in range(args.num_agents):
            env_agent_pos_list.append(envs.agents[agent_idx_in_env].state.pos[e_idx])
        current_positions[e_idx] = torch.stack(env_agent_pos_list, dim=0)
    
    next_done = torch.zeros(args.num_envs, args.num_agents, device=device) # [num_envs, num_agents]
    agent_steps_per_rollout = args.num_envs * args.num_steps * args.num_agents
    num_updates = args.total_timesteps // agent_steps_per_rollout


    for update in range(1, num_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # Temporary lists to store returns and lengths of completed episodes in this rollout
        completed_episode_returns = []
        completed_episode_lengths = []

        for step in range(0, args.num_steps):
            # global_step should count agent steps
            global_step += args.num_envs * args.num_agents # Increment by total agent steps in this env step
            
            raw_obs_storage[step] = next_gnn_obs
            positions_storage[step] = current_positions
            dones[step] = next_done.flatten() 

            with torch.no_grad():
                action, logprob, entropy, value = agent.get_action_and_value(next_gnn_obs, current_positions)

            values[step] = value.flatten() 
            actions[step] = action         
            logprobs[step] = logprob       

            action_array = action.reshape(args.num_envs, args.num_agents).cpu().numpy()
            action_list_for_env = [action_array[:, i] for i in range(args.num_agents)]
            next_obs_list_from_env, reward_list_from_env, done_from_env, info = envs.step(action_list_for_env)
            
            reward_tensor = torch.stack(reward_list_from_env, dim=1).to(device) 
            rewards[step] = reward_tensor.flatten() 

            # Update episodic counts
            env_reward_sum = reward_tensor.sum(dim=1).cpu().numpy()
            current_episode_returns += env_reward_sum
            current_episode_lengths += 1

            next_obs_stacked_env = torch.stack(next_obs_list_from_env, dim=1).to(device)
            next_obs_flat_for_id_env = next_obs_stacked_env.reshape(-1, obs_dim)
            next_obs_with_id_flat_env = add_agent_id(next_obs_flat_for_id_env, args.num_envs, args.num_agents, device)
            next_gnn_obs = next_obs_with_id_flat_env.reshape(args.num_envs, args.num_agents, obs_dim + args.num_agents)

            for e_idx in range(args.num_envs):
                env_agent_pos_list = []
                for agent_idx_in_env in range(args.num_agents):
                     env_agent_pos_list.append(envs.agents[agent_idx_in_env].state.pos[e_idx]) # VMAS specific
                current_positions[e_idx] = torch.stack(env_agent_pos_list, dim=0)
            
            # done_from_env is [num_envs]. If an env is done, all agents in it are considered done for GAE.
            # next_done should be [num_envs, num_agents] for storage and GAE logic
            next_done = done_from_env.to(device).unsqueeze(1).repeat(1, args.num_agents)

            # Check for completed episodes and log them
            for i in range(args.num_envs):
                if done_from_env[i]:
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
            next_value_agents = agent.get_value(next_gnn_obs, current_positions).flatten() 
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device) 
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        # next_done is [num_envs, num_agents] from the last step of the rollout
                        nextnonterminal = 1.0 - next_done.flatten().float() 
                        nextvalues_gae = next_value_agents
                    else:
                        # dones[t+1] is already flat [num_envs*num_agents]
                        nextnonterminal = 1.0 - dones[t + 1].float() 
                        nextvalues_gae = values[t + 1] 
                    delta = rewards[t] + args.gamma * nextvalues_gae * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done.flatten().float()
                        next_return_val = next_value_agents
                    else:
                        nextnonterminal = 1.0 - dones[t+1].float()
                        next_return_val = returns[t+1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return_val
                advantages = returns - values

        num_env_steps_in_buffer = args.num_envs * args.num_steps
        
        b_gnn_obs_input = raw_obs_storage.permute(1, 0, 2, 3).reshape(num_env_steps_in_buffer, args.num_agents, obs_dim + args.num_agents)
        b_gnn_pos_input = positions_storage.permute(1, 0, 2, 3).reshape(num_env_steps_in_buffer, args.num_agents, 2)

        b_actions_flat = actions.reshape(-1) 
        b_logprobs_flat = logprobs.reshape(-1)
        b_advantages_flat = advantages.reshape(-1)
        b_returns_flat = returns.reshape(-1)
        b_values_flat = values.reshape(-1) 

        env_step_indices_for_ppo_batch = np.arange(num_env_steps_in_buffer)

        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(env_step_indices_for_ppo_batch)
            for start in range(0, num_env_steps_in_buffer, args.minibatch_size):
                end = start + args.minibatch_size
                mb_env_step_inds = env_step_indices_for_ppo_batch[start:end] 

                current_mb_raw_obs = b_gnn_obs_input[mb_env_step_inds]
                current_mb_positions = b_gnn_pos_input[mb_env_step_inds]
                
                mb_agent_actions = b_actions_flat.view(num_env_steps_in_buffer, args.num_agents)[mb_env_step_inds].reshape(-1).long()
                mb_agent_logprobs_old = b_logprobs_flat.view(num_env_steps_in_buffer, args.num_agents)[mb_env_step_inds].reshape(-1)
                mb_agent_advantages = b_advantages_flat.view(num_env_steps_in_buffer, args.num_agents)[mb_env_step_inds].reshape(-1)
                mb_agent_returns = b_returns_flat.view(num_env_steps_in_buffer, args.num_agents)[mb_env_step_inds].reshape(-1)
                mb_agent_values_old = b_values_flat.view(num_env_steps_in_buffer, args.num_agents)[mb_env_step_inds].reshape(-1)
                
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    current_mb_raw_obs,
                    current_mb_positions,
                    action=mb_agent_actions 
                )
                newvalue = newvalue.flatten()

                logratio = newlogprob - mb_agent_logprobs_old
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
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
                # Removed the gradient print loop for cleaner output, can be re-added for debugging
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
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
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step) # clipfracs could be empty if update_epochs is 0 or 1
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print(f"SPS: {int(global_step / (time.time() - start_time))}")
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        
        if global_step >= next_save_step : # Check if current global_step has passed the next save point
            torch.save(agent.state_dict(), f"{run_name}_{global_step}.pth")
            print(f"SAVE MODEL in global_step = {global_step}")
            next_save_step += SAVE_INTERVAL

    torch.save(agent.state_dict(), f"{run_name}_final.pth") # Save final model
    
    writer.close()