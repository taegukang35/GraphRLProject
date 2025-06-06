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


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str,
        default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--scenario", type=str, default="navigation",
        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=10_000_000,
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
        default="ppo-implementation-details",
        help="Weights & Biases project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="Weights & Biases entity/team")

    # Algorithm specific arguments
    parser.add_argument("--num-agents", type=int, default=4,
        help="number of agents in the environment")
    parser.add_argument("--num-envs", type=int, default=600,
        help="number of parallel envs per worker (=> 600*100=60k frames/batch)")
    parser.add_argument("--num-steps", type=int, default=100,
        help="number of steps per env per rollout (100 => 60k frames)")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)),
        default=True, nargs="?", const=True,
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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class CentralizedAgent(nn.Module):
    def __init__(self, obs_dim, act_dim, n_agents):
        super().__init__()
        self.n_agents = n_agents

        # --- actor: decentralized, per-agent obs input ---
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, act_dim), std=0.01),
        )

        # --- critic: centralized, global obs input ---
        global_dim = obs_dim * n_agents
        self.critic = nn.Sequential(
            layer_init(nn.Linear(global_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )

    def get_action_and_logprob(self, obs_agent, action=None):
        """
        obs_agent: [batch_size_agents, obs_dim]
        returns:
          action: [batch_size_agents]
          logprob: [batch_size_agents]
          entropy: [batch_size_agents]
        """
        logits = self.actor(obs_agent)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def get_central_value(self, obs_global):
        """
        obs_global: [num_envs, obs_dim * n_agents]
        returns:
          value: [num_envs]  (scalar value per environment)
        """
        return self.critic(obs_global)

    def get_action(self, x, deterministic=True):
        logits = self.actor(x)
        if deterministic:
            return logits.argmax(dim=-1)
        dist = Categorical(logits=logits)
        return dist.sample()
    
        
if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.scenario}__{args.exp_name}__{args.seed}__{int(time.time())}"
    print(run_name)
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
    obs_list = envs.reset()
    obs_dim = obs_list[0].shape[-1]
    act_dim = envs.action_space[0].n
    
    agent = CentralizedAgent(obs_dim=obs_dim, act_dim=act_dim, n_agents=args.num_agents).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs * args.num_agents, obs_dim)).to(device)
    global_obs = torch.zeros((args.num_steps, args.num_envs, obs_dim * args.num_agents)).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs * args.num_agents)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs * args.num_agents)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs * args.num_agents)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs * args.num_agents)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs * args.num_agents)).to(device)

    # start the game
    global_step = 0
    episode_returns = np.zeros(args.num_envs, dtype=np.float32)
    episode_lengths = np.zeros(args.num_envs, dtype=np.int32)
    return_queue = deque(maxlen=100)
    length_queue = deque(maxlen=100)
    start_time = time.time()
    next_obs_list = envs.reset(seed=args.seed) # (num_envs, obs_dim) per agents
    next_obs_stack = torch.stack(next_obs_list, dim=1).to(device) # [num_envs, num_agents, obs_dim]
    next_global_obs = next_obs_stack.view(args.num_envs, -1) # [num_envs, num_agents * obs_dim]
    next_obs = next_obs_stack.view(-1, obs_dim) # [num_envs * num_agents, obs_dim]
    next_done = torch.zeros(args.num_envs * args.num_agents).to(device)
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            global_obs[step] = next_global_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, entropy = agent.get_action_and_logprob(next_obs)
                value = agent.get_central_value(next_global_obs) # [num_envs, 1]
                repeat_value = value.repeat(1, args.num_agents)  # [num_envs, num_agents]
                flat_value = repeat_value.view(-1) # [num_envs * num_agents]
                values[step] = flat_value

            actions[step] = action
            logprobs[step] = logprob

            # execute the game and log data
            action_array = action.view(args.num_envs, args.num_agents).cpu().numpy()
            action_list = [action_array[:, i] for i in range(args.num_agents)]
            next_obs_list, reward_list, done, info = envs.step(action_list)
            
            reward = torch.stack(reward_list, dim=1).to(device)
            env_rewards = torch.sum(reward, dim=1).cpu().numpy()
            episode_returns += env_rewards
            episode_lengths += 1

            rewards[step] = reward.flatten()

            next_obs = torch.stack(next_obs_list, dim=1).to(device) # [num_envs, num_agents, obs_dim]
            next_obs = next_obs.view(-1, obs_dim) # [num_envs * num_agents, obs_dim]
            next_global_obs = next_obs.view(args.num_envs, -1) # [num_envs, num_agents * obs_dim]
            next_done = done.to(device).unsqueeze(1).repeat(1, args.num_agents) # [num_envs, ] => [num_envs, num_agents]
            next_done = next_done.flatten() # [num_envs * num_agents, ]

            episode_ret = []
            episode_len = []
            
            for i in range(len(done)):
                if done[i]:
                    episode_ret.append(episode_returns[i])
                    episode_len.append(episode_lengths[i])
                    episode_returns[i] = 0
                    episode_lengths[i] = 0
            
            # logging episode return and length
            if episode_ret and global_step > 100:
                # print(f"global_step={global_step}, episodic_return={np.mean(episode_ret)}")
                writer.add_scalar("charts/episodic_return", np.mean(episode_ret), global_step)
                writer.add_scalar("charts/episodic_length", np.mean(episode_len), global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_central_value(next_global_obs) # [num_envs, 1]
            next_repeat_value = next_value.repeat(1, args.num_agents)  # [num_envs, num_agents]
            next_flat_value = next_repeat_value.view(-1, 1) # [num_envs * num_agents, 1]
            next_value = next_flat_value.reshape(1, -1)

            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done.float()
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1].float()
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done.float()
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1].float()
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        # flatten the batch
        b_obs = obs.reshape((-1, obs_dim))
        b_global_obs = global_obs.reshape((-1, obs_dim * args.num_agents))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,1))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy = agent.get_action_and_logprob(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = agent.get_central_value(b_global_obs[mb_inds])
                newvalue = newvalue.view(-1)

                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if global_step % 10_000_000 == 0 and global_step > 0:
            torch.save(agent.actor.state_dict(), f"{run_name}_policy_{global_step}.pth")
            torch.save(agent.critic.state_dict(), f"{run_name}_critic_{global_step}.pth")
            print("SAVE MODEL in global_step = ", global_step)

    torch.save(agent.actor.state_dict(), f"{run_name}_policy.pth")
    torch.save(agent.critic.state_dict(), f"{run_name}_critic.pth")
    
    writer.close()