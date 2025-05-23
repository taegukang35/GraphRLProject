import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import vmas


def parse_args():
    parser = argparse.ArgumentParser(description="VMAS RL Training Template")
    parser.add_argument("--scenario", type=str, default="navigation",
                        help="VMAS scenario name")
    parser.add_argument("--num-envs", type=int, default=16,
                        help="Number of parallel environments")
    parser.add_argument("--num-agents", type=int, default=8,
                        help="Number of agents per environment")
    parser.add_argument("--max-step", type=int, default=100,
                        help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (""cpu"" or ""cuda"")")
    return parser.parse_args()


def make_env(scenario, num_envs, continuous_actions, seed, device, n_agents):
    return vmas.make_env(
        scenario=scenario,
        num_envs=num_envs,
        device=device,
        continuous_actions=continuous_actions,
        seed=seed,
        n_agents=n_agents,
    )


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class SharedAgent(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, act_dim), std=0.01),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(x)

    def get_action(self, x, deterministic=True):
        logits = self.actor(x)
        if deterministic:
            return logits.argmax(dim=-1)
        dist = Categorical(logits=logits)
        return dist.sample()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Environment setup
    env = make_env(
        scenario=args.scenario,
        num_envs=args.num_envs,
        continuous_actions=False,
        seed=args.seed,
        device=device,
        n_agents=args.num_agents,
    )

    # Initial observation
    obs_list = env.reset()
    obs_dim = obs_list[0].shape[-1]
    act_dim = env.action_space[0].n

    # Agent
    agent = SharedAgent(obs_dim=obs_dim, act_dim=act_dim).to(device)

    print(f"Scenario: {args.scenario}, Envs: {args.num_envs}, Agents: {args.num_agents}")
    print(f"Obs dim: {obs_dim}, Act dim: {act_dim}")

    # Run loop
    for ep in range(1):  # single episode
        obs_list = env.reset() # obs_list contain (num_env, obs_len) for each agents
        print(len(obs_list))
        start_time = time.time()
        for step in range(args.max_step):
            # Stack observations: shape [num_envs, num_agents, obs_dim]
            obs_tensor = torch.stack(obs_list, dim=1).to(device)

            # Flatten for network: [num_envs * num_agents, obs_dim]
            flat_obs = obs_tensor.view(-1, obs_dim)

            with torch.no_grad():
                flat_action, flat_logp, flat_entropy, flat_value = agent.get_action_and_value(flat_obs)

            # Prepare actions per agent
            action_array = flat_action.view(args.num_envs, args.num_agents).cpu().numpy()
            action_list = [action_array[:, i] for i in range(args.num_agents)]

            # Step environment with a list of length num_agents
            next_obs_list, rewards, dones, infos = env.step(action_list)

            # Logging (optional)
            if dones.any():
                print(f"Episode done at step {step}")
                break

            obs_list = next_obs_list

        duration = time.time() - start_time
        print(f"Episode finished in {duration:.2f}s")