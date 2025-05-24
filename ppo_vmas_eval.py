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

include_agent_in_obs = [True, False][0]
scenario = ["navigation", "sampling"][0]
actor_path = "/home/tgkang/GraphRLProject/navigation__ppo_vmas_discrete_shared2__72__1748090726_policy.pth"
device = torch.device("cuda")
num_envs = 16
num_agents = 4
seed = 1

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def make_grid(frames, ncols=4):
    n = len(frames)
    h, w, c = frames[0].shape
    nrows = int(np.ceil(n / ncols))
    grid = np.zeros((h * nrows, w * ncols, c), dtype=frames[0].dtype)
    for idx, frame in enumerate(frames):
        row = idx // ncols
        col = idx % ncols
        grid[row*h:(row+1)*h, col*w:(col+1)*w, :] = frame
    return grid

def add_agent_id(obs, num_envs, num_agents, device):
    agent_ids = torch.eye(num_agents, device=device)  # [num_agents, num_agents]
    agent_ids = agent_ids.unsqueeze(0).repeat(num_envs, 1, 1)  # [num_envs, num_agents, num_agents]
    agent_ids = agent_ids.view(-1, num_agents)  # [num_envs * num_agents, num_agents]
    return torch.cat([obs, agent_ids], dim=-1)  # [num_envs * num_agents, obs_dim + num_agents]

class Agent(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, act_dim),
        )

    def get_action(self, x, deterministic=True):
        logits = self.actor(x)
        if deterministic:
            return logits.argmax(dim=-1)
        dist = Categorical(logits=logits)
        return dist.sample()
    

envs = vmas.make_env(
        scenario=scenario, # can be scenario name or BaseScenario class
        num_envs=num_envs,
        device=device, # "cpu", "cuda"
        continuous_actions=False,
        wrapper=None, 
        max_steps=None, # Defines the horizon. None is infinite horizon.
        seed=None, # Seed of the environment
        dict_spaces=False, # By default tuple spaces are used with each element in the tuple being an agent.
        # If dict_spaces=True, the spaces will become Dict with each key being the agent's name
        grad_enabled=False, # If grad_enabled the simulator is differentiable and gradients can flow from output to input
        terminated_truncated=False, # If terminated_truncated the simulator will return separate `terminated` and `truncated` flags in the `done()`, `step()`, and `get_from_scenario()` functions instead of a single `done` flag
        n_agents=num_agents, # Additional arguments you want to pass to the scenario initialization
    )

obs_list = envs.reset()
obs_dim = obs_list[0].shape[-1]
act_dim = envs.action_space[0].n

obs = torch.stack(obs_list, dim=1).to(device) # [num_envs, num_agents, obs_dim]
obs = obs.view(-1, obs_dim) # [num_envs * num_agents, obs_dim]

if include_agent_in_obs:
    obs = add_agent_id(obs, num_envs, num_agents, device) # [num_envs * num_agents, obs_dim + num_agents]

agent = Agent(obs_dim=obs_dim + num_agents if include_agent_in_obs else 0, act_dim=act_dim).to(device)
agent.actor.load_state_dict(torch.load(actor_path, map_location=device))
agent.actor.eval()

frame_list = []  # For creating a gif
MAX_STEP = 100
RENDER = True

for step in range(MAX_STEP):
    print(f"Step {step}")
    action = agent.get_action(obs, deterministic=True)
    action_array = action.view(num_envs, num_agents).cpu().numpy()
    action_list = [action_array[:, i] for i in range(num_agents)]
    
    obs_list, rews, dones, info = envs.step(action_list)
    obs = torch.stack(obs_list, dim=1).to(device) # [num_envs, num_agents, obs_dim]
    obs = obs.view(-1, obs_dim) # [num_envs * num_agents, obs_dim]
    if include_agent_in_obs:
        obs = add_agent_id(obs, num_envs, num_agents, device) # [num_envs * num_agents, obs_dim + num_agents]
    if RENDER:
        frame = []
        for i in range(num_envs):
            # envs.render returns H×W×C numpy array
            frame.append(envs.render(mode="rgb_array", env_index=i))
        grid_img = make_grid(frame, ncols=4)
        frame_list.append(grid_img)

if RENDER:
    from moviepy.editor import ImageSequenceClip
    fps=30
    clip = ImageSequenceClip(frame_list, fps=fps)
    clip.write_gif(f'{scenario}.gif', fps=fps)