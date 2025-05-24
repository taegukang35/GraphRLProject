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

scenario = ["navigation", "sampling"][1]
actor_path = "/home/tgkang/GraphRLProject/sampling__ppo_vmas_discrete_centralized__1__1748079694_policy.pth"
device = torch.device("cuda")
num_envs = 16
num_agents = 4
seed = 1

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

agent = Agent(obs_dim=obs_dim, act_dim=act_dim).to(device)
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
    if RENDER:
        frame = envs.render(mode="rgb_array")
        frame_list.append(frame)

if RENDER:
    from moviepy.editor import ImageSequenceClip
    fps=30
    clip = ImageSequenceClip(frame_list, fps=fps)
    clip.write_gif(f'{scenario}.gif', fps=fps)