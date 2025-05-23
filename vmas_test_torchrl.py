import vmas
from vmas.simulator.scenario import BaseScenario
import time
import argparse
import os
import random
import time
from distutils.util import strtobool
from torchrl.envs import EnvBase, VmasEnv
from tensordict import TensorDict

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import copy

MAX_STEP = 100
RENDER = True
NUM_ENVS = 16
SCENARIO = "navigation"
CONTINOUS = False
SEED = 0
DEVICE = "cuda" # cpu, cuda
NUM_AGENTS = 8

def make_env(
    scenario: str,
    num_envs: int,
    continuous_actions: bool,
    seed,
    device,
    config=None
    ):
    return lambda: VmasEnv(
        scenario=scenario,
        num_envs=num_envs,
        continuous_actions=continuous_actions,
        seed=seed,
        device=device,
        categorical_actions=True,
        clamp_actions=True,
        **config,
    )

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class SharedAgent(nn.Module):
    """
    shared critic network, shared actor network
    """
    def __init__(self, envs):
        super(SharedAgent, self).__init__()
        self.single_obs_n = envs.observation_space[0].shape[0]
        self.single_action_n = env.action_space[0].n
        self.agent_n = env.n_agents

        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.single_obs_n, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(self.single_obs_n, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, self.single_action_n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def get_action(self, x, deterministic=False):
        logits = self.actor(x)
        probs = Categorical(logits=logits)


print("::: ENV MAKE TEST :::")
env_maker = make_env(
    num_envs=NUM_ENVS, 
    continuous_actions=CONTINOUS,
    seed=SEED,
    scenario=SCENARIO,
    device=DEVICE,
    config={'n_agents': NUM_AGENTS}
    )

env = env_maker()
print(env.action_space)
print(env.observation_space)
print(env.observation_space[0].shape[0]) # single observation 
print(env.num_envs)
print(env.n_agents)


print("::: ENV STEP TEST :::")
init = env.reset(seed=SEED)
print(init.keys())
obs = init["agents"]["observation"]
done = init["done"]
terminated = init["terminated"]

print(obs.shape)
print(done.shape)

for step in range(MAX_STEP):
    print(f"Step {step}")
    actions = torch.zeros(NUM_ENVS, NUM_AGENTS).to(DEVICE)
    for i, agent in enumerate(env.agents):
        action = env.get_random_action(agent)
        actions[:, i] = action
    actions = actions.unsqueeze(-1)
    actions_td = TensorDict(
        source={"agents": TensorDict({"action": actions}, batch_size=(NUM_ENVS,))},
        batch_size=(NUM_ENVS,),
        device=DEVICE,
    )
    transition_dict = env.step(actions_td)
    obs = transition_dict["next"]["agents"]["observation"]
    done = transition_dict["next"]["done"]
    terminated = transition_dict["next"]["terminated"]
    break

# shared actor, shared critic model test
print("::: MODEL SHAPE TEST :::")
agent = SharedAgent(env).to(DEVICE)
init = env.reset(seed=SEED)
obs = init["agents"]["observation"] # (num_env, num_agents, obs_shape)

for step in range(MAX_STEP):
    print(f"Step {step}")
    obs = obs.reshape(-1, agent.single_obs_n)
    actions, log_probs, entropy, value = agent.get_action_and_value(obs)
    actions = actions.reshape(-1, agent.agent_n)
    actions_td = TensorDict(
        source={"agents": TensorDict({"action": actions}, batch_size=(NUM_ENVS,))},
        batch_size=(NUM_ENVS,),
        device=DEVICE,
    )
    transtion_dict = env.step(actions_td)
    obs = transition_dict["next"]["agents"]["observation"]
    done = transition_dict["next"]["done"]
    terminated = transition_dict["next"]["terminated"]