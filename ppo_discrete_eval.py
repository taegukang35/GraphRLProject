import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from ppo import Agent, make_env

import argparse
import os
import random
import time
from distutils.util import strtobool
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="CartPole-v1",
        help="the id of the gym environment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=25000,
        help="total timesteps of the experiments")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--num-envs", type=int, default=4,
        help="the number of parallel game environments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--actor-path", type=str, default="actor.pth")
    args = parser.parse_args()
    return args

def make_env(gym_id, seed):
    def thunk():
        env = gym.make(gym_id)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    # fix seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.gym_id, i) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    agent.actor.load_state_dict(torch.load(args.actor_path, map_location=device))
    agent.actor.eval()

    # TRY NOT TO MODIFY: start the game
    total_timesteps = args.total_timesteps
    global_step = 0
    start_time = time.time()
    obs, done = envs.reset(seed=args.seed)
    obs = torch.Tensor(obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    rewards = torch.zeros((total_timesteps + 1), args.num_envs).to(device)
    frame_list = []

    for step in tqdm(range(1, total_timesteps + 1)):
        with torch.no_grad():
            obs = torch.Tensor(obs).to(device)
            action, logprob, _, value = agent.get_action_and_value(obs)
            obs, reward, done, trucated, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)

    envs.close()