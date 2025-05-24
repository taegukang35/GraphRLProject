import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import vmas
import gym
import itertools

class GraphSageLayer(nn.Module):
    def __init__(self, dim_in: int,
                 dim_out: int,
                 agg_type: str):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.agg_type = agg_type
        self.act = nn.ReLU()

        if agg_type == 'gcn':
            self.weight = nn.Linear(dim_in, dim_out, bias=False)
            self.bias   = nn.Linear(dim_in, dim_out, bias=False)
        elif agg_type == 'mean':
            self.weight = nn.Linear(2 * dim_in, dim_out, bias=False)
        elif agg_type == 'maxpool':
            self.linear_pool = nn.Linear(dim_in, dim_in, bias=True)
            self.weight      = nn.Linear(2 * dim_in, dim_out, bias=False)
        else:
            raise RuntimeError(f"Unknown aggregation type: {agg_type}")

    def forward(self, feat: torch.Tensor,
                edge: torch.Tensor,
                degree: torch.Tensor) -> torch.Tensor:
        # feat: [N, dim_in], edge: [E,2], degree: [N]

        agg_vector = torch.zeros_like(feat)

        if self.agg_type == 'gcn':
            agg_vector.index_add_(0, edge[:,1], feat[edge[:,0]])
            inv = (1.0 / degree.clamp(min=1)).unsqueeze(-1)
            out = self.act(self.weight(agg_vector * inv) + self.bias(feat))

        elif self.agg_type == 'mean':
            agg_vector.index_add_(0, edge[:,1], feat[edge[:,0]])
            inv = (1.0 / degree.clamp(min=1)).unsqueeze(-1)
            cat = torch.cat([agg_vector * inv, feat], dim=-1)
            out = self.act(self.weight(cat))

        else:  # 'maxpool'
            src = self.act(self.linear_pool(feat))[edge[:,0]]
            idx = edge[:,1].unsqueeze(-1).expand_as(src)
            agg_vector.scatter_reduce_(0, idx, src, reduce='amax', include_self=False)
            cat = torch.cat([agg_vector, feat], dim=-1)
            out = self.act(self.weight(cat))

        return F.normalize(out, p=2, dim=-1)

def build_edge_lists(coords, n_agents, threshold: float):
    edges = [(i, i) for i in range(n_agents)]
    for i, j in itertools.combinations(range(n_agents), 2):
        dist = torch.dist(coords[i], coords[j], p=2)
        if dist <= threshold:
            edges.append((i, j))
    return edges

def build_knn_edge_lists(coords, n_agents, k: int):
    edges = []
    dist = torch.cdist(coords, coords, p=2)
    dist.fill_diagonal_(float('inf'))
    knn_idx = torch.topk(dist, k, largest=False).indices
    edges = [(i, i) for i in range(n_agents)]
    for i in range(n_agents):
        for j in knn_idx[i].tolist():
            edges.append((i, j))
    return edges

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
    def __init__(self, obs_dim: int, act_dim: int, dist: float = 1.0, agg_type='gcn'):
        super().__init__()
        self.dist = dist
        # GraphSAGE policy
        self.gsage1 = GraphSageLayer(obs_dim, 256, agg_type=agg_type)
        self.gsage2 = GraphSageLayer(256, 256, agg_type=agg_type)
        self.policy_head = layer_init(nn.Linear(256, act_dim), std=0.01)
        # Critic: MLP
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0)
        )

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, obs_dim]
        return self.critic(x).squeeze(-1)

    def get_action_and_logprob(self, x: torch.Tensor, positions, action=None):
        # x: [-1, num_agents, obs_dim]
        # positions: [-1, num_agents, 2]
        device = x.device
        num_agents = x.shape[1]
        # apply GCN per env
        h_list = []
        for e in range(x.shape[0]):
            feat_e = x[e] # [num_agents, obs_dim]
            coord_e = positions[e] # [num_agents, 2]
            edges_e = build_edge_lists(coord_e, num_agents, self.dist)
            edges_e = torch.tensor(edges_e).to(device) # [num_edges, 2]
            degree_e = torch.bincount(edges_e[:,1], minlength=num_agents).to(device) # [num_agents, ]
            h_e = self.gsage1(feat_e, edges_e, degree_e)
            h_e = self.gsage2(h_e, edges_e, degree_e) # [num_agents, obs_dim]
            h_list.append(h_e)
        # flatten back
        h = torch.cat(h_list, dim=0)  # [-1, obs_dim]
        print(h.shape)
        # policy logits
        logits = self.policy_head(h)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, logprob, entropy

if __name__ == "__main__":
    num_envs = 16
    num_agents = 8
    env = vmas.make_env(
            scenario="navigation", # can be scenario name or BaseScenario class
            num_envs=num_envs,
            device="cpu", # "cpu", "cuda"
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
    
    obs_list = env.reset()
    device = "cpu"
    obs_dim = obs_list[0].shape[-1]
    act_dim = env.action_space[0].n
    
    agent = GraphSageAgent(obs_dim=obs_dim + num_agents, act_dim=act_dim, dist=0.1, agg_type="gcn")

    obs = torch.stack(obs_list, dim=1).to(device) # [num_envs, num_agents, obs_dim]
    obs = obs.view(-1, obs_dim) # [num_envs * num_agents, obs_dim]
    obs = add_agent_id(obs, num_envs, num_agents, device) # [num_envs * num_agents, obs_dim + num_agents]
    obs = obs.view(-1, num_agents, obs_dim + num_agents) # [-1, num_agents, obs_dim]

    positions = []
    for e in range(num_envs):
        coords = torch.stack([agent.state.pos[e] for agent in env.agents], dim=0).to(device)  # [n_agents, pos_dim]
        positions.append(coords)
    positions = torch.stack(positions, dim=0).to(device)
    print(positions.shape)  # [-1, num_agents, 2]

    for coord in positions:
        edge1 = build_edge_lists(coord, num_agents, 1.0)
        edge2 = build_knn_edge_lists(coord, num_agents, 3)
        print(edge1, edge2)

    actions, logprob, entropy = agent.get_action_and_logprob(obs, positions)
    print(actions.shape)