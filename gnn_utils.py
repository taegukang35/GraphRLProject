import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import vmas
import gym
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphSageLayer(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, agg_type: str):
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

    def forward(self, feat: torch.Tensor, edge: torch.Tensor, degree: torch.Tensor) -> torch.Tensor:
        """
        feat: [N, dim_in]
        edge: [E, 2]
        degree: [N]
        """
        src = edge[:, 0]  # source node indices
        dst = edge[:, 1]  # destination node indices

        if self.agg_type == 'gcn':
            agg = torch.zeros_like(feat)
            agg = agg.index_add(0, dst, feat[src])  # SAFE: out-of-place
            inv_deg = (1.0 / degree.clamp(min=1)).unsqueeze(-1)
            out = self.act(self.weight(agg * inv_deg) + self.bias(feat))

        elif self.agg_type == 'mean':
            agg = torch.zeros_like(feat)
            agg = agg.index_add(0, dst, feat[src])  # SAFE: out-of-place
            inv_deg = (1.0 / degree.clamp(min=1)).unsqueeze(-1)
            agg = agg * inv_deg
            cat = torch.cat([agg, feat], dim=-1)
            out = self.act(self.weight(cat))

        elif self.agg_type == 'maxpool':
            src_transformed = self.act(self.linear_pool(feat))[src]  # [E, dim_in]
            idx = dst.unsqueeze(1).expand_as(src_transformed)
            agg = torch.zeros_like(feat)
            agg = agg.scatter_reduce(0, idx, src_transformed, reduce='amax', include_self=False)
            cat = torch.cat([agg, feat], dim=-1)
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