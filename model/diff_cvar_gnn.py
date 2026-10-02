import numpy as np
import torch
from torch import nn

from model.diff_cvar import DiffCVaRBFQP
from model.gat import GraphEncoder
from model.ppo_base import resolve_activation


class DiffCVaRBFQPGNN(DiffCVaRBFQP):
    """DiffCVaRBFQP whose heads read a GAT embedding of the full scene graph.

    Only the representation changes: the u_nom / beta / r_safe heads, the
    CVaR-BF constraints and the differentiable QP are inherited unchanged. The
    QP still receives the nearest ``qp_top_k`` human blocks (the env sorts
    humans nearest-first), so with qp_top_k=1 the safety layer sees exactly
    what the baseline's does.
    """

    def __init__(self, n_features, action_dim, hidden_dim=256,
                 gnn_embed_dim=64, gnn_hidden_dim=64, gnn_layers=2, qp_top_k=1, **kwargs):
        super().__init__(n_features, action_dim, hidden_dim=hidden_dim, **kwargs)
        self.qp_top_k = int(qp_top_k)
        if self.qp_top_k < 1 or 6 + 6 * self.qp_top_k > n_features:
            raise ValueError("qp_top_k must be in [1, number of human slots in obs]")
        self.graph_encoder = GraphEncoder(gnn_embed_dim, gnn_hidden_dim, gnn_layers)
        # fc1 now maps the ego embedding (not the flat obs) into the shared trunk.
        self.fc1 = nn.Linear(int(gnn_embed_dim), hidden_dim)

    def encode(self, obs):
        return self.act(self.fc1(self.graph_encoder(obs)))

    def qp_obs(self, obs):
        return obs[:, : 6 + 6 * self.qp_top_k]


class GraphCritic(nn.Module):
    """Value network with its own GAT encoder (not shared with the actor)."""

    def __init__(self, n_features, output_dim, hidden_dim=256, hidden_dim2=256, act="relu",
                 gnn_embed_dim=64, gnn_hidden_dim=64, gnn_layers=2, **kwargs):
        super().__init__()
        self.act = resolve_activation(act)
        self.graph_encoder = GraphEncoder(gnn_embed_dim, gnn_hidden_dim, gnn_layers)
        self.fc1 = nn.Linear(int(gnn_embed_dim), hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, hidden_dim2)
        self.fc31 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        obs = obs.to(self.fc1.weight.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        obs = obs.reshape(obs.size(0), -1)

        x = self.act(self.fc1(self.graph_encoder(obs)))
        x = self.act(self.fc21(x))
        return self.fc31(x)
