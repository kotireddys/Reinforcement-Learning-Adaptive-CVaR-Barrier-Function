"""Complete-graph construction and graph attention encoder.

The graph follows the online adaptive CBF GAT (tkkim-robot/online_adaptive_cbf,
nn_model/penn/gat.py): node features are a node-type one-hot, edge features are
[dx, dy, clearance, dvx, dvy], an edge MLP (psi1) embeds z_ij = [h_i, h_j, e_ij],
a score MLP (psi2) gives attention logits, softmax runs over each node's
neighbours, and a message MLP (psi3) produces the aggregated messages. The ego
node's embedding is the graph readout.

Differences from that reference:
  * Dense, masked, batched tensors instead of torch_geometric / torch_scatter,
    so a variable number of humans is handled with the env's mask and no extra
    dependency is needed.
  * Stacked layers (default 2). With a single round, the ego embedding only sees
    ego->j edges, so a complete graph behaves like a star graph. The first round
    lets every human aggregate the others (crowd interactions), the second
    aggregates that into the ego node.

Graph per sample, all positions in the ego-centred frame (ego at the origin):
  node 0       ego robot
  node 1       goal
  node 2..N+1  human slots (slots with mask=0 are padding and excluded)
"""

import torch
from torch import nn


NODE_TYPES = 3  # ego, goal, human
EDGE_DIM = 5    # dx, dy, clearance, dvx, dvy


def build_graph(obs):
    """Relative policy obs -> complete-graph tensors.

    obs: (B, 6 + N*6) as produced by absolute_obs_batch_to_relative:
         [p_r - p_g (2), v_r (2), theta, r_r, N * (p_r - p_h (2), v_h (2), r_h, mask)]

    Returns:
      node_x:    (B, n, 3)   node-type one-hot
      edge_attr: (B, n, n, 5) features of edge i->j: [p_j - p_i, clearance, v_j - v_i]
      adj:       (B, n, n)   bool, True where j is a neighbour of i
    with n = N + 2.
    """
    bsz = obs.size(0)
    n_h = (obs.size(1) - 6) // 6
    humans = obs[:, 6:].reshape(bsz, n_h, 6)
    zeros2 = obs.new_zeros(bsz, 1, 2)
    ones1 = obs.new_ones(bsz, 1)

    pos = torch.cat([zeros2, -obs[:, None, 0:2], -humans[:, :, 0:2]], dim=1)
    vel = torch.cat([obs[:, None, 2:4], zeros2, humans[:, :, 2:4]], dim=1)
    radius = torch.cat([obs[:, 5:6], obs.new_zeros(bsz, 1), humans[:, :, 4]], dim=1)
    valid = torch.cat([ones1, ones1, humans[:, :, 5]], dim=1) > 0.5

    n = n_h + 2
    node_type = torch.cat(
        [obs.new_zeros(bsz, 1, dtype=torch.long),
         obs.new_ones(bsz, 1, dtype=torch.long),
         obs.new_full((bsz, n_h), 2, dtype=torch.long)],
        dim=1,
    )
    node_x = nn.functional.one_hot(node_type, NODE_TYPES).to(obs.dtype)

    rel_pos = pos.unsqueeze(1) - pos.unsqueeze(2)  # [b, i, j] = p_j - p_i
    rel_vel = vel.unsqueeze(1) - vel.unsqueeze(2)
    dist = torch.sqrt((rel_pos ** 2).sum(-1) + 1e-8)
    clearance = (dist - radius.unsqueeze(1) - radius.unsqueeze(2)).clamp_min(0.0)
    edge_attr = torch.cat([rel_pos, clearance.unsqueeze(-1), rel_vel], dim=-1)

    not_self = ~torch.eye(n, dtype=torch.bool, device=obs.device)
    adj = valid.unsqueeze(1) & valid.unsqueeze(2) & not_self
    return node_x, edge_attr, adj


def _mlp(in_dim, hidden_dim, out_dim):
    return nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))


class GraphAttentionLayer(nn.Module):
    """One round of edge-conditioned attention message passing."""

    def __init__(self, node_dim, edge_dim, hidden_dim, out_dim):
        super().__init__()
        self.edge_mlp = _mlp(2 * node_dim + edge_dim, hidden_dim, out_dim)  # psi1
        self.score_mlp = _mlp(out_dim, out_dim, 1)                           # psi2
        self.msg_mlp = _mlp(out_dim, hidden_dim, out_dim)                    # psi3

    def forward(self, h, edge_attr, adj):
        n = h.size(1)
        h_i = h.unsqueeze(2).expand(-1, -1, n, -1)
        h_j = h.unsqueeze(1).expand(-1, n, -1, -1)
        q = self.edge_mlp(torch.cat([h_i, h_j, edge_attr], dim=-1))  # (B, n, n, out)

        # Masked softmax over neighbours j. A large negative (not -inf) keeps
        # rows without neighbours (padding nodes) NaN-free; multiplying by adj
        # then zeroes them.
        logits = self.score_mlp(q).squeeze(-1).masked_fill(~adj, -1e9)
        attn = torch.softmax(logits, dim=-1) * adj.to(h.dtype)

        out = (attn.unsqueeze(-1) * self.msg_mlp(q)).sum(dim=2)
        return out, attn


class GraphEncoder(nn.Module):
    """obs -> ego-node embedding of the complete ego/goal/humans graph."""

    def __init__(self, embed_dim=64, hidden_dim=64, num_layers=2):
        super().__init__()
        if int(num_layers) < 1:
            raise ValueError("GraphEncoder requires num_layers >= 1")
        dims = [NODE_TYPES] + [int(embed_dim)] * int(num_layers)
        self.layers = nn.ModuleList(
            GraphAttentionLayer(dims[i], EDGE_DIM, int(hidden_dim), dims[i + 1])
            for i in range(int(num_layers))
        )
        self.embed_dim = int(embed_dim)
        self.last_attn = None

    def forward(self, obs):
        h, edge_attr, adj = build_graph(obs)
        attns = []
        for idx, layer in enumerate(self.layers):
            h, attn = layer(h, edge_attr, adj)
            if idx < len(self.layers) - 1:
                h = torch.relu(h)
            attns.append(attn.detach())
        # (B, L, n, n) attention weights, kept for visualisation/diagnostics.
        self.last_attn = torch.stack(attns, dim=1)
        return h[:, 0]
