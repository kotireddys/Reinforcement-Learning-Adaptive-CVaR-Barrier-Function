"""Sanity checks for the GAT observation encoder.

Run: python scripts/test_gnn.py
"""

from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.diff_cvar import DiffCVaRBFQP
from model.diff_cvar_gnn import DiffCVaRBFQPGNN, GraphCritic
from model.gat import GraphEncoder, build_graph


N_HUMANS = 6
OBS_DIM = 6 + 6 * N_HUMANS


def random_obs(bsz, n_valid, seed=0):
    """Relative policy obs with the first n_valid human slots real, nearest-first."""
    g = torch.Generator().manual_seed(seed)
    obs = torch.zeros(bsz, OBS_DIM)
    obs[:, 0:2] = torch.randn(bsz, 2, generator=g) * 4.0     # p_r - p_g
    obs[:, 2:4] = torch.randn(bsz, 2, generator=g)           # v_r
    obs[:, 5] = 0.3                                          # r_r
    humans = obs[:, 6:].view(bsz, N_HUMANS, 6)
    dists = torch.sort(torch.rand(bsz, n_valid, generator=g) * 4.0 + 1.0, dim=1).values
    ang = torch.rand(bsz, n_valid, generator=g) * 2 * np.pi
    humans[:, :n_valid, 0] = dists * torch.cos(ang)
    humans[:, :n_valid, 1] = dists * torch.sin(ang)
    humans[:, :n_valid, 2:4] = torch.randn(bsz, n_valid, 2, generator=g)
    humans[:, :n_valid, 4] = 0.4
    humans[:, :n_valid, 5] = 1.0
    return obs


def test_graph_construction():
    obs = random_obs(4, n_valid=3)
    node_x, edge_attr, adj = build_graph(obs)
    n = N_HUMANS + 2
    assert node_x.shape == (4, n, 3) and edge_attr.shape == (4, n, n, 5) and adj.shape == (4, n, n)
    # ego->goal displacement is p_g - p_r = -(obs[0:2])
    assert torch.allclose(edge_attr[:, 0, 1, 0:2], -obs[:, 0:2])
    # ego->human0 displacement is p_h - p_r = -(block[0:2])
    assert torch.allclose(edge_attr[:, 0, 2, 0:2], -obs[:, 6:8])
    # no self edges, padded humans (slots 3..5 -> nodes 5..7) have no edges at all
    assert not adj[:, torch.arange(n), torch.arange(n)].any()
    assert not adj[:, 5:, :].any() and not adj[:, :, 5:].any()
    assert adj[:, :5, :5].sum().item() == 4 * (5 * 4)


def test_permutation_invariance():
    torch.manual_seed(0)
    enc = GraphEncoder(32, 32, 2)
    obs = random_obs(8, n_valid=N_HUMANS)
    perm = torch.randperm(N_HUMANS)
    obs_perm = obs.clone()
    obs_perm[:, 6:] = obs[:, 6:].view(8, N_HUMANS, 6)[:, perm].reshape(8, -1)
    assert torch.allclose(enc(obs), enc(obs_perm), atol=1e-5)


def test_padding_is_ignored():
    torch.manual_seed(0)
    enc = GraphEncoder(32, 32, 2)
    obs = random_obs(8, n_valid=2)
    garbage = obs.clone()
    garbage[:, 6 + 2 * 6:] = torch.randn_like(garbage[:, 6 + 2 * 6:]) * 10.0
    garbage[:, 6:].view(8, N_HUMANS, 6)[:, 2:, 5] = 0.0  # keep them padded
    assert torch.allclose(enc(obs), enc(garbage), atol=1e-5)
    # also robust when no human is visible: ego still attends to the goal
    empty = random_obs(8, n_valid=0)
    assert torch.isfinite(enc(empty)).all()


def test_qp_sees_baseline_input():
    torch.manual_seed(0)
    base = DiffCVaRBFQP(12, 2, safe_dist=0.75, vmax=1.5)
    gnn = DiffCVaRBFQPGNN(OBS_DIM, 2, safe_dist=0.75, vmax=1.5, gnn_embed_dim=32, gnn_hidden_dim=32)
    obs = random_obs(8, n_valid=4)
    assert torch.equal(gnn.qp_obs(obs), obs[:, :12])
    # Same u_nom/beta/r_safe and same QP input -> same safe action as the baseline QP.
    u_nom = torch.randn(8, 2)
    beta = torch.full((8,), 0.3)
    r_safe = torch.full((8,), 0.9)
    a = base._solve_single_integrator_qp(obs[:, :12], u_nom, beta, r_safe)
    b = gnn._solve_single_integrator_qp(gnn.qp_obs(obs), u_nom, beta, r_safe)
    assert torch.allclose(a, b, atol=1e-6)


def test_gradients_reach_encoder_through_qp():
    torch.manual_seed(0)
    actor = DiffCVaRBFQPGNN(OBS_DIM, 2, safe_dist=0.75, vmax=1.5, gnn_embed_dim=32, gnn_hidden_dim=32)
    critic = GraphCritic(OBS_DIM, 1, gnn_embed_dim=32, gnn_hidden_dim=32)
    obs = random_obs(16, n_valid=5)
    actor(obs).pow(2).sum().backward()
    critic(obs).sum().backward()
    for name, module in (("actor", actor), ("critic", critic)):
        grads = [p.grad for p in module.graph_encoder.parameters()]
        assert all(g is not None for g in grads), f"{name}: encoder param without grad"
        assert sum(g.abs().sum().item() for g in grads) > 0.0, f"{name}: zero encoder grad"
    assert actor.graph_encoder.last_attn.shape == (16, 2, N_HUMANS + 2, N_HUMANS + 2)


def main():
    tests = [
        test_graph_construction,
        test_permutation_invariance,
        test_padding_is_ignored,
        test_qp_sees_baseline_input,
        test_gradients_reach_encoder_through_qp,
    ]
    for test in tests:
        test()
        print(f"ok  {test.__name__}", flush=True)
    print("gnn tests passed", flush=True)


if __name__ == "__main__":
    main()
