# GNN observation encoder for DiffCVaR-BF-QP

**Question.** Does a graph attention encoder over the full scene, with the ego robot,
the goal and every visible human as nodes, give a better RL policy than the current
top-1 nearest-human observation? The CVaR barrier constraints and the differentiable
QP layer stay unchanged.

Model selection: `MODEL=diff_cvar_gnn` (GNN) vs `MODEL=diff_cvar` (baseline).

## 1. Where the single-obstacle selection lived

The env already observes every human. `SocialNav._get_obs` emits one 6-d slot per
possible human, sorted nearest-first, with a validity mask. The reduction to one human
happens downstream:

- `config/env/*.yaml: obs_top_k: 1`
- `crowd_sim/utils.py: select_top_k_obs` keeps the first `obs_top_k` slots. It is
  called in rollouts (`trainer/vec_ppo.py`), in training-time evaluation
  (`trainer/ppo.py`) and in `scripts/eval.py`.

That 12-d vector, `[p_r - p_g, v_r, θ, r_r | p_r - p_h, v_h, r_h, mask]`, feeds
**two consumers** inside `DiffCVaRBFQP`:

1. **Representation.** `fc1 → fc21/22/23 → (u_nom, β, r_safe)`.
2. **Safety layer.** `_extract_obstacle_blocks` produces one CVaR-BF constraint
   per human block for the QP.

## 2. Design: change only the representation

The brief asks to replace the representation and keep the safety constraints. The
two consumers are therefore **split**:

| | baseline `diff_cvar` | GNN `diff_cvar_gnn` |
|---|---|---|
| Pipeline passes | nearest human | all human slots (`obs_top_k` = 20) |
| Representation | MLP on the 12-d vector | GAT over ego + goal + all humans → ego embedding |
| QP constraints | nearest human | nearest human (`qp_top_k: 1`), **identical input** |
| Heads, CVaR-BF, QP solver | — | inherited unchanged |
| Critic | MLP on the 12-d vector | its own GAT, same architecture as the actor's |

With this split, the representation is the only difference between the two runs.
Handing all N humans to the QP as well would change the safety layer too. That is a
natural follow-up experiment (§7), but it would confound this one.

Implementation: `DiffCVaRBFQP.forward` calls two hooks, `encode(obs)` and `qp_obs(obs)`.
Their defaults reproduce the old forward pass bit-for-bit; this was verified with
identical weights, max |Δ| = 0. `DiffCVaRBFQPGNN` overrides only those two hooks.

## 3. Graph construction (`model/gat.py: build_graph`)

Each observation becomes a complete graph, in the ego-centred frame (ego at the
origin):

- **Nodes:** `0` ego, `1` goal, `2..N+1` human slots. Node features are the type
  one-hot `[ego, goal, human]`.
- **Edge features** for edge i→j: `[p_j − p_i (2), clearance_ij (1), v_j − v_i (2)]`,
  where `clearance = max(‖p_j − p_i‖ − r_i − r_j, 0)`. The goal has radius 0 and
  velocity 0, so the ego→goal edge carries the goal direction and the negated ego
  velocity. That is how the ego's own velocity enters the graph.
- **Adjacency:** all pairs of valid nodes, no self-loops. Padded slots (`mask=0`)
  get no edges in either direction, so a variable number of humans is handled
  exactly.

These are the node and edge features of the lab's online adaptive CBF GAT
(`tkkim-robot/online_adaptive_cbf`, `nn_model/penn/gat.py`; arXiv:2504.03038 p.7),
applied to this env's observation.

## 4. Attention network (`GraphAttentionLayer`, `GraphEncoder`)

Each layer follows the reference recipe:

```
z_ij  = [h_i, h_j, e_ij]
q_ij  = ψ1(z_ij)                       # edge embedding (MLP)
a_ij  = softmax_j∈N(i) ψ2(q_ij)         # attention over neighbours (masked)
h_i'  = Σ_j a_ij · ψ3(q_ij)             # aggregated message
```

The readout is the ego node's final embedding. It goes through `fc1` into the
unchanged trunk and heads.

Two deliberate differences from the reference:

1. **Two layers instead of one.** With a single round, the ego embedding depends only
   on the ego→j edges, so a "complete graph" collapses to a star graph. In round 1,
   every human aggregates information from the other humans and the goal: who is
   converging on whom, crowd density near each person. In round 2, the ego aggregates
   those interaction-aware embeddings. This is what makes the complete graph matter.
2. **Dense, masked, batched tensors** of shape `(B, n, n, ·)` instead of
   torch_geometric/torch_scatter. There's no extra dependency, a padded slot is just
   a masked row and column, and it vectorises over PPO minibatches. Masking uses a
   large negative logit followed by multiplication by the adjacency, which keeps
   rows with no neighbours NaN-free.

Size: embed 64, hidden 64, 2 layers. That is about 43k parameters per encoder; actor
and critic each have their own encoder, because they use separate Adam optimizers in
this PPO. Actor totals: 258k (GNN) vs 202k (baseline).

Properties checked in `scripts/test_gnn.py`:
- graph geometry and masking
- invariance to the order of human slots
- invariance to anything written in padded slots
- that the QP receives the same input and returns the same action as the baseline QP
- that gradients reach both GAT encoders through the differentiable QP

## 5. Experimental protocol

- Same seed, PPO hyperparameters, env, reward and 20M-step budget for both runs; only
  `MODEL` differs. Launcher: `scripts/launch_comparison.sh`.
- **32 parallel envs** for both runs. With 8 envs, about 135 steps/s means roughly
  42 h per run; the bottleneck is env stepping plus the per-step QP solve, not the
  network. 32 envs give about 310 steps/s, and because `VecPPO` only adds completed
  episodes, a batch is about 9.4k steps instead of 8.2k. 64 envs would have more
  than doubled the batch and halved the number of PPO updates, so they were not used.
- Training-time eval uses 5 episodes × 10 seeds (100–1000), not 20, purely for
  wall-clock reasons. It is used only to rank checkpoints.
- **Final numbers come from held-out seeds** (10000–11900, 200 episodes per cell). The
  training-time seeds pick the checkpoints, so reporting on them would be optimistically
  biased.
- **Density sweep: 10 / 20 / 30 humans**; training uses 20. The GNN's input size
  does not depend on N. With fewer humans its slots are padded; with more, it sees
  the 20 nearest, which is still more than the baseline's 1.
- Qualitative: side-by-side rollouts on the same held-out seeds, and GNN rollouts with
  the ego's last-layer attention drawn on the scene (`scripts/visualize_attention.py`).

Reproduce:

```bash
GPUS="1 2" WANDB_MODE=offline bash scripts/launch_comparison.sh trainer.eval_episodes=5
python scripts/compare_runs.py \
  --run baseline=outputs/social_nav_var_num/runs/baseline-diff_cvar-bs8192-ep8-lr1.0e-04-ent0.01 \
  --run gnn=outputs/social_nav_var_num/runs/gnn-diff_cvar_gnn-bs8192-ep8-lr1.0e-04-ent0.01 \
  --log baseline=logs/baseline.log --log gnn=logs/gnn.log --out results
```

## 6. Results

See `results/comparison.md`, `results/learning_curves.png` and
`results/density_sweep.png`; these are filled in from the server runs.

## 7. Caveats and next steps

- **One seed per model.** RL variance across seeds is large; a single-seed difference
  is suggestive, not conclusive. Next: 3–5 seeds per model.
- **Attention is not explanation.** The overlays show where the ego's last layer puts
  weight, after layer 1 has already mixed information between nodes.
- **The QP still constrains only the nearest human,** in both models. The GNN can
  *anticipate* other humans through `u_nom`, `β` and `r_safe`, but formal safety
  is still with respect to one human. Next: `qp_top_k > 1` for both models, so the
  safety layer sees the same humans the encoder does.
- **Pre-existing PPO details,** left unchanged to keep the comparison controlled:
  - GAE treats time-limit truncation as termination (no bootstrap).
  - Advantages are not normalised.
  - The KL early stop checks only the last minibatch of each epoch.

  Each is worth fixing in both models together.
- **Capacity is not matched.** The GNN actor has about 28% more parameters (258k vs
  202k), almost all in the encoder. A wider baseline MLP would be the control for
  "more parameters" as opposed to "graph structure".
- **Shared vs separate encoders.** Sharing one GAT between actor and critic would halve
  encoder compute and may help sample efficiency, but needs a joint optimizer.
