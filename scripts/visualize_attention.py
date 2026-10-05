"""Render GNN rollouts with the ego node's attention drawn on the scene.

For each step, a line is drawn from the robot to every node it attends to in the
last GAT layer (goal and visible humans); line width and opacity scale with the
attention weight, and the top weights are annotated. Saves one GIF per episode
plus a few PNG snapshots.

  python scripts/visualize_attention.py --save-dir outputs/.../runs/<gnn run> \
      --checkpoint outputs/.../ckpt_<step>.pt --seeds 10000,10100 [overrides]
"""

from pathlib import Path
import argparse
import sys

import imageio
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval import Evaluator, parse_seeds, policy_obs_from_env_obs, set_render_safe_distance


def frame_from_fig(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()


def draw_attention(env, abs_obs, weights, top_n=3):
    """weights: (n,) ego-row attention over [ego, goal, human slots...]."""
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    ax = base.ax
    robot = abs_obs[0:2]
    targets = [abs_obs[2:4]]  # goal
    n_slots = (abs_obs.size - 8) // 6
    slots = abs_obs[8:].reshape(n_slots, 6)
    targets += [slot[0:2] for slot in slots]
    w = np.asarray(weights[1: 1 + len(targets)], dtype=float)
    if w.size == 0 or w.max() <= 0:
        return
    order = np.argsort(-w)
    for rank, idx in enumerate(order):
        if w[idx] < 1e-3:
            continue
        x, y = targets[idx]
        color = "#16a34a" if idx == 0 else "#dc2626"  # goal green, humans red
        ax.plot([robot[0], x], [robot[1], y], color=color, linewidth=0.5 + 8.0 * w[idx],
                alpha=float(np.clip(0.25 + 0.75 * w[idx] / w.max(), 0.0, 1.0)), zorder=6)
        if rank < top_n:
            ax.annotate(f"{w[idx]:.2f}", (x, y), xytext=(6, 6), textcoords="offset points",
                        fontsize=13, color=color, fontweight="bold", zorder=7)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seeds", default="10000,10100,10200")
    parser.add_argument("--out", default="", help="Output dir (default: <save-dir>/attention_<ckpt>)")
    parser.add_argument("--snapshot-every", type=int, default=40)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    ev = Evaluator(args.save_dir, episodes_per_seed=1, overrides=args.overrides)
    state = torch.load(args.checkpoint, map_location=ev.device, weights_only=False)
    ev.model.load_state_dict(state["model"], strict=True)
    ev.model.eval()
    encoder = getattr(ev.model.actor, "graph_encoder", None)
    if encoder is None:
        raise SystemExit("checkpoint's actor has no graph_encoder (not a diff_cvar_gnn run)")

    out_dir = Path(args.out) if args.out else Path(args.save_dir) / f"attention_{Path(args.checkpoint).stem}{ev.output_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    from crowd_sim.utils import build_env
    env = build_env(ev.env_name, render_mode="rgb_array", config=ev.config)
    try:
        for seed in parse_seeds(args.seeds):
            obs, _ = env.reset(seed=seed)
            ev.model.reset_episode_cache()
            frames, done, info, step = [], False, {}, 0
            while not done:
                policy_obs = policy_obs_from_env_obs(obs, ev.obs_top_k)
                obs_t = torch.tensor(policy_obs, dtype=torch.float32, device=ev.device).unsqueeze(0)
                with torch.no_grad():
                    policy_action = ev.model.get_action_deterministic(obs_t)
                    set_render_safe_distance(env, ev.model.actor)
                    action = ev.model.policy_action_to_env_action(obs_t, policy_action)
                    action = action.cpu().numpy().astype(np.float32).squeeze(0)
                weights = encoder.last_attn[0, -1, 0].cpu().numpy()
                abs_obs = np.asarray(obs, dtype=np.float32)

                env.render()
                draw_attention(env, abs_obs, weights)
                frame = frame_from_fig((env.unwrapped if hasattr(env, "unwrapped") else env).fig)
                frames.append(frame)
                if step % args.snapshot_every == 0:
                    imageio.imwrite(out_dir / f"seed_{seed}_t{step:03d}.png", frame)

                obs, _r, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                step += 1

            outcome = "success" if info.get("is_success") else "collision" if info.get("is_collision") else "timeout"
            path = out_dir / f"seed_{seed}_{outcome}.gif"
            imageio.mimsave(path, frames, fps=10)
            print(f"{path} ({step} steps)", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
