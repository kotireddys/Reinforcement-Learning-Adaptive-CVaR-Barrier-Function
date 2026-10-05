"""Final baseline-vs-GNN comparison: held-out evaluation, learning curves, tables.

For each run, the top-k checkpoints by training-time eval return (ckpt_manifest.json)
are evaluated on held-out seeds (disjoint from the 100..1000 seeds used to select
checkpoints during training) at several crowd densities, in parallel processes.

  python scripts/compare_runs.py \
      --run baseline=outputs/social_nav_var_num/runs/baseline-diff_cvar-... \
      --run gnn=outputs/social_nav_var_num/runs/gnn-diff_cvar_gnn-... \
      --log baseline=logs/baseline.log --log gnn=logs/gnn.log \
      --densities 10,20,30 --out results
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

COLORS = {"baseline": "#2a78d6", "gnn": "#eb6834"}  # categorical slots 1, 2
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
UPDATE_RE = re.compile(r"^update\s+(\d+)/\d+ \| steps\s+(\d+) \| eval\s+(\S+) \| success (\S+) \| sps (\d+)")


def parse_pairs(items):
    out = {}
    for item in items or []:
        name, _, value = item.partition("=")
        if not value:
            raise SystemExit(f"expected name=value, got {item!r}")
        out[name] = value
    return out


def top_checkpoints(run_dir, k):
    manifest = json.loads((Path(run_dir) / "ckpt_manifest.json").read_text())
    ranked = sorted(manifest.items(), key=lambda kv: kv[1]["performance"], reverse=True)
    return [(Path(run_dir) / name, info) for name, info in ranked[:k] if (Path(run_dir) / name).exists()]


def run_eval(run_dir, ckpt, density, seeds, episodes, log_dir):
    tag = f"heldout_ck{ckpt.stem.split('_')[-1]}"
    overrides = [f"env.humans.num_humans={density}"]
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "eval.py"), "--save-dir", str(run_dir),
           "--checkpoint", str(ckpt), "--seeds", seeds, "--episodes-per-seed", str(episodes),
           "--tag", tag, *overrides]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    log_path = Path(log_dir) / f"{Path(run_dir).name}_{tag}_n{density}.log"
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"eval failed ({proc.returncode}), see {log_path}")
    suffix = "_" + re.sub(r"[^A-Za-z0-9]+", "_", "__".join(overrides)).strip("_")[:120] + f"_{tag}"
    result = json.loads((Path(run_dir) / f"eval_results{suffix}.json").read_text())
    return result[ckpt.name]


def parse_curve(log_path):
    steps, ret, succ = [], [], []
    text = Path(log_path).read_text(errors="replace").replace("\r", "\n")
    for line in text.splitlines():
        m = UPDATE_RE.match(line.strip())
        if m and m.group(3) != "nan":
            steps.append(int(m.group(2)))
            ret.append(float(m.group(3)))
            succ.append(float(m.group(4)))
    return np.array(steps), np.array(ret), np.array(succ)


def _style(ax, title, ylabel, xlabel):
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=10)
    ax.set_xlabel(xlabel, color=INK_MUTED, fontsize=10)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)


def _direct_labels(ax, items, min_gap_frac=0.07):
    """items: [(x, y, name)] at series ends; nudge labels apart vertically so they never overlap."""
    if not items:
        return
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * min_gap_frac
    items = sorted(items, key=lambda it: it[1])
    placed = []
    for x, y, name in items:
        y_label = y if not placed else max(y, placed[-1] + gap)
        placed.append(y_label)
        ax.annotate(name, (x, y), xytext=(x, y_label), textcoords="data", va="center", ha="left",
                    fontsize=10, color=INK, annotation_clip=False)
    x0, x1 = ax.get_xlim()
    ax.set_xlim(x0, x1 + (x1 - x0) * 0.12)
    for text in ax.texts[-len(items):]:
        text.set_x(text.get_position()[0] + (x1 - x0) * 0.02)


def plot_curves(curves, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), facecolor="#fcfcfb")
    for ax, idx, title, ylabel in ((axes[0], 1, "Eval return during training", "mean return"),
                                   (axes[1], 2, "Eval success rate during training", "success rate")):
        ax.set_facecolor("#fcfcfb")
        ends = []
        for name, curve in curves.items():
            if curve[0].size == 0:
                continue
            x = curve[0] / 1e6
            ax.plot(x, curve[idx], color=COLORS.get(name, INK), linewidth=2, label=name)
            ends.append((x[-1], curve[idx][-1], name))
        _style(ax, title, ylabel, "environment steps (M)")
        _direct_labels(ax, ends)
        ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.text(0.01, 0.01, "Training-time eval: 50 episodes on seeds 100-1000 (used for checkpoint selection).",
             fontsize=8, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_density(table, densities, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = (("success_rate", "Success rate"), ("collision_rate", "Collision rate"),
               ("timeout_rate", "Timeout rate"))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor="#fcfcfb")
    for ax, (key, title) in zip(axes, metrics):
        ax.set_facecolor("#fcfcfb")
        ends = []
        for name, rows in table.items():
            means = [rows[d]["best"][key] for d in densities]
            ax.plot(densities, means, color=COLORS.get(name, INK), linewidth=2, marker="o",
                    markersize=8, markeredgecolor="#fcfcfb", markeredgewidth=2, label=name)
            ends.append((densities[-1], means[-1], name))
        _style(ax, title, key.replace("_", " "), "number of humans")
        ax.set_xticks(densities)
        ax.set_ylim(-0.03, 1.03)
        _direct_labels(ax, ends)
        ax.legend(frameon=False, fontsize=9)
    fig.text(0.01, 0.01, "Best checkpoint per run, held-out seeds. Training density: 20 humans.",
             fontsize=8, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_markdown(table, densities, ckpts, args, out_path):
    lines = ["# Baseline vs GNN: held-out evaluation", ""]
    n_seeds = len(range(*map(int, args.seeds.split(":")))) if ":" in args.seeds else len(args.seeds.split(","))
    lines.append(f"Held-out seeds `{args.seeds}`, {args.episodes_per_seed} episodes per seed "
                 f"({n_seeds * args.episodes_per_seed} episodes per cell). "
                 "Checkpoints ranked by training-time eval return.")
    lines.append("")
    for name, ck in ckpts.items():
        lines.append(f"- **{name}** best checkpoint: `{ck[0][0].name}`; top-{len(ck)}: "
                     + ", ".join(f"`{c.name}`" for c, _ in ck))
    lines.append("")
    lines.append("## Best checkpoint")
    lines.append("")
    lines.append("| humans | model | success | collision | timeout | return (mean ± std) | episode length |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    for d in densities:
        for name in table:
            r = table[name][d]["best"]
            lines.append(f"| {d} | {name} | {r['success_rate']:.3f} | {r['collision_rate']:.3f} | "
                         f"{r['timeout_rate']:.3f} | {r['mean_return']:.2f} ± {r['std_return']:.2f} | "
                         f"{r['mean_episode_length']:.0f} |")
    if args.top_k > 1:
        lines += ["", f"## Mean over top-{args.top_k} checkpoints (robustness to checkpoint choice)", ""]
        lines.append("| humans | model | success | collision | timeout |")
        lines.append("|---:|---|---:|---:|---:|")
        for d in densities:
            for name in table:
                rs = table[name][d]["all"]
                f = lambda k: f"{np.mean([r[k] for r in rs]):.3f} ± {np.std([r[k] for r in rs]):.3f}"
                lines.append(f"| {d} | {name} | {f('success_rate')} | {f('collision_rate')} | {f('timeout_rate')} |")
    Path(out_path).write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="name=run_dir (repeatable)")
    parser.add_argument("--log", action="append", default=[], help="name=training log (repeatable)")
    parser.add_argument("--densities", default="10,20,30")
    parser.add_argument("--seeds", default="10000:12000:100", help="held-out seeds, start:stop:step")
    parser.add_argument("--episodes-per-seed", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    runs, logs = parse_pairs(args.run), parse_pairs(args.log)
    densities = [int(d) for d in args.densities.split(",")]
    out = Path(args.out)
    (out / "eval_logs").mkdir(parents=True, exist_ok=True)

    curves = {name: parse_curve(path) for name, path in logs.items() if Path(path).exists()}
    if curves:
        plot_curves(curves, out / "learning_curves.png")

    ckpts = {name: top_checkpoints(run_dir, args.top_k) for name, run_dir in runs.items()}
    jobs = [(name, runs[name], ck, d) for name in runs for ck, _ in ckpts[name] for d in densities]
    print(f"{len(jobs)} evaluations, {args.jobs} parallel", flush=True)

    results = {}
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_eval, run_dir, ck, d, args.seeds, args.episodes_per_seed, out / "eval_logs"):
                   (name, ck, d) for name, run_dir, ck, d in jobs}
        for fut, (name, ck, d) in futures.items():
            results[(name, ck.name, d)] = fut.result()
            r = results[(name, ck.name, d)]
            print(f"{name:9s} {ck.name} n={d:2d}  success {r['success_rate']:.3f}  "
                  f"collision {r['collision_rate']:.3f}  timeout {r['timeout_rate']:.3f}", flush=True)

    table = {}
    for name in runs:
        table[name] = {}
        for d in densities:
            all_r = [results[(name, ck.name, d)] for ck, _ in ckpts[name]]
            table[name][d] = {"best": all_r[0], "all": all_r}

    (out / "comparison.json").write_text(json.dumps(
        {name: {str(d): v for d, v in rows.items()} for name, rows in table.items()}, indent=2))
    plot_density(table, densities, out / "density_sweep.png")
    write_markdown(table, densities, ckpts, args, out / "comparison.md")
    print(f"wrote {out}/comparison.md, comparison.json, density_sweep.png"
          + (", learning_curves.png" if curves else ""), flush=True)


if __name__ == "__main__":
    main()
