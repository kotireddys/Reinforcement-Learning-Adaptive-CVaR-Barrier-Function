#!/usr/bin/env bash
# Final stage of the baseline-vs-GNN comparison, run after both trainings exit:
#   1. held-out evaluation + density sweep + learning curves  -> results/
#   2. side-by-side rollout GIFs on the same held-out seeds   -> results/rollouts/
#   3. GNN attention overlays                                 -> results/attention/
#
#   bash scripts/post_training.sh            # waits for running trainings first
set -euo pipefail

cd "$(dirname "$0")/.."
RUNS=outputs/social_nav_var_num/runs
BASE=${BASE:-$(ls -d "$RUNS"/baseline-diff_cvar-* | head -1)}
GNN=${GNN:-$(ls -d "$RUNS"/gnn-diff_cvar_gnn-* | head -1)}
BASE_LOG=${BASE_LOG:-logs/baseline.log}
GNN_LOG=${GNN_LOG:-logs/gnn.log}
OUT=${OUT:-results}
EVAL_SEEDS=${EVAL_SEEDS:-10000:12000:100}
EPISODES=${EPISODES:-10}
TOPK=${TOPK:-3}
DENSITIES=${DENSITIES:-10,20,30}
QUAL_SEEDS=${QUAL_SEEDS:-10000,10100,10200,10300,10400,10500}
JOBS=${JOBS:-32}
mkdir -p "$OUT/rollouts" "$OUT/attention" logs

while [ -z "${SKIP_WAIT:-}" ] && pgrep -u "$USER" -f "scripts/run\.py .*run_name=(baseline|gnn) " >/dev/null; do
  echo "$(date '+%F %T') waiting for training to finish..."
  sleep 600
done
echo "$(date '+%F %T') training finished; starting final evaluation" | tee -a logs/status.md

python scripts/compare_runs.py --run baseline="$BASE" --run gnn="$GNN" \
  --log baseline="$BASE_LOG" --log gnn="$GNN_LOG" \
  --densities "$DENSITIES" --seeds "$EVAL_SEEDS" --episodes-per-seed "$EPISODES" --top-k "$TOPK" \
  --jobs "$JOBS" --out "$OUT" 2>&1 | tee "$OUT/compare_runs.log"

best_ckpt() {
  python -c "import json,sys; m=json.load(open(sys.argv[1]+'/ckpt_manifest.json')); print(max(m, key=lambda k: m[k]['performance']))" "$1"
}

for run in "$BASE" "$GNN"; do
  ck=$(best_ckpt "$run")
  name=$([ "$run" = "$BASE" ] && echo baseline || echo gnn)
  python scripts/eval.py --save-dir "$run" --checkpoint "$run/$ck" --seeds "$QUAL_SEEDS" \
    --episodes-per-seed 1 --visualize --tag qual >"$OUT/rollouts/$name.log" 2>&1 &
done
ck=$(best_ckpt "$GNN")
for n in ${DENSITIES//,/ }; do
  python scripts/visualize_attention.py --save-dir "$GNN" --checkpoint "$GNN/$ck" --seeds "$QUAL_SEEDS" \
    --out "$OUT/attention/n$n" env.humans.num_humans=$n >"$OUT/attention/n$n.log" 2>&1 &
done
wait

for run in "$BASE" "$GNN"; do
  name=$([ "$run" = "$BASE" ] && echo baseline || echo gnn)
  cp -r "$run"/visualize_*qual "$OUT/rollouts/$name" 2>/dev/null || true
  cp "$run/ckpt_manifest.json" "$OUT/${name}_ckpt_manifest.json"
done
cp "$BASE_LOG" "$OUT/baseline_train.log"; cp "$GNN_LOG" "$OUT/gnn_train.log"
echo "$(date '+%F %T') post-training done -> $OUT" | tee -a logs/status.md
