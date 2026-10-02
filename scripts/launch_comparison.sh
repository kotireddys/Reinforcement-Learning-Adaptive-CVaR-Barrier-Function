#!/usr/bin/env bash
# Launch the baseline (diff_cvar) and GNN (diff_cvar_gnn) runs side by side in a
# detached tmux session, one GPU each, identical settings apart from the model.
#
#   GPUS="1 2" bash scripts/launch_comparison.sh [extra hydra overrides]
#
# Attach with `tmux attach -t comparison`; logs go to logs/<run>.log.
set -euo pipefail

cd "$(dirname "$0")/.."
read -r GPU_BASE GPU_GNN <<< "${GPUS:-0 0}"
SESSION="${SESSION:-comparison}"
EXTRA="$*"
mkdir -p logs

ENV_SETUP="export WANDB_MODE=${WANDB_MODE:-online} NUM_ENVS=${NUM_ENVS:-32}"
if [ -n "${CONDA_ENV:-}" ]; then
  ENV_SETUP="source \"\$(conda info --base)/etc/profile.d/conda.sh\" && conda activate ${CONDA_ENV} && ${ENV_SETUP}"
fi

launch() {
  local window=$1 gpu=$2 model=$3
  local cmd="cd '$PWD' && ${ENV_SETUP} && CUDA_VISIBLE_DEVICES=${gpu} MODEL=${model} RUN_NAME=${window} bash scripts/run_ppo.sh ${EXTRA} 2>&1 | tee logs/${window}.log; exec bash"
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux new-window -t "$SESSION" -n "$window" "bash -c $(printf %q "$cmd")"
  else
    tmux new-session -d -s "$SESSION" -n "$window" "bash -c $(printf %q "$cmd")"
  fi
}

launch baseline "$GPU_BASE" diff_cvar
launch gnn "$GPU_GNN" diff_cvar_gnn
tmux list-windows -t "$SESSION"
