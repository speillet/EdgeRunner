#!/bin/bash
# Sourced by the .slurm jobs: environment, index, copy of the data to each node's NVMe.
# Env: DS (store with cog/, bdtopo/, checkpoints/), NVME, OUT (default $DS/runs/samroad_<job>,
# + /fold<k> for the task k of an array),
# INDEX_DIR (default $DS/index if present, else rebuilt in $OUT/index by edgerunner.data.index),
# FOLDS (folds whose COGs are copied), CLEARML (1/0, keys CLEARML_* in .env), RUN_NAME.
# Leaves PY (venv python), ARGS (Hydra overrides for the NVMe copy) and run_train.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
if [ -f .env ]; then set -a; source .env; set +a; fi
DS=${DS:-/mnt/stores/store-DAI/pocs/datasets/road_graph_extractor}
NVME=${NVME:-/var/nvme/$USER/edgerunner}
OUT=${OUT:-$DS/runs/samroad_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}}
RUN_NAME=${RUN_NAME:-samroad}
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then  # cross-validation: one folder per fold
  OUT=$OUT/fold$SLURM_ARRAY_TASK_ID RUN_NAME=${RUN_NAME}_fold$SLURM_ARRAY_TASK_ID
fi
FOLDS=${FOLDS:-0 1 2 3 4}
CLEARML=${CLEARML:-1}
CKPT=sam_vit_b_01ec64.pth
echo "[$(date)] $SLURM_JOB_NAME nodes=$SLURM_JOB_NODELIST NVME=$NVME OUT=$OUT"

if [ "$CLEARML" = 1 ]; then
  : "${CLEARML_API_HOST:?set CLEARML_API_HOST and the other CLEARML_* vars in .env}"
  # the HTTP proxy rejects ClearML calls: reach its hosts directly
  hosts=$(for v in "$CLEARML_API_HOST" "${CLEARML_WEB_HOST:-}" "${CLEARML_FILES_HOST:-}"; do
            [ -n "$v" ] && echo "$v" | sed -E 's#^[a-z]+://([^:/]+).*#\1#'; done | sort -u | paste -sd,)
  export no_proxy="$hosts${no_proxy:+,$no_proxy}" NO_PROXY="$hosts${NO_PROXY:+,$NO_PROXY}"
fi

uv sync --locked --extra clearml --inexact  # fails if uv.lock is stale: same versions on every node
PY=$PWD/.venv/bin/python  # ranks call the venv directly: no concurrent uv runs
[ -f "$DS/checkpoints/$CKPT" ] || {
  echo "missing $DS/checkpoints/$CKPT: https://dl.fbaipublicfiles.com/segment_anything/$CKPT"
  exit 1; }
mkdir -p logs "$OUT"

# --- index: the official one once copied to $DS/index, else rebuilt (same folds) in the run ---
if [ -z "${INDEX_DIR:-}" ]; then
  INDEX_DIR=$DS/index
  if [ ! -f "$INDEX_DIR/tiles.parquet" ]; then
    INDEX_DIR=$OUT/index
    "$PY" -m edgerunner.data.index "$DS/cog" "$INDEX_DIR"
  fi
fi
[ -f "$INDEX_DIR/tiles.parquet" ] || { echo "missing $INDEX_DIR/tiles.parquet"; exit 1; }
echo "index: $INDEX_DIR"

# --- copy to every node, data_root layout: index/ cog/ bdtopo/troncon/ (incremental) ---
TILES=$PWD/logs/samroad_tiles_$SLURM_JOB_ID.txt
$PY -c 'import sys, pandas as pd
t = pd.read_parquet(sys.argv[1] + "/tiles.parquet", columns=["uri", "fold"])
print("\n".join(t[t["fold"].astype(str).isin(sys.argv[2].split())]["uri"]))' \
  "$INDEX_DIR" "$FOLDS" > "$TILES"
echo "copying $(wc -l < "$TILES") COGs per node"
export DS NVME TILES INDEX_DIR CKPT
srun --ntasks-per-node=1 --ntasks="$SLURM_JOB_NUM_NODES" bash "$PWD/sbatch/stage_data.sh"

ARGS=("data_root=$NVME" "sam_ckpt=$NVME/$CKPT" data.num_workers=10)
export PYTHONUNBUFFERED=1

# DDP on every GPU of the job: one task per GPU, Lightning reads the ranks from SLURM; one Hydra
# dir for all ranks. Arguments: Hydra overrides, applied last.
run_train() {
  local cml=false
  [ "$CLEARML" = 1 ] && cml=true
  export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
  # some nodes expose a soft-iWARP device that NCCL mistakes for InfiniBand: plain TCP on bond0
  export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1} NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond0}
  srun "$PY" -m edgerunner.train "${ARGS[@]}" "trainer.devices=$SLURM_NTASKS_PER_NODE" \
    "trainer.num_nodes=$SLURM_JOB_NUM_NODES" trainer.strategy=ddp "hydra.run.dir=$OUT" \
    "run_name=$RUN_NAME" "clearml=$cml" "$@"
  echo "[$(date)] done, outputs in $OUT"
}
