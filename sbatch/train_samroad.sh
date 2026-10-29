#!/bin/bash
# SAM-Road on IGN data: copies the data to each node's NVMe, then NODES x 3 A40 DDP.
#   just cluster [train|dev|smoke] [hydra overrides...]
#   (or: MODE=dev sbatch sbatch/train_samroad.sh data.batch_size=8)
# Env: DS (dataset store with cog/, bdtopo/, checkpoints/, index/), INDEX_DIR (default
# $DS/index: tiles.parquet + windows.parquet), NVME, OUT (default $DS/runs/samroad_<job>),
# FOLDS (folds whose COGs are copied), CLEARML (1/0, keys CLEARML_* in .env), RUN_NAME.
#SBATCH --job-name=samroad-ign
#SBATCH --partition=jz,jz-nvme
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=3
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=400G
#SBATCH --time=48:00:00
#SBATCH --output=logs/samroad_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
if [ -f .env ]; then set -a; source .env; set +a; fi
MODE=${MODE:-train}
DS=${DS:-/mnt/stores/store_dai/pocs/datasets/road_graph_extractor}
INDEX_DIR=${INDEX_DIR:-$DS/index}
[ -f "$INDEX_DIR/tiles.parquet" ] || { echo "missing $INDEX_DIR/tiles.parquet (set INDEX_DIR)"; exit 1; }
NVME=${NVME:-/var/nvme/$USER/edgerunner}
OUT=${OUT:-$DS/runs/samroad_$SLURM_JOB_ID}
FOLDS=${FOLDS:-0 1 2 3 4}
CLEARML=${CLEARML:-1}
CKPT=sam_vit_b_01ec64.pth
echo "[$(date)] mode=$MODE nodes=$SLURM_JOB_NODELIST NVME=$NVME OUT=$OUT"

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

# --- copy to every node, data_root layout: index/ cog/ bdtopo/troncon/ (incremental) ---
mkdir -p logs "$OUT"
TILES=$PWD/logs/samroad_tiles_$SLURM_JOB_ID.txt
$PY -c 'import sys, pandas as pd
t = pd.read_parquet(sys.argv[1] + "/tiles.parquet", columns=["uri", "fold"])
print("\n".join(t[t["fold"].astype(str).isin(sys.argv[2].split())]["uri"]))' \
  "$INDEX_DIR" "$FOLDS" > "$TILES"
echo "copying $(wc -l < "$TILES") COGs per node"
export DS NVME TILES INDEX_DIR CKPT
srun --ntasks-per-node=1 --ntasks="$SLURM_JOB_NUM_NODES" bash -c '
  set -euo pipefail
  t0=$(date +%s)
  mkdir -p "$NVME/index" "$NVME/cog" "$NVME/bdtopo"
  rsync -a "$INDEX_DIR/" "$NVME/index/"
  rsync -a "$DS/bdtopo/bretagne/troncon" "$NVME/bdtopo/"
  rsync -a "$DS/checkpoints/$CKPT" "$NVME/"
  split -n l/8 "$TILES" "/tmp/samroad_${SLURM_JOB_ID}_"  # 8 parallel copies from CIFS
  pids=()
  for f in /tmp/samroad_${SLURM_JOB_ID}_*; do
    rsync -a --files-from="$f" "$DS/cog/" "$NVME/cog/" &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  rm -f /tmp/samroad_${SLURM_JOB_ID}_*
  if [ "$failed" != 0 ]; then
    echo "COG staging failed on $(hostname)" >&2
    exit 1
  fi
  echo "$(hostname): $(du -sh "$NVME/cog" | cut -f1) of COGs in $(( $(date +%s) - t0 )) s"'

ARGS=("data_root=$NVME" "sam_ckpt=$NVME/$CKPT" data.num_workers=10 "$@")
export PYTHONUNBUFFERED=1
if [ "$MODE" = smoke ]; then
  "$PY" -m edgerunner.smoke --out "$OUT/smoke" "${ARGS[@]}"
  exit
fi

# --- train: one task per GPU, Lightning reads ranks from SLURM; one Hydra dir for all ranks ---
CML=false; [ "$CLEARML" = 1 ] && CML=true
ARGS+=(trainer.devices=3 "trainer.num_nodes=$SLURM_JOB_NUM_NODES" trainer.strategy=ddp
       "hydra.run.dir=$OUT" "run_name=${RUN_NAME:-samroad}" "clearml=$CML")
[ "$MODE" = dev ] && ARGS+=(trainer.fast_dev_run=2)
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
# some nodes expose a soft-iWARP device that NCCL mistakes for InfiniBand: plain TCP on bond0
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1} NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond0}
srun "$PY" -m edgerunner.train "${ARGS[@]}"
echo "[$(date)] done, outputs in $OUT"
