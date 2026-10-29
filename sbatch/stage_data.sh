#!/bin/bash
# Copie des données sur un nœud ; variables DS, NVME, TILES, INDEX_DIR, CKPT et SLURM_JOB_ID.
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
echo "$(hostname): $(du -sh "$NVME/cog" | cut -f1) of COGs in $(( $(date +%s) - t0 )) s"
