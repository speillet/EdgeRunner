#!/bin/bash
# Dataset check on the cluster (CPU only): label overlays, stats and throughput from the NVMe.
#   just cluster smoke [--n 64] [hydra overrides...]     (= sbatch sbatch/smoke.slurm ...)
# Overlays in $OUT/smoke. Env variables: see sbatch/common.sh. The first job on a node also
# copies the COGs.
#SBATCH --job-name=samroad-smoke
#SBATCH --partition=jz,jz-nvme
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

CLEARML=0
source "$SLURM_SUBMIT_DIR/sbatch/common.sh"
"$PY" -m edgerunner.smoke --out "$OUT/smoke" "${ARGS[@]}" "$@"
