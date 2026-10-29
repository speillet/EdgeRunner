#!/bin/bash
# 5-fold cross-validation on the internal cluster: array task k trains on the other folds of 0-4
# and validates on fold k (fold test never used). 2 folds at a time, each on 1 node x 3 A40.
#   just cluster cv [hydra overrides...]          (= sbatch sbatch/cv.slurm ...)
# Outputs: $DS/runs/samroad_<array job>/fold<k>/ (metrics.json = best checkpoint on fold k) and
# cv_summary.csv next to them, refreshed by every fold that ends.
# Resume fold k in its folder:
#   OUT=$DS/runs/samroad_<id> sbatch --array=k sbatch/cv.slurm ckpt_path=$DS/runs/samroad_<id>/fold<k>/ckpt/last.ckpt
# Env variables: see sbatch/common.sh.
#SBATCH --job-name=samroad-cv
#SBATCH --array=0-4%2
#SBATCH --partition=jz,jz-nvme
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=400G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%A_%a.out

source "$SLURM_SUBMIT_DIR/sbatch/common.sh"
K=$SLURM_ARRAY_TASK_ID
TRAIN=$(printf '%s\n' 0 1 2 3 4 | grep -vx "$K" | paste -sd,)
run_train "data.train_folds=[$TRAIN]" "data.val_folds=[$K]" "$@"
"$PY" -m edgerunner.cv "${OUT%/*}" || echo "no cross-validation summary yet"
