#!/bin/bash
# Full SAM-Road training on the internal cluster: 1 node x 3 A40 (DDP), data copied to its NVMe.
#   just cluster train [hydra overrides...]      (= sbatch sbatch/train.slurm ...)
#   resume: just cluster train ckpt_path=$DS/runs/samroad_<job>/ckpt/last.ckpt
# Env variables: see sbatch/common.sh.
#SBATCH --job-name=samroad-train
#SBATCH --partition=jz,jz-nvme
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=400G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out

source "$SLURM_SUBMIT_DIR/sbatch/common.sh"
run_train "$@"
