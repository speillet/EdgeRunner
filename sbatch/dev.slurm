#!/bin/bash
# Pipeline check on the cluster: same setup as train.slurm, 2 train + val batches per GPU.
#   just cluster dev [hydra overrides...]        (= sbatch sbatch/dev.slurm ...)
# Env variables: see sbatch/common.sh. The first job on a node also copies the COGs.
#SBATCH --job-name=samroad-dev
#SBATCH --partition=jz,jz-nvme
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=400G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

source "$SLURM_SUBMIT_DIR/sbatch/common.sh"
run_train trainer.fast_dev_run=2 "$@"
