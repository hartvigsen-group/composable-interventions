#!/bin/bash

#SBATCH --job-name=hydra-grid-search
#SBATCH --account=hartvigsen_lab
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=02:59:00
#SBATCH --output=slurm_logs/hydra_job_%A_%a.out
#SBATCH --error=slurm_logs/hydra_job_%A_%a.err

# Exclude node(s) with 40GB A100
#SBATCH --exclude=udc-an36-13


# Load necessary modules or activate virtual environment
source activate lm-compose

# Run the Python script with Hydra's grid search
# The SLURM_ARRAY_TASK_ID environment variable will be different for each job in the array
# python main.py --multirun edit_set=$SLURM_ARRAY_TASK_ID number_of_edits=50 edit=True\
#  compress=True save_ckpt=False method=prune sparsity_ratio=0.35\
#  tag=exp_memit_wanda35
python main.py wandb=online $@