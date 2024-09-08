#!/bin/bash

#SBATCH --job-name=hydra-grid-search
#SBATCH --account=hartvigsen_lab
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=03:59:00
#SBATCH --output=slurm_logs/hydra_job_%A_%a.out
#SBATCH --error=slurm_logs/hydra_job_%A_%a.err
#SBATCH --array=1-10

# udc-an28-1 is 40GB A100 which is not enough compute
# udc-an34-7 is having CUDA issues
# udc-an34-19 is having CUDA issues
#SBATCH --exclude=udc-an34-7,udc-an34-19,udc-an34-19

# Load necessary modules or activate virtual environment
source activate lm-compose

# Run the Python script with Hydra's grid search
# The SLURM_ARRAY_TASK_ID environment variable will be different for each job in the array
# python main.py --multirun edit_set=$SLURM_ARRAY_TASK_ID number_of_edits=50 edit=True\
#  compress=True save_ckpt=False method=prune sparsity_ratio=0.35\
#  tag=exp_memit_wanda35
python main.py edit_set=$SLURM_ARRAY_TASK_ID seed=$SLURM_ARRAY_TASK_ID edit_dataset=counterfact wandb=online $@